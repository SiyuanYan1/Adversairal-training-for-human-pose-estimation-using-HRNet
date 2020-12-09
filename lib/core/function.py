# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch

import torch.nn.functional as F

from core.evaluate import accuracy, calc_dists
from core.inference import get_final_preds, get_max_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from utils.utils import get_value

logger = logging.getLogger(__name__)


def train(config, train_loader, generator, discriminator,
                     criterion, optimizer, kt, epoch,
                     output_dir, tb_log_dir, writer_dict):
    """
    criterion: dict, all the loss functions for training.
    optimizer: dict, all the optimizers for training.

    Return:
        kt - returned balance parameter.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    d_losses = AverageMeter()
    combine_losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    generator.train()
    discriminator.train()

    # kt
    kt = kt

    end = time.time()
    for i, (images, resized_img, target, target_weight, meta) in enumerate(train_loader):
        # images - image
        # target - ground truth joint heatmap
        # target_weight - 0 for not provided joint, 1 for visible joint
        # occlusion - ground truth joint occlusion
        # occlusion_weight - 0 for not provided joint, 1 for occluded joint
        # meta - some other thing

        # measure data loading time
        images = images.cuda(non_blocking=True)
        resized_img = resized_img.cuda(non_blocking=True)

        # compute output (joint+occlusion heatmap)
        outputs = generator(images)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        # ----------------------------------------------------------------------------------------
        # Compute content loss
        # ----------------------------------------------------------------------------------------
        # if isinstance(outputs, list):
        #     heatmap_loss = criterion['heatmap_loss'](outputs[0], ground_truth, ground_truth_weight)
        #     for output in outputs[1:]:
        #         heatmap_loss += criterion['heatmap_loss'](output, ground_truth, ground_truth_weight)
        # else:
        #     output = outputs
        #     heatmap_loss = criterion['heatmap_loss'](output, ground_truth, ground_truth_weight)

        loss_content = criterion['heatmap_loss'](outputs, target, target_weight)

        # ----------------------------------------------------------------------------------------
        # Training D according to adv loss
        # ----------------------------------------------------------------------------------------
        # resized_img = torch.nn.functional.interpolate(images, scale_factor=0.25, mode='bilinear')
        # resized_img = resized_img.cuda(non_blocking=True)
        input_real = torch.cat([resized_img, target], dim=1)
        input_fake = torch.cat([resized_img, outputs], dim=1)

        loss_d_real = criterion['generative_loss'](discriminator(input_real), target)
        loss_d_fake = criterion['generative_loss'](discriminator(input_fake.detach()), outputs.detach())
        loss_d = loss_d_real - kt * loss_d_fake

        optimizer['d'].zero_grad()
        loss_d.backward()
        optimizer['d'].step()

        # ----------------------------------------------------------------------------------------
        # Training G
        # ----------------------------------------------------------------------------------------
        loss_g = criterion['generative_loss'](discriminator(input_fake), outputs)
        loss_combine = loss_content + config.GENERATOR.LAMBDA * loss_g

        optimizer['g'].zero_grad()
        loss_combine.backward()
        optimizer['g'].step()

        # ----------------------------------------------------------------------------------------
        # Update kt
        # ----------------------------------------------------------------------------------------
        loss_d_real_ = get_value(loss_d_real)
        loss_d_fake_ = get_value(loss_d_fake)
        balance = config.TRAIN.BALANCE_GAMMA * loss_d_real_ - loss_d_fake_ # / config.GENERATOR.LAMBDA  # Is this dividing good? Original impl. has this
        kt = kt + config.TRAIN.LAMBDA_KT * balance
        kt = min(1, max(0, kt))

        # measure accuracy and record loss
        losses.update(loss_content.item(), images.size(0))
        d_losses.update(loss_d.item(), images.size(0))
        combine_losses.update(loss_combine.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            # record output joint heatmap
            # put everything to cpu from here.
            # indices = torch.tensor([i for i in range(config.MODEL.NUM_JOINTS)])
            # output = torch.index_select(outputs.detach().cpu(), 1, indices)

            _, avg_acc, cnt, pred = accuracy(outputs.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=images.size(0) / batch_time.val, loss=losses,
                      acc=acc)
            # msg = 'Epoch: [{0}][{1}/{2}]\t' \
            #       'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            #       'Speed {speed:.1f} samples/s\t' \
            #       'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            #       'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
            #       'PoseLoss {pose_d_losses.val:.5f} ({pose_d_losses.avg:.5f})\t' \
            #       'ConfLoss {conf_d_losses.val:.5f} ({conf_d_losses.avg:.5f})\t' \
            #       'CombineLoss {combine_losses.val:.5f} ({combine_losses.avg:.5f})\t' \
            #       'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
            #           epoch, i, len(train_loader), batch_time=batch_time,
            #           speed=images.size(0) / batch_time.val,
            #           data_time=data_time, loss=losses,
            #           pose_d_losses=pose_d_losses,
            #           conf_d_losses=conf_d_losses,
            #           combine_losses=combine_losses,
            #           acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_content_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer.add_scalar('train_combined_loss', combine_losses.val, global_steps)
            writer.add_scalar('train_d_losses', d_losses.val, global_steps)
            writer.add_scalar('kt', kt, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, images, meta, target, pred * 4, outputs,
                              prefix)
    return kt


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
                        tb_log_dir, writer_dict=None):
    """
    criterion: dict, all the loss functions for training.
    optimizer: dict, all the optimizers for training.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, _, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)

            # In the paper it should be a single output, no idea why it can be a list, so I just comment it.
            # if isinstance(outputs, list):
            #     output = outputs[-1]
            # else:
            #     output = outputs

            # Take first num of joint of channels (the joint heatmap) as the final output.
            indices = torch.tensor([i for i in range(config.MODEL.NUM_JOINTS)]).cuda()
            output = torch.index_select(outputs, 1, indices)

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                # if isinstance(outputs_flipped, list):
                #     output_flipped = outputs_flipped[-1]
                # else:
                #     output_flipped = outputs_flipped

                # Only take joint heatmap
                output_flipped = torch.index_select(outputs_flipped, 1, indices)

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion['heatmap_loss'](output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred * 4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def get_p_fake_poseD(output, target, thr=0.05):
    """
    Calculate p_fake

    output, target: numpy array, [batch_size, n_joints, h, w]
    thr: if distance is small than threshold, assign corresponding p_i as 1

    p_fake: numpy array, [batch_size, num_joints]
    TODO: find the proper threshold here
    """
    pred, _ = get_max_preds(output)
    target, _ = get_max_preds(target)
    h = output.shape[2]
    w = output.shape[3]
    norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10

    batch_size = output.shape[0]
    num_joints = output.shape[1]
    p_fake = np.zeros((batch_size, num_joints))
    p_fake[np.swapaxes(calc_dists(pred, target, norm), 0, 1) < thr] = 1

    return p_fake.astype(np.float32)


def get_c_fake_confD(output, target, thr=1e-7):
    """
    Calculate c_fake,

    output, target: tensor, [batch_size, n_joints, h, w]

    c_fake, tensor, [batch_size, n_joints]
    """
    batch_size, n_joints = output.shape[0], output.shape[1]
    c_fake = torch.zeros((batch_size, n_joints)).cuda()
    dist = torch.sqrt(
        torch.mean(
            torch.square(output-target),
            dim=(2, 3)
        )
    )
    c_fake[dist < thr] = 1

    return c_fake


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
