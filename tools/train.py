# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss, PredictLoss, GenerativeLoss, LSGenerativeLoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import model


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # load model
    generator = eval('model.' + cfg.GENERATOR.NAME + '.get_generator')(
        cfg, is_train=True
    )
    discriminator = eval('model.' + cfg.DISCRIMINATOR.NAME + '.get_discriminator')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/model', cfg.GENERATOR.NAME + '.py'),
        final_output_dir)
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/model', cfg.DISCRIMINATOR.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )

    logger.info(get_model_summary(generator, dump_input))

    generator = torch.nn.DataParallel(generator, device_ids=cfg.GPUS).cuda()
    discriminator = torch.nn.DataParallel(discriminator, device_ids=cfg.GPUS).cuda()

    # init kt
    kt = cfg.TRAIN.INIT_KT

    # define loss function (criterion) and optimizer
    criterion = {}
    criterion['heatmap_loss'] = PredictLoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    criterion['generative_loss'] = LSGenerativeLoss().cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1

    # Define optimizer
    optimizer = {}
    optimizer['g'] = get_optimizer(cfg, cfg.TRAIN.G_OPTIMIZER, cfg.TRAIN.G_LR, generator)
    optimizer['d'] = get_optimizer(cfg, cfg.TRAIN.D_OPTIMIZER, cfg.TRAIN.D_LR, discriminator)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint_90.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['state_dict'])
        discriminator.load_state_dict(checkpoint['d_state_dict'])
        kt = checkpoint['kt']
        optimizer['g'].load_state_dict(checkpoint['optimizer'])
        optimizer['d'].load_state_dict(checkpoint['d_optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer['g'], cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )
    lr_d_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer['d'], cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # train for one epoch
        kt = train(cfg, train_loader, generator, discriminator, criterion, optimizer, kt, epoch,
                              final_output_dir, tb_log_dir, writer_dict)

        lr_scheduler.step()
        lr_d_scheduler.step()

        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, generator, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': generator.state_dict(),
            'd_state_dict': discriminator.state_dict(),
            'best_state_dict': generator.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer['g'].state_dict(),
            'd_optimizer': optimizer['d'].state_dict(),
            'kt': kt,
        }, best_model, final_output_dir, filename='checkpoint_{}.pth'.format(epoch))

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(generator.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
