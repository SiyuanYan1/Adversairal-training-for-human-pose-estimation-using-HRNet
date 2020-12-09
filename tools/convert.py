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
                        # default='../experiments/mpii/gan/w32_256x256_adam_lr1e-3.yaml',
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
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # writer_dict['writer'].add_graph(generator, dump_input)

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

    best_perf = 0.0
    best_model = False
    last_epoch = -1

    # Define optimizer
    optimizer = {}
    optimizer['g'] = get_optimizer(cfg, cfg.TRAIN.G_OPTIMIZER, cfg.TRAIN.G_LR, generator)
    optimizer['d'] = get_optimizer(cfg, cfg.TRAIN.D_OPTIMIZER, cfg.TRAIN.D_LR, discriminator)
    # optimizer['conf_d'] = get_optimizer(cfg, confidence_discriminator)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
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
    # print(generator.state_dict())
    # print('----------------------------------------------------------------------------------')
    # print(generator.module.state_dict())

    save_checkpoint({
            'model': cfg.MODEL.NAME,
            'state_dict': generator.module.state_dict(),
        }, False, final_output_dir, filename='hrgan_w32_256x192.pth')

if __name__ == '__main__':
    main()