import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import os

from model.basic import HourglassNet, HgResBlock
from model.pose_hrnet import PoseHighResolutionNet, conv3x3, BasicBlock, Bottleneck, HighResolutionModule


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError("Index {} is out of range".format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class ConvBnRelu(nn.Module):
    """
        A block of convolution, relu, batchnorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):

        super(ConvBnRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # changing this to instance norm due to small batch_size of 1
        self.bn   = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConvTripleBlock(nn.Module):
    """
        A block of 3 ConvBnRelu blocks.
        This triple block makes up a residual block as described in the paper
        Resolution h x w does not change across this block
    """

    def __init__(self, in_channels, out_channels):

        super(ConvTripleBlock, self).__init__()

        out_channels_half = out_channels // 2

        self.convblock1 = ConvBnRelu(in_channels, out_channels_half)
        self.convblock2 = ConvBnRelu(out_channels_half, out_channels_half, kernel_size=3, stride=1, padding=1)
        self.convblock3 = ConvBnRelu(out_channels_half, out_channels)

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)

        return x


class SkipLayer(nn.Module):
    """
        The skip connections are necessary for transferring global and local context
        Resolution h x w does not change across this block
    """

    def __init__(self, in_channels, out_channels):

        super(SkipLayer, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels

        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        if self.in_channels != self.out_channels:
            x = self.conv(x)

        return x


class Residual(nn.Module):
    """
        The highly used Residual block
        Resolution h x w does not change across this block
    """
    def __init__(self, in_channels, out_channels):

        super(Residual, self).__init__()

        self.convblock = ConvTripleBlock(in_channels, out_channels)
        self.skip 	   = SkipLayer(in_channels, out_channels)


    def forward(self, x):

        y = self.convblock(x)
        z = self.skip(x)
        out = y + z

        return out


class Hourglass(nn.Module):
    """
        Hourglass network - Core component of Generator
    """

    def __init__(self, num_channels, num_reductions=4, num_residual_modules=2):

        super(Hourglass, self).__init__()

        scale_factor = 2
        self.num_reductions = num_reductions

        skip = []
        for _ in range(num_residual_modules):
            skip.append(Residual(num_channels, num_channels))
        self.skip = nn.Sequential(*skip)

        self.pool = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)

        before_mid = []
        for _ in range(num_residual_modules):
            before_mid.append(Residual(num_channels, num_channels))
        self.before_mid = nn.Sequential(*before_mid)

        if (num_reductions > 1):
            self.sub_hourglass = Hourglass(num_channels, num_reductions - 1,
                                            num_residual_modules)
        else:
            mid_residual = []
            for _ in range(num_residual_modules):
                mid_residual.append(Residual(num_channels, num_channels))
            self.mid_residual = nn.Sequential(*mid_residual)

        end_residual = []
        for _ in range(num_residual_modules):
            end_residual.append(Residual(num_channels, num_channels))
        self.end_residual = nn.Sequential(*end_residual)

        self.up_sample = nn.Upsample(scale_factor=scale_factor, mode='nearest')


    def forward(self, x):
        y = self.pool(x)
        y = self.before_mid(y)

        if (self.num_reductions > 1):
            y = self.sub_hourglass(y)
        else:
            y = self.mid_residual(y)

        y = self.end_residual(y)
        y = self.up_sample(y)

        x = self.skip(x)

        out = x + y

        return out


class StackedHourglass(nn.Module):
    """
        Stacking hourglass - gives precursors to pose and occlusion heatmaps
    """

    def __init__(self, num_channels, hourglass_params):
        super(StackedHourglass, self).__init__()

        self.hg = []
        for _ in range(2):
            self.hg.append(Hourglass(num_channels, hourglass_params['num_reductions'], hourglass_params['num_residual_modules']))
        self.hg = ListModule(*self.hg)

        self.dim_reduction = nn.Conv2d(in_channels=2 * num_channels, out_channels=num_channels, kernel_size=1, stride=1)

    def forward(self, x):
        y = x
        out1 = self.hg[0](y)
        y = torch.cat((out1, y), dim=1)
        y = self.dim_reduction(y)
        out2 = self.hg[1](y)
        return [out1, out2]


class Discriminator(nn.Module):

    def __init__(self, cfg, in_channels):
        '''
            Initialisation of the Discriminator network
            Contains the necessary modules
            Input is pose and confidence heatmaps
            in_channels = num_joints x 2 + 3 (for the image) (Pose network)
            in_channels = num_joints x 2 (Confidence network)
        '''

        super(Discriminator, self).__init__()
        ## Define Layers Here ##
        num_channels = cfg.DISCRIMINATOR.RES_CHANNEL  # 512
        self.num_residuals = cfg.DISCRIMINATOR.NUM_RES  # 5
        self.heatmap_h, self.heatmap_w = cfg.DISCRIMINATOR.HEATMAP_SIZE  # [64, 64]
        self.residual = []
        self.residual.append(Residual(in_channels, num_channels))
        for _ in range(self.num_residuals-1):
            self.residual.append(Residual(num_channels, num_channels))

        self.residual = ListModule(*self.residual)

        self.max_pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(num_channels*(self.heatmap_h // (2**(self.num_residuals-1)))*(self.heatmap_w // (2**(self.num_residuals-1))), 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, cfg.MODEL.NUM_JOINTS)

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

    def forward(self, x):
        """
            Assuming num channels is 512 and in_channels, num_residuals = 5
        """
        # N x in_channels x 256 x 256
        x = self.residual[0](x)
        # N x 512 x 256 x 256

        for i in range(1, self.num_residuals):
            x = self.residual[i](x)
            x = self.max_pool(x)

        # N x 512 x 16 x 16
        x = x.view(x.shape[0], -1)
        # N x (512 * 16 * 16)
        x = self.fc1(x)
        # N x 128
        x = self.relu(x)
        # N x 128
        x = self.fc2(x)
        # N x 16
        x = torch.sigmoid(x)

        return x


class ReallyDummyDiscriminator(nn.Module):
    def __init__(self, cfg, in_channel=16*2+3):
        super(ReallyDummyDiscriminator, self).__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)

        self.bottle_neck = nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.fan = nn.Conv2d(16, 16, kernel_size=16)
        # self.fan = nn.AvgPool2d(kernel_size=16)

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

    def forward(self, x):

        # N x (2n+3) x 64 x64
        x = self.conv1(x)
        x = self.bn1(x)

        # N x 64 x 32 x 32
        x = self.conv2(x)
        x = self.bn2(x)

        # N x 128 x 16 x16
        x = self.bottle_neck(x)

        # N x 16 x 16 x 16
        x = self.fan(x)

        # N x 16
        x = torch.squeeze(x)
        # N x 16
        x = torch.sigmoid(x)

        return x


class HourglassDisNet(HourglassNet):
    def __init__(self, cfg, resBlock=HgResBlock):
        super().__init__(
            cfg.DISCRIMINATOR.N_STACKS,
            cfg.DISCRIMINATOR.N_MODULES,
            cfg.DISCRIMINATOR.FEATURES,
            cfg.DISCRIMINATOR.NUM_JOINTS,
            resBlock,
            cfg.DISCRIMINATOR.NUM_JOINTS+3)

    def _make_head(self):
        self.conv1 = nn.Conv2d(self.inplanes, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = self.resBlock(64, 128)
        self.res2 = self.resBlock(128, 128)
        self.res3 = self.resBlock(128, self.nFeat)

    def forward(self, x):

        # head
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        for i in range(self.nStacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            if i < (self.nStacks - 1):
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return score


class HRDisNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        self.inplanes = cfg['DISCRIMINATOR']['NUM_JOINTS'] + 3
        extra = cfg['MODEL']['EXTRA']
        super(HRDisNet, self).__init__()

        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],  # Predict joint points.
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading discriminator pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_discriminator(cfg, is_train=True, **kwargs):
    model = HRDisNet(cfg)

    if is_train and cfg['DISCRIMINATOR']['INIT_WEIGHTS']:
        # TODO: load pretrained discriminator later
        model.init_weights(cfg['DISCRIMINATOR']['PRETRAINED'])

    return model
