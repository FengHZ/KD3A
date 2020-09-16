import torch
import torch.nn as nn
import numpy as np
import random
import re


class _PreProcess(nn.Sequential):
    def __init__(self, num_input_channels, num_init_features=16, small_input=True):
        super(_PreProcess, self).__init__()
        if small_input:
            self.add_module('conv0',
                            nn.Conv2d(num_input_channels, num_init_features, kernel_size=3, stride=1, padding=1,
                                      bias=True))
        else:
            self.add_module('conv0',
                            nn.Conv2d(num_input_channels, num_init_features, kernel_size=7, stride=2, padding=3,
                                      bias=True))
            self.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                  ceil_mode=False))


class _WideResUnit(nn.Module):
    def __init__(self, num_input_features, num_output_features, stride=1, drop_rate=0.3):
        super(_WideResUnit, self).__init__()
        self.f_block = nn.Sequential()
        self.f_block.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.f_block.add_module('relu1', nn.LeakyReLU(inplace=True))
        self.f_block.add_module('conv1', nn.Conv2d(num_input_features, num_output_features,
                                                   kernel_size=3, stride=stride, padding=1, bias=False))
        self.f_block.add_module('dropout', nn.Dropout(drop_rate))
        self.f_block.add_module('norm2', nn.BatchNorm2d(num_output_features))
        self.f_block.add_module('relu2', nn.LeakyReLU(inplace=True))
        self.f_block.add_module('conv2', nn.Conv2d(num_output_features, num_output_features,
                                                   kernel_size=3, stride=1, padding=1, bias=False))

        if num_input_features != num_output_features or stride != 1:
            self.i_block = nn.Sequential()
            self.i_block.add_module('norm', nn.BatchNorm2d(num_input_features))
            self.i_block.add_module('relu', nn.LeakyReLU(inplace=True))
            self.i_block.add_module('conv',
                                    nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=stride,
                                              bias=False))

    def forward(self, x):
        new_features = self.f_block(x)
        if hasattr(self, "i_block"):
            x = self.i_block(x)
        return new_features + x


class _WideBlock(nn.Module):
    def __init__(self, input_channel, channel_width, block_depth, down_sample=False, drop_rate=0.0):
        super(_WideBlock, self).__init__()
        self.wide_block = nn.Sequential()
        for i in range(block_depth):
            if i == 0:
                unit = _WideResUnit(input_channel, channel_width, stride=int(1 + down_sample),
                                    drop_rate=drop_rate)
            else:
                unit = _WideResUnit(channel_width, channel_width, drop_rate=drop_rate)
            self.wide_block.add_module("wideunit%d" % (i + 1), unit)

    def forward(self, x):
        return self.wide_block(x)


class WideResNet(nn.Module):
    def __init__(self, num_input_channels=3, num_init_features=16, depth=28, width=2,
                 data_parallel=True, small_input=True, drop_rate=0.0):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        block_depth = (depth - 4) // 6
        widths = [int(v * width) for v in (16, 32, 64)]
        self._widths = widths
        self.encoder = nn.Sequential()
        pre_process = _PreProcess(num_input_channels, num_init_features, small_input=small_input)
        if data_parallel:
            pre_process = nn.DataParallel(pre_process)
        self.encoder.add_module("pre_process", pre_process)
        for idx, width in enumerate(widths):
            if idx == 0:
                wide_block = _WideBlock(num_init_features, width, block_depth, drop_rate=drop_rate)
            else:
                wide_block = _WideBlock(widths[idx - 1], width, block_depth, down_sample=True, drop_rate=drop_rate)
            if data_parallel:
                wide_block = nn.DataParallel(wide_block)
            self.encoder.add_module("wideblock%d" % (idx + 1), wide_block)
        trans = nn.Sequential()
        trans.add_module("norm", nn.BatchNorm2d(widths[-1]))
        trans.add_module("relu", nn.LeakyReLU(inplace=True))
        if data_parallel:
            trans = nn.DataParallel(trans)
        self.encoder.add_module('transition', trans)
        self.num_feature_channel = widths[-1]

    def forward(self, input_img):
        features = self.encoder(input_img)
        return features


def get_wide_resnet(name, drop_rate=0.0, input_channels=3, small_input=True, data_parallel=True):
    """
    :param name: wideresnet-depth-width type e.g. wideresnet-28-2
    :param drop_rate: the drop rate
    :param data_parallel:
    :param input_channels:
    :return: widresnet encoder
    """
    depth, width = re.findall(r'\d+', name)
    depth = eval(depth)
    width = eval(width)
    return WideResNet(depth=depth, width=width, drop_rate=drop_rate, num_input_channels=input_channels,
                      data_parallel=data_parallel, small_input=small_input)
