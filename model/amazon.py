import torch.nn as nn
import torch.nn.functional as F


class AmazonMLP(nn.Module):
    def __init__(self, data_parallel=True):
        super(AmazonMLP, self).__init__()
        encoder = nn.Sequential()
        encoder.add_module("fc1", nn.Linear(5000, 1000))
        # encoder.add_module("bn1", nn.BatchNorm1d(1000))
        encoder.add_module("relu1", nn.ReLU())
        encoder.add_module("fc2", nn.Linear(1000, 500))
        # encoder.add_module("bn2", nn.BatchNorm1d(500))
        encoder.add_module("relu2", nn.ReLU())
        encoder.add_module("fc3", nn.Linear(500, 100))
        # encoder.add_module("bn3", nn.BatchNorm1d(100))
        encoder.add_module("relu3", nn.ReLU())
        if data_parallel:
            self.encoder = nn.DataParallel(encoder)
        else:
            self.encoder = encoder

    def forward(self, x):
        feature = self.encoder(x)
        return feature


class AmazonClassifier(nn.Module):
    def __init__(self, data_parallel=True):
        super(AmazonClassifier, self).__init__()
        linear = nn.Sequential()
        linear.add_module("fc", nn.Linear(100, 2))
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, x):
        x = self.linear(x)
        return x

