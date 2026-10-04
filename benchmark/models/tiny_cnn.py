import torch.nn as nn


def build():
    # Fresh BN stats are all 1s and 0s, so the exporter dedups them behind Identity nodes.
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1, bias=True),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1, bias=True),
        nn.BatchNorm2d(32),
        nn.ReLU(),
    )
