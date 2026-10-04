import torchvision.models as tvm


def build():
    # Pretrained weights: real BN statistics, and its Conv2d layers have bias=False.
    return tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
