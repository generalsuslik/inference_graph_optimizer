import torchvision.models as tvm


def build():
    # Pretrained weights: real BN statistics, and its Conv2d layers have bias=False.
    return tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
