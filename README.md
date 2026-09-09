# ONNX Inference Graph Optimizer

## Current optimizations
### Operator fusing
1) Conv -> ReLu

`FuseConvRelu` (`fuse-conv-relu`) collapses a convolution feeding directly into
a ReLU into a single `FusedConvRelu` node, removing one node and one
intermediate activation buffer per match.
```
before:  x ─> Conv ─> hidden -> Relu -> y
after:   x -> FusedConvRelu ----------> y
```