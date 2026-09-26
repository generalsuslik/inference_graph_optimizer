# ONNX Inference Graph Optimizer

## Current optimizations
### Constant folding
1) Conv -> BatchNormalization
([paper](https://leimao.github.io/blog/Neural-Network-Batch-Normalization-Fusion/))
`FoldConvBn` (`fold-conv-bn`) bakes an inference-mode BatchNormalization into the
weight and bias of the Conv feeding it. BN is a per-channel affine map, so

```
scale = gamma / sqrt(var + eps)
W'    = W * scale                  (per output channel)
B'    = (B - mean) * scale + beta  (B = 0 if the Conv had no bias)
```
```
before:  x -> Conv -> intermediate -> BatchNormalization -> y
after:   x -> Conv(W', B') ----------------------------> y
```

Run it before `fuse-conv-relu` so that `Conv -> BatchNormalization -> Relu`
collapses into a single `FusedConvRelu`.

### Operator fusing
1) Conv -> ReLu

`FuseConvRelu` (`fuse-conv-relu`) collapses a convolution feeding directly into
a ReLU into a single `FusedConvRelu` node, removing one node and one
intermediate activation buffer per match.
```
before:  x -> Conv -> intermediate -> Relu -> y
after:   x -> FusedConvRelu ---------------> y
```
