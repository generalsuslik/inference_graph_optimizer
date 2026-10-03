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

### Dead code elimination
`DCE` (`dce`) removes every node whose results never reach a graph output, then
every initializer nothing reads any more, such as the old Conv weight and BN
parameters left behind by `fold-conv-bn`. It is a mark-and-sweep: walk back from
the graph outputs marking each producer as live, then delete the rest. This
drops a whole dead chain in one run, and keeps a multi-output node as long as
any one of its outputs is used. Initializers that are graph inputs or outputs
always stay.
```
before:  x -> Relu -> y          after:  x -> Relu -> y
         x -> a -> b -> unused
```

## Pipeline
`Pipeline::standard` runs the passes in this order, repeating until an
iteration changes nothing:

1. `fold-conv-bn`
2. `fuse-conv-relu`
3. `dce`
