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

2) Constant subexpressions

`FoldConstants` (`fold-constants`) evaluates every node whose inputs are all
initializers and turns its output into an initializer holding the result.
Exported with `do_constant_folding=False`, every Linear layer reads its weight
through a `Transpose` that runs on each inference; folding it leaves the MatMul
reading a pre-transposed constant.
```
before:  W -> Transpose -> Wt -> MatMul -> y      (W an initializer)
after:   Wt --------------------> MatMul -> y      (Wt an initializer)
```
It evaluates `Transpose`, `Sqrt`, `Relu`, and `Add`/`Sub`/`Mul`/`Div`/`Pow`
with numpy-style broadcasting. Only f32 tensors are folded, since that is all
the IR's tensors hold, so integer shape and mask arithmetic stays as it is.

### Operator fusing
The fusions are written as declarative patterns (`pattern.rs`). A name used
twice in a pattern must bind the same value, which is how a pattern says "this
value feeds two of my nodes"; `Add` and `Mul` match their inputs in either
order. A match is only reported if no intermediate it deletes is read outside
the match or is a graph output, so a rewrite never changes anything else.

1) Conv -> ReLu

`FuseConvRelu` (`fuse-conv-relu`) collapses a convolution feeding directly into
a ReLU into a single `FusedConvRelu` node, removing one node and one
intermediate activation buffer per match.
```
before:  x -> Conv -> intermediate -> Relu -> y
after:   x -> FusedConvRelu ---------------> y
```
ONNX has no standard fused convolution, so `FusedConvRelu` is exported as
onnxruntime's contrib op `com.microsoft:FusedConv` with `activation="Relu"`.
A model with it only runs on onnxruntime.

2) LayerNorm

`FuseLayerNorm` (`fuse-layer-norm`) collapses the decomposed LayerNorm that
exporters write below opset 17 into a single standard `LayerNormalization`:
```
d = x - ReduceMean(x)
y = d / Sqrt(ReduceMean(Pow(d, 2)) + eps) * gamma + beta
```
```
before:  x -> ReduceMean -> Sub -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Add -> y
         (9 nodes; x and d each feed two of them)
after:   x -> LayerNormalization(axis = -k, epsilon = eps) -> y
```
It only fuses a chain that really computes a LayerNorm:
- both `ReduceMean`s reduce the same trailing axes `-k..-1`, with `keepdims=1`;
- the exponent is exactly 2 and `eps` is a scalar;
- `gamma` and `beta` span exactly the normalized axes and have no size-1
  dimension that broadcasting could have stretched.

3) GELU

`FuseGelu` (`fuse-gelu`) collapses the exact (erf) GELU that exporters write
below opset 20 into a single standard `Gelu`:
```
y = x * (Erf(x / sqrt(2)) + 1) * 0.5
```
```
before:  x -> Div -> Erf -> Add -> Mul -> Mul -> y
         (5 nodes; x also feeds the first Mul)
after:   x -> Gelu -> y
```
The constants must be `sqrt(2)`, `1` and `0.5` (within 1e-6). The tanh
approximation is a different chain and is left alone.

#### Opset raises
`LayerNormalization` is standard from opset 17 and `Gelu` from opset 20, so a
fusion that emits one raises the model's default opset to that version if it
was older, and the exporter raises the IR version to match (opset 17 needs IR
8, opset 20 needs IR 9). `onnx-opt` prints the raise on its own line, because
the runtime must then support that opset.

The rest of the model is then read under the new opset, which onnx-opt does
not check. For example, from opset 18 `ReduceMean` takes its axes as an input
rather than an attribute, so a `ReduceMean` left outside a LayerNorm would be
invalid at opset 20. `onnx.checker` and the benchmark's differential test catch
this.

### Identity elimination
`EliminateIdentity` (`eliminate-identity`) removes Identity nodes by pointing
their consumers straight at the Identity's input. Exporters use Identity to
give one tensor a second name (the TorchScript exporter does it for
deduplicated initializers), which hides the constant from passes that expect an
initializer.
```
before:  W -> Identity -> W2 -> Conv
after:   W -----------------> Conv
```
An Identity whose output is a graph output stays, because graph outputs are
known by name.

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
iteration changes nothing (at most `--max-iterations`, 10 by default):

1. `eliminate-identity`, so later passes see initializers rather than
   Identity outputs
2. `fold-constants`, so the fusions see the results as initializers
3. `fold-conv-bn`
4. `fuse-conv-relu`, after `fold-conv-bn` so that `Conv -> BN -> Relu` becomes
   one node
5. `fuse-layer-norm`
6. `fuse-gelu`
7. `dce`, which removes what the rewrites left behind
