use onnx_ir::{Graph, NodeId, OpType, Tensor, ValueId};

use crate::pass::Pass;
use crate::pattern::{Pat, matches};

/// Folds an inference-mode BatchNormalization into the Conv that feeds it.
///
/// BN is a per-channel affine map, so it bakes into the Conv's weight and bias:
/// `scale = gamma / sqrt(var + eps)`, `W' = W * scale`, `B' = (B - mean) * scale + beta`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FoldConvBn;

impl Pass for FoldConvBn {
    fn name(&self) -> &'static str {
        "fold-conv-bn"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        let patterns = patterns();
        let mut folded = 0;
        for bn in graph.node_ids() {
            let Some(fold) = foldable(graph, &patterns, bn) else {
                continue;
            };
            let weight = graph.add_initializer(format!("fold_conv_bn_{}_w", fold.conv.0), fold.weight);
            let bias = graph.add_initializer(format!("fold_conv_bn_{}_b", fold.conv.0), fold.bias);
            // Drop the BatchNormalization first: it still owns the producer link on the outputs the
            // Conv is about to take over.
            graph.remove_node(bn);
            graph.set_node_inputs(fold.conv, vec![fold.input, weight, bias]);
            graph.set_node_outputs(fold.conv, fold.outputs);
            folded += 1;
        }
        folded
    }
}

struct Fold {
    conv: NodeId,
    input: ValueId,
    outputs: Vec<ValueId>,
    weight: Tensor,
    bias: Tensor, 
}

/// `Conv(x, w[, b]) -> BatchNormalization(gamma, beta, mean, var)`, one pattern per Conv arity.
/// The Conv output turns into the folded result, so the matcher keeps it from being read elsewhere.
fn patterns() -> [Pat; 2] {
    let bn = |conv| {
        Pat::node(
            "bn",
            OpType::BatchNormalization,
            [conv, Pat::constant("gamma"), Pat::constant("beta"), Pat::constant("mean"), Pat::constant("var")],
        )
    };
    [
        bn(Pat::node("conv", OpType::Conv, [Pat::value("x"), Pat::constant("w"), Pat::constant("b")])),
        bn(Pat::node("conv", OpType::Conv, [Pat::value("x"), Pat::constant("w")])),
    ]
}

fn foldable(graph: &Graph, patterns: &[Pat], bn: NodeId) -> Option<Fold> {
    let m = patterns.iter().flat_map(|p| matches(graph, p, bn)).next()?;
    let bn = graph.node(bn)?;
    if bn.attrs.get_int("training_mode") == Some(1) || bn.outputs.len() != 1 {
        return None;
    }

    let tensor = |name| graph.initializer(m.value(name));
    let weight = tensor("w")?;
    let bias = m.get("b").and_then(|b| graph.initializer(b));
    let [Some(gamma), Some(beta), Some(mean), Some(var)] = ["gamma", "beta", "mean", "var"].map(tensor) else {
        return None;
    };

    let channels = *weight.dims.first()?;
    if weight.numel() == 0 || [gamma, beta, mean, var].iter().any(|t| t.numel() != channels) || bias.is_some_and(|b| b.numel() != channels) {
        return None;
    }

    let eps = bn.attrs.get_float("epsilon").unwrap_or(1e-5);
    let scale: Vec<f32> = gamma
        .data
        .iter()
        .zip(&var.data)
        .map(|(g, v)| g / (v + eps).sqrt())
        .collect();

    let mut folded_weight = weight.data.clone();
    for (channel, scale) in folded_weight.chunks_mut(weight.numel() / channels).zip(&scale) {
        for w in channel {
            *w *= *scale;
        }
    }
    let folded_bias = (0..channels)
        .map(|c| {
            let b = bias.as_ref().map_or(0.0, |b| b.data[c]);
            (b - mean.data[c]) * scale[c] + beta.data[c]
        })
        .collect();
    Some(Fold {
        conv: m.node("conv"),
        input: m.value("x"),
        outputs: bn.outputs.clone(),
        weight: Tensor::new(weight.dims.clone(), folded_weight),
        bias: Tensor::new(vec![channels], folded_bias),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_ir::{Attr, Attrs};

    /// Builds `x -> Conv -> hidden -> BatchNormalization -> y` with two output channels,
    /// returning the Conv and the `hidden` value.
    ///
    /// With `epsilon = 1` the BN scale works out exactly to `[4 / sqrt(4), 3 / sqrt(1)] = [2, 3]`.
    fn conv_bn(with_bias: bool) -> (Graph, NodeId, ValueId) {
        let mut graph = Graph::new();
        let x = graph.add_value("x");
        let w = graph.add_initializer("w", Tensor::new(vec![2, 1, 1, 1], vec![1.0, 2.0]));
        let mut conv_inputs = vec![x, w];
        if with_bias {
            conv_inputs.push(graph.add_initializer("b", Tensor::new(vec![2], vec![0.5, -1.0])));
        }
        let scale = graph.add_initializer("scale", Tensor::new(vec![2], vec![4.0, 3.0]));
        let beta = graph.add_initializer("beta", Tensor::new(vec![2], vec![0.0, 3.0]));
        let mean = graph.add_initializer("mean", Tensor::new(vec![2], vec![1.0, 0.0]));
        let var = graph.add_initializer("var", Tensor::new(vec![2], vec![3.0, 0.0]));
        let hidden = graph.add_value("hidden");
        let y = graph.add_value("y");

        let conv = graph.add_node(OpType::Conv, "conv", conv_inputs, vec![hidden], Attrs::default());
        graph.add_node(
            OpType::BatchNormalization,
            "bn",
            vec![hidden, scale, beta, mean, var],
            vec![y],
            Attrs::default().with_float("epsilon", 1.0),
        );
        graph.inputs.push(x);
        graph.outputs.push(y);
        (graph, conv, hidden)
    }

    /// Data of the initializer feeding input `slot` of `node`.
    fn input_data(graph: &Graph, node: NodeId, slot: usize) -> &[f32] {
        let value = graph.node(node).unwrap().inputs[slot];
        &graph.initializer(value).expect("folded input is a constant").data
    }

    #[test]
    fn folds_bn_into_conv_weight_and_bias() {
        let (mut graph, conv, _) = conv_bn(true);
        let x = graph.inputs[0];
        let y = graph.outputs[0];

        assert_eq!(FoldConvBn.run(&mut graph), 1);

        assert_eq!(graph.node_count(), 1);
        let folded = graph.node(conv).expect("conv survives the fold");
        assert_eq!(folded.op, OpType::Conv);
        assert_eq!(folded.inputs[0], x);
        assert_eq!(folded.outputs, vec![y]);
        assert_eq!(graph.value(y).unwrap().producer, Some(conv));

        assert_eq!(input_data(&graph, conv, 1), [2.0, 6.0]);
        // (0.5 - 1) * 2 + 0, (-1 - 0) * 3 + 3
        assert_eq!(input_data(&graph, conv, 2), [-1.0, 0.0]);
    }

    #[test]
    fn adds_a_bias_to_a_conv_without_one() {
        let (mut graph, conv, _) = conv_bn(false);

        assert_eq!(FoldConvBn.run(&mut graph), 1);

        assert_eq!(graph.node(conv).unwrap().inputs.len(), 3);
        // (0 - 1) * 2 + 0, (0 - 0) * 3 + 3
        assert_eq!(input_data(&graph, conv, 2), [-2.0, 3.0]);
    }

    #[test]
    fn folds_regardless_of_padding_and_stride() {
        // BN acts on every Conv output, border ones included, and padded zeros stay zero
        // under W', so the fold is exact whatever the Conv geometry.
        let (mut graph, conv, _) = conv_bn(true);
        let attrs = &mut graph.attrs_mut(conv).unwrap().0;
        attrs.insert("pads".into(), Attr::Ints(vec![1, 1, 1, 1]));
        attrs.insert("strides".into(), Attr::Ints(vec![2, 2]));

        assert_eq!(FoldConvBn.run(&mut graph), 1);

        let folded = graph.node(conv).unwrap();
        assert_eq!(folded.attrs.get_ints("pads"), Some(&[1, 1, 1, 1][..]));
        assert_eq!(folded.attrs.get_ints("strides"), Some(&[2, 2][..]));
        assert_eq!(input_data(&graph, conv, 1), [2.0, 6.0]);
        assert_eq!(input_data(&graph, conv, 2), [-1.0, 0.0]);
    }

    #[test]
    fn keeps_conv_output_with_a_second_consumer() {
        let (mut graph, _, hidden) = conv_bn(true);
        let side = graph.add_value("side");
        graph.add_node(OpType::Identity, "id", vec![hidden], vec![side], Attrs::default());

        assert_eq!(FoldConvBn.run(&mut graph), 0);
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn keeps_conv_output_that_escapes_the_graph() {
        let (mut graph, _, hidden) = conv_bn(true);
        graph.outputs.push(hidden);

        assert_eq!(FoldConvBn.run(&mut graph), 0);
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn ignores_bn_in_training_mode() {
        let (mut graph, _, hidden) = conv_bn(true);
        let bn = graph.value(hidden).unwrap().consumers[0];
        graph.attrs_mut(bn).unwrap().0.insert("training_mode".into(), Attr::Int(1));

        assert_eq!(FoldConvBn.run(&mut graph), 0);
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn ignores_conv_with_non_constant_weight() {
        let mut graph = Graph::new();
        let x = graph.add_value("x");
        let w = graph.add_value("w");
        let hidden = graph.add_value("hidden");
        let params: Vec<_> = ["scale", "beta", "mean", "var"]
            .into_iter()
            .map(|name| graph.add_initializer(name, Tensor::filled(vec![1], 1.0)))
            .collect();
        let y = graph.add_value("y");
        graph.add_node(OpType::Conv, "conv", vec![x, w], vec![hidden], Attrs::default());
        let mut bn_inputs = vec![hidden];
        bn_inputs.extend(params);
        graph.add_node(OpType::BatchNormalization, "bn", bn_inputs, vec![y], Attrs::default());
        graph.inputs.extend([x, w]);
        graph.outputs.push(y);

        assert_eq!(FoldConvBn.run(&mut graph), 0);
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn ignores_bn_after_a_fused_conv_relu() {
        // The Relu sits between the Conv and the BN, so folding would change the result.
        let (mut graph, conv, _) = conv_bn(true);
        graph.set_op(conv, OpType::FusedConvRelu);

        assert_eq!(FoldConvBn.run(&mut graph), 0);
        assert_eq!(graph.node_count(), 2);
    }
}
