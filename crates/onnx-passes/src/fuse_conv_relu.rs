use onnx_ir::{Graph, NodeId, OpType, ValueId};

use crate::pass::Pass;

#[derive(Debug, Clone, Copy, Default)]
pub struct FuseConvRelu;

impl Pass for FuseConvRelu {
    fn name(&self) -> &'static str {
        "fuse-conv-relu"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        let mut fused = 0;
        for relu in graph.node_ids() {
            let Some((conv, outputs)) = fusable(graph, relu) else {
                continue;
            };
            // Drop the Relu first: it still owns the producer link on the outputs the
            // Conv is about to take over.
            graph.remove_node(relu);
            graph.set_op(conv, OpType::FusedConvRelu);
            graph.set_node_outputs(conv, outputs);
            fused += 1;
        }
        fused
    }
}

/// If `relu` can absorb into its producing Conv, returns that Conv and the outputs it
/// should take over.
fn fusable(graph: &Graph, relu: NodeId) -> Option<(NodeId, Vec<ValueId>)> {
    let relu = graph.node(relu)?;
    if relu.op != OpType::Relu {
        return None;
    }
    let [input] = relu.inputs[..] else {
        return None;
    };

    // The Conv output disappears into the fused node, so nothing else may need it.
    if graph.is_output(input) {
        return None;
    }
    let value = graph.value(input)?;
    if value.consumers.len() != 1 {
        return None;
    }

    let conv = value.producer?;
    if graph.node(conv)?.op != OpType::Conv {
        return None;
    }
    Some((conv, relu.outputs.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_ir::Attrs;

    /// Builds `x -> Conv -> hidden -> Relu -> y`, returning the two node ids.
    fn conv_relu() -> (Graph, NodeId, NodeId) {
        let mut graph = Graph::new();
        let x = graph.add_value("x");
        let hidden = graph.add_value("hidden");
        let y = graph.add_value("y");
        let conv = graph.add_node(OpType::Conv, "conv", vec![x], vec![hidden], Attrs::default());
        let relu = graph.add_node(OpType::Relu, "relu", vec![hidden], vec![y], Attrs::default());
        graph.inputs.push(x);
        graph.outputs.push(y);
        (graph, conv, relu)
    }

    #[test]
    fn fuses_conv_into_relu() {
        let (mut graph, conv, relu) = conv_relu();
        let y = graph.outputs[0];

        assert_eq!(FuseConvRelu.run(&mut graph), 1);

        assert!(graph.node(relu).is_none());
        assert_eq!(graph.node_count(), 1);

        let fused = graph.node(conv).expect("conv survives the fusion");
        assert_eq!(fused.op, OpType::FusedConvRelu);
        assert_eq!(fused.outputs, vec![y]);
        assert_eq!(graph.value(y).unwrap().producer, Some(conv));
    }

    #[test]
    fn keeps_conv_output_with_a_second_consumer() {
        let (mut graph, _, _) = conv_relu();
        let hidden = ValueId(1);
        let side = graph.add_value("side");
        graph.add_node(OpType::Identity, "id", vec![hidden], vec![side], Attrs::default());

        assert_eq!(FuseConvRelu.run(&mut graph), 0);
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn keeps_conv_output_that_escapes_the_graph() {
        let (mut graph, _, _) = conv_relu();
        graph.outputs.push(ValueId(1));

        assert_eq!(FuseConvRelu.run(&mut graph), 0);
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn ignores_relu_without_a_conv_producer() {
        let mut graph = Graph::new();
        let x = graph.add_value("x");
        let y = graph.add_value("y");
        graph.add_node(OpType::Relu, "relu", vec![x], vec![y], Attrs::default());
        graph.inputs.push(x);
        graph.outputs.push(y);

        assert_eq!(FuseConvRelu.run(&mut graph), 0);
        assert_eq!(graph.node_count(), 1);
    }
}
