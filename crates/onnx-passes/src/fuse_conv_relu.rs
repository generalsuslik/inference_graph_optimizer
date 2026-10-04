use onnx_ir::{Graph, OpType};

use crate::pass::Pass;
use crate::pattern::{Pat, matches};

#[derive(Debug, Clone, Copy, Default)]
pub struct FuseConvRelu;

impl Pass for FuseConvRelu {
    fn name(&self) -> &'static str {
        "fuse-conv-relu"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        // The Conv output disappears into the fused node; the matcher keeps it from being read elsewhere.
        let pattern = Pat::node("relu", OpType::Relu, [Pat::any_node("conv", OpType::Conv)]);
        let mut fused = 0;
        for relu in graph.node_ids() {
            let Some(m) = matches(graph, &pattern, relu).into_iter().next() else {
                continue;
            };
            let conv = m.node("conv");
            let outputs = graph.node(relu).expect("matched nodes are live").outputs.clone();
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

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_ir::{Attrs, NodeId, ValueId};

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
