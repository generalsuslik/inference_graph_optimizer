use onnx_ir::{Graph, NodeId, OpType, ValueId};

use crate::pass::Pass;

/// Removes Identity nodes by pointing their consumers straight at the Identity's input.
///
/// Exporters use Identity to give one tensor a second name (the TorchScript exporter does it for
/// deduplicated initializers), which hides constants from passes that expect an initializer.
#[derive(Debug, Clone, Copy, Default)]
pub struct EliminateIdentity;

impl Pass for EliminateIdentity {
    fn name(&self) -> &'static str {
        "eliminate-identity"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        let mut removed = 0;
        for identity in graph.node_ids() {
            let Some((input, output)) = removable(graph, identity) else {
                continue;
            };
            // A node reading `output` twice is listed twice.
            let mut consumers = graph.value(output).map_or_else(Vec::new, |v| v.consumers.clone());
            consumers.sort();
            consumers.dedup();
            for consumer in consumers {
                let inputs = graph.node(consumer).map_or_else(Vec::new, |n| {
                    n.inputs.iter().map(|&v| if v == output { input } else { v }).collect()
                });
                graph.set_node_inputs(consumer, inputs);
            }
            graph.remove_node(identity);
            removed += 1;
        }
        removed
    }
}

/// If `identity` can be bypassed, returns its input and output.
fn removable(graph: &Graph, identity: NodeId) -> Option<(ValueId, ValueId)> {
    let node = graph.node(identity)?;
    if node.op != OpType::Identity {
        return None;
    }
    let (&[input], &[output]) = (&node.inputs[..], &node.outputs[..]) else {
        return None;
    };
    // Graph outputs are known by name, so the value has to keep its producer.
    if graph.is_output(output) {
        return None;
    }
    Some((input, output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_ir::{Attrs, Tensor};

    #[test]
    fn rewires_consumers_to_the_identity_input() {
        // x -> Identity -> alias -> Add(alias, alias) -> y
        let mut graph = Graph::new();
        let x = graph.add_value("x");
        let alias = graph.add_value("alias");
        let y = graph.add_value("y");
        let identity = graph.add_node(OpType::Identity, "id", vec![x], vec![alias], Attrs::default());
        let add = graph.add_node(OpType::Add, "add", vec![alias, alias], vec![y], Attrs::default());
        graph.inputs.push(x);
        graph.outputs.push(y);

        assert_eq!(EliminateIdentity.run(&mut graph), 1);

        assert!(graph.node(identity).is_none());
        assert_eq!(graph.node(add).unwrap().inputs, vec![x, x]);
        assert_eq!(graph.value(x).unwrap().consumers, vec![add, add]);
        assert!(graph.value(alias).unwrap().consumers.is_empty());
    }

    #[test]
    fn collapses_a_chain_in_one_run() {
        let mut graph = Graph::new();
        let x = graph.add_value("x");
        let a = graph.add_value("a");
        let b = graph.add_value("b");
        let y = graph.add_value("y");
        graph.add_node(OpType::Identity, "id_a", vec![x], vec![a], Attrs::default());
        graph.add_node(OpType::Identity, "id_b", vec![a], vec![b], Attrs::default());
        let relu = graph.add_node(OpType::Relu, "relu", vec![b], vec![y], Attrs::default());
        graph.inputs.push(x);
        graph.outputs.push(y);

        assert_eq!(EliminateIdentity.run(&mut graph), 2);

        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.node(relu).unwrap().inputs, vec![x]);
    }

    #[test]
    fn keeps_identity_that_produces_a_graph_output() {
        let mut graph = Graph::new();
        let x = graph.add_value("x");
        let y = graph.add_value("y");
        graph.add_node(OpType::Identity, "id", vec![x], vec![y], Attrs::default());
        graph.inputs.push(x);
        graph.outputs.push(y);

        assert_eq!(EliminateIdentity.run(&mut graph), 0);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn exposes_aliased_bn_statistics_to_fold_conv_bn() {
        use crate::FoldConvBn;

        // The TorchScript export of a fresh BN: running_var aliases gamma, running_mean aliases beta.
        let mut graph = Graph::new();
        let x = graph.add_value("x");
        let w = graph.add_initializer("w", Tensor::filled(vec![1, 1, 1, 1], 2.0));
        let gamma = graph.add_initializer("gamma", Tensor::filled(vec![1], 1.0));
        let beta = graph.add_initializer("beta", Tensor::filled(vec![1], 0.0));
        let var = graph.add_value("running_var");
        let mean = graph.add_value("running_mean");
        let hidden = graph.add_value("hidden");
        let y = graph.add_value("y");
        graph.add_node(OpType::Identity, "id_var", vec![gamma], vec![var], Attrs::default());
        graph.add_node(OpType::Identity, "id_mean", vec![beta], vec![mean], Attrs::default());
        graph.add_node(OpType::Conv, "conv", vec![x, w], vec![hidden], Attrs::default());
        graph.add_node(
            OpType::BatchNormalization,
            "bn",
            vec![hidden, gamma, beta, mean, var],
            vec![y],
            Attrs::default(),
        );
        graph.inputs.push(x);
        graph.outputs.push(y);

        assert_eq!(FoldConvBn.run(&mut graph), 0);
        assert_eq!(EliminateIdentity.run(&mut graph), 2);
        assert_eq!(FoldConvBn.run(&mut graph), 1);
        assert_eq!(graph.node_count(), 1);
    }
}
