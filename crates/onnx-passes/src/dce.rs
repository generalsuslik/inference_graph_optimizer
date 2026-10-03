use std::collections::HashSet;

use onnx_ir::{Graph, NodeId, ValueId};

use crate::pass::Pass;

/// Removes every node whose results never reach a graph output, then every initializer
/// that nothing reads any more.
#[derive(Debug, Clone, Copy, Default)]
pub struct DCE;

impl Pass for DCE {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        let live = live_nodes(graph);
        let mut removed = 0;
        for node in graph.node_ids() {
            if !live.contains(&node) {
                graph.remove_node(node);
                removed += 1;
            }
        }

        let dead: Vec<ValueId> = graph
            .initializers()
            .map(|(id, _)| id)
            .filter(|&id| is_dead_initializer(graph, id))
            .collect();
        for id in dead {
            graph.remote_initializer(id);
            removed += 1;
        }
        removed
    }
}

fn live_nodes(graph: &Graph) -> HashSet<NodeId> {
    let mut live = HashSet::new();
    let mut pending = graph.outputs.clone();
    while let Some(value) = pending.pop() {
        let Some(producer) = graph.value(value).and_then(|v| v.producer) else {
            continue;
        };
        if live.insert(producer) && let Some(node) = graph.node(producer) {
            pending.extend(node.inputs.iter().copied());
        }
    }
    live
}

fn is_dead_initializer(graph: &Graph, id: ValueId) -> bool {
    graph.value(id).is_some_and(|v| v.consumers.is_empty())
        && !graph.is_output(id)
        && !graph.inputs.contains(&id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_ir::{Attrs, OpType, Tensor};

    #[test]
    fn leaves_a_clean_graph_alone() {
        let mut graph = Graph::new();
        let x = graph.add_value("x");
        let y = graph.add_value("y");
        graph.add_node(OpType::Relu, "relu", vec![x], vec![y], Attrs::default());
        graph.inputs.push(x);
        graph.outputs.push(y);

        assert_eq!(DCE.run(&mut graph), 0);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn removes_a_whole_dead_chain_in_one_run() {
        // x -> relu -> y is live; x -> a -> b -> unused hangs off the side.
        let mut graph = Graph::new();
        let x = graph.add_value("x");
        let y = graph.add_value("y");
        let mid = graph.add_value("mid");
        let unused = graph.add_value("unused");
        let relu = graph.add_node(OpType::Relu, "relu", vec![x], vec![y], Attrs::default());
        let a = graph.add_node(OpType::Relu, "a", vec![x], vec![mid], Attrs::default());
        let b = graph.add_node(OpType::Relu, "b", vec![mid], vec![unused], Attrs::default());
        graph.inputs.push(x);
        graph.outputs.push(y);

        assert_eq!(DCE.run(&mut graph), 2);

        assert!(graph.node(relu).is_some());
        assert!(graph.node(a).is_none());
        assert!(graph.node(b).is_none());
        assert_eq!(graph.value(x).unwrap().consumers, vec![relu]);
    }

    #[test]
    fn keeps_a_multi_output_node_when_one_output_is_used() {
        let mut graph = Graph::new();
        let x = graph.add_value("x");
        let used = graph.add_value("used");
        let unused = graph.add_value("unused");
        let split = graph.add_node(
            OpType::Other("Split".into()),
            "split",
            vec![x],
            vec![used, unused],
            Attrs::default(),
        );
        graph.inputs.push(x);
        graph.outputs.push(used);

        assert_eq!(DCE.run(&mut graph), 0);
        assert!(graph.node(split).is_some());
    }

    #[test]
    fn cleans_up_after_fold_conv_bn() {
        use crate::FoldConvBn;

        let mut graph = Graph::new();
        let x = graph.add_value("x");
        let w = graph.add_initializer("w", Tensor::filled(vec![1, 1, 1, 1], 1.0));
        let params: Vec<_> = ["gamma", "beta", "mean", "var"]
            .into_iter()
            .map(|name| graph.add_initializer(name, Tensor::filled(vec![1], 1.0)))
            .collect();
        let hidden = graph.add_value("hidden");
        let y = graph.add_value("y");
        graph.add_node(OpType::Conv, "conv", vec![x, w], vec![hidden], Attrs::default());
        let mut bn_inputs = vec![hidden];
        bn_inputs.extend(&params);
        graph.add_node(OpType::BatchNormalization, "bn", bn_inputs, vec![y], Attrs::default());
        graph.inputs.push(x);
        graph.outputs.push(y);

        assert_eq!(FoldConvBn.run(&mut graph), 1);
        // The old weight plus the four BN parameters are now unread.
        assert_eq!(DCE.run(&mut graph), 5);
        assert_eq!(graph.initializers().count(), 2);
    }
}