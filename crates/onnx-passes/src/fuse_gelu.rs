use std::f32::consts::SQRT_2;

use onnx_ir::{Attrs, Graph, NodeId, OpType, ValueId};

use crate::pass::Pass;
use crate::pattern::{Match, Pat, matches};

/// Collapses the exact (erf) GELU that exporters write below opset 20 into a single Gelu:
///
/// y = x * (Erf(x / sqrt(2)) + 1) * 0.5
///
/// `x` feeds both the Div and the first Mul. The tanh approximation is a different chain and is
/// left alone.
///
/// Gelu is standard from opset 20, so a fusion raises the graph's default opset to [`GELU_OPSET`]
/// if it was older.
#[derive(Debug, Clone, Copy, Default)]
pub struct FuseGelu;

/// The opset that standardized Gelu.
pub const GELU_OPSET: i64 = 20;

/// Every node of the pattern; the rewrite removes them all.
const NODES: [&str; 5] = ["div", "erf", "plus_one", "times_x", "y"];

impl Pass for FuseGelu {
    fn name(&self) -> &'static str {
        "fuse-gelu"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        let pattern = pattern();
        let mut fused = 0;
        for root in graph.node_ids() {
            let Some((nodes, x)) = matches(graph, &pattern, root).iter().find_map(|m| fusion(graph, m)) else {
                continue;
            };
            let outputs = graph.node(root).expect("matched nodes are live").outputs.clone();
            // Remove the chain first: its last Mul still owns the producer link on the outputs the
            // fused node takes over.
            for node in nodes {
                graph.remove_node(node);
            }
            graph.add_node(OpType::Gelu, format!("gelu_{}", root.0), vec![x], outputs, Attrs::default());
            fused += 1;
        }
        // The fused op only exists from opset 20 on, so the model has to declare it. Whether every
        // other op still means the same there is for onnx.checker and the differential test to say.
        if fused > 0 {
            graph.opset = graph.opset.map(|v| v.max(GELU_OPSET));
        }
        fused
    }
}

fn pattern() -> Pat {
    let div = Pat::node("div", OpType::Div, [Pat::value("x"), Pat::constant("sqrt2")]);
    let erf = Pat::node("erf", OpType::Erf, [div]);
    let plus_one = Pat::node("plus_one", OpType::Add, [erf, Pat::constant("one")]);
    let times_x = Pat::node("times_x", OpType::Mul, [Pat::value("x"), plus_one]);
    Pat::node("y", OpType::Mul, [times_x, Pat::constant("half")])
}

/// Checks what the pattern's shape cannot: that the constants make the chain a GELU.
fn fusion(graph: &Graph, m: &Match) -> Option<([NodeId; 5], ValueId)> {
    let scalar = |name, expected: f32| {
        graph
            .initializer(m.value(name))
            .is_some_and(|t| t.numel() == 1 && (t.data[0] - expected).abs() <= 1e-6)
    };
    if !(scalar("sqrt2", SQRT_2) && scalar("one", 1.0) && scalar("half", 0.5)) {
        return None;
    }
    Some((NODES.map(|name| m.node(name)), m.value("x")))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use onnx_ir::Tensor;

    struct Options {
        sqrt2: f32,
        one: f32,
        half: f32,
        /// Writes every Add and Mul with its inputs the other way round.
        swapped: bool,
        /// Shape of the `half` constant.
        half_dims: Vec<usize>,
    }

    impl Default for Options {
        fn default() -> Self {
            Options {
                sqrt2: SQRT_2,
                one: 1.0,
                half: 0.5,
                swapped: false,
                half_dims: vec![],
            }
        }
    }

    /// torch's opset-13 export of `nn.GELU()`, with every value reachable by name.
    struct Gelu {
        graph: Graph,
        values: HashMap<&'static str, ValueId>,
    }

    fn gelu(o: Options) -> Gelu {
        let mut graph = Graph::new();
        let mut values = HashMap::new();

        let x = graph.add_value("x");
        graph.inputs.push(x);
        let sqrt2 = graph.add_initializer("sqrt2", Tensor::new(vec![], vec![o.sqrt2]));
        let one = graph.add_initializer("one", Tensor::new(vec![], vec![o.one]));
        let numel = o.half_dims.iter().product();
        let half = graph.add_initializer("half", Tensor::new(o.half_dims.clone(), vec![o.half; numel]));
        values.extend([("x", x), ("sqrt2", sqrt2), ("one", one), ("half", half)]);

        let swap = |a, b| if o.swapped { vec![b, a] } else { vec![a, b] };
        let mut add = |graph: &mut Graph, op, name: &'static str, inputs| {
            let out = graph.add_value(name);
            graph.add_node(op, name, inputs, vec![out], Attrs::default());
            values.insert(name, out);
            out
        };

        let div = add(&mut graph, OpType::Div, "div", vec![x, sqrt2]);
        let erf = add(&mut graph, OpType::Erf, "erf", vec![div]);
        let plus_one = add(&mut graph, OpType::Add, "plus_one", swap(erf, one));
        let times_x = add(&mut graph, OpType::Mul, "times_x", swap(x, plus_one));
        let y = add(&mut graph, OpType::Mul, "y", swap(times_x, half));
        graph.outputs.push(y);

        Gelu { graph, values }
    }

    fn assert_fused(mut g: Gelu) {
        assert_eq!(FuseGelu.run(&mut g.graph), 1);

        let fused: Vec<_> = g.graph.nodes().filter(|(_, n)| n.op == OpType::Gelu).collect();
        let [(id, node)] = fused[..] else {
            panic!("expected one Gelu");
        };
        assert_eq!(node.inputs, vec![g.values["x"]]);
        assert_eq!(node.outputs, vec![g.values["y"]]);
        assert_eq!(g.graph.value(g.values["y"]).unwrap().producer, Some(id));
        for name in ["div", "erf", "plus_one", "times_x"] {
            assert!(g.graph.value(g.values[name]).unwrap().producer.is_none(), "`{name}` survived");
        }
    }

    fn assert_untouched(mut g: Gelu) {
        let before = g.graph.node_count();
        assert_eq!(FuseGelu.run(&mut g.graph), 0);
        assert_eq!(g.graph.node_count(), before);
    }

    #[test]
    fn fuses_torchs_erf_gelu() {
        let g = gelu(Options::default());
        assert_eq!(g.graph.node_count(), 5);
        assert_fused(g);
    }

    #[test]
    fn fuses_with_commutative_inputs_swapped() {
        assert_fused(gelu(Options { swapped: true, ..Default::default() }));
    }

    #[test]
    fn fuses_when_x_has_other_readers() {
        // `x` is an input to the fused node, not an intermediate, so it may feed anything else.
        let mut g = gelu(Options::default());
        let side = g.graph.add_value("side");
        g.graph.add_node(OpType::Relu, "side", vec![g.values["x"]], vec![side], Attrs::default());
        g.graph.outputs.push(side);
        assert_fused(g);
    }

    #[test]
    fn keeps_the_chain_when_an_intermediate_is_read_elsewhere() {
        let mut g = gelu(Options::default());
        let leak = g.graph.add_value("leak");
        g.graph.add_node(OpType::Identity, "leak", vec![g.values["erf"]], vec![leak], Attrs::default());
        g.graph.outputs.push(leak);
        assert_untouched(g);
    }

    #[test]
    fn keeps_the_chain_when_an_intermediate_is_a_graph_output() {
        let mut g = gelu(Options::default());
        g.graph.outputs.push(g.values["times_x"]);
        assert_untouched(g);
    }

    #[test]
    fn keeps_the_chain_for_other_constants() {
        assert_untouched(gelu(Options { sqrt2: 2.0, ..Default::default() }));
        assert_untouched(gelu(Options { one: 2.0, ..Default::default() }));
        assert_untouched(gelu(Options { half: 0.25, ..Default::default() }));
    }

    #[test]
    fn keeps_the_chain_when_a_constant_is_not_a_scalar() {
        assert_untouched(gelu(Options { half_dims: vec![2], ..Default::default() }));
    }

    #[test]
    fn declares_the_opset_gelu_needs() {
        let mut g = gelu(Options::default());
        g.graph.opset = Some(17);
        assert_eq!(FuseGelu.run(&mut g.graph), 1);
        assert_eq!(g.graph.opset, Some(GELU_OPSET));

        let mut untouched = gelu(Options { half: 0.25, ..Default::default() });
        untouched.graph.opset = Some(17);
        assert_eq!(FuseGelu.run(&mut untouched.graph), 0);
        assert_eq!(untouched.graph.opset, Some(17));
    }

    #[test]
    fn standard_pipeline_leaves_one_node_and_drops_the_constants() {
        let mut g = gelu(Options::default());
        g.graph.opset = Some(13);
        let report = crate::pipeline::Pipeline::standard(10).run(&mut g.graph);

        assert_eq!(report.nodes_after, 1);
        assert_eq!(report.rewrites.get("fuse-gelu"), Some(&1));
        assert_eq!((report.opset_before, report.opset_after), (Some(13), Some(20)));
        for name in ["sqrt2", "one", "half"] {
            assert!(g.graph.initializer(g.values[name]).is_none(), "`{name}` survived DCE");
        }
    }
}
