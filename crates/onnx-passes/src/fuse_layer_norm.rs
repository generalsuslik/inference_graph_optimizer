use onnx_ir::{Attrs, Graph, NodeId, OpType, ValueId};

use crate::pass::Pass;
use crate::pattern::{Match, Pat, matches};

/// Collapses the decomposed LayerNorm that exporters write below opset 17 into a single
/// LayerNormalization:
///
/// ```text
/// d = x - ReduceMean(x)
/// y = d / Sqrt(ReduceMean(Pow(d, 2)) + eps) * gamma + beta
/// ```
///
/// `x` and `d` each feed two nodes of the chain, so this is the pattern that leans on the matcher's
/// shared names and its check that no intermediate is read outside the match.
///
/// LayerNormalization is standard from opset 17, so a fusion raises the graph's default opset to
/// [`LAYER_NORM_OPSET`] if it was older.
#[derive(Debug, Clone, Copy, Default)]
pub struct FuseLayerNorm;

/// The opset that standardized LayerNormalization.
pub const LAYER_NORM_OPSET: i64 = 17;

/// Every node of the pattern; the rewrite removes them all.
const NODES: [&str; 9] = ["mean", "d", "pow", "var", "var_eps", "std", "norm", "scaled", "y"];

impl Pass for FuseLayerNorm {
    fn name(&self) -> &'static str {
        "fuse-layer-norm"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        let pattern = pattern();
        let mut fused = 0;
        for root in graph.node_ids() {
            let Some(fusion) = matches(graph, &pattern, root).iter().find_map(|m| fusion(graph, m)) else {
                continue;
            };
            let outputs = graph.node(root).expect("matched nodes are live").outputs.clone();
            // Remove the chain first: its last Add still owns the producer link on the outputs the
            // fused node takes over.
            for node in fusion.nodes {
                graph.remove_node(node);
            }
            graph.add_node(
                OpType::LayerNormalization,
                format!("layer_norm_{}", root.0),
                vec![fusion.x, fusion.gamma, fusion.beta],
                outputs,
                Attrs::default()
                    .with_int("axis", fusion.axis)
                    .with_float("epsilon", fusion.epsilon),
            );
            fused += 1;
        }
        // The fused op only exists from opset 17 on, so the model has to declare it. Whether every
        // other op still means the same there is for onnx.checker and the differential test to say.
        if fused > 0 {
            graph.opset = graph.opset.map(|v| v.max(LAYER_NORM_OPSET));
        }
        fused
    }
}

fn pattern() -> Pat {
    let mean = Pat::node("mean", OpType::ReduceMean, [Pat::value("x")]);
    let d = Pat::node("d", OpType::Sub, [Pat::value("x"), mean]);
    let pow = Pat::node("pow", OpType::Pow, [Pat::value("d"), Pat::constant("two")]);
    let var = Pat::node("var", OpType::ReduceMean, [pow]);
    let var_eps = Pat::node("var_eps", OpType::Add, [var, Pat::constant("eps")]);
    let std = Pat::node("std", OpType::Sqrt, [var_eps]);
    let norm = Pat::node("norm", OpType::Div, [d, std]);
    let scaled = Pat::node("scaled", OpType::Mul, [norm, Pat::constant("gamma")]);
    Pat::node("y", OpType::Add, [scaled, Pat::constant("beta")])
}

struct Fusion {
    nodes: [NodeId; 9],
    x: ValueId,
    gamma: ValueId,
    beta: ValueId,
    axis: i64,
    epsilon: f32,
}

/// Checks what the pattern's shape cannot: that the chain really computes a LayerNorm.
fn fusion(graph: &Graph, m: &Match) -> Option<Fusion> {
    let attrs = |name| &graph.node(m.node(name)).expect("matched nodes are live").attrs;
    let (mean, var) = (attrs("mean"), attrs("var"));

    // Without keepdims the mean loses the reduced axes and `x - mean` broadcasts against the wrong ones.
    if [mean, var].iter().any(|a| a.get_int("keepdims").is_some_and(|k| k != 1)) {
        return None;
    }
    let axes = mean.get_ints("axes")?;
    if var.get_ints("axes") != Some(axes) {
        return None;
    }
    // LayerNormalization normalizes a trailing block of axes. Without the input's rank only negative
    // axes are known to be trailing, so they must be exactly -k..-1.
    let k = axes.len() as i64;
    let mut sorted = axes.to_vec();
    sorted.sort_unstable();
    if k == 0 || sorted != (-k..0).collect::<Vec<_>>() {
        return None;
    }

    let tensor = |name| graph.initializer(m.value(name)).expect("the pattern matched an initializer");
    let (two, eps, gamma, beta) = (tensor("two"), tensor("eps"), tensor("gamma"), tensor("beta"));
    if two.data != [2.0] || eps.numel() != 1 {
        return None;
    }
    // LayerNormalization applies gamma and beta over the normalized axes. Mul and Add broadcast them
    // over whatever they line up with, which is the same thing only if they span exactly those axes
    // and have no size-1 dimension that could have been stretched.
    if gamma.dims.len() != axes.len() || beta.dims != gamma.dims || gamma.dims.contains(&1) {
        return None;
    }

    Some(Fusion {
        nodes: NODES.map(|name| m.node(name)),
        x: m.value("x"),
        gamma: m.value("gamma"),
        beta: m.value("beta"),
        axis: -k,
        epsilon: eps.data[0],
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use onnx_ir::Tensor;

    struct Options {
        mean_axes: Vec<i64>,
        var_axes: Vec<i64>,
        keepdims: Option<i64>,
        exponent: f32,
        gamma_dims: Vec<usize>,
        /// Writes every Add and Mul with its inputs the other way round.
        swapped: bool,
    }

    impl Default for Options {
        fn default() -> Self {
            Options {
                mean_axes: vec![-1],
                var_axes: vec![-1],
                keepdims: None,
                exponent: 2.0,
                gamma_dims: vec![4],
                swapped: false,
            }
        }
    }

    /// torch's opset-13 export of `nn.LayerNorm`, with every node and value reachable by name.
    struct LayerNorm {
        graph: Graph,
        nodes: HashMap<&'static str, NodeId>,
        values: HashMap<&'static str, ValueId>,
    }

    fn layer_norm(o: Options) -> LayerNorm {
        let mut graph = Graph::new();
        let mut values = HashMap::new();
        let mut nodes = HashMap::new();

        let x = graph.add_value("x");
        graph.inputs.push(x);
        let numel = o.gamma_dims.iter().product();
        let gamma = graph.add_initializer("gamma", Tensor::new(o.gamma_dims.clone(), (1..=numel).map(|i| i as f32).collect()));
        let beta = graph.add_initializer("beta", Tensor::filled(o.gamma_dims.clone(), 0.5));
        let two = graph.add_initializer("two", Tensor::new(vec![], vec![o.exponent]));
        let eps = graph.add_initializer("eps", Tensor::new(vec![], vec![1e-5]));
        values.extend([("x", x), ("gamma", gamma), ("beta", beta), ("two", two), ("eps", eps)]);

        let reduce = |axes: &[i64]| {
            let attrs = Attrs::default().with_ints("axes", axes.to_vec());
            match o.keepdims {
                Some(k) => attrs.with_int("keepdims", k),
                None => attrs,
            }
        };
        let swap = |a, b| if o.swapped { vec![b, a] } else { vec![a, b] };
        let mut add = |graph: &mut Graph, op, name: &'static str, inputs, attrs| {
            let out = graph.add_value(name);
            nodes.insert(name, graph.add_node(op, name, inputs, vec![out], attrs));
            values.insert(name, out);
            out
        };

        let mean = add(&mut graph, OpType::ReduceMean, "mean", vec![x], reduce(&o.mean_axes));
        let d = add(&mut graph, OpType::Sub, "d", vec![x, mean], Attrs::default());
        let pow = add(&mut graph, OpType::Pow, "pow", vec![d, two], Attrs::default());
        let var = add(&mut graph, OpType::ReduceMean, "var", vec![pow], reduce(&o.var_axes));
        let var_eps = add(&mut graph, OpType::Add, "var_eps", swap(var, eps), Attrs::default());
        let std = add(&mut graph, OpType::Sqrt, "std", vec![var_eps], Attrs::default());
        let norm = add(&mut graph, OpType::Div, "norm", vec![d, std], Attrs::default());
        let scaled = add(&mut graph, OpType::Mul, "scaled", swap(norm, gamma), Attrs::default());
        let y = add(&mut graph, OpType::Add, "y", swap(scaled, beta), Attrs::default());
        graph.outputs.push(y);

        LayerNorm { graph, nodes, values }
    }

    fn assert_fused(mut ln: LayerNorm, axis: i64) {
        assert_eq!(FuseLayerNorm.run(&mut ln.graph), 1);

        assert_eq!(ln.graph.node_count(), 1);
        let (id, fused) = ln.graph.nodes().next().unwrap();
        assert_eq!(fused.op, OpType::LayerNormalization);
        assert_eq!(fused.inputs, vec![ln.values["x"], ln.values["gamma"], ln.values["beta"]]);
        assert_eq!(fused.outputs, vec![ln.values["y"]]);
        assert_eq!(fused.attrs.get_int("axis"), Some(axis));
        assert_eq!(fused.attrs.get_float("epsilon"), Some(1e-5));
        assert_eq!(ln.graph.value(ln.values["y"]).unwrap().producer, Some(id));
    }

    fn assert_untouched(mut ln: LayerNorm) {
        let before = ln.graph.node_count();
        assert_eq!(FuseLayerNorm.run(&mut ln.graph), 0);
        assert_eq!(ln.graph.node_count(), before);
    }

    #[test]
    fn fuses_torchs_decomposed_layer_norm() {
        assert_fused(layer_norm(Options::default()), -1);
    }

    #[test]
    fn fuses_with_commutative_inputs_swapped() {
        assert_fused(layer_norm(Options { swapped: true, ..Default::default() }), -1);
    }

    #[test]
    fn fuses_with_explicit_keepdims() {
        assert_fused(layer_norm(Options { keepdims: Some(1), ..Default::default() }), -1);
    }

    #[test]
    fn fuses_over_several_trailing_axes() {
        let options = Options {
            mean_axes: vec![-1, -2],
            var_axes: vec![-1, -2],
            gamma_dims: vec![3, 4],
            ..Default::default()
        };
        assert_fused(layer_norm(options), -2);
    }

    #[test]
    fn keeps_the_chain_when_the_centered_input_is_read_elsewhere() {
        // `d` already feeds two nodes of the match; a third reader outside it must block the fusion.
        let mut ln = layer_norm(Options::default());
        let leak = ln.graph.add_value("leak");
        ln.graph.add_node(OpType::Identity, "leak", vec![ln.values["d"]], vec![leak], Attrs::default());
        ln.graph.outputs.push(leak);
        assert_untouched(ln);
    }

    #[test]
    fn keeps_the_chain_when_an_intermediate_is_a_graph_output() {
        let mut ln = layer_norm(Options::default());
        ln.graph.outputs.push(ln.values["mean"]);
        assert_untouched(ln);
    }

    #[test]
    fn keeps_the_chain_when_the_two_means_differ() {
        assert_untouched(layer_norm(Options { var_axes: vec![-2], ..Default::default() }));
    }

    #[test]
    fn keeps_the_chain_for_axes_not_known_to_be_trailing() {
        let options = Options {
            mean_axes: vec![2],
            var_axes: vec![2],
            ..Default::default()
        };
        assert_untouched(layer_norm(options));
    }

    #[test]
    fn keeps_the_chain_without_keepdims() {
        assert_untouched(layer_norm(Options { keepdims: Some(0), ..Default::default() }));
    }

    #[test]
    fn keeps_the_chain_for_another_exponent() {
        assert_untouched(layer_norm(Options { exponent: 3.0, ..Default::default() }));
    }

    #[test]
    fn keeps_the_chain_when_gamma_could_have_broadcast() {
        assert_untouched(layer_norm(Options { gamma_dims: vec![1], ..Default::default() }));
    }

    #[test]
    fn keeps_the_chain_when_a_reduction_has_no_axes() {
        let mut ln = layer_norm(Options::default());
        let mean = ln.nodes["mean"];
        ln.graph.attrs_mut(mean).unwrap().0.remove("axes");
        assert_untouched(ln);
    }

    #[test]
    fn declares_the_opset_layer_normalization_needs() {
        let mut ln = layer_norm(Options::default());
        ln.graph.opset = Some(13);

        assert_eq!(FuseLayerNorm.run(&mut ln.graph), 1);
        assert_eq!(ln.graph.opset, Some(LAYER_NORM_OPSET));
    }

    #[test]
    fn keeps_a_newer_opset() {
        let mut ln = layer_norm(Options::default());
        ln.graph.opset = Some(18);

        assert_eq!(FuseLayerNorm.run(&mut ln.graph), 1);
        assert_eq!(ln.graph.opset, Some(18));
    }

    #[test]
    fn leaves_the_opset_alone_without_a_fusion() {
        let mut ln = layer_norm(Options { keepdims: Some(0), ..Default::default() });
        ln.graph.opset = Some(13);

        assert_eq!(FuseLayerNorm.run(&mut ln.graph), 0);
        assert_eq!(ln.graph.opset, Some(13));
    }

    #[test]
    fn standard_pipeline_leaves_one_node_and_drops_the_scalars() {
        let mut ln = layer_norm(Options::default());
        ln.graph.opset = Some(13);
        let report = crate::pipeline::Pipeline::standard(10).run(&mut ln.graph);

        assert_eq!(report.nodes_after, 1);
        assert_eq!(report.rewrites.get("fuse-layer-norm"), Some(&1));
        assert_eq!((report.opset_before, report.opset_after), (Some(13), Some(17)));
        assert!(ln.graph.initializer(ln.values["two"]).is_none());
        assert!(ln.graph.initializer(ln.values["eps"]).is_none());
    }
}
