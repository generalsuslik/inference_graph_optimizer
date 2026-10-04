use onnx_ir::{Attrs, Graph, OpType, Tensor};

use crate::pass::Pass;

/// Evaluates nodes whose inputs are all constants and turns their output into an initializer.
///
/// Exported without the exporter's own constant folding, every Linear layer reads its weight
/// through a `Transpose` that runs on each inference; folding it leaves the MatMul reading a
/// pre-transposed constant. Only f32 is folded, since that is all the IR's tensors hold: integer
/// shape and mask arithmetic stays as it is.
#[derive(Debug, Clone, Copy, Default)]
pub struct FoldConstants;

impl Pass for FoldConstants {
    fn name(&self) -> &'static str {
        "fold-constants"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        let mut folded = 0;
        // Creation order is topological for imported graphs, so a chain of constant nodes folds
        // in a single run.
        for id in graph.node_ids() {
            let node = graph.node(id).expect("node_ids yields live nodes");
            let [output] = node.outputs[..] else {
                continue;
            };
            let Some(inputs) = node
                .inputs
                .iter()
                .map(|&v| graph.initializer(v))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let Some(tensor) = evaluate(&node.op, &node.attrs, &inputs) else {
                continue;
            };
            graph.remove_node(id);
            graph.set_initializer(output, tensor);
            folded += 1;
        }
        folded
    }
}

/// The result of `op` on constant `inputs`, or `None` for an op or input this pass cannot evaluate.
fn evaluate(op: &OpType, attrs: &Attrs, inputs: &[&Tensor]) -> Option<Tensor> {
    match (op, inputs) {
        (OpType::Transpose, [x]) => transpose(x, attrs.get_ints("perm")),
        (OpType::Sqrt, [x]) => Some(map(x, f32::sqrt)),
        (OpType::Relu, [x]) => Some(map(x, |v| v.max(0.0))),
        (OpType::Add, [a, b]) => broadcast(a, b, |x, y| x + y),
        (OpType::Sub, [a, b]) => broadcast(a, b, |x, y| x - y),
        (OpType::Mul, [a, b]) => broadcast(a, b, |x, y| x * y),
        (OpType::Div, [a, b]) => broadcast(a, b, |x, y| x / y),
        (OpType::Pow, [a, b]) => broadcast(a, b, f32::powf),
        _ => None,
    }
}

fn map(x: &Tensor, f: impl Fn(f32) -> f32) -> Tensor {
    Tensor::new(x.dims.clone(), x.data.iter().map(|&v| f(v)).collect())
}

/// Row-major strides of `dims`.
fn strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Visits every index of a `dims`-shaped tensor in row-major order, passing the offset each
/// `strides` set gives it.
fn for_each_offset<const N: usize>(dims: &[usize], strides: [&[usize]; N], mut f: impl FnMut([usize; N])) {
    let total: usize = dims.iter().product();
    let mut index = vec![0; dims.len()];
    for _ in 0..total {
        f(strides.map(|s| index.iter().zip(s).map(|(i, s)| i * s).sum()));
        for axis in (0..dims.len()).rev() {
            index[axis] += 1;
            if index[axis] < dims[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

/// ONNX Transpose: output axis `i` is input axis `perm[i]`, reversing the axes when `perm` is absent.
fn transpose(x: &Tensor, perm: Option<&[i64]>) -> Option<Tensor> {
    let rank = x.dims.len();
    let perm: Vec<usize> = match perm {
        Some(perm) => perm.iter().map(|&p| usize::try_from(p).ok()).collect::<Option<_>>()?,
        None => (0..rank).rev().collect(),
    };
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    if sorted != (0..rank).collect::<Vec<_>>() {
        return None;
    }

    let dims: Vec<usize> = perm.iter().map(|&p| x.dims[p]).collect();
    let input_strides = strides(&x.dims);
    // Walking the output in order, axis `i` moves through the input along axis `perm[i]`.
    let along: Vec<usize> = perm.iter().map(|&p| input_strides[p]).collect();
    let mut data = Vec::with_capacity(x.numel());
    for_each_offset(&dims, [&along], |[offset]| data.push(x.data[offset]));
    Some(Tensor::new(dims, data))
}

/// An elementwise op under ONNX (numpy) multidirectional broadcasting, or `None` if the shapes
/// do not broadcast.
fn broadcast(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32) -> Option<Tensor> {
    let rank = a.dims.len().max(b.dims.len());
    // Shapes align at their last axis; a missing leading axis acts as size 1.
    let padded = |dims: &[usize]| [vec![1; rank - dims.len()], dims.to_vec()].concat();
    let (da, db) = (padded(&a.dims), padded(&b.dims));
    let dims = da
        .iter()
        .zip(&db)
        .map(|(&x, &y)| match (x, y) {
            _ if x == y => Some(x),
            (1, _) => Some(y),
            (_, 1) => Some(x),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;

    // A broadcast axis does not move through its input.
    let along = |d: &[usize]| -> Vec<usize> {
        strides(d).iter().zip(d).map(|(&s, &n)| if n == 1 { 0 } else { s }).collect()
    };
    let (sa, sb) = (along(&da), along(&db));
    let mut data = Vec::with_capacity(dims.iter().product());
    for_each_offset(&dims, [&sa, &sb], |[i, j]| data.push(f(a.data[i], b.data[j])));
    Some(Tensor::new(dims, data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_ir::{NodeId, ValueId};

    /// A graph computing `op` on constants, returning the node and its output.
    fn constant_op(op: OpType, inputs: Vec<Tensor>, attrs: Attrs) -> (Graph, NodeId, ValueId) {
        let mut graph = Graph::new();
        let inputs = inputs
            .into_iter()
            .enumerate()
            .map(|(i, t)| graph.add_initializer(format!("c{i}"), t))
            .collect();
        let out = graph.add_value("out");
        let node = graph.add_node(op, "op", inputs, vec![out], attrs);
        graph.outputs.push(out);
        (graph, node, out)
    }

    fn folded(op: OpType, inputs: Vec<Tensor>, attrs: Attrs) -> Option<Tensor> {
        let (mut graph, node, out) = constant_op(op, inputs, attrs);
        if FoldConstants.run(&mut graph) == 0 {
            assert!(graph.node(node).is_some());
            return None;
        }
        assert!(graph.node(node).is_none());
        Some(graph.initializer(out).expect("the output became a constant").clone())
    }

    fn t(dims: &[usize], data: &[f32]) -> Tensor {
        Tensor::new(dims.to_vec(), data.to_vec())
    }

    fn assert_tensor(got: Option<Tensor>, dims: &[usize], data: &[f32]) {
        let got = got.expect("expected the node to fold");
        assert_eq!((got.dims.as_slice(), got.data.as_slice()), (dims, data));
    }

    #[test]
    fn transposes_a_matrix_by_default() {
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        let x = t(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        assert_tensor(folded(OpType::Transpose, vec![x], Attrs::default()), &[3, 2], &[1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn transposes_by_perm() {
        // x[i][j][k] = 100i + 10j + k with dims [2, 3, 2]; perm [1, 0, 2] swaps the first two axes.
        let data: Vec<f32> = (0..2)
            .flat_map(|i| (0..3).flat_map(move |j| (0..2).map(move |k| (100 * i + 10 * j + k) as f32)))
            .collect();
        let x = t(&[2, 3, 2], &data);
        let expected: Vec<f32> = (0..3)
            .flat_map(|j| (0..2).flat_map(move |i| (0..2).map(move |k| (100 * i + 10 * j + k) as f32)))
            .collect();
        let attrs = Attrs::default().with_ints("perm", vec![1, 0, 2]);
        assert_tensor(folded(OpType::Transpose, vec![x], attrs), &[3, 2, 2], &expected);
    }

    #[test]
    fn leaves_a_transpose_with_an_invalid_perm() {
        let x = t(&[2, 3], &[0.; 6]);
        let attrs = Attrs::default().with_ints("perm", vec![0, 0]);
        assert!(folded(OpType::Transpose, vec![x], attrs).is_none());
    }

    #[test]
    fn broadcasts_binary_ops() {
        let m = t(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let row = t(&[3], &[10., 20., 30.]);
        let column = t(&[2, 1], &[100., 200.]);
        let scalar = t(&[], &[2.]);

        assert_tensor(folded(OpType::Add, vec![m.clone(), row], Attrs::default()), &[2, 3], &[11., 22., 33., 14., 25., 36.]);
        assert_tensor(folded(OpType::Sub, vec![column, m.clone()], Attrs::default()), &[2, 3], &[99., 98., 97., 196., 195., 194.]);
        assert_tensor(folded(OpType::Mul, vec![scalar, m.clone()], Attrs::default()), &[2, 3], &[2., 4., 6., 8., 10., 12.]);
        assert_tensor(folded(OpType::Div, vec![m.clone(), t(&[1], &[2.])], Attrs::default()), &[2, 3], &[0.5, 1., 1.5, 2., 2.5, 3.]);
        assert_tensor(folded(OpType::Pow, vec![m, t(&[], &[2.])], Attrs::default()), &[2, 3], &[1., 4., 9., 16., 25., 36.]);
    }

    #[test]
    fn leaves_shapes_that_do_not_broadcast() {
        let a = t(&[2, 3], &[0.; 6]);
        let b = t(&[2], &[0.; 2]);
        assert!(folded(OpType::Add, vec![a, b], Attrs::default()).is_none());
    }

    #[test]
    fn applies_unary_ops() {
        assert_tensor(folded(OpType::Sqrt, vec![t(&[3], &[1., 4., 9.])], Attrs::default()), &[3], &[1., 2., 3.]);
        assert_tensor(folded(OpType::Relu, vec![t(&[3], &[-1., 0., 2.])], Attrs::default()), &[3], &[0., 0., 2.]);
    }

    #[test]
    fn leaves_ops_it_cannot_evaluate() {
        let x = t(&[2], &[1., 2.]);
        assert!(folded(OpType::Other("Shape".into()), vec![x], Attrs::default()).is_none());
    }

    #[test]
    fn leaves_nodes_with_a_computed_input() {
        let mut graph = Graph::new();
        let x = graph.add_value("x");
        graph.inputs.push(x);
        let c = graph.add_initializer("c", t(&[1], &[1.]));
        let out = graph.add_value("out");
        graph.add_node(OpType::Add, "add", vec![x, c], vec![out], Attrs::default());
        graph.outputs.push(out);

        assert_eq!(FoldConstants.run(&mut graph), 0);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn folds_a_chain_in_one_run() {
        // Transpose(Transpose(w)) is w again.
        let mut graph = Graph::new();
        let w = graph.add_initializer("w", t(&[2, 3], &[1., 2., 3., 4., 5., 6.]));
        let once = graph.add_value("once");
        let twice = graph.add_value("twice");
        graph.add_node(OpType::Transpose, "t1", vec![w], vec![once], Attrs::default());
        graph.add_node(OpType::Transpose, "t2", vec![once], vec![twice], Attrs::default());
        graph.outputs.push(twice);

        assert_eq!(FoldConstants.run(&mut graph), 2);
        assert_eq!(graph.node_count(), 0);
        let back = graph.initializer(twice).unwrap();
        assert_eq!((back.dims.as_slice(), back.data.as_slice()), (&[2, 3][..], &[1., 2., 3., 4., 5., 6.][..]));
    }

    #[test]
    fn standard_pipeline_feeds_matmul_a_pretransposed_weight() {
        // x -> MatMul(x, Transpose(w)) -> y, as torch exports a Linear without constant folding.
        let mut graph = Graph::new();
        let x = graph.add_value("x");
        graph.inputs.push(x);
        let w = graph.add_initializer("w", t(&[2, 3], &[1., 2., 3., 4., 5., 6.]));
        let wt = graph.add_value("wt");
        let y = graph.add_value("y");
        graph.add_node(OpType::Transpose, "t", vec![w], vec![wt], Attrs::default());
        let matmul = graph.add_node(OpType::MatMul, "mm", vec![x, wt], vec![y], Attrs::default());
        graph.outputs.push(y);

        let report = crate::pipeline::Pipeline::standard(10).run(&mut graph);

        assert_eq!(report.rewrites.get("fold-constants"), Some(&1));
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.node(matmul).unwrap().inputs, vec![x, wt]);
        assert_eq!(graph.initializer(wt).unwrap().data, [1., 4., 2., 5., 3., 6.]);
        assert!(graph.initializer(w).is_none(), "the untransposed weight is dead and DCE drops it");
    }
}
