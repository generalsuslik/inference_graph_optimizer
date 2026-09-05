//! Optimization passes over the [`onnx_ir`] graph.

use onnx_ir::Graph;

/// A single rewrite over the IR graph.
pub trait Pass {
    fn name(&self) -> &str;

    fn run(&self, graph: &mut Graph) -> usize;
}

/// Runs every pass in order, returning the total number of rewrites applied.
pub fn run_all(passes: &[&dyn Pass], graph: &mut Graph) -> usize {
    passes.iter().map(|pass| pass.run(graph)).sum()
}
