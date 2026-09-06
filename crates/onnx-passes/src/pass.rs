//! Optimization passes over the [`onnx_ir`] graph.

use onnx_ir::Graph;

pub trait Pass {
    fn name(&self) -> &'static str;

    fn run(&self, graph: &mut Graph) -> usize;
}
