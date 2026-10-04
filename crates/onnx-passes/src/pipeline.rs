use std::collections::BTreeMap;

use onnx_ir::Graph;

use crate::pass::{Pass};
use crate::{DCE, EliminateIdentity, FoldConvBn, FuseConvRelu, FuseLayerNorm};

#[derive(Debug, Default)]
pub struct Report {
    pub iterations: usize,
    pub rewrites: BTreeMap<&'static str, usize>,
    pub nodes_before: usize,
    pub nodes_after: usize,
    /// Default opset the graph declared before and after; a pass emitting a newer op raises it,
    /// which raises what the runtime has to support.
    pub opset_before: Option<i64>,
    pub opset_after: Option<i64>,
}

pub struct Pipeline {
    passes: Vec<Box<dyn Pass>>,
    max_iterations: usize,
}

impl Pipeline {
    pub fn new(passes: Vec<Box<dyn Pass>>, max_iterations: usize) -> Self {
        Self {
            passes,
            max_iterations,
        }
    }

    /// Every pass, in the order they should run. Identity elimination goes first so that
    /// constants hidden behind an alias reach the folds as initializers. BN folding goes
    /// before Conv+Relu fusion so a `Conv -> BatchNormalization -> Relu` chain collapses in one iteration.
    /// LayerNorm fusion touches none of their ops, and DCE goes last to sweep up whatever the
    /// rewrites left behind.
    pub fn standard(max_iterations: usize) -> Self {
        Self::new(
            vec![
                Box::new(EliminateIdentity),
                Box::new(FoldConvBn),
                Box::new(FuseConvRelu), 
                Box::new(FuseLayerNorm),
                Box::new(DCE)
            ],
            max_iterations,
        )
    }

    pub fn run(&self, g: &mut Graph) -> Report {
        let mut report = Report {
            nodes_before: g.node_count(),
            opset_before: g.opset,
            ..Default::default()
        };

        for it in 0..self.max_iterations {
            let mut changed = 0usize;
            for p in &self.passes {
                let n = p.run(g);
                if n > 0 {
                    *report.rewrites.entry(p.name()).or_default() += n;
                    changed += n;
                }
                if cfg!(debug_assertions) {
                    log::debug!(
                        "pass {} applied {} rewrites, graph now has {} nodes",
                        p.name(),
                        n,
                        g.node_count()
                    );
                    if let Err(errs) = g.validate() {
                        panic!("pass `{}` broke the graph:\n  {}", p.name(), errs.join("\n  "));
                    }
                }
            }
            if changed == 0 {
                report.iterations = it;
                break;
            }
            report.iterations = it + 1;
        }
        report.nodes_after = g.node_count();
        report.opset_after = g.opset;
        report
    }
}
