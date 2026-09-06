use std::collections::BTreeMap;

use onnx_ir::Graph;

use crate::pass::{Pass};

#[derive(Debug, Default)]
pub struct Report {
    pub iterations: usize,
    pub rewrites: BTreeMap<&'static str, usize>,
    pub nodes_before: usize,
    pub nodes_after: usize,
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

    pub fn run(&self, g: &mut Graph) -> Report {
        let mut report = Report {
            nodes_before: g.node_count(),
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
        report
    }
}
