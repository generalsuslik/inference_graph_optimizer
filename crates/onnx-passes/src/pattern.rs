//! Declarative subgraph patterns.
//!
//! A pattern is a tree rooted at the node a pass rewrites, with names on the parts the pass
//! needs. A name always denotes one value: using it twice requires both places to see the
//! same value, which is how a pattern says "this value feeds two of my nodes".
//!
//! [`matches`] only reports matches a pass can rewrite without changing anything else: every
//! matched node but the root must have all of its consumers inside the match, so the
//! intermediates a rewrite deletes are never needed elsewhere.

use std::collections::{HashMap, HashSet};

use onnx_ir::{Graph, NodeId, OpType, ValueId};

pub enum Pat {
    /// Any value.
    Value(&'static str),
    /// An initializer.
    Const(&'static str),
    /// The first output of a node.
    Node(NodePat),
}

pub struct NodePat {
    name: &'static str,
    op: OpType,
    /// `None` accepts any inputs; otherwise there must be exactly these, in this order, or in
    /// either order for a commutative op with two inputs.
    inputs: Option<Vec<Pat>>,
}

impl Pat {
    pub fn value(name: &'static str) -> Pat {
        Pat::Value(name)
    }

    pub fn constant(name: &'static str) -> Pat {
        Pat::Const(name)
    }

    pub fn node(name: &'static str, op: OpType, inputs: impl Into<Vec<Pat>>) -> Pat {
        Pat::Node(NodePat {
            name,
            op,
            inputs: Some(inputs.into()),
        })
    }

    /// A node whose inputs the pattern does not care about.
    pub fn any_node(name: &'static str, op: OpType) -> Pat {
        Pat::Node(NodePat { name, op, inputs: None })
    }
}

/// What a pattern's names were bound to. A node pattern's name binds both the node and its
/// output value.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Match {
    values: HashMap<&'static str, ValueId>,
    nodes: HashMap<&'static str, NodeId>,
}

impl Match {
    pub fn get(&self, name: &str) -> Option<ValueId> {
        self.values.get(name).copied()
    }

    /// The value bound to `name`, which the pattern must define.
    pub fn value(&self, name: &str) -> ValueId {
        self.get(name).unwrap_or_else(|| panic!("pattern binds no value `{name}`"))
    }

    /// The node bound to `name`, which the pattern must define as a node.
    pub fn node(&self, name: &str) -> NodeId {
        *self.nodes.get(name).unwrap_or_else(|| panic!("pattern binds no node `{name}`"))
    }

    fn bind(mut self, name: &'static str, value: ValueId, node: Option<NodeId>) -> Option<Self> {
        if self.values.get(name).is_some_and(|&bound| bound != value) {
            return None;
        }
        self.values.insert(name, value);
        if let Some(node) = node {
            self.nodes.insert(name, node);
        }
        Some(self)
    }
}

/// Every way `pattern` matches with its root at `root`. Add and Mul are tried with their inputs
/// in both orders, so several matches can come back; a pass takes the first one it accepts.
pub fn matches(graph: &Graph, pattern: &Pat, root: NodeId) -> Vec<Match> {
    let Pat::Node(pattern) = pattern else {
        panic!("a pattern must be rooted at a node");
    };
    match_node(graph, pattern, root, Match::default())
        .into_iter()
        .filter(|m| is_self_contained(graph, m, root))
        .collect()
}

fn match_value(graph: &Graph, pattern: &Pat, value: ValueId, m: Match) -> Vec<Match> {
    match pattern {
        Pat::Value(name) => m.bind(*name, value, None).into_iter().collect(),
        Pat::Const(name) => {
            if graph.initializer(value).is_none() {
                return Vec::new();
            }
            m.bind(*name, value, None).into_iter().collect()
        }
        Pat::Node(pattern) => {
            let Some(producer) = graph.value(value).and_then(|v| v.producer) else {
                return Vec::new();
            };
            if graph.node(producer).and_then(|n| n.outputs.first()) != Some(&value) {
                return Vec::new();
            }
            match_node(graph, pattern, producer, m)
        }
    }
}

fn match_node(graph: &Graph, pattern: &NodePat, id: NodeId, m: Match) -> Vec<Match> {
    let Some(node) = graph.node(id) else {
        return Vec::new();
    };
    if node.op != pattern.op {
        return Vec::new();
    }
    let Some(m) = node.outputs.first().and_then(|&output| m.bind(pattern.name, output, Some(id))) else {
        return Vec::new();
    };
    let Some(inputs) = &pattern.inputs else {
        return vec![m];
    };
    if inputs.len() != node.inputs.len() {
        return Vec::new();
    }

    let mut found = match_all(graph, inputs, &node.inputs, m.clone());
    if let [a, b] = node.inputs[..]
        && a != b
        && matches!(node.op, OpType::Add | OpType::Mul)
    {
        found.extend(match_all(graph, inputs, &[b, a], m));
    }
    found
}

/// Matches `patterns` against `values` pairwise, carrying every partial match forward.
fn match_all(graph: &Graph, patterns: &[Pat], values: &[ValueId], m: Match) -> Vec<Match> {
    patterns.iter().zip(values).fold(vec![m], |partial, (pattern, &value)| {
        partial
            .into_iter()
            .flat_map(|m| match_value(graph, pattern, value, m))
            .collect()
    })
}

/// Whether the match can be rewritten in isolation: no intermediate is read outside it or is a
/// graph output, and no value the rewrite would read is produced by a node it deletes.
fn is_self_contained(graph: &Graph, m: &Match, root: NodeId) -> bool {
    let inside: HashSet<NodeId> = m.nodes.values().copied().collect();

    let intermediates_stay_inside = inside.iter().filter(|&&n| n != root).all(|&n| {
        graph.node(n).is_some_and(|node| {
            node.outputs.iter().all(|&output| {
                !graph.is_output(output)
                    && graph
                        .value(output)
                        .is_some_and(|v| v.consumers.iter().all(|c| inside.contains(c)))
            })
        })
    });

    // A value capture that is not itself a node's name must come from outside the match;
    // otherwise it aliases an intermediate the rewrite is about to remove.
    let captures_come_from_outside = m
        .values
        .iter()
        .filter(|(name, _)| !m.nodes.contains_key(*name))
        .all(|(_, &value)| graph.value(value).and_then(|v| v.producer).is_none_or(|p| !inside.contains(&p)));

    intermediates_stay_inside && captures_come_from_outside
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_ir::{Attrs, Tensor};

    /// A graph builder that names values as it goes.
    struct Builder {
        graph: Graph,
        values: HashMap<&'static str, ValueId>,
    }

    impl Builder {
        fn new(inputs: &[&'static str]) -> Self {
            let mut b = Builder {
                graph: Graph::new(),
                values: HashMap::new(),
            };
            for &name in inputs {
                let id = b.graph.add_value(name);
                b.graph.inputs.push(id);
                b.values.insert(name, id);
            }
            b
        }

        fn constant(&mut self, name: &'static str) -> &mut Self {
            let id = self.graph.add_initializer(name, Tensor::filled(vec![1], 2.0));
            self.values.insert(name, id);
            self
        }

        fn node(&mut self, op: OpType, inputs: &[&'static str], output: &'static str) -> NodeId {
            let inputs = inputs.iter().map(|name| self.values[name]).collect();
            let out = self.graph.add_value(output);
            self.values.insert(output, out);
            self.graph.add_node(op, output, inputs, vec![out], Attrs::default())
        }

        fn v(&self, name: &str) -> ValueId {
            self.values[name]
        }
    }

    fn relu_of_conv() -> Pat {
        Pat::node("relu", OpType::Relu, [Pat::any_node("conv", OpType::Conv)])
    }

    /// `Div(d, Sqrt(Pow(d, two)))` with `d = Sub(x, ReduceMean(x))`: `x` and `d` both feed two
    /// nodes of the pattern, as in LayerNorm.
    fn shared_values() -> Pat {
        let mean = Pat::node("mean", OpType::ReduceMean, [Pat::value("x")]);
        let d = Pat::node("d", OpType::Sub, [Pat::value("x"), mean]);
        let pow = Pat::node("pow", OpType::Pow, [Pat::value("d"), Pat::constant("two")]);
        let sqrt = Pat::node("sqrt", OpType::Sqrt, [pow]);
        Pat::node("div", OpType::Div, [d, sqrt])
    }

    fn shared_values_graph() -> (Builder, NodeId) {
        let mut b = Builder::new(&["x"]);
        b.constant("two");
        b.node(OpType::ReduceMean, &["x"], "mean");
        b.node(OpType::Sub, &["x", "mean"], "d");
        b.node(OpType::Pow, &["d", "two"], "pow");
        b.node(OpType::Sqrt, &["pow"], "sqrt");
        let div = b.node(OpType::Div, &["d", "sqrt"], "div");
        b.graph.outputs.push(b.v("div"));
        (b, div)
    }

    #[test]
    fn binds_nodes_and_values_by_name() {
        let mut b = Builder::new(&["x"]);
        let conv = b.node(OpType::Conv, &["x"], "hidden");
        let relu = b.node(OpType::Relu, &["hidden"], "y");

        let [m] = &matches(&b.graph, &relu_of_conv(), relu)[..] else {
            panic!("expected exactly one match");
        };
        assert_eq!(m.node("relu"), relu);
        assert_eq!(m.node("conv"), conv);
        assert_eq!(m.value("conv"), b.v("hidden"));
        assert_eq!(m.value("relu"), b.v("y"));
    }

    #[test]
    fn rejects_the_wrong_op_or_arity() {
        let mut b = Builder::new(&["x", "y"]);
        b.node(OpType::Conv, &["x"], "hidden");
        let add = b.node(OpType::Add, &["hidden", "y"], "out");
        let relu_of_add = b.node(OpType::Relu, &["out"], "z");

        assert!(matches(&b.graph, &relu_of_conv(), relu_of_add).is_empty());
        // Relu takes one input; the Add has two.
        let one_input = Pat::node("add", OpType::Add, [Pat::value("a")]);
        assert!(matches(&b.graph, &one_input, add).is_empty());
    }

    #[test]
    fn const_matches_only_initializers() {
        let mut b = Builder::new(&["x", "y"]);
        b.constant("c");
        let with_const = b.node(OpType::Mul, &["x", "c"], "a");
        let with_input = b.node(OpType::Mul, &["x", "y"], "b");
        let pattern = Pat::node("mul", OpType::Mul, [Pat::value("x"), Pat::constant("c")]);

        assert_eq!(matches(&b.graph, &pattern, with_const).len(), 1);
        assert!(matches(&b.graph, &pattern, with_input).is_empty());
    }

    #[test]
    fn a_repeated_name_must_bind_the_same_value() {
        let pattern = Pat::node(
            "sub",
            OpType::Sub,
            [
                Pat::value("x"),
                Pat::node("mean", OpType::ReduceMean, [Pat::value("x")]),
            ],
        );
        let mut b = Builder::new(&["x", "other"]);
        b.node(OpType::ReduceMean, &["x"], "mean_x");
        b.node(OpType::ReduceMean, &["other"], "mean_other");
        let same = b.node(OpType::Sub, &["x", "mean_x"], "d1");
        let different = b.node(OpType::Sub, &["x", "mean_other"], "d2");

        assert_eq!(matches(&b.graph, &pattern, same).len(), 1);
        assert!(matches(&b.graph, &pattern, different).is_empty());
    }

    #[test]
    fn add_and_mul_match_either_input_order() {
        let pattern = Pat::node(
            "add",
            OpType::Add,
            [Pat::any_node("relu", OpType::Relu), Pat::constant("bias")],
        );
        let mut b = Builder::new(&["x"]);
        b.constant("bias");
        // A Relu each: one shared Relu would be read outside either match.
        b.node(OpType::Relu, &["x"], "r1");
        b.node(OpType::Relu, &["x"], "r2");
        let as_written = b.node(OpType::Add, &["r1", "bias"], "a");
        let swapped = b.node(OpType::Add, &["bias", "r2"], "b");

        assert_eq!(matches(&b.graph, &pattern, as_written).len(), 1);
        let [m] = &matches(&b.graph, &pattern, swapped)[..] else {
            panic!("expected exactly one match");
        };
        assert_eq!(m.value("bias"), b.v("bias"));
    }

    #[test]
    fn other_ops_keep_their_input_order() {
        let pattern = Pat::node(
            "sub",
            OpType::Sub,
            [Pat::any_node("relu", OpType::Relu), Pat::constant("c")],
        );
        let mut b = Builder::new(&["x"]);
        b.constant("c");
        b.node(OpType::Relu, &["x"], "r");
        let swapped = b.node(OpType::Sub, &["c", "r"], "s");

        assert!(matches(&b.graph, &pattern, swapped).is_empty());
    }

    #[test]
    fn a_value_may_feed_two_nodes_of_the_match() {
        let (b, div) = shared_values_graph();

        let [m] = &matches(&b.graph, &shared_values(), div)[..] else {
            panic!("expected exactly one match");
        };
        assert_eq!(m.value("x"), b.v("x"));
        assert_eq!(m.value("d"), b.v("d"));
    }

    #[test]
    fn rejects_an_intermediate_read_outside_the_match() {
        let (mut b, div) = shared_values_graph();
        b.node(OpType::Identity, &["d"], "leak");

        assert!(matches(&b.graph, &shared_values(), div).is_empty());
    }

    #[test]
    fn rejects_an_intermediate_that_is_a_graph_output() {
        let (mut b, div) = shared_values_graph();
        b.graph.outputs.push(b.v("sqrt"));

        assert!(matches(&b.graph, &shared_values(), div).is_empty());
    }

    #[test]
    fn the_root_output_may_be_read_anywhere() {
        let (mut b, div) = shared_values_graph();
        b.node(OpType::Identity, &["div"], "after");

        assert_eq!(matches(&b.graph, &shared_values(), div).len(), 1);
    }

    #[test]
    fn rejects_a_capture_that_aliases_an_intermediate() {
        // Add(a, Relu(y)) on Add(r, r) with r = Relu(y): `a` would be the Relu's output, which a
        // rewrite of the match deletes.
        let pattern = Pat::node(
            "add",
            OpType::Add,
            [Pat::value("a"), Pat::node("relu", OpType::Relu, [Pat::value("y")])],
        );
        let mut b = Builder::new(&["y"]);
        b.node(OpType::Relu, &["y"], "r");
        let add = b.node(OpType::Add, &["r", "r"], "out");

        assert!(matches(&b.graph, &pattern, add).is_empty());
    }
}
