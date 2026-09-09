use std::{collections::HashMap, fmt};

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct NodeId(pub u32);

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ValueId(pub u32);

impl NodeId {
    fn idx(self) -> usize {
        self.0 as usize
    }
}

impl ValueId {
    fn idx(self) -> usize {
        self.0 as usize
    }
}

pub struct Tensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(dims: Vec<usize>, data: Vec<f32>) -> Self {
        assert_eq!(dims.iter().product::<usize>(), data.len(), "Data length does not match tensor dimensions");
        Tensor { dims, data }
    }

    pub fn filled(dims: Vec<usize>, value: f32) -> Self {
        let size = dims.iter().product();
        Tensor {
            dims,
            data: vec![value; size],
        }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn zeros(dims: Vec<usize>) -> Self {
        let size = dims.iter().product();
        Tensor {
            dims,
            data: vec![0.0; size],
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum OpType {
    Conv,
    BatchNormalization,
    Relu,
    Add,
    Mul,
    MatMul,
    Gemm,
    Identity,
    // Result of a fusion pass; carries the same semantics as Conv followed by Relu.
    FusedConvRelu,
    Other(String),
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpType::Other(s) => write!(f, "{}", s),
            other => write!(f, "{:?}", other),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Attrs(pub HashMap<String, Attr>);

#[derive(Clone, Debug)]
pub enum Attr {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
}

impl Attrs {
    pub fn get_float(&self, key: &str) -> Option<f32> {
        match self.0.get(key) {
            Some(Attr::Float(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn get_int(&self, key: &str) -> Option<i64> {
        match self.0.get(key) {
            Some(Attr::Int(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.0.get(key) {
            Some(Attr::String(value)) => Some(value),
            _ => None,
        }
    }

    pub fn get_ints(&self, key: &str) -> Option<&[i64]> {
        match self.0.get(key) {
            Some(Attr::Ints(values)) => Some(values),
            _ => None,
        }
    }

    pub fn with_float(mut self, key: &str, value: f32) -> Self {
        self.0.insert(key.to_string(), Attr::Float(value));
        self
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: OpType,
    pub name: String,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub attrs: Attrs,
}

#[derive(Clone, Debug)]
pub struct Value {
    pub name: String,
    pub producer: Option<NodeId>,
    pub consumers: Vec<NodeId>,
}

#[derive(Default)]
pub struct Graph {
    nodes: Vec<Option<Node>>,
    values: Vec<Value>,
    initializers: HashMap<ValueId, Tensor>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        for (i, node_opt) in self.nodes.iter().enumerate() {
            if let Some(node) = node_opt {
                for &input in &node.inputs {
                    if input.idx() >= self.values.len() {
                        errors.push(format!(
                            "Node {} ({}) has input ValueId {} which is out of bounds",
                            i, node.name, input.0
                        ));
                    }
                }
                for &output in &node.outputs {
                    if output.idx() >= self.values.len() {
                        errors.push(format!(
                            "Node {} ({}) has output ValueId {} which is out of bounds",
                            i, node.name, output.0
                        ));
                    }
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn add_value(&mut self, name: impl Into<String>) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(Value {
            name: name.into(),
            producer: None,
            consumers: Vec::new(),
        });
        id
    }

    pub fn add_initializer(&mut self, name: impl Into<String>, tensor: Tensor) -> ValueId {
        let id = self.add_value(name);
        self.initializers.insert(id, tensor);
        id
    }

    pub fn add_node(
        &mut self,
        op: OpType,
        name: impl Into<String>,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        attrs: Attrs,
    ) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        for &v in &inputs {
            self.values[v.idx()].consumers.push(id);
        }
        for &v in &outputs {
            debug_assert!(self.values[v.idx()].producer.is_none(), "Value already has a producer");
            self.values[v.idx()].producer = Some(id);
        }
        self.nodes.push(Some(Node {
            op,
            name: name.into(),
            inputs,
            outputs,
            attrs,
        }));
        id
    }

    /// Returns the node at `id`, or `None` if it was never created or has been removed.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.idx())?.as_ref()
    }

    /// Iterates over the live nodes in creation order, skipping removed ones.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| node.as_ref().map(|node| (NodeId(i as u32), node)))
    }

    /// Snapshots the live node ids, so a pass can mutate the graph while walking it.
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes().map(|(id, _)| id).collect()
    }

    /// Number of live nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.iter().flatten().count()
    }

    pub fn value(&self, id: ValueId) -> Option<&Value> {
        self.values.get(id.idx())
    }

    pub fn values(&self) -> impl Iterator<Item = (ValueId, &Value)> {
        self.values
            .iter()
            .enumerate()
            .map(|(i, value)| (ValueId(i as u32), value))
    }

    /// Returns the constant backing `id`, if it is an initializer rather than a computed value.
    pub fn initializer(&self, id: ValueId) -> Option<&Tensor> {
        self.initializers.get(&id)
    }

    pub fn initializers(&self) -> impl Iterator<Item = (ValueId, &Tensor)> {
        self.initializers.iter().map(|(&id, tensor)| (id, tensor))
    }

    pub fn is_output(&self, id: ValueId) -> bool {
        self.outputs.contains(&id)
    }

    pub fn set_op(&mut self, node: NodeId, op: OpType) {
        if let Some(Some(n)) = self.nodes.get_mut(node.idx()) {
            n.op = op;
        }
    }

    pub fn attrs_mut(&mut self, node: NodeId) -> Option<&mut Attrs> {
        self.nodes.get_mut(node.idx())?.as_mut().map(|n| &mut n.attrs)
    }

    /// Repoints `node` at a new output list, keeping the producer links in sync.
    ///
    /// Values dropped from the list lose their producer and become dangling; it is the
    /// caller's job to know they are dead.
    pub fn set_node_outputs(&mut self, node: NodeId, outputs: Vec<ValueId>) {
        let Some(Some(n)) = self.nodes.get_mut(node.idx()) else {
            return;
        };
        let old = std::mem::replace(&mut n.outputs, outputs.clone());
        for v in old {
            if self.values[v.idx()].producer == Some(node) {
                self.values[v.idx()].producer = None;
            }
        }
        for v in outputs {
            debug_assert!(
                self.values[v.idx()].producer.is_none_or(|p| p == node),
                "Value already has a different producer"
            );
            self.values[v.idx()].producer = Some(node);
        }
    }

    /// Removes `node`, unlinking it from the values it read and wrote.
    ///
    /// The slot is tombstoned rather than compacted, so every other [`NodeId`] stays valid.
    pub fn remove_node(&mut self, node: NodeId) -> Option<Node> {
        let removed = self.nodes.get_mut(node.idx())?.take()?;
        for &v in &removed.inputs {
            self.values[v.idx()].consumers.retain(|&c| c != node);
        }
        for &v in &removed.outputs {
            if self.values[v.idx()].producer == Some(node) {
                self.values[v.idx()].producer = None;
            }
        }
        Some(removed)
    }
}
