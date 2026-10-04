use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail};
use onnx_ir::{Attr, Attrs, Graph, OpType, Tensor, ValueId};
use onnx_proto::attribute_proto::AttributeType;
use onnx_proto::tensor_proto::{DataLocation, DataType};
use onnx_proto::{AttributeProto, ModelProto, TensorProto};

use crate::{Model, is_default_domain};

impl Model {
    pub fn from_proto(mut proto: ModelProto) -> Result<Self> {
        let mut graph_proto = proto.graph.take().context("model has no graph")?;
        if !graph_proto.sparse_initializer.is_empty() {
            bail!("sparse initializers are not supported");
        }

        let mut graph = Graph::new();
        let mut names: HashMap<String, ValueId> = HashMap::new();
        let mut opaque_initializers = HashMap::new();
        let mut opaque_attrs = HashMap::new();
        let mut domains = HashMap::new();

        for init in std::mem::take(&mut graph_proto.initializer) {
            let name = init.name.clone().unwrap_or_default();
            if names.contains_key(&name) {
                bail!("initializer `{name}` is defined twice");
            }
            let id = match f32_tensor(&init) {
                Some(tensor) => graph.add_initializer(&name, tensor),
                None => {
                    let id = graph.add_value(&name);
                    opaque_initializers.insert(id, init);
                    id
                }
            };
            names.insert(name, id);
        }

        for input in &graph_proto.input {
            let name = input.name.clone().unwrap_or_default();
            // Before IR version 4 every initializer is also listed as a graph input.
            if names.contains_key(&name) {
                continue;
            }
            let id = graph.add_value(&name);
            names.insert(name, id);
            graph.inputs.push(id);
        }

        for node in std::mem::take(&mut graph_proto.node) {
            let node_name = node.name.clone().unwrap_or_default();
            // Outer values used inside a subgraph are invisible to the IR, so DCE could drop them.
            if node.attribute.iter().any(|a| a.g.is_some() || !a.graphs.is_empty()) {
                bail!("node `{node_name}` has a subgraph attribute; subgraphs are not supported");
            }

            let inputs = node
                .input
                .iter()
                .map(|name| value_for_input(&mut graph, &mut names, name))
                .collect();
            let outputs = node
                .output
                .iter()
                .map(|name| value_for_output(&mut graph, &mut names, name, &opaque_initializers))
                .collect::<Result<_>>()
                .with_context(|| format!("node `{node_name}`"))?;

            let domain = node.domain.clone().unwrap_or_default();
            let op = op_type(&domain, node.op_type.as_deref().unwrap_or_default());
            let (attrs, opaque) = split_attrs(node.attribute);

            let id = graph.add_node(op, node_name, inputs, outputs, attrs);
            if !opaque.is_empty() {
                opaque_attrs.insert(id, opaque);
            }
            if !is_default_domain(&domain) {
                domains.insert(id, domain);
            }
        }

        for output in &graph_proto.output {
            let name = output.name.as_deref().unwrap_or_default();
            let id = *names
                .get(name)
                .with_context(|| format!("graph output `{name}` is never produced"))?;
            graph.outputs.push(id);
        }

        graph
            .validate()
            .map_err(|errors| anyhow!("imported graph is invalid:\n  {}", errors.join("\n  ")))?;

        proto.graph = Some(graph_proto);
        Ok(Model {
            graph,
            proto,
            opaque_initializers,
            opaque_attrs,
            domains,
        })
    }
}

/// Looks up the value a node reads, creating it if nothing has defined it yet.
fn value_for_input(graph: &mut Graph, names: &mut HashMap<String, ValueId>, name: &str) -> ValueId {
    // An empty name is a skipped optional input; each one gets its own value.
    if name.is_empty() {
        return graph.add_value("");
    }
    *names.entry(name.to_string()).or_insert_with(|| graph.add_value(name))
}

/// Looks up the value a node writes, which must not have another producer.
fn value_for_output(
    graph: &mut Graph,
    names: &mut HashMap<String, ValueId>,
    name: &str,
    opaque_initializers: &HashMap<ValueId, TensorProto>,
) -> Result<ValueId> {
    // An empty name is an optional output nobody asked for.
    if name.is_empty() {
        return Ok(graph.add_value(""));
    }
    let id = value_for_input(graph, names, name);
    if graph.initializer(id).is_some() || opaque_initializers.contains_key(&id) {
        bail!("output `{name}` overwrites an initializer");
    }
    if graph.value(id).is_some_and(|v| v.producer.is_some()) || graph.inputs.contains(&id) {
        bail!("output `{name}` already has a producer");
    }
    Ok(id)
}

fn op_type(domain: &str, op: &str) -> OpType {
    if !is_default_domain(domain) {
        return OpType::Other(op.to_string());
    }
    match op {
        "Conv" => OpType::Conv,
        "BatchNormalization" => OpType::BatchNormalization,
        "Relu" => OpType::Relu,
        "Add" => OpType::Add,
        "Sub" => OpType::Sub,
        "Mul" => OpType::Mul,
        "Div" => OpType::Div,
        "Pow" => OpType::Pow,
        "Sqrt" => OpType::Sqrt,
        "ReduceMean" => OpType::ReduceMean,
        "MatMul" => OpType::MatMul,
        "Gemm" => OpType::Gemm,
        "Identity" => OpType::Identity,
        "LayerNormalization" => OpType::LayerNormalization,
        other => OpType::Other(other.to_string()),
    }
}

/// Splits attributes into the ones `Attr` can hold and the ones kept verbatim.
fn split_attrs(protos: Vec<AttributeProto>) -> (Attrs, Vec<AttributeProto>) {
    let mut attrs = Attrs::default();
    let mut opaque = Vec::new();
    for proto in protos {
        let kind = proto.r#type.and_then(|t| AttributeType::try_from(t).ok());
        let attr = match kind {
            Some(AttributeType::Int) => Some(Attr::Int(proto.i.unwrap_or_default())),
            Some(AttributeType::Float) => Some(Attr::Float(proto.f.unwrap_or_default())),
            Some(AttributeType::String) => {
                String::from_utf8(proto.s.clone().unwrap_or_default()).ok().map(Attr::String)
            }
            Some(AttributeType::Ints) => Some(Attr::Ints(proto.ints.clone())),
            _ => None,
        };
        match attr {
            Some(attr) if proto.ref_attr_name.is_none() => {
                attrs.0.insert(proto.name.unwrap_or_default(), attr);
            }
            _ => opaque.push(proto),
        }
    }
    (attrs, opaque)
}

/// Decodes `proto` if it is an f32 tensor stored inline; anything else stays opaque.
fn f32_tensor(proto: &TensorProto) -> Option<Tensor> {
    if proto.data_type != Some(DataType::Float as i32)
        || proto.data_location == Some(DataLocation::External as i32)
        || proto.segment.is_some()
    {
        return None;
    }
    let dims = proto
        .dims
        .iter()
        .map(|&d| usize::try_from(d).ok())
        .collect::<Option<Vec<_>>>()?;
    let data: Vec<f32> = match &proto.raw_data {
        Some(raw) if !raw.is_empty() => {
            if raw.len() % 4 != 0 {
                return None;
            }
            raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        _ => proto.float_data.clone(),
    };
    (dims.iter().product::<usize>() == data.len()).then(|| Tensor::new(dims, data))
}
