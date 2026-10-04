use std::collections::{BTreeSet, HashSet};

use anyhow::{Result, bail};
use onnx_ir::{Attr, Graph, NodeId, OpType, Tensor, ValueId};
use onnx_proto::attribute_proto::AttributeType;
use onnx_proto::tensor_proto::DataType;
use onnx_proto::{AttributeProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto};

use crate::{MS_DOMAIN, Model, is_default_domain};

impl Model {
    pub fn to_proto(&self) -> Result<ModelProto> {
        let graph = &self.graph;
        let name = |id: ValueId| graph.value(id).map_or_else(String::new, |v| v.name.clone());

        let mut nodes = Vec::new();
        let mut uses_ms_domain = false;
        for id in topo_order(graph)? {
            let node = graph.node(id).expect("topo_order yields live nodes");
            let mut attribute: Vec<AttributeProto> = node
                .attrs
                .0
                .iter()
                .map(|(key, attr)| attr_proto(key, attr))
                .collect();
            attribute.sort_by(|a, b| a.name.cmp(&b.name));
            attribute.extend(self.opaque_attrs.get(&id).into_iter().flatten().cloned());

            let (op_type, domain) = match &node.op {
                // onnxruntime's contrib op: Conv's inputs and attributes plus a fused activation.
                OpType::FusedConvRelu => {
                    uses_ms_domain = true;
                    attribute.push(attr_proto("activation", &Attr::String("Relu".into())));
                    ("FusedConv".to_string(), MS_DOMAIN.to_string())
                }
                op @ OpType::Other(_) => (op.to_string(), self.domains.get(&id).cloned().unwrap_or_default()),
                op => (op.to_string(), String::new()),
            };
            nodes.push(NodeProto {
                input: node.inputs.iter().map(|&v| name(v)).collect(),
                output: node.outputs.iter().map(|&v| name(v)).collect(),
                name: Some(node.name.clone()),
                op_type: Some(op_type),
                domain: (!domain.is_empty()).then_some(domain),
                attribute,
                ..Default::default()
            });
        }

        // Only initializers something still reads are written out.
        let needed = |id: ValueId| graph.value(id).is_some_and(|v| !v.consumers.is_empty()) || graph.is_output(id);
        let mut initializers: Vec<(ValueId, TensorProto)> = graph
            .initializers()
            .filter(|&(id, _)| needed(id))
            .map(|(id, tensor)| (id, f32_proto(name(id), tensor)))
            .chain(
                self.opaque_initializers
                    .iter()
                    .filter(|&(&id, _)| needed(id))
                    .map(|(&id, proto)| (id, proto.clone())),
            )
            .collect();
        initializers.sort_by_key(|&(id, _)| id);

        let mut model = self.proto.clone();
        let graph_proto = model.graph.as_mut().expect("imported models have a graph");

        // Graph inputs that were initializers (IR < 4) go when their initializer does.
        let kept_inputs: HashSet<String> = graph
            .inputs
            .iter()
            .chain(initializers.iter().map(|(id, _)| id))
            .map(|&id| name(id))
            .collect();
        graph_proto.input.retain(|vi| vi.name.as_ref().is_some_and(|n| kept_inputs.contains(n)));

        // Shape info for intermediates the passes removed would describe values that no longer exist.
        let produced: HashSet<&str> = nodes.iter().flat_map(|n| n.output.iter().map(String::as_str)).collect();
        graph_proto
            .value_info
            .retain(|vi| vi.name.as_deref().is_some_and(|n| produced.contains(n)));

        graph_proto.node = nodes;
        graph_proto.initializer = initializers.into_iter().map(|(_, proto)| proto).collect();

        if uses_ms_domain && !model.opset_import.iter().any(|o| o.domain.as_deref() == Some(MS_DOMAIN)) {
            model.opset_import.push(OperatorSetIdProto {
                domain: Some(MS_DOMAIN.to_string()),
                version: Some(1),
            });
        }

        if let Some(opset) = graph.opset {
            let default = model
                .opset_import
                .iter_mut()
                .find(|o| is_default_domain(o.domain.as_deref().unwrap_or_default()));
            match default {
                Some(entry) if entry.version.is_none_or(|v| v < opset) => entry.version = Some(opset),
                Some(_) => {}
                None => model.opset_import.push(OperatorSetIdProto {
                    domain: Some(String::new()),
                    version: Some(opset),
                }),
            }
            if let Some(ir) = min_ir_version(opset)
                && model.ir_version.is_some_and(|v| v < ir)
            {
                model.ir_version = Some(ir);
            }
        }
        Ok(model)
    }
}

/// The IR version each opset shipped with (`onnx.helper.VERSION_TABLE`); a model declaring the
/// opset needs at least that.
fn min_ir_version(opset: i64) -> Option<i64> {
    Some(match opset {
        13..=14 => 7,
        15..=18 => 8,
        19..=20 => 9,
        21..=22 => 10,
        _ => return None,
    })
}

/// Live nodes in an order where every producer comes before its consumers, as ONNX requires.
///
/// Ties go to the lower [`NodeId`], so a graph that is already sorted keeps its order.
fn topo_order(graph: &Graph) -> Result<Vec<NodeId>> {
    let producer_of = |v: ValueId| graph.value(v).and_then(|v| v.producer);
    let mut pending: std::collections::HashMap<NodeId, usize> = graph
        .nodes()
        .map(|(id, node)| (id, node.inputs.iter().filter(|&&v| producer_of(v).is_some()).count()))
        .collect();
    let mut ready: BTreeSet<NodeId> = pending.iter().filter(|&(_, &n)| n == 0).map(|(&id, _)| id).collect();

    let mut order = Vec::with_capacity(pending.len());
    while let Some(id) = ready.pop_first() {
        order.push(id);
        for &output in &graph.node(id).expect("pending holds live nodes").outputs {
            // A consumer is listed once per input slot reading the value, matching the count above.
            for &consumer in &graph.value(output).expect("node outputs exist").consumers {
                let count = pending.get_mut(&consumer).expect("consumers are live nodes");
                *count -= 1;
                if *count == 0 {
                    ready.insert(consumer);
                }
            }
        }
    }
    if order.len() != pending.len() {
        bail!("graph has a cycle");
    }
    Ok(order)
}

fn attr_proto(name: &str, attr: &Attr) -> AttributeProto {
    let mut proto = AttributeProto {
        name: Some(name.to_string()),
        ..Default::default()
    };
    let kind = match attr {
        Attr::Int(i) => {
            proto.i = Some(*i);
            AttributeType::Int
        }
        Attr::Float(f) => {
            proto.f = Some(*f);
            AttributeType::Float
        }
        Attr::String(s) => {
            proto.s = Some(s.clone().into_bytes());
            AttributeType::String
        }
        Attr::Ints(ints) => {
            proto.ints = ints.clone();
            AttributeType::Ints
        }
    };
    proto.r#type = Some(kind as i32);
    proto
}

fn f32_proto(name: String, tensor: &Tensor) -> TensorProto {
    TensorProto {
        name: Some(name),
        dims: tensor.dims.iter().map(|&d| d as i64).collect(),
        data_type: Some(DataType::Float as i32),
        raw_data: Some(tensor.data.iter().flat_map(|x| x.to_le_bytes()).collect()),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_passes::pipeline::Pipeline;
    use onnx_proto::{GraphProto, ValueInfoProto};

    fn tensor(name: &str, dims: &[i64], data: &[f32]) -> TensorProto {
        TensorProto {
            name: Some(name.into()),
            dims: dims.to_vec(),
            data_type: Some(DataType::Float as i32),
            raw_data: Some(data.iter().flat_map(|x| x.to_le_bytes()).collect()),
            ..Default::default()
        }
    }

    fn node(op: &str, inputs: &[&str], outputs: &[&str], attribute: Vec<AttributeProto>) -> NodeProto {
        NodeProto {
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            name: Some(format!("{op}_{}", outputs[0])),
            op_type: Some(op.into()),
            attribute,
            ..Default::default()
        }
    }

    fn info(name: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: Some(name.into()),
            ..Default::default()
        }
    }

    /// `x -> Conv -> h -> BatchNormalization -> b -> Relu -> y`, with two output channels.
    fn conv_bn_relu() -> ModelProto {
        ModelProto {
            ir_version: Some(8),
            opset_import: vec![OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(13),
            }],
            graph: Some(GraphProto {
                node: vec![
                    node("Conv", &["x", "w"], &["h"], vec![attr_proto("pads", &Attr::Ints(vec![0, 0, 0, 0]))]),
                    node(
                        "BatchNormalization",
                        &["h", "gamma", "beta", "mean", "var"],
                        &["b"],
                        vec![attr_proto("epsilon", &Attr::Float(1.0))],
                    ),
                    node("Relu", &["b"], &["y"], vec![]),
                ],
                initializer: vec![
                    tensor("w", &[2, 1, 1, 1], &[1.0, 2.0]),
                    tensor("gamma", &[2], &[4.0, 3.0]),
                    tensor("beta", &[2], &[0.0, 3.0]),
                    tensor("mean", &[2], &[1.0, 0.0]),
                    tensor("var", &[2], &[3.0, 0.0]),
                ],
                input: vec![info("x")],
                output: vec![info("y")],
                value_info: vec![info("h"), info("b")],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn graph(model: &ModelProto) -> &GraphProto {
        model.graph.as_ref().unwrap()
    }

    fn op_types(model: &ModelProto) -> Vec<&str> {
        graph(model).node.iter().map(|n| n.op_type.as_deref().unwrap()).collect()
    }

    fn initializer_names(model: &ModelProto) -> Vec<&str> {
        graph(model).initializer.iter().map(|t| t.name.as_deref().unwrap()).collect()
    }

    #[test]
    fn round_trips_an_unoptimized_model() {
        let original = conv_bn_relu();
        let exported = Model::from_proto(original.clone()).unwrap().to_proto().unwrap();

        assert_eq!(graph(&exported).node, graph(&original).node);
        assert_eq!(graph(&exported).initializer, graph(&original).initializer);
        assert_eq!(graph(&exported).input, graph(&original).input);
        assert_eq!(graph(&exported).output, graph(&original).output);
        assert_eq!(graph(&exported).value_info, graph(&original).value_info);
        assert_eq!(exported.opset_import, original.opset_import);
    }

    #[test]
    fn exports_the_optimized_graph_as_a_fused_conv() {
        let mut model = Model::from_proto(conv_bn_relu()).unwrap();
        Pipeline::standard(10).run(&mut model.graph);
        let exported = model.to_proto().unwrap();

        let [fused] = &graph(&exported).node[..] else {
            panic!("expected one node, got {:?}", op_types(&exported));
        };
        assert_eq!(fused.op_type.as_deref(), Some("FusedConv"));
        assert_eq!(fused.domain.as_deref(), Some(MS_DOMAIN));
        assert_eq!(fused.output, vec!["y"]);
        let activation = fused.attribute.iter().find(|a| a.name.as_deref() == Some("activation")).unwrap();
        assert_eq!(activation.s.as_deref(), Some(&b"Relu"[..]));
        assert!(fused.attribute.iter().any(|a| a.name.as_deref() == Some("pads")));

        // Only the folded weight and bias are left, and the intermediates lost their shape info.
        assert_eq!(initializer_names(&exported).len(), 2);
        assert!(graph(&exported).value_info.is_empty());
        assert!(
            exported
                .opset_import
                .iter()
                .any(|o| o.domain.as_deref() == Some(MS_DOMAIN) && o.version == Some(1))
        );
    }

    #[test]
    fn keeps_what_the_ir_cannot_model() {
        let shape = TensorProto {
            name: Some("shape".into()),
            dims: vec![2],
            data_type: Some(DataType::Int64 as i32),
            int64_data: vec![1, -1],
            ..Default::default()
        };
        let floats = AttributeProto {
            name: Some("scales".into()),
            r#type: Some(AttributeType::Floats as i32),
            floats: vec![0.5, 2.0],
            ..Default::default()
        };
        let mut custom = node("MyOp", &["r"], &["y"], vec![floats.clone()]);
        custom.domain = Some("my.domain".into());
        let original = ModelProto {
            graph: Some(GraphProto {
                node: vec![node("Reshape", &["x", "shape"], &["r"], vec![]), custom],
                initializer: vec![shape.clone()],
                input: vec![info("x")],
                output: vec![info("y")],
                ..Default::default()
            }),
            ..Default::default()
        };

        let model = Model::from_proto(original).unwrap();
        assert!(model.graph.initializers().next().is_none(), "int64 tensors stay out of the IR");
        let exported = model.to_proto().unwrap();

        assert_eq!(graph(&exported).initializer, vec![shape]);
        let custom = &graph(&exported).node[1];
        assert_eq!(custom.domain.as_deref(), Some("my.domain"));
        assert_eq!(custom.attribute, vec![floats]);
    }

    #[test]
    fn reads_float_data_as_well_as_raw_data() {
        let mut proto = conv_bn_relu();
        let w = &mut proto.graph.as_mut().unwrap().initializer[0];
        w.raw_data = None;
        w.float_data = vec![1.0, 2.0];

        let model = Model::from_proto(proto).unwrap();
        let (_, w) = model.graph.initializers().find(|&(id, _)| model.graph.value(id).unwrap().name == "w").unwrap();
        assert_eq!(w.data, [1.0, 2.0]);
    }

    #[test]
    fn drops_initializer_inputs_along_with_their_initializers() {
        // Before IR version 4, initializers are listed among the graph inputs too.
        let mut proto = conv_bn_relu();
        let graph_proto = proto.graph.as_mut().unwrap();
        graph_proto.input.extend(["w", "gamma", "beta", "mean", "var"].map(info));

        let mut model = Model::from_proto(proto).unwrap();
        assert_eq!(model.graph.inputs.len(), 1);
        Pipeline::standard(10).run(&mut model.graph);
        let exported = model.to_proto().unwrap();

        assert_eq!(graph(&exported).input, vec![info("x")]);
    }

    #[test]
    fn rejects_subgraphs() {
        let mut proto = conv_bn_relu();
        proto.graph.as_mut().unwrap().node[2].attribute.push(AttributeProto {
            name: Some("then_branch".into()),
            r#type: Some(AttributeType::Graph as i32),
            g: Some(GraphProto::default()),
            ..Default::default()
        });

        assert!(Model::from_proto(proto).is_err());
    }

    fn constant(output: &str, attr: AttributeProto) -> NodeProto {
        node("Constant", &[], &[output], vec![attr])
    }

    /// The IR tensor behind the value named `name`, if it is an f32 initializer.
    fn ir_initializer<'a>(model: &'a Model, name: &str) -> Option<&'a Tensor> {
        let (id, _) = model.graph.values().find(|(_, v)| v.name == name)?;
        model.graph.initializer(id)
    }

    #[test]
    fn turns_numeric_constants_into_initializers() {
        let value = AttributeProto {
            name: Some("value".into()),
            r#type: Some(AttributeType::Tensor as i32),
            t: Some(tensor("", &[], &[2.0])),
            ..Default::default()
        };
        let original = ModelProto {
            graph: Some(GraphProto {
                node: vec![
                    constant("two", value),
                    constant("eps", attr_proto("value_float", &Attr::Float(1e-5))),
                    constant("shape", attr_proto("value_ints", &Attr::Ints(vec![1, -1]))),
                    node("Pow", &["x", "two"], &["p"], vec![]),
                    node("Add", &["p", "eps"], &["a"], vec![]),
                    node("Reshape", &["a", "shape"], &["y"], vec![]),
                ],
                input: vec![info("x")],
                output: vec![info("y")],
                ..Default::default()
            }),
            ..Default::default()
        };

        let model = Model::from_proto(original).unwrap();
        assert_eq!(model.graph.node_count(), 3);
        let two = ir_initializer(&model, "two").expect("f32 constants enter the IR");
        assert_eq!((two.dims.as_slice(), two.data.as_slice()), (&[][..], &[2.0][..]));
        assert_eq!(ir_initializer(&model, "eps").unwrap().data, [1e-5]);
        assert!(ir_initializer(&model, "shape").is_none(), "int64 constants stay out of the IR");

        let exported = model.to_proto().unwrap();
        assert_eq!(op_types(&exported), ["Pow", "Add", "Reshape"]);
        assert_eq!(initializer_names(&exported), ["two", "eps", "shape"]);
        let shape = &graph(&exported).initializer[2];
        assert_eq!(shape.data_type, Some(DataType::Int64 as i32));
        assert_eq!(shape.dims, [2]);
        assert_eq!(shape.int64_data, [1, -1]);
    }

    #[test]
    fn keeps_string_constants_as_nodes() {
        let original = ModelProto {
            graph: Some(GraphProto {
                node: vec![
                    constant("s", attr_proto("value_string", &Attr::String("hi".into()))),
                    node("MyOp", &["s"], &["y"], vec![]),
                ],
                output: vec![info("y")],
                ..Default::default()
            }),
            ..Default::default()
        };

        let exported = Model::from_proto(original).unwrap().to_proto().unwrap();
        assert_eq!(op_types(&exported), ["Constant", "MyOp"]);
        assert!(graph(&exported).initializer.is_empty());
    }

    #[test]
    fn declares_a_raised_opset_with_the_ir_version_it_needs() {
        let mut proto = conv_bn_relu();
        proto.ir_version = Some(7);
        let mut model = Model::from_proto(proto).unwrap();
        assert_eq!(model.graph.opset, Some(13));

        model.graph.opset = Some(17);
        let exported = model.to_proto().unwrap();

        let default = OperatorSetIdProto {
            domain: Some(String::new()),
            version: Some(17),
        };
        assert_eq!(exported.opset_import, vec![default]);
        assert_eq!(exported.ir_version, Some(8));
    }

    #[test]
    fn exports_a_fused_layer_norm_at_opset_17() {
        // torch's opset-13 LayerNorm over the last axis, its scalars in Constant nodes.
        let axes = || vec![attr_proto("axes", &Attr::Ints(vec![-1]))];
        let original = ModelProto {
            ir_version: Some(7),
            opset_import: vec![OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(13),
            }],
            graph: Some(GraphProto {
                node: vec![
                    node("ReduceMean", &["x"], &["mean"], axes()),
                    node("Sub", &["x", "mean"], &["d"], vec![]),
                    constant("two", attr_proto("value_float", &Attr::Float(2.0))),
                    node("Pow", &["d", "two"], &["d2"], vec![]),
                    node("ReduceMean", &["d2"], &["var"], axes()),
                    constant("eps", attr_proto("value_float", &Attr::Float(1e-5))),
                    node("Add", &["var", "eps"], &["var_eps"], vec![]),
                    node("Sqrt", &["var_eps"], &["std"], vec![]),
                    node("Div", &["d", "std"], &["norm"], vec![]),
                    node("Mul", &["norm", "gamma"], &["scaled"], vec![]),
                    node("Add", &["scaled", "beta"], &["y"], vec![]),
                ],
                initializer: vec![tensor("gamma", &[4], &[1.0, 2.0, 3.0, 4.0]), tensor("beta", &[4], &[0.5; 4])],
                input: vec![info("x")],
                output: vec![info("y")],
                ..Default::default()
            }),
            ..Default::default()
        };

        let mut model = Model::from_proto(original).unwrap();
        Pipeline::standard(10).run(&mut model.graph);
        let exported = model.to_proto().unwrap();

        let [ln] = &graph(&exported).node[..] else {
            panic!("expected one node, got {:?}", op_types(&exported));
        };
        assert_eq!(ln.op_type.as_deref(), Some("LayerNormalization"));
        assert_eq!(ln.domain, None);
        assert_eq!(ln.input, ["x", "gamma", "beta"]);
        assert_eq!(ln.output, ["y"]);
        assert_eq!(initializer_names(&exported), ["gamma", "beta"]);
        assert_eq!(exported.opset_import[0].version, Some(17));
        assert_eq!(exported.ir_version, Some(8));
    }
}
