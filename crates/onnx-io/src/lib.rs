//! Conversion between ONNX protobuf models and the [`onnx_ir`] graph.

mod export;
mod import;

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use onnx_ir::{Graph, NodeId, ValueId};
use onnx_proto::{AttributeProto, ModelProto, TensorProto};

/// Domain of the onnxruntime contrib ops, `FusedConv` among them.
pub const MS_DOMAIN: &str = "com.microsoft";

/// An ONNX model opened for optimization.
///
/// [`Model::graph`] is what passes see and rewrite. Whatever the IR has no room for is kept
/// alongside it and goes back into the model on export.
pub struct Model {
    pub graph: Graph,
    /// The loaded model minus its nodes and initializers, which live in `graph` instead.
    proto: ModelProto,
    /// Initializers that are not plain f32 tensors, so the IR does not model them.
    opaque_initializers: HashMap<ValueId, TensorProto>,
    /// Node attributes of a kind `Attr` has no variant for.
    opaque_attrs: HashMap<NodeId, Vec<AttributeProto>>,
    /// Domains of imported nodes outside the default ONNX domain.
    domains: HashMap<NodeId, String>,
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_proto(onnx_proto::load_model(path)?)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        onnx_proto::save_model(path, &self.to_proto()?)
    }
}

fn is_default_domain(domain: &str) -> bool {
    domain.is_empty() || domain == "ai.onnx"
}
