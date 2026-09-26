use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use prost::Message;

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

pub fn load_model(path: impl AsRef<Path>) -> Result<ModelProto> {
    let path = path.as_ref();
    let data = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ModelProto::decode(&*data).with_context(|| format!("decoding {}", path.display()))
}

pub fn save_model(path: impl AsRef<Path>, model: &ModelProto) -> Result<()> {
    let path = path.as_ref();
    let data = model.encode_to_vec();
    fs::write(path, data).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_disk() {
        let model = ModelProto {
            ir_version: Some(9),
            graph: Some(GraphProto {
                name: Some("g".into()),
                node: vec![NodeProto {
                    op_type: Some("Relu".into()),
                    input: vec!["x".into()],
                    output: vec!["y".into()],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let path = std::env::temp_dir().join("onnx_proto_roundtrip.onnx");
        save_model(&path, &model).unwrap();
        let back = load_model(&path).unwrap();
        assert_eq!(back, model);
    }
}
