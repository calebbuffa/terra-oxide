//! Typed structs for the EXT_mesh_features glTF extension.
//!
//! Reference: https://github.com/CesiumGS/glTF/tree/proposal-EXT_mesh_features
use moderu::GltfExtension;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtMeshFeatures {
    pub feature_ids: Vec<FeatureId>,
}

impl GltfExtension for ExtMeshFeatures {
    const NAME: &'static str = "EXT_mesh_features";
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeatureId {
    pub feature_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribute: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub property_table: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}
