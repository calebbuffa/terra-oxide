//! Bentley iTwin REST API types.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ITwin {
    pub id: String,
    pub display_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RealityData {
    pub id: String,
    pub display_name: Option<String>,
    #[serde(rename = "type")]
    pub data_type: Option<String>,
    pub root_document: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IModelMeshExport {
    pub id: String,
    pub display_name: Option<String>,
    pub status: Option<String>,
}

/// Generic iTwin paged response.
#[derive(Debug, Clone, Deserialize)]
pub struct PagedResponse<T> {
    #[serde(flatten)]
    pub items_field: serde_json::Value,
    #[serde(skip)]
    _phantom: std::marker::PhantomData<T>,
}
