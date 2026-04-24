//! [`SkirtMeshMetadata`] - skirt range stored in primitive `extras`.
//!
//! Skirts are a seam-hiding technique: the edges of a terrain tile are
//! extruded downward so that neighbouring tiles with slightly different heights
//! don't reveal cracks. [`SkirtMeshMetadata`] records which part of the index
//! and vertex buffers contain the "real" geometry (as opposed to the added
//! skirt quads), enabling the upsampler to clip only the non-skirt triangles.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Private serde-able mirror of `SkirtMeshMetadata` whose field names match
///
/// `noSkirtRange` is a `[u32; 4]` matching
/// `[noSkirtIndicesBegin, noSkirtIndicesCount, noSkirtVerticesBegin, noSkirtVerticesCount]`.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SkirtMeshMetadataJson {
    no_skirt_range: [u32; 4],
    mesh_center: [f64; 3],
    skirt_west_height: f64,
    skirt_south_height: f64,
    skirt_east_height: f64,
    skirt_north_height: f64,
}

impl From<&SkirtMeshMetadata> for SkirtMeshMetadataJson {
    fn from(s: &SkirtMeshMetadata) -> Self {
        Self {
            no_skirt_range: [
                s.no_skirt_indices_begin,
                s.no_skirt_indices_count,
                s.no_skirt_vertices_begin,
                s.no_skirt_vertices_count,
            ],
            mesh_center: s.mesh_center,
            skirt_west_height: s.skirt_west_height,
            skirt_south_height: s.skirt_south_height,
            skirt_east_height: s.skirt_east_height,
            skirt_north_height: s.skirt_north_height,
        }
    }
}

impl From<SkirtMeshMetadataJson> for SkirtMeshMetadata {
    fn from(j: SkirtMeshMetadataJson) -> Self {
        Self {
            no_skirt_indices_begin: j.no_skirt_range[0],
            no_skirt_indices_count: j.no_skirt_range[1],
            no_skirt_vertices_begin: j.no_skirt_range[2],
            no_skirt_vertices_count: j.no_skirt_range[3],
            mesh_center: j.mesh_center,
            skirt_west_height: j.skirt_west_height,
            skirt_south_height: j.skirt_south_height,
            skirt_east_height: j.skirt_east_height,
            skirt_north_height: j.skirt_north_height,
        }
    }
}

/// Metadata that describes the skirt regions of a terrain mesh.
///
/// Stored in a `MeshPrimitive.extras` JSON object under the key
/// `"skirtMeshMetadata"`.
#[derive(Debug, Clone)]
pub struct SkirtMeshMetadata {
    /// Start index into the index buffer of the non-skirt geometry.
    pub no_skirt_indices_begin: u32,
    /// Length of the non-skirt index range.
    pub no_skirt_indices_count: u32,
    /// Start vertex index of the non-skirt geometry.
    pub no_skirt_vertices_begin: u32,
    /// Number of non-skirt vertices.
    pub no_skirt_vertices_count: u32,
    /// ECEF centre used as the relative origin for positions (`f32` relative
    /// positions are added to this to recover absolute ECEF).
    pub mesh_center: [f64; 3],
    pub skirt_west_height: f64,
    pub skirt_south_height: f64,
    pub skirt_east_height: f64,
    pub skirt_north_height: f64,
}

impl SkirtMeshMetadata {
    /// Parse from a `MeshPrimitive.extras` JSON value, returning `None` if the
    /// field is absent or malformed.
    pub fn parse_from_extras(extras: &Value) -> Option<Self> {
        let inner: SkirtMeshMetadataJson =
            serde_json::from_value(extras.get("skirtMeshMetadata")?.clone()).ok()?;
        Some(inner.into())
    }

    /// Serialise into the `"skirtMeshMetadata"` wrapper ready for insertion
    /// into a `MeshPrimitive.extras` JSON object.
    pub fn to_extras_value(&self) -> Value {
        serde_json::json!({ "skirtMeshMetadata": SkirtMeshMetadataJson::from(self) })
    }
}
