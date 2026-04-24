//! ArcGIS SceneServer client — I3S service discovery.

use super::types::{Extent, SpatialReference};
use crate::client::Client;
use crate::fetch::{AssetAccessor, FetchError, RequestPriority};
use orkester::Task;
use serde::Deserialize;
use std::sync::Arc;

/// Summary of a single I3S layer advertised by a SceneServer.
#[derive(Debug, Clone)]
pub struct SceneServerInfo {
    /// I3S layer type: `"3DObject"`, `"IntegratedMesh"`, `"Building"`, `"PointCloud"`.
    pub layer_type: String,
    pub id: u32,
    pub href: String,
    pub full_extent: Option<Extent>,
    pub spatial_reference: Option<SpatialReference>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SceneServerResponse {
    layers: Option<Vec<LayerEntry>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LayerEntry {
    id: u32,
    href: Option<String>,
    layer_type: Option<String>,
    full_extent: Option<Extent>,
    spatial_reference: Option<SpatialReference>,
}

/// Client for an ArcGIS SceneServer REST endpoint.
///
/// Used to discover I3S layers for the kiban I3S loader.
pub struct SceneServerClient {
    base_url: String,
    accessor: Arc<dyn AssetAccessor>,
}

impl SceneServerClient {
    pub fn new(base_url: impl Into<String>, accessor: Arc<dyn AssetAccessor>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            accessor,
        }
    }

    /// Fetch `{base_url}?f=pjson` and return layer info for I3S loader wiring.
    ///
    /// Only returns layers with recognised I3S types (3DObject, IntegratedMesh,
    /// Building, PointCloud). Other types are filtered out.
    pub fn discover(&self) -> Task<Result<Vec<SceneServerInfo>, FetchError>> {
        let url = format!("{}?f=pjson", self.base_url);
        let base = self.base_url.clone();
        self.accessor
            .get(&url, &[], RequestPriority::NORMAL, None)
            .map(move |result| {
                result.and_then(|resp| {
                    resp.check_status()?;
                    let parsed: SceneServerResponse = serde_json::from_slice(&resp.data)
                        .map_err(|e| FetchError::Json(e.to_string()))?;

                    let known_types = ["3DObject", "IntegratedMesh", "Building", "PointCloud"];
                    let infos = parsed
                        .layers
                        .unwrap_or_default()
                        .into_iter()
                        .filter(|l| {
                            l.layer_type
                                .as_deref()
                                .map(|t| known_types.contains(&t))
                                .unwrap_or(false)
                        })
                        .map(|l| SceneServerInfo {
                            layer_type: l.layer_type.unwrap_or_default(),
                            id: l.id,
                            href: l.href.unwrap_or_else(|| format!("{base}/layers/{}", l.id)),
                            full_extent: l.full_extent,
                            spatial_reference: l.spatial_reference,
                        })
                        .collect();

                    Ok(infos)
                })
            })
    }
}

impl Client for SceneServerClient {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn accessor(&self) -> &Arc<dyn AssetAccessor> {
        &self.accessor
    }
}
