//! [`LayerJsonTerrainLoader`] - quantized-mesh-1.0 terrain loader.
//!
//! Mirrors `Cesium3DTilesSelection::LayerJsonTerrainLoader`.
//!
//! Decoding of the binary quantized-mesh format is handled by
//! [`arazi::decode_quantized_mesh`]. This module owns only the network I/O,
//! `layer.json` parsing, quadtree descriptor construction, and
//! [`ContentLoader`] integration.

use std::sync::Arc;

use arazi::{QuantizedMeshResult, decode_quantized_mesh};
use courtier::{AssetAccessor, AssetResponse, RequestPriority};
use glam::DMat4;
use orkester::{Context, Task};
use terra::{Ellipsoid, GlobeRectangle, obb_for_region, tile_geometric_error};
use zukei::{QuadtreeTileID, QuadtreeTilingScheme, SpatialBounds};

use super::tileset_json::TilesetInitResult;
use crate::loader::{
    ContentLoader, HeightSampler, TileChildrenResult, TileLoadInput, TileLoadResult,
};
use crate::tile_store::{
    ContentKey, RefinementMode, TileDescriptor, TileFlags, TileId, TileKind, TileStore,
};

/// Errors that can occur while initialising a terrain loader from `layer.json`.
#[derive(Debug, thiserror::Error)]
pub enum TerrainLoaderError {
    #[error("I/O error: {0}")]
    Fetch(#[from] courtier::FetchError),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("layer.json missing required field: {0}")]
    MissingField(&'static str),
}

/// Terrain loader backed by a Cesium-style `layer.json` + quantized-mesh tiles.
///
/// Mirrors `Cesium3DTilesSelection::LayerJsonTerrainLoader`.
pub struct LayerJsonTerrainLoader {
    /// Directory URL from which tile URL templates are resolved.
    base_url: Arc<str>,
    /// URL templates from `layer.json["tiles"]`. `{z}`, `{x}`, `{y}` are
    /// substituted at load time. `{reverseY}` is also supported for TMS
    /// (inverted Y) convention.
    tile_url_templates: Vec<String>,
    /// `octvertexnormals-metadata` or similar; appended as `?extensions=…`
    /// query parameter.
    extensions_to_request: String,
    /// Maximum zoom level available (default 30).
    max_zoom: u32,
    /// Geographic 2x1 (EPSG:4326) or web-mercator 1x1 (EPSG:3857).
    tiling_scheme: QuadtreeTilingScheme,
    /// Reference ellipsoid.
    ellipsoid: Ellipsoid,
    /// HTTP headers forwarded from the creator.
    headers: Arc<[(String, String)]>,
    /// Network/file accessor.
    accessor: Arc<dyn AssetAccessor>,
}

impl LayerJsonTerrainLoader {
    /// Fetch `layer.json`, parse it, and return a ready [`TilesetInitResult`].
    ///
    /// Mirrors `LayerJsonTerrainLoader::createLoader`.
    pub fn create_loader(
        layer_json_url: impl Into<String>,
        headers: Vec<(String, String)>,
        accessor: Arc<dyn AssetAccessor>,
        bg: Context,
        ellipsoid: Ellipsoid,
    ) -> Task<Result<TilesetInitResult, TerrainLoaderError>> {
        let url: Arc<str> = layer_json_url.into().into();
        let headers: Arc<[(String, String)]> = headers.into();
        let accessor2 = Arc::clone(&accessor);
        let url2 = Arc::clone(&url);
        let headers2 = Arc::clone(&headers);

        accessor
            .get(&url, &headers, RequestPriority::HIGH, None)
            .then(
                &bg,
                move |io_result: Result<AssetResponse, courtier::FetchError>| {
                    let result = (|| -> Result<TilesetInitResult, TerrainLoaderError> {
                        let response = io_result?;
                        response.check_status()?;
                        let data = response.decompressed_data();

                        let parsed: serde_json::Value = serde_json::from_slice(data)?;

                        let projection_str = parsed["projection"].as_str().unwrap_or("EPSG:4326");
                        let (tiling_scheme, root_tiles_x) = if projection_str == "EPSG:3857" {
                            (QuadtreeTilingScheme::web_mercator(), 1u32)
                        } else {
                            (QuadtreeTilingScheme::geographic(), 2u32)
                        };

                        let tile_urls: Vec<String> = match parsed["tiles"].as_array() {
                            Some(arr) => arr
                                .iter()
                                .filter_map(|v| v.as_str().map(str::to_string))
                                .collect(),
                            None => return Err(TerrainLoaderError::MissingField("tiles")),
                        };
                        if tile_urls.is_empty() {
                            return Err(TerrainLoaderError::MissingField("tiles"));
                        }

                        let max_zoom = parsed["maxzoom"].as_u64().unwrap_or(30) as u32;

                        let available_exts: Vec<&str> = parsed["extensions"]
                            .as_array()
                            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
                            .unwrap_or_default();
                        let wanted = ["octvertexnormals", "metadata"];
                        let ext_query = wanted
                            .iter()
                            .filter(|&&w| available_exts.contains(&w))
                            .copied()
                            .collect::<Vec<_>>()
                            .join("-");

                        let attribution: Option<Arc<str>> = parsed["attribution"]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .map(Arc::from);

                        let base_url: Arc<str> = {
                            let s = url2.as_ref();
                            let base = s.rfind('/').map(|i| &s[..=i]).unwrap_or(s);
                            Arc::from(base)
                        };

                        let loader = LayerJsonTerrainLoader {
                            base_url,
                            tile_url_templates: tile_urls,
                            extensions_to_request: ext_query,
                            max_zoom,
                            tiling_scheme: tiling_scheme.clone(),
                            ellipsoid: ellipsoid.clone(),
                            headers: Arc::clone(&headers2),
                            accessor: Arc::clone(&accessor2),
                        };

                        let root = loader.build_root(root_tiles_x);

                        Ok(TilesetInitResult {
                            root,
                            loader: Arc::new(loader),
                            child_loaders: Vec::new(),
                            attribution,
                        })
                    })();

                    orkester::resolved(result)
                },
            )
    }

    fn build_root(&self, root_tiles_x: u32) -> TileDescriptor {
        let mut level0 = Vec::new();
        for x in 0..root_tiles_x {
            if let Some(desc) = self.node_descriptor_for(QuadtreeTileID::new(0, x, 0)) {
                level0.push(desc);
            }
        }
        TileDescriptor {
            bounds: SpatialBounds::Empty,
            geometric_error: f64::MAX,
            refinement: RefinementMode::Replace,
            kind: TileKind::EMPTY,
            flags: TileFlags::UNCONDITIONALLY_REFINED,
            content_keys: Vec::new(),
            world_transform: DMat4::IDENTITY,
            children: level0,
            content_bounds: None,
            viewer_request_volume: None,
            globe_rectangle: None,
            content_max_age: None,
            loader_index: None,
        }
    }

    fn node_descriptor_for(&self, id: QuadtreeTileID) -> Option<TileDescriptor> {
        let rect = self.tiling_scheme.tile_to_rectangle(id)?;
        let globe_rect = GlobeRectangle::new(
            rect.minimum_x,
            rect.minimum_y,
            rect.maximum_x,
            rect.maximum_y,
        );
        let bounds = obb_for_region(
            &self.ellipsoid,
            rect.minimum_x,
            rect.minimum_y,
            rect.maximum_x,
            rect.maximum_y,
        );
        let geometric_error = tile_geometric_error(&self.ellipsoid, id, &self.tiling_scheme);

        Some(TileDescriptor {
            bounds,
            geometric_error,
            refinement: RefinementMode::Replace,
            kind: TileKind::CONTENT,
            flags: if id.level < self.max_zoom {
                TileFlags::MIGHT_HAVE_LATENT_CHILDREN
            } else {
                TileFlags::empty()
            },
            content_keys: vec![ContentKey::Quadtree(id.level, id.x, id.y)],
            world_transform: DMat4::IDENTITY,
            children: Vec::new(),
            content_bounds: None,
            viewer_request_volume: None,
            globe_rectangle: Some(globe_rect),
            content_max_age: None,
            loader_index: None,
        })
    }

    fn tile_url(&self, id: QuadtreeTileID) -> String {
        let template = &self.tile_url_templates[0];
        let mut url = template
            .replace("{z}", &id.level.to_string())
            .replace("{x}", &id.x.to_string())
            .replace("{y}", &id.y.to_string());

        if url.contains("{reverseY}") {
            let tiles_y = self.tiling_scheme.tiles_y_at_level(id.level);
            let reverse_y = tiles_y.saturating_sub(1 + id.y);
            url = url.replace("{reverseY}", &reverse_y.to_string());
        }

        if !self.extensions_to_request.is_empty() {
            if url.contains('?') {
                url.push_str(&format!("&extensions={}", self.extensions_to_request));
            } else {
                url.push_str(&format!("?extensions={}", self.extensions_to_request));
            }
        }

        if url.starts_with("http://") || url.starts_with("https://") || url.starts_with("file://") {
            url
        } else {
            outil::resolve_url(&self.base_url, &url)
        }
    }
}

impl ContentLoader for LayerJsonTerrainLoader {
    fn load_tile(&self, input: TileLoadInput) -> Task<TileLoadResult> {
        let id = match input.content_keys.first() {
            Some(ContentKey::Quadtree(z, x, y)) => QuadtreeTileID::new(*z, *x, *y),
            _ => return orkester::resolved(TileLoadResult::failed()),
        };

        let rect = match self.tiling_scheme.tile_to_rectangle(id) {
            Some(r) => r,
            None => return orkester::resolved(TileLoadResult::failed()),
        };

        let url_arc: Arc<str> = Arc::from(self.tile_url(id).as_str());
        let headers = Arc::clone(&self.headers);
        let ellipsoid = self.ellipsoid.clone();
        let bg = input.runtime.background();

        self.accessor
            .get(&url_arc, &headers, RequestPriority::NORMAL, None)
            .then(
                &bg,
                move |io_result: Result<AssetResponse, courtier::FetchError>| {
                    let result = (|| -> TileLoadResult {
                        let response = match io_result {
                            Ok(r) => r,
                            Err(_) => return TileLoadResult::failed(),
                        };
                        if response.check_status().is_err() {
                            return TileLoadResult::failed();
                        }
                        let data = response.decompressed_data().to_vec();
                        match decode_quantized_mesh(
                            &data,
                            rect.minimum_x,
                            rect.minimum_y,
                            rect.maximum_x,
                            rect.maximum_y,
                            id.level,
                            &ellipsoid,
                        ) {
                            Ok(QuantizedMeshResult { model, .. }) => {
                                let mut r = TileLoadResult::gltf(model);
                                r.gltf_up_axis = zukei::Axis::Y;
                                r.source_url = Some(url_arc.to_string());
                                r
                            }
                            Err(_) => TileLoadResult::failed(),
                        }
                    })();
                    orkester::resolved(result)
                },
            )
    }

    fn create_children(
        &self,
        tile: TileId,
        store: &TileStore,
        _ellipsoid: &Ellipsoid,
    ) -> TileChildrenResult {
        let id = match store.content_keys(tile).first() {
            Some(ContentKey::Quadtree(z, x, y)) => QuadtreeTileID::new(*z, *x, *y),
            _ => return TileChildrenResult::None,
        };

        if id.level >= self.max_zoom {
            return TileChildrenResult::None;
        }

        let nl = id.level + 1;
        let x2 = id.x * 2;
        let y2 = id.y * 2;

        let children: Vec<TileDescriptor> = [
            QuadtreeTileID::new(nl, x2, y2),
            QuadtreeTileID::new(nl, x2 + 1, y2),
            QuadtreeTileID::new(nl, x2, y2 + 1),
            QuadtreeTileID::new(nl, x2 + 1, y2 + 1),
        ]
        .into_iter()
        .filter_map(|cid| self.node_descriptor_for(cid))
        .collect();

        if children.is_empty() {
            TileChildrenResult::None
        } else {
            TileChildrenResult::Children(children)
        }
    }

    fn height_sampler(&self) -> Option<&dyn HeightSampler> {
        Some(self)
    }
}

impl HeightSampler for LayerJsonTerrainLoader {
    /// Returns the ellipsoid surface height (0.0) as a fast analytical
    /// fallback. Actual terrain height requires tile geometry queries via
    /// [`crate::tileset::Tileset::sample_height`].
    fn sample_height(&self, _lon: f64, _lat: f64, _ellipsoid: &Ellipsoid) -> Option<f64> {
        Some(0.0)
    }
}
