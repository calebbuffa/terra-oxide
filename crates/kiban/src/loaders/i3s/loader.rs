//! [`I3sContentLoader`] - I3S (Indexed 3D Scene Layers) content loader.
//!
//! Loads an I3S scene layer, walks its node page tree, and produces
//! renderable tiles from I3S geometry buffers decoded into
//! [`moderu::GltfModel`] via `i3s-content`.

use std::sync::Arc;

use courtier::{AssetAccessor, AssetResponse, RequestPriority};
use glam::{DMat4, DQuat, DVec3};
use i3s::cmn::{GeometryBuffer, Node, Obb, SceneLayerInfo};
use i3s::{decode_geometry, geometry_url, layer_url, node_page_url, select_geometry_buffer};
use orkester::Task;
use zukei::{Axis, SpatialBounds};

use crate::loader::{
    ContentLoader, TileChildrenResult, TileContentKind, TileLoadInput, TileLoadResult,
    TileLoadResultState,
};
use crate::tile_store::{
    ContentKey, RefinementMode, TileDescriptor, TileFlags, TileId, TileKind, TileStore,
};

use super::node_pages::NodePageCache;

#[derive(Debug, thiserror::Error)]
pub enum I3sLoaderError {
    #[error("I/O: {0}")]
    Fetch(#[from] courtier::FetchError),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("layer has no node pages definition")]
    NoNodePages,
    #[error("no supported geometry buffer in layer")]
    NoGeometry,
}

/// Streams an I3S scene layer as a 3D Tiles–compatible tile tree.
///
/// ## Lifecycle
///
/// 1. [`I3sContentLoader::from_url`] fetches the `SceneLayerInfo` (layer 0),
///    selects the best geometry buffer, fetches node page 0, and builds the
///    root [`TileDescriptor`].
///
/// 2. `create_children` expands a node's page on demand (RetryLater while the
///    page fetch is in flight).
///
/// 3. `load_tile` fetches the node's geometry resource, decodes it via
///    `i3s::decode_geometry`, and returns a `Renderable` result.
pub struct I3sContentLoader {
    base_url: Arc<str>,
    headers: Arc<[(String, String)]>,
    accessor: Arc<dyn AssetAccessor>,
    /// Background context for node-page fetches from `create_children`.
    bg: orkester::Context,
    /// Selected buffer index within the chosen geometry definition.
    geom_buf_idx: usize,
    /// The selected geometry buffer descriptor.
    geom_buf: Arc<GeometryBuffer>,
    /// Number of nodes per node page.
    nodes_per_page: i64,
    /// Shared page cache.
    page_cache: Arc<NodePageCache>,
}

impl I3sContentLoader {
    /// Async factory: fetch `SceneLayerInfo`, select geometry, fetch page 0.
    ///
    /// Uses two chained `.then()` calls - no blocking anywhere.
    /// Returns `(root_descriptor, Arc<dyn ContentLoader>)` on success.
    pub fn from_url(
        base_url: impl Into<Arc<str>>,
        headers: impl Into<Arc<[(String, String)]>>,
        accessor: Arc<dyn AssetAccessor>,
        bg: orkester::Context,
    ) -> Task<Result<(TileDescriptor, Arc<dyn ContentLoader>), I3sLoaderError>> {
        let base_url: Arc<str> = base_url.into();
        let headers: Arc<[(String, String)]> = headers.into();

        let base_url2 = Arc::clone(&base_url);
        let headers2 = Arc::clone(&headers);
        let accessor2 = Arc::clone(&accessor);
        let bg2 = bg.clone();
        let bg3 = bg.clone();

        let lyr_url = layer_url(&base_url);

        // Step 1: fetch SceneLayerInfo.
        accessor
            .get(&lyr_url, &headers, RequestPriority::HIGH, None)
            .then(
                &bg,
                move |io_result: Result<AssetResponse, courtier::FetchError>| {
                    let parse_result = (|| -> Result<_, I3sLoaderError> {
                        let resp = io_result?;
                        resp.check_status()?;
                        let layer: SceneLayerInfo =
                            serde_json::from_slice(resp.decompressed_data())?;

                        let (geom_def_idx, geom_buf_idx) =
                            select_geometry_buffer(&layer).ok_or(I3sLoaderError::NoGeometry)?;
                        let nodes_per_page = layer
                            .node_pages
                            .as_ref()
                            .ok_or(I3sLoaderError::NoNodePages)?
                            .nodes_per_page;
                        let geom_buf = Arc::new(
                            layer.geometry_definitions[geom_def_idx].geometry_buffers[geom_buf_idx]
                                .clone(),
                        );
                        Ok((geom_buf_idx, geom_buf, nodes_per_page))
                    })();

                    match parse_result {
                        Err(e) => orkester::resolved(Err(e)),
                        Ok((geom_buf_idx, geom_buf, nodes_per_page)) => {
                            // Step 2: fetch node page 0 to build the root descriptor.
                            let page0_url = node_page_url(&base_url2, 0);
                            accessor2
                                .get(&page0_url, &headers2, RequestPriority::HIGH, None)
                                .then(
                                    &bg2,
                                    move |io_result2: Result<
                                        AssetResponse,
                                        courtier::FetchError,
                                    >| {
                                        let result = (|| -> Result<
                                        (TileDescriptor, Arc<dyn ContentLoader>),
                                        I3sLoaderError,
                                    > {
                                        let resp = io_result2?;
                                        resp.check_status()?;
                                        let page0: i3s::cmn::NodePage =
                                            serde_json::from_slice(resp.decompressed_data())?;

                                        let root_node =
                                            page0.nodes.first().ok_or_else(|| {
                                                I3sLoaderError::Fetch(courtier::FetchError::Json(
                                                    "page 0 is empty".into(),
                                                ))
                                            })?;

                                        let root_bounds = obb_to_bounds(&root_node.obb);
                                        let root_error =
                                            root_node.lod_threshold.unwrap_or(1024.0);
                                        let has_children = root_node
                                            .children
                                            .as_ref()
                                            .map_or(false, |c| !c.is_empty());
                                        let flags = if has_children {
                                            TileFlags::MIGHT_HAVE_LATENT_CHILDREN
                                        } else {
                                            TileFlags::empty()
                                        };

                                        let page_cache = Arc::new(NodePageCache::new());
                                        let loader: Arc<dyn ContentLoader> =
                                            Arc::new(I3sContentLoader {
                                                base_url: Arc::clone(&base_url2),
                                                headers: Arc::clone(&headers2),
                                                accessor: Arc::clone(&accessor2),
                                                bg: bg3,
                                                geom_buf_idx,
                                                geom_buf,
                                                nodes_per_page,
                                                page_cache,
                                            });

                                        let root_desc = TileDescriptor {
                                            bounds: root_bounds,
                                            geometric_error: root_error,
                                            refinement: RefinementMode::Replace,
                                            kind: TileKind::CONTENT,
                                            flags,
                                            content_keys: vec![ContentKey::Custom(Arc::new(
                                                I3sNodeKey(0),
                                            ))],
                                            world_transform: DMat4::IDENTITY,
                                            children: Vec::new(),
                                            content_bounds: None,
                                            viewer_request_volume: None,
                                            globe_rectangle: None,
                                            content_max_age: None,
                                            loader_index: None,
                                        };

                                        Ok((root_desc, loader))
                                    })();
                                        orkester::resolved(result)
                                    },
                                )
                        }
                    }
                },
            )
    }
}

impl ContentLoader for I3sContentLoader {
    fn load_tile(&self, input: TileLoadInput) -> Task<TileLoadResult> {
        let node_id = match input.content_keys.iter().find_map(|k| {
            if let ContentKey::Custom(arc) = k {
                arc.downcast_ref::<I3sNodeKey>().map(|k| k.0)
            } else {
                None
            }
        }) {
            Some(id) => id,
            None => return orkester::resolved(TileLoadResult::failed()),
        };

        let base_url = Arc::clone(&self.base_url);
        let headers = Arc::clone(&self.headers);
        let accessor = Arc::clone(&self.accessor);
        let geom_buf = Arc::clone(&self.geom_buf);
        let geom_buf_idx = self.geom_buf_idx;
        let bg = input.runtime.background();
        let geom_url = geometry_url(&base_url, node_id, node_id, geom_buf_idx);
        let geom_url2 = geom_url.clone();

        accessor
            .get(&geom_url, &headers, RequestPriority::NORMAL, None)
            .then(
                &bg,
                move |io_result: Result<AssetResponse, courtier::FetchError>| {
                    let result = (|| -> Option<TileLoadResult> {
                        let resp = io_result.ok()?;
                        resp.check_status().ok()?;
                        let model = decode_geometry(resp.decompressed_data(), &geom_buf, 0).ok()?;
                        Some(TileLoadResult {
                            content: TileContentKind::Gltf(model),
                            state: TileLoadResultState::Success,
                            gltf_up_axis: Axis::Y,
                            updated_bounds: None,
                            raster_overlay_details: None,
                            tile_initializer: None,
                            source_url: Some(geom_url2),
                        })
                    })();
                    orkester::resolved(result.unwrap_or_else(TileLoadResult::failed))
                },
            )
    }

    fn create_children(
        &self,
        tile: TileId,
        store: &TileStore,
        _ellipsoid: &terra::Ellipsoid,
    ) -> TileChildrenResult {
        let node_id = match store.content_keys(tile).iter().find_map(|k| {
            if let ContentKey::Custom(arc) = k {
                arc.downcast_ref::<I3sNodeKey>().map(|k| k.0)
            } else {
                None
            }
        }) {
            Some(id) => id,
            None => return TileChildrenResult::None,
        };

        let page_id = NodePageCache::page_id(node_id, self.nodes_per_page);
        let page_offset = NodePageCache::page_offset(node_id, self.nodes_per_page);

        match self.page_cache.get(page_id) {
            Some(Ok(page)) => match page.nodes.get(page_offset) {
                Some(node) => {
                    TileChildrenResult::Children(build_child_descriptors(node, self.nodes_per_page))
                }
                None => TileChildrenResult::None,
            },
            Some(Err(())) => TileChildrenResult::None,
            None => {
                let page_url = node_page_url(&self.base_url, page_id);
                self.page_cache.ensure_fetched(
                    page_id,
                    page_url,
                    Arc::clone(&self.accessor),
                    Arc::clone(&self.headers),
                    self.bg.clone(),
                );
                TileChildrenResult::RetryLater
            }
        }
    }
}

/// Custom content key payload: the I3S node index.
pub struct I3sNodeKey(pub u64);

fn obb_to_bounds(obb: &Obb) -> SpatialBounds {
    let zukei_obb = zukei::OrientedBoundingBox::from_quat(
        DVec3::from_array(obb.center),
        DQuat::from_xyzw(
            obb.quaternion[0],
            obb.quaternion[1],
            obb.quaternion[2],
            obb.quaternion[3],
        ),
        DVec3::from_array(obb.half_size),
    );
    SpatialBounds::from(zukei_obb)
}

fn build_child_descriptors(node: &Node, _nodes_per_page: i64) -> Vec<TileDescriptor> {
    let child_indices = match &node.children {
        Some(c) => c.as_slice(),
        None => return Vec::new(),
    };

    child_indices
        .iter()
        .map(|&child_idx| TileDescriptor {
            bounds: SpatialBounds::Empty,
            geometric_error: 0.0,
            refinement: RefinementMode::Replace,
            kind: TileKind::CONTENT,
            flags: TileFlags::MIGHT_HAVE_LATENT_CHILDREN,
            content_keys: vec![ContentKey::Custom(Arc::new(I3sNodeKey(child_idx as u64)))],
            world_transform: DMat4::IDENTITY,
            children: Vec::new(),
            content_bounds: None,
            viewer_request_volume: None,
            globe_rectangle: None,
            content_max_age: None,
            loader_index: None,
        })
        .collect()
}
