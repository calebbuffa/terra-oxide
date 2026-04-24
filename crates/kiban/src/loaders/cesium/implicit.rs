use courtier::AssetResponse;
use tairu::{TileFormat, decode_tile};

use super::axis_to_up;
use crate::loader::TileLoadResult;

/// Fetch bytes, detect format, decode -- shared by both implicit loaders.
fn decode_response(
    url: &str,
    io_result: Result<AssetResponse, courtier::FetchError>,
    gltf_up_axis: zukei::Axis,
) -> TileLoadResult {
    let response = match io_result {
        Ok(r) => r,
        Err(_) => return TileLoadResult::failed(),
    };
    if response.check_status().is_err() {
        return TileLoadResult::failed();
    }
    let data = response.decompressed_data();
    if data.is_empty() {
        return TileLoadResult::empty();
    }
    let format = TileFormat::detect(url, data);
    match decode_tile(data, &format, axis_to_up(gltf_up_axis), None) {
        Some(model) => {
            let mut r = TileLoadResult::gltf(model);
            r.gltf_up_axis = gltf_up_axis;
            r
        }
        None => TileLoadResult::empty(),
    }
}

pub mod quadtree {

    use std::sync::Arc;

    use courtier::{AssetAccessor, AssetResponse, RequestPriority};
    use glam::DMat4;
    use orkester::Task;
    use tairu::{SubtreeAvailability, implicit_tiling};
    use zukei::QuadtreeTileID;

    use crate::loader::{ContentLoader, TileChildrenResult, TileLoadInput, TileLoadResult};
    use crate::tile_store::{
        ContentKey, RefinementMode, TileDescriptor, TileFlags, TileId, TileKind,
    };

    use crate::loaders::cesium::{ImplicitLoaderShared, fetch_subtree};

    pub struct ImplicitQuadtreeLoader {
        inner: ImplicitLoaderShared,
    }

    impl ImplicitQuadtreeLoader {
        pub fn new(
            base_url: Arc<str>,
            headers: Arc<[(String, String)]>,
            accessor: Arc<dyn AssetAccessor>,
            content_url_template: impl Into<String>,
            subtree_url_template: impl Into<String>,
            subtree_levels: u32,
            available_levels: u32,
            root_bounds: zukei::SpatialBounds,
            gltf_up_axis: zukei::Axis,
        ) -> Self {
            Self {
                inner: ImplicitLoaderShared::new(
                    base_url,
                    headers,
                    accessor,
                    content_url_template,
                    subtree_url_template,
                    subtree_levels,
                    available_levels,
                    root_bounds,
                    gltf_up_axis,
                ),
            }
        }

        fn subtree_root_for(&self, tile: QuadtreeTileID) -> QuadtreeTileID {
            tile.subtree_root(self.inner.subtree_levels)
        }

        fn level_group_for(&self, subtree_root: QuadtreeTileID) -> usize {
            self.inner.level_group_for_level(subtree_root.level)
        }

        fn morton_for(subtree_root: QuadtreeTileID) -> u64 {
            subtree_root.morton_index()
        }
    }

    impl ContentLoader for ImplicitQuadtreeLoader {
        fn load_tile(&self, input: TileLoadInput) -> Task<TileLoadResult> {
            let tile_id = match input.content_keys.first() {
                Some(ContentKey::Quadtree(level, x, y)) => QuadtreeTileID::new(*level, *x, *y),
                _ => return orkester::resolved(TileLoadResult::failed()),
            };

            let subtree_root = self.subtree_root_for(tile_id);
            let level_group = self.level_group_for(subtree_root);
            let morton = Self::morton_for(subtree_root);

            // Check if the subtree is already loaded.
            let maybe_av = {
                let subtrees = self.inner.loaded_subtrees.lock().unwrap();
                subtrees.get(&(level_group, morton)).cloned()
            };

            if let Some(av) = maybe_av {
                // Subtree is loaded - check content availability.
                if !av.is_content_available_quad(subtree_root, tile_id, 0) {
                    return orkester::resolved(TileLoadResult::empty());
                }
                // Fetch the tile content.
                let url: Arc<str> = implicit_tiling::utlx::resolve_url_quad(
                    &self.inner.base_url,
                    &self.inner.content_url_template,
                    tile_id,
                )
                .into();
                let accessor = Arc::clone(&self.inner.accessor);
                let headers = Arc::clone(&self.inner.headers);
                let bg_ctx = input.runtime.background();
                let url_clone = Arc::clone(&url);
                let gltf_up_axis = self.inner.gltf_up_axis;

                return accessor
                    .get(&url, &headers, RequestPriority::NORMAL, None)
                    .then(
                        &bg_ctx,
                        move |io_result: Result<AssetResponse, courtier::FetchError>| {
                            orkester::resolved(super::decode_response(
                                &url_clone,
                                io_result,
                                gltf_up_axis,
                            ))
                        },
                    );
            }

            // Subtree not yet loaded -- fetch it, cache it, and return retry_later.
            let subtree_url: Arc<str> = implicit_tiling::utlx::resolve_url_quad(
                &self.inner.base_url,
                &self.inner.subtree_url_template,
                subtree_root,
            )
            .into();

            fetch_subtree(
                subtree_url,
                Arc::clone(&self.inner.accessor),
                Arc::clone(&self.inner.headers),
                tairu::SubdivisionScheme::Quadtree,
                self.inner.subtree_levels,
                level_group,
                morton,
                Arc::clone(&self.inner.loaded_subtrees),
                input.runtime.background(),
            )
        }

        fn create_children(
            &self,
            tile: TileId,
            store: &crate::tile_store::TileStore,
            _ellipsoid: &terra::Ellipsoid,
        ) -> TileChildrenResult {
            let tile_id = match store.content_keys(tile).first() {
                Some(ContentKey::Quadtree(level, x, y)) => QuadtreeTileID::new(*level, *x, *y),
                _ => QuadtreeTileID::new(0, 0, 0),
            };

            let subtree_root = self.subtree_root_for(tile_id);
            let level_group = self.level_group_for(subtree_root);
            let morton = Self::morton_for(subtree_root);

            let maybe_av = {
                let subtrees = self.inner.loaded_subtrees.lock().unwrap();
                subtrees.get(&(level_group, morton)).cloned()
            };

            match maybe_av {
                Some(av) => {
                    let children = populate_subtree(
                        &av,
                        self.inner.subtree_levels,
                        subtree_root,
                        tile_id,
                        store.geometric_error(tile),
                        store.refinement(tile),
                        store.world_transform(tile),
                        &self.inner.root_bounds,
                    );
                    if children.is_empty() {
                        TileChildrenResult::None
                    } else {
                        TileChildrenResult::Children(children)
                    }
                }
                // Subtree not yet cached; create_children is sync so signal retry.
                None => TileChildrenResult::RetryLater,
            }
        }
    }

    /// Build child [TileDescriptor]s for an implicit quadtree tile from a loaded
    /// subtree, mirroring cesium-native's populateSubtree.
    fn populate_subtree(
        av: &SubtreeAvailability,
        subtree_levels: u32,
        subtree_root: QuadtreeTileID,
        tile_id: QuadtreeTileID,
        parent_geometric_error: f64,
        refinement: RefinementMode,
        world_transform: DMat4,
        root_bounds: &zukei::SpatialBounds,
    ) -> Vec<TileDescriptor> {
        let relative_tile_level = tile_id.level - subtree_root.level;
        if relative_tile_level >= subtree_levels {
            return Vec::new();
        }

        let child_ids = tile_id.children();
        let child_error = parent_geometric_error * 0.5;

        child_ids
            .into_iter()
            .filter_map(|child_id| {
                let relative_child_level = relative_tile_level + 1;
                let child_morton = subtree_root.relative_morton_index(child_id);
                if relative_child_level == subtree_levels {
                    // Leaf of this subtree -- child subtree boundary.
                    if !av.is_child_subtree_available(child_morton) {
                        return None;
                    }
                    let bounds = root_bounds.subdivide_quad(child_id);
                    let globe_rectangle =
                        crate::loaders::cesium::tileset_json::globe_rect_from_bounds(&bounds);
                    Some(TileDescriptor::implicit_child(
                        bounds,
                        child_error,
                        refinement,
                        TileKind::EMPTY,
                        TileFlags::MIGHT_HAVE_LATENT_CHILDREN,
                        vec![ContentKey::Quadtree(child_id.level, child_id.x, child_id.y)],
                        world_transform,
                        globe_rectangle,
                    ))
                } else {
                    if !av.is_tile_available(relative_child_level, child_morton) {
                        return None;
                    }
                    let has_content =
                        av.is_content_available(relative_child_level, child_morton, 0);
                    let kind = if has_content {
                        TileKind::CONTENT
                    } else {
                        TileKind::EMPTY
                    };
                    let content_keys = if has_content {
                        vec![ContentKey::Quadtree(child_id.level, child_id.x, child_id.y)]
                    } else {
                        Vec::new()
                    };
                    let bounds = root_bounds.subdivide_quad(child_id);
                    let globe_rectangle =
                        crate::loaders::cesium::tileset_json::globe_rect_from_bounds(&bounds);
                    Some(TileDescriptor::implicit_child(
                        bounds,
                        child_error,
                        refinement,
                        kind,
                        TileFlags::empty(),
                        content_keys,
                        world_transform,
                        globe_rectangle,
                    ))
                }
            })
            .collect()
    }
}

pub mod octree {
    use std::sync::Arc;

    use courtier::{AssetAccessor, RequestPriority};
    use glam::DMat4;
    use orkester::Task;
    use tairu::{SubtreeAvailability, implicit_tiling};
    use zukei::OctreeTileID;

    use crate::loader::{ContentLoader, TileChildrenResult, TileLoadInput, TileLoadResult};
    use crate::tile_store::{
        ContentKey, RefinementMode, TileDescriptor, TileFlags, TileId, TileKind,
    };

    use crate::loaders::cesium::{ImplicitLoaderShared, fetch_subtree};

    pub struct ImplicitOctreeLoader {
        inner: ImplicitLoaderShared,
    }

    impl ImplicitOctreeLoader {
        pub fn new(
            base_url: Arc<str>,
            headers: Arc<[(String, String)]>,
            accessor: Arc<dyn AssetAccessor>,
            content_url_template: impl Into<String>,
            subtree_url_template: impl Into<String>,
            subtree_levels: u32,
            available_levels: u32,
            root_bounds: zukei::SpatialBounds,
            gltf_up_axis: zukei::Axis,
        ) -> Self {
            Self {
                inner: ImplicitLoaderShared::new(
                    base_url,
                    headers,
                    accessor,
                    content_url_template,
                    subtree_url_template,
                    subtree_levels,
                    available_levels,
                    root_bounds,
                    gltf_up_axis,
                ),
            }
        }

        fn subtree_root_for(&self, tile: OctreeTileID) -> OctreeTileID {
            tile.subtree_root(self.inner.subtree_levels)
        }

        fn level_group_for(&self, subtree_root: OctreeTileID) -> usize {
            self.inner.level_group_for_level(subtree_root.level)
        }

        fn morton_for(subtree_root: OctreeTileID) -> u64 {
            subtree_root.morton_index()
        }
    }

    impl ContentLoader for ImplicitOctreeLoader {
        fn load_tile(&self, input: TileLoadInput) -> Task<TileLoadResult> {
            let tile_id = match input.content_keys.first() {
                Some(ContentKey::Octree(level, x, y, z)) => OctreeTileID::new(*level, *x, *y, *z),
                _ => return orkester::resolved(TileLoadResult::failed()),
            };

            let subtree_root = self.subtree_root_for(tile_id);
            let level_group = self.level_group_for(subtree_root);
            let morton = Self::morton_for(subtree_root);

            // Check if the subtree is already loaded.
            let maybe_av = {
                let subtrees = self.inner.loaded_subtrees.lock().unwrap();
                subtrees.get(&(level_group, morton)).cloned()
            };

            if let Some(av) = maybe_av {
                // Subtree is loaded - check content availability.
                if !av.is_content_available_oct(subtree_root, tile_id, 0) {
                    return orkester::resolved(TileLoadResult::empty());
                }
                // Fetch the tile content.
                let url: Arc<str> = implicit_tiling::utlx::resolve_url_oct(
                    &self.inner.base_url,
                    &self.inner.content_url_template,
                    tile_id,
                )
                .into();
                let accessor = Arc::clone(&self.inner.accessor);
                let headers = Arc::clone(&self.inner.headers);
                let bg_ctx = input.runtime.background();
                let url_clone = Arc::clone(&url);
                let gltf_up_axis = self.inner.gltf_up_axis;

                return accessor
                    .get(&url, &headers, RequestPriority::NORMAL, None)
                    .then(&bg_ctx, move |io_result| {
                        orkester::resolved(super::decode_response(
                            &url_clone,
                            io_result,
                            gltf_up_axis,
                        ))
                    });
            }

            // Subtree not yet loaded - fetch it, cache it, and return retry_later.
            let subtree_url: Arc<str> = implicit_tiling::utlx::resolve_url_oct(
                &self.inner.base_url,
                &self.inner.subtree_url_template,
                subtree_root,
            )
            .into();

            fetch_subtree(
                subtree_url,
                Arc::clone(&self.inner.accessor),
                Arc::clone(&self.inner.headers),
                tairu::SubdivisionScheme::Octree,
                self.inner.subtree_levels,
                level_group,
                morton,
                Arc::clone(&self.inner.loaded_subtrees),
                input.runtime.background(),
            )
        }

        fn create_children(
            &self,
            tile: TileId,
            store: &crate::tile_store::TileStore,
            _ellipsoid: &terra::Ellipsoid,
        ) -> TileChildrenResult {
            let tile_id = match store.content_keys(tile).first() {
                Some(ContentKey::Octree(level, x, y, z)) => OctreeTileID::new(*level, *x, *y, *z),
                _ => OctreeTileID::new(0, 0, 0, 0),
            };

            let subtree_root = self.subtree_root_for(tile_id);
            let level_group = self.level_group_for(subtree_root);
            let morton = Self::morton_for(subtree_root);

            let maybe_av = {
                let subtrees = self.inner.loaded_subtrees.lock().unwrap();
                subtrees.get(&(level_group, morton)).cloned()
            };

            match maybe_av {
                Some(av) => {
                    let children = populate_octree_subtree(
                        &av,
                        self.inner.subtree_levels,
                        subtree_root,
                        tile_id,
                        store.geometric_error(tile),
                        store.refinement(tile),
                        store.world_transform(tile),
                        &self.inner.root_bounds,
                    );
                    if children.is_empty() {
                        TileChildrenResult::None
                    } else {
                        TileChildrenResult::Children(children)
                    }
                }
                // Subtree not yet cached; create_children is sync so signal retry.
                None => TileChildrenResult::RetryLater,
            }
        }
    }

    /// Build child [TileDescriptor]s for an implicit octree tile from a loaded
    /// subtreed.
    fn populate_octree_subtree(
        av: &SubtreeAvailability,
        subtree_levels: u32,
        subtree_root: OctreeTileID,
        tile_id: OctreeTileID,
        parent_geometric_error: f64,
        refinement: RefinementMode,
        world_transform: DMat4,
        root_bounds: &zukei::SpatialBounds,
    ) -> Vec<TileDescriptor> {
        let relative_level = tile_id.level - subtree_root.level;
        if relative_level >= subtree_levels {
            return Vec::new();
        }

        let child_ids = tile_id.children();
        let child_error = parent_geometric_error * 0.5;

        child_ids
            .into_iter()
            .filter_map(|child_id| {
                let rel_child_level = relative_level + 1;
                let child_morton = subtree_root.relative_morton_index(child_id);

                if rel_child_level == subtree_levels {
                    // Leaf of this subtree -- child subtree boundary.
                    if !av.is_child_subtree_available(child_morton) {
                        return None;
                    }
                    let bounds = root_bounds.subdivide_oct(child_id);
                    let globe_rectangle =
                        crate::loaders::cesium::tileset_json::globe_rect_from_bounds(&bounds);
                    Some(TileDescriptor::implicit_child(
                        bounds,
                        child_error,
                        refinement,
                        TileKind::EMPTY,
                        TileFlags::MIGHT_HAVE_LATENT_CHILDREN,
                        vec![ContentKey::Octree(
                            child_id.level,
                            child_id.x,
                            child_id.y,
                            child_id.z,
                        )],
                        world_transform,
                        globe_rectangle,
                    ))
                } else {
                    if !av.is_tile_available(rel_child_level, child_morton) {
                        return None;
                    }
                    let has_content = av.is_content_available(rel_child_level, child_morton, 0);
                    let kind = if has_content {
                        TileKind::CONTENT
                    } else {
                        TileKind::EMPTY
                    };
                    let content_keys = if has_content {
                        vec![ContentKey::Octree(
                            child_id.level,
                            child_id.x,
                            child_id.y,
                            child_id.z,
                        )]
                    } else {
                        Vec::new()
                    };
                    let bounds = root_bounds.subdivide_oct(child_id);
                    let globe_rectangle =
                        crate::loaders::cesium::tileset_json::globe_rect_from_bounds(&bounds);
                    Some(TileDescriptor::implicit_child(
                        bounds,
                        child_error,
                        refinement,
                        kind,
                        TileFlags::empty(),
                        content_keys,
                        world_transform,
                        globe_rectangle,
                    ))
                }
            })
            .collect()
    }
}

pub mod s2 {
    use std::sync::Arc;

    use courtier::{AssetAccessor, AssetResponse, RequestPriority};
    use glam::DMat4;
    use orkester::Task;
    use tairu::{SubtreeAvailability, implicit_tiling};
    use tairu::{TileFormat, decode_tile};
    use terra::{Cartographic, Ellipsoid, GlobeRectangle};
    use zukei::{BoundingSphere, S2CellId, SpatialBounds};

    use crate::loader::{ContentLoader, TileChildrenResult, TileLoadInput, TileLoadResult};
    use crate::tile_store::{
        ContentKey, RefinementMode, TileDescriptor, TileFlags, TileId, TileKind,
    };

    use crate::loaders::cesium::{ImplicitLoaderShared, axis_to_up, fetch_subtree};

    pub struct ImplicitS2Loader {
        inner: ImplicitLoaderShared,
        /// Root S2 cells at the top of this implicit tileset.
        root_cells: Vec<S2CellId>,
        minimum_height: f64,
        maximum_height: f64,
    }

    impl ImplicitS2Loader {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            base_url: Arc<str>,
            headers: Arc<[(String, String)]>,
            accessor: Arc<dyn AssetAccessor>,
            content_url_template: impl Into<String>,
            subtree_url_template: impl Into<String>,
            subtree_levels: u32,
            available_levels: u32,
            root_cells: Vec<S2CellId>,
            minimum_height: f64,
            maximum_height: f64,
            gltf_up_axis: zukei::Axis,
        ) -> Self {
            // S2 root bounds are derived from the root cells; use a unit sphere as
            // a conservative placeholder — the traversal bounds are per-tile.
            let root_bounds = SpatialBounds::Sphere(BoundingSphere::new(
                glam::DVec3::ZERO,
                6_371_000.0, // Earth mean radius in metres — conservative root bound
            ));
            Self {
                inner: ImplicitLoaderShared::new(
                    base_url,
                    headers,
                    accessor,
                    content_url_template,
                    subtree_url_template,
                    subtree_levels,
                    available_levels,
                    root_bounds,
                    gltf_up_axis,
                ),
                root_cells,
                minimum_height,
                maximum_height,
            }
        }

        /// Which face/subtree-group does this cell live in?
        fn subtree_root_for(&self, cell: S2CellId) -> S2CellId {
            cell.parent_at_level(
                (cell.level() / self.inner.subtree_levels) * self.inner.subtree_levels,
            )
        }

        fn level_group_for(&self, subtree_root: S2CellId) -> usize {
            self.inner.level_group_for_level(subtree_root.level())
        }

        /// Cache key: (level_group, hilbert_pos_at_subtree_root).
        fn cache_key(subtree_root: S2CellId) -> u64 {
            subtree_root.raw()
        }
    }

    impl ContentLoader for ImplicitS2Loader {
        fn load_tile(&self, input: TileLoadInput) -> Task<TileLoadResult> {
            let cell = match input.content_keys.first() {
                Some(ContentKey::S2(raw)) => S2CellId::from_raw(*raw),
                _ => return orkester::resolved(TileLoadResult::failed()),
            };

            let subtree_root = self.subtree_root_for(cell);
            let level_group = self.level_group_for(subtree_root);
            let cache_key = Self::cache_key(subtree_root);

            // Check if subtree is already loaded.
            let maybe_av = {
                let subtrees = self.inner.loaded_subtrees.lock().unwrap();
                subtrees.get(&(level_group, cache_key)).cloned()
            };

            if let Some(av) = maybe_av {
                let hilbert_pos = cell.subtree_hilbert_position(subtree_root);
                let rel_level = (cell.level() - subtree_root.level()) as u64;
                if !av.is_content_available(rel_level as u32, hilbert_pos, 0) {
                    return orkester::resolved(TileLoadResult::empty());
                }
                let url: Arc<str> = implicit_tiling::utlx::resolve_url_s2(
                    &self.inner.base_url,
                    &self.inner.content_url_template,
                    cell,
                )
                .into();
                let accessor = Arc::clone(&self.inner.accessor);
                let headers = Arc::clone(&self.inner.headers);
                let bg_ctx = input.runtime.background();
                let url_clone = Arc::clone(&url);
                let gltf_up_axis = self.inner.gltf_up_axis;

                return accessor
                    .get(&url, &headers, RequestPriority::NORMAL, None)
                    .then(
                        &bg_ctx,
                        move |io_result: Result<AssetResponse, courtier::FetchError>| {
                            orkester::resolved(decode_s2_response(
                                &url_clone,
                                io_result,
                                gltf_up_axis,
                            ))
                        },
                    );
            }

            // Subtree not yet loaded — fetch it and return retry_later.
            let subtree_url: Arc<str> = implicit_tiling::utlx::resolve_url_s2(
                &self.inner.base_url,
                &self.inner.subtree_url_template,
                subtree_root,
            )
            .into();

            fetch_subtree(
                subtree_url,
                Arc::clone(&self.inner.accessor),
                Arc::clone(&self.inner.headers),
                tairu::SubdivisionScheme::S2,
                self.inner.subtree_levels,
                level_group,
                cache_key,
                Arc::clone(&self.inner.loaded_subtrees),
                input.runtime.background(),
            )
        }

        fn create_children(
            &self,
            tile: TileId,
            store: &crate::tile_store::TileStore,
            ellipsoid: &Ellipsoid,
        ) -> TileChildrenResult {
            let cell = match store.content_keys(tile).first() {
                Some(ContentKey::S2(raw)) => S2CellId::from_raw(*raw),
                // Root tile: expand all root cells.
                _ => {
                    if self.root_cells.is_empty() {
                        return TileChildrenResult::None;
                    }
                    let children = self
                        .root_cells
                        .iter()
                        .map(|&c| {
                            cell_to_descriptor(
                                c,
                                store.geometric_error(tile),
                                store.refinement(tile),
                                store.world_transform(tile),
                                self.minimum_height,
                                self.maximum_height,
                                false,
                                ellipsoid,
                            )
                        })
                        .collect();
                    return TileChildrenResult::Children(children);
                }
            };

            let subtree_root = self.subtree_root_for(cell);
            let level_group = self.level_group_for(subtree_root);
            let cache_key = Self::cache_key(subtree_root);

            let maybe_av = {
                let subtrees = self.inner.loaded_subtrees.lock().unwrap();
                subtrees.get(&(level_group, cache_key)).cloned()
            };

            match maybe_av {
                Some(av) => {
                    let children = populate_s2_subtree(
                        &av,
                        self.inner.subtree_levels,
                        subtree_root,
                        cell,
                        store.geometric_error(tile),
                        store.refinement(tile),
                        store.world_transform(tile),
                        self.minimum_height,
                        self.maximum_height,
                        ellipsoid,
                    );
                    if children.is_empty() {
                        TileChildrenResult::None
                    } else {
                        TileChildrenResult::Children(children)
                    }
                }
                None => TileChildrenResult::RetryLater,
            }
        }
    }

    /// Build child descriptors for an S2 implicit tile from a loaded subtree.
    fn populate_s2_subtree(
        av: &SubtreeAvailability,
        subtree_levels: u32,
        subtree_root: S2CellId,
        cell: S2CellId,
        parent_geometric_error: f64,
        refinement: RefinementMode,
        world_transform: DMat4,
        min_height: f64,
        max_height: f64,
        ellipsoid: &Ellipsoid,
    ) -> Vec<TileDescriptor> {
        let relative_level = cell.level() - subtree_root.level();
        if relative_level >= subtree_levels {
            return Vec::new();
        }

        let child_error = parent_geometric_error * 0.5;

        cell.children()
            .iter()
            .filter_map(|&child| {
                let child_hilbert = child.subtree_hilbert_position(subtree_root);
                let child_rel_level = relative_level + 1;

                if child_rel_level == subtree_levels {
                    // Subtree boundary.
                    if !av.is_child_subtree_available(child_hilbert) {
                        return None;
                    }
                    Some(cell_to_descriptor(
                        child,
                        child_error,
                        refinement,
                        world_transform,
                        min_height,
                        max_height,
                        true,
                        ellipsoid,
                    ))
                } else {
                    if !av.is_tile_available(child_rel_level, child_hilbert) {
                        return None;
                    }
                    let has_content = av.is_content_available(child_rel_level, child_hilbert, 0);
                    Some(cell_to_descriptor_with_content(
                        child,
                        child_error,
                        refinement,
                        world_transform,
                        min_height,
                        max_height,
                        has_content,
                        false,
                        ellipsoid,
                    ))
                }
            })
            .collect()
    }

    /// Build a [`TileDescriptor`] for an S2 cell at a subtree leaf (child subtree
    /// boundary).  Sets `MIGHT_HAVE_LATENT_CHILDREN`.
    fn cell_to_descriptor(
        cell: S2CellId,
        geometric_error: f64,
        refinement: RefinementMode,
        world_transform: DMat4,
        min_height: f64,
        max_height: f64,
        latent: bool,
        ellipsoid: &Ellipsoid,
    ) -> TileDescriptor {
        cell_to_descriptor_with_content(
            cell,
            geometric_error,
            refinement,
            world_transform,
            min_height,
            max_height,
            false,
            latent,
            ellipsoid,
        )
    }

    fn cell_to_descriptor_with_content(
        cell: S2CellId,
        geometric_error: f64,
        refinement: RefinementMode,
        world_transform: DMat4,
        min_height: f64,
        max_height: f64,
        has_content: bool,
        latent: bool,
        ellipsoid: &Ellipsoid,
    ) -> TileDescriptor {
        let bounds = s2_cell_bounds(cell, min_height, max_height, ellipsoid);
        let globe_rectangle = s2_cell_globe_rectangle(cell);
        let kind = if has_content || latent {
            if latent {
                TileKind::EMPTY
            } else {
                TileKind::CONTENT
            }
        } else {
            TileKind::EMPTY
        };
        let flags = if latent {
            TileFlags::MIGHT_HAVE_LATENT_CHILDREN
        } else {
            TileFlags::empty()
        };
        let content_keys = if has_content || latent {
            vec![ContentKey::S2(cell.raw())]
        } else {
            Vec::new()
        };

        TileDescriptor::implicit_child(
            bounds,
            geometric_error,
            refinement,
            kind,
            flags,
            content_keys,
            world_transform,
            globe_rectangle,
        )
    }

    /// Approximate bounding sphere for an S2 cell with a height range.
    ///
    /// Converts the cell's lat/lon corners to ECEF via the ellipsoid, then wraps
    /// them in a bounding sphere.
    fn s2_cell_bounds(
        cell: S2CellId,
        min_height: f64,
        max_height: f64,
        ellipsoid: &Ellipsoid,
    ) -> SpatialBounds {
        let (west, south, east, north) = cell.lat_lon_bounds_radians();
        // Eight corners: 4 lat/lon corners × 2 heights.
        let corners = [
            (west, south, min_height),
            (east, south, min_height),
            (west, north, min_height),
            (east, north, min_height),
            (west, south, max_height),
            (east, south, max_height),
            (west, north, max_height),
            (east, north, max_height),
        ];

        let ecef_corners: Vec<glam::DVec3> = corners
            .iter()
            .map(|&(lon, lat, h)| ellipsoid.cartographic_to_ecef(Cartographic::new(lon, lat, h)))
            .collect();

        let center = ecef_corners
            .iter()
            .fold(glam::DVec3::ZERO, |acc, &p| acc + p)
            / ecef_corners.len() as f64;

        let radius = ecef_corners
            .iter()
            .map(|&p| (p - center).length())
            .fold(0.0_f64, f64::max);

        SpatialBounds::Sphere(BoundingSphere::new(center, radius))
    }

    fn s2_cell_globe_rectangle(cell: S2CellId) -> Option<GlobeRectangle> {
        let (west, south, east, north) = cell.lat_lon_bounds_radians();
        Some(GlobeRectangle::new(west, south, east, north))
    }

    fn decode_s2_response(
        url: &str,
        io_result: Result<AssetResponse, courtier::FetchError>,
        gltf_up_axis: zukei::Axis,
    ) -> TileLoadResult {
        let response = match io_result {
            Ok(r) => r,
            Err(_) => return TileLoadResult::failed(),
        };
        if response.check_status().is_err() {
            return TileLoadResult::failed();
        }
        let data = response.decompressed_data();
        if data.is_empty() {
            return TileLoadResult::empty();
        }
        let format = TileFormat::detect(url, data);
        match decode_tile(data, &format, axis_to_up(gltf_up_axis), None) {
            Some(model) => {
                let mut r = TileLoadResult::gltf(model);
                r.gltf_up_axis = gltf_up_axis;
                r
            }
            None => TileLoadResult::empty(),
        }
    }
}
