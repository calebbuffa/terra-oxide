mod ellipsoid;
mod implicit;
mod terrain;
mod tileset_json;

pub use arazi::{QuantizedMeshResult, decode_quantized_mesh};
pub use ellipsoid::EllipsoidTilesetLoader;
pub use implicit::octree::ImplicitOctreeLoader;
pub use implicit::quadtree::ImplicitQuadtreeLoader;
pub use implicit::s2::ImplicitS2Loader;
pub use terrain::{LayerJsonTerrainLoader, TerrainLoaderError};
pub use tileset_json::{TilesetInitResult, TilesetJsonError, TilesetJsonLoader};
pub use zukei::QuadtreeTileRectangularRange;

use std::sync::{Arc, Mutex};

use courtier::{AssetAccessor, RequestPriority};
use orkester::{Context, Task};
use outil::BoundedFifoCache;
use tairu::{SubdivisionScheme, SubtreeAvailability};

use crate::loader::TileLoadResult;

/// Convert a [`zukei::Axis`] to a [`moderu::UpAxis`] for tagging glTF extras.
pub(super) fn axis_to_up(axis: zukei::Axis) -> moderu::UpAxis {
    match axis {
        zukei::Axis::X => moderu::UpAxis::X,
        zukei::Axis::Y => moderu::UpAxis::Y,
        zukei::Axis::Z => moderu::UpAxis::Z,
    }
}

/// Maximum number of subtree availability objects held in each implicit loader's
/// cache.  Each entry is typically a few KB; 512 entries ~ a few MB.
pub(super) const DEFAULT_MAX_SUBTREE_CACHE_ENTRIES: usize = 512;

/// Fields and constructor shared by [`ImplicitQuadtreeLoader`] and
/// [`ImplicitOctreeLoader`].
///
/// Both loaders carry identical network/cache configuration.  Centralising
/// them here eliminates the duplicate struct definitions and ensures the
/// initialisation logic (in particular the `level_groups` calculation) lives
/// in exactly one place.
pub(super) struct ImplicitLoaderShared {
    pub base_url: Arc<str>,
    pub headers: Arc<[(String, String)]>,
    pub accessor: Arc<dyn AssetAccessor>,
    pub content_url_template: String,
    pub subtree_url_template: String,
    pub subtree_levels: u32,
    pub available_levels: u32,
    /// Bounding volume of the root implicit tile (level 0).
    /// Used to subdivide child bounds.
    pub root_bounds: zukei::SpatialBounds,
    /// Up-axis declared in the parent tileset's `asset.gltfUpAxis`.
    /// Propagated into each loaded tile's `extras["gltfUpAxis"]`.
    pub gltf_up_axis: zukei::Axis,
    /// FIFO-bounded cache of subtree availability data keyed by
    /// `(level_group, morton_code)`.
    /// `Arc` so task closures can clone a handle without borrowing the loader.
    pub loaded_subtrees: Arc<Mutex<BoundedFifoCache<(usize, u64), SubtreeAvailability>>>,
}

impl ImplicitLoaderShared {
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
            base_url,
            headers,
            accessor,
            content_url_template: content_url_template.into(),
            subtree_url_template: subtree_url_template.into(),
            subtree_levels,
            available_levels,
            root_bounds,
            gltf_up_axis,
            loaded_subtrees: Arc::new(Mutex::new(BoundedFifoCache::new(
                DEFAULT_MAX_SUBTREE_CACHE_ENTRIES,
            ))),
        }
    }

    /// Level-group index for a subtree whose root is at `level`.
    ///
    /// Called by each concrete loader's type-specific `level_group_for` helper.
    pub fn level_group_for_level(&self, level: u32) -> usize {
        (level / self.subtree_levels) as usize
    }
}

/// Fetch a subtree file, parse it, store it in the cache, and return
/// [`TileLoadResult::retry_later`] so the caller retries tile loading.
///
/// Both implicit loaders use an identical subtree-fetch workflow - the only
/// difference is the [`SubdivisionScheme`] passed to [`tairu::parse_subtree`].
pub(super) fn fetch_subtree(
    subtree_url: Arc<str>,
    accessor: Arc<dyn AssetAccessor>,
    headers: Arc<[(String, String)]>,
    scheme: SubdivisionScheme,
    subtree_levels: u32,
    level_group: usize,
    morton: u64,
    loaded_subtrees: Arc<Mutex<BoundedFifoCache<(usize, u64), SubtreeAvailability>>>,
    bg_ctx: Context,
) -> Task<TileLoadResult> {
    accessor
        .get(&subtree_url, &headers, RequestPriority::HIGH, None)
        .then(
            &bg_ctx,
            move |io_result: Result<_, courtier::FetchError>| {
                let result = (|| -> TileLoadResult {
                    let response = match io_result {
                        Ok(r) => r,
                        Err(_) => return TileLoadResult::failed(),
                    };
                    if response.check_status().is_err() {
                        return TileLoadResult::failed();
                    }
                    let av = match tairu::parse_subtree(
                        response.decompressed_data(),
                        scheme,
                        subtree_levels,
                    ) {
                        Ok(a) => a,
                        Err(_) => return TileLoadResult::failed(),
                    };
                    {
                        let mut subtrees = loaded_subtrees.lock().unwrap();
                        subtrees.insert((level_group, morton), av);
                    }
                    TileLoadResult::retry_later()
                })();
                orkester::resolved(result)
            },
        )
}
