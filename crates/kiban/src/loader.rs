//! [`ContentLoader`] - the central loader trait for `kiban`.
//!
//! This is the Rust equivalent of cesium-native's `TilesetContentLoader`.
//! Every tile source (3D Tiles JSON, implicit quadtree/octree, terrain,
//! ellipsoid, Cesium ion) implements this single trait.  `ContentManager`
//! calls it and treats all sources uniformly.
//!
//! # Invariant
//!
//! **All loader implementations return [`moderu::GltfModel`] as the terminal
//! renderable type** - the same invariant as cesium-native's
//! `CesiumGltf::Model`.  Empty tiles and external tileset references are
//! handled via the other [`TileContentKind`] variants.

use std::sync::Arc;

use courtier::AssetAccessor;
use glam::DMat4;
use orkester::Task;
use terra::Ellipsoid;

use crate::async_runtime::AsyncRuntime;
use crate::tile_store::{ContentKey, RefinementMode, TileDescriptor, TileId, TileStore};

/// Everything a loader needs to produce a [`TileLoadResult`].
///
/// Fields from the [`TileStore`] are snapshotted at dispatch time so that
/// background tasks never hold a reference into the store.
///
/// Analogous to cesium-native's `TileLoadInput`.
pub struct TileLoadInput {
    /// The tile being loaded.
    pub tile: TileId,
    /// Content keys snapshotted from `TileStore::content_keys(tile)`.
    pub content_keys: Vec<ContentKey>,
    /// World transform snapshotted from `TileStore::world_transform(tile)`.
    pub world_transform: DMat4,
    /// Refinement mode snapshotted from `TileStore::refinement(tile)`.
    pub refinement: RefinementMode,
    /// Network/file accessor.
    pub accessor: Arc<dyn AssetAccessor>,
    /// HTTP headers to forward with every request (e.g. `Authorization`).
    pub headers: Arc<[(String, String)]>,
    /// Async runtime for background work and main-thread finalization.
    pub runtime: AsyncRuntime,
    /// Reference ellipsoid for geospatial maths.
    pub ellipsoid: Arc<Ellipsoid>,
    /// Maximum screen-space error threshold configured on this tileset.
    pub maximum_screen_space_error: f64,
}

/// How a tile load resolved.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileLoadResultState {
    /// Load succeeded; all fields in [`TileLoadResult`] apply.
    Success,
    /// Load failed; no fields apply.  The tile will be marked failed.
    Failed,
    /// Background work is still in progress; the loader asks to be retried
    /// next frame (e.g. waiting for a subtree file).
    RetryLater,
}

/// The content a tile load produced.
pub enum TileContentKind {
    /// No geometry at this tile level (empty tile).
    Empty,
    /// This tile pointed to an external `tileset.json`; the caller should
    /// insert `root_descriptor` as children of the current tile and register
    /// any `child_loaders` with the `ContentManager`.
    External {
        root_descriptor: TileDescriptor,
        child_loaders: Vec<Arc<dyn ContentLoader>>,
    },
    /// A renderable glTF model.  This is the primary terminal content type.
    Gltf(moderu::GltfModel),
    /// Escape hatch for non-glTF content (point clouds, voxels, proprietary
    /// meshes, etc.).
    ///
    /// kiban emits a [`TileEvent::CustomTileLoaded`] event for any tile whose
    /// load produces this variant.  Downcast the inner `Arc<dyn Any + Send +
    /// Sync>` to your concrete content type in your event handler.
    ///
    /// The tile is *not* automatically marked ready by kiban - call
    /// [`Layer::mark_tile_ready`] yourself once you have prepared the content
    /// for rendering, just as you would for `Renderable` tiles.
    ///
    /// [`TileEvent::CustomTileLoaded`]: crate::TileEvent::CustomTileLoaded
    /// [`Layer::mark_tile_ready`]: crate::Layer::mark_tile_ready
    Custom(std::sync::Arc<dyn std::any::Any + Send + Sync>),
}

/// Return value of [`ContentLoader::load_tile`].
///
/// Analogous to cesium-native's `TileLoadResult`.
pub struct TileLoadResult {
    pub content: TileContentKind,
    pub state: TileLoadResultState,
    /// Up-axis convention for glTF content in this tile.
    /// Mirrors `TileLoadResult::glTFUpAxis` in cesium-native.
    /// Defaults to `Y` (legacy glTF convention).
    pub gltf_up_axis: zukei::Axis,
    /// Updated bounding volume if the loader computed a tighter one from the
    /// content (e.g. from the glTF mesh AABB).
    pub updated_bounds: Option<zukei::SpatialBounds>,
    /// Details needed to drape raster overlays onto this tile.
    pub raster_overlay_details: Option<RasterOverlayDetails>,
    /// Optional callback executed on the **main thread** after the content
    /// pipeline completes, used to mutate `TileStore` state that is only
    /// safe to touch from the main thread (e.g. update transforms discovered
    /// in the tile content).
    pub tile_initializer: Option<Box<dyn FnOnce(&mut TileStore, TileId) + Send>>,
    /// Source URL of the tile resource, for error reporting.
    ///
    /// Set by loaders that know the URL at load time; `None` for procedural
    /// tiles.  Threaded into [`TileLoadErrorDetails::url`] on failure.
    pub source_url: Option<String>,
}

impl TileLoadResult {
    pub fn empty() -> Self {
        Self {
            content: TileContentKind::Empty,
            state: TileLoadResultState::Success,
            gltf_up_axis: zukei::Axis::Y,
            updated_bounds: None,
            raster_overlay_details: None,
            tile_initializer: None,
            source_url: None,
        }
    }

    pub fn failed() -> Self {
        Self {
            content: TileContentKind::Empty,
            state: TileLoadResultState::Failed,
            gltf_up_axis: zukei::Axis::Y,
            updated_bounds: None,
            raster_overlay_details: None,
            tile_initializer: None,
            source_url: None,
        }
    }

    pub fn retry_later() -> Self {
        Self {
            content: TileContentKind::Empty,
            state: TileLoadResultState::RetryLater,
            gltf_up_axis: zukei::Axis::Y,
            updated_bounds: None,
            raster_overlay_details: None,
            tile_initializer: None,
            source_url: None,
        }
    }

    pub fn gltf(model: moderu::GltfModel) -> Self {
        Self {
            content: TileContentKind::Gltf(model),
            state: TileLoadResultState::Success,
            gltf_up_axis: zukei::Axis::Y,
            updated_bounds: None,
            raster_overlay_details: None,
            tile_initializer: None,
            source_url: None,
        }
    }

    pub fn external(
        root_descriptor: TileDescriptor,
        child_loaders: Vec<Arc<dyn ContentLoader>>,
    ) -> Self {
        Self {
            content: TileContentKind::External {
                root_descriptor,
                child_loaders,
            },
            state: TileLoadResultState::Success,
            gltf_up_axis: zukei::Axis::Y,
            updated_bounds: None,
            raster_overlay_details: None,
            tile_initializer: None,
            source_url: None,
        }
    }
}

/// UV-generation metadata produced alongside renderable content so that raster
/// overlays can be draped onto the tile without re-loading.
#[derive(Clone, Debug)]
pub struct RasterOverlayDetails {
    /// Geographic rectangle covered by this tile's content.
    pub rectangle: terra::GlobeRectangle,
    /// Which UV channel the loader wrote overlay coordinates into.
    pub uv_channel: u32,
}

/// Return value of [`ContentLoader::create_children`].
pub enum TileChildrenResult {
    /// Children are ready; insert them into the tile tree.
    Children(Vec<TileDescriptor>),
    /// Loader needs more time (e.g. subtree fetch in flight).  Try again next
    /// frame.
    RetryLater,
    /// This tile has no children.
    None,
}

/// Optional fast height-query shortcut (e.g. analytical ellipsoid terrain).
///
/// Equivalent to cesium-native's `ITilesetHeightSampler`.
pub trait HeightSampler: Send + Sync {
    fn sample_height(&self, longitude: f64, latitude: f64, ellipsoid: &Ellipsoid) -> Option<f64>;
}

/// The central abstraction for fetching tile content.
///
/// Equivalent to cesium-native's `TilesetContentLoader`.
///
/// # Design
///
/// - **`dyn ContentLoader`** - loaders are stored as trait objects so the
///   `CesiumIonLoader` decorator can swap its inner loader at runtime once it
///   discovers the asset type (same pattern as cesium-native's
///   `_pAggregatedLoader`).
/// - **Returns `GltfModel`** - every concrete loader converts its native
///   format (B3DM, quantized mesh, I3S, …) into a `moderu::GltfModel` before
///   returning.  `ContentManager` is format-blind above this boundary.
pub trait ContentLoader: Send + Sync {
    /// Fetch and decode content for `input.tile`.
    ///
    /// May run on a background thread.  The returned `Task` resolves on the
    /// thread pool; any main-thread work (GPU upload) is posted to
    /// `input.runtime.main()`.
    fn load_tile(&self, input: TileLoadInput) -> Task<TileLoadResult>;

    /// Expand latent children for a tile.
    ///
    /// Called when the selection algorithm finds a tile with
    /// `MIGHT_HAVE_LATENT_CHILDREN` set and no current children.  For implicit
    /// tilesets this triggers a subtree fetch; for explicit tilesets it is
    /// never called (all children are present from the start).
    fn create_children(
        &self,
        tile: TileId,
        store: &TileStore,
        ellipsoid: &Ellipsoid,
    ) -> TileChildrenResult;

    /// Optional fast height sampler (analytical terrain loaders provide this).
    fn height_sampler(&self) -> Option<&dyn HeightSampler> {
        None
    }
}

/// Plugin hook that can prevent individual tiles from being refined or rendered.
///
/// Equivalent to cesium-native's `ITileExcluder`.
///
/// When any excluder returns `true` from [`should_exclude`], the traversal
/// renders the tile at its current level of detail and **does not recurse into
/// its children**.  This is useful for features such as bounding-box region
/// culling, layer masking, or debug overrides.
///
/// # Thread safety
///
/// Implementations must be `Send + Sync` because `kiban` may call them from a
/// background thread during traversal.
pub trait TileExcluder: Send + Sync {
    /// Called once at the start of each [`select`] invocation (once per
    /// view-group update).  Use this to cache per-frame state if needed.
    ///
    /// Mirrors `ITileExcluder::startNewFrame`.
    fn start_new_frame(&self) {}

    /// Return `true` to exclude `tile` from refinement this frame.
    ///
    /// When `true` the tile is treated as a leaf: it will be placed in the
    /// render list at its current detail level and its children will not be
    /// visited.
    ///
    /// Mirrors `ITileExcluder::shouldExcludeTile`.
    fn should_exclude(&self, tile: TileId, store: &TileStore) -> bool;
}
