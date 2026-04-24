//! [`TraversalStrategy`] - pluggable tile selection algorithm.
//!
//! The traversal strategy determines *how* the tile tree is walked each frame:
//! which nodes are selected for rendering, which need loading, and in what
//! order.  The engine ships two built-in strategies:
//!
//! - [`DefaultTraversalStrategy`] - standard two-phase DFS.  Always selects the
//!   best available ancestor when refined children are not yet loaded.
//! - [`SkipLodTraversalStrategy`] - skip-LOD DFS.  May skip intermediate LOD
//!   levels, loading fine detail directly while coarser tiles are still
//!   loading.
//!
//! # Extension
//!
//! Implement [`TraversalStrategy`] and set it via [`SelectionContext::strategy`]
//! to add a new selection mode - for example a *most-detailed* mode for picking
//! or flight-destination preloading - without modifying the existing traversal
//! code.
//!
//! [`SelectionContext`]: crate::traversal::SelectionContext

use crate::frame_decision::{ExpandResult, SelectionOutput};
use crate::selection_state::TileStates;
use crate::tile_store::TileId;
use crate::traversal::{
    FrameLocals, SelectionContext, TraversalBuffers, compute_default_dynamic_sse_densities,
    compute_default_fog_densities, select_base_inner, select_skip_inner,
};

/// Plug-in interface for the per-frame tile selection algorithm.
///
/// The engine pre-computes frame-level values and then delegates the actual
/// DFS to this trait.  Two hooks - [`fog_densities`] and
/// [`dynamic_sse_densities`] - let custom strategies substitute different
/// atmospheric or LOD-reduction models without re-implementing the DFS.
///
/// [`fog_densities`]: TraversalStrategy::fog_densities
/// [`dynamic_sse_densities`]: TraversalStrategy::dynamic_sse_densities
pub trait TraversalStrategy: Send + Sync + 'static {
    /// Return per-view fog densities for this frame.
    ///
    /// Called by the engine *before* [`execute`] and the result is forwarded
    /// to it unchanged.  The default implementation reads the
    /// `culling.fog_density_table` from `ctx.options` and applies the
    /// WGS-84-height-corrected interpolation.
    ///
    /// Override to substitute a custom atmospheric model (e.g. weather-driven
    /// fog, volumetric haze) without modifying the DFS traversal.
    ///
    /// Return an empty `Vec` to disable fog culling regardless of options.
    ///
    /// [`execute`]: TraversalStrategy::execute
    fn fog_densities(&self, ctx: &SelectionContext<'_>) -> Vec<f64> {
        compute_default_fog_densities(ctx)
    }

    /// Return per-view dynamic SSE reduction densities for this frame.
    ///
    /// Called by the engine *before* [`execute`] and the result is forwarded
    /// to it unchanged.  The default implementation computes a horizon-based
    /// reduction from `ctx.options.lod.dynamic_detail_reduction_density`.
    ///
    /// Override to use a different LOD-reduction formula (e.g. altitude-based,
    /// motion-blur-based) without modifying the DFS traversal.
    ///
    /// Return an empty `Vec` to disable dynamic SSE reduction regardless of
    /// options.
    ///
    /// [`execute`]: TraversalStrategy::execute
    fn dynamic_sse_densities(&self, ctx: &SelectionContext<'_>) -> Vec<f64> {
        compute_default_dynamic_sse_densities(ctx)
    }

    /// Run the DFS selection loop for one frame.
    ///
    /// `fog_densities` and `dynamic_sse_densities` are pre-computed via
    /// [`Self::fog_densities`] and [`Self::dynamic_sse_densities`] before
    /// this call; they are provided read-only so the strategy does not need
    /// to recompute them.
    fn execute(
        &self,
        ctx: &SelectionContext<'_>,
        frame: &FrameLocals,
        state: &mut TileStates,
        buffers: &mut TraversalBuffers,
        expand: &mut dyn FnMut(TileId) -> ExpandResult,
        fog_densities: &[f64],
        dynamic_sse_densities: &[f64],
    ) -> SelectionOutput;
}

/// Standard two-phase base traversal.
///
/// Equivalent to CesiumJS / cesium-native's default (non-skip-LOD) selection.
/// Always falls back to the best ready ancestor when children are not
/// yet loaded, preventing holes in the render set.
pub struct DefaultTraversalStrategy;

impl TraversalStrategy for DefaultTraversalStrategy {
    fn execute(
        &self,
        ctx: &SelectionContext<'_>,
        frame: &FrameLocals,
        state: &mut TileStates,
        buffers: &mut TraversalBuffers,
        expand: &mut dyn FnMut(TileId) -> ExpandResult,
        fog_densities: &[f64],
        dynamic_sse_densities: &[f64],
    ) -> SelectionOutput {
        select_base_inner(
            ctx,
            frame,
            state,
            buffers,
            expand,
            fog_densities,
            dynamic_sse_densities,
        )
    }
}

/// Skip-LOD two-phase traversal.
///
/// Equivalent to CesiumJS / cesium-native's `Cesium3DTilesetSkipTraversal`.
/// Skips intermediate LOD levels so fine-detail tiles can start loading
/// before their coarser ancestors have arrived.
pub struct SkipLodTraversalStrategy;

impl TraversalStrategy for SkipLodTraversalStrategy {
    fn execute(
        &self,
        ctx: &SelectionContext<'_>,
        frame: &FrameLocals,
        state: &mut TileStates,
        buffers: &mut TraversalBuffers,
        expand: &mut dyn FnMut(TileId) -> ExpandResult,
        fog_densities: &[f64],
        dynamic_sse_densities: &[f64],
    ) -> SelectionOutput {
        select_skip_inner(
            ctx,
            frame,
            state,
            buffers,
            expand,
            fog_densities,
            dynamic_sse_densities,
        )
    }
}
