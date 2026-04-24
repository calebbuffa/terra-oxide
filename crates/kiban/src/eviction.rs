//! Pluggable tile eviction policy.
//!
//! [`EvictionPolicy`] determines whether a loaded tile's content is stale and
//! should be re-fetched.  It is called once per visited tile during traversal,
//! before the load-state is inspected.
//!
//! The default implementation is [`MaxAgeEvictionPolicy`], which marks a tile
//! as expiring when its `content_max_age` field (set by the loader) has elapsed.

use crate::selection_state::{TileLoadState, TileStatus};
use crate::tile_store::{TileId, TileStore};

/// Determines whether a loaded tile should be marked as expiring and re-fetched.
///
/// Implement this trait to add custom cache-invalidation logic without modifying
/// the traversal.  Examples: version-tag comparison, ETag-based invalidation,
/// server-push expiry signals.
///
/// # Contract
///
/// - Only called for tiles in the `Renderable` state.
/// - Return `true` to transition the tile to `Expiring` (continues rendering
///   old content while a re-fetch is queued).
/// - Return `false` to leave the tile `Renderable`.
pub trait EvictionPolicy: Send + Sync {
    /// Return `true` if `tile` should be transitioned to `Expiring`.
    fn should_evict(
        &self,
        tile: TileId,
        status: &TileStatus,
        store: &TileStore,
        now_secs: u64,
    ) -> bool;

    /// Adjust the effective screen-space error threshold in response to memory
    /// pressure.
    ///
    /// Called once per frame by [`Layer`] *before* the selection traversal.
    /// The default implementation returns `current_sse` unchanged (no
    /// adjustment).  Override to apply custom memory-pressure logic - e.g.
    /// the built-in [`BudgetEvictionPolicy`] applies a 2% ramp-up when
    /// resident bytes exceed the configured cache budget.
    ///
    /// # Arguments
    /// - `current_sse` - the `memory_adjusted_sse` value from the previous
    ///   frame (starts equal to `LayerOptions::maximum_screen_space_error`).
    /// - `resident_bytes` - total bytes of currently loaded tile content.
    /// - `budget_bytes` - `LoadingOptions::max_cached_bytes`.
    /// - `overflow_bytes` - `LoadingOptions::maximum_cache_overflow_bytes`.
    ///
    /// # Returns
    /// The new effective SSE for this frame.  The engine clamps it from below
    /// to `LayerOptions::maximum_screen_space_error` after this call.
    ///
    /// [`Layer`]: crate::Layer
    fn adjust_sse(
        &self,
        current_sse: f64,
        _resident_bytes: usize,
        _budget_bytes: usize,
        _overflow_bytes: usize,
    ) -> f64 {
        current_sse
    }
}

/// Evict tiles whose `content_max_age` duration (set by the loader) has elapsed.
///
/// This is the default policy.  Tiles whose descriptor has `content_max_age: None`
/// are never evicted by this policy.
///
/// This policy does **not** apply memory-pressure SSE adjustment.  Wrap it
/// with [`BudgetEvictionPolicy`] (or use [`LayerOptions::default`]) to get
/// both behaviours.
pub struct MaxAgeEvictionPolicy;

impl EvictionPolicy for MaxAgeEvictionPolicy {
    fn should_evict(
        &self,
        tile: TileId,
        status: &TileStatus,
        store: &TileStore,
        now_secs: u64,
    ) -> bool {
        if status.load_state != TileLoadState::Ready {
            return false;
        }
        if let Some(max_age) = store.content_max_age(tile) {
            let loaded_secs = status.content_loaded_secs;
            loaded_secs > 0 && now_secs > loaded_secs.saturating_add(max_age.as_secs())
        } else {
            false
        }
    }
}

/// Eviction policy that never evicts content.
///
/// Use when content is static and `content_max_age` should be ignored.
pub struct NeverEvictPolicy;

impl EvictionPolicy for NeverEvictPolicy {
    fn should_evict(
        &self,
        _node: TileId,
        _status: &TileStatus,
        _store: &TileStore,
        _now_secs: u64,
    ) -> bool {
        false
    }
}

/// Eviction policy that wraps another policy and adds memory-pressure SSE
/// adjustment.
///
/// When resident content bytes exceed `budget + overflow`, the effective SSE
/// is multiplied by `ramp_factor` (default `1.02`) each frame, forcing the
/// traversal to request coarser tiles and reduce memory pressure.  When bytes
/// fall back below `budget`, the SSE is divided by `ramp_factor` each frame
/// until it reaches the configured minimum (`LayerOptions::maximum_screen_space_error`).
///
/// This is the behaviour from CesiumJS `Cesium3DTileset._memoryAdjustedScreenSpaceError`.
///
/// # Default
///
/// [`LayerOptions::default`] sets the eviction policy to
/// `BudgetEvictionPolicy { inner: MaxAgeEvictionPolicy, ramp_factor: 1.02 }`.
pub struct BudgetEvictionPolicy<E: EvictionPolicy> {
    /// The underlying content-expiry policy.
    pub inner: E,
    /// Multiplicative ramp applied per-frame while over budget.
    /// Must be `> 1.0`.  Defaults to `1.02` (2% per frame).
    pub ramp_factor: f64,
}

impl<E: EvictionPolicy> BudgetEvictionPolicy<E> {
    /// Wrap `inner` with a 2%-per-frame memory-pressure ramp.
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            ramp_factor: 1.02,
        }
    }
}

impl<E: EvictionPolicy> EvictionPolicy for BudgetEvictionPolicy<E> {
    fn should_evict(
        &self,
        tile: TileId,
        status: &TileStatus,
        store: &TileStore,
        now_secs: u64,
    ) -> bool {
        self.inner.should_evict(tile, status, store, now_secs)
    }

    fn adjust_sse(
        &self,
        current_sse: f64,
        resident_bytes: usize,
        budget_bytes: usize,
        overflow_bytes: usize,
    ) -> f64 {
        if resident_bytes > budget_bytes.saturating_add(overflow_bytes) {
            current_sse * self.ramp_factor
        } else if resident_bytes < budget_bytes {
            current_sse / self.ramp_factor
        } else {
            current_sse
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selection_state::{TileLoadState, TileStatus};
    use crate::tile_store::{TileDescriptor, TileId};
    use std::time::Duration;

    fn make_status(state: TileLoadState, content_loaded_secs: u64) -> TileStatus {
        TileStatus {
            load_state: state,
            content_loaded_secs,
            ..TileStatus::DEFAULT
        }
    }

    fn make_store_with_max_age(max_age: Option<Duration>) -> (TileStore, TileId) {
        let desc = TileDescriptor {
            content_max_age: max_age,
            ..TileDescriptor::interior(zukei::SpatialBounds::Empty, 1.0, Vec::new())
        };
        let store = TileStore::from_descriptor(desc);
        let root = store.root();
        (store, root)
    }

    #[test]
    fn max_age_not_renderable_is_not_evicted() {
        let (store, root) = make_store_with_max_age(Some(Duration::from_secs(10)));
        let status = make_status(TileLoadState::Loading, 100);
        assert!(!MaxAgeEvictionPolicy.should_evict(root, &status, &store, 200));
    }

    #[test]
    fn max_age_no_max_age_is_not_evicted() {
        let (store, root) = make_store_with_max_age(None);
        let status = make_status(TileLoadState::Ready, 100);
        assert!(!MaxAgeEvictionPolicy.should_evict(root, &status, &store, 200));
    }

    #[test]
    fn max_age_within_age_is_not_evicted() {
        let (store, root) = make_store_with_max_age(Some(Duration::from_secs(60)));
        // Loaded at t=100, now at t=150 - still within the 60-second window.
        let status = make_status(TileLoadState::Ready, 100);
        assert!(!MaxAgeEvictionPolicy.should_evict(root, &status, &store, 150));
    }

    #[test]
    fn max_age_exceeded_is_evicted() {
        let (store, root) = make_store_with_max_age(Some(Duration::from_secs(60)));
        // Loaded at t=100, now at t=161 - 61 s elapsed > 60 s max age.
        let status = make_status(TileLoadState::Ready, 100);
        assert!(MaxAgeEvictionPolicy.should_evict(root, &status, &store, 161));
    }

    #[test]
    fn never_evict_always_returns_false() {
        let (store, root) = make_store_with_max_age(Some(Duration::from_secs(0)));
        let status = make_status(TileLoadState::Ready, 0);
        assert!(!NeverEvictPolicy.should_evict(root, &status, &store, 9999));
    }
}
