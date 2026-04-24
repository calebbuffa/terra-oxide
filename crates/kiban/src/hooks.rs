//! [`FrameHook`] - extension points for each tick phase.
//!
//! `kiban`'s `tick()` runs a sequence of phases every frame (drain completed
//! loads -> select tiles -> dispatch new loads -> evict stale tiles -> rebuild
//! render set).  [`FrameHook`] lets you observe or intercept each phase
//! boundary without modifying the engine source.
//!
//! # Use cases
//!
//! - **Metrics / telemetry** - measure per-phase timings, tile counts.
//! - **Debug overlays** - dump the selection output after traversal.
//! - **Adaptive quality** - inspect `ViewUpdateResult` after selection and
//!   adjust `maximum_screen_space_error` for the next frame.
//! - **Replay / serialisation** - snapshot state at deterministic points.
//!
//! # Extension
//!
//! Implement [`FrameHook`] and register it via [`TileOptions::hooks`] or
//! [`TilesetOptions::hooks`].  All hooks run on the same thread as `tick()`.
//! Hook methods have default no-op implementations so you only override what
//! you need.

use crate::frame_decision::{LoadEvent, SelectionOutput};

/// Observer hook called at key points during a `tick()` execution.
///
/// All methods have default no-op implementations.
pub trait FrameHook: Send + Sync + 'static {
    /// Called immediately after completed load results have been drained from
    /// background threads and turned into [`LoadEvent`]s.
    ///
    /// `events` is the slice of events produced this frame (may be empty).
    fn after_drain(&self, _events: &[LoadEvent]) {}

    /// Called immediately after the tile selection traversal completes,
    /// before the output is merged into the [`ViewGroup`].
    ///
    /// `output` contains the raw traversal result including render list,
    /// load requests, and per-tile SSE values.
    ///
    /// [`ViewGroup`]: crate::ViewGroup
    fn after_select(&self, _output: &SelectionOutput) {}

    /// Called immediately after the load queue has been dispatched to
    /// background tasks.  `dispatched_count` is the number of new loads
    /// started this frame.
    fn after_dispatch(&self, _dispatched_count: usize) {}

    /// Called once per `tick()` after all phases have completed.
    fn after_tick(&self) {}
}
