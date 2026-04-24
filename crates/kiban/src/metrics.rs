//! Frame performance metrics for observability.

/// Performance metrics from the last [`Layer::tick`] call.
///
/// Use [`Layer::last_frame_metrics`] to retrieve metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameMetrics {
    /// Time spent in tile selection traversal (milliseconds).
    pub selection_time_ms: f64,

    /// Time spent processing load completions (milliseconds).
    pub load_processing_time_ms: f64,

    /// Time spent dispatching new load requests (milliseconds).
    pub load_dispatch_time_ms: f64,

    /// Number of tiles currently in the load queue (pending downloads).
    pub load_queue_size: usize,

    /// Number of tiles loaded in the last frame.
    pub tiles_loaded_this_frame: usize,

    /// Number of tiles evicted in the last frame.
    pub tiles_evicted_this_frame: usize,

    /// Total number of tiles currently in the render set.
    pub selected_tile_count: usize,

    /// Estimated total memory usage of selected tiles (bytes).
    /// Note: This is an estimate; actual memory depends on your ContentLoader implementation.
    pub estimated_memory_bytes: u64,

    /// Number of tiles visited by the traversal this frame (includes culled tiles).
    pub tiles_visited: usize,

    /// Number of tiles frustum- or fog-culled during traversal this frame.
    pub tiles_culled: usize,

    /// Number of tiles that have permanently failed to load.
    pub tiles_failed: usize,

    /// Number of tiles that loaded as geometry-free (empty) this frame.
    pub tiles_empty: usize,

    /// Number of tiles currently in-flight (network request pending).
    pub tiles_in_flight: usize,

    /// Number of tiles that were refined (children pushed onto the work queue)
    /// during traversal this frame.
    pub tiles_refined: usize,
}

impl FrameMetrics {
    /// Total time spent in tile selection and loading this frame (milliseconds).
    pub fn total_time_ms(&self) -> f64 {
        self.selection_time_ms + self.load_processing_time_ms + self.load_dispatch_time_ms
    }
}
