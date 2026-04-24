//! Per-frame camera motion tracking.
//!
//! Isolates the "how fast is the camera moving and how long has it been
//! still" state that several dispatch gates (foveated time-delay,
//! cull-requests-while-moving) depend on. Extracted from [`LoadScheduler`]
//! so the scheduler no longer carries this orthogonal bookkeeping.

use glam::DVec3;

/// Tracks frame-to-frame camera motion.
///
/// Designed so the caller simply hands over the current camera position and
/// frame delta time; the tracker derives displacement magnitude and
/// stationary-duration.
#[derive(Default)]
pub(crate) struct CameraMotionTracker {
    previous_position: Option<DVec3>,
    delta_magnitude: f64,
    time_since_stopped_secs: f64,
}

impl CameraMotionTracker {
    /// Update with the current frame's primary-view camera position and
    /// elapsed frame time.
    ///
    /// `movement_threshold` is the displacement (in world units) below which
    /// the camera is considered stationary for the stopped-time accumulator.
    pub fn update(&mut self, current_position: DVec3, delta_time: f32, movement_threshold: f64) {
        self.delta_magnitude = match self.previous_position {
            Some(prev) => (current_position - prev).length(),
            None => 0.0,
        };
        if self.delta_magnitude < movement_threshold {
            self.time_since_stopped_secs += delta_time as f64;
        } else {
            self.time_since_stopped_secs = 0.0;
        }
        self.previous_position = Some(current_position);
    }

    /// Most recent frame-to-frame displacement magnitude (world units).
    pub fn delta_magnitude(&self) -> f64 {
        self.delta_magnitude
    }

    /// Seconds the camera has been stationary (displacement below threshold).
    pub fn time_since_stopped_secs(&self) -> f64 {
        self.time_since_stopped_secs
    }
}
