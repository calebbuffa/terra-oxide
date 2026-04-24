//! [`DispatchGate`] - pluggable load-dispatch gating.
//!
//! Before the [`LoadScheduler`] dispatches a load request it passes the
//! request through a chain of gates.  Each gate can veto a request by
//! returning `false` from [`DispatchGate::should_dispatch`].  If *any* gate
//! vetoes, the request is skipped for this frame (it will be retried next
//! frame).
//!
//! # Extension
//!
//! Implement [`DispatchGate`] and register it with [`TileOptions::gates`] /
//! [`TilesetOptions::gates`].  The engine always prepends the two built-in
//! gates ([`FoveatedTimeDelayGate`] and [`CullWhileMovingGate`]) before any
//! user-provided gates.
//!
//! [`LoadScheduler`]: crate::load_scheduler::LoadScheduler

use crate::frame_decision::{LoadRequest, PriorityGroup};

/// Context passed to each [`DispatchGate`] during load dispatch.
pub struct DispatchContext {
    /// Seconds the primary camera has been stationary.
    pub time_since_camera_stopped_secs: f64,
    /// World-space displacement magnitude of the primary camera this frame.
    pub camera_delta_magnitude: f64,
}

/// Plug-in interface for gating individual load requests before dispatch.
///
/// Implement this trait to veto specific requests per-frame.  All built-in
/// gates and user-provided gates are evaluated in order; the first `false`
/// skips the request.
pub trait DispatchGate: Send + Sync + 'static {
    /// Return `true` if `req` should be dispatched this frame, `false` to
    /// skip it (it will be retried next frame).
    fn should_dispatch(&self, req: &LoadRequest, ctx: &DispatchContext) -> bool;
}

/// Gate that defers `Deferred`-priority requests until the camera has been
/// stationary for at least `time_delay` seconds.
///
/// Matches CesiumJS `foveatedTimeDelay` behaviour.
pub struct FoveatedTimeDelayGate {
    /// Minimum seconds of camera stillness before `Deferred` requests fire.
    pub time_delay: f64,
}

impl DispatchGate for FoveatedTimeDelayGate {
    fn should_dispatch(&self, req: &LoadRequest, ctx: &DispatchContext) -> bool {
        if req.priority.group == PriorityGroup::DEFERRED {
            ctx.time_since_camera_stopped_secs >= self.time_delay
        } else {
            true
        }
    }
}

/// Gate that suppresses non-`Urgent` requests when the camera is moving fast
/// relative to a tile's bounding radius.
///
/// Matches CesiumJS `cullRequestsWhileMoving` behaviour.
pub struct CullWhileMovingGate {
    /// Multiplier applied to the per-frame camera displacement.  Requests are
    /// suppressed when `multiplier * delta / radius >= 1.0`.
    pub multiplier: f64,
}

impl DispatchGate for CullWhileMovingGate {
    fn should_dispatch(&self, req: &LoadRequest, ctx: &DispatchContext) -> bool {
        if req.priority.group == PriorityGroup::URGENT || ctx.camera_delta_magnitude == 0.0 {
            return true;
        }
        let radius = req.bounding_radius.max(1.0);
        self.multiplier * ctx.camera_delta_magnitude / radius < 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_decision::{LoadPriority, PriorityGroup};
    use crate::tile_store::TileId;

    fn req(group: PriorityGroup, radius: f64) -> LoadRequest {
        LoadRequest {
            tile: TileId::from_slot(0),
            priority: LoadPriority { group, score: 0.0 },
            raw_distance: 0.0,
            raw_foveated_factor: 0.0,
            raw_depth: 0,
            raw_reverse_sse: 0.0,
            bounding_radius: radius,
        }
    }

    fn ctx(stopped_secs: f64, delta: f64) -> DispatchContext {
        DispatchContext {
            time_since_camera_stopped_secs: stopped_secs,
            camera_delta_magnitude: delta,
        }
    }

    #[test]
    fn foveated_gate_passes_non_deferred_regardless_of_time() {
        let gate = FoveatedTimeDelayGate { time_delay: 10.0 };
        let c = ctx(0.0, 0.0);
        assert!(gate.should_dispatch(&req(PriorityGroup::NORMAL, 1.0), &c));
        assert!(gate.should_dispatch(&req(PriorityGroup::URGENT, 1.0), &c));
        assert!(gate.should_dispatch(&req(PriorityGroup::PRELOAD, 1.0), &c));
    }

    #[test]
    fn foveated_gate_blocks_deferred_before_delay() {
        let gate = FoveatedTimeDelayGate { time_delay: 0.5 };
        let c = ctx(0.1, 0.0); // 0.1 s < 0.5 s
        assert!(!gate.should_dispatch(&req(PriorityGroup::DEFERRED, 1.0), &c));
    }

    #[test]
    fn foveated_gate_passes_deferred_after_delay() {
        let gate = FoveatedTimeDelayGate { time_delay: 0.2 };
        let c = ctx(1.0, 0.0); // 1.0 s > 0.2 s
        assert!(gate.should_dispatch(&req(PriorityGroup::DEFERRED, 1.0), &c));
    }

    #[test]
    fn cull_gate_always_passes_urgent() {
        let gate = CullWhileMovingGate { multiplier: 1000.0 };
        let c = ctx(0.0, 9999.0);
        assert!(gate.should_dispatch(&req(PriorityGroup::URGENT, 1.0), &c));
    }

    #[test]
    fn cull_gate_passes_when_camera_still() {
        let gate = CullWhileMovingGate { multiplier: 60.0 };
        let c = ctx(0.0, 0.0);
        assert!(gate.should_dispatch(&req(PriorityGroup::NORMAL, 1.0), &c));
    }

    #[test]
    fn cull_gate_blocks_small_tile_when_moving_fast() {
        // 60 * 100 / 1 = 6000 >= 1 -> blocked
        let gate = CullWhileMovingGate { multiplier: 60.0 };
        let c = ctx(0.0, 100.0);
        assert!(!gate.should_dispatch(&req(PriorityGroup::NORMAL, 1.0), &c));
    }

    #[test]
    fn cull_gate_passes_large_tile_when_moving() {
        // 60 * 1 / 10000 = 0.006 < 1 -> passes
        let gate = CullWhileMovingGate { multiplier: 60.0 };
        let c = ctx(0.0, 1.0);
        assert!(gate.should_dispatch(&req(PriorityGroup::NORMAL, 10000.0), &c));
    }
}
