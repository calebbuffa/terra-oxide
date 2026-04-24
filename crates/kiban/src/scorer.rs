//! [`LoadPriorityScorer`] - pluggable load priority scoring.
//!
//! After each DFS traversal pass, the engine has collected a list of
//! [`LoadRequest`]s, each carrying raw metrics (distance, foveated factor,
//! depth, reverse-SSE).  The scorer's job is to map those raw values into a
//! single `f32` `priority.score` that the [`LoadScheduler`] uses to order
//! dispatch.
//!
//! # Extension
//!
//! Implement [`LoadPriorityScorer`] and pass your implementation to
//! [`SelectionContext`] to swap the scoring algorithm without touching the
//! traversal.  The default implementation is [`WeightedComponentScorer`].
//!
//! [`LoadScheduler`]: crate::load_scheduler::LoadScheduler
//! [`SelectionContext`]: crate::traversal::SelectionContext

use crate::frame_decision::LoadRequest;

/// Plug-in interface for computing final `priority.score` values on a
/// collection of [`LoadRequest`]s after a traversal pass.
///
/// The scorer receives the full mutable slice so it can inspect global
/// statistics (min/max ranges) before writing per-request scores.
pub trait LoadPriorityScorer: Send + Sync + 'static {
    /// Assign `priority.score` to every request in `loads`.
    ///
    /// The scorer is free to read *and* mutate the slice.  It must set
    /// `req.priority.score` on every entry; leftover entries will keep
    /// whatever value they had before this call.
    fn score(&self, loads: &mut Vec<LoadRequest>);
}

/// Default scorer: 4-component weighted composite.
///
/// Each raw metric is min-max normalised across the current frame's load
/// list, then blended with equal `0.25` weights:
///
/// ```text
/// score = 0.25 * norm_depth   (higher = deeper leaf, when prefer_leaves)
///       + 0.25 * norm_distance (higher = closer)
///       + 0.25 * norm_foveated (higher = more central)
///       + 0.25 * norm_rsse    (higher = larger reverse-SSE, i.e. more urgent)
/// ```
///
/// Matches the scoring used in CesiumJS `Cesium3DTileset._requestTiles`.
pub struct WeightedComponentScorer {
    /// When `true`, deeper (leaf) tiles score higher on the depth component.
    /// When `false`, shallower tiles score higher (prefer roots).
    pub prefer_leaves: bool,
}

impl Default for WeightedComponentScorer {
    fn default() -> Self {
        Self {
            prefer_leaves: false,
        }
    }
}

impl LoadPriorityScorer for WeightedComponentScorer {
    fn score(&self, loads: &mut Vec<LoadRequest>) {
        if loads.is_empty() {
            return;
        }

        let mut min_dist = f64::MAX;
        let mut max_dist = f64::MIN;
        let mut min_ff = f64::MAX;
        let mut max_ff = f64::MIN;
        let mut min_d = u32::MAX;
        let mut max_d = 0u32;
        let mut min_r = f64::MAX;
        let mut max_r = f64::MIN;

        for req in loads.iter() {
            if req.raw_distance < min_dist {
                min_dist = req.raw_distance;
            }
            if req.raw_distance > max_dist {
                max_dist = req.raw_distance;
            }
            if req.raw_foveated_factor < min_ff {
                min_ff = req.raw_foveated_factor;
            }
            if req.raw_foveated_factor > max_ff {
                max_ff = req.raw_foveated_factor;
            }
            if req.raw_depth < min_d {
                min_d = req.raw_depth;
            }
            if req.raw_depth > max_d {
                max_d = req.raw_depth;
            }
            if req.raw_reverse_sse < min_r {
                min_r = req.raw_reverse_sse;
            }
            if req.raw_reverse_sse > max_r {
                max_r = req.raw_reverse_sse;
            }
        }

        let rng_dist = (max_dist - min_dist).max(1e-10);
        let rng_ff = (max_ff - min_ff).max(1e-10);
        let rng_depth = (max_d as f64 - min_d as f64).max(1.0);
        let rng_rsse = (max_r - min_r).max(1e-10);

        for req in loads.iter_mut() {
            let n_dist = 1.0 - (req.raw_distance - min_dist) / rng_dist;
            let n_ff = 1.0 - (req.raw_foveated_factor - min_ff) / rng_ff;
            let n_d_raw = (req.raw_depth as f64 - min_d as f64) / rng_depth;
            let n_depth = if self.prefer_leaves {
                n_d_raw
            } else {
                1.0 - n_d_raw
            };
            let n_rsse = 1.0 - (req.raw_reverse_sse - min_r) / rng_rsse;
            req.priority.score =
                (0.25 * n_depth + 0.25 * n_dist + 0.25 * n_ff + 0.25 * n_rsse) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_decision::{LoadPriority, PriorityGroup};
    use crate::tile_store::TileId;

    fn make_request(distance: f64, ff: f64, depth: u32, rsse: f64) -> LoadRequest {
        LoadRequest {
            tile: TileId::from_slot(0),
            priority: LoadPriority {
                group: PriorityGroup::NORMAL,
                score: 0.0,
            },
            raw_distance: distance,
            raw_foveated_factor: ff,
            raw_depth: depth,
            raw_reverse_sse: rsse,
            bounding_radius: 1.0,
        }
    }

    #[test]
    fn empty_list_is_noop() {
        let scorer = WeightedComponentScorer::default();
        let mut loads = vec![];
        scorer.score(&mut loads);
        assert!(loads.is_empty());
    }

    #[test]
    fn single_request_gets_score_one() {
        let scorer = WeightedComponentScorer::default();
        let mut loads = vec![make_request(100.0, 0.5, 3, 0.1)];
        scorer.score(&mut loads);
        // With a single request all ranges collapse to zero -> score should be 1.0
        // (each n_x = 1 - 0/eps = 1, except n_d_raw = 0/1 = 0 -> n_depth = 1 with prefer_leaves=false)
        assert!(
            (loads[0].priority.score - 1.0).abs() < 1e-4,
            "score={}",
            loads[0].priority.score
        );
    }

    #[test]
    fn closer_tile_scores_higher_distance() {
        let scorer = WeightedComponentScorer::default();
        let mut loads = vec![
            make_request(10.0, 0.0, 1, 0.0),  // close
            make_request(100.0, 0.0, 1, 0.0), // far
        ];
        scorer.score(&mut loads);
        // Close tile should score higher (n_dist larger for smaller distance)
        assert!(loads[0].priority.score > loads[1].priority.score);
    }

    #[test]
    fn prefer_leaves_inverts_depth_component() {
        let scorer_leaves = WeightedComponentScorer {
            prefer_leaves: true,
        };
        let scorer_parents = WeightedComponentScorer {
            prefer_leaves: false,
        };
        let mut loads_l = vec![
            make_request(0.0, 0.0, 1, 0.0),
            make_request(0.0, 0.0, 5, 0.0),
        ];
        let mut loads_p = loads_l.clone();
        scorer_leaves.score(&mut loads_l);
        scorer_parents.score(&mut loads_p);
        // prefer_leaves: depth=5 should score higher
        assert!(loads_l[1].priority.score > loads_l[0].priority.score);
        // prefer_parents: depth=1 should score higher
        assert!(loads_p[0].priority.score > loads_p[1].priority.score);
    }
}
