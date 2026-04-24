//! Per-tile selection state tracked across frames.
//!
//! Private to `kiban` - this is the selection engine's bookkeeping state.
//! It is indexed by [`NodeIndex::slot()`] for O(1) access.

use crate::frame_decision::LoadEvent;
use crate::tile_store::TileId;

/// Node lifecycle state machine.
///
/// ```text
/// Unloaded -> Queued -> Loading -> Renderable
///                │        │
///                └────────┴── RetryScheduled -> Queued
///
/// Renderable -> Expiring       (content exceeded content_max_age; still renders)
/// Expiring   -> Loading        (re-fetch dispatched)
/// Expiring   -> Renderable     (new content arrived)
///
/// Any state -> Failed   (permanent)
/// Any state -> Evicted  (memory pressure)
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TileLoadState {
    Unloaded,
    Loading,
    Ready,
    /// Content is loaded and renderable but has exceeded its max age.
    /// The tile continues to render its old content while a re-fetch is queued.
    /// Mirrors CesiumJS `tile.contentExpired` + `_expiredContent` fallback.
    Expiring,
    RetryScheduled,
    Failed,
    Evicted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileRefinementResult {
    None,
    /// Tile was frustum-culled and skipped this frame (shouldVisit == false).
    /// Mirrors `TileSelectionState::Result::Culled`.
    Culled,
    Selected,
    Refined,
    /// Tile was selected for rendering then later removed from the render list
    /// because its ancestor became the fallback (some sibling subtrees not
    /// yet renderable).  Equivalent to `TileSelectionState::Result::RenderedAndKicked`.
    SelectedAndKicked,
    /// Tile was being refined (children in render list) but the whole subtree
    /// was kicked in favour of an ancestor.  Equivalent to
    /// `TileSelectionState::Result::RefinedAndKicked`.
    RefinedAndKicked,
}

impl TileRefinementResult {
    /// Transition to the kicked variant.  Mirrors `TileSelectionState::kick()`.
    #[inline]
    pub fn kick(self) -> Self {
        match self {
            Self::Selected => Self::SelectedAndKicked,
            Self::Refined => Self::RefinedAndKicked,
            // None and Culled have no kicked variant - pass through unchanged.
            other => other,
        }
    }

    /// The result prior to kicking.  Mirrors `TileSelectionState::getOriginalResult()`.
    #[inline]
    pub fn original(self) -> Self {
        match self {
            Self::SelectedAndKicked => Self::Selected,
            Self::RefinedAndKicked => Self::Refined,
            other => other,
        }
    }
}

/// Per-tile tracking state.
#[derive(Clone, Debug)]
pub struct TileStatus {
    pub load_state: TileLoadState,
    pub retry_count: u8,
    /// Frame on which to attempt the next retry (backoff).
    pub next_retry_frame: u64,
    pub last_result: TileRefinementResult,
    /// Importance score from the last traversal (higher = keep longer).
    pub importance: f32,
    /// Wall-clock milliseconds (since UNIX epoch) when this tile first entered
    /// the `fading_in` list.  `0` means it is not currently fading in.
    /// Used to compute [`SelectionState::fade_percentage`].
    pub fade_in_ms: u64,
    /// Wall-clock seconds (since Unix epoch) when this tile was last
    /// visited as Rendered or Refined.  `0` = never touched.
    /// Used by eviction: nodes untouched for longer than the grace
    /// period are candidates for unloading.
    pub last_touched_secs: u64,
    /// Current LOD fade-in percentage in [0.0, 1.0].
    /// Updated by ContentManager after each frame's _updateLodTransitions step.
    /// Mirrors cesium-native's `TileRenderContent::getLodTransitionFadePercentage()`.
    /// Used in traversal to stop kicking children once the parent is fully visible.
    pub lod_fade_pct: f32,
    /// Wall-clock seconds when content last became `Renderable`.
    /// Used to detect when `content_max_age` has been exceeded.
    pub content_loaded_secs: u64,
    /// Nearest ancestor that has content or has the potential to have content.
    /// Updated top-down each frame in execute_traversal_skip.
    /// Mirrors CesiumJS `tile._ancestorWithContent`.
    pub ancestor_with_content: Option<crate::tile_store::TileId>,
    /// Nearest ancestor whose content is currently available (Renderable).
    /// Used as the fallback when a desired skip-LOD tile is not yet loaded.
    /// Mirrors CesiumJS `tile._ancestorWithContentAvailable`.
    pub ancestor_with_content_available: Option<crate::tile_store::TileId>,
    /// SSE of this tile as computed in the most-recent skip traversal visit.
    /// Ancestors are always visited before descendants (DFS), so this value
    /// is fresh when `reachedSkippingThreshold` reads it for any descendant.
    pub traversal_sse: f64,
    /// Depth of this tile in the most-recent skip traversal visit.
    pub traversal_depth: u32,
    /// Approximate byte size of this tile's renderer-ready content.
    ///
    /// Set when the content pipeline completes.  Used for byte-budget eviction.
    /// Mirrors the per-tile byte accounting previously held in
    /// `ContentManager::content_byte_sizes`.
    pub content_byte_size: u32,
}

impl TileStatus {
    pub(crate) const DEFAULT: TileStatus = TileStatus {
        load_state: TileLoadState::Unloaded,
        retry_count: 0,
        next_retry_frame: 0,
        last_result: TileRefinementResult::None,
        importance: 0.0,
        fade_in_ms: 0,
        last_touched_secs: 0,
        lod_fade_pct: 0.0,
        content_loaded_secs: 0,
        ancestor_with_content: None,
        ancestor_with_content_available: None,
        traversal_sse: 0.0,
        traversal_depth: 0,
        content_byte_size: 0,
    };
}

/// Dense per-tile selection state, indexed by `NodeIndex::slot()`.
///
/// Grows on demand; O(1) read/write with no hashing.
pub struct TileStates {
    statuses: Vec<TileStatus>,
    pub frame_index: u64,
}

impl TileStates {
    pub fn new() -> Self {
        Self {
            statuses: Vec::new(),
            frame_index: 0,
        }
    }

    pub fn advance_frame(&mut self) {
        self.frame_index += 1;
    }

    /// O(1) read. Returns the static default for nodes not yet seen.
    #[inline(always)]
    pub fn get(&self, tile: TileId) -> &TileStatus {
        self.statuses
            .get(tile.slot())
            .unwrap_or(&TileStatus::DEFAULT)
    }

    /// O(1) write. Grows the backing vec if needed.
    #[inline(always)]
    pub fn get_mut(&mut self, tile: TileId) -> &mut TileStatus {
        let s = tile.slot();
        if s >= self.statuses.len() {
            self.statuses
                .resize_with(s + 1, || TileStatus::DEFAULT.clone());
        }
        &mut self.statuses[s]
    }

    pub fn mark_ready(&mut self, tile: TileId) {
        let now_secs = outil::time_now_secs();
        let s = self.get_mut(tile);
        s.load_state = TileLoadState::Ready;
        s.content_loaded_secs = now_secs;
    }

    pub fn mark_expiring(&mut self, tile: TileId) {
        self.get_mut(tile).load_state = TileLoadState::Expiring;
    }

    pub fn mark_loading(&mut self, tile: TileId) {
        self.get_mut(tile).load_state = TileLoadState::Loading;
    }

    pub fn mark_failed(&mut self, tile: TileId) {
        self.get_mut(tile).load_state = TileLoadState::Failed;
    }

    pub fn mark_evicted(&mut self, tile: TileId) {
        self.get_mut(tile).load_state = TileLoadState::Evicted;
    }

    /// Iterate over all `Renderable` nodes whose `last_touched_secs` is more
    /// than `grace_secs` seconds behind `current_secs`.  These are candidates
    /// for content eviction.  Nodes with `last_touched_secs == 0` (never
    /// visited) are not returned - they were never loaded so there's nothing
    /// to evict.
    pub fn stale(&self, current_secs: u64, grace_secs: u64) -> impl Iterator<Item = TileId> + '_ {
        use crate::tile_store::TileId;
        self.statuses
            .iter()
            .enumerate()
            .filter(move |(_, s)| {
                s.load_state == TileLoadState::Ready
                    && s.last_touched_secs != 0
                    && current_secs.saturating_sub(s.last_touched_secs) > grace_secs
            })
            .map(|(slot, _)| TileId::from_slot(slot as u32))
    }

    /// Returns a 0.0–1.0 fade-in progress for a tile, clamped.
    ///
    /// `current_ms` - current wall-clock time in milliseconds since UNIX epoch.
    /// `transition_length_secs` - length of the fade in seconds (from `StreamingOptions`).
    ///
    /// Returns `0.0` if the tile is not fading in, `1.0` when the transition is complete.
    pub fn fade_percentage(
        &self,
        tile: TileId,
        current_ms: u64,
        transition_length_secs: f32,
    ) -> f32 {
        let status = self.get(tile);
        if status.fade_in_ms == 0 || transition_length_secs <= 0.0 {
            return 0.0;
        }
        let elapsed_ms = current_ms.saturating_sub(status.fade_in_ms);
        let transition_ms = (transition_length_secs * 1000.0) as u64;
        if transition_ms == 0 {
            return 1.0;
        }
        (elapsed_ms as f32 / transition_ms as f32).clamp(0.0, 1.0)
    }

    /// Read-only slice over all tile statuses (for load-progress metrics).
    pub fn statuses(&self) -> &[TileStatus] {
        &self.statuses
    }

    pub fn schedule_retry(&mut self, tile: TileId, retry_frame: u64) {
        let status = self.get_mut(tile);
        status.load_state = TileLoadState::RetryScheduled;
        status.retry_count = status.retry_count.saturating_add(1);
        status.next_retry_frame = retry_frame;
    }

    /// Returns a 0.0–100.0 load-completion percentage.
    pub fn compute_load_progress(&self) -> f32 {
        let mut renderable = 0usize;
        let mut loading = 0usize;
        for status in &self.statuses {
            match status.load_state {
                TileLoadState::Ready => renderable += 1,
                TileLoadState::Loading => loading += 1,
                _ => {}
            }
        }
        let total = renderable + loading;
        if total == 0 {
            return 100.0;
        }
        (renderable as f32 / total as f32) * 100.0
    }

    /// Returns (tiles_loading, tiles_renderable, tiles_failed, resident_bytes).
    ///
    /// Mirrors the statistics exposed by CesiumJS `Cesium3DTilesetStatistics`:
    /// `numberOfPendingRequests` / `numberOfTilesWithContentReady` /
    /// `numberOfAttemptedRequests` / `totalMemoryUsageInBytes`.
    pub fn load_stats(&self) -> (u32, u32, u32, usize) {
        let mut loading = 0u32;
        let mut renderable = 0u32;
        let mut failed = 0u32;
        let mut bytes = 0usize;
        for status in &self.statuses {
            match status.load_state {
                TileLoadState::Loading => loading += 1,
                TileLoadState::Ready | TileLoadState::Expiring => {
                    renderable += 1;
                    bytes += status.content_byte_size as usize;
                }
                TileLoadState::Failed => failed += 1,
                _ => {}
            }
        }
        (loading, renderable, failed, bytes)
    }

    /// Mark all nodes as evicted (used by `Tileset::unload_all`).
    pub fn mark_all_evicted(&mut self) {
        for status in &mut self.statuses {
            status.load_state = TileLoadState::Evicted;
        }
    }

    /// Apply a normalised load event, driving the tile's state-machine transition.
    ///
    /// This is the single authoritative transition function: rather than
    /// scattering `mark_failed` / `schedule_retry` / etc. across the codebase,
    /// callers construct a `LoadEvent` and pass it here.
    ///
    /// `StoreInit` and `Loaded` variants carry no state transition - they are
    /// handled by `ContentCache` and the `Tileset` pipeline step respectively.
    pub(crate) fn apply(&mut self, event: &LoadEvent) {
        match event {
            LoadEvent::Empty { tile } => self.mark_ready(*tile),
            LoadEvent::Failed { tile, .. } => self.mark_failed(*tile),
            LoadEvent::RetryLater { tile } => {
                let retry_count = self.get(*tile).retry_count;
                let backoff = (1u64 << retry_count.min(6)).min(64);
                let retry_frame = self.frame_index + backoff;
                self.schedule_retry(*tile, retry_frame);
            }
            // Loaded / Custom: tile stays Loading until mark_tile_ready() is called.
            LoadEvent::Loaded { .. } | LoadEvent::Custom { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_decision::LoadEvent;
    use crate::tile_store::TileId;

    fn tile(slot: u32) -> TileId {
        TileId::from_slot(slot)
    }

    #[test]
    fn new_node_is_unloaded() {
        let states = TileStates::new();
        assert_eq!(states.get(tile(0)).load_state, TileLoadState::Unloaded);
    }

    #[test]
    fn mark_renderable_transitions_and_records_time() {
        let mut states = TileStates::new();
        states.mark_ready(tile(0));
        let s = states.get(tile(0));
        assert_eq!(s.load_state, TileLoadState::Ready);
        assert!(s.content_loaded_secs > 0);
    }

    #[test]
    fn mark_loading_then_expiring() {
        let mut states = TileStates::new();
        states.mark_loading(tile(1));
        assert_eq!(states.get(tile(1)).load_state, TileLoadState::Loading);
        states.mark_ready(tile(1));
        states.mark_expiring(tile(1));
        assert_eq!(states.get(tile(1)).load_state, TileLoadState::Expiring);
    }

    #[test]
    fn mark_failed_and_evicted() {
        let mut states = TileStates::new();
        states.mark_ready(tile(2));
        states.mark_evicted(tile(2));
        assert_eq!(states.get(tile(2)).load_state, TileLoadState::Evicted);

        states.mark_ready(tile(3));
        states.mark_failed(tile(3));
        assert_eq!(states.get(tile(3)).load_state, TileLoadState::Failed);
    }

    #[test]
    fn stale_renderables_returns_old_nodes() {
        let mut states = TileStates::new();
        states.mark_ready(tile(0));
        states.get_mut(tile(0)).last_touched_secs = 100;

        states.mark_ready(tile(1));
        states.get_mut(tile(1)).last_touched_secs = 200;

        // current=250, grace=60: tile(0) is 150s stale, tile(1) is 50s stale
        let stale: Vec<TileId> = states.stale(250, 60).collect();
        assert!(stale.contains(&tile(0)));
        assert!(!stale.contains(&tile(1)));
    }

    #[test]
    fn stale_renderables_skips_never_touched() {
        let mut states = TileStates::new();
        states.mark_ready(tile(0));
        // last_touched_secs stays 0 (never visited in traversal)
        let stale: Vec<TileId> = states.stale(9999, 0).collect();
        assert!(stale.is_empty());
    }

    #[test]
    fn load_stats_counts_correctly() {
        let mut states = TileStates::new();
        states.mark_loading(tile(0));
        states.mark_ready(tile(1));
        states.get_mut(tile(1)).content_byte_size = 1024;
        states.mark_ready(tile(2));
        states.get_mut(tile(2)).content_byte_size = 512;
        states.mark_failed(tile(3));

        let (loading, renderable, failed, bytes) = states.load_stats();
        assert_eq!(loading, 1);
        assert_eq!(renderable, 2);
        assert_eq!(failed, 1);
        assert_eq!(bytes, 1536);
    }

    #[test]
    fn load_progress_empty_is_100() {
        let states = TileStates::new();
        assert_eq!(states.compute_load_progress(), 100.0);
    }

    #[test]
    fn load_progress_one_renderable_one_loading() {
        let mut states = TileStates::new();
        states.mark_ready(tile(0));
        states.mark_loading(tile(1));
        assert!((states.compute_load_progress() - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_empty_marks_renderable() {
        let mut states = TileStates::new();
        states.apply(&LoadEvent::Empty { tile: tile(0) });
        assert_eq!(states.get(tile(0)).load_state, TileLoadState::Ready);
    }

    #[test]
    fn apply_failed_marks_failed() {
        let mut states = TileStates::new();
        states.apply(&LoadEvent::Failed {
            tile: tile(0),
            url: None,
            message: "oops".into(),
        });
        assert_eq!(states.get(tile(0)).load_state, TileLoadState::Failed);
    }

    #[test]
    fn apply_retry_later_schedules_retry() {
        let mut states = TileStates::new();
        states.frame_index = 10;
        states.apply(&LoadEvent::RetryLater { tile: tile(0) });
        let s = states.get(tile(0));
        assert_eq!(s.load_state, TileLoadState::RetryScheduled);
        // backoff for retry_count=0: 1 << 0 = 1, so retry_frame = 10 + 1 = 11
        assert_eq!(s.next_retry_frame, 11);
    }

    #[test]
    fn schedule_retry_increments_count() {
        let mut states = TileStates::new();
        states.schedule_retry(tile(0), 5);
        let s = states.get(tile(0));
        assert_eq!(s.load_state, TileLoadState::RetryScheduled);
        assert_eq!(s.retry_count, 1);
        assert_eq!(s.next_retry_frame, 5);

        states.schedule_retry(tile(0), 10);
        assert_eq!(states.get(tile(0)).retry_count, 2);
    }
}
