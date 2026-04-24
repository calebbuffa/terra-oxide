//! Concurrent fetch-once state map.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// Per-key fetch state for [`FetchOnceMap`].
enum FetchState<V> {
    /// A fetch is in flight; callers should retry later.
    Fetching,
    /// The value is ready.
    Ready(Arc<V>),
    /// The fetch failed; callers should propagate failure.
    Failed,
}

/// Thread-safe map that tracks the state of concurrent one-shot fetches.
///
/// Each key can be in one of three states:
///
/// | State | Meaning |
/// |-------|---------|
/// | *absent* | Not yet requested |
/// | `Fetching` | A fetch is in flight; subsequent callers should return `RetryLater` |
/// | `Ready(v)` | The value is available |
/// | `Failed` | The fetch failed permanently |
///
/// Once a key transitions to `Ready` or `Failed` it stays there for the
/// lifetime of the map (no eviction). This makes the type suitable for
/// session-scoped caches such as I3S node-page caches where re-fetching on
/// failure would loop forever.
///
/// The map itself carries no fetch logic — the caller is responsible for
/// initiating and completing the fetch.
///
/// # Example
///
/// ```rust
/// # use std::sync::Arc;
/// # use outil::FetchOnceMap;
/// let cache: FetchOnceMap<u64, String> = FetchOnceMap::new();
///
/// // Caller checks whether it should start a fetch.
/// if cache.try_start(&42) {
///     // ... kick off async work ...
///     cache.set_ready(42, "result".to_owned());
/// }
///
/// assert_eq!(cache.get(&42), Some(Ok(Arc::new("result".to_owned()))));
/// ```
pub struct FetchOnceMap<K, V> {
    entries: Mutex<HashMap<K, FetchState<V>>>,
}

impl<K, V> FetchOnceMap<K, V>
where
    K: Eq + Hash,
{
    /// Create a new, empty map.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Attempt to claim the fetch slot for `key`.
    ///
    /// Returns `true` if the slot was unclaimed (absent) and is now marked
    /// `Fetching`. The caller **must** call [`set_ready`](Self::set_ready) or
    /// [`set_failed`](Self::set_failed) to complete the fetch.
    ///
    /// Returns `false` if the key is already `Fetching`, `Ready`, or `Failed`.
    pub fn try_start(&self, key: &K) -> bool
    where
        K: Clone,
    {
        let mut guard = self.entries.lock().unwrap();
        if guard.contains_key(key) {
            return false;
        }
        guard.insert(key.clone(), FetchState::Fetching);
        true
    }

    /// Mark `key` as successfully fetched with `value`.
    pub fn set_ready(&self, key: K, value: V) {
        self.entries
            .lock()
            .unwrap()
            .insert(key, FetchState::Ready(Arc::new(value)));
    }

    /// Mark `key` as permanently failed.
    pub fn set_failed(&self, key: K) {
        self.entries.lock().unwrap().insert(key, FetchState::Failed);
    }

    /// Look up the current state of `key` without triggering a fetch.
    ///
    /// Returns:
    /// - `None` — key is absent or still `Fetching` (caller should retry)
    /// - `Some(Ok(v))` — ready
    /// - `Some(Err(()))` — failed
    pub fn get(&self, key: &K) -> Option<Result<Arc<V>, ()>> {
        match self.entries.lock().unwrap().get(key)? {
            FetchState::Ready(v) => Some(Ok(Arc::clone(v))),
            FetchState::Failed => Some(Err(())),
            FetchState::Fetching => None,
        }
    }
}

impl<K, V> Default for FetchOnceMap<K, V>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_key_returns_none() {
        let m: FetchOnceMap<u32, &str> = FetchOnceMap::new();
        assert_eq!(m.get(&1), None);
    }

    #[test]
    fn try_start_claims_slot() {
        let m: FetchOnceMap<u32, &str> = FetchOnceMap::new();
        assert!(m.try_start(&1));
        assert!(!m.try_start(&1)); // already claimed
        assert_eq!(m.get(&1), None); // still fetching -> None
    }

    #[test]
    fn set_ready_makes_value_available() {
        let m: FetchOnceMap<u32, &str> = FetchOnceMap::new();
        m.try_start(&1);
        m.set_ready(1, "hello");
        assert_eq!(m.get(&1), Some(Ok(Arc::new("hello"))));
    }

    #[test]
    fn set_failed_returns_err() {
        let m: FetchOnceMap<u32, &str> = FetchOnceMap::new();
        m.try_start(&1);
        m.set_failed(1);
        assert_eq!(m.get(&1), Some(Err(())));
    }

    #[test]
    fn ready_slot_cannot_be_reclaimed() {
        let m: FetchOnceMap<u32, &str> = FetchOnceMap::new();
        m.try_start(&1);
        m.set_ready(1, "v");
        assert!(!m.try_start(&1)); // already done
    }
}
