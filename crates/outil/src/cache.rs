//! FIFO-eviction bounded cache.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

/// A cache with a fixed entry limit that evicts the oldest inserted entry when
/// the limit is reached.
///
/// Lookup and insertion are both O(1) amortised. The eviction policy is
/// strict FIFO (insertion order); the most-recently-inserted entry is never
/// evicted before older entries.
///
/// Intended for caching network-fetched blobs (subtree availability data, node
/// pages, …) where the working set is small and LRU complexity is unnecessary.
pub struct BoundedFifoCache<K, V> {
    map: HashMap<K, V>,
    order: VecDeque<K>,
    max_entries: usize,
}

impl<K, V> BoundedFifoCache<K, V>
where
    K: Eq + Hash + Clone,
{
    /// Create a new cache bounded to `max_entries` total entries.
    ///
    /// # Panics
    /// Panics if `max_entries` is zero.
    pub fn new(max_entries: usize) -> Self {
        assert!(max_entries > 0, "max_entries must be positive");
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            max_entries,
        }
    }

    /// Look up a value by key.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    /// Insert a key-value pair, evicting the oldest entry if the cache is full.
    ///
    /// If `key` is already present the value is updated in place without
    /// changing the insertion order or evicting any entry.
    pub fn insert(&mut self, key: K, value: V) {
        if self.map.contains_key(&key) {
            self.map.insert(key, value);
            return;
        }
        while self.order.len() >= self.max_entries {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// `true` when the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert_get() {
        let mut c = BoundedFifoCache::new(3);
        c.insert("a", 1);
        c.insert("b", 2);
        assert_eq!(c.get(&"a"), Some(&1));
        assert_eq!(c.get(&"b"), Some(&2));
        assert_eq!(c.get(&"z"), None);
    }

    #[test]
    fn evicts_oldest_when_full() {
        let mut c = BoundedFifoCache::new(2);
        c.insert("a", 1);
        c.insert("b", 2);
        c.insert("c", 3); // evicts "a"
        assert_eq!(c.get(&"a"), None);
        assert_eq!(c.get(&"b"), Some(&2));
        assert_eq!(c.get(&"c"), Some(&3));
    }

    #[test]
    fn update_does_not_evict() {
        let mut c = BoundedFifoCache::new(2);
        c.insert("a", 1);
        c.insert("b", 2);
        c.insert("a", 99); // update, no eviction
        assert_eq!(c.len(), 2);
        assert_eq!(c.get(&"a"), Some(&99));
        assert_eq!(c.get(&"b"), Some(&2));
    }

    #[test]
    fn len_and_is_empty() {
        let mut c: BoundedFifoCache<i32, i32> = BoundedFifoCache::new(4);
        assert!(c.is_empty());
        c.insert(1, 10);
        assert_eq!(c.len(), 1);
        assert!(!c.is_empty());
    }
}
