//! Node-page cache for I3S loaders.
//!
//! I3S node trees are delivered as fixed-size pages of nodes rather than
//! individual node documents.  This module provides a shared, thread-safe
//! cache that fetches each page exactly once.

use std::sync::Arc;

use courtier::{AssetAccessor, AssetResponse, RequestPriority};
use i3s::cmn::NodePage;
use orkester::Context;
use outil::FetchOnceMap;

/// Opaque cache key: the 0-based page identifier.
pub(super) type PageId = u64;

/// Thread-safe cache of I3S node pages.
///
/// Shared by all `load_tile` / `create_children` calls on the same layer.
pub(super) struct NodePageCache {
    inner: Arc<FetchOnceMap<PageId, NodePage>>,
}

impl NodePageCache {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(FetchOnceMap::new()),
        }
    }

    /// Page ID and within-page offset for a global node index.
    pub fn page_id(node_idx: u64, nodes_per_page: i64) -> PageId {
        node_idx / nodes_per_page as u64
    }

    pub fn page_offset(node_idx: u64, nodes_per_page: i64) -> usize {
        (node_idx % nodes_per_page as u64) as usize
    }

    /// Kick off an async fetch for `page_id` if not already in flight.
    ///
    /// Returns immediately.  Callers should return `RetryLater` and call
    /// [`get`] on subsequent frames to check completion.
    ///
    /// Safe to call repeatedly — subsequent calls are no-ops while a fetch is
    /// in flight or after it has resolved.
    pub fn ensure_fetched(
        &self,
        page_id: PageId,
        page_url: String,
        accessor: Arc<dyn AssetAccessor>,
        headers: Arc<[(String, String)]>,
        bg: Context,
    ) {
        if !self.inner.try_start(&page_id) {
            return; // already fetching, ready, or failed
        }

        // Fire-and-forget: the task populates the cache when done.
        let cache = Arc::clone(&self.inner);
        drop::<orkester::Task<()>>(
            accessor
                .get(&page_url, &headers, RequestPriority::HIGH, None)
                .then(
                    &bg,
                    move |io_result: Result<AssetResponse, courtier::FetchError>| {
                        match io_result {
                            Ok(resp) if resp.check_status().is_ok() => {
                                match serde_json::from_slice::<NodePage>(resp.decompressed_data()) {
                                    Ok(page) => cache.set_ready(page_id, page),
                                    Err(_) => cache.set_failed(page_id),
                                }
                            }
                            _ => cache.set_failed(page_id),
                        }
                        orkester::resolved(())
                    },
                ),
        );
    }

    /// Look up a ready page without triggering a fetch.
    pub fn get(&self, page_id: PageId) -> Option<Result<Arc<NodePage>, ()>> {
        self.inner.get(&page_id)
    }
}
