//! Multi-consumer completion cell for [`Handle`](crate::Handle).
//!
//! Unlike [`TaskCell`](crate::task_cell::TaskCell) - which transfers ownership
//! of the result to a single consumer - `SharedCell<T>` stores the result once
//! and clones it for each consumer.  Requires `T: Clone + Send`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::Waker;

use crate::error::AsyncError;

type SharedCallback<T> = Box<dyn FnOnce(Result<T, AsyncError>) + Send + 'static>;

/// Slot index used by a `Handle` clone to identify its waker inside `SharedCell`.
/// `usize::MAX` is the sentinel for "no slot allocated yet".
pub(crate) type WakerSlot = usize;
pub(crate) const NO_SLOT: WakerSlot = usize::MAX;

enum CellState<T> {
    Pending {
        callbacks: Vec<SharedCallback<T>>,
        /// Per-clone waker slots.  Index into this vec is the `WakerSlot`
        /// stored in each `Handle` clone.  `None` means the slot was freed
        /// (handle polled after completion or dropped).
        wakers: Vec<Option<Waker>>,
        /// Indices of freed waker slots available for reuse.
        free_wakers: Vec<usize>,
    },
    Ready {
        result: Result<T, AsyncError>,
    },
}

/// Multi-consumer completion cell.
///
/// Stores a `Result<T, AsyncError>` written exactly once. Clones the result
/// for each registered callback and for each blocking waiter.
pub struct SharedCell<T: Clone + Send + 'static> {
    /// Fast path: `true` once [`complete`](Self::complete) has been called.
    ready: AtomicBool,
    state: Mutex<CellState<T>>,
    condvar: Condvar,
}

impl<T: Clone + Send + 'static> SharedCell<T> {
    pub(crate) fn new() -> Self {
        Self {
            ready: AtomicBool::new(false),
            state: Mutex::new(CellState::Pending {
                callbacks: Vec::new(),
                wakers: Vec::new(),
                free_wakers: Vec::new(),
            }),
            condvar: Condvar::new(),
        }
    }

    /// Returns `true` once [`complete`](Self::complete) has been called.
    #[inline]
    pub(crate) fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Allocate a waker slot for a `Handle` clone.  Returns the slot index.
    /// Reuses freed slots when available.
    pub(crate) fn alloc_waker_slot(cell: &Arc<Self>) -> WakerSlot {
        if cell.is_ready() {
            return NO_SLOT;
        }
        let mut st = cell.state.lock().unwrap_or_else(|p| p.into_inner());
        match &mut *st {
            CellState::Ready { .. } => NO_SLOT,
            CellState::Pending {
                wakers,
                free_wakers,
                ..
            } => {
                if let Some(idx) = free_wakers.pop() {
                    wakers[idx] = None;
                    idx
                } else {
                    wakers.push(None);
                    wakers.len() - 1
                }
            }
        }
    }

    /// Free a previously-allocated waker slot.  Called from `Handle::drop`.
    pub(crate) fn free_waker_slot(cell: &Arc<Self>, slot: WakerSlot) {
        if slot == NO_SLOT || cell.is_ready() {
            return;
        }
        let mut st = cell.state.lock().unwrap_or_else(|p| p.into_inner());
        if let CellState::Pending {
            wakers,
            free_wakers,
            ..
        } = &mut *st
        {
            if slot < wakers.len() {
                wakers[slot] = None;
                free_wakers.push(slot);
            }
        }
    }

    /// Register or update the waker for a slot.  Returns `true` if the cell
    /// was already complete (caller should return `Poll::Ready` immediately).
    pub(crate) fn register_waker(cell: &Arc<Self>, slot: WakerSlot, waker: &Waker) -> bool {
        if cell.is_ready() {
            return true;
        }
        let mut st = cell.state.lock().unwrap_or_else(|p| p.into_inner());
        match &mut *st {
            CellState::Ready { .. } => true,
            CellState::Pending { wakers, .. } => {
                if slot < wakers.len() {
                    match &wakers[slot] {
                        Some(w) if w.will_wake(waker) => {}
                        _ => wakers[slot] = Some(waker.clone()),
                    }
                }
                false
            }
        }
    }

    /// Store the result and fire all registered callbacks with owned clones.
    ///
    /// # Panics
    ///
    /// Panics if called more than once.
    pub(crate) fn complete(&self, result: Result<T, AsyncError>) {
        let (callbacks, wakers) = {
            let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
            let old = std::mem::replace(
                &mut *st,
                CellState::Ready {
                    result: result.clone(),
                },
            );
            self.ready.store(true, Ordering::Release);
            match old {
                CellState::Pending {
                    callbacks, wakers, ..
                } => (callbacks, wakers),
                CellState::Ready { .. } => panic!("SharedCell completed twice"),
            }
        };
        self.condvar.notify_all();
        // Wake all registered Handle pollers.
        for waker in wakers.into_iter().flatten() {
            waker.wake();
        }
        // Fire registered callbacks.
        for cb in callbacks {
            cb(result.clone());
        }
    }

    /// Clone the stored result. Returns `None` if not yet complete.
    pub(crate) fn get(&self) -> Option<Result<T, AsyncError>> {
        if !self.is_ready() {
            return None;
        }
        let st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        match &*st {
            CellState::Ready { result } => Some(result.clone()),
            CellState::Pending { .. } => None,
        }
    }

    /// Block the current thread until the cell is complete, then return a
    /// cloned copy of the result.
    pub(crate) fn wait_and_get(&self) -> Result<T, AsyncError> {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            match &*st {
                CellState::Ready { result } => return result.clone(),
                CellState::Pending { .. } => {
                    st = self.condvar.wait(st).unwrap_or_else(|p| p.into_inner());
                }
            }
        }
    }

    /// Register a callback that receives an owned `Result<T, AsyncError>` (cloned
    /// from the stored result) when this cell completes. If already complete,
    /// fires immediately.
    pub(crate) fn on_complete(
        cell: Arc<Self>,
        f: impl FnOnce(Result<T, AsyncError>) + Send + 'static,
    ) {
        let mut f: Option<SharedCallback<T>> = Some(Box::new(f));
        let to_fire = {
            let mut st = cell.state.lock().unwrap_or_else(|p| p.into_inner());
            match &mut *st {
                CellState::Ready { result } => Some(result.clone()),
                CellState::Pending { callbacks, .. } => {
                    callbacks.push(f.take().unwrap());
                    None
                }
            }
        };
        if let Some(r) = to_fire {
            f.take().unwrap()(r);
        }
    }
}
