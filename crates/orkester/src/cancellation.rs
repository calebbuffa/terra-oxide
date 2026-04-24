use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::{AsyncError, ErrorCode};
use crate::shared_cell::SharedCell;
use crate::task::{Handle, Resolver, Task, TaskInner, create_pair};
use crate::task_cell::TaskCell;

type Callback = Box<dyn FnOnce() + Send + 'static>;

/// Identifier returned by [`CancellationToken::on_cancel`] that lets the
/// caller unregister a previously-installed callback with
/// [`CancellationToken::unregister`] once the associated work has completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CancelRegistration(u64);

/// Cooperative cancellation token. Cheap to clone (shared `Arc`).
///
/// Use [`CancellationToken::new`] to create a fresh token, pass clones to
/// any number of tasks via [`Task::with_cancellation`], then call
/// [`CancellationToken::cancel`] to signal them all.
///
/// Cancellation is cooperative - it does not abort running work.  Instead,
/// any task that has not yet completed will be rejected with an
/// [`AsyncError`] whose message is `"cancelled"`.
#[derive(Clone)]
pub struct CancellationToken {
    inner: Arc<TokenInner>,
}

struct TokenInner {
    signalled: AtomicBool,
    callbacks: Mutex<CallbackSlab>,
}

/// Slab storage for cancel callbacks. Callbacks can be removed by id so that
/// long-lived tokens (e.g. per-scene) don't accumulate entries from tasks
/// that completed normally without cancellation.
struct CallbackSlab {
    /// `None` slots are reusable holes; `Some` slots hold a live callback.
    slots: Vec<Option<Callback>>,
    /// Stack of vacant indices for O(1) reuse.
    free: Vec<usize>,
    /// Monotonic generation so `CancelRegistration` ids don't collide
    /// across reuse of the same slot index.
    next_gen: u64,
    /// Generation stamped on each occupied slot, parallel to `slots`.
    slot_gen: Vec<u64>,
}

impl CallbackSlab {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            next_gen: 1,
            slot_gen: Vec::new(),
        }
    }

    fn insert(&mut self, cb: Callback) -> CancelRegistration {
        let gen_id = self.next_gen;
        self.next_gen = self.next_gen.wrapping_add(1);
        let idx = if let Some(idx) = self.free.pop() {
            self.slots[idx] = Some(cb);
            self.slot_gen[idx] = gen_id;
            idx
        } else {
            self.slots.push(Some(cb));
            self.slot_gen.push(gen_id);
            self.slots.len() - 1
        };
        CancelRegistration(encode(idx, gen_id))
    }

    /// Remove a callback by registration id. Returns the callback if the
    /// slot is still occupied with the matching generation.
    fn remove(&mut self, reg: CancelRegistration) -> Option<Callback> {
        let (idx, gen_id) = decode(reg.0);
        if idx >= self.slots.len() || self.slot_gen[idx] != gen_id {
            return None;
        }
        let cb = self.slots[idx].take();
        if cb.is_some() {
            self.free.push(idx);
        }
        cb
    }

    /// Drain all live callbacks for firing on cancel.
    fn drain(&mut self) -> Vec<Callback> {
        let mut out = Vec::with_capacity(self.slots.len() - self.free.len());
        for slot in self.slots.iter_mut() {
            if let Some(cb) = slot.take() {
                out.push(cb);
            }
        }
        self.slots.clear();
        self.slot_gen.clear();
        self.free.clear();
        out
    }
}

#[inline]
fn encode(idx: usize, gen_id: u64) -> u64 {
    // Pack (idx, gen) into 64 bits: low 24 bits = idx, high 40 bits = gen.
    // 24 bits = 16M concurrent registrations per token, plenty for our use.
    debug_assert!(idx < (1 << 24));
    ((gen_id & ((1u64 << 40) - 1)) << 24) | ((idx as u64) & ((1u64 << 24) - 1))
}

#[inline]
fn decode(packed: u64) -> (usize, u64) {
    let idx = (packed & ((1u64 << 24) - 1)) as usize;
    let gen_id = packed >> 24;
    (idx, gen_id)
}

impl CancellationToken {
    /// Create a new, unsignalled token.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(TokenInner {
                signalled: AtomicBool::new(false),
                callbacks: Mutex::new(CallbackSlab::new()),
            }),
        }
    }

    /// Signal cancellation. Fires all registered callbacks.
    pub fn cancel(&self) {
        if self
            .inner
            .signalled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let cbs = {
                let mut guard = self.inner.callbacks.lock().expect("token lock");
                guard.drain()
            };
            for cb in cbs {
                cb();
            }
        }
    }

    /// Returns `true` once [`cancel`](Self::cancel) has been called.
    pub fn is_cancelled(&self) -> bool {
        self.inner.signalled.load(Ordering::Acquire)
    }

    /// Register a callback that fires when the token is cancelled.
    /// If already cancelled, the callback fires immediately and the returned
    /// [`CancelRegistration`] refers to no live slot (unregistering is a no-op).
    ///
    /// This can be used to hook into external cancellation sources - for
    /// example, cancelling an in-flight HTTP request when the token is
    /// signalled.
    pub fn on_cancel(&self, cb: impl FnOnce() + Send + 'static) -> CancelRegistration {
        if self.is_cancelled() {
            cb();
            return CancelRegistration(0);
        }

        // Lock-before-check-before-insert to avoid a race with cancel().
        let mut guard = self.inner.callbacks.lock().expect("token lock");
        if self.inner.signalled.load(Ordering::Acquire) {
            drop(guard);
            cb();
            return CancelRegistration(0);
        }
        guard.insert(Box::new(cb))
    }

    /// Remove a previously-registered cancel callback. Used by
    /// [`Task::with_cancellation`] and similar wrappers to release the
    /// callback's captured state once the underlying work has completed
    /// normally, so long-lived tokens don't accumulate dead closures.
    ///
    /// No-op if the id is stale, already fired, or never existed.
    pub fn unregister(&self, reg: CancelRegistration) {
        if reg.0 == 0 || self.is_cancelled() {
            return;
        }
        let mut guard = self.inner.callbacks.lock().expect("token lock");
        let _ = guard.remove(reg);
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + 'static> Task<T> {
    /// Attach a cancellation token. If the token is signalled before the
    /// upstream task completes, the returned task rejects with a
    /// `"cancelled"` error.
    pub fn with_cancellation(self, token: &CancellationToken) -> Task<T> {
        if token.is_cancelled() {
            let (resolver, task) = create_pair();
            resolver.reject(AsyncError::with_code(ErrorCode::Cancelled, "cancelled"));
            return task;
        }

        let (resolver, output) = create_pair::<T>();
        // Single Mutex<Option<Resolver>> acts as both shared state and claim
        // mechanism: whichever path calls .take() first wins; the other sees None.
        let shared: Arc<Mutex<Option<Resolver<T>>>> = Arc::new(Mutex::new(Some(resolver)));
        let reg_slot: Arc<Mutex<Option<CancelRegistration>>> = Arc::new(Mutex::new(None));
        let token_for_unreg = token.clone();

        // Path 1: upstream completes first -> forward result.
        let sp1 = Arc::clone(&shared);
        let reg_slot_1 = Arc::clone(&reg_slot);
        match self.inner {
            TaskInner::Ready(result) => {
                if let Some(resolver) = sp1.lock().expect("cancel resolver lock").take() {
                    match result {
                        Some(Ok(v)) => resolver.resolve(v),
                        Some(Err(e)) => resolver.reject(e),
                        None => resolver.reject(AsyncError::msg("Task already consumed")),
                    }
                }
            }
            TaskInner::Pending(cell) => {
                TaskCell::on_complete(cell, move |result| {
                    if let Some(resolver) = sp1.lock().expect("cancel resolver lock").take() {
                        match result {
                            Ok(v) => resolver.resolve(v),
                            Err(e) => resolver.reject(e),
                        }
                        // Unregister the cancel callback since we completed normally.
                        if let Some(reg) = reg_slot_1.lock().expect("reg slot lock").take() {
                            token_for_unreg.unregister(reg);
                        }
                    }
                });
            }
        }

        // Path 2: token cancelled first -> reject.
        let sp2 = shared;
        let reg = token.on_cancel(move || {
            if let Some(resolver) = sp2.lock().expect("cancel resolver lock").take() {
                resolver.reject(AsyncError::with_code(ErrorCode::Cancelled, "cancelled"));
            }
        });
        *reg_slot.lock().expect("reg slot lock") = Some(reg);

        output
    }
}

impl<T: Clone + Send + 'static> Handle<T> {
    /// Attach a cancellation token. If the token is signalled before the
    /// upstream task completes, the returned task rejects with a
    /// `"cancelled"` error. Does NOT consume the shared task.
    pub fn with_cancellation(&self, token: &CancellationToken) -> Task<T> {
        if token.is_cancelled() {
            let (resolver, task) = create_pair();
            resolver.reject(AsyncError::with_code(ErrorCode::Cancelled, "cancelled"));
            return task;
        }

        let source = Arc::clone(&self.cell);
        let (resolver, output) = create_pair::<T>();
        let shared: Arc<Mutex<Option<Resolver<T>>>> = Arc::new(Mutex::new(Some(resolver)));
        let reg_slot: Arc<Mutex<Option<CancelRegistration>>> = Arc::new(Mutex::new(None));
        let token_for_unreg = token.clone();

        // Path 1: upstream completes first -> forward result.
        let sp1 = Arc::clone(&shared);
        let reg_slot_1 = Arc::clone(&reg_slot);
        SharedCell::on_complete(source, move |result| {
            if let Some(resolver) = sp1.lock().expect("cancel resolver lock").take() {
                match result {
                    Ok(v) => resolver.resolve(v),
                    Err(e) => resolver.reject(e),
                }
                // Unregister the cancel callback since we completed normally.
                if let Some(reg) = reg_slot_1.lock().expect("reg slot lock").take() {
                    token_for_unreg.unregister(reg);
                }
            }
        });

        // Path 2: token cancelled first -> reject.
        let sp2 = shared;
        let reg = token.on_cancel(move || {
            if let Some(resolver) = sp2.lock().expect("cancel resolver lock").take() {
                resolver.reject(AsyncError::with_code(ErrorCode::Cancelled, "cancelled"));
            }
        });
        *reg_slot.lock().expect("reg slot lock") = Some(reg);

        output
    }
}
