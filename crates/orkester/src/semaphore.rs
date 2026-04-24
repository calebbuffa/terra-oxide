//! Counting semaphore for concurrency limiting.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::task::{Resolver, Task};

/// A counting semaphore that limits concurrent access to a resource.
///
/// Call [`acquire`](Semaphore::acquire) to obtain a [`SemaphorePermit`].
/// When the permit is dropped, the slot is released back to the semaphore,
/// potentially waking a queued acquirer.
///
/// # Example
///
/// ```rust,ignore
/// let sem = Semaphore::new(3);
/// let permit = sem.acquire_blocking(); // blocks if 3 permits already held
/// // ... do work ...
/// drop(permit); // releases back to the semaphore
/// ```
#[derive(Clone)]
pub struct Semaphore {
    inner: Arc<SemaphoreInner>,
}

struct SemaphoreInner {
    state: Mutex<SemaphoreState>,
}

struct SemaphoreState {
    permits: usize,
    max_permits: usize,
    waiters: VecDeque<Resolver<()>>,
}

impl Semaphore {
    /// Create a semaphore with `permits` available slots.
    ///
    /// # Panics
    ///
    /// Panics if `permits` is 0.
    pub fn new(permits: usize) -> Self {
        assert!(permits > 0, "semaphore requires at least 1 permit");
        Self {
            inner: Arc::new(SemaphoreInner {
                state: Mutex::new(SemaphoreState {
                    permits,
                    max_permits: permits,
                    waiters: VecDeque::new(),
                }),
            }),
        }
    }

    /// Acquire a permit, **blocking** the current thread until one is available.
    ///
    /// Returns a [`SemaphorePermit`] that releases the slot when dropped.
    ///
    /// # Note
    ///
    /// The returned permit **must** be bound to a variable. Dropping it
    /// immediately releases the permit in the same statement, which is
    /// almost certainly a bug:
    /// ```text
    /// let _permit = sem.acquire_blocking(); // ✓ holds until _permit is dropped
    /// sem.acquire_blocking();               // ✗ released immediately!
    /// ```
    ///
    /// # Deadlock Hazard
    ///
    /// **Do not call this from a [`ThreadPool`](crate::ThreadPool) worker thread
    /// or from a [`WorkQueue`](crate::WorkQueue) continuation.**
    ///
    /// `acquire_blocking` parks the calling thread until a permit is available.
    /// If the thread that would eventually *release* the permit (e.g. another
    /// task dispatched on the same pool) cannot run because the worker is
    /// parked here, the program deadlocks.
    ///
    /// For safe alternatives that never park any thread use
    /// [`acquire_async`](Self::acquire_async) or [`Task::with_semaphore`].
    ///
    /// Panics on WebAssembly (via [`Task::block`]) because that platform has
    /// no way to park threads.
    #[must_use = "dropping the permit immediately releases it"]
    pub fn acquire_blocking(&self) -> SemaphorePermit {
        #[cfg(not(target_arch = "wasm32"))]
        debug_assert!(
            !crate::thread_pool::is_pool_thread(),
            "Semaphore::acquire_blocking called from a ThreadPool worker thread — \
             this will deadlock if the permit holder is queued on the same pool. \
             Use acquire_async() or Task::with_semaphore() instead."
        );
        match self.try_acquire_or_enqueue() {
            Ok(permit) => permit,
            Err(task) => {
                // Block until our resolver is resolved.
                let _ = task.block();
                self.make_permit()
            }
        }
    }

    /// Acquire a permit asynchronously.
    ///
    /// Returns a [`Task`] that resolves to a [`SemaphorePermit`] once a slot
    /// becomes available. Unlike [`acquire`](Self::acquire) this never parks
    /// the calling thread, so it is safe from the main thread, from task
    /// continuations, and from WebAssembly.
    ///
    /// If a permit is already available the task resolves immediately.
    pub fn acquire_async(&self) -> Task<SemaphorePermit> {
        match self.try_acquire_or_enqueue() {
            Ok(permit) => crate::resolved(permit),
            Err(task) => {
                let inner = Arc::clone(&self.inner);
                task.map(move |_| SemaphorePermit { inner })
            }
        }
    }

    /// Try to take a permit immediately. If none is available, enqueue a
    /// resolver and return the task that will fire when the permit is
    /// handed off by a future `Drop`.
    ///
    /// Returns `Ok(permit)` on the fast path, `Err(task)` on the slow path.
    fn try_acquire_or_enqueue(&self) -> Result<SemaphorePermit, Task<()>> {
        {
            let mut state = self.inner.state.lock().expect("semaphore lock");
            if state.permits > 0 {
                state.permits -= 1;
                return Ok(self.make_permit());
            }
        }

        let (resolver, task) = crate::task::create_pair::<()>();
        let mut state = self.inner.state.lock().expect("semaphore lock");
        // Re-check after acquiring lock (permit may have been released).
        if state.permits > 0 {
            state.permits -= 1;
            drop(state);
            resolver.resolve(());
            return Ok(self.make_permit());
        }
        state.waiters.push_back(resolver);
        Err(task)
    }

    #[inline]
    fn make_permit(&self) -> SemaphorePermit {
        SemaphorePermit {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Try to acquire a permit without blocking.
    ///
    /// Returns `Some(permit)` if a slot was available, `None` otherwise.
    #[must_use = "dropping the permit immediately releases it"]
    pub fn try_acquire(&self) -> Option<SemaphorePermit> {
        let mut state = self.inner.state.lock().expect("semaphore lock");
        if state.permits > 0 {
            state.permits -= 1;
            Some(self.make_permit())
        } else {
            None
        }
    }

    /// Returns the number of permits currently available.
    pub fn available_permits(&self) -> usize {
        self.inner.state.lock().expect("semaphore lock").permits
    }

    /// Returns the maximum number of permits.
    pub fn max_permits(&self) -> usize {
        self.inner.state.lock().expect("semaphore lock").max_permits
    }
}

/// RAII guard returned by [`Semaphore::acquire`].
///
/// Dropping this releases one permit back to the semaphore.
pub struct SemaphorePermit {
    inner: Arc<SemaphoreInner>,
}

impl fmt::Debug for Semaphore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        f.debug_struct("Semaphore")
            .field("available", &state.permits)
            .field("max", &state.max_permits)
            .finish()
    }
}

impl Drop for SemaphorePermit {
    fn drop(&mut self) {
        // Acquire lock with panic-safe recovery. If the lock is poisoned, recover
        // by using the inner state. This ensures we never panic in drop.
        let mut state = match self.inner.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                // Lock was poisoned. Use the poisoned guard to recover state.
                // This lets us clean up even if another thread panicked.
                poisoned.into_inner()
            }
        };

        if let Some(waiter) = state.waiters.pop_front() {
            // Give the permit directly to the next waiter.
            // Wrap in catch_unwind to prevent panics in user code from aborting during drop.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                waiter.resolve(());
            }));
        } else {
            state.permits += 1;
        }
    }
}

impl<T: Send + 'static> Task<T> {
    /// Gate this task on a semaphore permit.
    ///
    /// The permit is acquired **asynchronously**: if no permit is available
    /// the returned task is pending until one is released, and no thread is
    /// parked in the meantime. Once acquired the permit is held for the
    /// lifetime of the upstream task (including any time spent waiting on
    /// `self`), and released when the result is delivered.
    ///
    /// Use this to cap how many tasks can be *in flight* at once (for
    /// example, limiting concurrent HTTP requests) without risking the
    /// main-thread deadlock that `Semaphore::acquire` can cause.
    pub fn with_semaphore(self, sem: &Semaphore) -> Task<T> {
        sem.acquire_async().map(move |permit| {
            self.map(move |v| {
                drop(permit);
                v
            })
        })
    }
}
