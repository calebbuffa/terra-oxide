use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crossbeam_deque::{Injector, Steal, Stealer, Worker};

use crate::executor::Executor;

type Task = Box<dyn FnOnce() + Send + 'static>;

thread_local! {
    static CURRENT_POOL_ID: Cell<Option<u64>> = const { Cell::new(None) };
}

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

struct PoolShared {
    id: u64,
    /// Lock-free global injection queue. Tasks submitted from outside threads
    /// land here; workers steal from it when their local deque is empty.
    injector: Injector<Task>,
    /// One stealer per worker - used by other workers to steal tasks.
    stealers: Vec<Stealer<Task>>,
    /// Per-worker park/unpark: workers sleep here when fully idle.
    /// `notify_one` wakes exactly one sleeper per injected task.
    mutex: Mutex<Parked>,
    condvar: Condvar,
}

struct Parked {
    sleeping: usize,
    shutdown: bool,
}

struct PoolInner {
    shared: Arc<PoolShared>,
    _workers: Vec<thread::JoinHandle<()>>,
}

impl Drop for PoolInner {
    fn drop(&mut self) {
        {
            let mut p = self.shared.mutex.lock().unwrap_or_else(|g| g.into_inner());
            p.shutdown = true;
        }
        self.shared.condvar.notify_all();
    }
}

/// Try to pop from local, then steal from injector or any sibling worker.
/// Returns `Some(task)` if work was found, `None` if all queues are empty.
fn find_task(local: &Worker<Task>, shared: &PoolShared) -> Option<Task> {
    // 1. Local deque first (LIFO - best cache locality for chained continuations).
    local.pop().or_else(|| {
        // 2. Drain the global injector into local, then take one.
        loop {
            match shared.injector.steal_batch_and_pop(local) {
                Steal::Success(t) => return Some(t),
                Steal::Empty => break,
                Steal::Retry => continue,
            }
        }
        // 3. Try stealing from sibling workers.
        for stealer in &shared.stealers {
            loop {
                match stealer.steal_batch_and_pop(local) {
                    Steal::Success(t) => return Some(t),
                    Steal::Empty => break,
                    Steal::Retry => continue,
                }
            }
        }
        None
    })
}

/// Work-stealing thread pool matching the performance profile of `async++`'s
/// `threadpool_scheduler` used by CesiumAsync.
///
/// Architecture:
/// - One [`Injector`] (lock-free MPMC) for external task submission.
/// - Per-worker [`Worker`] deques (LIFO) for local task execution and
///   continuation chaining with zero contention.
/// - All other workers can [`Stealer::steal`] from any peer when idle.
/// - Workers park on a `Condvar` only when all queues are truly empty;
///   `notify_one` wakes exactly one worker per new task.
#[derive(Clone)]
pub struct ThreadPool {
    inner: Arc<PoolInner>,
}

impl Executor for ThreadPool {
    fn execute(&self, task: Box<dyn FnOnce() + Send + 'static>) {
        self.schedule(task);
    }

    fn is_current(&self) -> bool {
        CURRENT_POOL_ID.with(|slot| slot.get() == Some(self.inner.shared.id))
    }
}

impl ThreadPool {
    pub fn new(number_of_threads: usize) -> Self {
        let number_of_threads = number_of_threads.max(1);
        let id = NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed);

        // Build one Worker per thread and collect their Stealers for sharing.
        let mut locals: Vec<Worker<Task>> =
            (0..number_of_threads).map(|_| Worker::new_lifo()).collect();
        let stealers: Vec<Stealer<Task>> = locals.iter().map(|w| w.stealer()).collect();

        let shared = Arc::new(PoolShared {
            id,
            injector: Injector::new(),
            stealers,
            mutex: Mutex::new(Parked {
                sleeping: 0,
                shutdown: false,
            }),
            condvar: Condvar::new(),
        });

        let mut workers = Vec::with_capacity(number_of_threads);
        for (index, local) in locals.drain(..).enumerate() {
            let shared = Arc::clone(&shared);
            let thread_name = format!("orkester-pool-{id}-worker-{index}");
            let handle = thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
                    CURRENT_POOL_ID.with(|slot| slot.set(Some(id)));

                    'outer: loop {
                        // Run all available work before considering sleep.
                        while let Some(task) = find_task(&local, &shared) {
                            task();
                        }

                        // All queues appear empty. Acquire the sleeping-count
                        // lock to close the lost-wakeup window: if a producer
                        // pushes *before* we increment `sleeping`, its
                        // notify_one fires after we're already waiting;
                        // if it pushes *after* we increment `sleeping`, the
                        // notify arrives while we're in `condvar.wait`.
                        //
                        // IMPORTANT: do NOT call find_task (which consumes
                        // via steal_batch_and_pop) while holding this lock.
                        // Use is_empty() checks only so tasks are not dropped.
                        let mut guard = shared.mutex.lock().unwrap_or_else(|g| g.into_inner());

                        // Re-check emptiness without consuming anything.
                        let has_work = !local.is_empty()
                            || !shared.injector.is_empty()
                            || shared.stealers.iter().any(|s| !s.is_empty());

                        if has_work {
                            // Something arrived between the outer loop and here
                            // - drop the lock and loop back to consume it.
                            drop(guard);
                            continue 'outer;
                        }

                        if guard.shutdown {
                            break 'outer;
                        }

                        guard.sleeping += 1;
                        guard = shared
                            .condvar
                            .wait(guard)
                            .unwrap_or_else(|g| g.into_inner());
                        guard.sleeping -= 1;

                        let shutdown = guard.shutdown;
                        drop(guard);

                        if shutdown {
                            break 'outer;
                        }
                        // Loop back to find_task at top of 'outer.
                    }

                    CURRENT_POOL_ID.with(|slot| slot.set(None));
                })
                .expect("failed to spawn thread-pool worker");
            workers.push(handle);
        }

        Self {
            inner: Arc::new(PoolInner {
                shared,
                _workers: workers,
            }),
        }
    }

    fn schedule(&self, task: Task) {
        self.inner.shared.injector.push(task);
        // Wake one sleeping worker; if all are busy they will find it when
        // they finish their current task via find_task stealing from injector.
        let sleeping = self
            .inner
            .shared
            .mutex
            .lock()
            .unwrap_or_else(|g| g.into_inner())
            .sleeping;
        if sleeping > 0 {
            self.inner.shared.condvar.notify_one();
        }
    }

    /// Return a [`Context`](crate::Context) that routes tasks into this pool.
    pub fn context(&self) -> crate::context::Context {
        crate::context::Context::new(self.clone())
    }
}

/// Returns `true` if the calling thread is currently running inside a
/// `ThreadPool` worker.  Used by `Semaphore::acquire_blocking` to catch
/// inadvertent blocking calls from pool threads in debug builds.
pub(crate) fn is_pool_thread() -> bool {
    CURRENT_POOL_ID.with(|slot| slot.get().is_some())
}

impl Default for ThreadPool {
    /// Creates a thread pool with `max(available_cpus - 1, 1)` threads.
    ///
    /// Reserves one CPU for the main/UI thread. Falls back to 4 CPUs if
    /// `available_parallelism()` is unavailable.
    fn default() -> Self {
        let cpus = thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(4);
        Self::new((cpus.saturating_sub(1)).max(1))
    }
}
