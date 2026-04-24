//! Detached tasks and join sets.

use std::fmt;
use std::sync::{Arc, Condvar, Mutex};

use crate::error::AsyncError;
use crate::task::{Task, TaskInner};
use crate::task_cell::TaskCell;

/// A collection of tasks that can be joined together.
pub struct JoinSet<T: Send + 'static> {
    entries: Vec<Arc<TaskCell<T>>>,
    /// Shared condvar pair that all registered callbacks notify.
    /// Allocated once at construction and reused across every `join_next` call.
    notifier: Arc<(Mutex<()>, Condvar)>,
}

impl<T: Send + 'static> JoinSet<T> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            notifier: Arc::new((Mutex::new(()), Condvar::new())),
        }
    }

    /// Add a task to the set.
    pub fn push(&mut self, task: Task<T>) {
        let cell = match task.inner {
            TaskInner::Pending(cell) => cell,
            TaskInner::Ready(result) => {
                let cell = Arc::new(TaskCell::new());
                match result {
                    Some(Ok(v)) => cell.complete(Ok(v)),
                    Some(Err(e)) => cell.complete(Err(e)),
                    None => cell.complete(Err(AsyncError::msg("Task already consumed"))),
                }
                cell
            }
        };

        // Register a completion callback once here so that join_next() never
        // needs to re-register callbacks on every call.
        let notifier = Arc::clone(&self.notifier);
        TaskCell::on_ready(&cell, move || {
            // Hold the lock while notifying so the waiter cannot miss the signal.
            let _guard = notifier.0.lock().expect("join push notify lock");
            notifier.1.notify_all();
        });

        self.entries.push(cell);
    }

    /// Number of tasks in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the set contains no tasks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Block until all tasks complete. Returns results in insertion order.
    pub fn block_all(self) -> Vec<Result<T, AsyncError>> {
        self.entries
            .into_iter()
            .map(|cell| {
                cell.wait_until_ready();
                cell.take_result()
                    .unwrap_or_else(|| Err(AsyncError::msg("JoinSet: missing result")))
            })
            .collect()
    }

    /// Block until the next task completes, and return its result.
    /// Returns `None` when the set is empty.
    ///
    /// This method is free of missed-wakeup races: readiness is checked
    /// while holding the condvar mutex, so a completion signal that arrives
    /// between the scan and the `wait` is never lost.
    ///
    /// The `notifier` condvar pair is allocated once when the `JoinSet` is
    /// created and shared with all completion callbacks, so this method
    /// performs no heap allocations in the steady state.
    pub fn join_next(&mut self) -> Option<Result<T, AsyncError>> {
        if self.entries.is_empty() {
            return None;
        }

        loop {
            // Acquire lock first, then scan - this prevents a completing
            // thread from notifying between the scan and wait().
            let guard = self.notifier.0.lock().expect("join_next condvar lock");
            for i in 0..self.entries.len() {
                if self.entries[i].is_ready() {
                    drop(guard);
                    let cell = self.entries.swap_remove(i);
                    return Some(
                        cell.take_result()
                            .unwrap_or_else(|| Err(AsyncError::msg("JoinSet: missing result"))),
                    );
                }
            }
            // Atomically releases the lock and waits.
            drop(self.notifier.1.wait(guard).expect("join_next condvar wait"));
        }
    }
}

impl<T: Send + 'static> Default for JoinSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + 'static> fmt::Debug for JoinSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JoinSet")
            .field("len", &self.entries.len())
            .finish()
    }
}

impl<T: Send + 'static> Extend<Task<T>> for JoinSet<T> {
    fn extend<I: IntoIterator<Item = Task<T>>>(&mut self, iter: I) {
        for task in iter {
            self.push(task);
        }
    }
}

impl<T: Send + 'static> FromIterator<Task<T>> for JoinSet<T> {
    fn from_iter<I: IntoIterator<Item = Task<T>>>(iter: I) -> Self {
        let mut set = JoinSet::new();
        set.extend(iter);
        set
    }
}
