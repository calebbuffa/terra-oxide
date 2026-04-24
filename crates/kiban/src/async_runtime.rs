use std::time::Duration;

use orkester::{Context, Task, WorkQueue};

/// Platform-agnostic async runtime for kiban.
///
/// Holds a `background` [`Context`] (any executor - `ThreadPool` on native,
/// `WasmExecutor` on WebAssembly, or a custom backend) and a user-pumped
/// [`WorkQueue`] for main-thread continuations.
///
/// The choice of background executor belongs to the *application layer*, not
/// to kiban. Construct with [`AsyncRuntime::new`] and pass whatever context
/// is appropriate for the target platform.
#[derive(Clone)]
pub struct AsyncRuntime {
    background: Context,
    work_queue: WorkQueue,
}

impl AsyncRuntime {
    /// Create a runtime with the given background execution context.
    ///
    /// # Native example
    /// ```rust,ignore
    /// let runtime = AsyncRuntime::new(orkester::ThreadPool::new(4).context());
    /// ```
    ///
    /// # WASM example
    /// ```rust,ignore
    /// let runtime = AsyncRuntime::new(orkester::Context::new(orkester::WasmExecutor));
    /// ```
    pub fn new(background: Context) -> Self {
        Self {
            background,
            work_queue: WorkQueue::new(),
        }
    }

    /// Context for background work (off main thread on native; inline on WASM).
    pub fn background(&self) -> Context {
        self.background.clone()
    }

    /// Context for main-thread continuations (always the `WorkQueue`).
    pub fn main(&self) -> Context {
        self.work_queue.context()
    }

    pub fn flush_main(&mut self) -> usize {
        self.work_queue.flush()
    }

    pub fn pump_main(&mut self) -> bool {
        self.work_queue.pump()
    }

    /// Execute pending main-thread tasks until the queue is empty or `budget`
    /// has elapsed. Returns the number of tasks executed.
    pub fn flush_timed(&mut self, budget: Duration) -> usize {
        self.work_queue.flush_timed(budget)
    }

    pub fn start_task<T: Send + 'static>(
        &self,
        task: impl FnOnce() -> T + Send + 'static,
    ) -> Task<T> {
        self.background().run(task)
    }
}
