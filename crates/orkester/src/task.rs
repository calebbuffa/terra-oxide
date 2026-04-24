use crate::context::Context;
use crate::error::{AsyncError, ErrorCode};
use crate::executor::Executor;
use crate::shared_cell::{NO_SLOT, SharedCell, WakerSlot};
use crate::task_cell::TaskCell;
use std::fmt::{self, Debug, Formatter};
use std::future::Future as StdFuture;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

/// A one-shot producer that completes a paired [`Task`].
///
/// Resolving or rejecting consumes the `Resolver`. If dropped without
/// resolving, the paired task is automatically rejected with
/// [`ErrorCode::Dropped`](crate::ErrorCode::Dropped).
pub struct Resolver<T: Send + 'static> {
    cell: Option<Arc<TaskCell<T>>>,
}

impl<T: Send + 'static> Resolver<T> {
    pub(crate) fn new(cell: Arc<TaskCell<T>>) -> Self {
        Self { cell: Some(cell) }
    }

    /// Resolve the paired task with a value.
    pub fn resolve(mut self, value: T) {
        if let Some(cell) = self.cell.take() {
            cell.complete(Ok(value));
        }
    }

    /// Reject the paired task with an error.
    pub fn reject(mut self, error: impl Into<AsyncError>) {
        if let Some(cell) = self.cell.take() {
            cell.complete(Err(error.into()));
        }
    }
}

impl<T: Send + 'static> Drop for Resolver<T> {
    fn drop(&mut self) {
        if let Some(cell) = self.cell.take() {
            cell.complete(Err(AsyncError::with_code(
                ErrorCode::Dropped,
                "Resolver dropped without resolving",
            )));
        }
    }
}

/// Internal state of a `Task<T>`.
///
/// `Ready` holds a synchronous result (zero heap allocation).
/// `Pending` is backed by a `TaskCell` for async completion.
pub(crate) enum TaskInner<T: Send + 'static> {
    Ready(Option<Result<T, AsyncError>>),
    Pending(Arc<TaskCell<T>>),
}

/// Single-consumer async task.
///
/// Move-only. Use [`.share()`](Task::share) to convert to a cloneable
/// [`Handle<T>`]. Implements [`std::future::Future`] for async/await.
///
/// # Consumption
///
/// A `Task<T>` can only be polled to completion **once**. After the first
/// successful poll returns `Poll::Ready`, subsequent polls return
/// `Err(AsyncError)` with [`ErrorCode::Dropped`]. If you need multiple
/// consumers, call [`.share()`](Task::share) to obtain a [`Handle<T>`]
/// (requires `T: Clone`).
pub struct Task<T: Send + 'static> {
    pub(crate) inner: TaskInner<T>,
}

impl<T: Send + 'static> Task<T> {
    /// Create a task that is already resolved with a value.
    #[inline]
    pub(crate) fn ready(value: T) -> Self {
        Self {
            inner: TaskInner::Ready(Some(Ok(value))),
        }
    }

    /// Create a task that is already rejected with an error.
    #[inline]
    pub(crate) fn ready_err(error: AsyncError) -> Self {
        Self {
            inner: TaskInner::Ready(Some(Err(error))),
        }
    }
}

impl<T: Send + 'static> Debug for Task<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let variant = match &self.inner {
            TaskInner::Ready(Some(Ok(_))) => "Ready(Ok)",
            TaskInner::Ready(Some(Err(_))) => "Ready(Err)",
            TaskInner::Ready(None) => "Taken",
            TaskInner::Pending(s) => {
                return f
                    .debug_struct("Task")
                    .field("state", &(Arc::as_ptr(s) as usize))
                    .finish();
            }
        };
        f.debug_struct("Task").field("state", &variant).finish()
    }
}

pub(crate) fn create_pair<T: Send + 'static>() -> (Resolver<T>, Task<T>) {
    let cell = Arc::new(TaskCell::new());
    let resolver = Resolver::new(Arc::clone(&cell));
    let task = Task {
        inner: TaskInner::Pending(cell),
    };
    (resolver, task)
}

/// Output-type adapter for [`Context::run`] and related combinators.
///
/// This trait is already implemented for `T` (plain values) and `Task<T>`
/// (chained tasks). Closures passed to [`Context::run`], [`Task::then`],
/// [`Task::map`], etc. may return either type - both are handled identically.
///
/// You do not need to implement this trait yourself.
pub trait ResolveOutput<T: Send + 'static>: Send + 'static {
    fn resolve_into(self, resolver: Resolver<T>);
    fn into_task(self) -> Task<T>;
}

impl<T> ResolveOutput<T> for T
where
    T: Send + 'static,
{
    fn resolve_into(self, resolver: Resolver<T>) {
        resolver.resolve(self);
    }
    #[inline]
    fn into_task(self) -> Task<T> {
        Task::ready(self)
    }
}

impl<T> ResolveOutput<T> for Task<T>
where
    T: Send + 'static,
{
    fn resolve_into(self, resolver: Resolver<T>) {
        self.pipe_to(resolver);
    }
    #[inline]
    fn into_task(self) -> Task<T> {
        self
    }
}

impl<T: Send + 'static> Task<T> {
    fn consumed_error() -> AsyncError {
        AsyncError::msg("Task already consumed")
    }

    fn pipe_to(self, resolver: Resolver<T>) {
        match self.inner {
            TaskInner::Ready(Some(Ok(value))) => resolver.resolve(value),
            TaskInner::Ready(Some(Err(error))) => resolver.reject(error),
            TaskInner::Ready(None) => resolver.reject(Self::consumed_error()),
            TaskInner::Pending(cell) => {
                TaskCell::on_complete(cell, move |result| match result {
                    Ok(value) => resolver.resolve(value),
                    Err(error) => resolver.reject(error),
                });
            }
        }
    }

    #[inline]
    pub fn is_ready(&self) -> bool {
        match &self.inner {
            TaskInner::Ready(Some(_)) => true,
            TaskInner::Ready(None) => false,
            TaskInner::Pending(state) => state.is_ready(),
        }
    }

    /// Non-blocking poll: returns `Some(result)` if the task has completed,
    /// consuming the value, or `None` if still in-flight.
    ///
    /// Designed for use with `Vec::retain_mut` to drain a collection of
    /// in-flight tasks without blocking.
    ///
    /// ```rust,ignore
    /// tasks.retain_mut(|(key, task)| {
    ///     match task.poll_ready() {
    ///         Some(result) => { handle(key, result); false }
    ///         None         => true,
    ///     }
    /// });
    /// ```
    pub fn poll_ready(&mut self) -> Option<Result<T, AsyncError>> {
        match &self.inner {
            TaskInner::Ready(Some(_)) => {
                let TaskInner::Ready(v) =
                    std::mem::replace(&mut self.inner, TaskInner::Ready(None))
                else {
                    unreachable!()
                };
                Some(v.unwrap())
            }
            TaskInner::Ready(None) => None,
            TaskInner::Pending(cell) => {
                if cell.is_ready() {
                    let TaskInner::Pending(cell) =
                        std::mem::replace(&mut self.inner, TaskInner::Ready(None))
                    else {
                        unreachable!()
                    };
                    Some(
                        cell.take_result()
                            .unwrap_or_else(|| Err(Self::consumed_error())),
                    )
                } else {
                    None
                }
            }
        }
    }

    pub fn block(self) -> Result<T, AsyncError> {
        match self.inner {
            TaskInner::Ready(Some(result)) => result,
            TaskInner::Ready(None) => Err(Self::consumed_error()),
            TaskInner::Pending(cell) => {
                cell.wait_until_ready();
                cell.take_result()
                    .unwrap_or_else(|| Err(Self::consumed_error()))
            }
        }
    }

    fn then_with_executor<U, F, R>(self, executor: Arc<dyn Executor>, f: F) -> Task<U>
    where
        U: Send + 'static,
        F: FnOnce(T) -> R + Send + 'static,
        R: ResolveOutput<U>,
    {
        match self.inner {
            TaskInner::Ready(Some(Ok(value))) => {
                if executor.is_current() {
                    f(value).into_task()
                } else {
                    let (resolver, next) = create_pair::<U>();
                    executor.execute(Box::new(move || f(value).resolve_into(resolver)));
                    next
                }
            }
            TaskInner::Ready(Some(Err(error))) => Task::ready_err(error),
            TaskInner::Ready(None) => Task::ready_err(Self::consumed_error()),
            TaskInner::Pending(cell) => {
                let (resolver, next) = create_pair::<U>();
                TaskCell::on_complete(cell, move |result| match result {
                    Ok(value) => {
                        let run = move || f(value).resolve_into(resolver);
                        if executor.is_current() {
                            run();
                        } else {
                            executor.execute(Box::new(run));
                        }
                    }
                    Err(error) => resolver.reject(error),
                });
                next
            }
        }
    }

    fn catch_with_executor<F, R>(self, executor: Arc<dyn Executor>, f: F) -> Task<T>
    where
        F: FnOnce(AsyncError) -> R + Send + 'static,
        R: ResolveOutput<T>,
    {
        match self.inner {
            TaskInner::Ready(Some(Ok(value))) => Task::ready(value),
            TaskInner::Ready(Some(Err(error))) => {
                if executor.is_current() {
                    f(error).into_task()
                } else {
                    let (resolver, next) = create_pair::<T>();
                    executor.execute(Box::new(move || f(error).resolve_into(resolver)));
                    next
                }
            }
            TaskInner::Ready(None) => Task::ready_err(Self::consumed_error()),
            TaskInner::Pending(cell) => {
                let (resolver, next) = create_pair::<T>();
                TaskCell::on_complete(cell, move |result| match result {
                    Ok(value) => resolver.resolve(value),
                    Err(error) => {
                        let run = move || f(error).resolve_into(resolver);
                        if executor.is_current() {
                            run();
                        } else {
                            executor.execute(Box::new(run));
                        }
                    }
                });
                next
            }
        }
    }

    /// Transform the value inline (on the completing thread).
    /// Equivalent to `.then(Context::IMMEDIATE, f)`.
    pub fn map<U, F, R>(self, f: F) -> Task<U>
    where
        U: Send + 'static,
        F: FnOnce(T) -> R + Send + 'static,
        R: ResolveOutput<U>,
    {
        self.then(&Context::IMMEDIATE, f)
    }

    /// Chain a continuation in the given scheduling context.
    pub fn then<U, F, R>(self, context: &Context, f: F) -> Task<U>
    where
        U: Send + 'static,
        F: FnOnce(T) -> R + Send + 'static,
        R: ResolveOutput<U>,
    {
        match context.executor_opt() {
            Some(executor) => self.then_with_executor(executor, f),
            None => self.then_immediate(f),
        }
    }

    /// Chain an async continuation in the given scheduling context.
    pub fn then_async<U, F, Fut>(self, context: &Context, f: F) -> Task<U>
    where
        U: Send + 'static,
        F: FnOnce(T) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = U> + Send + 'static,
    {
        match self.inner {
            TaskInner::Ready(Some(Ok(value))) => {
                let fut = f(value);
                match context.executor_opt() {
                    Some(executor) => {
                        let (resolver, next) = create_pair::<U>();
                        executor.spawn(Box::pin(async move {
                            resolver.resolve(fut.await);
                        }));
                        next
                    }
                    None => Task::ready(crate::block_on::block_on(fut)),
                }
            }
            TaskInner::Ready(Some(Err(error))) => Task::ready_err(error),
            TaskInner::Ready(None) => Task::ready_err(Self::consumed_error()),
            TaskInner::Pending(cell) => {
                let (resolver, next) = create_pair::<U>();
                let executor = context.executor_opt();
                TaskCell::on_complete(cell, move |result| match result {
                    Ok(value) => {
                        let fut = f(value);
                        match executor {
                            Some(executor) => {
                                executor.spawn(Box::pin(async move {
                                    resolver.resolve(fut.await);
                                }));
                            }
                            None => {
                                resolver.resolve(crate::block_on::block_on(fut));
                            }
                        }
                    }
                    Err(error) => resolver.reject(error),
                });
                next
            }
        }
    }

    fn then_immediate<U, F, R>(self, f: F) -> Task<U>
    where
        U: Send + 'static,
        F: FnOnce(T) -> R + Send + 'static,
        R: ResolveOutput<U>,
    {
        match self.inner {
            TaskInner::Ready(Some(Ok(value))) => f(value).into_task(),
            TaskInner::Ready(Some(Err(error))) => Task::ready_err(error),
            TaskInner::Ready(None) => Task::ready_err(Self::consumed_error()),
            TaskInner::Pending(cell) => {
                let (resolver, next) = create_pair::<U>();
                TaskCell::on_complete(cell, move |result| match result {
                    Ok(value) => f(value).resolve_into(resolver),
                    Err(error) => resolver.reject(error),
                });
                next
            }
        }
    }

    /// Recover from an error inline (on the completing thread).
    /// Equivalent to `.catch(Context::IMMEDIATE, f)`.
    pub fn or_else<F, R>(self, f: F) -> Task<T>
    where
        F: FnOnce(AsyncError) -> R + Send + 'static,
        R: ResolveOutput<T>,
    {
        self.catch(&Context::IMMEDIATE, f)
    }

    /// Catch an error in the given scheduling context.
    pub fn catch<F, R>(self, context: &Context, f: F) -> Task<T>
    where
        F: FnOnce(AsyncError) -> R + Send + 'static,
        R: ResolveOutput<T>,
    {
        match context.executor_opt() {
            Some(executor) => self.catch_with_executor(executor, f),
            None => self.catch_immediate(f),
        }
    }

    fn catch_immediate<F, R>(self, f: F) -> Task<T>
    where
        F: FnOnce(AsyncError) -> R + Send + 'static,
        R: ResolveOutput<T>,
    {
        match self.inner {
            TaskInner::Ready(Some(Ok(value))) => Task::ready(value),
            TaskInner::Ready(Some(Err(error))) => f(error).into_task(),
            TaskInner::Ready(None) => Task::ready_err(Self::consumed_error()),
            TaskInner::Pending(cell) => {
                let (resolver, next) = create_pair::<T>();
                TaskCell::on_complete(cell, move |result| match result {
                    Ok(value) => resolver.resolve(value),
                    Err(error) => f(error).resolve_into(resolver),
                });
                next
            }
        }
    }

    /// Wrap this task with a timeout.
    pub fn with_timeout(self, duration: Duration) -> Task<T> {
        crate::combinators::timeout(self, duration)
    }
}

impl<T, E> Task<Result<T, E>>
where
    T: Send + 'static,
    E: Send + 'static,
{
    /// Chain a fallible continuation. Propagates `Err(e)` without invoking `f`.
    pub fn and_then<U, F>(self, context: &Context, f: F) -> Task<Result<U, E>>
    where
        U: Send + 'static,
        F: FnOnce(T) -> Result<U, E> + Send + 'static,
    {
        self.then(context, move |result| match result {
            Ok(v) => f(v),
            Err(e) => Err(e),
        })
    }
}

impl<T: Send + 'static> Task<T> {
    /// Combine this task with `other`, resolving when **both** complete.
    pub fn join<U: Send + 'static>(self, other: Task<U>) -> Task<(T, U)> {
        crate::combinators::join(self, other)
    }

    /// Convert to a cloneable [`Handle<T>`].
    pub fn share(self) -> Handle<T>
    where
        T: Clone,
    {
        let shared = Arc::new(SharedCell::new());
        let sc = Arc::clone(&shared);
        match self.inner {
            TaskInner::Ready(Some(result)) => sc.complete(result),
            TaskInner::Ready(None) => sc.complete(Err(Self::consumed_error())),
            TaskInner::Pending(cell) => {
                TaskCell::on_complete(cell, move |result| sc.complete(result));
            }
        }
        Handle {
            cell: shared,
            waker_slot: NO_SLOT,
        }
    }
}

impl<T: Send + 'static> Unpin for Task<T> {}

impl<T: Send + 'static> StdFuture for Task<T> {
    type Output = Result<T, AsyncError>;

    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match &mut this.inner {
            TaskInner::Ready(slot) => match slot.take() {
                Some(result) => Poll::Ready(result),
                None => Poll::Ready(Err(Self::consumed_error())),
            },
            TaskInner::Pending(cell) => {
                if cell.is_ready() {
                    Poll::Ready(
                        cell.take_result()
                            .unwrap_or_else(|| Err(Self::consumed_error())),
                    )
                } else {
                    // SAFETY: Task<T> is single-consumer (move-only, not Clone).
                    unsafe {
                        cell.register_waker(cx.waker());
                    }
                    Poll::Pending
                }
            }
        }
    }
}

/// Cloneable multi-consumer async task.
///
/// Create via [`Task::share()`]. Multiple clones observe the same underlying
/// result. Closures passed to [`then`](Handle::then) receive `T` by value
/// (cloned from the stored result).
///
/// `Handle<T>` implements [`Future`](std::future::Future) so it can be
/// directly `.await`'d in async code. Each clone registers its waker directly
/// inside the shared [`SharedCell`](crate::shared_cell::SharedCell) using a
/// pre-allocated slot index, so `Handle::clone()` no longer allocates a
/// separate `Arc<Mutex<Option<Waker>>>`.
///
/// Requires `T: Clone + Send`.
pub struct Handle<T: Clone + Send + 'static> {
    pub(crate) cell: Arc<SharedCell<T>>,
    /// Index into the waker slot table inside `SharedCell`.
    /// `NO_SLOT` means no slot has been allocated yet (first poll will allocate).
    waker_slot: WakerSlot,
}

impl<T: Clone + Send + 'static> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            cell: Arc::clone(&self.cell),
            // Each clone gets an independent waker slot, allocated lazily on
            // the first poll.  No Arc or Mutex allocation here.
            waker_slot: NO_SLOT,
        }
    }
}

impl<T: Clone + Send + 'static> Drop for Handle<T> {
    fn drop(&mut self) {
        if self.waker_slot != NO_SLOT {
            SharedCell::free_waker_slot(&self.cell, self.waker_slot);
        }
    }
}

impl<T: Clone + Send + 'static> Debug for Handle<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handle")
            .field("cell", &(Arc::as_ptr(&self.cell) as usize))
            .finish()
    }
}

impl<T: Clone + Send + 'static> Handle<T> {
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.cell.is_ready()
    }

    /// Returns a clone of the result if ready, or `None` if still pending.
    #[inline]
    pub fn get(&self) -> Option<Result<T, AsyncError>> {
        self.cell.get()
    }

    /// Block the current thread until the result is available.
    pub fn block(&self) -> Result<T, AsyncError> {
        self.cell.wait_and_get()
    }

    fn then_with_executor<U, F, R>(&self, executor: Arc<dyn Executor>, f: F) -> Task<U>
    where
        U: Send + 'static,
        F: FnOnce(T) -> R + Send + 'static,
        R: ResolveOutput<U>,
    {
        let source = Arc::clone(&self.cell);
        let (resolver, next_task) = create_pair::<U>();

        SharedCell::on_complete(source, move |result| match result {
            Ok(value) => {
                let run = move || f(value).resolve_into(resolver);
                if executor.is_current() {
                    run();
                } else {
                    executor.execute(Box::new(run));
                }
            }
            Err(error) => resolver.reject(error),
        });

        next_task
    }

    /// Transform the value inline (on the completing thread).
    pub fn map<U, F, R>(&self, f: F) -> Task<U>
    where
        U: Send + 'static,
        F: FnOnce(T) -> R + Send + 'static,
        R: ResolveOutput<U>,
    {
        self.then(&Context::IMMEDIATE, f)
    }

    /// Chain a continuation in the given scheduling context.
    pub fn then<U, F, R>(&self, context: &Context, f: F) -> Task<U>
    where
        U: Send + 'static,
        F: FnOnce(T) -> R + Send + 'static,
        R: ResolveOutput<U>,
    {
        match context.executor_opt() {
            Some(executor) => self.then_with_executor(executor, f),
            None => self.then_immediate(f),
        }
    }

    fn then_immediate<U, F, R>(&self, f: F) -> Task<U>
    where
        U: Send + 'static,
        F: FnOnce(T) -> R + Send + 'static,
        R: ResolveOutput<U>,
    {
        let source = Arc::clone(&self.cell);
        let (resolver, next_task) = create_pair::<U>();

        SharedCell::on_complete(source, move |result| match result {
            Ok(value) => f(value).resolve_into(resolver),
            Err(error) => resolver.reject(error),
        });

        next_task
    }
}

impl<T: Clone + Send + 'static> Unpin for Handle<T> {}

/// `Handle<T>` implements `Future` so it can be `.await`'d in async code.
///
/// On the first poll the handle allocates a waker slot inside the shared
/// [`SharedCell`] and stores the current waker there.  Subsequent polls update
/// the slot in place.  When the cell completes it wakes all slots in a single
/// locked sweep — no per-clone Arc or Mutex allocation is needed.
impl<T: Clone + Send + 'static> StdFuture for Handle<T> {
    type Output = Result<T, AsyncError>;

    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        // Fast path: already complete.
        if let Some(result) = this.cell.get() {
            return Poll::Ready(result);
        }

        // Allocate a slot on the first poll.
        if this.waker_slot == NO_SLOT {
            this.waker_slot = SharedCell::alloc_waker_slot(&this.cell);
            // alloc_waker_slot returns NO_SLOT if already complete.
            if this.waker_slot == NO_SLOT {
                if let Some(result) = this.cell.get() {
                    return Poll::Ready(result);
                }
                // Spurious NO_SLOT (shouldn't happen): treat as pending.
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
        }

        // Register or update the waker.  Returns true if already complete.
        if SharedCell::register_waker(&this.cell, this.waker_slot, cx.waker()) {
            if let Some(result) = this.cell.get() {
                return Poll::Ready(result);
            }
        }

        Poll::Pending
    }
}
