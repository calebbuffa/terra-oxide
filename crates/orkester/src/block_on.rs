#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::future::Future;
    use std::pin::pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake};

    struct ThreadWaker(std::thread::Thread);

    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    /// Block the current thread until `future` completes.
    ///
    /// Uses thread parking for efficient waiting. Each call to `wake()` unparks
    /// the blocked thread so it can re-poll.
    pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
        let waker = Arc::new(ThreadWaker(std::thread::current())).into();
        let mut cx = Context::from_waker(&waker);
        let mut future = pin!(future);

        loop {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(output) => return output,
                Poll::Pending => std::thread::park(),
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::block_on;

#[cfg(target_arch = "wasm32")]
pub(crate) fn block_on<F: Future>(_future: F) -> F::Output {
    panic!(
        "block_on is not available in WebAssembly. \
         Drive futures through WorkQueue::pump() in the browser's render loop instead."
    );
}
