//! Synchronous multi-listener events.
//!
//! Two types work together:
//!
//! | Type | Role | `Clone` |
//! |------|------|---------|
//! | [`Event<A>`] | Fire-side - owned by the emitting struct, private | No |
//! | [`EventListener<A>`] | Subscribe-side - handed to consumers, public | Yes |
//!
//! This split makes it impossible for external code to accidentally call
//! `raise()`. Only the struct that owns the `Event<A>` can fire it.
//!
//! # Deadlock safety
//!
//! `raise()` snapshots the listener list *while holding the lock*, then calls
//! each listener *after releasing the lock*. A listener may therefore safely
//! call [`subscribe`](EventListener::subscribe) or drop a
//! [`SubscriptionHandle`] from inside the callback without deadlocking.
//!
//! # Example
//!
//! ```rust,ignore
//! let evt: Event<u32> = Event::new();
//! let listener = evt.listener();          // hand this to consumers
//!
//! let _h = listener.subscribe(|n| println!("got {n}"));
//! evt.raise(42);  // prints "got 42"
//!
//! drop(_h);       // unsubscribe
//! evt.raise(1);   // nothing printed
//! ```

use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ListenerId(u64);

type ListenerFn<A> = Arc<dyn Fn(&A) + Send + Sync + 'static>;

struct EventInner<A: Send + 'static> {
    listeners: Vec<(ListenerId, ListenerFn<A>)>,
    next_id: u64,
}

impl<A: Send + 'static> EventInner<A> {
    fn new() -> Self {
        Self {
            listeners: Vec::new(),
            next_id: 1,
        }
    }

    fn insert(&mut self, f: ListenerFn<A>) -> ListenerId {
        let id = ListenerId(self.next_id);
        self.next_id += 1;
        self.listeners.push((id, f));
        id
    }

    fn remove(&mut self, id: ListenerId) {
        self.listeners.retain(|(lid, _)| *lid != id);
    }

    /// Snapshot function pointers while holding the lock.
    fn snapshot(&self) -> Vec<ListenerFn<A>> {
        self.listeners.iter().map(|(_, f)| Arc::clone(f)).collect()
    }
}

/// RAII guard returned by [`EventListener::subscribe`].
///
/// Drop this value to unsubscribe the listener. **The handle must be stored** -
/// dropping it immediately unsubscribes.
///
/// `SubscriptionHandle` is type-erased so handles for different event types
/// can be stored in the same `Vec<SubscriptionHandle>`.
pub struct SubscriptionHandle(Box<dyn Send + 'static>);

struct RemoveOnDrop<A: Send + 'static> {
    inner: Arc<Mutex<EventInner<A>>>,
    id: ListenerId,
}

impl<A: Send + 'static> Drop for RemoveOnDrop<A> {
    fn drop(&mut self) {
        if let Ok(mut g) = self.inner.lock() {
            g.remove(self.id);
        }
    }
}

/// Fire-side event handle.
///
/// Create with [`Event::new`], keep private in your struct, and hand out
/// [`EventListener`]s via [`Event::listener`].  Only the owner of an `Event`
/// can call [`raise`](Self::raise).
///
/// Not `Clone` - intentionally. Cloneability would let consumers smuggle a
/// firing capability out of your API boundary.
pub struct Event<A: Send + 'static> {
    inner: Arc<Mutex<EventInner<A>>>,
}

impl<A: Send + 'static> Event<A> {
    /// Create a new event with no listeners.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(EventInner::new())),
        }
    }

    /// Return a subscribe-only [`EventListener`] backed by this event.
    ///
    /// Clone the listener freely and hand it to consumers. All clones share
    /// the same underlying listener list - subscribing via any clone registers
    /// against this event.
    pub fn listener(&self) -> EventListener<A> {
        EventListener {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Raise the event, calling every registered listener with `args`.
    ///
    /// The listener list is snapshotted *before* any listener is invoked.
    /// Listeners may safely subscribe or unsubscribe during the raise.
    pub fn raise(&self, args: A) {
        let listeners = self
            .inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .snapshot();
        for f in listeners {
            f(&args);
        }
    }

    /// Number of currently registered listeners.
    pub fn listener_count(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .listeners
            .len()
    }
}

impl<A: Send + 'static> Default for Event<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Subscribe-only view of an [`Event<A>`].
///
/// Obtained from [`Event::listener`]. `Clone` - hand copies to closures,
/// structs, etc. All clones share the same underlying listener list.
///
/// This type intentionally has **no `raise` method** - only the struct that
/// owns the `Event<A>` can fire it.
#[derive(Clone)]
pub struct EventListener<A: Send + 'static> {
    inner: Arc<Mutex<EventInner<A>>>,
}

impl<A: Send + 'static> EventListener<A> {
    /// Register `f` as a listener.
    ///
    /// Returns a [`SubscriptionHandle`] that unsubscribes `f` when dropped.
    ///
    /// # Important
    ///
    /// **Store the returned handle.** Dropping it immediately is equivalent
    /// to never subscribing.
    #[must_use = "dropping the SubscriptionHandle immediately unsubscribes; store it"]
    pub fn subscribe<F: Fn(&A) + Send + Sync + 'static>(&self, f: F) -> SubscriptionHandle {
        let id = self
            .inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(Arc::new(f));
        SubscriptionHandle(Box::new(RemoveOnDrop {
            inner: Arc::clone(&self.inner),
            id,
        }))
    }

    /// Number of currently registered listeners.
    pub fn listener_count(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .listeners
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn single_listener_called() {
        let evt: Event<u32> = Event::new();
        let count = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&count);
        let _h = evt.listener().subscribe(move |n: &u32| {
            c.fetch_add(*n, Ordering::SeqCst);
        });
        evt.raise(3);
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn multiple_listeners_all_called() {
        let evt: Event<u32> = Event::new();
        let count = Arc::new(AtomicU32::new(0));
        let c1 = Arc::clone(&count);
        let c2 = Arc::clone(&count);
        let _h1 = evt.listener().subscribe(move |_| {
            c1.fetch_add(1, Ordering::SeqCst);
        });
        let _h2 = evt.listener().subscribe(move |_| {
            c2.fetch_add(1, Ordering::SeqCst);
        });
        evt.raise(0);
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn drop_handle_unsubscribes() {
        let evt: Event<u32> = Event::new();
        let count = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&count);
        let h = evt.listener().subscribe(move |_| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        evt.raise(0);
        assert_eq!(count.load(Ordering::SeqCst), 1);
        drop(h);
        evt.raise(0);
        assert_eq!(count.load(Ordering::SeqCst), 1); // unchanged
    }

    #[test]
    fn drop_handle_during_raise_no_deadlock() {
        let evt: Event<u32> = Event::new();
        let listener = evt.listener();
        // Use a second listener clone to subscribe from inside the callback.
        let listener2 = listener.clone();
        let inner_handle: Arc<Mutex<Option<SubscriptionHandle>>> = Arc::new(Mutex::new(None));
        let inner_handle_clone = Arc::clone(&inner_handle);
        let inner_count = Arc::new(AtomicU32::new(0));
        let ic = Arc::clone(&inner_count);
        let h_inner = listener.subscribe(move |_| {
            ic.fetch_add(1, Ordering::SeqCst);
        });
        *inner_handle.lock().unwrap() = Some(h_inner);
        let _h_outer = listener.subscribe(move |_| {
            let _ = listener2.listener_count();
            inner_handle_clone.lock().unwrap().take(); // drops h_inner
        });
        evt.raise(0); // should not deadlock
        assert_eq!(inner_count.load(Ordering::SeqCst), 1);
        evt.raise(0); // inner now removed
        assert_eq!(inner_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn subscribe_during_raise_no_deadlock() {
        let evt: Event<u32> = Event::new();
        let listener = evt.listener();
        let listener2 = listener.clone();
        let late_count = Arc::new(AtomicU32::new(0));
        let lc = Arc::clone(&late_count);
        let _h = listener.subscribe(move |_| {
            let lc2 = Arc::clone(&lc);
            let _h2 = listener2.subscribe(move |_| {
                lc2.fetch_add(1, Ordering::SeqCst);
            });
            std::mem::forget(_h2); // keep alive without storing
        });
        evt.raise(0); // should not deadlock; late listener NOT called this raise
        assert_eq!(late_count.load(Ordering::SeqCst), 0);
    }
}
