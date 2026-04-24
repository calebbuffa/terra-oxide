//! [`AckSignal<K>`] - a `#[must_use]` acknowledgement handle.
//!
//! Used to signal completion back to a producer after asynchronous work
//! finishes.  The signal is embedded in an event; the receiver calls
//! `.signal()` after their work is done, or `.cancel()` to explicitly
//! opt out without triggering the must-use warning.
//!
//! # Typical pattern
//!
//! ```rust,ignore
//! // Producer side - create a factory + receiver pair once:
//! let (factory, rx) = orkester::ack_channel::<TileId>(256);
//!
//! // Per event - embed a signal in the event payload:
//! let sig = factory.make(tile_id);
//! event_queue.push(MyEvent { tile: tile_id, data, sig });
//!
//! // Consumer side - call signal after populating content store:
//! match event {
//!     MyEvent { tile, data, sig } => {
//!         my_store.insert(tile, data);
//!         sig.signal();   // producer's rx receives tile_id
//!     }
//! }
//!
//! // Producer drains acks:
//! while let Ok(tile) = rx.try_recv() {
//!     mark_ready(tile);
//! }
//! ```
//!
//! # Cancellation
//!
//! Call `.cancel()` to explicitly discard the signal without triggering the
//! `#[must_use]` warning.  This is the correct path when you decide not to
//! prepare content for a tile (e.g. it was evicted before you processed the
//! event).

use crate::channel;
use crate::channel::{Receiver, Sender};

/// Factory for producing [`AckSignal<K>`] instances that all deliver to the
/// same [`Receiver<K>`].
///
/// Cheap to clone - backed by an `Arc`-shared sender.
#[derive(Clone)]
pub struct AckSignalFactory<K: Send + 'static> {
    tx: Sender<K>,
}

impl<K: Send + 'static> AckSignalFactory<K> {
    pub(crate) fn new(tx: Sender<K>) -> Self {
        Self { tx }
    }

    /// Create a new [`AckSignal`] that delivers `key` when signalled.
    pub fn make(&self, key: K) -> AckSignal<K> {
        AckSignal {
            key: Some(key),
            tx: self.tx.clone(),
        }
    }
}

/// Create a bounded acknowledgement channel.
///
/// Returns an [`AckSignalFactory<K>`] (the write side - distribute to
/// consumers) and a [`Receiver<K>`] (the read side - drain from the
/// producer/coordinator).
pub fn ack_channel<K: Send + 'static>(capacity: usize) -> (AckSignalFactory<K>, Receiver<K>) {
    let (tx, rx) = channel::mpsc(capacity.max(1));
    (AckSignalFactory::new(tx), rx)
}

/// A `#[must_use]` acknowledgement handle.
///
/// Call [`.signal()`](AckSignal::signal) after your work is done, or
/// [`.cancel()`](AckSignal::cancel) to explicitly opt out.
///
/// Dropping without calling either produces a compiler warning.
#[must_use = "call .signal() when done, or .cancel() to opt out; dropping without either is a bug"]
pub struct AckSignal<K: Send + 'static> {
    /// `None` after signal/cancel to suppress the drop warning.
    key: Option<K>,
    tx: Sender<K>,
}

impl<K: Send + 'static> AckSignal<K> {
    /// Signal completion.  Delivers `key` to the paired [`Receiver`].
    ///
    /// Safe to call from any thread.  Call after populating any data store
    /// the producer is waiting on, to guarantee the store is populated before
    /// the producer observes the signal.
    pub fn signal(mut self) {
        if let Some(key) = self.key.take() {
            let _ = self.tx.send(key);
        }
    }

    /// Explicitly cancel this signal without delivering a key.
    ///
    /// Use this when you decide not to prepare content (e.g. the tile was
    /// evicted before you processed the event).  The producer will not receive
    /// an acknowledgement - it must handle the absence of a signal gracefully
    /// (e.g. via a timeout or eviction path).
    pub fn cancel(mut self) {
        self.key = None;
        // drops silently - no delivery, no warning
    }
}

impl<K: Send + 'static> Drop for AckSignal<K> {
    fn drop(&mut self) {
        // If key is still Some, the signal was dropped without signal() or cancel().
        // We can't panic here (could be inside a panic unwind), so just log.
        if self.key.is_some() {
            // No std logger dependency - use eprintln for now.
            // Users will see the #[must_use] compiler warning before hitting this.
            #[cfg(debug_assertions)]
            eprintln!(
                "[orkester] AckSignal dropped without signal() or cancel() - \
                 the producer will never receive this acknowledgement"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_delivers_key() {
        let (factory, rx) = ack_channel::<u32>(4);
        let sig = factory.make(42u32);
        sig.signal();
        assert_eq!(rx.try_recv().unwrap(), 42);
    }

    #[test]
    fn cancel_delivers_nothing() {
        let (factory, rx) = ack_channel::<u32>(4);
        let sig = factory.make(42u32);
        sig.cancel();
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn multiple_signals_from_same_factory() {
        let (factory, rx) = ack_channel::<u32>(16);
        for i in 0..8 {
            factory.make(i).signal();
        }
        let mut received: Vec<u32> = Vec::new();
        while let Ok(v) = rx.try_recv() {
            received.push(v);
        }
        assert_eq!(received.len(), 8);
    }

    #[test]
    fn factory_is_clone() {
        let (factory, rx) = ack_channel::<u32>(4);
        let factory2 = factory.clone();
        factory.make(1).signal();
        factory2.make(2).signal();
        let mut received = [rx.try_recv().unwrap(), rx.try_recv().unwrap()];
        received.sort();
        assert_eq!(received, [1, 2]);
    }
}
