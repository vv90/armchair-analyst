//! A coalescing single-cell mailbox: a one-slot channel that keeps only the **latest** value sent.
//! This is the client-side port of aa-cli's `latest_slot` — the transport that feeds fresh reserve
//! snapshots to the self-clocked optimizer worker ([`crate::optimizer::run`]).
//!
//! Unlike an unbounded FIFO, a producer that outruns the consumer never piles up work: each
//! [`LatestSender::send`] overwrites the pending value, so the consumer always takes the newest
//! snapshot and older un-taken ones are dropped. That is exactly the coalescing the optimizer needs —
//! it should grind the freshest reserves, not a backlog of stale ones. [`LatestReceiver::wait_take`]
//! blocks until a value is present (or the slot closes); [`LatestReceiver::try_take`] returns
//! immediately, which is how the worker distinguishes "fresh reserves arrived" (→ step them) from
//! "nothing new" (→ keep grinding the current session).
//!
//! Either end closing (explicitly or on `Drop`) wakes the other, so a torn-down runtime or a finished
//! worker unblocks its counterpart cleanly. Everything is panic-free: a poisoned lock degrades to a
//! typed error rather than unwinding.

use std::sync::{Arc, Condvar, Mutex};

/// Create a coalescing slot, returning its sender and receiver halves. Both share one `Arc`-backed
/// cell; dropping either half closes the slot for the other.
pub(crate) fn latest_slot<T>() -> (LatestSender<T>, LatestReceiver<T>) {
    let inner = Arc::new(LatestSlot {
        state: Mutex::new(LatestSlotState {
            latest: None,
            closed: false,
        }),
        changed: Condvar::new(),
    });

    (
        LatestSender {
            inner: inner.clone(),
        },
        LatestReceiver { inner },
    )
}

/// Why a [`LatestSender::send`] could not deliver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LatestSendError {
    /// The receiver half has been dropped / the slot was closed.
    Closed,
    /// The slot's lock was poisoned by a panic elsewhere.
    Poisoned,
}

/// Why a [`LatestReceiver`] take could not produce a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LatestReceiveError {
    /// The sender half has been dropped / the slot was closed and is empty.
    Closed,
    /// The slot's lock was poisoned by a panic elsewhere.
    Poisoned,
}

/// The producing half. Sends overwrite the pending value (latest-wins).
pub(crate) struct LatestSender<T> {
    inner: Arc<LatestSlot<T>>,
}

/// The consuming half. Takes remove and return the pending value.
pub(crate) struct LatestReceiver<T> {
    inner: Arc<LatestSlot<T>>,
}

struct LatestSlot<T> {
    state: Mutex<LatestSlotState<T>>,
    changed: Condvar,
}

struct LatestSlotState<T> {
    latest: Option<T>,
    closed: bool,
}

impl<T> LatestSender<T> {
    /// Place `value` in the slot, overwriting any un-taken previous value, and wake a waiting
    /// receiver. Errors only if the slot is closed or its lock is poisoned.
    pub(crate) fn send(&self, value: T) -> Result<(), LatestSendError> {
        match self.inner.state.lock() {
            Ok(state) if state.closed => Err(LatestSendError::Closed),
            Ok(mut state) => {
                state.latest = Some(value);
                self.inner.changed.notify_one();
                Ok(())
            }
            Err(_) => Err(LatestSendError::Poisoned),
        }
    }
}

impl<T> LatestReceiver<T> {
    /// Block until a value is present, returning it and clearing the slot. Returns
    /// [`LatestReceiveError::Closed`] if the slot closes while empty.
    pub(crate) fn wait_take(&self) -> Result<T, LatestReceiveError> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| LatestReceiveError::Poisoned)?;

        loop {
            if let Some(value) = state.latest.take() {
                return Ok(value);
            }

            if state.closed {
                return Err(LatestReceiveError::Closed);
            }

            state = self
                .inner
                .changed
                .wait(state)
                .map_err(|_| LatestReceiveError::Poisoned)?;
        }
    }

    /// Take the pending value if one is present without blocking: `Ok(Some(_))` when a value was
    /// waiting, `Ok(None)` when the slot is (currently) empty but open, `Err(Closed)` when the slot is
    /// empty and closed.
    pub(crate) fn try_take(&self) -> Result<Option<T>, LatestReceiveError> {
        match self.inner.state.lock() {
            Ok(mut state) => {
                if let Some(value) = state.latest.take() {
                    Ok(Some(value))
                } else if state.closed {
                    Err(LatestReceiveError::Closed)
                } else {
                    Ok(None)
                }
            }
            Err(_) => Err(LatestReceiveError::Poisoned),
        }
    }
}

impl<T> LatestSlot<T> {
    fn close(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.closed = true;
            self.changed.notify_all();
        }
    }
}

impl<T> Drop for LatestSender<T> {
    fn drop(&mut self) {
        self.inner.close();
    }
}

impl<T> Drop for LatestReceiver<T> {
    fn drop(&mut self) {
        self.inner.close();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    use super::*;

    #[test]
    fn wait_take_returns_sent_value() {
        let (sender, receiver) = latest_slot();
        sender.send(1).expect("send");
        assert_eq!(receiver.wait_take(), Ok(1));
    }

    #[test]
    fn multiple_sends_keep_only_latest_value() {
        let (sender, receiver) = latest_slot();
        sender.send(1).expect("send 1");
        sender.send(2).expect("send 2");
        sender.send(3).expect("send 3");
        // Coalescing: only the newest survives; nothing is queued behind it.
        assert_eq!(receiver.wait_take(), Ok(3));
        assert_eq!(receiver.try_take(), Ok(None));
    }

    #[test]
    fn try_take_returns_pending_value_once() {
        let (sender, receiver) = latest_slot();
        sender.send(1).expect("send");
        assert_eq!(receiver.try_take(), Ok(Some(1)));
        assert_eq!(receiver.try_take(), Ok(None));
    }

    #[test]
    fn wait_take_blocks_until_value_is_sent() {
        let (sender, receiver) = latest_slot();
        let (ready_sender, ready_receiver) = mpsc::channel();
        let started_at = Instant::now();

        let handle = thread::spawn(move || {
            ready_sender.send(()).expect("signal ready");
            receiver.wait_take()
        });

        ready_receiver
            .recv_timeout(Duration::from_millis(250))
            .expect("worker ready");
        thread::sleep(Duration::from_millis(10));
        sender.send(7).expect("send");

        assert_eq!(handle.join().expect("join"), Ok(7));
        assert!(started_at.elapsed() >= Duration::from_millis(10));
    }

    #[test]
    fn close_wakes_waiting_receiver() {
        let (sender, receiver) = latest_slot::<u8>();
        let (ready_sender, ready_receiver) = mpsc::channel();

        let handle = thread::spawn(move || {
            ready_sender.send(()).expect("signal ready");
            receiver.wait_take()
        });

        ready_receiver
            .recv_timeout(Duration::from_millis(250))
            .expect("worker ready");
        // Dropping the sender closes the slot, waking the blocked receiver with Closed.
        drop(sender);

        assert_eq!(
            handle.join().expect("join"),
            Err(LatestReceiveError::Closed)
        );
    }

    #[test]
    fn send_after_receiver_dropped_returns_closed_error() {
        let (sender, receiver) = latest_slot();
        drop(receiver);
        assert_eq!(sender.send(1), Err(LatestSendError::Closed));
    }
}
