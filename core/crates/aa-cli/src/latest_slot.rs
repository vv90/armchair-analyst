use std::sync::{Arc, Condvar, Mutex};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LatestSendError {
    Closed,
    Poisoned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LatestReceiveError {
    Closed,
    Poisoned,
}

pub(crate) struct LatestSender<T> {
    inner: Arc<LatestSlot<T>>,
}

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

    pub(crate) fn close(&self) -> Result<(), LatestSendError> {
        self.inner.close().map_err(|_| LatestSendError::Poisoned)
    }
}

impl<T> LatestReceiver<T> {
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
    fn close(&self) -> Result<(), ()> {
        let mut state = self.state.lock().map_err(|_| ())?;

        state.closed = true;
        self.changed.notify_all();

        Ok(())
    }
}

impl<T> Drop for LatestSender<T> {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

impl<T> Drop for LatestReceiver<T> {
    fn drop(&mut self) {
        let _ = self.inner.close();
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };

    use super::*;

    #[test]
    fn wait_take_returns_sent_value() {
        let (sender, receiver) = latest_slot();

        sender.send(1).unwrap();

        assert_eq!(receiver.wait_take(), Ok(1));
    }

    #[test]
    fn multiple_sends_keep_only_latest_value() {
        let (sender, receiver) = latest_slot();

        sender.send(1).unwrap();
        sender.send(2).unwrap();
        sender.send(3).unwrap();

        assert_eq!(receiver.wait_take(), Ok(3));
        assert_eq!(receiver.try_take(), Ok(None));
    }

    #[test]
    fn try_take_returns_pending_value_once() {
        let (sender, receiver) = latest_slot();

        sender.send(1).unwrap();

        assert_eq!(receiver.try_take(), Ok(Some(1)));
        assert_eq!(receiver.try_take(), Ok(None));
    }

    #[test]
    fn wait_take_blocks_until_value_is_sent() {
        let (sender, receiver) = latest_slot();
        let (ready_sender, ready_receiver) = mpsc::channel();
        let started_at = Instant::now();

        let handle = thread::spawn(move || {
            ready_sender.send(()).unwrap();
            receiver.wait_take()
        });

        ready_receiver
            .recv_timeout(Duration::from_millis(250))
            .unwrap();
        thread::sleep(Duration::from_millis(10));
        sender.send(7).unwrap();

        assert_eq!(handle.join().unwrap(), Ok(7));
        assert!(started_at.elapsed() >= Duration::from_millis(10));
    }

    #[test]
    fn close_wakes_waiting_receiver() {
        let (sender, receiver) = latest_slot::<u8>();
        let (ready_sender, ready_receiver) = mpsc::channel();

        let handle = thread::spawn(move || {
            ready_sender.send(()).unwrap();
            receiver.wait_take()
        });

        ready_receiver
            .recv_timeout(Duration::from_millis(250))
            .unwrap();
        sender.close().unwrap();

        assert_eq!(handle.join().unwrap(), Err(LatestReceiveError::Closed));
    }

    #[test]
    fn send_after_close_returns_closed_error() {
        let (sender, _receiver) = latest_slot();

        sender.close().unwrap();

        assert_eq!(sender.send(1), Err(LatestSendError::Closed));
    }
}
