use std::{
    fs::{self, File},
    io::{self, Write},
    path::Path,
    sync::mpsc::{self, Sender},
    thread::{self, JoinHandle},
    time::{SystemTime, UNIX_EPOCH},
};

/// Non-blocking, channel-backed logger.
///
/// `log` only sends a line over a channel; a dedicated writer thread owns the
/// file and performs all I/O, so no runtime or effect thread ever blocks on disk.
#[derive(Clone)]
pub(crate) struct Logger {
    sender: Sender<String>,
    /// Wall-clock millis when the run started; every line is stamped with its offset from here so
    /// latency can be read straight off the log. The absolute base is in the `run started_millis=…`
    /// header.
    started_millis: u128,
}

impl Logger {
    /// Opens a distinctly-named log file for this run under `logs/` and spawns
    /// the writer thread that drains logged lines to it.
    pub(crate) fn create_for_run() -> io::Result<Logger> {
        create_in(Path::new("logs"))
    }

    /// A logger that discards everything it is given. Used by tests.
    pub(crate) fn sink() -> Logger {
        let (sender, _receiver) = mpsc::channel();
        Logger {
            sender,
            started_millis: 0,
        }
    }

    /// Enqueues a line for the writer thread, prefixed with `t=<millis since run start>` so the log
    /// carries timing. Never blocks and never panics; if the writer is gone the line is dropped.
    pub(crate) fn log(&self, line: &str) {
        let offset = unix_millis().saturating_sub(self.started_millis);
        let _ = self.sender.send(format!("t={offset} {line}"));
    }
}

fn create_in(dir: &Path) -> io::Result<Logger> {
    fs::create_dir_all(dir)?;

    let started_millis = unix_millis();
    let pid = std::process::id();
    let path = dir.join(format!("aa-cli-{started_millis}-{pid}.log"));

    let mut file = File::create(&path)?;
    writeln!(file, "run started_millis={started_millis} pid={pid}")?;

    let (sender, _handle) = spawn_writer(file);

    Ok(Logger {
        sender,
        started_millis,
    })
}

fn spawn_writer(mut file: File) -> (Sender<String>, JoinHandle<()>) {
    let (sender, receiver) = mpsc::channel::<String>();
    let handle = thread::spawn(move || {
        for line in receiver {
            let _ = writeln!(file, "{line}");
            let _ = file.flush();
        }
    });

    (sender, handle)
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    #[test]
    fn sink_logger_log_is_noop_and_does_not_panic() {
        let logger = Logger::sink();

        logger.log("discarded");
    }

    #[test]
    fn log_forwards_lines_to_the_channel_in_order() {
        let (sender, receiver) = mpsc::channel();
        let logger = Logger {
            sender,
            started_millis: 0,
        };

        logger.log("first");
        logger.log("second");
        drop(logger);

        let lines = receiver.iter().collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        assert!(
            lines[0].starts_with("t="),
            "missing timestamp: {}",
            lines[0]
        );
        assert!(
            lines[0].ends_with(" first"),
            "unexpected line: {}",
            lines[0]
        );
        assert!(
            lines[1].ends_with(" second"),
            "unexpected line: {}",
            lines[1]
        );
    }

    #[test]
    fn log_stamps_each_line_with_a_non_decreasing_offset() {
        let (sender, receiver) = mpsc::channel();
        let logger = Logger {
            sender,
            started_millis: 0,
        };

        logger.log("first");
        logger.log("second");
        drop(logger);

        let offsets = receiver
            .iter()
            .map(|line| {
                line.strip_prefix("t=")
                    .and_then(|rest| rest.split_whitespace().next())
                    .and_then(|millis| millis.parse::<u128>().ok())
                    .expect("each line carries a numeric t= offset")
            })
            .collect::<Vec<_>>();

        assert_eq!(offsets.len(), 2);
        assert!(
            offsets[1] >= offsets[0],
            "offsets went backwards: {offsets:?}"
        );
    }

    #[test]
    fn writer_thread_writes_each_line_to_the_file() -> io::Result<()> {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir)?;
        let path = dir.join("writer.log");
        let file = File::create(&path)?;

        let (sender, handle) = spawn_writer(file);
        sender.send("alpha".to_owned()).unwrap();
        sender.send("beta".to_owned()).unwrap();
        drop(sender);
        handle.join().unwrap();

        assert_eq!(fs::read_to_string(&path)?, "alpha\nbeta\n");

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn create_in_writes_a_distinct_run_file_with_header() -> io::Result<()> {
        let dir = unique_temp_dir();

        let _logger = create_in(&dir)?;

        let entries = fs::read_dir(&dir)?.collect::<Result<Vec<_>, _>>()?;
        assert_eq!(entries.len(), 1);
        let name = entries[0].file_name().into_string().unwrap();
        assert!(name.starts_with("aa-cli-"), "unexpected name: {name}");
        assert!(name.ends_with(".log"), "unexpected name: {name}");
        // The header is written synchronously before the writer thread spawns.
        assert!(fs::read_to_string(entries[0].path())?.starts_with("run started_millis="));

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    fn unique_temp_dir() -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nonce = COUNTER.fetch_add(1, Ordering::Relaxed);

        std::env::temp_dir().join(format!(
            "aa-cli-logger-test-{}-{}",
            std::process::id(),
            nonce
        ))
    }
}
