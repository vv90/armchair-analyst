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
        Logger { sender }
    }

    /// Enqueues a line for the writer thread. Never blocks and never panics;
    /// if the writer is gone the line is silently dropped.
    pub(crate) fn log(&self, line: &str) {
        let _ = self.sender.send(line.to_owned());
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

    Ok(Logger { sender })
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
        let logger = Logger { sender };

        logger.log("first");
        logger.log("second");
        drop(logger);

        let lines = receiver.iter().collect::<Vec<_>>();
        assert_eq!(lines, vec!["first".to_owned(), "second".to_owned()]);
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
        assert!(
            fs::read_to_string(entries[0].path())?.starts_with("run started_millis=")
        );

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
