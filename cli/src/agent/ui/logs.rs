//! The log, captured for the chat screen: each `tracing` line goes to the UI instead of stderr, which the screen
//! owns while it runs.

use std::io::Write;

use tokio::sync::mpsc;
use tracing::Level;
use tracing_subscriber::fmt::MakeWriter;

/// One formatted log line.
#[derive(Clone, Debug)]
pub(crate) struct LogLine {
    pub(crate) level: Level,
    pub(crate) text: String,
}

/// Where the chat's log goes: a `tracing_subscriber` writer whose lines arrive at [`Logs`].
#[derive(Clone)]
pub(crate) struct LogSink(mpsc::UnboundedSender<LogLine>);

/// The other end: log lines waiting to be shown.
pub(crate) struct Logs(pub(crate) mpsc::UnboundedReceiver<LogLine>);

pub(crate) fn channel() -> (LogSink, Logs) {
    let (sink, logs) = mpsc::unbounded_channel();
    (LogSink(sink), Logs(logs))
}

impl Logs {
    /// Prints what has not been shown to stderr, for the lines that come after the screen is gone (the prompt
    /// cache being saved, say).
    pub(crate) fn print_rest(&mut self) {
        while let Ok(line) = self.0.try_recv() {
            eprintln!("{}", line.text);
        }
    }
}

/// Collects one event's formatted text and sends it when the formatter is done with it.
pub(crate) struct LineWriter {
    sink: mpsc::UnboundedSender<LogLine>,
    level: Level,
    text: Vec<u8>,
}

impl Write for LineWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.text.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for LineWriter {
    fn drop(&mut self) {
        let text = String::from_utf8_lossy(&self.text);
        let text = text.trim();
        if !text.is_empty() {
            // Once the chat has ended nobody is listening, which is fine.
            let _ = self.sink.send(LogLine { level: self.level, text: text.to_owned() });
        }
    }
}

impl<'a> MakeWriter<'a> for LogSink {
    type Writer = LineWriter;

    fn make_writer(&'a self) -> LineWriter {
        LineWriter { sink: self.0.clone(), level: Level::INFO, text: Vec::new() }
    }

    fn make_writer_for(&'a self, meta: &tracing::Metadata<'_>) -> LineWriter {
        LineWriter { sink: self.0.clone(), level: *meta.level(), text: Vec::new() }
    }
}
