"""Configures the application's logging for dual output.

This module is responsible for setting up the root logger to send logs
to two distinct destinations: a user-friendly, colorized terminal output
that suppresses stack traces, and a detailed file output that includes
full tracebacks for debugging purposes.

Classes:
    CustomTerminalFormatter: A custom formatter for clean, user-friendly
                             terminal output that suppresses tracebacks.

Functions:
    setup_logging: Configures the root logger with the dual-handler setup.

"""
# Copyright 2025 Gabriel Carvalho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
from pathlib import Path

import colorlog


class CustomTerminalFormatter(colorlog.ColoredFormatter):
    """Custom log formatter for clean terminal output.

    This formatter is designed to provide a user-friendly console experience.
    INFO level logs are displayed as clean messages, while other levels
    (WARNING, ERROR) are prefixed with their colored level names.
    Crucially, it guarantees that no tracebacks are ever printed to the
    console.

    Attributes:
        info_formatter: A specific formatter for INFO-level logs.

    """

    def __init__(self) -> None:
        """Initialize the CustomTerminalFormatter."""
        info_format = "%(message)s"
        other_format = "%(log_color)s%(levelname)s:%(reset)s %(message)s"
        super().__init__(
            fmt=other_format,
            log_colors={
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        self.info_formatter = colorlog.ColoredFormatter(info_format)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record, suppressing console stack traces.

        This method temporarily removes exception info from the record before
        formatting, ensuring no traceback is printed to the console handler.
        The exception info is restored afterward, allowing other handlers (like
        a file handler) to still log the full traceback.

        Args:
            record: The log record to be formatted.

        Returns:
            The formatted string for the console.

        """
        original_exc_info = record.exc_info
        original_exc_text = record.exc_text

        record.exc_info = None
        record.exc_text = None

        try:
            if record.levelno == logging.INFO:
                return self.info_formatter.format(record)
            return super().format(record)
        finally:
            record.exc_info = original_exc_info
            record.exc_text = original_exc_text


def setup_logging(log_file_path: Path, quiet_level: int) -> None:
    """Configure the root logger for dual-handler output.

    This function clears any existing handlers and sets up two new ones:
    - A console handler with a custom formatter for clean, user-facing logs.
    - A file handler that logs detailed information for debugging.

    The verbosity of the console handler is controlled by the quiet_level.

    Args:
        log_file_path: The destination path for the detailed log file.
        quiet_level: An integer controlling console verbosity
                    (0=info, 1=warning, 2+=silent).

    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    logging.getLogger("content_summarizer").setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    console_handler = colorlog.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    if quiet_level == 1:
        console_handler.setLevel(logging.WARNING)
    if quiet_level >= 2:
        console_handler.setLevel(logging.CRITICAL + 1)
    console_handler.setFormatter(CustomTerminalFormatter())

    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")

    file_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
