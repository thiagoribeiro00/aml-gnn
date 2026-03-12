"""
Structured logging adapter for terminal and cloud debugging.
Supports JSON format (cloud) and human-readable format (terminal).
No emojis in any log output.
"""

import logging
import json
import time
import uuid
import functools
from typing import Optional
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """JSON log formatter for cloud environments (GCP/AWS compatible)."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id

        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        # Capture extra fields
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "timestamp", "severity", "logger",
            "line", "duration_ms", "correlation_id"
        }
        
        extra_data = {k: v for k, v in record.__dict__.items() if k not in standard_attrs}
        if extra_data:
            log_entry["data"] = extra_data

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for terminal debugging."""

    LEVEL_COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, self.RESET)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        parts = [
            f"{color}[{ts}]{self.RESET}",
            f"{color}[{record.levelname:>8s}]{self.RESET}",
            f"[{record.name}]",
            record.getMessage(),
        ]

        if hasattr(record, "correlation_id"):
            parts.insert(3, f"[cid:{record.correlation_id[:8]}]")

        if hasattr(record, "duration_ms"):
            parts.append(f"(took {record.duration_ms:.1f}ms)")

        # Display non-standard attributes
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "duration_ms", "correlation_id"
        }
        extra_data = {k: v for k, v in record.__dict__.items() if k not in standard_attrs}
        if extra_data:
            parts.append(f"| {extra_data}")

        msg = " ".join(parts)

        if record.exc_info and record.exc_info[1]:
            msg += f"\n  -> Exception: {type(record.exc_info[1]).__name__}: {record.exc_info[1]}"

        return msg


class StructuredLogger:
    """
    Logger factory that produces structured loggers with correlation ID support.
    """

    _configured = False
    _environment = "dev"

    @classmethod
    def configure(cls, level: str = "INFO", environment: str = "dev") -> None:
        """
        Configure root logging. Call once at application startup.
        """
        if cls._configured:
            return

        cls._environment = environment
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Remove existing handlers
        root_logger.handlers.clear()

        handler = logging.StreamHandler()
        if environment == "prod":
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(ConsoleFormatter())

        root_logger.addHandler(handler)
        cls._configured = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance. Auto-configures on first call if needed."""
        if not cls._configured:
            cls.configure()
        return logging.getLogger(name)

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a unique correlation ID for tracing operations."""
        return str(uuid.uuid4())


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)
            start_time = time.perf_counter()
            _logger.debug(
                "Starting: %s",
                func.__qualname__,
            )
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                _logger.info(
                    "Completed: %s",
                    func.__qualname__,
                    extra={"duration_ms": elapsed_ms},
                )
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                _logger.error(
                    "Failed: %s - %s: %s",
                    func.__qualname__,
                    type(e).__name__,
                    str(e),
                    extra={"duration_ms": elapsed_ms},
                )
                raise
        return wrapper
    return decorator


class MetricsCollector:
    """Simple in-memory metrics collector for pipeline observability."""

    def __init__(self):
        self._metrics: dict = {}
        self._logger = StructuredLogger.get_logger("metrics")

    def record(self, name: str, value: float, unit: str = "") -> None:
        """Record a numeric metric."""
        self._metrics[name] = {"value": value, "unit": unit}
        self._logger.debug(
            "Metric recorded: %s = %s %s",
            name, value, unit,
        )

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        if name not in self._metrics:
            self._metrics[name] = {"value": 0, "unit": "count"}
        self._metrics[name]["value"] += amount

    def get(self, name: str) -> Optional[float]:
        """Get a metric value."""
        entry = self._metrics.get(name)
        return entry["value"] if entry else None

    def summary(self) -> dict:
        """Return all collected metrics."""
        return dict(self._metrics)

    def log_summary(self) -> None:
        """Log all metrics as a structured summary."""
        self._logger.info(
            "Metrics summary",
            extra={"metrics": self._metrics},
        )


# Global metrics instance
metrics = MetricsCollector()
