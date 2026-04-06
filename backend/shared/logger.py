import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_LEVEL = logging.DEBUG if os.getenv("CODESPECTRA_ENV") == "development" else logging.INFO

# Patterns that should never appear in log files.
# Each tuple is (compiled_pattern, replacement_string).
_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Bearer / token auth headers
    (re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), r"\1[REDACTED]"),
    # API key in JSON / query string / kwargs
    (re.compile(r"(\"?api[_-]?key\"?\s*[:=]\s*[\"']?)[A-Za-z0-9\-_]{8,}([\"']?)", re.IGNORECASE), r"\1[REDACTED]\2"),
    # OpenAI-style keys (sk-...)
    (re.compile(r"(sk-[A-Za-z0-9]{8})[A-Za-z0-9\-_]+"), r"\1[REDACTED]"),
    # Generic long hex/base64 that looks like a token (40+ chars after = or :)
    (re.compile(r"(password|token|secret|key)\s*[:=]\s*[\"']?[A-Za-z0-9+/\-_]{20,}[\"']?", re.IGNORECASE), r"\1=[REDACTED]"),
]


class _RedactingFormatter(logging.Formatter):
    """Formatter that scrubs credential patterns from log messages."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        for pattern, replacement in _REDACT_PATTERNS:
            msg = pattern.sub(replacement, msg)
        return msg


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("codespectra")
    logger.setLevel(LOG_LEVEL)

    if logger.handlers:
        return logger

    fmt = _RedactingFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console handler (captured by Electron as stderr)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Rotating file handler — max 5 MB × 3 files = 15 MB total
    data_dir = os.getenv("CODESPECTRA_DATA_DIR")
    if data_dir:
        log_path = Path(data_dir) / "logs" / "backend.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path,
            encoding="utf-8",
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


logger = _build_logger()
