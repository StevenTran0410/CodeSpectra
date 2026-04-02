import logging
import os
import sys
from pathlib import Path

LOG_LEVEL = logging.DEBUG if os.getenv("CODESPECTRA_ENV") == "development" else logging.INFO


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("codespectra")
    logger.setLevel(LOG_LEVEL)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console handler (captured by Electron as stderr)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File handler (optional — only when data dir is set)
    data_dir = os.getenv("CODESPECTRA_DATA_DIR")
    if data_dir:
        log_path = Path(data_dir) / "logs" / "backend.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


logger = _build_logger()
