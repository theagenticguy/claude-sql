"""Loguru configuration for claude-sql CLI.

Single ``configure_logging()`` helper that removes the default loguru handler
and re-installs a stderr handler honoring ``--verbose`` / ``--quiet`` flags and
``LOGURU_LEVEL``.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

_FORMAT = "<green>{time:HH:mm:ss}</green> <level>{level:<7}</level> {extra} {message}"


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Install the stderr loguru handler for claude-sql.

    Parameters
    ----------
    verbose
        If true, set level to ``DEBUG`` (takes precedence over ``quiet``).
    quiet
        If true, set level to ``ERROR``.
    """
    logger.remove()
    if verbose:
        level = "DEBUG"
    elif quiet:
        level = "ERROR"
    else:
        level = os.getenv("LOGURU_LEVEL", "INFO")
    logger.add(
        sys.stderr,
        level=level,
        format=_FORMAT,
        backtrace=False,
        diagnose=False,
    )
