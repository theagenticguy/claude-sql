"""Loguru configuration for claude-sql.

Single ``configure_logging()`` helper that removes the default loguru handler
and re-installs a stderr handler honoring ``--verbose`` / ``--quiet`` flags and
``LOGURU_LEVEL``.

Also provides ``loguru_before_sleep`` so workers can pass tenacity a
loguru-native callback (instead of stdlib ``logging.getLogger`` +
``before_sleep_log``). Keeps the package single-logger across the board.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from tenacity import RetryCallState

_FORMAT = "<green>{time:HH:mm:ss}</green> <level>{level:<7}</level> {extra} {message}"


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:  # noqa: FBT001, FBT002 — CLI flag pass-through
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


def loguru_before_sleep(level: str = "WARNING") -> Callable[[RetryCallState], None]:
    """Return a tenacity ``before_sleep`` callback that logs via loguru.

    Replaces the historical ``before_sleep_log(stdlib_logger, level)`` shape
    that pulled stdlib ``logging`` into otherwise loguru-only modules just
    to satisfy tenacity's API. The format string mirrors tenacity's own
    :func:`tenacity.before_sleep_log`: function name, sleep seconds, and
    the exception or returned value that triggered the retry.

    Parameters
    ----------
    level
        Loguru level name (``"DEBUG" | "INFO" | "WARNING" | …``).

    Returns
    -------
    Callable
        A function with the tenacity ``before_sleep`` signature
        ``(RetryCallState) -> None``.
    """

    def _before_sleep(retry_state: RetryCallState) -> None:
        if retry_state.outcome is None or retry_state.next_action is None:
            return
        if retry_state.fn is None:
            fn_name = "<unknown>"
        else:
            fn_name = getattr(retry_state.fn, "__qualname__", repr(retry_state.fn))
        if retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            verb, value = "raised", f"{exc.__class__.__name__}: {exc}"
        else:
            verb, value = "returned", retry_state.outcome.result()
        logger.log(
            level,
            "Retrying {} in {:.3g} seconds as it {} {}.",
            fn_name,
            retry_state.next_action.sleep,
            verb,
            value,
        )

    return _before_sleep
