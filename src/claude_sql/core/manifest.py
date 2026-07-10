"""Self-describing manifest derived from the cyclopts command tree.

An agent driving ``claude-sql`` should learn the whole command surface from one
call instead of scraping ``--help`` per subcommand.  :func:`build_manifest`
walks a cyclopts :class:`~cyclopts.App` by introspection -- command names,
per-parameter types / choices / defaults / required-ness, the sub-app tree --
and folds in the stable :data:`~claude_sql.core.output.EXIT_CODES` contract and
the ``--format`` output conventions.  The result is a plain ``dict`` ready for
``json.dumps``; the ``manifest`` CLI command and the ``docs/reference`` doc
generator both render it, so neither can drift from the code.

Layering note: this lives in ``core`` (the bottom layer) and must not import
``claude_sql.app``.  The caller passes the assembled ``App`` in, so the
introspection stays here without an upward import.

cyclopts 4.21 introspection contract (verified against the installed version,
2026-07-11):

* ``for name in app`` yields command + sub-app names plus the ``--help`` /
  ``-h`` / ``--version`` meta flags, which we skip.
* ``app[name]`` returns the child ``App``; a leaf command has a
  ``default_command`` function, a sub-app has ``default_command is None`` and is
  itself iterable.
* ``app.name`` is a ``tuple[str, ...]`` -- take the first element.
* ``sub.assemble_argument_collection(parse_docstring=True)`` yields ``Argument``
  objects exposing ``.name`` (display name), ``.required`` (bool),
  ``.get_choices()`` (tuple for ``Literal`` / ``StrEnum``, else ``None``),
  ``.field_info.default``, ``.is_positional_only``, and ``.keys`` (non-empty for
  flags flattened from a shared dataclass such as ``Common``).  ``Argument`` has
  no ``.help`` attribute in 4.21 -- per-parameter help, when present, lives on
  ``.parameter.help``.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, cast

from claude_sql.core.output import EXIT_CODES, OutputFormat

if TYPE_CHECKING:
    from collections.abc import Callable

    from cyclopts import App

# Meta entries cyclopts injects into every ``App`` iteration; not real commands.
_META_NAMES = frozenset({"--help", "-h", "--version"})

# Parameter container names that are not user-facing flags.  ``*`` is the
# cyclopts placeholder for a flattened dataclass (here ``Common``); its member
# flags appear as their own ``Argument`` entries alongside it.
_CONTAINER_NAMES = frozenset({"*"})

# The shared ``Common`` flags flattened onto every subcommand.  We surface them
# once at the top level of the manifest and mark them per-command with a boolean
# instead of repeating seven identical entries under all 29 commands.
_GLOBAL_FLAG_NAMES = frozenset({"--verbose", "--quiet", "--glob", "--subagent-glob", "--format"})

API_VERSION = "1"


def _first_paragraph(doc: str | None) -> str:
    """Return the summary line of a docstring (text up to the first blank line)."""
    if not doc:
        return ""
    return inspect.cleandoc(doc).split("\n\n", 1)[0].strip()


def _app_name(app: App) -> str:
    """Return an ``App``'s display name (cyclopts stores it as a tuple)."""
    name = app.name
    if isinstance(name, tuple):
        return str(name[0]) if name else ""
    return str(name)


def _call_version_callable(version: object) -> str:
    """Invoke a cyclopts ``version`` callable with no arguments.

    cyclopts types ``App.version`` as ``None | str | Callable[..., str] |
    Callable[..., Coroutine[Any, Any, str]]`` (see ``cyclopts.core.App``); ty
    resolves the attrs-generated attribute to an unknown-signature callable, so
    the cast documents the contract cyclopts itself relies on internally
    (``App.version_print`` calls it with no arguments).

    ``claude-sql`` only ever sets a synchronous callable (``format_version``);
    guard the async-``Callable`` branch of that union explicitly rather than
    silently stringifying an un-awaited coroutine into manifest garbage.
    """
    result = cast("Callable[[], str]", version)()
    if inspect.iscoroutine(result):
        result.close()  # avoid a "coroutine was never awaited" leak warning
        raise TypeError(
            "App.version is an async callable; build_manifest only supports "
            "a synchronous version callable or a plain string."
        )
    return result


def _app_version(app: App) -> str | None:
    """Resolve an ``App``'s version to its first line.

    ``claude-sql`` sets ``version`` to a callable (``format_version``) that
    returns a multi-line banner including the local entrypoint path.  We call it
    and keep only the first line so the manifest carries the version string
    (``claude-sql 1.2.1``) without machine-specific paths that would make a
    committed doc non-deterministic.
    """
    version = app.version
    if version is None:
        return None
    resolved = _call_version_callable(version) if callable(version) else version
    return str(resolved).splitlines()[0].strip() if resolved else None


def _type_name(hint: Any) -> str:
    """Best-effort human name for a type hint (``str``, ``int``, ``Path``, ...)."""
    return getattr(hint, "__name__", None) or str(hint)


def _describe_parameter(arg: Any) -> dict[str, Any]:
    """Render one cyclopts ``Argument`` into a manifest parameter entry."""
    field_info = getattr(arg, "field_info", None)
    default = getattr(field_info, "default", inspect.Parameter.empty)
    has_default = default is not inspect.Parameter.empty
    choices = arg.get_choices()
    param_help = getattr(getattr(arg, "parameter", None), "help", None)
    # ``is_positional_only`` is a bound method in cyclopts 4.21, not a property,
    # so it must be called -- ``bool(<method>)`` is always truthy.
    positional_only = arg.is_positional_only
    positional = bool(positional_only() if callable(positional_only) else positional_only)
    return {
        "name": arg.name,
        "type": _type_name(arg.hint),
        "required": bool(arg.required),
        "positional": positional,
        "choices": list(choices) if choices else None,
        "default": default if has_default else None,
        "help": param_help or None,
    }


def _describe_command(name: str, sub: App) -> dict[str, Any]:
    """Render one leaf command into a manifest command entry."""
    fn = sub.default_command
    parameters: list[dict[str, Any]] = []
    for arg in sub.assemble_argument_collection(parse_docstring=True):
        if arg.name in _CONTAINER_NAMES or arg.name in _GLOBAL_FLAG_NAMES:
            # ``*`` is the Common container; the global flags are surfaced once
            # at the manifest top level (see ``global_flags``).
            continue
        parameters.append(_describe_parameter(arg))
    return {
        "name": name,
        "summary": _first_paragraph(inspect.getdoc(fn) if fn else None),
        "parameters": parameters,
        "supports_format": True,
    }


def _walk(app: App, prefix: str = "") -> list[dict[str, Any]]:
    """Recursively collect leaf commands, threading sub-app names into ``name``."""
    commands: list[dict[str, Any]] = []
    for name in app:
        if name in _META_NAMES:
            continue
        child = app[name]
        full = f"{prefix}{name}"
        if child.default_command is None:
            # A sub-app (e.g. ``cache`` / ``skills``): recurse; its leaves get a
            # space-joined name like ``cache compact``.
            commands.extend(_walk(child, prefix=f"{full} "))
        else:
            commands.append(_describe_command(full, child))
    return commands


def _global_flags(app: App) -> list[dict[str, Any]]:
    """Describe the shared ``Common`` flags once, read off any real command.

    Every subcommand flattens the same ``Common`` dataclass, so we introspect
    the first leaf command and pull out the entries whose names are in
    :data:`_GLOBAL_FLAG_NAMES`.
    """
    for name in app:
        if name in _META_NAMES:
            continue
        child = app[name]
        if child.default_command is None:
            nested = _global_flags(child)
            if nested:
                return nested
            continue
        flags = [
            _describe_parameter(arg)
            for arg in child.assemble_argument_collection(parse_docstring=True)
            if arg.name in _GLOBAL_FLAG_NAMES
        ]
        if flags:
            return flags
    return []


def build_manifest(app: App) -> dict[str, Any]:
    """Build the machine-readable manifest for ``app`` by introspection.

    Parameters
    ----------
    app
        The assembled cyclopts ``App`` (passed in so ``core`` need not import
        ``claude_sql.app``).

    Returns
    -------
    dict
        ``{apiVersion, cli, version, summary, commands, global_flags,
        output_formats, exit_codes, conventions}`` -- JSON-serializable.
    """
    return {
        "apiVersion": API_VERSION,
        "cli": _app_name(app),
        "version": _app_version(app),
        "summary": _first_paragraph(app.help),
        "commands": _walk(app),
        "global_flags": _global_flags(app),
        "output_formats": [fmt.value for fmt in OutputFormat],
        "exit_codes": dict(EXIT_CODES),
        "conventions": [
            "Every command accepts --format {auto,table,json,ndjson,csv}; "
            "auto emits a human table on a TTY and JSON on a pipe.",
            "Flags attach to the subcommand, not the binary "
            "(claude-sql query --format json 'SELECT 1').",
            "Errors carry a stable exit code (see exit_codes); non-TTY stderr "
            'carries a JSON {"error": {"kind", "message", "hint"}} payload.',
            "Cost-incurring commands (embed, classify, trajectory, conflicts, "
            "friction, analyze) default to --dry-run; pass --no-dry-run to spend.",
        ],
    }
