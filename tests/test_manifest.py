"""Tests for ``claude_sql.core.manifest`` -- the cyclopts introspection walk.

These exercise :func:`build_manifest` against the *real* ``claude-sql`` app
(not a toy fixture app) so a signature change that breaks the introspection
walk fails here instead of silently shipping a hollowed-out manifest.
"""

from __future__ import annotations

import json

from cyclopts import App

from claude_sql.app.cli import app
from claude_sql.core.manifest import _global_flags, build_manifest
from claude_sql.core.output import EXIT_CODES, OutputFormat


def test_build_manifest_is_json_serializable() -> None:
    manifest = build_manifest(app)
    # The real emit_json path is json.dumps(payload, indent=2, default=str).
    json.dumps(manifest, indent=2, default=str)


def test_build_manifest_is_deterministic() -> None:
    first = json.dumps(build_manifest(app), default=str)
    second = json.dumps(build_manifest(app), default=str)
    assert first == second


def test_build_manifest_top_level_shape() -> None:
    manifest = build_manifest(app)
    assert manifest["apiVersion"] == "1"
    assert manifest["cli"] == "claude-sql"
    assert isinstance(manifest["version"], str)
    assert manifest["version"].startswith("claude-sql ")
    # format_version()'s multi-line banner (entrypoint path, "installed from
    # source") must not leak into the manifest -- only the version line.
    assert "\n" not in manifest["version"]
    assert manifest["output_formats"] == [fmt.value for fmt in OutputFormat]
    assert manifest["exit_codes"] == EXIT_CODES


def test_build_manifest_covers_every_command_including_subapps() -> None:
    manifest = build_manifest(app)
    names = {c["name"] for c in manifest["commands"]}
    # Root commands, spanning read/ingest/eval/provenance groups.
    assert {"schema", "query", "explain", "kappa", "bind", "resolve", "manifest"} <= names
    # Sub-app leaves get a space-joined name, not a bare "compact"/"ls".
    assert {"cache compact", "cache migrate", "skills sync", "skills ls"} <= names
    # No sub-app itself (as opposed to its leaves) should appear as a command.
    assert "cache" not in names
    assert "skills" not in names


def test_build_manifest_omits_global_flags_and_container_from_commands() -> None:
    manifest = build_manifest(app)
    leaked = {
        (cmd["name"], p["name"])
        for cmd in manifest["commands"]
        for p in cmd["parameters"]
        if p["name"] in {"--verbose", "--quiet", "--glob", "--subagent-glob", "--format", "*"}
    }
    assert not leaked
    global_names = {f["name"] for f in manifest["global_flags"]}
    assert global_names == {"--verbose", "--quiet", "--glob", "--subagent-glob", "--format"}


def test_build_manifest_query_command_parameters() -> None:
    manifest = build_manifest(app)
    by_name = {c["name"]: c for c in manifest["commands"]}
    query = by_name["query"]
    params = {p["name"]: p for p in query["parameters"]}
    assert params["SQL"]["required"] is True
    assert params["SQL"]["positional"] is True
    assert params["--profile-json"]["required"] is False
    assert params["--profile-json"]["positional"] is False
    assert params["--profile-json"]["default"] is False


def test_build_manifest_captures_enum_choices() -> None:
    manifest = build_manifest(app)
    format_flag = next(f for f in manifest["global_flags"] if f["name"] == "--format")
    assert format_flag["choices"] == ["auto", "table", "json", "ndjson", "csv"]
    assert format_flag["default"] == OutputFormat.AUTO


def test_build_manifest_required_keyword_only_parameter() -> None:
    # ``freeze --panel`` is keyword-only with no default -- required=True.
    manifest = build_manifest(app)
    by_name = {c["name"]: c for c in manifest["commands"]}
    panel = next(p for p in by_name["freeze"]["parameters"] if p["name"] == "--panel")
    assert panel["required"] is True
    assert panel["default"] is None


def test_build_manifest_zero_param_command_has_empty_parameter_list() -> None:
    # ``manifest`` itself takes only the flattened ``Common`` container -- once
    # the global flags and the ``*`` container are stripped, nothing should be
    # left. Pins the filter in ``_describe_command`` against a regression that
    # would leak ``common``/``*`` into a command with no real parameters.
    manifest = build_manifest(app)
    by_name = {c["name"]: c for c in manifest["commands"]}
    assert by_name["manifest"]["parameters"] == []
    assert by_name["shell"]["parameters"] == []


def test_global_flags_falls_through_to_empty_list_when_none_match() -> None:
    # A synthetic app with a leaf command that carries no Common-style flags at
    # all -- exercises the ``return []`` fallthrough that is a dead path on the
    # real app (every real leaf carries the flattened ``Common`` flags).
    toy = App(name="toy")

    @toy.command
    def leaf(x: int = 1) -> None:
        """A leaf with no --verbose/--quiet/--glob/--subagent-glob/--format."""

    assert _global_flags(toy) == []


def test_global_flags_recurses_into_sub_app_to_find_flags() -> None:
    # A synthetic app whose ONLY command lives under a sub-app -- exercises the
    # recursion branch in ``_global_flags`` (dead in the real app, since
    # ``shell`` -- the first registered command -- is already a leaf).
    toy = App(name="toy")
    sub = App(name="sub")
    toy.command(sub)

    @sub.command
    def leaf(*, verbose: bool = False) -> None:
        """A sub-app leaf carrying one Common-style flag name."""

    flags = _global_flags(toy)
    assert [f["name"] for f in flags] == ["--verbose"]
