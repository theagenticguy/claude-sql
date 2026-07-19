"""Regenerate ``docs/reference/cli-manifest.md`` from the CLI manifest.

The manifest (:mod:`claude_sql.core.manifest`) is derived from the cyclopts
command tree by introspection, so this generator can never describe a flag
that doesn't exist. Run it directly to refresh the doc after adding or
changing a command:

    uv run python scripts/gen_cli_manifest_doc.py

Wired as the ``docs:cli-manifest`` mise task (``sources`` = ``cli.py`` +
``manifest.py``, ``outputs`` = the generated doc) so CI can detect drift by
running it and diffing against the committed file.
"""

from __future__ import annotations

from pathlib import Path

from claude_sql.app.cli import app
from claude_sql.core.manifest import build_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "docs" / "reference" / "cli-manifest.md"


def _render_table(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return ["_No parameters._", ""]
    lines = [
        "| parameter | type | required | choices | default | help |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        choices = ", ".join(str(c) for c in row["choices"]) if row["choices"] else "—"
        default = f"`{row['default']}`" if row["default"] is not None else "—"
        help_text = row["help"] or "—"
        lines.append(
            f"| `{row['name']}` | {row['type']} | "
            f"{'yes' if row['required'] else 'no'} | {choices} | {default} | {help_text} |"
        )
    lines.append("")
    return lines


def render_markdown(manifest: dict[str, object]) -> str:
    """Render the manifest dict as a Markdown reference doc."""
    lines: list[str] = [
        f"# {manifest['cli']} — CLI manifest (generated)",
        "",
        "**Do not hand-edit.** Regenerate with "
        "`uv run python scripts/gen_cli_manifest_doc.py` (or `mise run docs:cli-manifest`) "
        "after adding, renaming, or reshaping a command — it is derived by introspection "
        "from `src/claude_sql/app/cli.py` via `claude_sql.core.manifest.build_manifest`, "
        "so it can never describe a flag that doesn't exist. Machine-readable equivalent: "
        "`claude-sql manifest --format json`.",
        "",
        f"Version at generation time: `{manifest['version']}`.",
        "",
        str(manifest["summary"]),
        "",
        "## Conventions",
        "",
    ]
    lines += [f"- {c}" for c in manifest["conventions"]]  # type: ignore[union-attr]
    lines += [
        "",
        "## Global flags",
        "",
        "Every command below accepts these in addition to its own parameters.",
        "",
    ]
    lines += _render_table(manifest["global_flags"])  # type: ignore[arg-type]
    lines += [
        "## Exit codes",
        "",
        "| name | code |",
        "|---|---|",
    ]
    for name, code in manifest["exit_codes"].items():  # type: ignore[union-attr]
        lines.append(f"| `{name}` | `{code}` |")
    lines += ["", "## Commands", ""]
    for cmd in manifest["commands"]:  # type: ignore[union-attr]
        lines.append(f"### `{manifest['cli']} {cmd['name']}`")
        lines.append("")
        if cmd["summary"]:
            lines.append(str(cmd["summary"]))
            lines.append("")
        lines += _render_table(cmd["parameters"])  # type: ignore[arg-type]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    """Build the manifest, render it, and write ``docs/reference/cli-manifest.md``."""
    manifest = build_manifest(app)
    OUTPUT_PATH.write_text(render_markdown(manifest))
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
