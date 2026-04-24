"""Tests for ``claude_sql.skills_catalog`` — the filesystem walker."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from claude_sql import skills_catalog
from claude_sql.config import Settings


def _write_skill(dir_: Path, name: str, description: str) -> None:
    dir_.mkdir(parents=True)
    (dir_ / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n"
    )


def _write_command(dir_: Path, stem: str, description: str, *, hint: str | None = None) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    frontmatter = f'description: "{description}"'
    if hint is not None:
        frontmatter += f'\nargument-hint: "{hint}"'
    (dir_ / f"{stem}.md").write_text(f"---\n{frontmatter}\n---\n\n# {stem}\n")


def _plugin_manifest(version_dir: Path, name: str, version: str) -> None:
    meta_dir = version_dir / ".claude-plugin"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "plugin.json").write_text(json.dumps({"name": name, "version": version}))


def _build_fixture_layout(root: Path) -> None:
    """Lay out a realistic slice of ~/.claude/skills + plugins/cache."""
    user_skills = root / "skills"
    _write_skill(user_skills / "gitnexus-cli", "gitnexus-cli", "Run gitnexus CLI.")

    cache = root / "plugins" / "cache"
    # Two versions of the same plugin; newest (1.29.0) should win.
    old_version = cache / "personal-plugins" / "personal-plugins" / "1.27.1"
    new_version = cache / "personal-plugins" / "personal-plugins" / "1.29.0"
    _plugin_manifest(old_version, "personal-plugins", "1.27.1")
    _plugin_manifest(new_version, "personal-plugins", "1.29.0")
    _write_skill(old_version / "skills" / "erpaval", "erpaval", "Old description.")
    _write_skill(new_version / "skills" / "erpaval", "erpaval", "New description.")
    _write_skill(
        new_version / "skills" / "browser-automation",
        "browser-automation",
        "Drive a browser.",
    )
    _write_command(
        new_version / "commands",
        "ralph-loop",
        "Run the ralph loop.",
        hint="PROMPT [--max-iterations N]",
    )


def _settings_for(root: Path) -> Settings:
    return Settings(
        user_skills_dir=root / "skills",
        plugins_cache_dir=root / "plugins" / "cache",
        skills_catalog_parquet_path=root / "skills_catalog.parquet",
    )


def test_sync_dry_run_counts(tmp_path: Path) -> None:
    _build_fixture_layout(tmp_path)
    settings = _settings_for(tmp_path)
    stats = skills_catalog.sync(settings, dry_run=True)
    # 1 user skill + (erpaval x2 + browser-automation x2 + ralph-loop x2)
    # + 15 builtins.  Dry run writes nothing.
    assert stats["builtins"] == len(skills_catalog.BUILTIN_SLASH_COMMANDS)
    assert stats["skills"] == 1 + 2 + 2  # user-skill + 2x plugin-skill pairs
    assert stats["commands"] == 2  # bare + namespaced plugin-command rows
    assert not (tmp_path / "skills_catalog.parquet").exists()


def test_sync_writes_parquet(tmp_path: Path) -> None:
    _build_fixture_layout(tmp_path)
    settings = _settings_for(tmp_path)
    skills_catalog.sync(settings, dry_run=False)

    out = tmp_path / "skills_catalog.parquet"
    assert out.exists()
    df = pl.read_parquet(out)

    # Newest version wins: plugin_version for erpaval is 1.29.0, not 1.27.1.
    erp = df.filter(pl.col("skill_id") == "erpaval")
    assert erp.height == 1
    assert erp["plugin_version"].item() == "1.29.0"
    assert erp["description"].item() == "New description."

    # Namespaced companion row exists with the same plugin version.
    ns = df.filter(pl.col("skill_id") == "personal-plugins:erpaval")
    assert ns.height == 1
    assert ns["plugin"].item() == "personal-plugins"
    assert ns["source_kind"].item() == "plugin-skill"

    # Plugin command present under both bare and namespaced keys.
    bare = df.filter(pl.col("skill_id") == "ralph-loop")
    ns_cmd = df.filter(pl.col("skill_id") == "personal-plugins:ralph-loop")
    assert bare.height == 1
    assert ns_cmd.height == 1
    assert bare["source_kind"].item() == "plugin-command"
    assert bare["argument_hint"].item() == "PROMPT [--max-iterations N]"

    # Built-ins: /clear is present and tagged.
    clear = df.filter(pl.col("skill_id") == "clear")
    assert clear.height == 1
    assert clear["source_kind"].item() == "builtin"

    # User-skill takes precedence over plugin-skill for duplicate bare names
    # (gitnexus-cli only exists in ~/.claude/skills here).
    gn = df.filter(pl.col("skill_id") == "gitnexus-cli")
    assert gn.height == 1
    assert gn["source_kind"].item() == "user-skill"


def test_sync_missing_roots_are_skipped(tmp_path: Path) -> None:
    """A setup with no user skills and no plugin cache still yields the builtins."""
    settings = _settings_for(tmp_path)  # directories don't exist
    stats = skills_catalog.sync(settings, dry_run=False)
    assert stats["skills"] == 0
    assert stats["commands"] == 0
    assert stats["builtins"] == len(skills_catalog.BUILTIN_SLASH_COMMANDS)
    df = pl.read_parquet(tmp_path / "skills_catalog.parquet")
    assert set(df["source_kind"].unique().to_list()) == {"builtin"}


def test_parse_frontmatter_tolerates_broken_yaml(tmp_path: Path) -> None:
    """A malformed SKILL.md yields an empty dict rather than aborting."""
    broken = tmp_path / "broken" / "SKILL.md"
    broken.parent.mkdir(parents=True)
    broken.write_text("---\nname: oops\n: this is invalid yaml\n---\n")
    # Direct call — the public sync path would just skip with a warning; this
    # pins the helper's contract directly.
    meta = skills_catalog._parse_frontmatter(broken)
    assert meta == {}
