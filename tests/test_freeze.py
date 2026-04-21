"""Unit tests for freeze + replay."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from claude_sql import freeze


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ~/.claude/studies at a tmp dir so tests don't touch the user's home."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    # os.path.expanduser is what freeze uses; monkeypatching HOME is enough
    return fake_home


@pytest.fixture
def rubric(tmp_path: Path) -> Path:
    p = tmp_path / "rubric.yaml"
    p.write_text("axes:\n  - correction_required\n  - fabrication_present\n", encoding="utf-8")
    return p


def test_freeze_produces_deterministic_sha(isolated_home: Path, rubric: Path) -> None:
    s1 = freeze.freeze(rubric, panel_shortnames=("kimi-k2.5", "deepseek-v3.2"))
    s2 = freeze.freeze(rubric, panel_shortnames=("kimi-k2.5", "deepseek-v3.2"))
    assert s1.manifest_sha == s2.manifest_sha


def test_freeze_changes_sha_on_rubric_edit(isolated_home: Path, rubric: Path) -> None:
    s1 = freeze.freeze(rubric, panel_shortnames=("kimi-k2.5",))
    rubric.write_text(rubric.read_text() + "\n  - new_axis\n", encoding="utf-8")
    s2 = freeze.freeze(rubric, panel_shortnames=("kimi-k2.5",))
    assert s1.manifest_sha != s2.manifest_sha


def test_freeze_changes_sha_on_panel_change(isolated_home: Path, rubric: Path) -> None:
    s1 = freeze.freeze(rubric, panel_shortnames=("kimi-k2.5",))
    s2 = freeze.freeze(rubric, panel_shortnames=("kimi-k2.5", "deepseek-v3.2"))
    assert s1.manifest_sha != s2.manifest_sha


def test_freeze_writes_manifest_and_rubric(isolated_home: Path, rubric: Path) -> None:
    s = freeze.freeze(rubric, panel_shortnames=("kimi-k2.5",))
    study_dir = Path(os.path.expanduser("~/.claude/studies")) / s.manifest_sha
    assert (study_dir / "manifest.json").exists()
    assert (study_dir / "rubric.yaml").exists()
    payload = json.loads((study_dir / "manifest.json").read_text())
    assert payload["manifest_sha"] == s.manifest_sha
    assert payload["panel_shortnames"] == ["kimi-k2.5"]


def test_replay_round_trip(isolated_home: Path, rubric: Path) -> None:
    s = freeze.freeze(
        rubric,
        panel_shortnames=("kimi-k2.5", "deepseek-v3.2"),
        embed_model_id="global.cohere.embed-v4:0",
    )
    replayed = freeze.replay(s.manifest_sha)
    assert replayed.manifest_sha == s.manifest_sha
    assert replayed.panel_shortnames == ("kimi-k2.5", "deepseek-v3.2")


def test_replay_missing_raises(isolated_home: Path) -> None:
    with pytest.raises(FileNotFoundError):
        freeze.replay("0000000000000000")


def test_unknown_judge_shortname_rejected(isolated_home: Path, rubric: Path) -> None:
    with pytest.raises(KeyError):
        freeze.freeze(rubric, panel_shortnames=("not-a-judge",))


def test_missing_rubric_raises(isolated_home: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        freeze.freeze(tmp_path / "nope.yaml", panel_shortnames=("kimi-k2.5",))


def test_list_studies_returns_every_frozen(isolated_home: Path, rubric: Path) -> None:
    s1 = freeze.freeze(rubric, panel_shortnames=("kimi-k2.5",))
    s2 = freeze.freeze(rubric, panel_shortnames=("kimi-k2.5", "deepseek-v3.2"))
    summaries = freeze.list_studies()
    shas = {row["manifest_sha"] for row in summaries}
    assert s1.manifest_sha in shas
    assert s2.manifest_sha in shas
