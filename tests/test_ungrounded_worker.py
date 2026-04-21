"""Unit tests for ungrounded_worker."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from claude_sql import ungrounded_worker as uw


def test_extract_path_claim() -> None:
    claims = uw.extract_claims("See /efs/lalsaado/workplace/bonk/config.py for details.")
    paths = [c for c in claims if c.kind == "path"]
    assert any(c.entity == "/efs/lalsaado/workplace/bonk/config.py" for c in paths)


def test_extract_function_claim() -> None:
    claims = uw.extract_claims("We call handler.run() on each tick.")
    funcs = [c for c in claims if c.kind == "function"]
    assert any(c.entity == "handler.run" for c in funcs)


def test_extract_flag_claim() -> None:
    claims = uw.extract_claims("Pass --no-dry-run to actually spend.")
    flags = [c for c in claims if c.kind == "flag"]
    assert any(c.entity == "--no-dry-run" for c in flags)


def test_extract_env_var_claim() -> None:
    claims = uw.extract_claims("CLAUDE_SQL_CONCURRENCY=8 bumps parallelism.")
    envs = [c for c in claims if c.kind == "env_var"]
    assert any(c.entity == "CLAUDE_SQL_CONCURRENCY" for c in envs)


def test_extract_work_item_id() -> None:
    claims = uw.extract_claims("Tracked by wi_abc123def456.")
    ids = [c for c in claims if c.kind == "id"]
    assert any(c.entity == "wi_abc123def456" for c in ids)


def test_extract_cli_subcommand() -> None:
    claims = uw.extract_claims("Run `claude-sql judge` to dispatch.")
    clis = [c for c in claims if c.kind == "cli"]
    assert any(c.entity == "judge" for c in clis)


def test_grounded_when_entity_seen_in_tool_output() -> None:
    text = "File /home/user/code.py exists."
    tool_output = "ls /home/user/ -> code.py config.toml\ncat /home/user/code.py: def main()..."
    claims = uw.extract_claims(text)
    checks = uw.check_claims(claims, tool_output)
    paths = [c for c in checks if c["claim_kind"] == "path"]
    assert len(paths) >= 1
    assert paths[0]["grounded"] is True
    assert paths[0]["tool_output_hits"] >= 1


def test_ungrounded_when_entity_absent() -> None:
    text = "The handler.magic_method() fires on boot."
    tool_output = "grep -r 'magic_method' src/\n(no results)"
    claims = uw.extract_claims(text)
    checks = uw.check_claims(claims, tool_output)
    funcs = [c for c in checks if c["claim_kind"] == "function"]
    assert len(funcs) >= 1
    assert funcs[0]["grounded"] is False
    assert funcs[0]["tool_output_hits"] == 0


def test_detect_produces_stable_schema() -> None:
    turns = [
        uw.Turn(
            session_id="s1",
            turn_idx=0,
            assistant_text="The config at /etc/foo.conf matters.",
            tool_output_text="cat /etc/foo.conf -> key=value",
        ),
        uw.Turn(
            session_id="s1",
            turn_idx=1,
            assistant_text="Call handler.unknown_method() anytime.",
            tool_output_text="grep handler -> handler.run",
        ),
    ]
    df = uw.detect(turns, freeze_sha="deadbeefcafe0000")
    assert set(df.columns) == {
        "session_id",
        "turn_idx",
        "claim_entity",
        "claim_kind",
        "grounded",
        "tool_output_hits",
        "freeze_sha",
    }
    # The fabricated handler.unknown_method should be ungrounded
    ungrounded = df.filter(~pl.col("grounded"))
    assert ungrounded.height >= 1
    assert "handler.unknown_method" in ungrounded["claim_entity"].to_list()


def test_summarize_ranks_by_ungrounded_rate() -> None:
    rows = [
        # s1: 2 claims, 1 ungrounded -> rate 0.5
        {
            "session_id": "s1",
            "turn_idx": 0,
            "claim_entity": "a",
            "claim_kind": "path",
            "grounded": True,
            "tool_output_hits": 1,
            "freeze_sha": "x",
        },
        {
            "session_id": "s1",
            "turn_idx": 1,
            "claim_entity": "b",
            "claim_kind": "path",
            "grounded": False,
            "tool_output_hits": 0,
            "freeze_sha": "x",
        },
        # s2: 2 claims, 2 ungrounded -> rate 1.0 (ranks first)
        {
            "session_id": "s2",
            "turn_idx": 0,
            "claim_entity": "c",
            "claim_kind": "path",
            "grounded": False,
            "tool_output_hits": 0,
            "freeze_sha": "x",
        },
        {
            "session_id": "s2",
            "turn_idx": 1,
            "claim_entity": "d",
            "claim_kind": "path",
            "grounded": False,
            "tool_output_hits": 0,
            "freeze_sha": "x",
        },
    ]
    df = pl.DataFrame(rows)
    summary = uw.summarize(df)
    assert summary["session_id"][0] == "s2"
    assert summary["ungrounded_rate"][0] == 1.0


def test_detect_empty_returns_schema_stable_frame() -> None:
    df = uw.detect([], freeze_sha="x")
    assert df.height == 0
    assert "claim_entity" in df.columns


def test_to_parquet_roundtrip(tmp_path: Path) -> None:
    df = uw.detect(
        [
            uw.Turn(
                session_id="s1",
                turn_idx=0,
                assistant_text="See /a/b.py",
                tool_output_text="cat /a/b.py",
            )
        ],
        freeze_sha="x",
    )
    p = tmp_path / "u.parquet"
    uw.to_parquet(df, p)
    back = pl.read_parquet(p)
    assert back.height == df.height
