"""Unit tests for blind_handover stripper."""

from __future__ import annotations

from claude_sql import blind_handover


def test_strips_slack_user_ids() -> None:
    txt = "Laith is U088Y6R64E9 and Sanju is U0AD7QKCTU6."
    got = blind_handover.strip_text(txt)
    assert "U088Y6R64E9" not in got.text
    assert "U0AD7QKCTU6" not in got.text
    assert "[user]" in got.text
    assert got.n_user_ids == 2


def test_strips_channel_and_team_ids() -> None:
    txt = "Channel C0AN06GNR9S on team T016M3G1GHZ; DM D0ADHAFGJPR."
    got = blind_handover.strip_text(txt)
    assert "C0AN06GNR9S" not in got.text
    assert "T016M3G1GHZ" not in got.text
    assert "D0ADHAFGJPR" not in got.text
    assert got.n_channel_ids == 3


def test_strips_mrkdwn_user_and_channel_refs() -> None:
    txt = "Hey <@U088Y6R64E9>, see <#C0ADWPJRGUE|bonk-place>. <!here>"
    got = blind_handover.strip_text(txt)
    assert "<@U" not in got.text
    assert "<#C" not in got.text
    assert "<!here>" not in got.text


def test_strips_persona_markers() -> None:
    txt = ":moyai: Bonk: here is the data. Clod: responds. -- Bonk"
    got = blind_handover.strip_text(txt)
    assert ":moyai:" not in got.text
    assert "Bonk" not in got.text
    assert "Clod" not in got.text
    assert got.n_persona_markers >= 3


def test_strips_radio_protocol_tokens() -> None:
    txt = "Reply coming. over :radio:"
    got = blind_handover.strip_text(txt)
    assert "over :radio:" not in got.text
    assert "[end-of-turn]" in got.text
    assert got.n_protocol_tokens == 1


def test_strips_work_items_and_thread_ts() -> None:
    txt = "wi_abc123def456 in thread 1776804739.101309"
    got = blind_handover.strip_text(txt)
    assert "wi_abc123def456" not in got.text
    assert "1776804739.101309" not in got.text
    assert got.n_system_ids == 2


def test_strips_mcp_tool_names() -> None:
    txt = "I called mcp__slack__slack_post_markdown and it worked."
    got = blind_handover.strip_text(txt)
    assert "mcp__slack__slack_post_markdown" not in got.text
    assert "[tool]" in got.text
    assert got.n_tool_names == 1


def test_strips_session_uuid() -> None:
    txt = "Session 1ab5c690-5e1e-41d2-8ac1-538accc012f1 was warm."
    got = blind_handover.strip_text(txt)
    assert "1ab5c690-5e1e-41d2-8ac1-538accc012f1" not in got.text
    assert got.n_system_ids == 1


def test_preserves_non_identity_text() -> None:
    txt = "The detector ran in 2.3s and caught 4 fabrications."
    got = blind_handover.strip_text(txt)
    assert "2.3s" in got.text
    assert "4 fabrications" in got.text


def test_original_hash_is_stable_16_chars() -> None:
    h = blind_handover.original_hash("session-xyz")
    assert len(h) == 16
    assert h == blind_handover.original_hash("session-xyz")
    assert h != blind_handover.original_hash("session-abc")


def test_multiple_runs_collapse_to_single_space() -> None:
    txt = "Hey   U088Y6R64E9   did   the   thing."
    got = blind_handover.strip_text(txt)
    assert "   " not in got.text
