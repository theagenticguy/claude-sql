"""Multi-provider judge worker: runs a panel of Bedrock models over sessions.

Uses the Bedrock **Converse API** (``bedrock-runtime.converse``) which is
the only path that works uniformly across Anthropic, Moonshot, DeepSeek,
MiniMax, Mistral, Z.AI, Qwen, Writer, and NVIDIA Nemotron on Bedrock.

For each (session, axis) the worker dispatches one call per judge and
writes a parquet row with the score and free-text rationale.  No tool_use
machinery — the prompt instructs each judge to output `score=<int>\\n
rationale=<text>` and we parse it.  Stable across model families; the
tradeoff is we rely on rubric-prompted discipline rather than schema
enforcement.  That is deliberate: a structured-output API tied to one
provider would leak within-family bias back into the ensemble.

Cost guard: defaults to ``dry_run=True`` per the project convention —
emits a plan dict with ``(n_calls, est_tokens, est_dollars)`` rather
than spending.
"""

from __future__ import annotations

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import polars as pl
from botocore.config import Config as BotoConfig
from loguru import logger

from claude_sql import judges as judge_catalog
from claude_sql.judges import Judge

# Rough per-1M-token cost estimates (USD).  Conservative; updated as
# Bedrock publishes list prices.  Missing entries fall through to a
# 2.00/10.00 fallback.
_JUDGE_PRICING: dict[str, tuple[float, float]] = {
    "moonshotai.kimi-k2.5": (0.60, 2.50),
    "moonshot.kimi-k2-thinking": (1.00, 4.00),
    "deepseek.v3.2": (0.30, 1.20),
    "minimax.minimax-m2.5": (0.50, 2.00),
    "zai.glm-5": (0.60, 2.40),
    "qwen.qwen3-next-80b-a3b": (0.80, 3.20),
    "mistral.mistral-large-3-675b-instruct": (3.00, 12.00),
    "mistral.magistral-small-2509": (0.40, 1.60),
    "writer.palmyra-x5-v1:0": (2.50, 10.00),
    "nvidia.nemotron-super-3-120b": (1.20, 4.80),
    "anthropic.claude-opus-4-7": (15.00, 75.00),
    "anthropic.claude-sonnet-4-6": (3.00, 15.00),
    "amazon.nova-2-lite-v1:0:256k": (0.06, 0.24),
}
_FALLBACK_PRICE: tuple[float, float] = (2.00, 10.00)


@dataclass(frozen=True)
class GradePlan:
    """Dry-run plan: what ``run`` would do if ``dry_run=False``."""

    n_sessions: int
    n_judges: int
    n_axes: int
    n_calls: int
    est_input_tokens: int
    est_output_tokens: int
    est_usd: float


@dataclass(frozen=True)
class Axis:
    """One scoring axis from the rubric."""

    name: str
    description: str
    levels: dict[int, str]  # score -> level description
    detector_vs_grader: str  # "detector" or "grader"


def parse_rubric(rubric_yaml: str) -> list[Axis]:
    """Parse a rubric YAML into ``Axis`` objects.

    Minimal YAML subset: key:value pairs, ``-`` list items, 2-space
    indent.  We avoid a pyyaml dep here; rubrics are agent-authored
    and shape-constrained.

    Expected shape::

        axes:
          - name: correction_required
            description: "..."
            detector_vs_grader: detector
            levels:
              0: "no correction needed"
              1: "correction needed"
    """
    import yaml  # lazy import; add pyyaml to deps if unused elsewhere

    data = yaml.safe_load(rubric_yaml) or {}
    raw = data.get("axes") or []
    if not isinstance(raw, list):
        raise ValueError("rubric.axes must be a list")
    out: list[Axis] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"rubric.axes[{i}] must be a mapping")
        name = entry.get("name")
        if not name:
            raise ValueError(f"rubric.axes[{i}] is missing 'name'")
        out.append(
            Axis(
                name=str(name),
                description=str(entry.get("description", "")),
                levels={int(k): str(v) for k, v in (entry.get("levels") or {}).items()},
                detector_vs_grader=str(entry.get("detector_vs_grader", "grader")),
            )
        )
    if not out:
        raise ValueError("rubric has zero axes")
    return out


def render_prompt(session_text: str, axis: Axis) -> str:
    """Render the single-turn grading prompt.

    Prompt discipline, not schema enforcement, is what we rely on
    across non-Anthropic judges.  Keep it minimal and explicit.
    """
    levels_block = "\n".join(f"  {k}: {v}" for k, v in sorted(axis.levels.items()))
    return (
        f"You are scoring an agent transcript on ONE axis.\n"
        f"\n"
        f"Axis name: {axis.name}\n"
        f"Axis description: {axis.description}\n"
        f"Scoring levels:\n{levels_block}\n"
        f"\n"
        f"Respond in EXACTLY this format, nothing else:\n"
        f"score=<int>\n"
        f"rationale=<one-paragraph explanation>\n"
        f"\n"
        f"=== TRANSCRIPT BEGIN ===\n"
        f"{session_text}\n"
        f"=== TRANSCRIPT END ===\n"
    )


_SCORE_RE = re.compile(r"^\s*score\s*=\s*(-?\d+)\s*$", re.MULTILINE)
_RATIONALE_RE = re.compile(r"^\s*rationale\s*=\s*(.*)$", re.MULTILINE | re.DOTALL)


def parse_judge_response(text: str) -> tuple[int | None, str]:
    """Extract (score, rationale) from a free-form judge response.

    Tolerant: if the format is slightly off (JSON object, extra prose
    before the fields, etc.), we still try to recover a score.  If we
    cannot parse an integer score, return ``(None, <raw text>)`` so
    callers can log the refusal and continue.
    """
    m = _SCORE_RE.search(text)
    score: int | None = int(m.group(1)) if m else None
    if score is None:
        # Try JSON-shaped fallback
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "score" in obj:
                score = int(obj["score"])
        except (ValueError, TypeError):
            pass
    r = _RATIONALE_RE.search(text)
    rationale = r.group(1).strip() if r else text.strip()
    return score, rationale


# ---------------------------------------------------------------------------
# Bedrock Converse dispatch
# ---------------------------------------------------------------------------


def _bedrock_client(region: str = "us-east-1"):
    """Return a tuned boto3 bedrock-runtime client."""
    cfg = BotoConfig(
        region_name=region,
        retries={"max_attempts": 0, "mode": "standard"},
        read_timeout=120,
        connect_timeout=10,
    )
    return boto3.client("bedrock-runtime", config=cfg)


def _converse_once(
    client: Any, model_id: str, prompt: str, max_tokens: int = 4096, temperature: float = 0.0
) -> str:
    """One Bedrock Converse call; returns the model's plain text response."""
    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )
    msg = resp.get("output", {}).get("message", {})
    parts = msg.get("content", []) or []
    for p in parts:
        if "text" in p:
            return p["text"]
    return ""


# ---------------------------------------------------------------------------
# Planning (dry-run)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Approximate tokens as ``chars / 4`` — standard Bedrock heuristic."""
    return max(1, len(text) // 4)


def plan(
    sessions: list[tuple[str, str]],
    panel: list[Judge],
    axes: list[Axis],
    out_tokens: int = 256,
) -> GradePlan:
    """Estimate call count + token + dollar cost of a grade run."""
    n_calls = len(sessions) * len(panel) * len(axes)
    total_in = 0
    total_out = 0
    total_usd = 0.0
    for _, text in sessions:
        for axis in axes:
            prompt = render_prompt(text, axis)
            tin = estimate_tokens(prompt)
            tout = out_tokens
            for j in panel:
                total_in += tin
                total_out += tout
                pin, pout = _JUDGE_PRICING.get(j.model_id, _FALLBACK_PRICE)
                total_usd += (tin / 1_000_000) * pin + (tout / 1_000_000) * pout
    return GradePlan(
        n_sessions=len(sessions),
        n_judges=len(panel),
        n_axes=len(axes),
        n_calls=n_calls,
        est_input_tokens=total_in,
        est_output_tokens=total_out,
        est_usd=round(total_usd, 4),
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeScore:
    """One row of the output parquet."""

    session_id: str
    axis: str
    judge_shortname: str
    judge_model_id: str
    score: int
    rationale: str
    freeze_sha: str


async def _grade_one(
    loop: asyncio.AbstractEventLoop,
    executor: ThreadPoolExecutor,
    client: Any,
    session_id: str,
    session_text: str,
    judge: Judge,
    axis: Axis,
    freeze_sha: str,
) -> JudgeScore | None:
    prompt = render_prompt(session_text, axis)
    try:
        text = await loop.run_in_executor(
            executor, lambda: _converse_once(client, judge.model_id, prompt)
        )
    except Exception as exc:  # noqa: BLE001 — log + skip; the study has 10+ judges
        logger.warning("judge {} failed on {}/{}: {}", judge.shortname, session_id, axis.name, exc)
        return None
    score, rationale = parse_judge_response(text)
    if score is None:
        logger.warning(
            "judge {} returned unparseable response on {}/{}: {!r}",
            judge.shortname,
            session_id,
            axis.name,
            text[:2000],
        )
        # Persist the full text as the rationale so post-hoc triage can
        # see what the judge actually returned. score=None is the sentinel.
        return JudgeScore(
            session_id=session_id,
            axis=axis.name,
            judge_shortname=judge.shortname,
            judge_model_id=judge.model_id,
            score=-1,
            rationale=f"[unparseable] {rationale}",
            freeze_sha=freeze_sha,
        )
    return JudgeScore(
        session_id=session_id,
        axis=axis.name,
        judge_shortname=judge.shortname,
        judge_model_id=judge.model_id,
        score=score,
        rationale=rationale,
        freeze_sha=freeze_sha,
    )


async def run_async(
    sessions: list[tuple[str, str]],
    panel: list[Judge],
    axes: list[Axis],
    freeze_sha: str,
    concurrency: int = 4,
    region: str = "us-east-1",
) -> list[JudgeScore]:
    """Grade every (session, judge, axis) triple concurrently."""
    client = _bedrock_client(region=region)
    loop = asyncio.get_running_loop()
    out: list[JudgeScore] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        sem = asyncio.Semaphore(concurrency)

        async def bounded(coro):
            async with sem:
                return await coro

        tasks = [
            bounded(_grade_one(loop, executor, client, sid, text, judge, axis, freeze_sha))
            for sid, text in sessions
            for judge in panel
            for axis in axes
        ]
        results = await asyncio.gather(*tasks)
    out.extend(r for r in results if r is not None)
    return out


def to_parquet(scores: list[JudgeScore], path: Path) -> None:
    """Write scores to parquet with a stable schema."""
    if not scores:
        logger.warning("no scores to write; emitting empty parquet for schema stability")
    df = pl.DataFrame(
        {
            "session_id": [s.session_id for s in scores],
            "axis": [s.axis for s in scores],
            "judge_shortname": [s.judge_shortname for s in scores],
            "judge_model_id": [s.judge_model_id for s in scores],
            "score": [s.score for s in scores],
            "rationale": [s.rationale for s in scores],
            "freeze_sha": [s.freeze_sha for s in scores],
        },
        schema={
            "session_id": pl.String,
            "axis": pl.String,
            "judge_shortname": pl.String,
            "judge_model_id": pl.String,
            "score": pl.Int64,
            "rationale": pl.String,
            "freeze_sha": pl.String,
        },
    )
    df.write_parquet(path)


def run(
    sessions: list[tuple[str, str]],
    panel_shortnames: list[str],
    rubric_yaml_path: Path,
    freeze_sha: str,
    out_parquet: Path,
    *,
    dry_run: bool = True,
    concurrency: int = 4,
    region: str = "us-east-1",
) -> GradePlan | list[JudgeScore]:
    """Synchronous entry point used by the CLI."""
    panel = judge_catalog.panel(panel_shortnames)
    axes = parse_rubric(rubric_yaml_path.read_text(encoding="utf-8"))
    p = plan(sessions, panel, axes)
    logger.info(
        "judge plan: {} calls across {} sessions × {} judges × {} axes; "
        "~{} in-tok, ~{} out-tok, ~${:.4f}",
        p.n_calls,
        p.n_sessions,
        p.n_judges,
        p.n_axes,
        p.est_input_tokens,
        p.est_output_tokens,
        p.est_usd,
    )
    if dry_run:
        return p
    scores = asyncio.run(
        run_async(sessions, panel, axes, freeze_sha, concurrency=concurrency, region=region)
    )
    to_parquet(scores, out_parquet)
    return scores
