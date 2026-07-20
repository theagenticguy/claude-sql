"""Pydantic v2 classification schemas (pure domain models).

These are the domain contract for the LLM-analytics plane: one pydantic model
per structured-output call shape. They are pure (pydantic only, no boto3 /
duckdb / adapters), so they live in the innermost hexagon.

The Bedrock wire-format flattening (``_bedrock_schema`` / ``_flatten``) and the
four live ``*_SCHEMA`` constants derived from these models live in
:mod:`claude_sql.infrastructure.bedrock.structured_output` — that flattening is a
Bedrock-subset concern and belongs in the Bedrock infrastructure adapter, which
imports the model classes from here.

Rehomed here from ``core/schemas.py`` in the v2 hexagonal final cut (T-8-2): the
transitional ``core`` package is dissolved and the pydantic models are pure
domain types.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SessionClassification(BaseModel):
    """Classify an entire Claude Code session.

    One row per session, written to ``session_classifications.parquet``.
    Used to build the ``session_classifications`` DuckDB view.
    """

    model_config = ConfigDict(extra="forbid")

    autonomy_tier: Literal["manual", "assisted", "autonomous"] = Field(
        ...,
        description=(
            "How much the agent drove the work.  "
            "'manual': the user typed every instruction and confirmed each step. "
            "'assisted': the agent took initiative but the user course-corrected often. "
            "'autonomous': the agent ran multi-step work end-to-end with minimal user "
            "intervention -- tier-3 work."
        ),
    )
    work_category: Literal[
        "sde",
        "admin",
        "strategy_business",
        "events",
        "thought_leadership",
        "other",
    ] = Field(
        ...,
        description=(
            "Dominant activity.  "
            "'sde': software engineering, debugging, code review, tests, CI. "
            "'admin': expense reports, scheduling, low-signal email triage, routine ops. "
            "'strategy_business': business analysis, competitive research, strategic memos, "
            "proposals. "
            "'events': event planning/logistics, speaker prep, agenda building. "
            "'thought_leadership': writing for external audiences (blog posts, conference "
            "abstracts, LinkedIn). "
            "'other': use only when nothing else fits."
        ),
    )
    success: Literal["success", "partial", "failure", "unknown"] = Field(
        ...,
        description=(
            "Did the session complete its stated goal? "
            "'success': the user's goal was clearly met. "
            "'partial': the goal was reached with caveats or leftover TODOs. "
            "'failure': the session ended without achieving the goal. "
            "'unknown': insufficient signal to judge."
        ),
    )
    goal: str = Field(
        ...,
        min_length=1,
        max_length=280,
        description=(
            "ONE sentence summarizing the user's goal, inferred from the opening user "
            "messages and overall arc.  Present tense, <= 280 chars.  Example: "
            '"Refactor the auth middleware to use the new token rotator."'
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Classifier self-assessed confidence 0.0-1.0. Use <0.5 for genuinely "
            "ambiguous sessions."
        ),
    )


class TrajectoryWindow(BaseModel):
    """One windowed turn-pair classification (RFC 0002 §3.4 / §4.1).

    A *window* binds two adjacent text turns from one session — the
    previous turn (``prev_uuid``) and the current turn (``curr_uuid``).
    The very first window of a session has ``prev_uuid is None`` plus a
    synthetic ``prev_sentiment``: this lets the parquet hold one row per
    text-turn instead of one fewer than the turn count.

    The model emits one such object per window; the array form
    :class:`TrajectoryArrayResult` carries the per-chunk batch.
    """

    model_config = ConfigDict(extra="forbid")

    prev_uuid: str | None = Field(
        ...,
        description=(
            "UUID of the prior text turn in this session, or null for the "
            "session-first window. The host pipeline echoes (prev_uuid, "
            "curr_uuid) back to verify per-window completeness."
        ),
    )
    curr_uuid: str = Field(
        ...,
        min_length=1,
        description=(
            "UUID of the current text turn — the window's anchor. Must "
            "match exactly one of the curr_uuid values supplied in the "
            "<window> XML payload."
        ),
    )
    prev_sentiment: Literal["negative", "neutral", "positive"] | None = Field(
        ...,
        description=(
            "Polarity of the prior turn (or null on session-first). "
            "'negative' = frustration / pushback / blocked. 'neutral' = "
            "factual / procedural / acknowledgement (majority class). "
            "'positive' = excitement / approval / momentum."
        ),
    )
    curr_sentiment: Literal["negative", "neutral", "positive"] = Field(
        ...,
        description="Polarity of the current turn — same three labels as prev_sentiment.",
    )
    delta: float | None = Field(
        ...,
        description=(
            "curr_sentiment - prev_sentiment encoded as integer in "
            "{-2,-1,0,1,2} (negative=-1, neutral=0, positive=1, then "
            "subtract). null when prev is null. The ``delta`` field is the "
            "load-bearing signal for downstream sentiment-arc analytics."
        ),
    )
    is_transition: bool = Field(
        ...,
        description=(
            "True when the *current* turn is pure filler / acknowledgement "
            "with no substantive content (e.g. 'ok let me check', "
            "'running...', 'done.'). Independent of prev_sentiment."
        ),
    )
    transition_kind: Literal[
        "frustration_spike",
        "resolution",
        "reset",
        "drift",
        "clarification",
        "none",
    ] = Field(
        ...,
        description=(
            "Categorical label for the shape of the prev→curr transition. "
            "'frustration_spike' = neutral/positive → negative. "
            "'resolution' = negative → neutral/positive (problem fixed). "
            "'reset' = abrupt topic change unrelated to prev. "
            "'drift' = same polarity but new sub-topic. "
            "'clarification' = curr restates / refines prev's substance. "
            "'none' = no salient transition (the majority class — use it)."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Classifier self-confidence 0.0-1.0. Use <0.5 when the cue is "
            "ambiguous or the prev/curr polarities are both neutral with "
            "no visible salience."
        ),
    )


class TrajectoryArrayResult(BaseModel):
    """Sonnet returns this — the array of windows for one chunk.

    The host pipeline verifies completeness by echoing the
    (prev_uuid, curr_uuid) tuples in the request payload back against
    the returned ``windows``. Missing windows trigger one bounded retry;
    persistent misses are stamped with neutral placeholders so the
    pipeline never wedges on a single refused chunk.
    """

    model_config = ConfigDict(extra="forbid")

    windows: list[TrajectoryWindow] = Field(
        ...,
        description=(
            "One TrajectoryWindow per (prev_uuid, curr_uuid) supplied in "
            "the request. Order should match the input window order; the "
            "host pipeline does not rely on order but ordered output is "
            "easier for an auditor to skim."
        ),
    )


class ConflictPair(BaseModel):
    """A single stance-conflict pair, keyed on the two opposing turn UUIDs.

    The v1.0 storage shape is pair-keyed on ``(turn_a_uuid, turn_b_uuid)``.
    The pair-scanner that emits one row per *adjacent turn pair* is RFC §4.2
    work scheduled for v1.1; for v1.0 the whole-session prompt still runs
    but it must now name the two specific turns whose stances clash.
    """

    model_config = ConfigDict(extra="forbid")

    turn_a_uuid: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description=(
            "UUID of the first turn that holds stance A.  Pull from the "
            "``[uuid=...]`` headers in the bound transcript -- copy verbatim, "
            "do NOT invent or paraphrase.  Together with ``turn_b_uuid`` this "
            "is the natural key of a conflict row."
        ),
    )
    turn_b_uuid: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description=(
            "UUID of the second turn that holds the opposing stance.  Same "
            "rules as ``turn_a_uuid`` -- copy verbatim from the transcript "
            "headers.  Must differ from ``turn_a_uuid``."
        ),
    )
    conflict_kind: Literal["disagreement", "correction", "reversal", "impasse"] = Field(
        ...,
        description=(
            "Shape of the conflict.  "
            "'disagreement': two parties hold opposing positions and discuss them. "
            "'correction': one party explicitly tells the other their answer/action "
            "was wrong (e.g. 'no, not that, do X instead'). "
            "'reversal': the same party flips their own earlier position "
            "('actually let's NOT do X'). "
            "'impasse': both sides restate their positions without converging "
            "and the topic stalls."
        ),
    )
    severity: Literal["low", "medium", "high"] = Field(
        ...,
        description=(
            "How consequential the conflict is for the session outcome.  "
            "'low': a minor course nudge with little downstream impact. "
            "'medium': changes the implementation approach or scope but stays "
            "inside the original goal. "
            "'high': blocks progress, reverses a major decision, or "
            "fundamentally changes the goal."
        ),
    )
    agent_position: str = Field(
        ...,
        min_length=1,
        max_length=280,
        description=(
            "One-sentence summary of the agent's stance in this conflict, "
            "phrased in the agent's own framing.  Strip pleasantries; keep "
            "the substantive claim."
        ),
    )
    user_position: str = Field(
        ...,
        min_length=1,
        max_length=280,
        description=(
            "One-sentence summary of the user's stance, phrased the way the "
            "user phrased it.  Same length budget as ``agent_position``."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Classifier self-confidence 0.0-1.0 that this is a real, "
            "substantive conflict (not deliberation, not collaboration, not "
            "an accepted-risk caveat).  Use <0.5 for borderline cases."
        ),
    )


class ConflictsResult(BaseModel):
    """Sonnet's response: zero or more conflict pairs.

    An empty list is valid and common.  When the model returns an empty
    list the worker writes ZERO rows for that session -- the legacy
    ``empty=True`` sentinel row is gone.  Sessions with no conflicts simply
    don't appear in ``session_conflicts`` (or in the derived
    ``conflicts_summary`` view).
    """

    model_config = ConfigDict(extra="forbid")

    conflicts: list[ConflictPair] = Field(
        default_factory=list,
        description=(
            "Zero or more stance-conflict pairs.  Each entry names the two "
            "turn UUIDs whose stances clash, the kind/severity, both party "
            "positions, and a confidence score.  Report only substantive "
            "technical or strategic conflicts -- skip trivial style "
            "disagreements."
        ),
    )


class UserFrictionSignal(BaseModel):
    """Classify a single short user message for friction signals.

    A "friction signal" is anything in the user's utterance that implies the
    agent's last turn fell short of expectations: an impatient status ping,
    a one-word question pointing at a missed artifact (``screenshot?``,
    ``tests?``), confusion about what happened, a hard interruption, a
    correction, or open frustration.

    Applied only to user-role messages below
    :class:`Settings.friction_max_chars` (default 300).  Long user messages
    are almost always genuine task turns rather than interrupt/confusion
    signals -- the filter keeps Bedrock cost linear in the interesting slice.
    """

    model_config = ConfigDict(extra="forbid")

    label: Literal[
        "status_ping",
        "unmet_expectation",
        "confusion",
        "interruption",
        "correction",
        "frustration",
        "none",
    ] = Field(
        ...,
        description=(
            "Dominant friction category for this single user message.  "
            "'status_ping': the user asks about progress/ETA ('how's it going?', "
            "'any update?', 'status?', 'where are we?'). "
            "'unmet_expectation': a one- or two-word question that points at "
            "something the agent should have done proactively but didn't "
            "('screenshot?', 'tests?', 'link?', 'diff?'). "
            "'confusion': the user signals they don't understand the output or "
            "state ('what does that mean?', 'why did you do X?', 'I don't get it'). "
            "'interruption': the user cuts the agent off or redirects mid-task "
            "('wait', 'stop', 'hold on', 'actually...', 'before you do that'). "
            "'correction': the user tells the agent its last action/answer was "
            "wrong ('no, not that', 'that's wrong', 'nope', 'try again'). "
            "'frustration': terse annoyance or sarcasm ('ugh', 'seriously?', "
            "'are you kidding', 'really?'). "
            "'none': ordinary task turn with no friction signal -- a substantive "
            "instruction, a plain question, an acknowledgement. USE THIS FOR "
            "THE MAJORITY OF MESSAGES."
        ),
    )
    rationale: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "One short sentence (<=200 chars) naming the specific phrase or "
            "structural cue that triggered the label.  Use an empty-ish "
            "placeholder like 'ordinary instruction' when label='none'."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Classifier self-confidence 0.0-1.0.  Use <0.5 when the message is "
            "genuinely ambiguous between 'none' and a friction label."
        ),
    )


__all__ = [
    "ConflictPair",
    "ConflictsResult",
    "SessionClassification",
    "TrajectoryArrayResult",
    "TrajectoryWindow",
    "UserFrictionSignal",
]
