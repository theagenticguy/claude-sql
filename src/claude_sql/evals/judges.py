"""Cross-provider Bedrock judge panel catalog.

Used by the ``judge`` subcommand to dispatch agent-output grading to a
panel of foundation models with diverse training lineages.  Goal:
ensemble *disagreement* surfaces bias; within-family agreement is the
bias we are measuring, not the signal.

All model IDs below are validated against ``aws bedrock list-foundation-models``
in ``us-east-1`` as of 2026-04-21 (lalsaado-handson profile).

Shortnames are stable CLI aliases so ``--panel kimi-k2.5,deepseek-v3.2``
does not rot every time a provider changes its ID suffix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

JudgeFamily = Literal["anthropic", "amazon", "non-anthropic-non-amazon"]
JudgeRole = Literal["judge", "bulk", "embed", "within-family-holdout"]


@dataclass(frozen=True)
class Judge:
    """One Bedrock foundation model wired into the judge panel."""

    shortname: str
    model_id: str
    provider: str
    family: JudgeFamily
    role: JudgeRole
    notes: str


#: Primary ensemble: non-Anthropic, non-Amazon judges for cross-lineage
#: variance.  Eight distinct training corpora across Chinese, North
#: American, European labs.
#:
#: Mistral Large 3 and Magistral-Small were dropped after the first
#: gym run (study ``ab0bf2eeb481fdd2``, 2026-04-21) because they
#: produced 4/50 unparseable responses \u2014 worst rubric discipline of
#: the panel.  They remain in ``EXCLUDED_JUDGES`` below so the history
#: is self-documenting and they can be re-opted-in via ``--panel``.
PRIMARY_PANEL: tuple[Judge, ...] = (
    Judge(
        shortname="kimi-k2.5",
        model_id="moonshotai.kimi-k2.5",
        provider="Moonshot AI",
        family="non-anthropic-non-amazon",
        role="judge",
        notes="Chinese lineage, strong general reasoning. Primary tie-breaker judge.",
    ),
    Judge(
        shortname="kimi-k2-thinking",
        model_id="moonshot.kimi-k2-thinking",
        provider="Moonshot AI",
        family="non-anthropic-non-amazon",
        role="judge",
        notes="Thinking variant; use on fabrication_present where CoT matters.",
    ),
    Judge(
        shortname="deepseek-v3.2",
        model_id="deepseek.v3.2",
        provider="DeepSeek",
        family="non-anthropic-non-amazon",
        role="judge",
        notes="Different training corpus from Kimi; good disagreement signal.",
    ),
    Judge(
        shortname="minimax-m2.5",
        model_id="minimax.minimax-m2.5",
        provider="MiniMax",
        family="non-anthropic-non-amazon",
        role="judge",
        notes="Third Chinese-lab vote for ensemble diversity.",
    ),
    Judge(
        shortname="glm-5",
        model_id="zai.glm-5",
        provider="Z.AI",
        family="non-anthropic-non-amazon",
        role="judge",
        notes="GLM-5; GLM-4.7 available as lightweight fallback.",
    ),
    Judge(
        shortname="qwen3-next-80b",
        model_id="qwen.qwen3-next-80b-a3b",
        provider="Qwen (Alibaba)",
        family="non-anthropic-non-amazon",
        role="judge",
        notes="Qwen3-Next 80B MoE; structured-output strong.",
    ),
    Judge(
        shortname="palmyra-x5",
        model_id="writer.palmyra-x5-v1:0",
        provider="Writer",
        family="non-anthropic-non-amazon",
        role="judge",
        notes="Enterprise-text purpose-trained; different bias axis.",
    ),
    Judge(
        shortname="nemotron-super-3",
        model_id="nvidia.nemotron-super-3-120b",
        provider="NVIDIA",
        family="non-anthropic-non-amazon",
        role="judge",
        notes="NVIDIA's largest open reasoning model.",
    ),
)

#: Judges evaluated and dropped.  Reachable via ``resolve()`` /
#: ``--panel`` for anyone who wants to re-test them, but not part of
#: the default ensemble.
EXCLUDED_JUDGES: tuple[Judge, ...] = (
    Judge(
        shortname="mistral-large-3",
        model_id="mistral.mistral-large-3-675b-instruct",
        provider="Mistral AI",
        family="non-anthropic-non-amazon",
        role="judge",
        notes=(
            "Dropped 2026-04-21 after study ab0bf2eeb481fdd2: produced 4/50 "
            "unparseable responses (worst rubric discipline in panel)."
        ),
    ),
    Judge(
        shortname="magistral-small",
        model_id="mistral.magistral-small-2509",
        provider="Mistral AI",
        family="non-anthropic-non-amazon",
        role="judge",
        notes=(
            "Dropped 2026-04-21 alongside mistral-large-3 (Mistral family "
            "entirely excluded pending rubric-discipline fix)."
        ),
    ),
)

#: Within-family holdout judges.  Kept explicitly so the study can
#: *measure* the within-family bias rather than silently avoid it.
WITHIN_FAMILY_HOLDOUT: tuple[Judge, ...] = (
    Judge(
        shortname="opus-4-7",
        model_id="global.anthropic.claude-opus-4-7",
        provider="Anthropic",
        family="anthropic",
        role="within-family-holdout",
        notes=(
            "Delta vs non-Anthropic ensemble = the bias we are measuring. "
            "Uses global CRIS profile; direct model ID rejects on-demand."
        ),
    ),
    Judge(
        shortname="sonnet-4-6",
        model_id="global.anthropic.claude-sonnet-4-6",
        provider="Anthropic",
        family="anthropic",
        role="within-family-holdout",
        notes="Intra-family agreement is its own data point. Uses global CRIS.",
    ),
)

#: Bulk Amazon lane: cheap, fast, current-gen only.  Nova Pro v1 is
#: explicitly excluded as stale.
BULK_PANEL: tuple[Judge, ...] = (
    Judge(
        shortname="nova-2-lite",
        model_id="amazon.nova-2-lite-v1:0:256k",
        provider="Amazon",
        family="amazon",
        role="bulk",
        notes="Bulk classifier: hedge counting, entity spotting, reversal markers.",
    ),
    Judge(
        shortname="nova-2-mm-embed",
        model_id="amazon.nova-2-multimodal-embeddings-v1:0",
        provider="Amazon",
        family="amazon",
        role="embed",
        notes="Embedding path for dedup-neighbors cosine-contamination filter.",
    ),
)

#: Flat lookup keyed by shortname (CLI-facing) and by model_id (internal).
#: Includes ``EXCLUDED_JUDGES`` so ``--panel mistral-large-3`` still
#: resolves for anyone re-testing a dropped judge.
_ALL: tuple[Judge, ...] = (
    *PRIMARY_PANEL,
    *WITHIN_FAMILY_HOLDOUT,
    *BULK_PANEL,
    *EXCLUDED_JUDGES,
)
_BY_SHORTNAME: dict[str, Judge] = {j.shortname: j for j in _ALL}
_BY_MODEL_ID: dict[str, Judge] = {j.model_id: j for j in _ALL}


def resolve(name: str) -> Judge:
    """Resolve a shortname or model ID to a ``Judge``.

    Raises ``KeyError`` with the full catalog when the name is unknown —
    agents parsing stderr get a concrete hint on what is available.
    """
    if name in _BY_SHORTNAME:
        return _BY_SHORTNAME[name]
    if name in _BY_MODEL_ID:
        return _BY_MODEL_ID[name]
    available = ", ".join(sorted(_BY_SHORTNAME))
    raise KeyError(f"unknown judge {name!r}; available: {available}")


def panel(names: list[str] | tuple[str, ...]) -> list[Judge]:
    """Resolve a list of shortnames/model IDs into Judge records, preserving order."""
    return [resolve(n) for n in names]


def all_primary() -> tuple[Judge, ...]:
    """Return the full primary (non-within-family) panel."""
    return PRIMARY_PANEL


def all_within_family() -> tuple[Judge, ...]:
    """Return the within-family holdout panel."""
    return WITHIN_FAMILY_HOLDOUT


def all_bulk() -> tuple[Judge, ...]:
    """Return the Amazon bulk/embed lane."""
    return BULK_PANEL


def all_excluded() -> tuple[Judge, ...]:
    """Return judges that were evaluated and dropped from the primary panel."""
    return EXCLUDED_JUDGES


def catalog() -> list[Judge]:
    """Every judge in the catalog, primary first, then holdout, bulk, excluded."""
    return [*PRIMARY_PANEL, *WITHIN_FAMILY_HOLDOUT, *BULK_PANEL, *EXCLUDED_JUDGES]
