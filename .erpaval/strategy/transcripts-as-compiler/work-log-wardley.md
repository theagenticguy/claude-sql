# Product strategy work log — wardley-cartographer

**Status:** COMPLETE
**Role:** wardley-cartographer
**Slug:** transcripts-as-compiler
**Working directory:** `.erpaval/strategy/transcripts-as-compiler/`
**Your output file:** `.erpaval/strategy/transcripts-as-compiler/wardley-packet.md`

## 1. Objective

Map the prompt→agent→diff→transcript→review value chain on Wardley's Genesis→Custom→Product→Commodity axis. Emit a build-vs-buy read per component, 3–5 gameplay moves, and a map-implied diagnosis for the Rumelt architect.

## 2. Scope

- **Input**: `framing.md` (read in full), `rumelt-packet.md` if already in flight, external evidence on coding-agent runtimes, transcript formats, vector DB landscape, AI code review products
- **Output**: `.erpaval/strategy/transcripts-as-compiler/wardley-packet.md`
- **Role reference**: `${CLAUDE_PLUGIN_ROOT}/skills/product-strategy/references/roles/wardley-cartographer.md`
- **Framework file**: `${CLAUDE_PLUGIN_ROOT}/skills/product-strategy/references/frameworks/wardley-maps.md`

Sections: User Need, Value Chain, Evolution Axis (with evidence), Climatic Patterns, Gameplay Moves, Build-vs-Buy Read, Map-implied Diagnosis, Attribution Note, Citations.

Out of scope: writing the kernel; composing the memo.

## 3. Inputs

- `framing.md`
- Sibling packets if in flight
- Web searches (2–4 max) to verify evolution-stage claims with concrete dates and provider counts:
    - Coding-agent runtime landscape 2026 (Claude Code, Cursor, Amp, Windsurf, Copilot Workspace, Cline, Aider)
    - Vector index market 2025–2026 (DuckDB VSS, pgvector, Pinecone, Weaviate, Milvus, Turbopuffer, Chroma)
    - Embedding provider landscape (Cohere Embed v4, OpenAI text-embed-3, Voyage, Bedrock Titan)
    - AI code review / PR-sidebar products (GitHub Copilot code review, Sourcegraph Cody Review, Cursor Bugbot)
    - Transcript artifact standards (git notes, commit trailers, OpenTelemetry GenAI semconv, OpenAI Responses API conversations)

## 4. Success criteria

Every component on the value chain has a stage with cited evidence. Build-vs-Buy call per component. At least 3 gameplay moves naming specific components + climatic patterns. Map-implied diagnosis is one tight paragraph.

## 5. Anti-goals

- Don't hand-wave evolution stages — every claim has a citation
- Don't draw a map that ignores what claude-sql already consumes (Bedrock, DuckDB VSS, Cohere Embed v4, HDBSCAN, UMAP, networkx louvain) — ground in reality
- Don't run "Build" on anything Commodity

---

## Work log

- **Read inputs.** `framing.md` (COMPLETE), `rumelt-packet.md` (skeleton — no diagnosis yet), `minto-outline.md` (skeleton), CLAUDE.md project instructions. Wardley reference + role reference located under `/efs/lalsaado/workplace/personal-plugins/personal-plugins/skills/product-strategy/references/`.
- **Grounding searches (4, parallel).** Coding-agent runtime landscape 2026, vector DB providers 2025–2026, AI PR-review product landscape 2026, OpenTelemetry GenAI semconv status. All cited inline in the packet.
- **Key findings:**
  - Coding-agent runtime has ≥7 credible products (Claude Code, Cursor, Windsurf, Copilot, Cline, Aider, Devin) + a $12.8B market — clearly **Product** stage, tipping toward **Commodity** on the low end.
  - Vector DB has ≥8 production providers — **Product → Commodity**.
  - AI PR-review is a Red Queen race (Copilot Review, Cursor Bugbot, Graphite Diamond, Sourcegraph Cody, Greptile) — all ship diff-commentary; **none ship transcript provenance**. That's the Genesis gap.
  - OTel GenAI semconv is in Development/experimental — transcript-format-standardization is early-Genesis and real.
- **Filled packet in numbered order:** User need → Value chain → Evolution axis (with cited evidence per row) → Climatic patterns → Gameplay → Build-vs-buy → Contradictions → Map-implied diagnosis → Attribution → Citations. Flipped status.

---

## Validation

- [x] Value chain dependencies go downward (11 components, top = reviewer surface, leaf = compute)
- [x] Every component's stage cites evidence (URLs + CLAUDE.md refs + framing.md refs inline in the evolution table)
- [x] Build-vs-buy matches stage (Genesis=Build; Custom=Build+extend; Product=Consume; Commodity=Consume). Contradictions section explicitly checks for inversions.
- [x] Gameplay references specific components (reviewer surface, binding, analytics, runtime) and specific climatic patterns (fragmentation, Red Queen, agent-runtime commoditization)
- [x] Map-implied diagnosis is one paragraph for the Rumelt architect to lift
- [x] Status flipped to COMPLETE

---

## Summary

Map filled end-to-end, grounded in 4 bounded web searches (2026-05-08) plus CLAUDE.md + framing.md citations. Central finding: top 3 components (reviewer surface, PR↔transcript binding, bound-transcript analytics) are Genesis/Custom; everything from the agent runtime down is Product/Commodity with ≥7 providers per row. Binding sits on the Genesis/Product boundary, which confirms the framing's binding-as-crux hypothesis. Map-implied diagnosis is a single paragraph the Rumelt architect can lift verbatim.
