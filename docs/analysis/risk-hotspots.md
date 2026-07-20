# claude-sql · Risk hotspots

Risk here is a composite of two signals the repository actually emits: 30-day git churn (rename-followed, so a file's history survives the hexagonal reshape) and suppression density — the count of `# noqa` / `type: ignore` escape hatches a file carries out of the strict ruff 32-family and ty-strict gates. Source-level static analysis is clean: `uv run ruff check src/` returns zero findings, `.sarif/bandit.sarif` has zero Python-source results, and every semgrep finding lands on `.github/*.yml` or `pyproject.toml`, not on `src/**/*.py`. TODO/FIXME markers are effectively absent (two in all of `src/`). With no source findings to grade, the Open-findings column reports the suppression count as the finding-pressure proxy. The score is `1.0 × commits + 2.0 × suppressions + 0.001 × LOC`, the last term a blast-radius tiebreak.

Two limitations shape the reading. First, the 30-day window (`2026-06-21` to `2026-07-20`) is dominated by the hexagonal reshape that landed inside it — commits `4038edb`..`7670b4c` moved every module from the old `core`/`analytics`/`app` tree into `domain`/`application`/`infrastructure`/`interfaces`. That churn is migration, not defect activity: the files at the top of this table are there because they were relocated and rewired, not because they broke. Genuine in-window behavior changes are few (`4bf1981` simhash vectorization in ingest, `d3fd346` lance index API, and three `fix(analytics)`/`fix(skills)` commits that touched the SQL now in `duckdb_views.py`). Read the Trend column as "recently disturbed," not "recently buggy." Second, this is a solo repository — every file's top owner is Laith Al-Saadoon at 100% commit share, so the Top-owner column carries no bus-factor signal beyond confirming single ownership. Trend thresholds derive from the 69-file churn distribution (median 1, σ 1.10): `↑ rising` = more than 2 commits, `→ flat` otherwise; no file falls below the zero floor, so `↓ falling` is unused.

| File | Trend | Open findings | Top owner | Citation |
|---|---|---|---|---|
| `interfaces/cli/app.py` | ↑ rising | 2 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/interfaces/cli/app.py` (2121 LOC) |
| `infrastructure/duckdb_views.py` | ↑ rising | 1 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/infrastructure/duckdb_views.py` (2394 LOC) |
| `domain/structure/community.py` | → flat | 4 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/domain/structure/community.py` (326 LOC) |
| `application/use_cases/trajectory.py` | ↑ rising | 2 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/application/use_cases/trajectory.py` (903 LOC) |
| `infrastructure/settings.py` | ↑ rising | 1 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/infrastructure/settings.py` (544 LOC) |
| `application/use_cases/community.py` | → flat | 2 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/application/use_cases/community.py` (434 LOC) |
| `domain/errors.py` | → flat | 2 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/domain/errors.py` (149 LOC) |
| `infrastructure/lance_store.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/infrastructure/lance_store.py` (300 LOC) |
| `infrastructure/embedding/cohere_bedrock.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/infrastructure/embedding/cohere_bedrock.py` (256 LOC) |
| `infrastructure/llm_analytics/sonnet_bedrock.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/infrastructure/llm_analytics/sonnet_bedrock.py` (81 LOC) |
| `application/use_cases/embed.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/application/use_cases/embed.py` (394 LOC) |
| `application/use_cases/ingest.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/application/use_cases/ingest.py` (365 LOC) |

## Per-file drill-down

### `interfaces/cli/app.py`

What's there: the cyclopts CLI composition root — the `Common` parameter dataclass, settings/format resolution helpers, provider-override plumbing, and the ~18 subcommands that wire the application use-cases to a terminal (`src/claude_sql/interfaces/cli/app.py:180`, `src/claude_sql/interfaces/cli/app.py:250`). It is the single thickest human-authored surface in the tree and the only module that legitimately reaches across every other layer, which is exactly why a change here has the widest blast radius.

Recent activity: 5 commits in 30 days, all reshape-driven (`7670b4c` extracted peek/search/community SQL out of the old CLI, `4038edb` relocated it, `968d3d2`/`077f362`/`a0ec803` added provider flags and dropped eval/provenance commands). `↑ rising`. Owners: Laith Al-Saadoon, 100%. Findings: 2 suppression warnings (`# noqa` at `src/claude_sql/interfaces/cli/app.py:61` and `src/claude_sql/interfaces/cli/app.py:66`); no scanner errors.

### `infrastructure/duckdb_views.py`

What's there: every DuckDB view and macro DDL for the analytics surface, split across `register_raw`, `register_views`, `register_macros`, `register_vss`, and `register_analytics` (`src/claude_sql/infrastructure/duckdb_views.py:490`, `src/claude_sql/infrastructure/duckdb_views.py:1244`). At 2394 LOC it is the largest file in the repo, and its correctness is SQL-string correctness — the class of bug the type checker and ruff cannot see, which is what makes its size a risk rather than mere bulk.

Recent activity: 6 commits in 30 days — the highest raw count — but the same reshape moves plus three genuine in-window `fix(analytics)`/`fix(skills)` commits (`c2645b4`, `aa2dc3f`, `66b7a07`) that corrected macro logic before the code was relocated here. `↑ rising`. Owners: Laith Al-Saadoon, 100%. Findings: 1 suppression warning (`# noqa` at `src/claude_sql/infrastructure/duckdb_views.py:1899`); no scanner errors.

### `domain/structure/community.py`

What's there: the Leiden+CPM community-detection math over a mutual-kNN cosine graph of session centroids — graph construction, the auto-γ resolution profile, the CPM partition run, and the warn-only disconnected-component check (`src/claude_sql/domain/structure/community.py:96`, `src/claude_sql/domain/structure/community.py:175`). This is the file with the highest suppression density in the tree, and it ranks top-3 on that finding-proxy alone despite flat churn.

Recent activity: 1 commit in 30 days (relocated by the reshape, otherwise untouched). `→ flat`. Owners: Laith Al-Saadoon, 100%. Findings: 4 suppression warnings — all `B009` (`getattr` with a constant) reaching into the C-backed `leidenalg`/`igraph` objects whose attributes ruff's static model cannot resolve (`src/claude_sql/domain/structure/community.py:129`, `src/claude_sql/domain/structure/community.py:135`); no scanner errors. The suppressions are a legitimate FFI accommodation, not latent bugs, but they mark the spot where the strict gates stop protecting the code.

### `application/use_cases/trajectory.py`

What's there: the windowed per-session trajectory classifier — chunking each session into ≤16-window anchor-sharing groups, echoing `(prev_uuid, curr_uuid)` pairs back from the LLM, running one bounded retry of missing windows, and stamping neutral placeholder rows so a refusing chunk never wedges the pipeline (`src/claude_sql/application/use_cases/trajectory.py:10`, `src/claude_sql/application/use_cases/trajectory.py:419`). At 903 LOC it is the most intricate control flow in the application layer, mixing chunk math, retry accounting, and structured-output parsing.

Recent activity: 3 commits in 30 days (reshape relocation plus the Strands/LLM-analytics rewire in `968d3d2`, then storage-port wiring in `d473338`). `↑ rising`. Owners: Laith Al-Saadoon, 100%. Findings: 2 suppression warnings (`src/claude_sql/application/use_cases/trajectory.py:556`, `src/claude_sql/application/use_cases/trajectory.py:751`); no scanner errors.

### `infrastructure/settings.py`

What's there: the `Settings` BaseSettings object and its ~20 default-path factory functions covering every cache location, glob, and Bedrock knob (`src/claude_sql/infrastructure/settings.py:33`, `src/claude_sql/infrastructure/settings.py:60`). It moved out of the old `core` package in the reshape and is now a wide configuration seam that almost every other module reads, so a wrong default here propagates silently rather than failing loud.

Recent activity: 5 commits in 30 days — the reshape relocation plus provider-config additions from the pluggable-embeddings and Strands work (`077f362`, `968d3d2`). `↑ rising`. Owners: Laith Al-Saadoon, 100%. Findings: 1 suppression warning (`# noqa` at `src/claude_sql/infrastructure/settings.py:407`); no scanner errors.

## See also

- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 4 shared source citations
- [claude-sql · Tech debt](../insights/tech-debt.md) — 4 shared source citations
- [claude-sql · Module map](../architecture/module-map.md) — 3 shared source citations
- [claude-sql · Processes](../behavior/processes.md) — 3 shared source citations
- [claude-sql · Contract map](../insights/contract-map.md) — 3 shared source citations
