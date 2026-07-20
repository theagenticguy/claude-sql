# claude-sql · Ownership

"Share" here is commit share over the available git history — 216 commits spanning 2026-04-19 to 2026-07-20 (~3 months). For each folder the top owner's share is that owner's commits touching the folder divided by all commits touching it, as a whole-percent integer. A share above 70% is the bus-factor threshold: at that concentration, one contributor holds the working context for the folder, and their unavailability stalls changes there.

This repo is near-solo. Three git identities — `alsaadoonlaith@gmail.com`, `9553966+theagenticguy@users.noreply.github.com`, and `lalsaado@amazon.com` — are all one human, Laith Al-Saadoon, and are merged into a single owner ("Laith") for every share below. The only other contributors are two bots: `dependabot[bot]` (lockfile bumps) and an agent bot (agent commits). Across the whole repo, Laith authored 200 of 216 commits (93%); bots account for the remaining 16. There is no `CODEOWNERS` file, so every owner figure is git-derived with nothing to reconcile against. Ownership analysis is therefore effectively "one person" — the table below is best read as a map of where that single-owner knowledge concentration is most load-bearing, not as a contest between people.

One structural caveat: the hexagonal reshape (commits 4038edb..7670b4c) created the `domain / application / infrastructure / interfaces` layer paths recently. Counting commits against those literal paths without `--follow` undercounts them to 2-4 each, so the ranking uses the aggregate `src/claude_sql` tree (all history, pre- and post-reshape) as the honest source-code bus-factor signal.

| Folder | Top owner | Share | Total contributors |
|---|---|---|---|
| `src/claude_sql` | Laith Al-Saadoon | 99% | 2 |
| `proofs` | Laith Al-Saadoon | 100% | 1 |
| `mise.toml` | Laith Al-Saadoon | 100% | 1 |
| `tests` | Laith Al-Saadoon | 97% | 2 |
| `README.md` | Laith Al-Saadoon | 96% | 2 |
| `docs` | Laith Al-Saadoon | 96% | 2 |
| `uv.lock` | Laith Al-Saadoon | 94% | 2 |
| `pyproject.toml` | Laith Al-Saadoon | 91% | 2 |
| `.erpaval/solutions` | Laith Al-Saadoon | 84% | 2 |
| `.github/workflows` | dependabot[bot] | 37% | 3 |

## Single points of failure

Every source, test, docs, and proof path in the repo is a single point of failure on one human. The bullets below name the load-bearing concentrations; the mitigation for all of them is the same in kind — get a second human into the review loop — and the codebase-wide fix is to bring on at least one co-maintainer before the bus factor of 1 becomes a delivery risk.

- `src/claude_sql` — Laith Al-Saadoon (99%). Add a co-maintainer with commit access and route all four hexagonal layers through paired review so the port/adapter contracts are held by more than one person.
- `proofs` — Laith Al-Saadoon (100%). Run a knowledge-transfer session on the Lean 4 invariant proofs, since this tree is newest and the smallest talent pool understands it.
- `mise.toml` — Laith Al-Saadoon (100%). Document the task/gate wiring in-repo so the five-gate `check` pipeline can be maintained by a second contributor.
- `tests` — Laith Al-Saadoon (97%). Cross-train a reviewer on the fixture-directory and Bedrock-mock conventions so test authorship is not a single-owner skill.
- `README.md` — Laith Al-Saadoon (96%). Fold user-facing docs review into any future co-maintainer's onboarding so the public surface has a second editor.
- `docs` — Laith Al-Saadoon (96%). Assign a second reader for the architecture and v2 design docs so the design intent survives one person's absence.
- `uv.lock` — Laith Al-Saadoon (94%). Keep dependabot's automated bumps flowing (already a partial mitigation) and add a second human approver for lockfile PRs.
- `pyproject.toml` — Laith Al-Saadoon (91%). Pair-review changes to the ruff selector set, import-linter contract, and Bedrock/DuckDB pins, since these gate every other contributor's build.
- `.erpaval/solutions` — Laith Al-Saadoon (84%). Treat the captured best-practices notes as shared team memory and invite other contributors to append lessons so the knowledge base is not one author's journal.

`.github/workflows` is the one folder Laith does not outright own: dependabot (7 commits) and the agent bot (5) split it with Laith (7), leaving no author above 37%. This is bot churn from automated CI-config bumps, not a second human maintainer — the CI pipeline is still Laith's to reason about, and it is a single point of failure in practice even though the raw share does not cross 70%.
