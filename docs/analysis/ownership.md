# claude-sql · Ownership

"Share" here means a folder's commit share over the available git history: of all commits that touched a folder, the percentage authored by its single most-active contributor. The history window runs from the first commit on `2026-04-19` to the latest on `2026-05-14`, about four weeks (`git log --pretty=format:%ai`); `HEAD` holds 121 commits (`git rev-list --count HEAD`). A top-owner share above 70% signals a bus-factor risk — one person holds nearly all the change knowledge for that path, so their absence stalls work there.

This project is near-solo. `git shortlog -sne --all` lists five identities, but three are aliases of one person (Laith Al-Saadoon, via `alsaadoonlaith@gmail.com` with 141 commits, `9553966+theagenticguy@users.noreply.github.com` with 41, and `lalsaado@amazon.com` with 9, totaling 191) and the remaining two are automation bots (`bonk-ai[bot]`, 14 commits; `dependabot[bot]`, 7). After merging the aliases, the effective human bus factor is one — below the five-contributor floor for meaningful folder-level ownership analysis. The table below merges Laith's three emails into a single owner and is included for completeness; the bus-factor finding is trivial: one person can speak for nearly every folder.

| Folder | Top owner | Share | Total contributors |
|---|---|---|---|
| `.claude/skills` | Laith Al-Saadoon | 100% | 1 |
| `.erpaval/strategy` | Laith Al-Saadoon | 100% | 1 |
| `src/claude_sql/core` | Laith Al-Saadoon | 100% | 1 |
| `src/claude_sql/analytics` | Laith Al-Saadoon | 100% | 1 |
| `src/claude_sql/evals` | Laith Al-Saadoon | 100% | 1 |
| `src/claude_sql/provenance` | Laith Al-Saadoon | 100% | 1 |
| `src/claude_sql/app` | Laith Al-Saadoon | 100% | 1 |
| `tests` | Laith Al-Saadoon | 96% | 2 |
| `docs` | Laith Al-Saadoon | 93% | 2 |
| `.erpaval/solutions` | Laith Al-Saadoon | 82% | 2 |
| `docs/adr` | Laith Al-Saadoon | 67% | 2 |
| `.github/workflows` | bonk-ai[bot] | 42% | 3 |

Notes on the table:

- The five `src/claude_sql/*` layer folders (`core`, `analytics`, `evals`, `provenance`, `app` — PEP 420 namespace sub-packages of the one `claude-sql` package) each carry exactly one commit — they were all created in a single refactor, `a5589ee` "collapse 5-package workspace into one claude-sql package (#81)" (`git log --oneline -- src/claude_sql/`), which folded the short-lived `packages/*` workspace (introduced by `982595b` "split into 5 PEP 420 namespace packages under uv workspace (#60)") back under one distribution. Their 100% share reflects that single bulk commit, not deep contribution history.
- `.github/workflows` is the only folder where a human is not the top committer: `bonk-ai[bot]` (5 commits) edges Laith (4) and `dependabot[bot]` (3). It remains automation plus one human.

## Single points of failure

The entire codebase is effectively one owner (see intro). With a human bus factor of one, the mitigation that matters is repository-level rather than per-folder: onboard a second human contributor, add a `CODEOWNERS` file naming a backup reviewer (none exists today), and run a knowledge-transfer pass over the `src/claude_sql/*` layer source tree so the namespace-sub-package layout and intent are not held by a single person.
