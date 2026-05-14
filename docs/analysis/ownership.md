# claude-sql · Ownership

"Share" is each folder's top contributor's commit count divided by that folder's total commits, as a whole percent. The window is the project's full git history on `HEAD`. At report time `HEAD` had 115 commits. A folder where the top owner holds more than 70% has one effective owner. If that person is unreachable, change in that folder slows.

`claude-sql` is a near-solo project. `git shortlog -sn --all` lists three contributors. Laith Al-Saadoon has 170 commits across all refs. `bonk-ai[bot]` has 14. `dependabot[bot]` has 6. That sits under the 5-contributor floor for meaningful folder-level ownership analysis. The table is included for completeness. The bus-factor finding is trivial: one person can speak for nearly every folder.

| Folder | Top owner | Share | Total contributors |
|---|---|---|---|
| `src/claude_sql/` | Laith Al-Saadoon | 98% | 2 |
| `tests/` | Laith Al-Saadoon | 95% | 2 |
| `docs/` | Laith Al-Saadoon | 93% | 2 |
| `.erpaval/strategy/` | Laith Al-Saadoon | 100% | 1 |
| `.claude/skills/` | Laith Al-Saadoon | 100% | 1 |
| `docs/rfc/` | Laith Al-Saadoon | 100% | 1 |
| `docs/queries/` | Laith Al-Saadoon | 100% | 1 |
| `.github/actions/` | Laith Al-Saadoon | 100% | 1 |
| `.erpaval/solutions/` | Laith Al-Saadoon | 79% | 2 |
| `docs/adr/` | Laith Al-Saadoon | 67% | 2 |
| `.github/workflows/` | bonk-ai[bot] | 45% | 3 |

## Single points of failure

The entire codebase is effectively one owner (see intro).
