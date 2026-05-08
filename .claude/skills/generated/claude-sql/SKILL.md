---
name: claude-sql
description: "Skill for the Claude_sql area of claude-sql. 92 symbols across 16 files."
---

# Claude_sql

92 symbols | 16 files | Cohesion: 94%

## When to Use

- Working with code in `src/`
- Understanding how sync, resolve_format, emit_dataframe work
- Modifying claude_sql-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `src/claude_sql/judge_worker.py` | parse_rubric, _bedrock_client, run_async, bounded, to_parquet (+7) |
| `src/claude_sql/session_text.py` | session_text_corpus, _load_session_order, _load_messages_text, _load_tool_calls, _load_tool_results (+4) |
| `src/claude_sql/skills_catalog.py` | _parse_frontmatter, _coerce_str, _read_plugin_manifest, _walk_user_skills, _walk_plugins (+3) |
| `src/claude_sql/checkpointer.py` | _strip_tz, filter_unchanged, _stale_or_equal, mark_completed, _attach_tz (+3) |
| `src/claude_sql/output.py` | resolve_format, emit_dataframe, emit_json, to_payload, classify_duckdb_error (+2) |
| `src/claude_sql/freeze.py` | to_dict, _read_rubric, _git_commit_sha, _studies_root, freeze (+2) |
| `src/claude_sql/embed_worker.py` | discover_unembedded, _build_bedrock_client, _invoke_bedrock_sync, _embed_one_batch, embed_documents_async (+2) |
| `src/claude_sql/retry_queue.py` | _connect, _backoff_delta, enqueue, drain, mark_done (+1) |
| `src/claude_sql/kappa_worker.py` | cohens_kappa, bootstrap_kappa_ci, compute_pairwise, fleiss_kappa, bootstrap_fleiss_ci (+1) |
| `src/claude_sql/ungrounded_worker.py` | extract_claims, push, count_in, check_claims, detect |

## Entry Points

Start here when exploring this area:

- **`sync`** (Function) — `src/claude_sql/skills_catalog.py:296`
- **`resolve_format`** (Function) — `src/claude_sql/output.py:53`
- **`emit_dataframe`** (Function) — `src/claude_sql/output.py:61`
- **`emit_json`** (Function) — `src/claude_sql/output.py:103`
- **`to_payload`** (Function) — `src/claude_sql/output.py:126`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `sync` | Function | `src/claude_sql/skills_catalog.py` | 296 |
| `resolve_format` | Function | `src/claude_sql/output.py` | 53 |
| `emit_dataframe` | Function | `src/claude_sql/output.py` | 61 |
| `emit_json` | Function | `src/claude_sql/output.py` | 103 |
| `to_payload` | Function | `src/claude_sql/output.py` | 126 |
| `classify_duckdb_error` | Function | `src/claude_sql/output.py` | 172 |
| `emit_error` | Function | `src/claude_sql/output.py` | 202 |
| `run_or_die` | Function | `src/claude_sql/output.py` | 220 |
| `to_dict` | Function | `src/claude_sql/freeze.py` | 66 |
| `freeze` | Function | `src/claude_sql/freeze.py` | 114 |
| `replay` | Function | `src/claude_sql/freeze.py` | 150 |
| `list_studies` | Function | `src/claude_sql/freeze.py` | 171 |
| `discover_unembedded` | Function | `src/claude_sql/embed_worker.py` | 84 |
| `embed_documents_async` | Function | `src/claude_sql/embed_worker.py` | 248 |
| `embed_query` | Function | `src/claude_sql/embed_worker.py` | 318 |
| `run_backfill` | Function | `src/claude_sql/embed_worker.py` | 349 |
| `enqueue` | Function | `src/claude_sql/retry_queue.py` | 78 |
| `drain` | Function | `src/claude_sql/retry_queue.py` | 113 |
| `mark_done` | Function | `src/claude_sql/retry_queue.py` | 143 |
| `pending_count` | Function | `src/claude_sql/retry_queue.py` | 167 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Run → Render_prompt` | cross_community | 4 |
| `Run → _converse_once` | cross_community | 4 |
| `Run → Parse_judge_response` | cross_community | 4 |
| `Sync → _parse_frontmatter` | intra_community | 4 |
| `Sync → _coerce_str` | intra_community | 4 |
| `Sync → _read_plugin_manifest` | intra_community | 4 |
| `Run → Estimate_tokens` | cross_community | 3 |
| `Run → _bedrock_client` | intra_community | 3 |
| `Run → Bounded` | intra_community | 3 |
| `Detect → Push` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "sync"})` — see callers and callees
2. `gitnexus_query({query: "claude_sql"})` — find related execution flows
3. Read key files listed above for implementation details
