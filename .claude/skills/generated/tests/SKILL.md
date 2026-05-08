---
name: tests
description: "Skill for the Tests area of claude-sql. 67 symbols across 11 files."
---

# Tests

67 symbols | 11 files | Cohesion: 97%

## When to Use

- Working with code in `tests/`
- Understanding how test_sessions_count, test_todo_events_count, test_todo_state_current_last_wins work
- Modifying tests-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/test_sql_views.py` | _connect, test_sessions_count, test_todo_events_count, test_todo_state_current_last_wins, test_todo_events_forward_compat (+12) |
| `tests/test_skill_views.py` | _connect, test_skill_invocations_tool, test_skill_invocations_slash, test_skill_invocations_combined_count, test_skill_usage_without_catalog (+6) |
| `tests/test_skills_catalog.py` | _write_skill, _write_command, _plugin_manifest, _build_fixture_layout, _settings_for (+3) |
| `tests/test_llm_worker.py` | _make_mock_client, _captured_body, test_invoke_body_uses_output_config_not_tool_use, test_invoke_body_has_adaptive_thinking_by_default, test_invoke_body_drops_thinking_when_disabled (+2) |
| `tests/test_install_source.py` | _write_receipt, test_read_install_source_directory, test_read_install_source_git, test_read_install_source_malformed_toml, test_read_install_source_skips_other_tools (+1) |
| `tests/test_v2_analytics.py` | _make_classifications_parquet, _make_clusters_parquet, _make_cluster_terms_parquet, analytics_settings, test_schemas_additional_properties_false (+1) |
| `tests/test_session_bounds.py` | _write_session_jsonl, _user_text_record, fixture_con |
| `tests/test_community_worker.py` | _write_jsonl, _msg, connected_settings |
| `tests/test_schemas.py` | _walk, test_all_object_subschemas_forbid_additional |
| `tests/test_friction_worker.py` | _make_user_friction_parquet, test_register_analytics_creates_user_friction_view |

## Entry Points

Start here when exploring this area:

- **`test_sessions_count`** (Function) — `tests/test_sql_views.py:315`
- **`test_todo_events_count`** (Function) — `tests/test_sql_views.py:321`
- **`test_todo_state_current_last_wins`** (Function) — `tests/test_sql_views.py:332`
- **`test_todo_events_forward_compat`** (Function) — `tests/test_sql_views.py:343`
- **`test_task_spawns_detected`** (Function) — `tests/test_sql_views.py:353`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `test_sessions_count` | Function | `tests/test_sql_views.py` | 315 |
| `test_todo_events_count` | Function | `tests/test_sql_views.py` | 321 |
| `test_todo_state_current_last_wins` | Function | `tests/test_sql_views.py` | 332 |
| `test_todo_events_forward_compat` | Function | `tests/test_sql_views.py` | 343 |
| `test_task_spawns_detected` | Function | `tests/test_sql_views.py` | 353 |
| `test_subagent_sessions_count` | Function | `tests/test_sql_views.py` | 362 |
| `test_subagent_sessions_meta_join` | Function | `tests/test_sql_views.py` | 371 |
| `test_macros_cost_estimate` | Function | `tests/test_sql_views.py` | 384 |
| `test_macros_todo_velocity` | Function | `tests/test_sql_views.py` | 395 |
| `test_macros_subagent_fanout` | Function | `tests/test_sql_views.py` | 403 |
| `test_model_used_macro` | Function | `tests/test_sql_views.py` | 409 |
| `test_explain_has_pushdown_markers` | Function | `tests/test_sql_views.py` | 415 |
| `test_describe_all_covers_every_view` | Function | `tests/test_sql_views.py` | 427 |
| `test_list_macros_includes_all` | Function | `tests/test_sql_views.py` | 452 |
| `test_invoke_body_uses_output_config_not_tool_use` | Function | `tests/test_llm_worker.py` | 29 |
| `test_invoke_body_has_adaptive_thinking_by_default` | Function | `tests/test_llm_worker.py` | 58 |
| `test_invoke_body_drops_thinking_when_disabled` | Function | `tests/test_llm_worker.py` | 72 |
| `test_invoke_parses_output_field` | Function | `tests/test_llm_worker.py` | 86 |
| `test_invoke_falls_back_to_text_block` | Function | `tests/test_llm_worker.py` | 100 |
| `test_skill_invocations_tool` | Function | `tests/test_skill_views.py` | 235 |

## How to Explore

1. `gitnexus_context({name: "test_sessions_count"})` — see callers and callees
2. `gitnexus_query({query: "tests"})` — find related execution flows
3. Read key files listed above for implementation details
