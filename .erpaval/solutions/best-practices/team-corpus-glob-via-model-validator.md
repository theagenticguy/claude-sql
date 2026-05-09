---
title: Re-derive Settings glob fields via model_validator — never break user pins
track: knowledge
category: best-practices
module: src/claude_sql/config.py
component: pydantic-settings v2
severity: info
tags: [pydantic-settings, model_validator, defaults, user-pin, settings-derivation]
applies_when:
  - "Adding a new Settings field that should re-derive other Settings fields when set"
  - "Need to distinguish 'user explicitly pinned a value' from 'value is at factory default'"
pattern: |
  When one Settings field (e.g. `team_corpus_root`) should re-derive sibling fields (e.g.
  `default_glob`, `subagent_glob`, `subagent_meta_glob`), use `@model_validator(mode="after")`
  and detect "user pinned" by comparing against the **factory function output**, never the
  literal default string:

      @model_validator(mode="after")
      def _derive_team_corpus_globs(self) -> Self:
          if self.team_corpus_root is None:
              return self
          # Detect user-pinned fields by calling the factory; never compare to literal strings.
          user_pinned = (
              self.default_glob != _default_glob()
              or self.subagent_glob != _default_subagent_glob()
              or self.subagent_meta_glob != _default_subagent_meta_glob()
          )
          if user_pinned:
              return self  # respect the pin
          root = self.team_corpus_root.expanduser().resolve()
          object.__setattr__(self, "default_glob", f"{root}/*/projects/*/*.jsonl")
          object.__setattr__(self, "subagent_glob", f"{root}/*/projects/*/subagents/agent-*.jsonl")
          object.__setattr__(self, "subagent_meta_glob", f"{root}/*/projects/*/subagents/meta-*.jsonl")
          return self

  Key disciplines:
    - Compare to factory output, NOT a literal string. If someone refactors the factory,
      the literal-string check silently mis-detects "pinned" → breaks user overrides.
    - Use `object.__setattr__` to mutate frozen-by-default Settings (mirror existing
      validator precedent like `_resolve_concurrency_alias`).
    - Return `Self` (from stdlib `typing` on Python 3.13+).
    - Replace, do not union — when team_corpus_root is set, the personal corpus root
      should NOT also apply (else dry-run cost estimates double-count).
example_files:
  - src/claude_sql/config.py        # the model_validator
  - tests/test_team_corpus.py       # tests assert user-pin precedence + env-var path
---

# Why this matters

The naive approach — "if `default_glob == '~/.claude/projects/*/*.jsonl'` then override" — looks
fine until a future refactor changes the factory default. The literal string check then silently
mis-detects "pinned" vs "default" and either trashes user overrides or fails to derive when it
should. Calling the factory is one extra call but is refactor-safe.

The `object.__setattr__` pattern mirrors the existing precedent in claude-sql's
`_resolve_concurrency_alias` validator. Don't use plain attribute assignment on a `BaseSettings`
inside a validator — pydantic's frozen semantics will reject it depending on `model_config`.

# Example

```python
# tests/test_team_corpus.py — the user-pin test
def test_team_corpus_user_pinned_glob_wins(tmp_path: Path) -> None:
    explicit = "custom/*.jsonl"
    s = Settings(team_corpus_root=tmp_path, default_glob=explicit)
    assert s.default_glob == explicit  # user pin wins
    # Subagent globs stay at factory defaults (no partial rewrite when user pinned the primary)
    assert s.subagent_glob == _default_subagent_glob()
```
