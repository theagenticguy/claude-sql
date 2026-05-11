---
title: Pinned tool.uv required-version needs mise-shimmed PATH for git hooks
track: knowledge
category: conventions
module: dev environment / git hooks
component: uv + mise + lefthook
severity: warning
tags: [uv, mise, lefthook, pre-commit, pyproject]
applies_when:
  - pyproject.toml has `[tool.uv] required-version = ">=X.Y.Z"`
  - mise manages uv (`mise.toml` `[tools] uv = "..."`)
  - lefthook (or any git hook) calls `uv run` directly
  - the developer's interactive shell was started before the new uv version was installed
pattern: |
  When `[tool.uv] required-version` is bumped in `pyproject.toml`, every uv
  invocation aborts with "Required uv version >=X.Y.Z does not match the
  running version A.B.C" until the binary on PATH actually meets the pin.
  `mise install uv@latest` puts a new binary under
  `~/.local/share/mise/installs/uv/<new>/...` but it is NOT automatically
  promoted to PATH for the running shell — `mise activate` only re-binds
  `$PATH` on the next directory change or shell rehash.

  Critical: git hooks (pre-commit, pre-push, commit-msg) run under whatever
  PATH the parent shell exports at the moment `git commit` / `git push` is
  invoked. If the shell's PATH still points at the old uv install dir, the
  hook fails with the version error and the commit / push is aborted.

  Two fixes:
  1. (Quickest, in-session): prepend the new uv install dir before the git
     command:
     `export PATH="$HOME/.local/share/mise/installs/uv/<new>/uv-<arch>:$PATH"`
  2. (Persistent): re-source the shell rc, or run `mise activate <shell>` in
     a new shell, so the mise-managed `~/.local/share/mise/shims/uv` is the
     resolution winner. Verify with `which uv && uv --version`.
example_files:
  - mise.toml
  - pyproject.toml
---

# Why this matters

A bumped `[tool.uv] required-version` is a silent landmine for any
mid-session contributor: their `mise run check` works (because mise tasks
re-resolve through their own shim), but `git commit` blows up because the
git hook inherits the parent shell's stale PATH. The error message points at
`uv self update`, which doesn't work when uv is mise-managed (the mise-
installed binary is not write-allowed for self-update).

# Example

```bash
# Symptoms
$ git commit -m "..."
┃  ty ❯ error: Required uv version `>=0.11.7` does not match the running version `0.10.7`.
       Update `uv` by running `uv self update`.

# In-session fix (one-shot)
$ export PATH="$HOME/.local/share/mise/installs/uv/0.11.12/uv-aarch64-apple-darwin:$PATH"
$ git commit -m "..."
✔️ ruff-lint
✔️ ruff-format
✔️ ty
✔️ commitizen

# Persistent fix
$ mise reshim
$ exec zsh                # or open a fresh terminal so PATH picks up the new shim
$ which uv                # /Users/...mise/shims/uv
$ uv --version            # 0.11.12 (or higher)
```
