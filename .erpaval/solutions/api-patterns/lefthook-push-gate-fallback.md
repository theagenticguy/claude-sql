# Lefthook pre-push `@{push}` gate needs the `HEAD~` fallback on first push

**Category:** api-patterns
**Tags:** lefthook, git-hooks, pre-push, push-files, fresh-branch
**Applies to:** lefthook v2.x pre-push hooks that scope by changed files

## Bug

A common lefthook pattern scopes a pre-push job to "did Python change
since the last push?":

```yaml
# BROKEN on first push of a new branch
- name: pytest
  files: "git diff --name-only @{push} HEAD"
  glob: "**/*.py"
  run: "{uv} pytest tests/ --no-header -q"
```

On the **first push** of a branch with no configured upstream,
`@{push}` resolves to nothing and `git` exits `128` ("no upstream
configured for branch '<name>'"). Lefthook interprets that as
command failure and blocks the push — even though no upstream is
exactly the situation where you want pytest to gate the whole branch,
not a pointless "since last push" subset.

**Observed:** `fatal: no upstream configured for branch
'chore/stack-modernization'` → `pytest exit status 128` → push
blocked.

## Fix

Chain a `HEAD~` fallback with `||`:

```yaml
- name: pytest
  files: "git diff --name-only @{push} HEAD || git diff --name-only HEAD~"
  glob: "**/*.py"
  run: "{uv} pytest tests/ --no-header -q"
  skip:
    - merge
    - rebase
  fail_text: "pytest failed — run `mise run test` locally before pushing."
```

On pushes where `@{push}` resolves, you get the scoped diff. On first
push (or any push where upstream isn't set), the fallback produces
`HEAD~`'s diff, which for a brand-new-branch-with-one-commit is
effectively "all files added in this branch" — correct gating.

## Why the "v2.1 idioms" advice got it wrong

Recent lefthook v2.1 guidance suggests dropping the fallback because
"`@{push}` always resolves on a real push." In practice, **the very
first push** is a real push, and `@{push}` does *not* resolve until
after the `git push -u` sets up upstream tracking. Drop-the-fallback
research that ignores first-push is incorrect.

## See also

- Lefthook config reference: https://lefthook.dev/configuration/
- Original research packet (with the bad recommendation):
  `.erpaval/sessions/session-638df1/research-lefthook.yaml`
- Fix applied in claude-sql commit `43f0960` on
  `chore/stack-modernization` (session-638df1).
