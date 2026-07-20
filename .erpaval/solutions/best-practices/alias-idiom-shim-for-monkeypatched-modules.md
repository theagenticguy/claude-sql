# Alias-idiom shim (`sys.modules` swap) for modules whose internals get monkeypatched

**Category:** best-practices
**Tags:** refactoring, monkeypatch, shims, module-moves, pytest, ty
**Session:** session-4fb0fd (v2 hexagonal reshape)

## Lesson

When relocating a module during a large refactor, a plain re-export shim
(`from new.home import name`) is only safe when tests patch the shim's
**entrypoints**. If any test patches a module-**internal** private and then
calls the entrypoint expecting the patch to bite inside (e.g.
`monkeypatch.setattr(worker, "_classify_chunk", fake)` then `worker.run(...)`),
a re-export shim silently breaks the patch: the test rebinds the shim's
attribute while the moved entrypoint resolves names in its own (real) module.

**Fix:** the alias idiom —

```python
# old/path.py (shim)
from new.home import entrypoint, _private_a, _private_b  # for ty's static resolution
import sys as _sys
from new import home as _real
_sys.modules[__name__] = _real
```

The `sys.modules` swap makes the historic import path and the real module the
**same object**, so internal-private patches, by-string patches
(`monkeypatch.setattr("old.path.fn", ...)`), and lazy `from old.path import x`
all keep working with zero test edits. The explicit re-imports above the swap
are dead at runtime but required so `ty` resolves `from old.path import name`
statically.

## How to choose

1. Grep the test suite for BOTH `from old.path import` names AND
   `setattr(old_module, "_private"` / `setattr("old.path._private"` forms.
2. Entrypoint-only patches → plain named re-export shim is fine (and cleaner).
3. Any internal-private patch-then-call-entrypoint pattern → alias idiom, always.

## Why

Discovered independently by two Wave-3 agents (T-3-3 trajectory/conflicts,
T-3-4 classify/friction) after prototyping both shim shapes. Plain shims kept
`test_analyze_chain` green (entrypoint patches) but broke
`test_friction_worker_llm` / `test_trajectory_coverage` (internal patches).
Related: [[read-side-dedup-companions-write-side-replace]] for cache-shape moves.
