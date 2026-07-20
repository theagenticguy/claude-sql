"""Interfaces layer (L-interfaces) for the v2 hexagonal reshape.

The outermost hexagon: driving adapters that translate the outside world into
use-case calls. Today that is the cyclopts CLI, output rendering, and the
install banner; the importable ``ClaudeSql(...)`` facade lands here too. This
layer is the composition root — it builds concrete infrastructure adapters and
injects them into application use-cases. Depends on every inner layer; nothing
depends on it.

Additive in MIGRATION Phase C step 1 (T-1-1): this wave creates the package
skeleton only. ``cli.py`` / ``output.py`` / the install banner move here in T-7.
"""

from __future__ import annotations
