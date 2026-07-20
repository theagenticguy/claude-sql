"""Application layer (L-application) for the v2 hexagonal reshape.

Owns the port Protocols (``ports.py``) that use-cases depend on, and — as later
waves land — the use-case orchestrations (the analytics worker entrypoints and
the ``analyze`` chain lifted out of the CLI). Depends inward on ``domain`` only;
never on ``infrastructure`` or ``interfaces``. Concrete adapters are injected by
the composition root (``interfaces/cli``).

Additive in MIGRATION Phase C step 1 (T-1-1): this wave lands the port surface
only. The import-linter contract does not yet name these packages; that lands in
T-5.
"""

from __future__ import annotations
