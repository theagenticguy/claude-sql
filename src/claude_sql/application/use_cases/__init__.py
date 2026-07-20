"""Use-case orchestrations for the v2 hexagonal reshape.

Each module here is an application-layer entrypoint that composes domain math
with injected ports/adapters. Depends inward on ``domain`` and on
``application.ports``; never on ``interfaces``.
"""

from __future__ import annotations
