"""CLI interface package for the v2 hexagonal reshape.

Destination for the cyclopts command surface (``cli.py``), output rendering
(``output.py``), and the install banner, moved here in MIGRATION Phase C step 7.
Each subcommand becomes a thin composition root that builds the adapters it
needs and injects them into the application use-case, preserving the
lazy-import discipline (adapters constructed inside command bodies, not at
module top — pinned by ``test_cli_import_is_lean``).

Additive in T-1-1: package skeleton only.
"""

from __future__ import annotations
