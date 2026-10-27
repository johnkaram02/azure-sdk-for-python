# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[3]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# Resolve the package root before importing helpers when the sample is run directly.
from samples.util import agent_tracing, agent_tracing_async  # pylint: disable=wrong-import-position

__all__ = ["agent_tracing", "agent_tracing_async"]
