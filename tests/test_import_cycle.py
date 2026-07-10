"""Regression tests for the app.py <-> agent_loop.py import cycle (issue #17).

`app.py` and `agent_loop.py` used to reference `repl.UI` only under
`TYPE_CHECKING`, papering over what would otherwise be a real circular
import (repl.py imports App). Both now depend on a UI protocol defined
in `agentsh.ui_protocol`, a leaf module neither of them cycles back
through. These tests import each module first, in isolated subprocesses,
to catch a reintroduced cycle regardless of which module happens to be
imported first -- the failure mode a same-process test would miss since
Python caches already-imported modules.
"""

import subprocess
import sys

import pytest

from agentsh import ui_protocol
from agentsh.repl import UI

_MODULE_IMPORT_ORDERS = [
    ["agentsh.app", "agentsh.agent_loop", "agentsh.repl"],
    ["agentsh.agent_loop", "agentsh.app", "agentsh.repl"],
    ["agentsh.repl", "agentsh.app", "agentsh.agent_loop"],
    ["agentsh.main"],
]


@pytest.mark.parametrize("modules", _MODULE_IMPORT_ORDERS)
def test_modules_import_cleanly_in_any_order(modules: list[str]) -> None:
    """Each module imports without NameError or ImportError, in isolation.

    Run in a subprocess so no module is already cached from a prior test,
    and with sys.dont_write_bytecode so a stale .pyc can't mask a
    reintroduced eager-evaluation NameError.
    """
    script = "; ".join(f"import {m}" for m in modules)
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        capture_output=True,
        text=True,
        cwd="src",
    )
    assert result.returncode == 0, result.stderr


def test_repl_ui_satisfies_shared_protocol() -> None:
    """repl.UI structurally satisfies agentsh.ui_protocol.UI.

    Confirms the concrete implementation still matches the shape that
    app.py and agent_loop.py now type against, without either importing
    repl.py.
    """
    from unittest.mock import MagicMock

    ui = UI(MagicMock())
    assert isinstance(ui, ui_protocol.UI)
