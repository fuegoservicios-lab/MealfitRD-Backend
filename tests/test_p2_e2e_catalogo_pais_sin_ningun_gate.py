"""Guards de G33: el catálogo vivo por país tiene un gate e2e acotado y read-only."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = BACKEND_ROOT / ".github" / "workflows" / "ci.yml"
LIVE_TEST = BACKEND_ROOT / "tests" / "test_p1_country_system_f2.py"


def test_ci_declares_optional_live_country_catalog_gate() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "Country catalog live (read-only)" in workflow
    start = workflow.index("Country catalog live (read-only)")
    end = workflow.index("\n      - name:", start)
    step = workflow[start:end]
    assert "tests/test_p1_country_system_f2.py" in step
    assert '-m "e2e"' in step
    assert "secrets.NEON_DATABASE_URL" in step
    assert "NEON_DATABASE_URL" in step and "exit 0" in step
    assert 'pytest tests/ -m "e2e"' not in step, (
        "el gate abrió todos los e2e; @pytest.mark.e2e también habilita escrituras"
    )


def test_live_country_subset_does_not_call_write_facade() -> None:
    tree = ast.parse(LIVE_TEST.read_text(encoding="utf-8"), filename=str(LIVE_TEST))
    write_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "execute_sql_write"
    ]
    assert not write_calls, f"el subconjunto dejó de ser read-only: líneas {write_calls}"


def test_local_deploy_gate_runs_live_subset_when_neon_is_configured() -> None:
    run_ci = BACKEND_ROOT.parent / "scripts" / "run_ci.ps1"
    if not run_ci.is_file():
        pytest.skip(f"repo workspace ausente: {run_ci.parent.parent}")
    source = run_ci.read_text(encoding="utf-8")
    assert '"tests/test_p1_country_system_f2.py"' in source
    assert '"e2e"' in source
    assert "NEON_DATABASE_URL" in source
    assert "Backend country catalog live (read-only)" in source
