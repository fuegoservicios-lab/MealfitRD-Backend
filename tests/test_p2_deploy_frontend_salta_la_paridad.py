"""Guards de G34: un deploy frontend conserva la paridad con backend."""

from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_frontend_cross_repo_marker_is_registered_and_auto_classified() -> None:
    ini = (BACKEND_ROOT / "pytest.ini").read_text(encoding="utf-8")
    conftest = (BACKEND_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "frontend_cross_repo:" in ini
    assert "def pytest_collection_modifyitems" in conftest
    assert "frontend_cross_repo" in conftest
    assert "def _is_frontend_cross_repo_test_file" in conftest


def test_country_parity_file_is_in_cross_repo_subject_set() -> None:
    from conftest import _is_frontend_cross_repo_test_file

    country_parity = BACKEND_ROOT / "tests" / "test_p1_country_system_f0.py"
    backend_only = BACKEND_ROOT / "tests" / "test_synonyms.py"
    assert _is_frontend_cross_repo_test_file(country_parity)
    assert not _is_frontend_cross_repo_test_file(backend_only)


def test_frontend_deploy_uses_backend_cross_repo_only_mode() -> None:
    workspace = BACKEND_ROOT.parent
    run_ci = workspace / "scripts" / "run_ci.ps1"
    deploy = workspace / "deploy-mealfit.ps1"
    if not run_ci.is_file() or not deploy.is_file():
        pytest.skip(f"repo workspace ausente: {workspace}")

    run_source = run_ci.read_text(encoding="utf-8")
    deploy_source = deploy.read_text(encoding="utf-8")
    assert "[switch]$BackendCrossRepoOnly" in run_source
    assert '-m "frontend_cross_repo and not e2e"' in run_source
    assert "@('-BackendCrossRepoOnly')" in deploy_source

