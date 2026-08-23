"""Guard del guard de G32: la auditoría debe mirar el CI backend ejecutable."""

from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
COUNTRY_COVERAGE_TEST = TESTS_ROOT / "test_p2_ci_country_coverage.py"


def test_country_ci_guard_discovers_backend_and_workspace_workflows() -> None:
    source = COUNTRY_COVERAGE_TEST.read_text(encoding="utf-8")
    assert 'BACKEND_ROOT / ".github" / "workflows" / "ci.yml"' in source
    assert 'WORKSPACE_ROOT / ".github" / "workflows" / "ci.yml"' in source
    assert "pytest.skip" not in source


def test_country_ci_guard_fails_when_no_workflow_exists() -> None:
    source = COUNTRY_COVERAGE_TEST.read_text(encoding="utf-8")
    assert "assert workflows" in source or "assert _WORKFLOWS" in source
    assert "no encontré ningún workflow" in source

