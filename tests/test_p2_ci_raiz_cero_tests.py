"""Guards de G31: la telemetría e2e vive en el CI ejecutable del backend."""

from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = BACKEND_ROOT / ".github" / "workflows" / "ci.yml"


def _workflow() -> str:
    assert WORKFLOW.is_file(), f"workflow backend ausente: {WORKFLOW}"
    return WORKFLOW.read_text(encoding="utf-8")


def test_backend_owns_unverified_e2e_notice_before_root_ci_is_removed() -> None:
    source = _workflow()
    assert "Report unverified e2e count" in source
    step = source[source.index("Report unverified e2e count") :]
    assert 'pytest tests/ -m "e2e" --collect-only' in step
    assert "::notice" in step
    assert "if: always()" in step
    assert "exit 1" not in step and "|| false" not in step

