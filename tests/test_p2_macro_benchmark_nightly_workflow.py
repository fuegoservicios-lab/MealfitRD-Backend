"""[P2-NIGHTLY-WORKFLOW-RELOCATE · 2026-06-18] (audit fresco P2) El workflow nightly de no-regresión de
precisión de macros (gap-audit G13) vivía SOLO en el workspace-root `.github/workflows/` — que NO es repo
git → GitHub Actions del repo backend nunca lo ejecutaba. La "validación continua de precisión" era de facto
inexistente en CI. Este test ancla su ubicación versionada (backend/.github/workflows/) para que un futuro
move/delete falle CI. La ACTIVACIÓN (secrets DEEPSEEK/NEON/COHERE) sigue siendo acción explícita del owner;
el job auto-salta sin ellos.
"""
from __future__ import annotations

from pathlib import Path

_BE = Path(__file__).resolve().parent.parent
_YML = _BE / ".github" / "workflows" / "macro-benchmark-nightly.yml"


def test_nightly_workflow_present_in_backend_repo():
    assert _YML.exists(), (
        "macro-benchmark-nightly.yml debe vivir bajo backend/.github/workflows/ (alcanzable por los "
        "GitHub Actions del repo backend). Si solo está en el workspace-root, nunca corre."
    )


def test_nightly_workflow_invokes_benchmark_gate():
    txt = _YML.read_text(encoding="utf-8")
    assert "benchmark_macro_compliance.py" in txt, "el workflow debe invocar el script de benchmark"
    assert "macro_baseline.json" in txt, "el gate debe anclar contra el baseline commiteado"
    assert "--max-mape-rise" in txt and "--max-band-drop" in txt, "el gate de no-regresión debe pasar tolerancias"
    assert "P2-NIGHTLY-WORKFLOW-RELOCATE" in txt, "marker de relocación ausente"
    # Paths ROOT-relative: el repo backend tiene scripts/tests/requirements en la raíz (sin subdir backend/).
    assert "backend/requirements.txt" not in txt, "el path NO debe llevar prefijo backend/ (raíz del repo = backend)"
    assert "backend/scripts/" not in txt, "el path del script NO debe llevar prefijo backend/"
    assert "scripts/benchmark_macro_compliance.py" in txt and "tests/fixtures/macro_baseline.json" in txt


def test_benchmark_script_and_baseline_exist():
    assert (_BE / "scripts" / "benchmark_macro_compliance.py").exists()
    assert (_BE / "tests" / "fixtures" / "macro_baseline.json").exists()
