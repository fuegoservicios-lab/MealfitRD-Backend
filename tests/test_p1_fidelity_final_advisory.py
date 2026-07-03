"""[P1-FIDELITY-FINAL-ADVISORY · 2026-07-03] Fidelity de pool → advisory en intento final.

Follow-up del gym baseline (docs/gym_baseline_2026_07_03.json): el eje "entrega" promedió 45.5
porque 13/20 planes se entregaron con `_review_failed_but_delivered` — y el minado del log mostró
que el driver dominante (63 rechazos) era el gate de skeleton-fidelity: el day-gen moderno diverge
del protein_pool del planner LEGÍTIMAMENTE (gates de variedad same-day que piden proteínas
distintas, biblioteca de transformaciones, taste, apetecibilidad), el gate fallaba los 3 intentos
y era el ÚNICO gate de calidad sin degradación a advisory en el final → plan marcado + alert I5
aunque slots/creatividad/coherencia puntuaran ~99.

Cierra: en el intento FINAL, si el ÚNICO error de ensamblaje es fidelity (sin estructurales),
degrada a `_skeleton_fidelity_advisory_final` (espejo exacto de slot/variety/dish/sodium) y el
plan se entrega limpio. Intentos 1..N-1 conservan retry con severity high (presión intacta).
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_marker_bumped():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-FIDELITY-FINAL-ADVISORY" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-03"


def test_knob_default_on():
    assert re.search(
        r'SKELETON_FIDELITY_FINAL_ADVISORY\s*=\s*_env_bool\("MEALFIT_SKELETON_FIDELITY_FINAL_ADVISORY",\s*True\)',
        _GO,
    ), "el knob debe nacer ON (la serie del gym ya midió el problema — playbook medir→actuar completado)"


def test_final_advisory_branch_wired():
    idx = _GO.index("assembly_errors = skeleton_fidelity_errors + structural_coherence_errors")
    blk = _GO[idx:idx + 4000]
    # la degradación aplica SOLO si: knob ON + hay fidelity + NO hay estructurales + intento final
    assert "SKELETON_FIDELITY_FINAL_ADVISORY" in blk
    assert "not structural_coherence_errors" in blk, \
        "errores ESTRUCTURALES jamás degradan a advisory (siguen siendo retry/entrega marcada)"
    assert "_sf_attempt >= MAX_ATTEMPTS" in blk, "la degradación es SOLO en el intento final"
    assert '_skeleton_fidelity_advisory_final"] = True' in blk, "el flag advisory debe persistir en el plan"
    # el path de rechazo sigue existiendo (intentos 1..N-1 y errores estructurales)
    assert "approved = False" in blk
    assert '_severity_max(severity, "high")' in blk, \
        "fidelity en intentos no-finales conserva severity high (presión de retry intacta)"


def test_advisory_branch_never_extends_issues():
    """En la rama advisory el plan NO debe cargar los assembly_errors como issues de rechazo
    (eso re-marcaría review_passed=False aguas abajo)."""
    idx = _GO.index("_sf_final_advisory = (")
    blk_advisory = _GO[idx:_GO.index("else:", idx)]
    assert "issues.extend(assembly_errors)" not in blk_advisory
