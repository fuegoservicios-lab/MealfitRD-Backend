"""[P1-PROD-AUDIT-1 · 2026-05-23] Coverage gate (`fail_under` en CI) es
decisión de producto deferred a >500 MAU — NO un gap técnico.

Gap aparente (audit production-readiness 2026-05-23, B-P1-7):
    El audit external flageó "Coverage sin `fail_under` en CI — 774 tests
    pero % real desconocido".

Decisión documentada en `.coveragerc` (P3-COVERAGE-HEATMAP · 2026-05-20):
    > "Sin fail_under / sin gates en CI todavía (decisión MVP <100 MAU).
    >  Cuando crucemos 500 MAU o agreguemos 2do dev:
    >    - Añadir `fail_under = 60` (baseline conservador tras primera medición).
    >    - Activar el job CI con `pytest --cov=. --cov-fail-under=60`.
    >    - Subir el threshold cada PR."

Razón:
    Coverage gate sin medición prior produce False Positives a escala:
      - Tests que cubren caminos no-crítico (utility helpers) inflan el %
        pero no protegen contra los modos de fallo reales (IDOR, lost-update,
        deadlock).
      - Coverage % es proxy proxy de calidad de tests — el gap real es
        tests funcionales E2E que exerciten paths críticos.

    El repo ya tiene ~770 tests parser-based + funcionales que enforzan
    invariantes específicas (I1-I8 plan_id lifecycle, P0-AGENT-1 user_id
    override, P1-AUDIT-3 historial exemption, etc.). Estos son MÁS valiosos
    que coverage % alto, y son lo que el repo prioriza.

    Cuando crucemos 500 MAU:
      (a) Medir baseline real con `./scripts/run_coverage.sh`.
      (b) Setear `fail_under = X` donde X = baseline - 5%.
      (c) Activar job CI `backend-coverage`.
      (d) Subir threshold incrementalmente.

Este test ancla la decisión:
    Si alguien añade `fail_under = N` a `.coveragerc` sin pasar por la
    decisión (sin documentar el bump a 500 MAU + medición previa), el
    test falla con copy explicativo.

    Análogo al patrón:
      - `test_p3_i18n_deferred.py` (i18n deferred to global expansion).
      - `test_p3_chat_safety_off_decision.py` (safety_settings relajados).

Tooltip-anchor: P1-PROD-AUDIT-1-COVERAGE-DECISION | audit 2026-05-23.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_COVERAGERC = _BACKEND_ROOT / ".coveragerc"


def test_coveragerc_exists():
    assert _COVERAGERC.exists(), (
        f".coveragerc ausente en {_COVERAGERC}. La decisión deferred "
        f"se basa en este archivo; si fue movido, actualizar este test."
    )


def test_coveragerc_has_no_fail_under_active():
    """Si `fail_under = N` está activo (no comentado) sin justificación,
    falla. La decisión es: NO gate hasta >500 MAU.

    Si alguien legítimamente cruzó 500 MAU + midió + decidió activar:
      - Bumpear cap aquí: actualizar este test para esperar
        `fail_under = X` y documentar en commit msg.
    """
    text = _COVERAGERC.read_text(encoding="utf-8")
    # Match `fail_under = N` que NO esté precedido por `#`.
    # Más simple: split por línea, filtrar comments.
    active_lines = [
        line for line in text.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    active_text = "\n".join(active_lines)
    m = re.search(r"fail_under\s*=\s*(\d+)", active_text)
    assert m is None, (
        f"`.coveragerc` activó `fail_under = {m.group(1) if m else '?'}` sin "
        f"pasar por la decisión documentada (>500 MAU + medición previa).\n\n"
        f"Si genuinamente cruzaste 500 MAU:\n"
        f"  (a) Documentar en commit msg el % medido (./scripts/run_coverage.sh).\n"
        f"  (b) Setear `fail_under = X` donde X = baseline - 5%.\n"
        f"  (c) Activar job `backend-coverage` en `.github/workflows/ci.yml`.\n"
        f"  (d) Actualizar este test para esperar el valor configurado.\n"
    )


def test_coveragerc_documents_decision():
    """`.coveragerc` debe documentar inline la decisión deferred. Sin esto,
    futuro mantenedor no entiende POR QUÉ no hay gate.
    """
    text = _COVERAGERC.read_text(encoding="utf-8")
    keywords = ["MAU", "fail_under", "decisión", "gate", "500"]
    found = [k for k in keywords if k in text or k.lower() in text.lower()]
    assert len(found) >= 3, (
        f".coveragerc NO documenta inline la decisión deferred ({len(found)}/5 "
        f"keywords presentes: {found}). Sin documentación, un PR futuro podría "
        f"añadir `fail_under = 80` arbitrario rompiendo CI sin entender el "
        f"trade-off. Restaurar comentario P3-COVERAGE-HEATMAP."
    )


def test_decision_referenced_in_p3_coverage_heatmap():
    """El P-fix `P3-COVERAGE-HEATMAP` debe ser mencionado en .coveragerc
    o tener un test homónimo que ancla la decisión (auto-satisfecho por
    este test).
    """
    text = _COVERAGERC.read_text(encoding="utf-8")
    has_marker = "P3-COVERAGE-HEATMAP" in text or "I6" in text
    assert has_marker, (
        "`.coveragerc` perdió referencia al P-fix `P3-COVERAGE-HEATMAP` "
        "(o `I6`). Sin marker, la decisión queda huérfana en el código."
    )
