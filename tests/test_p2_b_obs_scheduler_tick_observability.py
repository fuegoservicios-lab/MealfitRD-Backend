"""[P2-B-OBS · 2026-05-11] Anchor + cross-link del P-fix de observability
del autohealer del scheduler.

Bug original (audit final 2026-05-11):
    `_resolve_stale_scheduler_alerts` y `_alert_scheduler_cascade_missed`
    solo emitían pipeline_metrics en el "hot path":
      - El sweep solo loggeaba INFO cuando barría ≥1 alert.
      - El detector solo persistía `_scheduler_cascade_autoheal` cuando
        detectaba cascada.

    En path "cold" (sweep healthy con 0 alerts, detector healthy sin
    cascada) ambos producían 0 filas en `pipeline_metrics`. Eso es
    indistinguible de "el cron está MISSED" — post-incidente no había
    forma de answer "¿estuvo corriendo durante la ventana del fail?".

Fix (P2-B-OBS):
    1. `_resolve_stale_scheduler_alerts` emite SIEMPRE
       `pipeline_metrics._scheduler_alerts_sweep_tick` (count=0 cuando
       healthy, count=N cuando hay backlog).
    2. `_alert_scheduler_cascade_missed` emite SIEMPRE
       `pipeline_metrics._scheduler_cascade_check_tick` (con
       metadata.cascade_detected). El emit ocurre ANTES del early-return
       cuando no hay cascada — sino el tick no aparecería en path
       healthy (defeats the purpose).
    3. Tests detallados de cobertura en
       `test_p0_new_2_autoheal_cascade_sweep.py` (mismo archivo que
       agrupa los tests del autohealer: P0-NEW-1-AUTOHEAL,
       P0-NEW-2-AUTOHEAL, P2-B-OBS).

Este archivo existe principalmente como anchor del cross-link
requerido por `test_p2_hist_audit_14_marker_test_link.py`: el slug
del marker `_LAST_KNOWN_PFIX` (`p2_b_obs` tras strip de dashes) debe
matchear al menos un archivo `tests/test_p2_b_obs*.py`. Mover los
8 tests detallados aquí mejoraría cohesión por-Pfix pero rompería
git blame / split-cleanliness — preferimos un anchor minimal.
"""
from __future__ import annotations

from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DETAILED_TESTS = _BACKEND_ROOT / "tests" / "test_p0_new_2_autoheal_cascade_sweep.py"


def test_detailed_observability_tests_exist():
    """Los 8 tests detallados de P2-B-OBS viven en
    `test_p0_new_2_autoheal_cascade_sweep.py`. Si alguien borra ese
    archivo, este anchor falla y el marker `_LAST_KNOWN_PFIX` queda
    huérfano (cross-link roto)."""
    assert _DETAILED_TESTS.exists(), (
        f"Archivo SSOT de tests P2-B-OBS desaparecido: {_DETAILED_TESTS}. "
        "Los 8 tests detallados de observability viven ahí (no en este "
        "anchor). Restaurar el archivo o relocar los tests."
    )


def test_detailed_tests_cover_sweep_tick():
    """El archivo SSOT debe cubrir el sweep tick (`_scheduler_alerts_sweep_tick`)."""
    text = _DETAILED_TESTS.read_text(encoding="utf-8")
    assert "_scheduler_alerts_sweep_tick" in text, (
        "Tests de sweep tick desaparecieron de "
        "`test_p0_new_2_autoheal_cascade_sweep.py`. El gap original "
        "(sweep invisible en path healthy) reaparece sin ese test."
    )


def test_detailed_tests_cover_cascade_check_tick():
    """El archivo SSOT debe cubrir el cascade check tick
    (`_scheduler_cascade_check_tick`)."""
    text = _DETAILED_TESTS.read_text(encoding="utf-8")
    assert "_scheduler_cascade_check_tick" in text, (
        "Tests de cascade check tick desaparecieron de "
        "`test_p0_new_2_autoheal_cascade_sweep.py`. El gap original "
        "(detector invisible en path healthy) reaparece sin ese test."
    )


def test_implementation_uses_obs_tick_in_cron_tasks():
    """Sanity: el código de producción debe seguir emitiendo ambos ticks.
    Defensa contra refactor que borre el emit pero deje los tests verdes
    (e.g. extrayendo a helper que pase por mock)."""
    cron_py = _BACKEND_ROOT / "cron_tasks.py"
    text = cron_py.read_text(encoding="utf-8")
    assert "_scheduler_alerts_sweep_tick" in text, (
        "`cron_tasks.py` ya no emite `_scheduler_alerts_sweep_tick`. "
        "Tests pasan por mocks pero producción se queda sin métrica — "
        "regresión de P2-B-OBS."
    )
    assert "_scheduler_cascade_check_tick" in text, (
        "`cron_tasks.py` ya no emite `_scheduler_cascade_check_tick`. "
        "Tests pasan por mocks pero producción se queda sin métrica — "
        "regresión de P2-B-OBS."
    )
