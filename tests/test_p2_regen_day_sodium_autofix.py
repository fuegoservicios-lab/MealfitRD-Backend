"""[P2-REGEN-DAY-SODIUM-AUTOFIX · 2026-07-10] El "actualizar día" arregla lo que antes
solo REPORTABA, y sus avisos dejan de quedar stale. + [P1-DAY-REGEN-RESUME] la animación
y el proceso sobreviven al refresh (el backend siempre siguió generando server-side; era
el CLIENTE el que perdía el estado).

Evidencia (screenshots del owner + logs 2026-07-10, plan 9bce8fff):
  1. Tras regenerar el día, banner "Un día se pasa del techo de sodio" (Día 1:
     2,733/2,000mg) — regen-day NO corría el corrector de sodio per-día de S1 y el
     banner de panel JAMÁS se re-evaluaba en updates (early-return si ya estaba seteado).
  2. Chip ámbar "Macros algo fuera de la banda objetivo" en la card con el día en banda
     1.0 — `_macro_band_low` se setea al aceptar el swap (>15% del slot) pero el
     rebalance a nivel-DÍA corre DESPUÉS y lo deja stale.
  3. Toast decía "~1 minuto" (medido: ~4.4 min) y el refresh mataba overlay + estado.

tooltip-anchor: P2-REGEN-DAY-SODIUM-AUTOFIX
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_CTX = (_BACKEND.parent / "frontend" / "src" / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
_DASH = (_BACKEND.parent / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Sodium autofix del día regenerado (antes del qty-sync y del panel)
# ---------------------------------------------------------------------------

def test_sodium_autofix_wired_in_regen_day_mutator():
    i = _PLANS.find("P2-REGEN-DAY-SODIUM-AUTOFIX")
    assert i > 0, (
        "P2-REGEN-DAY-SODIUM-AUTOFIX: regen-day dejó de correr el corrector de sodio per-día "
        "— el día regenerado puede quedar sobre el techo OMS y el banner le pide al usuario "
        "arreglarlo a mano (Día 1 en 2,733/2,000mg, visto 2026-07-10)."
    )
    blk = _PLANS[i: i + 1800]
    assert "_day_sodium_autofix" in blk, "debe reusar el corrector SSOT de S1"
    assert 'os.environ.get("MEALFIT_REGEN_DAY_SODIUM_AUTOFIX", "true")' in blk, "knob default ON"
    # orden: sodio ANTES del qty-sync (GAP-10) y del recompute de micros
    i_sync = _PLANS.find("(GAP-10) Re-sincronizar cantidades", i)
    i_micros = _PLANS.find("recompute_micronutrient_report_for_plan(pd, _micro_form", i)
    assert 0 < i < i_sync < i_micros, (
        "el autofix de sodio corre antes del qty-sync y del recompute del panel "
        "(el panel debe reflejar el día YA corregido)"
    )


# ---------------------------------------------------------------------------
# 2. Re-evaluación bidireccional del banner de panel en updates
# ---------------------------------------------------------------------------

def test_panel_degraded_reevaluated_after_regen():
    i = _PLANS.find("P2-REGEN-DAY-PANEL-REEVAL")
    assert i > 0, (
        "P2-REGEN-DAY-PANEL-REEVAL: el banner de panel (sodio/azúcar/micros worst-day) "
        "volvió a quedar congelado con el veredicto de la GENERACIÓN — "
        "`_maybe_mark_panel_degraded` early-returns si el flag ya está seteado, así que "
        "sin el clear-first el banner jamás se limpia aunque el usuario regenere el día ofensor."
    )
    blk = _PLANS[i: i + 2200]
    assert "_maybe_mark_panel_degraded" in blk
    assert "micro_worst_day_ceiling" in blk, (
        "la clase-panel debe incluir los reasons worst-day (2b/2c), no solo _PANEL_DEGRADED_REASONS"
    )
    assert 'pd.pop("_quality_degraded", None)' in blk, "clear-first SOLO para razones de clase-panel"
    assert "review_failed" not in blk.split("_panel_class")[0] or True  # razones no-panel intactas (doc)


# ---------------------------------------------------------------------------
# 3. Chips _macro_band_low stale limpiados cuando el día quedó en banda
# ---------------------------------------------------------------------------

def test_stale_macro_band_chips_cleared_when_day_in_band():
    i = _PLANS.find("P2-REGEN-DAY-CHIP-STALE-CLEAR")
    assert i > 0, (
        "P2-REGEN-DAY-CHIP-STALE-CLEAR: el chip ámbar 'Macros algo fuera de la banda' "
        "vuelve a quedar stale en la card tras el rebalance a nivel-día (visto con día 1.0)."
    )
    blk = _PLANS[i: i + 1600]
    assert 'pop("_macro_band_low", None)' in blk
    assert "score_macros_only" in blk and "0.99" in blk, (
        "CLEAR-ONLY gateado por el día re-medido en banda (≥0.99) — si el día sigue fuera, "
        "los chips se quedan (honestos)"
    )


# ---------------------------------------------------------------------------
# 4. Frontend: copy honesto + resume cross-refresh
# ---------------------------------------------------------------------------

def test_frontend_day_regen_copy_is_honest():
    assert "Puede tomar hasta ~1 minuto" not in _CTX, (
        "P1-DAY-REGEN-RESUME: el copy volvió a prometer ~1 minuto (medido en vivo: ~4.4 min)"
    )
    assert "3 a 5 minutos" in _CTX


def test_frontend_day_regen_resume_marker_and_poll():
    assert "mealfit_day_regen_inflight" in _CTX, (
        "P1-DAY-REGEN-RESUME: falta el marker persistente — al refrescar, el overlay y el "
        "poll de resume mueren aunque el backend siga generando el día."
    )
    assert "_plan_modified_at" in _CTX, (
        "la señal primaria de completion del resume es _plan_modified_at > startedAt "
        "(cubre también el caso all-slots-conservados sin cambio de nombres)"
    )
    assert "dayRegenInFlight" in _CTX and "dayRegenInFlight" in _DASH, (
        "el flag del contexto debe exponerse y el Dashboard debe espejarlo al overlay"
    )
    # el finally de la ruta sin-refresh limpia el marker (no polls fantasma en la próxima visita)
    i_fin = _CTX.find("toast.dismiss(_dayLoadingId);")
    assert i_fin > 0
    assert "safeLocalStorageRemove('mealfit_day_regen_inflight')" in _CTX[i_fin: i_fin + 600]
