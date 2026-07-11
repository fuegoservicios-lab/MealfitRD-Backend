"""[P1-PLAN-FREEZE · 2026-07-11] Congelación del plan por Nevera vacía (decisión de
producto del owner, versión con gracia):

recordar (24h vacía) → congelar (48h: chunks pausados + `_frozen_at`, el contador de
días queda congelado DE VERDAD) → reanudar al reponer (corrimiento de fechas por los
días congelados + chunks resume + push; hook inmediato en /restock + sweep backstop)
→ archivar a los 30 días congelado. **La CUENTA jamás se toca** (cero borrado/
desactivación automática — línea roja del diseño).

tooltip-anchor: P1-PLAN-FREEZE
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_DASH = (_BACKEND.parent / "frontend" / "src" / "pages" / "Dashboard.jsx")


# ---------------------------------------------------------------------------
# 1. La escalera completa vive en el sweep
# ---------------------------------------------------------------------------

def test_sweep_ladder_states():
    i = _CRON.find("def _plan_freeze_sweep")
    assert i > 0, "el sweep de congelación desapareció"
    blk = _CRON[i: i + 9000]
    assert 'MEALFIT_PLAN_FREEZE_ENABLED' in blk, "kill switch"
    assert 'MEALFIT_PLAN_FREEZE_GRACE_HOURS", 48' in blk, "gracia 48h (nevera vacía día 0-1 es NORMAL)"
    assert 'MEALFIT_PLAN_FREEZE_REMINDER_HOURS", 24' in blk, "recordatorio a las 24h"
    assert 'MEALFIT_PLAN_FREEZE_ARCHIVE_DAYS", 30' in blk, "archivo a los 30 días congelado"
    assert "_count_meaningful_pantry_items" in blk, "mismo SSOT del mínimo significativo"
    assert "'{_frozen_at}'" in blk and "'{_freeze_reminder_at}'" in blk and "'{_frozen_archived_at}'" in blk


def test_never_touches_accounts():
    # Línea roja: el sweep JAMÁS borra/desactiva cuentas ni filas de user_profiles.
    i = _CRON.find("def _plan_freeze_sweep")
    blk = _CRON[i: i + 12000]
    assert "DELETE FROM user_profiles" not in blk
    assert "UPDATE user_profiles" not in blk
    assert "la CUENTA" in blk or "cuenta" in blk.lower(), "la doctrina queda documentada en el código"


def test_all_meal_plans_mutations_filter_user_id():
    # I2: toda mutación de meal_plans del feature lleva AND user_id (freeze, reminder,
    # archive, unfreeze, shift).
    i = _CRON.find("[P1-PLAN-FREEZE · 2026-07-11] Congelación")
    blk = _CRON[i:]
    import re
    _updates = re.findall(r"UPDATE meal_plans[\s\S]{0,400}?WHERE[^\"]+", blk)
    assert _updates, "esperaba UPDATEs de meal_plans en el feature"
    for u in _updates:
        assert "user_id = %s" in u, f"mutación sin filtro user_id (I2): {u[:120]}"


# ---------------------------------------------------------------------------
# 2. Congelar = pausar chunks; reanudar = shift de fechas + chunks resume
# ---------------------------------------------------------------------------

def test_freeze_pauses_chunks_and_resume_shifts_dates():
    assert "status = 'pending_user_action', updated_at = NOW() " in _CRON.replace("\n", " ") or \
        "SET status = 'pending_user_action'" in _CRON, "congelar pausa los chunks pending"
    i_sh = _CRON.find("def _shift_plan_dates_for_freeze")
    assert i_sh > 0
    blk = _CRON[i_sh: i_sh + 2600]
    for key in ("_plan_start_date", "plan_start_date", "grocery_start_date", "cycle_start_date"):
        assert key in blk, f"el shift debe cubrir la fecha-ancla {key}"
    assert "execute_after = execute_after + make_interval(days => %s)" in blk, (
        "los chunks futuros también se corren (el calendario completo se congela)"
    )
    i_rs = _CRON.find("def _resume_frozen_plan")
    blk_rs = _CRON[i_rs: i_rs + 3200]
    assert "_shift_plan_dates_for_freeze" in blk_rs
    assert "status = 'pending'" in blk_rs, "reanudar despierta los chunks pausados"
    assert "- '_frozen_at' - '_freeze_reminder_at'" in blk_rs, "flags limpiados al reanudar"


# ---------------------------------------------------------------------------
# 3. Hook inmediato en /restock + cron registrado
# ---------------------------------------------------------------------------

def test_restock_hook_and_cron_registration():
    i = _PLANS.find("from cron_tasks import try_unfreeze_plan_for_user")
    assert i > 0, "el hook de descongelamiento inmediato desapareció de /restock"
    assert '"plan_unfrozen": _unfrozen_now' in _PLANS, "el frontend sabe si se descongeló"
    assert 'id="plan_freeze_sweep"' in _CRON, "cron no registrado en el scheduler"
    assert 'MEALFIT_PLAN_FREEZE_SWEEP_INTERVAL_MIN", 60' in _CRON


# ---------------------------------------------------------------------------
# 4. Banner del Dashboard
# ---------------------------------------------------------------------------

def test_dashboard_frozen_banner():
    src = _DASH.read_text(encoding="utf-8")
    assert "planData?._frozen_at && (" in src, "el banner de plan congelado desapareció"
    assert "Tus días NO están corriendo" in src, "copy honesto del contador congelado"
    assert "Reponer mi Nevera" in src, "CTA accionable a /pantry"


def test_marker_anchored_in_source():
    assert _CRON.count("P1-PLAN-FREEZE") >= 4
