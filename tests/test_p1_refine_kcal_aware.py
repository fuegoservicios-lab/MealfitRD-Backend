"""[P1-REFINE-KCAL-AWARE · 2026-07-06] El trigger del refinador global de porciones (S1 assemble +
update helper `apply_update_macro_engine` + regen-day inline) miraba SOLO P/C/F contra [0.90,1.12].
La banda de kcal del scoreboard (±5% [0.95,1.05]) es MÁS ESTRECHA → un día con macros dentro de la
banda laxa pero kcal fuera del techo estrecho (el miss de banda MÁS COMÚN: forense 20 planes → celda
kcal fuera en 53% de los días) se SALTABA el refinador, que sí pondera kcal y puede cerrarlo.

Validado en la ruta de producción (refiner+truth-up) sobre 3 días reales: fcc7a9f0 d3 pasó 3/4→4/4
(kcal 105%→104%), cero regresión. Este test ancla el arm kcal (helper unit + knob + wiring en los 3 sitios).
"""
from pathlib import Path
import graph_orchestrator as go

_GO = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
_PL = (Path(go.__file__).resolve().parent / "routers" / "plans.py").read_text(encoding="utf-8")


# --- unit: _day_kcal_out_of_refine_band -------------------------------------

def test_all_at_target_is_in_band():
    # sp=pg, sc=cg, sf=fg → kcal == target → dentro de [0.95,1.05] → False (no refine).
    assert go._day_kcal_out_of_refine_band(120, 200, 60, 120, 200, 60) is False


def test_macros_in_loose_band_but_kcal_over_ceiling():
    # cada macro al 108% (dentro de [0.90,1.12]) → kcal 108% (fuera de [0.95,1.05]) → True.
    pg, cg, fg = 100.0, 100.0, 100.0
    sp = sc = sf = 108.0
    # sanity: cada macro en la banda laxa de macros
    for v, t in ((sp, pg), (sc, cg), (sf, fg)):
        assert 0.90 <= v / t <= 1.12
    assert go._day_kcal_out_of_refine_band(sp, sc, sf, pg, cg, fg) is True


def test_kcal_within_5pct_is_in_band():
    # cada macro al 103% → kcal 103% → dentro de [0.95,1.05] → False.
    assert go._day_kcal_out_of_refine_band(103, 103, 103, 100, 100, 100) is False


def test_kcal_under_floor_triggers():
    # 92% → fuera del piso 0.95 → True (también cubre el undershoot).
    assert go._day_kcal_out_of_refine_band(92, 92, 92, 100, 100, 100) is True


def test_zero_target_fail_safe():
    assert go._day_kcal_out_of_refine_band(0, 0, 0, 0, 0, 0) is False
    assert go._day_kcal_out_of_refine_band(50, 50, 50, 0, 0, 0) is False


def test_fats_skew_drives_kcal_out_even_with_pc_in():
    # P y C al 100%, F al 110% (dentro de banda macro) → kcal = (400+400+990)/(400+400+900)=1789/1700=1.052
    # → fuera de 1.05 por poco → True (el caso fats-empuja-kcal del forense, fats 39% out).
    assert go._day_kcal_out_of_refine_band(100, 100, 110, 100, 100, 100) is True


# --- knob -------------------------------------------------------------------

def test_knob_exists_default_on():
    assert hasattr(go, "REFINE_KCAL_AWARE")
    assert go.REFINE_KCAL_AWARE is True  # default ON


# --- wiring en los 3 sitios (paridad SSOT) ----------------------------------

def test_wired_in_s1_assemble():
    # el trigger inline de S1 (`_out_of_band = any(...)` con las sumas _sp_r/_sc_r/_sf_r) aplica el arm kcal.
    i = _GO.index("_out_of_band = any(")   # 1ra ocurrencia = inline de S1 (la del update es `def _out_of_band`)
    blk = _GO[i:i + 900]
    assert "_sp_r" in blk, "ancla equivocada (no es el inline de S1)"
    assert "REFINE_KCAL_AWARE" in blk
    assert "_day_kcal_out_of_refine_band(" in blk


def test_wired_in_update_helper():
    i = _GO.index("def apply_update_macro_engine(")
    blk = _GO[i:i + 9000]
    assert "REFINE_KCAL_AWARE" in blk
    assert "_day_kcal_out_of_refine_band(" in blk


def test_wired_in_regen_day_inline():
    i = _PL.index("refine_day_portions_integer as _rdi_rd")
    blk = _PL[i - 600:i + 1200]
    assert "_rka_rf" in blk and "_kob_rf(" in blk, "regen-day inline no aplica el arm kcal"


def test_solver_clamp_direction_telemetry_present():
    # [P1-SOLVER-CLAMP-DIRECTION · 2026-07-06] la telemetría del solver desglosa saturación ARRIBA (≥3.5×,
    # quería más → sub-entrega) vs ABAJO (≤0.3×, quería menos) — desambigua la opción (b) con datos de flota.
    assert "P1-SOLVER-CLAMP-DIRECTION" in _GO
    assert "_solver_clamp_saturated_hi" in _GO and "_solver_clamp_saturated_lo" in _GO
    assert "saturated_meals_hi" in _GO and "saturated_meals_lo" in _GO


def test_marker_anchored():
    assert "P1-REFINE-KCAL-AWARE" in _GO
    assert "P1-REFINE-KCAL-AWARE" in _PL
