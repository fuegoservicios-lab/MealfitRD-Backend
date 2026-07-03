"""[P1-UPDATE-MACRO-PARITY · 2026-07-03] (audit v6 · P1-1) Paridad del MOTOR de macros en updates.

El motor de S1 (`_rebalance_day_macros_to_target` + `refine_day_portions_integer`) corría SOLO en
form-gen: un swap/chat-modify que dejaba el día fuera de banda se entregaba con banner (band-parity)
pero sin palanca correctiva de P/C/F a nivel día. Cierra:
  1. Helper SSOT `apply_update_macro_engine` (graph_orchestrator): rebalance → refine 5g → qty-sync,
     por cada día fuera de banda [0.90, 1.12]. Skip pantry-strict; renal protein-preserving sin refine.
  2. Cableado en swap-persist (plans.py), chat-modify ×2 pasadas (tools.py) y refine en regen-day
     (plans.py, con guard pantry never-worse-than-current).
  3. Bugfix: el recheck post-quantize de assemble pasaba `[_rq_d]` (dict del DÍA) donde la función
     espera la lista de MEALS → `movable=0` siempre = no-op silencioso. Ahora pasa `_rq_meals`.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_PL = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_TO = (_BACKEND / "tools.py").read_text(encoding="utf-8")


def test_marker_bumped():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-UPDATE-MACRO-PARITY" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-03"


# ════════════════════════════════════════════════════════════════════════════
# Knob: default ON + registrado
# ════════════════════════════════════════════════════════════════════════════
def test_knob_default_on_in_source():
    assert re.search(
        r'UPDATE_MACRO_ENGINE_ENABLED\s*=\s*_env_bool\("MEALFIT_UPDATE_MACRO_ENGINE",\s*True\)', _GO
    ), "el knob MEALFIT_UPDATE_MACRO_ENGINE debe nacer ON (default True)"


def test_knob_registered():
    import graph_orchestrator as go
    snap = go.get_knobs_registry_snapshot()
    assert "MEALFIT_UPDATE_MACRO_ENGINE" in snap


# ════════════════════════════════════════════════════════════════════════════
# Funcional — fake DB determinista (mismo patrón de test_p1_next_level_batch)
# ════════════════════════════════════════════════════════════════════════════
_DENS = {"pollo": {"kcal": 1.65, "protein": 0.31, "carbs": 0.0, "fats": 0.036},
         "arroz": {"kcal": 1.30, "protein": 0.027, "carbs": 0.28, "fats": 0.003},
         "aguacate": {"kcal": 1.60, "protein": 0.02, "carbs": 0.085, "fats": 0.147}}


class _RefDB:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*g de (\w+)", str(s).lower())
        if not m:
            return None
        g, food = float(m.group(1)), m.group(2)
        d = _DENS.get(food)
        return {k: v * g for k, v in d.items()} if d else None


def _mk_plan(renal: bool = False):
    # Día entregado: P≈75 / C≈69 / F≈16 vs target P90/C70/F22 → proteína .83 y grasa .73 FUERA de banda.
    plan = {
        "macros": {"protein": 90, "carbs": 70, "fats": 22},
        "calories": 850,
        "days": [{"day": 1, "meals": [
            {"meal": "Almuerzo", "name": "Pollo con arroz", "protein": 40, "carbs": 27, "fats": 12,
             "ingredients": ["120g de pollo", "80g de arroz", "50g de aguacate"],
             "ingredients_raw": ["120g de pollo", "80g de arroz", "50g de aguacate"]},
            {"meal": "Cena", "name": "Pollo con arroz II", "protein": 35, "carbs": 42, "fats": 4,
             "ingredients": ["100g de pollo", "150g de arroz"],
             "ingredients_raw": ["100g de pollo", "150g de arroz"]},
        ]}],
    }
    if renal:
        plan["renal_protein_cap"] = {"applied": True, "protein_g": 60}
    return plan


def _delivered(plan):
    db = _RefDB()
    out = {"protein": 0.0, "carbs": 0.0, "fats": 0.0}
    for d in plan["days"]:
        for m in d["meals"]:
            for s in m["ingredients"]:
                mc = db.macros_from_ingredient_string(s)
                if mc:
                    for k in out:
                        out[k] += mc.get(k) or 0.0
    return out


def test_engine_pulls_out_of_band_day_into_band(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "UPDATE_MACRO_ENGINE_ENABLED", True)
    plan = _mk_plan()
    n = go.apply_update_macro_engine(plan, surface="test", db=_RefDB())
    assert n == 1, "el día fuera de banda debe ser tocado"
    got = _delivered(plan)
    for key, tgt in (("protein", 90.0), ("carbs", 70.0), ("fats", 22.0)):
        ratio = got[key] / tgt
        assert 0.88 <= ratio <= 1.14, f"{key} fuera de banda tras el motor: {ratio:.3f}"


def test_engine_noop_when_in_band(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "UPDATE_MACRO_ENGINE_ENABLED", True)
    plan = _mk_plan()
    # targets = entrega actual → en banda → 0 días tocados, plan intacto
    plan["macros"] = {"protein": 75, "carbs": 69, "fats": 16}
    before = copy.deepcopy(plan)
    assert go.apply_update_macro_engine(plan, surface="test", db=_RefDB()) == 0
    assert plan == before


def test_engine_skips_pantry_strict(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "UPDATE_MACRO_ENGINE_ENABLED", True)
    plan = _mk_plan()
    before = copy.deepcopy(plan)
    assert go.apply_update_macro_engine(plan, surface="test", db=_RefDB(), pantry_strict=True) == 0
    assert plan == before, "pantry-strict = skip total (escalar = 'comprar más')"


def test_engine_knob_off_noop(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "UPDATE_MACRO_ENGINE_ENABLED", False)
    plan = _mk_plan()
    before = copy.deepcopy(plan)
    assert go.apply_update_macro_engine(plan, surface="test", db=_RefDB()) == 0
    assert plan == before


def test_engine_renal_protein_preserving(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "UPDATE_MACRO_ENGINE_ENABLED", True)
    plan = _mk_plan(renal=True)
    pollo_before = [s for d in plan["days"] for m in d["meals"]
                    for s in m["ingredients"] if "pollo" in s]
    go.apply_update_macro_engine(plan, surface="test", db=_RefDB())
    pollo_after = [s for d in plan["days"] for m in d["meals"]
                   for s in m["ingredients"] if "pollo" in s]
    assert pollo_before == pollo_after, \
        "renal: las líneas proteína-dominantes JAMÁS se escalan (cap KDIGO manda)"


def test_engine_failsafe_on_garbage():
    import graph_orchestrator as go
    assert go.apply_update_macro_engine(None, surface="test") == 0
    assert go.apply_update_macro_engine({"days": "corrupted"}, surface="test", db=_RefDB()) == 0


# ════════════════════════════════════════════════════════════════════════════
# Cableado en las 3 superficies (parser-based; anclado a tooltip-anchors)
# ════════════════════════════════════════════════════════════════════════════
def test_wired_in_swap_persist_before_micros_recompute():
    assert "apply_update_macro_engine as _ume_sw" in _PL
    idx_engine = _PL.index("apply_update_macro_engine as _ume_sw")
    idx_micros = _PL.index("recompute_micronutrient_report_for_plan(plan_data, _micro_form, db=None)")
    assert idx_engine < idx_micros, \
        "el motor debe correr ANTES del recompute de micros (el panel ve el estado final)"
    assert 'surface="swap_persist", pantry_strict=_ps_swap' in _PL


def test_wired_in_chat_modify_both_passes():
    # pasada pre-listas (las listas precomputadas reflejan el estado final)
    assert "apply_update_macro_engine as _ume_pre" in _TO
    # pasada fresh dentro del lock (converge con las listas)
    assert "apply_update_macro_engine as _ume_cm" in _TO
    idx_engine = _TO.index("apply_update_macro_engine as _ume_cm")
    idx_micros = _TO.index("recompute_micronutrient_report_for_plan(plan_data_fresh, _micro_form_cm, db=None)")
    assert idx_engine < idx_micros, "en el fresh, el motor corre ANTES del recompute de micros"


def test_wired_in_regen_day_refine_with_pantry_guard():
    assert "refine_day_portions_integer as _rdi_rd" in _PL
    blk_start = _PL.index("refine_day_portions_integer as _rdi_rd")
    blk = _PL[blk_start:blk_start + 3000]
    assert "_day_exceeds_pantry" in blk, \
        "el refine de regen-day debe revalidar contra la Nevera ORIGINAL (never-worse-than-current)"
    assert "new_meals[:] = _pre_rf" in blk, "y revertir si rompe pantry"
    assert "not _renal_capped" in _PL[max(0, blk_start - 2000):blk_start], \
        "renal: skip del refine (el cap KDIGO manda)"


def test_band_parity_still_runs_after_engine_in_swap():
    # el motor NO reemplaza el contrato de banda: band-parity sigue corriendo después
    # (y ahora puede LIMPIAR el banner si el motor reparó la banda).
    idx_engine = _PL.index("apply_update_macro_engine as _ume_sw")
    idx_parity = _PL.index('_ubp_sw(plan_data, surface="swap_persist"')
    assert idx_engine < idx_parity


# ════════════════════════════════════════════════════════════════════════════
# Bugfix recheck post-quantize (assemble): [_rq_d] → _rq_meals
# ════════════════════════════════════════════════════════════════════════════
def test_postquantize_recheck_passes_meals_not_day():
    assert not re.search(r"_rebalance_day_macros_to_target\(\s*\[_rq_d\]", _GO), \
        "el recheck post-quantize pasaba [_rq_d] (day dict) — movable=0 siempre = no-op silencioso"
    assert "_rq_meals, float(_cg or 0), float(_fg or 0), _rq_db," in _GO
