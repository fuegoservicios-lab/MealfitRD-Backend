"""[P1-SWAP-RENAL-DAY-VIEW · 2026-07-30] (audit solver+seeder v5 · P1-3)

`/swap-meal/persist` es la ÚNICA superficie que no le pasa el plan entero a
`apply_update_macro_engine`: arma un "day view" de 1 día para re-cuadrar solo el día tocado.
Ese view llevaba 4 claves — `days`, `macros`, `calories`, `main_goal` — y NO `renal_protein_cap`.

El motor lee `plan_data.get("renal_protein_cap")` para decidir su guard renal:

    _renal = bool((plan_data.get("renal_protein_cap") or {}).get("applied"))
    _tp = 0.0 if _renal else _pg          # rebalance protein-preserving
    ...
    if not _renal and GLOBAL_DAY_REFINE_ENABLED:   # el refine mueve líneas proteína-dominantes

Con la clave ausente, `_renal` es SIEMPRE False en esa superficie: el rebalance re-apunta la
proteína al target COMPLETO del día y el refine escala líneas proteína-dominantes hasta 2×/línea
— en un plan con cap KDIGO aplicado, que el repo trata como fail-hard de seguridad iatrogénica.
Nada aguas abajo lo repara: `_ume_sw` (que sí ve el plan completo) tiene rama renal NO-TOUCH por
diseño, y `apply_update_condition_ceilings` mide sodio/K/Mg/fibra, no proteína.

El fix es una clave — pero el test que importa es el de PARIDAD: cualquier lectura futura de
`plan_data` dentro del motor debe estar replicada en el view, o el próximo guard nacerá inerte
en esta superficie exactamente igual que este. Es la lección de P1-UPDATE-PROTAGONIST-FLOOR:
un gate que declina por falta de contexto solo es correcto si ALGUIEN puede suministrarlo.
"""
from __future__ import annotations

import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel: str) -> str:
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")
_PLANS = _read(os.path.join("routers", "plans.py"))


# ═════════════════ 1 · el helper SSOT del view ═════════════════

def test_engine_day_view_carries_renal_cap():
    import graph_orchestrator as go
    plan = {
        "days": [{"meals": []}, {"meals": []}],
        "macros": {"protein": "60 g"},
        "calories": 1800,
        "main_goal": "lose_fat",
        "renal_protein_cap": {"applied": True, "cap_g": 55},
    }
    view = go._engine_day_view(plan, plan["days"][1])
    assert view["days"] == [plan["days"][1]]
    assert view["days"][0] is plan["days"][1], "el view comparte el dict: las mutaciones persisten"
    assert view["renal_protein_cap"] == {"applied": True, "cap_g": 55}
    assert view["macros"] is plan["macros"]
    assert view["calories"] == 1800
    assert view["main_goal"] == "lose_fat"


def test_engine_day_view_tolerates_missing_keys():
    import graph_orchestrator as go
    view = go._engine_day_view({}, {"meals": []})
    assert view["days"] == [{"meals": []}]
    assert view.get("renal_protein_cap") is None


def test_renal_guard_actually_engages_through_the_view(monkeypatch):
    """Funcional end-to-end del guard: con el view construido por el helper, el motor NO invoca
    el refine (que mueve proteína) y le pasa target_protein=0 al rebalance."""
    import graph_orchestrator as go
    seen = {"refine": 0, "target_protein": None}

    def _reb(_meals, _cg, _fg, db, target_protein=None, **kw):
        seen["target_protein"] = target_protein
        return 1

    monkeypatch.setattr(go, "_rebalance_day_macros_to_target", _reb)
    monkeypatch.setattr(go, "_trim_day_fats_to_target", lambda *a, **kw: 0)
    monkeypatch.setattr(go, "_sync_recipe_step_quantities", lambda *a, **kw: None)
    monkeypatch.setattr(go, "cap_dm2_high_gi_portions", lambda *a, **kw: 0)
    monkeypatch.setattr(go, "cap_bariatric_portions", lambda *a, **kw: 0)

    import portion_solver as ps
    monkeypatch.setattr(ps, "refine_day_portions_integer",
                        lambda *a, **kw: seen.__setitem__("refine", seen["refine"] + 1) or 0)

    day = {"meals": [{"protein": 20, "carbs": 40, "fats": 10, "cals": 330}]}
    plan = {"days": [day], "macros": {"protein": "100 g", "carbs": "200 g", "fats": "60 g"},
            "renal_protein_cap": {"applied": True}}
    go.apply_update_macro_engine(go._engine_day_view(plan, day), surface="swap_persist_day",
                                 db=object(), form_data={})
    assert seen["target_protein"] == 0.0, "renal ⇒ rebalance protein-preserving"
    assert seen["refine"] == 0, "renal ⇒ el refine (que mueve proteína) NO corre"


def test_without_renal_cap_the_engine_is_unchanged(monkeypatch):
    """Regresión inversa: un plan SIN cap renal debe comportarse exactamente como antes."""
    import graph_orchestrator as go
    seen = {"target_protein": None}
    monkeypatch.setattr(go, "_rebalance_day_macros_to_target",
                        lambda _m, _c, _f, db, target_protein=None, **kw:
                        seen.__setitem__("target_protein", target_protein) or 1)
    monkeypatch.setattr(go, "_trim_day_fats_to_target", lambda *a, **kw: 0)
    monkeypatch.setattr(go, "GLOBAL_DAY_REFINE_ENABLED", False)
    monkeypatch.setattr(go, "_sync_recipe_step_quantities", lambda *a, **kw: None)
    day = {"meals": [{"protein": 20, "carbs": 40, "fats": 10, "cals": 330}]}
    plan = {"days": [day], "macros": {"protein": "100 g", "carbs": "200 g", "fats": "60 g"}}
    go.apply_update_macro_engine(go._engine_day_view(plan, day), surface="s", db=object())
    assert seen["target_protein"] == 100.0


# ═════════════════ 2 · el callsite usa el helper ═════════════════

def test_swap_persist_day_uses_the_helper():
    """El view inline es el que se quedó atrás. Si vuelve un dict literal, este test lo caza."""
    i = _PLANS.index("P1-SWAP-PERSIST-DAY-BAND")
    blk = _PLANS[i:_PLANS.index("[P1-SWAP-PERSIST-DAY-BAND] no-op", i)]
    assert "_engine_day_view" in blk, "el day-view debe venir del helper SSOT, no de un literal"
    assert '"days": [day], "macros"' not in blk, "quedó el literal inline (el que olvidó el cap renal)"


# ═════════════════ 3 · PARIDAD: el view cubre TODO lo que el motor lee ═════════════════

def _engine_body() -> str:
    i = _GO.index("def apply_update_macro_engine(")
    return _GO[i:_GO.index("\ndef ", i + 10)]


def _view_body() -> str:
    i = _GO.index("def _engine_day_view(")
    return _GO[i:_GO.index("\ndef ", i + 10)]


def test_view_replicates_every_plan_data_key_the_engine_reads():
    """LA prueba de la clase, no del síntoma.

    `renal_protein_cap` se perdió porque el view era un literal escrito a mano y el motor añadió
    una lectura después. Este test deriva las claves del CÓDIGO del motor (no de una lista
    hardcodeada — memoria 07-29: anclar un conteo/lista caduca solo) y exige que el view las
    replique todas. El próximo guard que lea `plan_data.get("X")` rompe este test hasta que X
    viaje en el view."""
    keys = set(re.findall(r'plan_data\.get\(\s*"([a-z_]+)"', _engine_body()))
    assert "renal_protein_cap" in keys, "sanity: el motor debe seguir leyendo el cap renal"
    assert "macros" in keys and "days" in keys, "sanity del regex"
    view = _view_body()
    faltan = sorted(k for k in keys if f'"{k}"' not in view)
    assert not faltan, (
        f"el motor lee estas claves de plan_data y el day-view de swap-persist NO las replica: "
        f"{faltan}. Un guard alimentado por una clave ausente nace INERTE en esa superficie "
        f"(exactamente lo que pasó con renal_protein_cap)."
    )
