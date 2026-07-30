"""[P1-UPDATE-RECAP-ALL-SURFACES · 2026-07-30] (audit solver+seeder v5 · P1-1)

`P1-UPDATE-CLINICAL-RECAP` (v4) añadió el re-cap de los caps clínicos de porción
(`cap_dm2_high_gi_portions` + `cap_bariatric_portions`) DENTRO de `apply_update_macro_engine`,
pero lo gateó a `isinstance(form_data, dict) and form_data`. El fix se scope-ó a las 4
superficies user-facing de update (swap-persist ×2, chat-modify ×2) y dejó fuera las 4
superficies de generación/expand, que invocan el MISMO motor SIN `form_data` teniéndolo
en scope:

  1. `recipe_expand`          — routers/plans.py, `_expand_clin` está 11 líneas más abajo.
  2. `budget_convergence`     — graph_orchestrator, `form_data` vivo en el mismo bloque.
  3. `budget_convergence_t2`  — chunk worker (semanas 2+), ídem.
  4. `form_gen_final_closer`  — el wrapper `reconcile_all_macros_band_post_finalize` ni
                                 siquiera ACEPTABA el parámetro; el shield pre-INSERT
                                 (db_plans) ya extrae el form_data para el recompute de micros.

Consecuencia medida en el audit: en el happy path de form-gen los caps corren en assemble y
en la capa clínica determinista — TODOS antes de estos pases. Ningún pase posterior re-capea
(la cadena finalize del INSERT no contiene cap_dm2/cap_bariatric). Un DM2 cuya reconciliación
de presupuesto deja un día fuera de banda recibe la batata re-inflada de 150 g a 300-375 g
(rebalance factor hasta 2.5 + refine hasta 2×/línea) y el plan se PERSISTE así.

El segundo defecto es de observabilidad y es lo que mantuvo el gap invisible: la rama sin
form_data emitía `logger.debug` — el nivel que nadie mira. Pasa a `warning` agregado por
llamada (no por día, para no inundar).
"""
from __future__ import annotations

import logging
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel: str) -> str:
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")
_PLANS = _read(os.path.join("routers", "plans.py"))
_DBP = _read("db_plans.py")


def _line_at(src: str, needle: str) -> str:
    """Devuelve la LÍNEA (más su continuación hasta el `)` de cierre) que contiene `needle`.

    Anclaje por orden relativo — nunca ventana de bytes fija (4 caducaron en una sesión;
    memoria 2026-07-29). Se corta en el primer salto de línea que deje paréntesis
    balanceados, así que sobrevive a que la llamada se parta en varias líneas.
    """
    i = src.index(needle)
    j = src.rindex("\n", 0, i) + 1
    depth = 0
    out = []
    for k in range(j, len(src)):
        ch = src[k]
        out.append(ch)
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "\n" and depth <= 0:
            break
    return "".join(out)


# ═════════════════ 1 · los 4 callsites pasan su form_data ═════════════════

def test_recipe_expand_passes_form_data():
    """`_expand_clin` (medicalConditions hidratadas del perfil vía `_enrich_clinical_from_profile`)
    ya viaja al recompute de micros 11 líneas más abajo — el motor debe recibir el mismo dict."""
    call = _line_at(_PLANS, 'surface="recipe_expand"')
    assert "form_data=_expand_clin" in call, call


def test_budget_convergence_passes_form_data():
    call = _line_at(_GO, 'surface="budget_convergence", db=')
    assert "form_data=" in call, call


def test_budget_convergence_t2_passes_form_data():
    """El chunk worker de semanas 2+ es la superficie más silenciosa: corre en background,
    sin usuario mirando, y persiste directo."""
    call = _line_at(_GO, 'surface="budget_convergence_t2"')
    assert "form_data=" in call, call


def test_final_closer_accepts_form_data_and_shield_supplies_it():
    """El wrapper del shield pre-INSERT (`reconcile_all_macros_band_post_finalize`) es el
    ÚLTIMO pase que mueve cantidades antes del INSERT: cubre TODOS los paths (form-gen,
    partial, SSE-fallback, merge T1 del chunk worker)."""
    import inspect

    import graph_orchestrator as go
    assert "form_data" in inspect.signature(go.reconcile_all_macros_band_post_finalize).parameters
    call = _line_at(_DBP, "_ramb(_pd")
    assert "form_data=" in call, call
    # El shield YA extrae exactamente este form_data para el recompute de micros — cero
    # plumbing nuevo, misma expresión (si una cambia, la otra debe cambiar con ella).
    assert '_pd.get("form_data") or data.get("form_data")' in _DBP


# ═════════════════ 2 · propagación funcional (no solo textual) ═════════════════

def test_final_closer_propagates_form_data_to_engine(monkeypatch):
    import graph_orchestrator as go
    seen = {}

    def _fake(plan_data, *, surface, db=None, pantry_strict=False, form_data=None):
        seen["surface"] = surface
        seen["form_data"] = form_data
        return 1

    monkeypatch.setattr(go, "apply_update_macro_engine", _fake)
    fd = {"medicalConditions": ["Diabetes Tipo 2"]}
    assert go.reconcile_all_macros_band_post_finalize({"days": []}, form_data=fd) == 1
    assert seen["surface"] == "form_gen_final_closer"
    assert seen["form_data"] is fd


def test_final_closer_without_form_data_still_runs(monkeypatch):
    """Compat: el wrapper se sigue pudiendo llamar sin form_data (callers legacy/tests).
    El motor recibe None y su fail-safe interno decide — no explota."""
    import graph_orchestrator as go
    monkeypatch.setattr(go, "apply_update_macro_engine",
                        lambda plan_data, **kw: 0 if kw.get("form_data") is None else 99)
    assert go.reconcile_all_macros_band_post_finalize({"days": []}) == 0


# ═════════════════ 3 · la omisión deja de ser silenciosa ═════════════════

def _plan_out_of_band() -> dict:
    return {
        "macros": {"protein": "100 g", "carbs": "200 g", "fats": "60 g"},
        "days": [{"meals": [{"protein": 20, "carbs": 40, "fats": 10, "cals": 330}]}],
    }


def _stub_engine(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "_rebalance_day_macros_to_target", lambda *a, **kw: 1)
    monkeypatch.setattr(go, "_trim_day_fats_to_target", lambda *a, **kw: 0)
    monkeypatch.setattr(go, "GLOBAL_DAY_REFINE_ENABLED", False)
    monkeypatch.setattr(go, "_sync_recipe_step_quantities", lambda *a, **kw: None)
    return go


def test_missing_form_data_emits_warning_not_debug(monkeypatch, caplog):
    """`logger.debug` es el nivel que nadie mira: el gap vivió un día entero invisible en 4
    superficies. Con WARNING, un callsite nuevo que olvide su form_data se ve en el log de prod.

    Ancla el NIVEL a propósito — un test que captura en el nivel equivocado pasa por la línea
    equivocada (30 capturas verdes por eso, memoria 2026-07-29)."""
    go = _stub_engine(monkeypatch)
    caplog.set_level(logging.WARNING, logger=go.logger.name)
    go.apply_update_macro_engine(_plan_out_of_band(), surface="surface_sin_form_data", db=object())
    hits = [r for r in caplog.records
            if r.levelno == logging.WARNING and "P1-UPDATE-CLINICAL-RECAP" in r.getMessage()]
    assert len(hits) == 1, [r.getMessage() for r in caplog.records]
    assert "surface_sin_form_data" in hits[0].getMessage()


def test_warning_is_aggregated_not_per_day(monkeypatch, caplog):
    """Un plan de 30 días fuera de banda no debe producir 30 warnings idénticos."""
    go = _stub_engine(monkeypatch)
    caplog.set_level(logging.WARNING, logger=go.logger.name)
    plan = _plan_out_of_band()
    plan["days"] = plan["days"] * 30
    go.apply_update_macro_engine(plan, surface="s", db=object())
    hits = [r for r in caplog.records if "P1-UPDATE-CLINICAL-RECAP" in r.getMessage()]
    assert len(hits) == 1
    assert "30" in hits[0].getMessage(), "el warning debe declarar cuántos días quedaron sin re-cap"


def test_with_form_data_no_warning(monkeypatch, caplog):
    go = _stub_engine(monkeypatch)
    monkeypatch.setattr(go, "cap_dm2_high_gi_portions", lambda *a, **kw: 0)
    monkeypatch.setattr(go, "cap_bariatric_portions", lambda *a, **kw: 0)
    caplog.set_level(logging.WARNING, logger=go.logger.name)
    go.apply_update_macro_engine(_plan_out_of_band(), surface="s", db=object(),
                                 form_data={"medicalConditions": ["Diabetes Tipo 2"]})
    assert not [r for r in caplog.records if "P1-UPDATE-CLINICAL-RECAP" in r.getMessage()]
