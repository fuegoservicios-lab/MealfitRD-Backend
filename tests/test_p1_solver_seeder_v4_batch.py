"""[P1-SOLVER-SEEDER-V4 · 2026-07-29] Los 5 P1 del audit solver+seeder v4.

- P1-SOLVER-KCAL-ROW-REDUNDANT: la fila kcal del LSQ es `4P+4C+9F` medido (p50 de divergencia 0.1%
  en el catálogo) pero, con filas en unidades ABSOLUTAS, su peso EFECTIVO es `w·b²` → con w=1.2 se
  llevaba el 98.2% del objetivo. Default 1.2 → 0.1 (+10.1 pp de convergencia en 416 comidas vivas).
- P1-SOLVER-LSQ-ITERS: el criterio de parada de `_box_lsq` no dispara (99% de las comidas agotan el
  tope de 150 barridos; p50 necesario = 82.560) → tope promovido a knob, default 400 (+7.3 pp).
- P1-REFINE-RAW-BY-FOOD: el refinador escribía `ingredients_raw[idx]` con guard `idx < len(raw)` —
  más flojo que sus pases hermanos → escalaba el alimento EQUIVOCADO en la lista de compras.
- P1-CLOSER-CONDITION-SSOT: los 3 detectores de condición del micro-closer eran listas de stems
  PARALELAS al SSOT de constants.py; el renal cubría 2 de 14 términos y fallaba ABIERTO en el resto.
- P1-UPDATE-CLINICAL-RECAP: en las superficies de update el motor de macros corre DESPUÉS de la capa
  clínica y nadie re-aplicaba los caps de porción → el rebalance re-inflaba la batata capada del DM2.

Los tests parser-based anclan los tooltip-anchors: un renombre falla aquí antes de tocar producción.
"""
from __future__ import annotations

import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel: str) -> str:
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_PS = _read("portion_solver.py")
_GO = _read("graph_orchestrator.py")
_CONST = _read("constants.py")
_PLANS = _read(os.path.join("routers", "plans.py"))
_TOOLS = _read("tools.py")


# ═════════════════ P1-1 · P1-SOLVER-KCAL-ROW-REDUNDANT ═════════════════

def test_p1_1_kcal_weight_default_lowered():
    assert 'SOLVER_W_KCAL = _envf("MEALFIT_SOLVER_W_KCAL", 0.1' in _PS
    import portion_solver as ps
    assert ps.SOLVER_W_KCAL == 0.1
    # los otros tres NO se tocan (el cambio es sobre la fila redundante, no sobre la preferencia)
    assert (ps.SOLVER_W_PROTEIN, ps.SOLVER_W_CARBS, ps.SOLVER_W_FATS) == (1.5, 1.1, 1.4)


def test_p1_1_anchor_and_normalize_guardrail_present():
    """El anchor vive en el código Y el aviso anti-normalización también: la 'cura' natural del
    diagnóstico (normalizar filas) fue MEDIDA y regresa −27.1 pp. Sin este párrafo, el próximo audit
    vuelve a proponerla."""
    assert "P1-SOLVER-KCAL-ROW-REDUNDANT" in _PS
    assert "NO \"arreglar\" esto normalizando las filas" in _PS
    assert "−27.1 pp" in _PS or "-27.1 pp" in _PS


@pytest.mark.parametrize("tgt", [
    {"kcal": 700, "protein": 45, "carbs": 70, "fats": 25},    # almuerzo
    {"kcal": 520, "protein": 32, "carbs": 62, "fats": 14},    # desayuno
    {"kcal": 300, "protein": 24, "carbs": 30, "fats": 9},     # merienda
])
def test_p1_1_kcal_row_dominance_reduced(tgt):
    """El peso EFECTIVO de cada ecuación del LSQ es `w·b²` (filas absolutas). Con el default previo
    (1.2) la fila kcal se llevaba **98.2-98.4%** del objetivo; con 0.1 baja a **81.7-84.0%**.

    ⚠️ HONESTIDAD DEL UMBRAL: bajar el peso NO devuelve los pesos declarados a su significado — la
    fila kcal SIGUE dominando, solo que menos. Es lo que compra los +10.1 pp de convergencia medidos
    en 416 comidas vivas, ni un ápice más. El umbral 0.90 ancla esa mejora concreta (revertir a 1.2
    lo rompe) sin pretender una propiedad que el cambio no tiene. Cerrar el resto de la brecha exige
    re-tunear los CUATRO pesos contra el harness de comidas vivas — trabajo aparte, con A/B."""
    import portion_solver as ps
    w = {"kcal": ps.SOLVER_W_KCAL, "protein": ps.SOLVER_W_PROTEIN,
         "carbs": ps.SOLVER_W_CARBS, "fats": ps.SOLVER_W_FATS}
    eff = {m: w[m] * (float(tgt[m]) ** 2) for m in w}
    share_kcal = eff["kcal"] / sum(eff.values())
    assert share_kcal <= 0.90, (
        f"la fila kcal se lleva {share_kcal:.1%} del objetivo — volvió al régimen dominante previo "
        f"(pesos efectivos {({k: round(v) for k, v in eff.items()})})")


def test_p1_1_protein_outweighs_carbs_and_fats_effectively():
    """Corolario clínico del cambio: con la fila kcal contenida, el peso efectivo de la PROTEÍNA
    manda sobre carbos y grasa (es lo que `SOLVER_W_PROTEIN=1.5` siempre quiso decir)."""
    import portion_solver as ps
    tgt = {"kcal": 718, "protein": 52, "carbs": 72, "fats": 24}
    w = {"protein": ps.SOLVER_W_PROTEIN, "carbs": ps.SOLVER_W_CARBS, "fats": ps.SOLVER_W_FATS}
    eff = {m: w[m] * (float(tgt[m]) ** 2) for m in w}
    assert eff["protein"] > eff["fats"]
    # la inversión medida en el audit: carbos pesaba 1.41× proteína. El ratio sigue existiendo por
    # b_carbs > b_protein, pero se ancla para que no crezca en silencio.
    assert eff["carbs"] / eff["protein"] < 1.6, f"carbos sobre-pesa a proteína: {eff}"


# ═════════════════ P1-2 · P1-SOLVER-LSQ-ITERS ═════════════════

def test_p1_2_iters_knob_defined_with_validator():
    assert 'SOLVER_LSQ_ITERS = _envi("MEALFIT_SOLVER_LSQ_ITERS", 400' in _PS
    assert "50 <= v <= 20000" in _PS
    assert "P1-SOLVER-LSQ-ITERS" in _PS
    import portion_solver as ps
    assert ps.SOLVER_LSQ_ITERS == 400


def test_p1_2_knob_registered_in_registry():
    """Convención del repo: todo MEALFIT_* se auto-registra → visible en /health/version."""
    import portion_solver  # noqa: F401  (fuerza la lectura del knob)
    from knobs import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()
    assert "MEALFIT_SOLVER_LSQ_ITERS" in snap
    assert snap["MEALFIT_SOLVER_LSQ_ITERS"]["default"] == 400


def test_p1_2_box_lsq_defaults_to_knob_not_hardcoded_150():
    import inspect

    import portion_solver as ps
    sig = inspect.signature(ps._box_lsq)
    assert sig.parameters["iters"].default is None, (
        "`iters` volvió a ser un default plano — el knob deja de tener efecto")
    assert "iters = SOLVER_LSQ_ITERS" in _PS


def test_p1_2_iters_actually_changes_the_solution():
    """Prueba de que el parámetro está VIVO: el mismo problema resuelto con 1 barrido y con el knob
    da factores distintos. Si alguien vuelve a ignorar `iters`, esto falla."""
    import portion_solver as ps
    A = [[247, 216, 80, 45], [46.5, 4.2, 1.0, 0.0], [0.0, 47.0, 4.3, 0.0], [5.4, 0.5, 7.3, 5.0]]
    b = [700.0, 45.0, 70.0, 25.0]
    w = [ps.SOLVER_W_KCAL, ps.SOLVER_W_PROTEIN, ps.SOLVER_W_CARBS, ps.SOLVER_W_FATS]
    x_one = ps._box_lsq(A, b, w, 0.3, [3.5] * 4, ps.SOLVER_LSQ_REG, iters=1)
    x_knob = ps._box_lsq(A, b, w, 0.3, [3.5] * 4, ps.SOLVER_LSQ_REG)
    assert x_one != x_knob

    def _err(x):
        return sum(w[r] * (sum(A[r][i] * x[i] for i in range(4)) - b[r]) ** 2 for r in range(4))

    assert _err(x_knob) < _err(x_one), "más barridos deberían reducir el error del objetivo"


# ═════════════════ P1-3 · P1-REFINE-RAW-BY-FOOD ═════════════════

def test_p1_3_blind_index_write_is_gone():
    """Anti-regresión del bug exacto: el write posicional a raw sin exigir largos iguales."""
    assert "P1-REFINE-RAW-BY-FOOD" in _PS
    assert 'REFINE_RAW_BY_FOOD = _envb("MEALFIT_REFINE_RAW_BY_FOOD", True)' in _PS
    # el guard viejo `isinstance(raw, list) and idx < len(raw)` no debe existir en el refinador
    assert "isinstance(raw, list) and idx < len(raw)" not in _PS
    # y el contrato nuevo sí: el índice se gana con paralelismo VERIFICADO, no con el largo
    # [P2-RAW-PAIR-BY-FOOD · 2026-07-29] el largo solo es el primer filtro barato.
    assert "_parallel = len(raw) == len(_disp_orig)" in _PS
    assert "_raw_display_parallel_by_food as _par_ok" in _PS


class _FakeDB:
    """Catálogo mínimo: 'X g de <alimento>' → macros proporcionales a los gramos."""

    _PER_G = {"pollo": (1.65, 0.31, 0.0, 0.036), "arroz": (1.30, 0.027, 0.28, 0.003),
              "aceite": (8.84, 0.0, 0.0, 1.0), "cebolla": (0.40, 0.011, 0.093, 0.001)}

    @staticmethod
    def _parse(s: str):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s).lower())
        if not m:
            return None, 0.0
        grams = float(m.group(1).replace(",", "."))
        for food in _FakeDB._PER_G:
            if food in str(s).lower():
                return food, grams
        return None, grams

    def macros_from_ingredient_string(self, s):
        food, grams = self._parse(s)
        if not food or grams <= 0:
            return None
        k, p, c, f = self._PER_G[food]
        return {"kcal": k * grams, "protein": p * grams, "carbs": c * grams, "fats": f * grams}

    def grams_from_ingredient_string(self, s):
        return self._parse(s)[1]


def _grams_of(line: str) -> float:
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(line).lower())
    return float(m.group(1).replace(",", ".")) if m else 0.0


def test_p1_3_parallel_lists_still_use_index_path():
    """Largos iguales → camino por índice (exacto y barato). No debe cambiar el comportamiento."""
    from portion_solver import refine_day_portions_integer
    meal = {"name": "Pollo con arroz",
            "ingredients": ["150 g de pollo", "200 g de arroz"],
            "ingredients_raw": ["150 g de pollo", "200 g de arroz"]}
    moves = refine_day_portions_integer([meal], {"kcal": 900, "protein": 70, "carbs": 40, "fats": 12},
                                        _FakeDB(), floor_g=20.0, cap_g=400.0)
    assert moves > 0
    assert meal["ingredients"] == meal["ingredients_raw"], (
        f"lockstep roto en el camino por índice: {meal['ingredients']} vs {meal['ingredients_raw']}")


def test_p1_3_misaligned_lists_do_not_scale_the_wrong_food():
    """EL BUG: raw trae una línea extra al PRINCIPIO ('Sal al gusto' / 'cebolla' — el caso que el
    tracer midió como NORMAL). Con el guard viejo, el factor del pollo (idx 0) aterrizaba sobre la
    línea 0 de raw, que es OTRO alimento. Aquí se exige que eso ya no ocurra."""
    from portion_solver import refine_day_portions_integer
    meal = {"name": "Pollo con arroz",
            "ingredients": ["150 g de pollo", "200 g de arroz"],
            "ingredients_raw": ["30 g de cebolla", "150 g de pollo", "200 g de arroz"]}
    moves = refine_day_portions_integer([meal], {"kcal": 900, "protein": 70, "carbs": 40, "fats": 12},
                                        _FakeDB(), floor_g=20.0, cap_g=400.0)
    assert moves > 0, "el refinador debería haber movido algo en el display"
    raw = meal["ingredients_raw"]
    # la cebolla NO se toca: no es una línea que el refinador movió en el display
    assert _grams_of(raw[0]) == pytest.approx(30.0), (
        f"el factor del pollo aterrizó sobre la cebolla — el bug sigue vivo: {raw}")
    # y el pollo del raw sigue al pollo del display (mapeo por alimento)
    assert "pollo" in raw[1].lower()
    assert _grams_of(raw[1]) == pytest.approx(_grams_of(meal["ingredients"][0])), (
        f"raw del pollo desincronizado del display: {raw} vs {meal['ingredients']}")


def test_p1_3_knob_off_skips_sync_instead_of_blind_index():
    """Rollback: con el knob OFF el sync se SALTA (raw intacto). Nunca vuelve al índice ciego."""
    import portion_solver as ps
    from portion_solver import refine_day_portions_integer
    meal = {"name": "Pollo con arroz",
            "ingredients": ["150 g de pollo", "200 g de arroz"],
            "ingredients_raw": ["30 g de cebolla", "150 g de pollo", "200 g de arroz"]}
    _prev = ps.REFINE_RAW_BY_FOOD
    ps.REFINE_RAW_BY_FOOD = False
    try:
        refine_day_portions_integer([meal], {"kcal": 900, "protein": 70, "carbs": 40, "fats": 12},
                                    _FakeDB(), floor_g=20.0, cap_g=400.0)
    finally:
        ps.REFINE_RAW_BY_FOOD = _prev
    assert meal["ingredients_raw"] == ["30 g de cebolla", "150 g de pollo", "200 g de arroz"]


# ═════════════════ P1-4 · P1-CLOSER-CONDITION-SSOT ═════════════════

def test_p1_4_adhoc_renal_detector_removed_from_closer():
    assert "P1-CLOSER-CONDITION-SSOT" in _GO
    assert '"renal" in str(c).lower() or "erc" in str(c).lower()' not in _GO
    assert "_renal = _is_renal_condition(_fd)" in _GO
    assert "_dm2 = _is_diabetes_condition(_fd)" in _GO
    assert "_dyslip_or_hta = _is_dyslip_or_hta_condition(_fd)" in _GO


@pytest.mark.parametrize("condition", [
    "Nefropatía diabética", "Insuficiencia renal crónica", "diálisis peritoneal",
    "CKD estadio 3", "creatinina alta", "problema del riñón", "glomerulonefritis",
])
def test_p1_4_renal_freetext_variants_now_detected(condition):
    """El detector ad-hoc solo veía 'renal'/'erc'. Como NO hay chip renal en el formulario, todo
    perfil renal llega como texto libre — justo donde estas variantes son las probables."""
    import graph_orchestrator as go
    assert go._is_renal_condition({"medicalConditions": [condition]}) is True


@pytest.mark.parametrize("condition", [
    "Colesterol", "trigliceridos altos", "hipertenso", "tensión alta",
    "LDL alto", "hipercolesterolemia", "presión alta",
])
def test_p1_4_dyslip_hta_ssot_covers_freetext(condition):
    import graph_orchestrator as go
    assert go._is_dyslip_or_hta_condition({"medicalConditions": [condition]}) is True


@pytest.mark.parametrize("condition", ["glicemia alta", "T2DM", "intolerancia a la glucosa",
                                       "hiperglucemia", "DM-2"])
def test_p1_4_diabetes_ssot_covers_what_adhoc_missed(condition):
    import graph_orchestrator as go
    assert go._is_diabetes_condition({"medicalConditions": [condition]}) is True


def test_p1_4_migrated_stems_live_in_the_ssot_not_in_a_parallel_list():
    """Los stems propios del closer se movieron a constants.py — si vuelven a duplicarse, esto
    deja de ser la SSOT y el drift regresa."""
    assert '"glicem", "glucem",' in _CONST
    assert '"dislip", "colesterol", "hiperlip", "trigliceridos",' in _CONST
    assert '"hipertens",' in _CONST


def test_p1_4_dyslip_helper_fails_secure_without_ssot(monkeypatch):
    """Sin el SSOT importable, el helper asume la condición (no escalar queso/embutido sobre duda)."""
    import builtins

    import graph_orchestrator as go
    _real = builtins.__import__

    def _boom(name, *a, **kw):
        if name == "constants":
            raise ImportError("simulado")
        return _real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert go._is_dyslip_or_hta_condition({"medicalConditions": []}) is True


# ═════════════════ P1-5 · P1-UPDATE-CLINICAL-RECAP ═════════════════

def test_p1_5_knob_and_signature():
    assert 'UPDATE_MACRO_ENGINE_CLINICAL_RECAP = _env_bool("MEALFIT_UPDATE_MACRO_ENGINE_CLINICAL_RECAP", True)' in _GO
    import inspect

    import graph_orchestrator as go
    assert go.UPDATE_MACRO_ENGINE_CLINICAL_RECAP is True
    assert "form_data" in inspect.signature(go.apply_update_macro_engine).parameters


def test_p1_5_recap_runs_after_sizing_and_before_qty_sync():
    """El ORDEN es la corrección entera: los caps clínicos deben correr DESPUÉS del rebalance/refine
    (que es quien re-infla) y ANTES del qty-sync de pasos (última mutación textual)."""
    i_fats = _GO.index("_trim_day_fats_to_target(_meals, float(_fg), db, tol=FATS_POSTCLOSER_RELEVEL_TOL)")
    # Ancla ÚNICA a propósito: el marker `P1-UPDATE-CLINICAL-RECAP` aparece también en el comentario
    # del knob (muy arriba del archivo), así que `.index()` sobre el marker aterrizaría ahí y el test
    # pasaría en vacío. Lección de la colisión por prefijo compartido de esta misma semana.
    assert _GO.count("Re-aplica los caps clínicos") == 1
    i_recap = _GO.index("Re-aplica los caps clínicos")
    i_sync = _GO.index("_sync_recipe_step_quantities(_m)", i_recap)
    assert i_fats < i_recap < i_sync


def test_p1_5_all_update_surfaces_pass_form_data():
    """Las 4 superficies user-facing de update pasan su form_data SERVER-SIDE. Sin él, el re-cap se
    omite en silencio y el gap sigue abierto exactamente donde el audit lo encontró."""
    assert "form_data=_micro_form)" in _PLANS
    assert _PLANS.count("form_data=_micro_form)") >= 2, "swap_persist y swap_persist_day"
    assert _TOOLS.count("form_data=_micro_form_cm)") >= 2, "chat_modify pre-listas y fresh"


def test_p1_5_recap_invoked_on_touched_day(monkeypatch):
    """Funcional: con un día fuera de banda y `form_data` presente, los dos caps clínicos se invocan
    sobre ESE día. Es el escenario del audit (swap de un DM2 que re-infla la batata capada)."""
    import graph_orchestrator as go
    calls = {"dm2": 0, "bar": 0}

    monkeypatch.setattr(go, "_rebalance_day_macros_to_target",
                        lambda *a, **kw: 1)
    monkeypatch.setattr(go, "_trim_day_fats_to_target", lambda *a, **kw: 0)
    monkeypatch.setattr(go, "GLOBAL_DAY_REFINE_ENABLED", False)
    monkeypatch.setattr(go, "_sync_recipe_step_quantities", lambda *a, **kw: None)

    def _dm2(days, fd, db=None, **kw):
        calls["dm2"] += 1
        assert len(days) == 1, "el re-cap debe correr sobre el día tocado, no sobre el plan entero"
        return 1

    monkeypatch.setattr(go, "cap_dm2_high_gi_portions", _dm2)
    monkeypatch.setattr(go, "cap_bariatric_portions",
                        lambda days, fd, db=None, **kw: calls.__setitem__("bar", calls["bar"] + 1) or 0)

    plan = {
        "macros": {"protein": "100 g", "carbs": "200 g", "fats": "60 g"},
        "days": [{"meals": [{"protein": 20, "carbs": 40, "fats": 10, "cals": 330}]}],
    }
    touched = go.apply_update_macro_engine(plan, surface="swap_persist", db=object(),
                                           form_data={"medicalConditions": ["Diabetes Tipo 2"]})
    assert touched == 1
    assert calls["dm2"] == 1 and calls["bar"] == 1


def test_p1_5_recap_skipped_without_form_data(monkeypatch):
    """Sin form_data no se puede conocer la condición → el re-cap NO corre (y el motor sigue igual
    que antes). Ancla el fail-safe, no el fail-open silencioso.

    [P1-UPDATE-RECAP-ALL-SURFACES · 2026-07-30] El vehículo era `surface="budget_convergence"` —
    una superficie REAL que en v4 no pasaba form_data. Eso convertía este test en documentación
    del hueco: certificaba como correcto el estado que el audit v5 identificó como el bug. Ahora
    las 4 superficies de generación pasan su form_data y el vehículo es un nombre neutro: lo que
    este test protege es el CONTRATO de la función (fail-safe sin form_data), no el hueco.
    La omisión ya no es silenciosa — `test_p1_update_recap_all_surfaces.py` ancla el warning."""
    import graph_orchestrator as go
    calls = {"n": 0}
    monkeypatch.setattr(go, "_rebalance_day_macros_to_target", lambda *a, **kw: 1)
    monkeypatch.setattr(go, "_trim_day_fats_to_target", lambda *a, **kw: 0)
    monkeypatch.setattr(go, "GLOBAL_DAY_REFINE_ENABLED", False)
    monkeypatch.setattr(go, "_sync_recipe_step_quantities", lambda *a, **kw: None)
    monkeypatch.setattr(go, "cap_dm2_high_gi_portions",
                        lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1) or 0)

    plan = {
        "macros": {"protein": "100 g", "carbs": "200 g", "fats": "60 g"},
        "days": [{"meals": [{"protein": 20, "carbs": 40, "fats": 10, "cals": 330}]}],
    }
    go.apply_update_macro_engine(plan, surface="caller_legacy_sin_form_data", db=object())
    assert calls["n"] == 0
