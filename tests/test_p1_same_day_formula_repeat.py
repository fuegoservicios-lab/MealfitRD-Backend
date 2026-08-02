"""[P1-SAME-DAY-FORMULA-REPEAT · 2026-08-02] Fórmula repetida el mismo día (queja de creatividad).

Caso real 2026-08-02: el plan sirvió Desayuno "Bowl Cremoso de Lechosa y Avena Tostada con
granola y canela" + Merienda "Avena Cremosa con canela, mango y almendras Tostadas" — misma
FÓRMULA (avena+canela+fruta+frutos secos tostados en un bowl cremoso), solo cambió la fruta. El
gate `same_day_protein_repeats` no lo veía (sin proteína repetida) porque las BASES
(avena/arroz/yuca/plátano) y la FÓRMULA no tenían guard.

Este archivo ancla el detector warn-only `_same_day_formula_repeat_pairs` (junto al de proteína
en `build_variety_report`, `graph_orchestrator.py`) y su wiring en `plan_quality_index.py`:
  A) el caso real detecta (bowl lechosa+avena vs avena cremosa+mango).
  B) el caso negativo NO detecta (avena cremosa desayuno + arepitas de avena saladas merienda —
     formato Y perfil distintos).
  C) el gate de proteína (`same_day_protein_repeats`) sigue intacto — este detector es un campo
     SEPARADO, no lo reemplaza ni lo toca.
  D) warn-only: nunca bloquea (`ok`/gates existentes no dependen de este campo).
  E) NO staple-aware a propósito (decisión de producto separada de la exención de proteína).
"""
import pytest

from graph_orchestrator import (
    build_variety_report,
    _same_day_formula_repeat_pairs,
    _meal_formula_signature,
)


def _meal(name, ingredients=None):
    return {"name": name, "ingredients": ingredients or []}


# ---------------------------------------------------------------------------
# A) Caso real: detecta
# ---------------------------------------------------------------------------

def test_real_case_bowl_lechosa_avena_vs_avena_cremosa_mango_detects():
    plan = {"days": [{"day": 1, "meals": [
        _meal("Bowl Cremoso de Lechosa y Avena Tostada con granola y canela"),
        _meal("Avena Cremosa con canela, mango y almendras Tostadas"),
    ]}]}
    rep = build_variety_report(plan)
    assert rep["same_day_formula_repeats"] == 1, rep["issues"]
    assert any("comparten fórmula" in issue for issue in rep["issues"])


def test_real_case_pairs_helper_directly():
    from constants import strip_accents
    meals = [
        _meal("Bowl Cremoso de Lechosa y Avena Tostada con granola y canela"),
        _meal("Avena Cremosa con canela, mango y almendras Tostadas"),
    ]
    pairs = _same_day_formula_repeat_pairs(meals, strip_accents)
    assert len(pairs) == 1
    _m_a, _m_b, shared = pairs[0]
    assert shared == {"canela", "fruta"}


# ---------------------------------------------------------------------------
# B) Caso negativo: formato Y perfil distintos → NO detecta
# ---------------------------------------------------------------------------

def test_negative_case_avena_cremosa_vs_arepitas_saladas_does_not_detect():
    plan = {"days": [{"day": 1, "meals": [
        _meal("Avena cremosa"),
        _meal("Arepitas de avena saladas"),
    ]}]}
    rep = build_variety_report(plan)
    assert rep["same_day_formula_repeats"] == 0, rep["issues"]


def test_negative_case_same_base_different_format_and_profile_full_names():
    """Espejo con nombres completos (con acompañantes) — el formato horneado+perfil salado
    de las arepitas debe seguir ganando sobre compartir base+canela/fruta."""
    plan = {"days": [{"day": 1, "meals": [
        _meal("Avena Cremosa con canela y mango"),
        _meal("Arepitas de Avena saladas con queso"),
    ]}]}
    rep = build_variety_report(plan)
    assert rep["same_day_formula_repeats"] == 0, rep["issues"]


def test_negative_case_different_base_same_format_does_not_detect():
    """Misma familia de formato (bowl cremoso), BASE distinta (avena vs. yuca) → no es la misma
    fórmula."""
    plan = {"days": [{"day": 1, "meals": [
        _meal("Bowl Cremoso de Avena con canela y mango"),
        _meal("Bowl Cremoso de Yuca con canela y mango"),
    ]}]}
    rep = build_variety_report(plan)
    assert rep["same_day_formula_repeats"] == 0, rep["issues"]


def test_negative_case_only_one_shared_accompaniment_does_not_detect():
    """Misma base+formato pero SOLO 1 clase de acompañante compartida (canela) → no alcanza el
    umbral de ≥2."""
    plan = {"days": [{"day": 1, "meals": [
        _meal("Avena Cremosa con canela"),
        _meal("Batido de Avena con canela y coco"),
    ]}]}
    rep = build_variety_report(plan)
    # comparten base=avena, formato=cremoso_frio, savory=False; acompañantes: {canela} vs
    # {canela, coco} → intersección {canela} = 1 clase, no dispara.
    assert rep["same_day_formula_repeats"] == 0, rep["issues"]


def test_negative_case_different_days_never_compared():
    """El detector es SAME-DAY — el mismo par en días distintos no cuenta."""
    plan = {"days": [
        {"day": 1, "meals": [_meal("Bowl Cremoso de Lechosa y Avena Tostada con granola y canela")]},
        {"day": 2, "meals": [_meal("Avena Cremosa con canela, mango y almendras Tostadas")]},
    ]}
    rep = build_variety_report(plan)
    assert rep["same_day_formula_repeats"] == 0, rep["issues"]


def test_meal_without_recognized_base_returns_none_signature():
    from constants import strip_accents
    assert _meal_formula_signature(_meal("Pechuga de Pollo a la Plancha"), strip_accents) is None


# ---------------------------------------------------------------------------
# C) El gate de proteína sigue intacto (campo separado)
# ---------------------------------------------------------------------------

def test_protein_gate_unaffected_by_formula_detector():
    """Huevo repetido el mismo día sigue contando en `same_day_protein_repeats`, y el nuevo
    campo `same_day_formula_repeats` no lo pisa ni lo duplica (avena no está en ninguno de los
    dos platos de este caso)."""
    plan = {"days": [{"day": 1, "meals": [
        _meal("Batido con claras de huevo"),
        _meal("Tortilla de huevos"),
    ]}]}
    rep = build_variety_report(plan)
    assert rep["same_day_protein_repeats"] == 1
    assert rep["same_day_formula_repeats"] == 0


def test_both_gates_can_fire_independently_same_day():
    """Un día puede disparar AMBOS: proteína repetida Y fórmula repetida, en pares
    potencialmente distintos de comidas — los contadores son independientes."""
    plan = {"days": [{"day": 1, "meals": [
        _meal("Bowl Cremoso de Lechosa y Avena Tostada con granola y canela"),
        _meal("Avena Cremosa con canela, mango y almendras Tostadas"),
        _meal("Revoltillo de huevos con vegetales"),
        _meal("Tortilla de huevos con casabe"),
    ]}]}
    rep = build_variety_report(plan)
    assert rep["same_day_formula_repeats"] == 1
    assert rep["same_day_protein_repeats"] == 1


# ---------------------------------------------------------------------------
# D) Warn-only: nunca bloquea
# ---------------------------------------------------------------------------

def test_formula_repeat_feeds_issues_but_report_ok_field_is_advisory():
    """`issues` no vacío ⇒ `ok=False`, pero eso es advisory (igual que fruit_repeats/
    sweet_savory_clash — NINGÚN caller trata `ok` como gate duro; los gates reales filtran por
    campos numéricos específicos con su propio knob, ver `same_day_protein_repeats` +
    VARIETY_GATE_SAME_DAY_PROTEIN)."""
    plan = {"days": [{"day": 1, "meals": [
        _meal("Bowl Cremoso de Lechosa y Avena Tostada con granola y canela"),
        _meal("Avena Cremosa con canela, mango y almendras Tostadas"),
    ]}]}
    rep = build_variety_report(plan)
    assert rep["same_day_formula_repeats"] == 1
    assert rep["ok"] is False  # advisory: refleja que `issues` no está vacío, no bloquea nada


def test_quality_index_wires_same_day_formula_repeats():
    """El índice de calidad (`plan_quality_index`) penaliza `same_day_formula_repeats` igual
    que los demás defectos de variedad — cuenta, no bloquea."""
    from plan_quality_index import compute_plan_quality_index
    plan = {"days": [{"day": 1, "meals": [
        _meal("Bowl Cremoso de Lechosa y Avena Tostada con granola y canela"),
        _meal("Avena Cremosa con canela, mango y almendras Tostadas"),
    ]}]}
    rep = build_variety_report(plan)
    plan["variety_report"] = rep
    result = compute_plan_quality_index(plan, variety_report=rep)
    variedad = result.get("componentes", result).get("variedad", result) if isinstance(result, dict) else None
    # Estructura exacta la decide plan_quality_index; solo verificamos que el defecto
    # 'same_day_formula_repeats' aparece contado en algún nivel del resultado.
    import json
    assert "same_day_formula_repeats" in json.dumps(result)


# ---------------------------------------------------------------------------
# E) NO staple-aware a propósito
# ---------------------------------------------------------------------------

def test_staple_declared_does_not_exempt_formula_repeat():
    """A diferencia de `same_day_protein_repeats`, declarar 'avena' como básico NO exime la
    fórmula repetida — son decisiones de producto separadas (repetir INGREDIENTE con técnica
    distinta vs. clonar la fórmula entera)."""
    plan = {"days": [{"day": 1, "meals": [
        _meal("Bowl Cremoso de Lechosa y Avena Tostada con granola y canela"),
        _meal("Avena Cremosa con canela, mango y almendras Tostadas"),
    ]}]}
    # user_staples normalmente son labels de PROTEÍNA (_SAME_DAY_PROTEIN_GATE_LABELS); "avena"
    # no es una de ellas, pero el punto del test es que el detector de fórmula ni siquiera
    # CONSULTA `user_staples` — pasar un set no vacío no cambia el resultado.
    rep_sin_staples = build_variety_report(plan, user_staples=None)
    rep_con_staples = build_variety_report(plan, user_staples={"pollo", "huevo"})
    assert rep_sin_staples["same_day_formula_repeats"] == 1
    assert rep_con_staples["same_day_formula_repeats"] == 1
