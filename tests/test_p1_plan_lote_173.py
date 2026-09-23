# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-173 · 2026-09-23] Segunda vuelta de la batería REAL del generador (RD), sobre lo que el 172 dejó a la vista.

1. **Embarazo, a la tercera**: con el 172 el plan ya no caía al de emergencia, pero el primer intento seguía rechazado
   CRÍTICO por «queso blanco fresco sin pasteurizar»: el queso que el cerrador de proteína añade DESPUÉS de las
   sustituciones clínicas no llevaba la etiqueta, y los PASOS seguían diciendo «queso blanco fresco». La primera
   corrida pedía además la leche. Ahora las etiquetas corren también al final de la cadena de calidad (antes del
   revisor y en el escudo del guardado), sobre la lista, la compra y los pasos.
2. **El plan de emergencia salía SIN lista de compras** (batería real, embarazo: `aggregated_shopping_list = []`).
3. **La pechuga de pavo FRESCA se mostraba en «lonjas»** (20 g la lonja, presentación de embutido): el revisor la leyó
   como embutido alto en sodio en un plan de hipertensión y lo rechazó CRÍTICO.
4. **HTA y DM2 en el prompt**: el revisor rechazó CRÍTICO por queso en exceso y polvo de hornear (sodio oculto) y por
   almidones apilados en el desayuno (pan + cebada + mango). Las reglas del prompt no decían ni lo uno ni lo otro."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ─────────────────────────── 1. Embarazo: leche, pasos y al final de la cadena ───────────────────────────

def test_embarazo_leche_y_pasos():
    import embarazo_seguro as e
    plan = {"days": [{"meals": [{"name": "Mangú con queso", "ingredients": ["30 g de queso blanco fresco", "200 ml de leche",
                                                                            "60 ml de leche de coco"],
                                 "ingredients_raw": ["30 g de queso blanco fresco", "200 ml de leche"],
                                 "recipe": ["Desmenuza el queso blanco fresco encima y sirve con la ricotta."]}]}]}
    assert e.etiquetar(plan, {"medicalConditions": ["Embarazo"]}) == 1
    m = plan["days"][0]["meals"][0]
    assert m["ingredients"][1] == "200 ml de leche pasteurizada"
    assert m["ingredients"][2] == "60 ml de leche de coco", "la bebida vegetal no se etiqueta"
    assert "queso blanco fresco pasteurizado encima" in m["recipe"][0]
    assert "ricotta pasteurizada" in m["recipe"][0], "la ricotta es femenina"
    assert e.etiquetar(plan, {"medicalConditions": ["Embarazo"]}) == 0, "idempotente"


def test_lactancia_yogur_pasteurizado_y_mero_a_tilapia():
    """Batería real (Lactancia): «mero o tilapia» + 175 g de atún y el yogur sin «pasteurizado» → dos rechazos CRÍTICOS."""
    import embarazo_seguro as e
    assert e._linea_pescado("150 g de mero o tilapia") == "150 g de tilapia"
    assert e._linea_pescado("150 g de filete de mero") == "150 g de filete de tilapia"
    assert e._linea_queso("135 g de yogurt griego natural").endswith("pasteurizado")
    assert e._linea_queso("20 g de yogurt de coco sin azúcar") == "20 g de yogurt de coco sin azúcar"
    plan = {"days": [{"meals": [{"name": "Mero guisado", "ingredients": ["150 g de mero"]}]}]}
    assert e.etiquetar(plan, {"medicalConditions": ["Lactancia"]}) == 1, "lactancia usa la misma regla que embarazo"


def test_hta_bajo_en_sodio_en_queso_y_lata():
    import etiquetas_clinicas as ec
    plan = {"days": [{"meals": [{"name": "Casabe con queso y atún",
                                 "ingredients": ["30 g de queso blanco fresco", "80 g de atún en agua", "20 g de queso parmesano"],
                                 "ingredients_raw": ["30 g de queso blanco fresco", "80 g de atún en agua"]}]}]}
    assert ec.etiquetar(plan, {"medicalConditions": ["Hipertensión"]}) == 1
    m = plan["days"][0]["meals"][0]
    assert m["ingredients"][0].endswith("bajo en sodio") and m["ingredients"][1].endswith("bajo en sodio")
    assert m["ingredients"][2] == "20 g de queso parmesano"
    assert ec.etiquetar(plan, {"medicalConditions": ["Hipertensión"]}) == 0, "idempotente"
    assert ec.etiquetar({"days": [{"meals": [{"name": "x", "ingredients": ["30 g de queso blanco"]}]}]},
                        {"medicalConditions": ["Ninguna"]}) == 0


def test_dm2_cambia_el_cundeamor_por_tayota():
    import condition_rules as cr
    dm2 = next(r for r in cr.CONDITION_RULES if r.id == "dm2")
    assert any("cundeamor" in toks and repl == "Tayota" for toks, repl, *_ in dm2.substitutions)


def test_las_etiquetas_corren_al_final_de_la_cadena_de_calidad():
    """El cerrador de proteína añade el cottage DESPUÉS de las sustituciones clínicas: la etiqueta tiene que correr en el
    escudo (`_finalize_plan_data_for_insert`), que es también la cola de assemble ANTES del revisor."""
    src = _src("db_plans.py")
    i = src.index("def _finalize_plan_data_for_insert")
    fin = src.index("\ndef ", i + 10)
    cuerpo = src[i:fin]
    assert "etiquetas_clinicas" in cuerpo and ".etiquetar(" in cuerpo


def test_el_escudo_etiqueta_un_plan_de_embarazo(monkeypatch):
    import db_plans
    monkeypatch.setattr(db_plans, "_build_clinical_form",
                        lambda _u: {"medicalConditions": ["Embarazo"], "allergies": [], "dislikes": []})
    pd = {"days": [{"meals": [{"name": "Casabe con queso", "meal": "Merienda", "ingredients": ["30 g de queso blanco fresco"],
                               "ingredients_raw": ["30 g de queso blanco fresco"], "recipe": ["Sirve el queso blanco fresco."],
                               "cals": 120, "protein": 7, "carbs": 10, "fats": 6}]}],
          "calories": 1800, "macros": {"protein": "110g", "carbs": "200g", "fats": "60g"}}
    data = {"user_id": "u-emb", "plan_data": pd}
    db_plans._finalize_plan_data_for_insert(data)
    m = data["plan_data"]["days"][0]["meals"][0]
    assert any("pasteuriz" in x for x in m["ingredients"]), m["ingredients"]


# ─────────────────────────── 2. El plan de emergencia lleva lista ───────────────────────────

def test_el_plan_de_emergencia_recompone_su_lista():
    go = _src("graph_orchestrator.py")
    i = go.index("_apply_final_defense_guardrails(\n") if "_apply_final_defense_guardrails(\n" in go else go.index("_apply_final_defense_guardrails(")
    j = go.index("_apply_final_defense_guardrails(", i + 10)   # la llamada, no la definición
    bloque = go[j:j + 1500]
    assert "_is_fallback" in bloque and "_recompute_aggregates_after_swap(final_state)" in bloque


# ─────────────────────────── 3. Pavo fresco en gramos ───────────────────────────

def test_la_pechuga_de_pavo_fresca_no_se_muestra_en_lonjas():
    import humanize_ingredients as hz
    assert "pechuga de pavo" not in hz.DOMINICAN_HOUSEHOLD_MEASURES
    out = hz.humanize_ingredient("195 g de pechuga de pavo")
    assert "lonja" not in out, out
    assert "jamon" in hz.DOMINICAN_HOUSEHOLD_MEASURES, "el embutido de verdad sí va en lonjas"


# ─────────────────────────── 4. HTA y DM2 en el prompt ───────────────────────────

def test_hta_pide_queso_medido_sin_polvo_de_hornear_y_pavo_fresco():
    import condition_rules as cr
    hta = next(r for r in cr.CONDITION_RULES if r.id == "hta")
    txt = hta.prompt_block.lower()
    assert "polvo de hornear" in txt and "queso" in txt and "30 g" in txt and "lonjas" in txt


def test_dm2_una_sola_fuente_de_almidon_por_comida():
    import condition_rules as cr
    dm2 = next(r for r in cr.CONDITION_RULES if r.id == "dm2")
    assert "una sola fuente principal de almidón" in dm2.prompt_block.lower()


# ─────────────────────────── marcador ───────────────────────────

def test_marker_173():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 173 and m.group(2) >= "2026-09-23"
