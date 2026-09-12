# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-23 · 2026-09-12] C2 del plan de pendientes (CUL-P0-03): el contrato sobre la receta FINAL.

Los pasos nombran cantidades y la lista de ingredientes es la que compra, cuenta macros y llena la Nevera. Entre
generar y persistir, la lista la mutan media docena de reparadores; los sincronizadores de pasos que ya existían
corren en puntos fijos, y *una reparación que corre después del sincronizador deja lista y pasos desincronizados*.
Medido en el corpus fijo (64 comidas): 47 V7a, 38 V7e, 11 V6, 4 V4 — 100 de 111 hallazgos de capa 1 son cantidades;
54 de las 68 comidas que el dueño marcó con defecto lo describen así en su nota.

`recipe_contract.reconcile_step_quantities` reescribe en los pasos la cantidad que contradice a la lista, familia por
familia, con las guardas que aquí se anclan; corre ÚLTIMO en los dos finalizadores del persist boundary. Y V4 pasa a
atribuir los gramos por GRAMÁTICA («70 g de nabo, 265 g de tomate»), porque un medidor que atribuye por cercanía no
puede alimentar un reparador.
"""
from __future__ import annotations

import copy
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import culinary_coherence as cc  # noqa: E402
import recipe_contract as rc  # noqa: E402


def _row(name, aliases=(), category="vegetal", rte=False, prep=("hervir", "saltear", "crudo")):
    return {"name": name, "aliases": list(aliases), "category": category, "ready_to_eat": rte, "prep_methods": list(prep)}


_CAT = [_row("Huevo", ["huevos"], "proteina"), _row("Ajo", ["dientes de ajo", "diente de ajo"]), _row("Cebolla"),
        _row("Tomate", ["tomates"]), _row("Maní", ["mani"], "fruto seco", True, ["crudo", "tostar"]),
        _row("Pan integral", ["pan integral personal"], "cereal", True, ["tostar"]), _row("Avena", [], "cereal"),
        _row("Nabo"), _row("Pepino"), _row("Mandarina", ["mandarinas"], "fruta", True, ["crudo"]),
        _row("Yogur natural", ["yogurt natural"], "lacteo", True, ["crudo"])]
_INDEX = cc.build_culinary_index(_CAT)


def _meal(ings, pasos, name="Plato"):
    return {"meal": "Almuerzo", "name": name, "ingredients": list(ings), "recipe": list(pasos)}


def _checks(meal):
    return sorted(v["check"] for v in cc.culinary_contract_scan({"days": [{"day": 1, "meals": [meal]}]}, _CAT))


# ────────────────────────────────────────────────────────────── la lista tiene la última palabra

def test_un_paso_que_pide_mas_piezas_de_las_compradas_se_recorta_a_la_lista():
    m = _meal(["3 huevos", "1 rebanada de pan integral"], ["Mise en place: casca 6 huevos; tuesta 1 rebanada de pan integral."])
    assert "V7e" in _checks(m)
    r = rc.reconcile_step_quantities(m, _INDEX)
    assert m["recipe"][0] == "Mise en place: casca 3 huevos; tuesta 1 rebanada de pan integral."
    assert r["reescritas"] == 1 and r["familias"] == {"pieza": 1} and "V7e" not in _checks(m)


def test_una_sola_mencion_se_alinea_en_las_dos_direcciones():
    m = _meal(["15 g de avena", "60 g de maní"], ["Mise en place: mide 60 g de avena y 15 g de maní."])
    r = rc.reconcile_step_quantities(m, _INDEX)
    assert m["recipe"][0] == "Mise en place: mide 15 g de avena y 60 g de maní."
    assert r["reescritas"] == 2 and r["familias"] == {"g": 2}


def test_las_unidades_contables_concuerdan_en_numero_y_articulo():
    m = _meal(["1 rebanada de pan integral", "½ diente de ajo"],
              ["El Toque de Fuego: tuesta las 2 rebanadas de pan integral; pica 1 diente de ajo."])
    assert "V6" in _checks(m)
    rc.reconcile_step_quantities(m, _INDEX)
    assert m["recipe"][0] == "El Toque de Fuego: tuesta la 1 rebanada de pan integral; pica ½ diente de ajo."
    assert "V6" not in _checks(m)


def test_la_gramatica_de_los_gramos_manda_sobre_la_cercania():
    """«70 g de nabo, 265 g de tomate»: por cercanía los 265 g eran del nabo (2 de 4 V4 del corpus)."""
    assert cc._v4_grams_by_food(cc._norm("corta 70 g de nabo, 265 g de tomate y 100 g de pepino"), _INDEX) == \
        {"Nabo": 70.0, "Tomate": 265.0, "Pepino": 100.0}
    assert cc._v4_grams_by_food(cc._norm("mide ¼ taza de yogurt natural (90 g), 5 g de maní"), _INDEX) == \
        {"Yogur natural": 90.0, "Maní": 5.0}
    # sin dueño gramatical no hay atribución: «(90 g), 5 g» tras un alimento desconocido
    assert cc._v4_grams_by_food(cc._norm("mide algo raro (90 g), 5 g de maní"), _INDEX) == {"Maní": 5.0}
    m = _meal(["70 g de nabo", "265 g de tomate"], ["Mise en place: corta 70 g de nabo, 100 g de tomate."])
    rc.reconcile_step_quantities(m, _INDEX)
    assert m["recipe"][0] == "Mise en place: corta 70 g de nabo, 265 g de tomate."


def test_el_partitivo_sobra_con_un_entero():
    m = _meal(["1 cebolla"], ["Mise en place: pica ¼ de cebolla."])
    rc.reconcile_step_quantities(m, _INDEX)
    assert m["recipe"][0] == "Mise en place: pica 1 cebolla."


# ────────────────────────────────────────────────────────────── lo que NO se toca, y por qué

def test_cruzar_el_singular_plural_a_ciegas_no_se_hace():
    m = _meal(["2.5 tomates"], ["Mise en place: pica 1 tomate."])
    r = rc.reconcile_step_quantities(m, _INDEX)
    assert m["recipe"][0] == "Mise en place: pica 1 tomate." and r["sin_reparar"] == {"gramatical": 1}
    assert "V7a" in _checks(m), "V7a sigue viéndolo: el contrato no lo esconde"


def test_un_reparto_entre_pasos_no_se_sube_pero_el_exceso_si_se_recorta():
    m = _meal(["2 huevos"], ["Mise en place: bate 1 huevo.", "El Toque de Fuego: añade 1 huevo más.", "Montaje: casca 3 huevos."])
    r = rc.reconcile_step_quantities(m, _INDEX)
    assert m["recipe"][:2] == ["Mise en place: bate 1 huevo.", "El Toque de Fuego: añade 1 huevo más."]
    assert m["recipe"][2] == "Montaje: casca 2 huevos."
    assert r["sin_reparar"] == {"reparto": 2} and r["reescritas"] == 1


def test_un_conteo_de_compra_con_gramos_no_infla_el_paso():
    """«1 cebolla (25 g)»: el 1 es de COMPRA y los 25 g lo que se usa. Subir «¼ de cebolla» a 1 cuadruplicaría el plato."""
    m = _meal(["1 cebolla (25 g)"], ["Mise en place: pica ¼ de cebolla."])
    r = rc.reconcile_step_quantities(m, _INDEX)
    assert m["recipe"][0] == "Mise en place: pica ¼ de cebolla." and r["sin_reparar"] == {"conteo_con_gramos": 1}


def test_notas_rangos_y_aproximaciones_quedan_fuera():
    m = _meal(["3 huevos", "4 mandarinas", "90 g de maní"],
              ["⚠ Seguridad alimentaria: cocina bien los 6 huevos.", "💡 Ajustamos 6 huevos.",
               "Se reemplazó 6 huevos por yogur.", "Mise en place: pela 1-2 mandarinas; mide ≈ 30 g de maní."])
    r = rc.reconcile_step_quantities(m, _INDEX)
    assert r["reescritas"] == 0 and m["recipe"][3] == "Mise en place: pela 1-2 mandarinas; mide ≈ 30 g de maní."


def test_la_tolerancia_del_medidor_es_la_del_reparador():
    m = _meal(["100 g de avena", "2 huevos"], ["Mise en place: mide 90 g de avena; casca 2.05 huevos."])
    assert rc.reconcile_step_quantities(m, _INDEX)["reescritas"] == 0, "10 % en gramos y 0,05 piezas están dentro de V4/V7e"


def test_idempotente_fail_open_y_raw_intacto():
    m = _meal(["3 huevos"], ["Mise en place: casca 6 huevos."])
    m["ingredients_raw"] = ["3 huevos"]
    rc.reconcile_step_quantities(m, _INDEX)
    una = list(m["recipe"])
    assert rc.reconcile_step_quantities(m, _INDEX)["reescritas"] == 0 and m["recipe"] == una
    assert m["ingredients_raw"] == ["3 huevos"]
    assert rc.reconcile_step_quantities(None, _INDEX)["reescritas"] == 0        # type: ignore[arg-type]
    assert rc.reconcile_step_quantities({"ingredients": ["3 huevos"], "recipe": "no soy lista"}, _INDEX)["reescritas"] == 0
    assert rc.reconcile_step_quantities(m, {})["reescritas"] == 0


def test_formatear_cantidad_como_escriben_los_pasos():
    assert [rc.formatear_cantidad(v) for v in (3, 0.5, 1.5, 2.5, 1 / 3, 0.25, 2.7)] == ["3", "½", "1½", "2½", "⅓", "¼", "2.7"]


# ────────────────────────────────────────────────────────────── el gancho: ÚLTIMO en los dos finalizadores

def test_knob_modos_y_default_repair(monkeypatch):
    monkeypatch.delenv("MEALFIT_RECIPE_FINAL_CONTRACT", raising=False)
    assert rc.final_contract_mode() == "repair"
    for v, esperado in (("shadow", "shadow"), ("off", "off"), ("REPAIR", "repair"), ("bogus", "repair")):
        monkeypatch.setenv("MEALFIT_RECIPE_FINAL_CONTRACT", v)
        assert rc.final_contract_mode() == esperado


def test_apply_final_contract_repair_shadow_y_off(monkeypatch):
    monkeypatch.setattr(rc, "_index_default", lambda db=None: _INDEX)
    days = [{"day": 1, "meals": [_meal(["3 huevos"], ["Mise en place: casca 6 huevos."])]}]
    monkeypatch.setenv("MEALFIT_RECIPE_FINAL_CONTRACT", "shadow")
    assert rc.apply_final_contract(copy.deepcopy(days), None) == "recipe_contract_shadow=1"
    d2 = copy.deepcopy(days)
    rc.apply_final_contract(d2, None)
    assert d2[0]["meals"][0]["recipe"][0] == "Mise en place: casca 6 huevos.", "en sombra no se toca"
    assert d2[0]["meals"][0][rc.TELEMETRIA_KEY] == {"modo": "shadow", "reescritas": 1, "familias": {"pieza": 1}, "sin_reparar": {}}
    monkeypatch.setenv("MEALFIT_RECIPE_FINAL_CONTRACT", "repair")
    d3 = copy.deepcopy(days)
    assert rc.apply_final_contract(d3, None) == "recipe_contract=1"
    assert d3[0]["meals"][0]["recipe"][0] == "Mise en place: casca 3 huevos."
    assert d3[0]["meals"][0][rc.TELEMETRIA_KEY]["modo"] == "repair"
    monkeypatch.setenv("MEALFIT_RECIPE_FINAL_CONTRACT", "off")
    d4 = copy.deepcopy(days)
    assert rc.apply_final_contract(d4, None) == "" and rc.TELEMETRIA_KEY not in d4[0]["meals"][0]
    monkeypatch.setenv("MEALFIT_RECIPE_FINAL_CONTRACT", "repair")
    m = _meal(["3 huevos"], ["Mise en place: casca 6 huevos."])
    assert rc.apply_final_contract_meal(m, None) == 1 and m["recipe"][0] == "Mise en place: casca 3 huevos."
    limpio = _meal(["3 huevos"], ["Mise en place: casca 3 huevos."])
    assert rc.apply_final_contract_meal(limpio, None) == 0 and rc.TELEMETRIA_KEY not in limpio


def test_sin_catalogo_el_contrato_no_toca_nada_y_lo_dice(monkeypatch):
    monkeypatch.setenv("MEALFIT_RECIPE_FINAL_CONTRACT", "repair")
    monkeypatch.setattr(rc, "_index_default", lambda db=None: {})
    days = [{"day": 1, "meals": [_meal(["3 huevos"], ["Mise en place: casca 6 huevos."])]}]
    assert rc.apply_final_contract(days, None) == "recipe_contract=sin_catalogo"
    assert days[0]["meals"][0]["recipe"][0] == "Mise en place: casca 6 huevos."


def test_el_gancho_es_lo_ultimo_en_los_dos_finalizadores():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8", errors="ignore")
    assert "from recipe_contract import apply_final_contract as _recipe_final_contract" in src
    i = src.index("def finalize_plan_data_coherence(")
    j = src.index("\ndef ", i + 10)
    cuerpo = src[i:j]
    a = cuerpo.index("[P1-RECONCILE-LAST-WORD] no-op")
    b = cuerpo.index("_rfc = _recipe_final_contract(days, db)")
    c = cuerpo.rindex('return (total, ", ".join(parts))')
    assert a < b < c, "el contrato corre después del reconciliador display↔raw y justo antes del return"
    assert "P1-PLAN-LOTE-23-FINAL-CONTRACT" in cuerpo
    k = src.index("def finalize_single_meal_recipe_coherence(")
    cuerpo2 = src[k:src.index("\ndef ", k + 10)]
    h = cuerpo2.index("total += _recipe_final_contract_meal(meal, db)")
    assert cuerpo2.index("contract-lint en update no-op") < h < cuerpo2.index("[P1-UPDATE-RECIPE-FINALIZE] {total} fix(es)")


def test_ground_meat_sync_extraido_y_reexportado():
    import graph_orchestrator as go
    assert go._ground_meat_step_noun_sync is rc._ground_meat_step_noun_sync
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8", errors="ignore")
    assert "def _ground_meat_step_noun_sync" not in src and "_ground_meat_step_noun_sync(days)" in src
    days = [{"meals": [{"name": "x", "ingredients": ["150 g de pollo molido"], "recipe": ["Cocina la pechuga de pollo 8 min."]}]}]
    assert rc._ground_meat_step_noun_sync(days) == 1 and days[0]["meals"][0]["recipe"][0] == "Cocina el pollo molido 8 min."


def test_el_god_file_bajo_no_subio():
    n = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8", errors="ignore").count("\n")
    assert n <= 53_100


# ────────────────────────────────────────────────────────────── medido sobre el corpus fijo

def test_sobre_el_corpus_fijo_el_contrato_cierra_las_cantidades_y_es_idempotente():
    from culinary_corpus import cargar, filas_para_medir
    c = cargar(_BACKEND / "scripts" / "data" / "culinary_corpus_2026_09_12.json")
    cat = c["catalogo_filas"]
    index = cc.build_culinary_index(cat)
    import collections
    antes, despues = collections.Counter(), collections.Counter()
    segunda = 0
    for f in filas_para_medir(c):
        pd = copy.deepcopy(f["plan_data"])
        for v in cc.culinary_contract_scan(pd, cat):
            antes[v["check"]] += 1
        rc.reconcile_days(pd["days"], index)
        segunda += rc.reconcile_days(pd["days"], index)["reescritas"]
        for v in cc.culinary_contract_scan(pd, cat):
            despues[v["check"]] += 1
    assert antes["V7e"] >= 30 and antes["V6"] >= 8, "el corpus congelado tiene el defecto que este lote cierra"
    assert despues["V7e"] <= 5 and despues["V6"] == 0 and despues["V4"] == 0
    assert despues["V7a"] >= 30, "lo gramatical se deja a V7a a propósito"
    assert segunda == 0


def test_el_medidor_existe_y_solo_lee():
    src = (_BACKEND / "scripts" / "medir_contrato_receta_final.py").read_text(encoding="utf-8")
    assert "conn.read_only = True" in src and "UPDATE " not in src and "INSERT " not in src
    assert "reconcile_step_quantities" in src and "idempotente" in src


# ────────────────────────────────────────────────────────────── docs, plan, marker

def test_docs_plan_y_marker():
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "## El contrato sobre la receta final (C2 · `P1-PLAN-LOTE-23`" in doc
    assert "reconcile_step_quantities" in doc and "grams_owner" in doc and "MEALFIT_RECIPE_FINAL_CONTRACT" in doc
    knobs = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "| `MEALFIT_RECIPE_FINAL_CONTRACT` | `repair` |" in knobs
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert "| C2 | ✅ 2026-09-12 · contrato final de cantidades" in plan
    app = (_BACKEND / "app.py").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'_LAST_KNOWN_PFIX = "([^"]+)"', app)
    assert m and m.group(1).split("·")[-1].strip() >= "2026-09-12"
    assert "P1-PLAN-LOTE-23" in inspect.getsource(rc)
