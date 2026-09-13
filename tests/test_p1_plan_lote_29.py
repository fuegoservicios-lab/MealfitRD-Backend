# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-29 · 2026-09-12] CUL-P1-04 · el reparador de estructura: adapta la receta a lo que el plato declara SIN
tocar la lista ni una cifra de compra (la nutrición no se mueve), y el paso degradado del camino sin LLM deja de crear
huérfanos.

Lo que se prueba:
  · las tres reparaciones (tortilla → salteado antes del huevo; wrap → lo que cierra + el resto al lado; crema → sólo el
    líquido que espesa + el resto como bebida): lista intacta, V9 deja de acusar, idempotente;
  · lo que no se puede reparar se descarta y se dice (crema sin líquido que apartar);
  · el paso (4) del contrato final: `estructura` en la telemetría sólo si reparó; `shadow` no toca el plato;
  · `degradar_paso` conserva los otros alimentos y cuece lo seco; sin otros, el texto de siempre (tests viejos intactos);
  · el bench del camino degradado imita al cron (sólo V1/V2);
  · docs, plan, marcador.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

import dish_structure as ds  # noqa: E402
import recipe_repair as rr  # noqa: E402
import recipe_contract as rc  # noqa: E402
import culinary_coherence as cc  # noqa: E402

_CAT = [
    {"name": "Lentejas", "aliases": ["lenteja"], "category": "Legumbres", "prep_methods": ["guisado"]},
    {"name": "Leche descremada", "aliases": ["leche"], "category": "Lácteos", "prep_methods": ["ninguno"]},
    {"name": "Pechuga de pollo", "aliases": ["pollo"], "category": "Proteínas", "prep_methods": ["plancha"]},
    {"name": "Tortilla de trigo", "aliases": ["tortilla de trigo integral"], "category": "Granos", "prep_methods": ["tostado"]},
    {"name": "Lechuga", "aliases": [], "category": "Vegetales", "prep_methods": ["crudo"]},
    {"name": "Espinaca", "aliases": ["espinacas"], "category": "Vegetales", "prep_methods": ["salteado"]},
    {"name": "Tomate", "aliases": ["tomates"], "category": "Vegetales", "prep_methods": ["crudo"]},
    {"name": "Clara de huevo", "aliases": ["claras de huevo", "claras"], "category": "Proteínas", "prep_methods": ["cocido"]},
    {"name": "Cebolla", "aliases": ["cebollas"], "category": "Vegetales", "prep_methods": ["sofrito"]},
    {"name": "Ajo", "aliases": ["ajos"], "category": "Vegetales", "prep_methods": ["sofrito"]},
    {"name": "Habichuelas rojas", "aliases": ["habichuelas"], "category": "Legumbres", "prep_methods": ["guisado"]},
    {"name": "Casabe", "aliases": [], "category": "Granos", "prep_methods": ["tostado"], "ready_to_eat": True},
]
_IDX = cc.build_culinary_index(_CAT)


def _meal(name, ings, rec):
    return {"meal": "Almuerzo", "name": name, "ingredients": list(ings), "ingredients_raw": list(ings), "recipe": list(rec)}


CREMA = _meal("Crema de lentejas", ["10 g de lentejas", "300 ml de leche descremada"], ["Licúa las lentejas con la leche hasta obtener una crema espesa."])
WRAP = _meal("Wrap de pollo", ["40 g de tortilla de trigo", "250 g de pechuga de pollo", "80 g de lechuga"], ["Rellena la tortilla con el pollo y la lechuga."])
TORT = _meal("Tortilla de claras con espinaca y tomate", ["4 claras de huevo", "60 g de espinaca", "50 g de tomate"],
             ["Mise en place: bate las claras en un bol.", "El Toque de Fuego: vierte las claras en la sartén con la espinaca y el tomate crudos y cuaja."])


# ─────────────── las tres reparaciones ───────────────

def test_la_tortilla_gana_el_salteado_antes_del_huevo_y_la_lista_no_se_toca():
    m = copy.deepcopy(TORT)
    r = rr.reparar_estructura(m)
    assert r["aplicado"] == ["tortilla_vegetales_crudos"] and m["ingredients"] == TORT["ingredients"]
    assert len(m["recipe"]) == 3 and m["recipe"][1].startswith("El Toque de Fuego: Saltea espinaca y tomate")
    assert "escúrrelos bien antes de añadir las claras" in m["recipe"][1]
    assert ds.relaciones(m) == [], "reparada, V9 deja de acusar"
    assert rr.reparar_estructura(m)["aplicado"] == [] and len(m["recipe"]) == 3, "idempotente"


def test_el_wrap_se_cierra_con_lo_que_cabe_y_el_resto_va_al_lado():
    m = copy.deepcopy(WRAP)
    r = rr.reparar_estructura(m)
    assert r["aplicado"] == ["wrap_desproporcionado"] and m["ingredients"] == WRAP["ingredients"]
    nuevo = m["recipe"][-1]
    assert nuevo.startswith("Montaje: rellena la tortilla de trigo con lo que cierra (unos 150 g del relleno)")
    assert "sirve el resto del relleno (~180 g) al lado" in nuevo
    assert ds.relaciones(m) == [] and rr.reparar_estructura(m)["aplicado"] == []


def test_la_crema_usa_solo_el_liquido_que_espesa_y_el_resto_es_bebida():
    m = copy.deepcopy(CREMA)
    r = rr.reparar_estructura(m)
    assert r["aplicado"] == ["crema_sin_espesante"] and m["ingredients"] == CREMA["ingredients"]
    nuevo = m["recipe"][-1]
    assert "usa solo 30 ml de leche descremada en la crema" in nuevo and "270 ml de leche descremada restantes como bebida al lado" in nuevo
    assert ds.relaciones(m) == [] and rr.reparar_estructura(m)["aplicado"] == []
    # V7d no ve un sobrante: 30 + 270 = 300 ml en los pasos frente a 300 ml en la lista
    assert not cc._v7d_masa_sobrante(1, m, _IDX)


def test_lo_que_no_se_puede_reparar_se_descarta_y_se_dice(monkeypatch):
    poca = _meal("Crema de lentejas", ["60 g de lentejas", "160 ml de leche descremada"], ["Licúa hasta obtener una crema espesa."])
    assert ds.relaciones(poca) == [], "con 0,38 g/ml no se acusa nada"
    apenas = _meal("Crema de lentejas", ["20 g de lentejas", "170 ml de leche descremada"], ["Licúa hasta obtener una crema espesa."])
    assert [x["tipo"] for x in ds.relaciones(apenas)] == ["crema_sin_espesante"]
    r = rr.reparar_estructura(copy.deepcopy(apenas))
    assert r["aplicado"] == ["crema_sin_espesante"], "una crema acusada siempre deja ≥ 90 ml que servir aparte: se repara"
    # el camino «no se puede»: un reparador que no encuentra con qué (aquí, simulado) se declara, no se calla
    monkeypatch.setitem(rr._REPARADORES, "crema_sin_espesante", lambda meal, comp: None)
    r2 = rr.reparar_estructura(copy.deepcopy(apenas))
    assert r2["aplicado"] == [] and r2["descartado"] == ["crema_sin_espesante"]
    assert rr.reparar_estructura({"name": "x"}) == {"aplicado": [], "descartado": [], "cambios": []}


# ─────────────── el paso (4) del contrato final ───────────────

def test_el_contrato_final_repara_la_estructura_y_lo_anota_solo_si_reparo(monkeypatch):
    monkeypatch.setenv("MEALFIT_RECIPE_FINAL_CONTRACT", "repair")
    m = copy.deepcopy(WRAP)
    r = rc.reconcile_meal(m, _IDX)
    assert r["estructura"] == 1 and r["cambios_estructura"][0]["tipo"] == "wrap_desproporcionado"
    m2 = copy.deepcopy(WRAP)
    rc._aplicar_meal(m2, _IDX, "repair", None)
    assert m2[rc.TELEMETRIA_KEY]["estructura"] == 1 and m2["ingredients"] == WRAP["ingredients"]
    limpio = _meal("Pollo a la plancha", ["150 g de pechuga de pollo"], ["Asa el pollo 8 min por lado."])
    rc._aplicar_meal(limpio, _IDX, "repair", None)
    assert rc.TELEMETRIA_KEY not in limpio, "sin nada que reparar no se anota nada"


def test_en_shadow_se_anota_lo_que_habria_hecho_y_el_plato_no_cambia():
    m = copy.deepcopy(CREMA)
    rc._aplicar_meal(m, _IDX, "shadow", None)
    assert m["recipe"] == CREMA["recipe"] and m[rc.TELEMETRIA_KEY]["modo"] == "shadow" and m[rc.TELEMETRIA_KEY]["estructura"] == 1


def test_el_paso_4_va_detras_de_cantidades_y_formas():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    body = src.split("def reconcile_meal(")[1].split("\ndef ")[0]
    assert body.index("reconcile_step_quantities(meal, index)") < body.index("_reparar_estructura(meal)")
    assert 'if r.get("estructura"):' in src and "P1-PLAN-LOTE-29" in src


# ─────────────── el camino sin LLM ───────────────

def test_degradar_paso_conserva_los_otros_alimentos_y_cuece_lo_seco():
    assert rr.degradar_paso("Hornea el Casabe.", "Casabe", _IDX, "El Toque de Fuego: ") == ["El Toque de Fuego: Sirve el Casabe."]
    pasos = rr.degradar_paso("Sofríe la cebolla, el ajo y las habichuelas con el casabe.", "Casabe", _IDX, "")
    assert pasos[-1] == "Sirve el Casabe con Cebolla, Ajo y Habichuelas rojas."
    assert pasos[0].startswith("Cocina Habichuelas rojas según su envase") and len(pasos) == 2
    solo = rr.degradar_paso("Hornea el casabe con la cebolla.", "Casabe", _IDX, "Montaje: ")
    assert solo == ["Montaje: Sirve el Casabe con Cebolla."]


def test_el_degradador_del_cron_usa_al_reparador_y_no_deja_huerfanos():
    import cron_tasks as ct
    day = {"meals": [{"meal": "Cena", "ingredients": ["40 g de casabe", "50 g de cebolla", "100 g de habichuelas rojas"],
                      "recipe": ["El Toque de Fuego: Hornea el casabe con la cebolla y las habichuelas."]}]}
    n = ct._degrade_offending_steps(day, [{"check": "V1", "food": "Casabe", "meal": "Cena"}], _IDX)
    rec = day["meals"][0]["recipe"]
    assert n == 1 and rec[-1] == "El Toque de Fuego: Sirve el Casabe con Cebolla y Habichuelas rojas."
    assert rec[0].startswith("El Toque de Fuego: Cocina Habichuelas rojas")
    v3 = [v for v in cc.culinary_contract_scan({"days": [day]}, _CAT) if v["check"] in ("V3", "V7c")]
    assert not v3, v3
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert "from recipe_repair import degradar_paso as _dp" in src


def test_el_bench_degrada_solo_por_v1_v2_como_el_cron():
    src = (_BACKEND / "scripts" / "bench_superficies_culinarias.py").read_text(encoding="utf-8")
    assert 'if v.get("check") in ("V1", "V2")]' in src.split("def _ad_degradado")[1].split("\ndef ")[0]


# ─────────────── docs, plan, marcador ───────────────

def test_docs_plan_y_marcador():
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-29" in doc and "recipe_repair" in doc and "no 21" in doc
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-29" in plan
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 29 and m.group(2) >= "2026-09-12"
    assert "P1-PLAN-LOTE-29-RECIPE-REPAIR" in (_BACKEND / "recipe_repair.py").read_text(encoding="utf-8")
    assert len((_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8").splitlines()) <= 53100
