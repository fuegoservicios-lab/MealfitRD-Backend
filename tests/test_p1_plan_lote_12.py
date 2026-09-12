# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-12 · 2026-09-11] C8-b: el rótulo «Jugo de limón» que pidió el dueño.

Al curar C8 el dueño pidió que «Frutas picadas con limón» leyera «Jugo de limón 10 g» y no «Limón 10 g». El catálogo
tiene «jugo de limón» como ALIAS de la fila «Limón», así que el registry compilaba `ok`… pero el día determinista
construía su catálogo como `{name: fila}` a secas y `catalogo.get("Jugo de limón")` daba `None`: `construir_comida`
descartaba el plato. Una plantilla muerta que compilaba verde — un alias que resuelve en el compilador y no en el
consumidor.

El cierre no es un segundo resolutor (la lección de P1-DIET-CANON-SSOT): `deterministic_day._CatalogoPorNombre`
responde por nombre canónico y, si no, con EL MISMO resolutor que el compilador del registry
(`dish_registry.build_catalog_index` + `resolve_constituent`). El rótulo que ve el usuario es el `name` del
constituyente; las macros salen de la fila canónica.

Y lo que NO hizo falta, medido antes de añadirlo: la lista de compras y la Nevera ya resolvían el alias solas —
`shopping_calculator._parse_quantity` devuelve `normalize_name(name_raw)` («Limón») y aplica P1-CITRUS-JUICE-YIELD
(×2,86: 10 g de jugo son ~29 g de limón entero). `pantry_names_match("Jugo de limón", "Limón")` da `False` a
propósito, pero la línea nunca llega así al matcher. *Una función aislada que dice «no casa» no prueba que el camino
no case.*

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _rows():
    return [
        {"name": "Limón", "aliases": ["limones", "jugo de limón"], "kcal_per_100g": 29.0, "protein_g_per_100g": 1.1,
         "carbs_g_per_100g": 9.3, "fats_g_per_100g": 0.3},
        {"name": "Guineo", "aliases": ["banana"], "kcal_per_100g": 89.0, "protein_g_per_100g": 1.1,
         "carbs_g_per_100g": 22.8, "fats_g_per_100g": 0.3},
        # un alias que colisiona con el nombre canónico de OTRA fila: el canónico gana
        {"name": "Pavo", "aliases": [], "kcal_per_100g": 135.0, "protein_g_per_100g": 29.0,
         "carbs_g_per_100g": 0.0, "fats_g_per_100g": 1.7},
        {"name": "Jamón de pavo", "aliases": ["pavo", "jamón pavo"], "kcal_per_100g": 104.0, "protein_g_per_100g": 17.0,
         "carbs_g_per_100g": 2.0, "fats_g_per_100g": 3.0},
    ]


_PASOS = ["Pela y corta la fruta en cuadritos de bocado.",
          "Agrega el jugo de limón y mezcla con suavidad.",
          "Sírvelo frío, recién hecho, en un bol."]


# ─────────────────────────── 1. el catálogo del día responde por alias con el resolutor del compilador ───────────────────────────

def test_el_catalogo_del_dia_resuelve_un_alias_como_lo_hace_el_compilador():
    import deterministic_day as dd
    cat = dd._CatalogoPorNombre(_rows())
    assert cat.get("Jugo de limón")["name"] == "Limón"
    assert cat["Jugo de limón"]["name"] == "Limón"
    assert "Jugo de limón" in cat
    assert cat.get("jugo de LIMON")["name"] == "Limón", "misma normalización que el compilador (acentos, mayúsculas)"
    assert cat.get("Limones")["name"] == "Limón", "plural simple, como resolve_constituent"


def test_lo_que_no_esta_en_el_catalogo_sigue_sin_estar():
    import deterministic_day as dd
    cat = dd._CatalogoPorNombre(_rows())
    assert cat.get("Zapote") is None and cat.get("Zapote", {}) == {}
    assert "Zapote" not in cat
    with pytest.raises(KeyError):
        cat["Zapote"]


def test_el_nombre_canonico_gana_al_alias_y_el_resolutor_no_anade_filas():
    import deterministic_day as dd
    cat = dd._CatalogoPorNombre(_rows())
    assert cat.get("Pavo")["name"] == "Pavo", "el alias «pavo» de Jamón de pavo no pisa la fila Pavo"
    assert cat.get("Limón")["name"] == "Limón"
    assert len(cat) == 4 and {r["name"] for r in cat.values()} == {"Limón", "Guineo", "Pavo", "Jamón de pavo"}
    assert set(cat) == {"Limón", "Guineo", "Pavo", "Jamón de pavo"}, "iterar da las filas canónicas, no los alias"


def test_construir_comida_sirve_el_rotulo_del_registry_con_las_macros_de_la_fila(monkeypatch):
    """El usuario lee «10 g de Jugo de limón»; kcal y macros salen de la fila «Limón»."""
    import deterministic_day as dd
    import recipe_library as rl
    monkeypatch.setattr(rl, "recipe_for_dish_name", lambda name, country="DO": list(_PASOS))
    t = {"template_id": "tpl_prueba", "name": "Frutas de prueba con limón", "slots": ["merienda"],
         "constituents": [{"name": "Guineo", "grams": 80.0, "canonical": "Guineo"},
                          {"name": "Jugo de limón", "grams": 10.0, "canonical": "Limón"}]}
    m = dd.construir_comida(t, 1.0, dd._CatalogoPorNombre(_rows()), "merienda", "DO")
    assert m is not None, "el plato con un constituyente por alias tiene que poder servirse"
    assert m["ingredients"] == ["80 g de Guineo", "10 g de Jugo de limón"]
    assert m["calories"] == round(80 * 0.89 + 10 * 0.29)
    assert m["_recipe_source"] == "library" and m["_meal_source"] == "deterministic"


def test_build_day_construye_su_catalogo_con_el_resolutor():
    src = _src("deterministic_day.py")
    assert "catalogo = _CatalogoPorNombre(get_master_ingredients() or [])" in src
    assert "tooltip-anchor: _CatalogoPorNombre (test_p1_plan_lote_12.py)" in src
    assert 'catalogo = {str(r.get("name")): r for r in (get_master_ingredients() or [])}' not in src, (
        "vuelve el catálogo por nombre exacto: los constituyentes por alias mueren en silencio")


# ─────────────────────────── 2. la lista y la Nevera ya resolvían el alias: no se añadió maquinaria ───────────────────────────

def test_la_nevera_rechaza_a_proposito_el_alias_con_calificativo_en_el_matcher():
    """Verdad medida y NO problema: la línea llega al matcher ya canonicalizada (ver los dos tests de abajo)."""
    from constants import pantry_names_match
    assert pantry_names_match("Jugo de limón", "Limón", use_catalog_aliases=False) is False


def test_el_parser_de_lineas_canonicaliza_y_aplica_el_rendimiento_del_jugo():
    """`_parse_quantity` es el camino real de «me lo comí» y de la lista: nombre por `normalize_name`, cantidad ×1/0,35."""
    import shopping_calculator as sc
    assert sc._calculate_yield_multiplier("Jugo de limón") == pytest.approx(2.8571, abs=1e-3)
    assert sc._calculate_yield_multiplier("Limón") == 1.0
    src = _src("shopping_calculator.py")
    i = src.find("def _parse_quantity(")
    assert "return qty, unit, normalize_name(name_raw).strip()" in src[i:i + 6000], "el parser devuelve el nombre CANÓNICO"


def test_normalize_name_resuelve_el_alias_a_limon_segun_el_baseline_committed():
    """El baseline C3 es la verdad committed de `normalize_name` contra el catálogo vivo (test_p1_country_system_f2)."""
    m = json.loads(_src("scripts/data/do_corpus_retarget_baseline_2026_08_18.json"))["mapping"]
    assert m.get("jugo de limón") == "Limón" and m.get("limones") == "Limón"


def test_no_se_anadio_un_peldano_para_el_alias():
    """Lo medido antes de cerrar: producción ya resolvía. Un peldaño «2b» o un mapa en el plato serían maquinaria inerte."""
    assert "_ingredient_canonical" not in _src("deterministic_day.py")
    assert "canonical_hints" not in _src("db_inventory.py") and "canonical_hints" not in _src("routers/diary.py")
    assert "P1-PLAN-LOTE-12" in _src("docs/pantry_name_resolution.md") and "Mide la capa que corre" in _src("docs/pantry_name_resolution.md")


# ─────────────────────────── 3. el dato: el registry dice lo que pidió el dueño ───────────────────────────

def test_el_registry_dice_jugo_de_limon_y_resuelve_a_limon():
    reg = json.loads(_src("data/registry/dish_registry_do_v1.json"))
    t = {x["template_id"]: x for x in reg["templates"]}["tpl_0ae55a98c3e3"]
    jl = [c for c in t["constituents"] if c["name"] == "Jugo de limón"]
    assert jl and jl[0]["canonical"] == "Limón" and jl[0]["grams"] == 10.0 and jl[0]["ingredient_id"] == "limon"
    assert t["status"] == "ok" and t["excluded"] == []
    assert "Limón" not in [c["name"] for c in t["constituents"]], "el rótulo viejo no convive con el nuevo"
    tabla = json.loads(_src("data/dish_constituents_do.json"))["templates"]["Frutas picadas con limón"]
    assert {"grams": 10.0, "name": "Jugo de limón"} in tabla["constituents"]


def test_la_receta_habla_del_jugo_y_la_procedencia_ya_no_dice_no_aplicado():
    lib = json.loads(_src("data/registry/recipe_library_do_v1.json"))
    pasos = lib["por_id"]["tpl_0ae55a98c3e3"]["pasos"]
    assert "agrega el jugo de limón" in pasos[3] and "suavidad" in pasos[3]
    p = lib["procedencia"]
    assert "P1-PLAN-LOTE-12" in p["curacion_humana_c8"] and "no aplicado" not in p["curacion_humana_c8"].lower()
    assert "P1-PLAN-LOTE-12" in p["veredicto_humano"] and "salvo uno" not in p["veredicto_humano"]
    assert "v7b 2026-09-11" in lib["revision"]


def test_la_firma_curatorial_sigue_al_snapshot():
    reg = json.loads(_src("data/registry/dish_registry_do_v1.json"))
    rev = json.loads(_src("data/registry/cultural_curation_review_v1.json"))
    assert rev["profiles"]["dominican_criolla"]["snapshot_hash"] == reg["snapshot_hash"]
    assert "reanchor_note_2026_09_11b" in rev
    bench = json.loads(_src("data/registry/cultural_benchmark_v1.json"))
    assert bench["profiles"]["dominican_criolla"]["snapshot_hash"] == reg["snapshot_hash"]


# ─────────────────────────── docs y marker ───────────────────────────

def test_los_docs_cuentan_la_segunda_decision():
    dd = _src("docs/deterministic_day.md")
    assert "P1-PLAN-LOTE-12" in dd and "MISMO resolutor" in dd and "_CatalogoPorNombre" in dd
    assert "P1-CITRUS-JUICE-YIELD" in dd, "el doc dice por qué la Nevera no necesitó nada"
    assert "P1-PLAN-LOTE-12" in _src("docs/plan_pendientes_2026_09_11.md")


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-12 · 2026-09-11]" in _src("app.py")
    # [P1-PLAN-LOTE-13 · 2026-09-12] «no anterior a este lote», no «igual a hoy»: el pin de la fecha y del prefijo
    # `P1-PLAN-` rompía 12 tests el primer día en que otro P-fix bumpeaba el marker.
    assert app._LAST_KNOWN_PFIX.split("·")[-1].strip() >= "2026-09-11", app._LAST_KNOWN_PFIX
