# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-16 · 2026-09-12] Decimosexto lote del plan de pendientes: A9, la curación del DUEÑO aplicada.

Los 19 platos de P1-CATALOGO-PROTEINA-DESAYUNO (DO, pendientes desde el 09-09) y los 9 almuerzos de despensa de E9 (ES,
pendientes desde el 09-12) los juzgó el dueño plato a plato en la ficha interactiva A9: 28/28 «con cambios». Aplicados:

  · 2 renombres con id estable (`TEMPLATE_ALIASES`): «Huevo duro con aguacate y sal mínima» → «Huevo duro con aguacate»;
    «Macarrones con salsa de tomate y queso curado» → «Coditos con salsa de tomate y queso gouda».
  · 16 técnicas corregidas con id estable: la técnica entra en `mint_template_id` igual que el nombre, así que hace
    falta el mismo tipo de puente (`TEMPLATE_MINT_TECHNIQUE`: nombre actual → técnica con la que se acuñó el id).
  · `spec` por constituyente (estado en que se pesa / cómo se usa) y `prep_notes` por plantilla ES, que el compilador
    ahora deja pasar al snapshot: el `name` sigue siendo la fila del catálogo y la nutrición NO se mueve.
  · Las 19 recetas DO reescritas al pie de la letra (recipe_library v8).
  · Lo que NO se aplicó tal cual, dicho: «aguanta N d»/«solo despensa» eran la etiqueta de la ficha para la vida de los
    INGREDIENTES crudos (`logistics.days_fresh_min`/`pantry_only`), no del plato hecho — 28 de 28 veredictos lo
    señalaron porque la ficha lo etiquetaba mal. *Si un número interno se le enseña a alguien, la etiqueta es parte
    del dato.*

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

DO_IDS = {
    "tpl_6a5089265418": "Revoltillo de claras con espinaca y casabe",
    "tpl_36409c4726eb": "Mangú de plátano verde con atún y cebolla encurtida",
    "tpl_b991edb7ec87": "Avena cocida con claras de huevo y maní",
    "tpl_7f0b6fc54350": "Yogurt griego con avena tostada y maní",
    "tpl_4e48570063d2": "Sardinas guisadas con casabe y tomate",
    "tpl_86da7bdca661": "Tortilla de yuca con atún y queso blanco",
    "tpl_cc64fe57c76f": "Batida de leche con claras, avena y guineo",
    "tpl_e1265f14a8b4": "Queso cottage con casabe, tomate y aguacate",
    "tpl_8b37ecfd46e1": "Tilapia al horno con yuca y cebolla",
    "tpl_b83a2e9e039a": "Huevo duro con aguacate",
    "tpl_af7f6c3725a6": "Atún en agua con casabe y cebolla",
    "tpl_b29321fc8917": "Yogurt griego con guineo",
    "tpl_3a0517b6dc80": "Queso cottage con lechosa",
    "tpl_612f9a6638b2": "Batida de leche con claras y avena",
    "tpl_6a5fe80136a7": "Sardinas en lata con casabe",
    "tpl_7ec10278eed6": "Queso cottage con casabe y tomate",
    "tpl_53d21a09b17f": "Pechuga desmenuzada con casabe y cebolla",
    "tpl_385a17ed7bed": "Tofu salteado con salsa de soya y cebolla",
    "tpl_02895f974ed3": "Queso de hoja con tomate y cebolla",
}
ES_IDS = {
    "tpl_250f13abcd15": "Patatas guisadas con huevo escalfado y pimentón",
    "tpl_28e7588a586c": "Arroz con garbanzos, zanahoria y pimentón",
    "tpl_50cf720cd524": "Coditos con salsa de tomate y queso gouda",
    "tpl_65b8503d7106": "Ensalada de garbanzos con huevo duro, aceitunas y cebolla",
    "tpl_6e7dacc8a2ef": "Garbanzos guisados con pimentón y huevo duro",
    "tpl_9a2990dc877a": "Judías blancas estofadas con patata, laurel y pimentón",
    "tpl_a1e6ad600e91": "Trinxat de col y patata con huevo",
    "tpl_c7303228987a": "Lentejas estofadas con patata y zanahoria",
    "tpl_fa89b5d4bec3": "Arroz a la cubana con huevo y salsa de tomate",
}
RENOMBRES = {
    "Huevo duro con aguacate": "Huevo duro con aguacate y sal mínima",
    "Coditos con salsa de tomate y queso gouda": "Macarrones con salsa de tomate y queso curado",
}


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _json(rel: str):
    return json.loads(_src(rel))


@pytest.fixture(scope="module")
def reg_do():
    return _json("data/registry/dish_registry_do_v1.json")


@pytest.fixture(scope="module")
def reg_es():
    return _json("data/registry/dish_registry_es_v1.json")


@pytest.fixture(scope="module")
def lib():
    return _json("data/registry/recipe_library_do_v1.json")


def _by_id(reg):
    return {t["template_id"]: t for t in reg["templates"]}


# ─────────────────────────── ids estables: renombres y técnicas ───────────────────────────

def test_los_28_conservan_su_template_id_y_compilan_integros(reg_do, reg_es):
    for reg, ids in ((reg_do, DO_IDS), (reg_es, ES_IDS)):
        by = _by_id(reg)
        for tid, name in ids.items():
            assert tid in by, (tid, name, "el id histórico desapareció del snapshot")
            t = by[tid]
            assert t["name"] == name, (tid, t["name"])
            assert t["status"] == "ok" and t["excluded"] == [], (name, t["excluded"])
    assert len(reg_do["templates"]) == 193 and len(reg_es["templates"]) == 133


def test_el_renombre_y_la_tecnica_nueva_acunan_el_mismo_id_que_antes():
    import plan_policy
    fuentes = {t["name"]: t for t in _json("data/dish_templates.json")["templates"]}
    fuentes_es = {t["name"]: t for t in _json("data/dish_templates_es.json")["templates"]}
    for nuevo, viejo in RENOMBRES.items():
        assert plan_policy.TEMPLATE_ALIASES.get(nuevo) == viejo, nuevo
    for tid, name in DO_IDS.items():
        assert plan_policy.mint_template_id(fuentes[name], "do") == tid, (name, "el renombre o la técnica movió el id")
    for tid, name in ES_IDS.items():
        assert plan_policy.mint_template_id(fuentes_es[name], "es") == tid, (name, "el renombre o la técnica movió el id")


def test_el_puente_de_tecnica_esta_anclado_y_no_tiene_llaves_huerfanas():
    import plan_policy
    src = _src("plan_policy.py")
    assert "tooltip-anchor: TEMPLATE_MINT_TECHNIQUE (test_p1_plan_lote_16.py)" in src
    assert "minted_technique = TEMPLATE_MINT_TECHNIQUE.get(name, template.get(\"technique\"))" in src
    nombres = {t["name"]: t for t in _json("data/dish_templates.json")["templates"]}
    nombres.update({t["name"]: t for t in _json("data/dish_templates_es.json")["templates"]})
    assert len(plan_policy.TEMPLATE_MINT_TECHNIQUE) == 16
    for name, minted in plan_policy.TEMPLATE_MINT_TECHNIQUE.items():
        assert name in nombres, (name, "puente a un plato que no existe")
        assert nombres[name]["technique"] != minted, (name, "el puente sobra: la técnica actual es la acuñada")


def test_las_tecnicas_dicen_lo_que_el_plato_hace(reg_do, reg_es):
    by = _by_id(reg_do)
    assert by["tpl_7f0b6fc54350"]["technique"] == "tostado + montaje en frío"
    assert by["tpl_cc64fe57c76f"]["technique"] == "cocido + batido"
    assert by["tpl_8b37ecfd46e1"]["technique"] == "hervido + horneado"
    assert by["tpl_53d21a09b17f"]["technique"] == "hervido + tostado"
    assert by["tpl_b29321fc8917"]["technique"] == "montaje en frío"
    assert not any(t["technique"] == "crudo" for tid, t in by.items() if tid in DO_IDS), "ningún plato con casabe tostado o claras cocidas es «crudo»"
    bes = _by_id(reg_es)
    assert bes["tpl_50cf720cd524"]["technique"] == "hervido + sofrito"
    assert bes["tpl_a1e6ad600e91"]["technique"] == "hervido + majado + salteado"
    assert bes["tpl_fa89b5d4bec3"]["technique"] == "hervido + sartén"


# ─────────────────────────── spec y prep_notes atraviesan el compilador ───────────────────────────

def test_inline_constituent_conserva_spec_y_optional():
    import dish_registry as dr
    c = dr._inline_constituent({"name": "Avena", "grams": 55, "spec": " en hojuelas ", "optional": True, "otra": 1})
    assert c == {"name": "Avena", "grams": 55.0, "spec": "en hojuelas", "optional": True}
    assert dr._inline_constituent({"name": "Avena", "g": 55}) == {"name": "Avena", "grams": 55.0}


def test_el_spec_llega_al_snapshot_sin_mover_nombre_ni_nutricion(reg_do, reg_es):
    by = _by_id(reg_do)
    avena = {c["name"]: c for c in by["tpl_b991edb7ec87"]["constituents"]}
    assert avena["Avena"]["canonical"] == "Avena" and "hojuelas" in avena["Avena"]["spec"]
    assert "tostado" in avena["Maní"]["spec"] and "sin sal" in avena["Maní"]["spec"]
    sard = {c["name"]: c for c in by["tpl_4e48570063d2"]["constituents"]}["Sardinas en lata"]
    assert "escurridas" in sard["spec"] and sard["canonical"] == "Sardinas en lata"
    huevo = {c["name"]: c for c in by["tpl_b83a2e9e039a"]["constituents"]}["Huevo"]
    assert "sin cáscara" in huevo["spec"] and huevo["grams"] == 100.0
    bes = _by_id(reg_es)
    lent = {c["name"]: c for c in bes["tpl_c7303228987a"]["constituents"]}
    assert lent["Comino"]["spec"] == "molido" and lent["Pimentón"]["spec"] == "en polvo" and "secas" in lent["Lentejas"]["spec"]
    assert "se retira" in lent["Laurel"]["spec"]
    # el dueño pidió «recalcular con el catálogo»: ya era el cálculo — ninguna fila cambió, ninguna cifra se mueve
    assert by["tpl_6a5089265418"]["nutrition_per_serving"]["kcal"] == 343.25
    assert by["tpl_385a17ed7bed"]["nutrition_per_serving"]["sodium_mg"] == 252.0, "8 g de la fila «Salsa de soya» (2 890 mg/100 g) + tofu + cebolla"
    assert by["tpl_385a17ed7bed"]["nutrition_unknown"] == {}


def test_las_prep_notes_de_las_nueve_es_viajan_en_editorial(reg_es):
    bes = _by_id(reg_es)
    for tid, name in ES_IDS.items():
        notes = (bes[tid].get("editorial") or {}).get("prep_notes")
        assert notes and len(notes) > 40, (name, "sin prep_notes")
    assert "remojo" in bes["tpl_28e7588a586c"]["editorial"]["prep_notes"].lower()
    assert "4 g" in bes["tpl_fa89b5d4bec3"]["editorial"]["prep_notes"] and "6 g" in bes["tpl_fa89b5d4bec3"]["editorial"]["prep_notes"], "los 10 g de aceite, repartidos explícitamente"
    assert "batido" in bes["tpl_a1e6ad600e91"]["editorial"]["prep_notes"].lower(), "cómo entra el huevo en el trinxat"


def test_las_plantillas_do_no_llevan_prep_notes_porque_tienen_receta(reg_do):
    assert not any((t.get("editorial") or {}).get("prep_notes") for t in reg_do["templates"])


# ─────────────────────────── las 19 recetas, al pie de la letra ───────────────────────────

def test_las_19_recetas_v8_aplican_los_veredictos(lib):
    por = lib["por_id"]
    P = {tid: por[tid]["pasos"] for tid in DO_IDS}
    for tid in DO_IDS:
        assert P[tid] and all(isinstance(s, str) and s for s in P[tid]), tid
        assert not any("aguanta" in s for s in P[tid]), (tid, "la conservación no es un paso de la receta")
    # casabe: sartén seco a fuego medio, sin quemarse, sin la alternativa del horno
    for tid in ("tpl_6a5089265418", "tpl_4e48570063d2", "tpl_e1265f14a8b4", "tpl_af7f6c3725a6", "tpl_6a5fe80136a7", "tpl_7ec10278eed6", "tpl_53d21a09b17f"):
        j = " ".join(P[tid])
        assert "sartén seco a fuego medio" in j and "queme" in j, tid
        assert "en el horno" not in j, tid
    assert "también se pesa sin cáscara" in P["tpl_6a5089265418"][0]
    # mangú: cebolla en el vinagre SIN agua; el aceite repartido mitad y resto; sal en el puré sin la afirmación del hervor
    assert "sin agua" in P["tpl_36409c4726eb"][0] and "removiendo" in P["tpl_36409c4726eb"][0]
    assert "MITAD del aceite" in P["tpl_36409c4726eb"][2] and "sal de la ficha" in P["tpl_36409c4726eb"][2] and "fregadero" not in P["tpl_36409c4726eb"][2]
    assert "resto del aceite" in P["tpl_36409c4726eb"][4]
    # la regla de la biblioteca sigue en pie: los GRAMOS que dictó el dueño viven en la ficha (constituents.grams + spec),
    # el paso dice el ESTADO del peso — sin cifras la misma receta sirve para cualquier porción (R3 lo ancló primero).
    cifra = re.compile(r"\d+\s*(ml|g\b|gramos|cucharada)", re.I)
    assert not [s for tid in DO_IDS for s in P[tid] if cifra.search(s)], "volvió una cantidad de ingrediente a un paso"
    # maní: tostado sin sal, sin la alternativa de tostar crudo
    for tid in ("tpl_b991edb7ec87", "tpl_7f0b6fc54350"):
        j = " ".join(P[tid])
        assert "maní tostado sin sal" in j and "si está crudo" not in j, tid
    assert "enfriar del todo" in P["tpl_7f0b6fc54350"][2]
    # tortilla de yuca: fibra central, entibiar, 2 cm, centro cuajado
    j6 = " ".join(P["tpl_86da7bdca661"])
    assert "fibra central" in j6 and "entibiar" in j6 and "2 cm" in j6 and "completamente cuajado" in j6 and "cruda y ya pelada" in j6
    # batidas: claras pesadas crudas, cocidas hasta cuajar (minutos orientativos), colador fino
    for tid in ("tpl_cc64fe57c76f", "tpl_612f9a6638b2"):
        j = " ".join(P[tid])
        assert "claras crudas" in j and "orientativos" in j and "colador fino" in j and "sin cocción" in j, tid
    # cottage: mezclar y pesar sin escurrir, sin la afirmación sobre los macros al escurrir
    for tid in ("tpl_e1265f14a8b4", "tpl_3a0517b6dc80", "tpl_7ec10278eed6"):
        j = " ".join(P[tid])
        assert "sin escurrir" in j and "macros" not in j, tid
    assert "Lava el tomate" in " ".join(P["tpl_e1265f14a8b4"]) and "Lava el tomate" in " ".join(P["tpl_7ec10278eed6"])
    assert "Lava la lechosa" in P["tpl_3a0517b6dc80"][0]
    # tilapia y pechuga conservan sus temperaturas; la yuca desecha el agua y pierde la fibra central
    assert "63 °C" in " ".join(P["tpl_8b37ecfd46e1"]) and "fibra central" in P["tpl_8b37ecfd46e1"][3] and "fregadero" not in " ".join(P["tpl_8b37ecfd46e1"])
    assert "74 °C" in P["tpl_53d21a09b17f"][0] and "sin piel ni hueso" in P["tpl_53d21a09b17f"][0] and "Opcional" in P["tpl_53d21a09b17f"][2]
    # huevo duro: 10-12 minutos hasta firmes, sin «exactos»; la sal se añade
    j10 = " ".join(P["tpl_b83a2e9e039a"])
    assert "10-12 minutos" in j10 and "exactos" not in j10 and "firmes" in j10 and "Añade la sal" in j10
    # atún: limón opcional; yogurt: bolsa térmica; tofu: escurrido antes de prensar, cubos de 2 cm, antiadherente
    assert "Opcional" in P["tpl_af7f6c3725a6"][1]
    assert "bolsa térmica" in P["tpl_b29321fc8917"][1]
    j18 = " ".join(P["tpl_385a17ed7bed"])
    assert "antes de prensarlo" in j18 and "2 cm" in j18 and "antiadherente" in j18
    assert "Lava el tomate y pela la cebolla" in P["tpl_02895f974ed3"][0] and "Opcional" in P["tpl_02895f974ed3"][2]


def test_la_procedencia_cuenta_la_curacion_a9_sin_inflarla(lib):
    p = lib["procedencia"]
    assert "v8 2026-09-12" in lib["revision"]
    assert "A9" in p["veredicto_humano"] and "MUESTRA" in p["veredicto_humano"], "22 con juicio individual; el resto sigue siendo muestra"
    a9 = p.get("curacion_humana_a9") or ""
    assert "INGREDIENTES" in a9 and "no el dato" in a9, "lo no aplicado tal cual se declara con su razón"
    assert "sin_cantidades" in a9, "los gramos dictados viven en la ficha, no en el paso — y se dice por qué"
    assert "252 mg" in a9 and "2 890" in a9


# ─────────────────────────── SSOT de constituyentes, firma, benchmark, baseline ───────────────────────────

def test_la_tabla_curada_sigue_siendo_ssot_y_valida():
    import importlib.util
    p = _BACKEND / "scripts" / "check_dish_constituents_do.py"
    spec = importlib.util.spec_from_file_location("_lote16_check", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.check()
    assert r["ok"] is True and r["fallos"] == [], r["fallos"]
    tabla = _json("data/dish_constituents_do.json")["templates"]
    assert "Huevo duro con aguacate" in tabla and "Huevo duro con aguacate y sal mínima" not in tabla
    specs = [c for c in tabla["Tofu salteado con salsa de soya y cebolla"]["constituents"] if c.get("spec")]
    assert len(specs) == 2


def test_la_firma_curatorial_y_el_benchmark_apuntan_al_snapshot_recompilado(reg_do, reg_es):
    rev = _json("data/registry/cultural_curation_review_v1.json")
    bench = _json("data/registry/cultural_benchmark_v1.json")
    for pid, reg in (("dominican_criolla", reg_do), ("spain_mediterranea", reg_es)):
        prof = rev["profiles"][pid]
        assert prof["snapshot_hash"] == reg["snapshot_hash"], (pid, "firma caducada: recompilaste sin re-anclar")
        assert "pendiente_de_juicio_humano" not in prof, (pid, "A9 ya se juzgó")
        a9 = prof["juicio_humano_a9"]
        assert a9["fecha"] == "2026-09-12" and "no el dato" in a9["no_aplicado_tal_cual"]
        assert any("P1-PLAN-LOTE-16" in d for d in prof["decisions"])
        assert bench["profiles"][pid]["snapshot_hash"] == reg["snapshot_hash"], (pid, "informe desfasado — `python cultural_benchmark.py --write`")
    assert len(rev["profiles"]["dominican_criolla"]["juicio_humano_a9"]["platos"]) == 19
    assert len(rev["profiles"]["spain_mediterranea"]["juicio_humano_a9"]["platos"]) == 9
    assert "reanchor_note_2026_09_12b" in rev and "INGREDIENTES" in rev["human_review_notes_a9"]


def test_el_baseline_c3_y_los_tests_vecinos_hablan_del_nombre_nuevo():
    m = _json("scripts/data/do_corpus_retarget_baseline_2026_08_18.json")["mapping"]
    assert "Huevo duro con aguacate" in m and "Huevo duro con aguacate y sal mínima" not in m
    assert '"Huevo duro con aguacate",' in _src("tests/test_p1_catalogo_proteina_desayuno.py")
    assert "Coditos con salsa de tomate y queso gouda" in _src("tests/test_p1_plan_lote_13.py")


# ─────────────────────────── docs y marker ───────────────────────────

def test_los_docs_cuentan_el_lote():
    plan = _src("docs/plan_pendientes_2026_09_11.md")
    assert re.search(r"^\| A9 \| ✅ 2026-09-12", plan, re.M), "A9 cerrado en el Estado del plan"
    f6 = _src("docs/dish_registry_f6.md")
    for frag in ("`spec`", "prep_notes", "TEMPLATE_MINT_TECHNIQUE", "INGREDIENTES"):
        assert frag in f6, frag


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-16 · 2026-09-12]" in _src("app.py")
    assert app._LAST_KNOWN_PFIX.split("·")[-1].strip() >= "2026-09-12", app._LAST_KNOWN_PFIX
