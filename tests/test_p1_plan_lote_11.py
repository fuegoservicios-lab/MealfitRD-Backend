# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-11 · 2026-09-11] Undécimo lote del plan de pendientes: C8, la curación del DUEÑO.

Las 4 plantillas DO que compilaban `partial` desde ARQ27-P0-02 (Menta, Salami de pavo, Chillo, Zapote) no tenían
receta congelada y el día determinista no podía servirlas. El dueño las juzgó plato a plato en una ficha interactiva:

  · Frutas picadas con limón y menta → ✏️ «Frutas picadas con limón»: la menta no se compra y queda opcional en el
    último paso; el guineo se pesa sin cáscara; el limón se exprime y se mezcla suave.
  · Mangú con salami de pavo a la plancha (versión magra) → ✏️ «Mangú con jamón de pavo a la plancha»: el
    constituyente siempre fue jamón; «versión magra» cae porque no se especifica la grasa; la mitad del aceite a la
    cebolla y el resto al mangú.
  · Chillo al horno con vegetales y batata asada → ✏️ «Filete de pescado blanco al horno con vegetales y batata»:
    batata en cubos de unos 2 cm sobre papel de hornear (sin aceite no se pega), condimentos opcionales, tiempo
    ajustado al grosor, los vegetales siguen solos si el pescado termina antes.
  · Batida de zapote ligera (leche descremada) → ❌: falta el zapote, ingrediente principal. Fuera hasta que el
    catálogo tenga la fila con pulpa comestible, nutrición y precio verificado.

Dos decisiones técnicas que estos tests fijan:
  1. Un renombre NO cambia el `template_id` (`plan_policy.TEMPLATE_ALIASES`): la biblioteca, los blueprints vivos y
     las fichas citan el id histórico.
  2. La tabla curada `data/dish_constituents_do.json` es el SSOT; su script generador ya no la reproduce.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_REG = _BACKEND / "data" / "registry" / "dish_registry_do_v1.json"
_LIB = _BACKEND / "data" / "registry" / "recipe_library_do_v1.json"

IDS = {
    "tpl_0ae55a98c3e3": ("Frutas picadas con limón", "Frutas picadas con limón y menta", "merienda"),
    "tpl_4afb7ed71229": ("Mangú con jamón de pavo a la plancha", "Mangú con salami de pavo a la plancha (versión magra)", "desayuno"),
    "tpl_81b612498f83": ("Filete de pescado blanco al horno con vegetales y batata", "Chillo al horno con vegetales y batata asada", "cena"),
}
RETIRADA = "Batida de zapote ligera (leche descremada)"


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def reg():
    return json.loads(_REG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lib():
    return json.loads(_LIB.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def por_id(reg):
    return {t["template_id"]: t for t in reg["templates"]}


# ─────────────────────────── el registry: las cuatro cerradas, ninguna parcial ───────────────────────────

def test_las_tres_renombradas_compilan_integras_con_su_id_historico(por_id):
    for tid, (nuevo, viejo, franja) in IDS.items():
        t = por_id[tid]
        assert t["name"] == nuevo and t["editorial"]["display_name"]["es"] == nuevo, (tid, t["name"])
        assert t["status"] == "ok" and t["excluded"] == [], (tid, t["excluded"])
        assert t["slots"] == [franja]
    nombres = {t["name"] for t in por_id.values()}
    assert not (nombres & {v[1] for v in IDS.values()}), "los nombres heredados siguen en el registry"


def test_la_batida_de_zapote_salio_del_registry_y_de_las_fuentes(reg):
    assert "tpl_e63e01c38ad3" not in {t["template_id"] for t in reg["templates"]}
    assert not [t["name"] for t in reg["templates"] if "zapote" in t["name"].lower()]
    fuentes = json.loads(_src("data/dish_templates.json"))["templates"]
    assert RETIRADA not in {t["name"] for t in fuentes}
    tabla = json.loads(_src("data/dish_constituents_do.json"))["templates"]
    assert RETIRADA not in tabla


def test_el_registry_do_no_tiene_parciales_ni_declaraciones_sin_resolver(reg):
    st = reg["stats"]
    assert st["partial"] == 0 and st["excluded"] == 0 and st["ok"] == st["templates"] == 193, st
    assert st["resolution_pct"] == 100.0
    tabla = json.loads(_src("data/dish_constituents_do.json"))["templates"]
    assert all(not v.get("declared_unresolved") for v in tabla.values()), "vuelve una promesa sin resolver"
    assert "ES EL SSOT" in json.loads(_src("data/dish_constituents_do.json"))["_note"], (
        "la nota debe decir que el JSON manda: su script ya no lo reproduce (75 entradas divergen)")


def test_los_constituyentes_del_mangu_y_del_pescado_son_los_que_el_titulo_promete(por_id):
    assert "Jamón de pavo" in [c["name"] for c in por_id["tpl_4afb7ed71229"]["constituents"]]
    assert "Filete de pescado blanco" in [c["name"] for c in por_id["tpl_81b612498f83"]["constituents"]]
    # [P1-PLAN-LOTE-12] el rótulo que pidió el dueño; resuelve a la fila «Limón» por alias del catálogo
    assert "Jugo de limón" in [c["name"] for c in por_id["tpl_0ae55a98c3e3"]["constituents"]]


# ─────────────────────────── el id no se mueve con el renombre ───────────────────────────

def test_el_renombre_conserva_el_template_id():
    import plan_policy
    for tid, (nuevo, viejo, _f) in IDS.items():
        assert plan_policy.TEMPLATE_ALIASES.get(nuevo) == viejo, nuevo
    fuentes = {t["name"]: t for t in json.loads(_src("data/dish_templates.json"))["templates"]}
    for tid, (nuevo, viejo, _f) in IDS.items():
        assert plan_policy.mint_template_id(fuentes[nuevo], "do") == tid, (nuevo, "el renombre movió el id")


def test_el_alias_esta_anclado_en_el_codigo():
    assert "tooltip-anchor: TEMPLATE_ALIASES (test_p1_plan_lote_11.py)" in _src("plan_policy.py")


# ─────────────────────────── la biblioteca: las tres recetas, con los cambios que dictó ───────────────────────────

def test_las_tres_recetas_existen_y_cubren_todo_el_registry(lib, reg):
    for tid, (_n, _v, franja) in IDS.items():
        r = lib["por_id"][tid]
        assert r["franja"] == franja and 3 <= len(r["pasos"]) <= 5, tid
    assert "tpl_e63e01c38ad3" not in lib["por_id"]
    assert set(lib["por_id"]) == {t["template_id"] for t in reg["templates"]}, "toda plantilla del registry tiene receta"
    assert lib["recetas"] == len(lib["por_id"]) == 193


def test_frutas_picadas_guineo_sin_cascara_limon_exprimido_y_menta_opcional(lib):
    pasos = lib["por_id"]["tpl_0ae55a98c3e3"]["pasos"]
    txt = " ".join(pasos)
    assert "sin cáscara" in pasos[2]
    assert "jugo de limón" in pasos[3] and "suavidad" in pasos[3]
    assert "menta" in pasos[-1].lower() and "opcional" in pasos[-1]
    assert txt.lower().count("menta") == 1, "la menta sólo aparece como opcional, en el último paso"


def test_mangu_jamon_a_74_grados_y_el_aceite_repartido_mitad_y_resto(lib):
    pasos = lib["por_id"]["tpl_4afb7ed71229"]["pasos"]
    assert "74 °C" in pasos[2] and "jamón de pavo" in pasos[2]
    assert "la mitad del aceite" in pasos[3] and "cebolla" in pasos[3]
    assert "el resto del aceite" in pasos[4] and "mangú" in pasos[4]
    assert "salami" not in " ".join(pasos).lower()


def test_pescado_papel_de_hornear_condimentos_opcionales_y_tiempo_por_grosor(lib):
    pasos = lib["por_id"]["tpl_81b612498f83"]["pasos"]
    assert "papel de hornear" in pasos[0] and "2 cm" in pasos[0]
    assert "opcionales" in pasos[2] and "limón" in pasos[2]
    assert "grosor" in pasos[3] and "63 °C" in pasos[3] and "lascas" in pasos[3]
    assert "siguen duros" in pasos[4]
    assert "chillo" not in " ".join(pasos).lower()


def test_la_procedencia_cuenta_la_curacion_humana_sin_inflarla(lib):
    p = lib["procedencia"]
    assert p.get("curacion_humana_c8") and "Jugo de limón" in p["curacion_humana_c8"], (
        "lo único no aplicado literal se declara, no se disimula")
    assert "MUESTRA" in p["veredicto_humano"] and "C8" in p["veredicto_humano"]
    assert "v7 2026-09-11" in lib["revision"]


# ─────────────────────────── firma, benchmark y baseline siguen al snapshot ───────────────────────────

def test_la_firma_curatorial_y_el_benchmark_apuntan_al_snapshot_recompilado(reg):
    rev = json.loads(_src("data/registry/cultural_curation_review_v1.json"))
    prof = rev["profiles"]["dominican_criolla"]
    assert prof["snapshot_hash"] == reg["snapshot_hash"], "firma caducada: recompilaste sin re-anclar"
    assert prof["retirados_por_el_dueno"][0]["template_id"] == "tpl_e63e01c38ad3"
    assert "reanchor_note_2026_09_11" in rev
    bench = json.loads(_src("data/registry/cultural_benchmark_v1.json"))
    assert bench["profiles"]["dominican_criolla"]["snapshot_hash"] == reg["snapshot_hash"], (
        "informe desfasado — corre `python cultural_benchmark.py --write`")


def test_el_baseline_c3_habla_de_los_nombres_nuevos():
    m = json.loads(_src("scripts/data/do_corpus_retarget_baseline_2026_08_18.json"))["mapping"]
    for _tid, (nuevo, viejo, _f) in IDS.items():
        assert nuevo in m and viejo not in m, nuevo
    assert RETIRADA not in m


# ─────────────────────────── docs y marker ───────────────────────────

def test_los_docs_cuentan_el_cierre_y_lo_que_no_se_aplico():
    dd = _src("docs/deterministic_day.md")
    assert "las cerró el **dueño** el 2026-09-11" in dd and "193 `ok`, 0 `partial`" in dd
    assert "Jugo de limón" in dd and "es el SSOT" in dd
    assert re.search(r"^\| C8 \| ✅ 2026-09-11", _src("docs/plan_pendientes_2026_09_11.md"), re.M)
    assert "[P1-PLAN-LOTE-11 · 2026-09-11 · C8] Las cuatro están cerradas" in _src("docs/arq27_f1_seleccion.md")


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-11 · 2026-09-11]" in _src("app.py")
    assert app._LAST_KNOWN_PFIX.startswith("P1-PLAN-") and "2026-09-11" in app._LAST_KNOWN_PFIX
