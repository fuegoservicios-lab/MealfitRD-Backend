# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-796 · 2026-09-28] Alergia a frutos secos (y las otras nueve del formulario) × los 6 países.

Un revisor encontró que `_allergen_pool_item_banned("frutos secos", ["frutos secos"])` devolvía False y que la línea dura
del prompt de un alérgico a lácteos y frutos secos terminaba ofreciéndole «…frutos secos con fruta». La causa no estaba en
el pool ni en el prompt: el vocabulario de la clase listaba cada fruto seco por su nombre (almendra, nuez, pistacho…) y no
el nombre GENÉRICO, así que «10 g de frutos secos mixtos sin sal» —línea real de la batería (rdfinal/dm2_insulinafin)—
pasaba limpia por el escáner, que es la ÚNICA capa que ven el pool del esqueleto, las sugerencias del prompt, el tamiz del
camino degradado, el día determinista y el backstop final.

La misma sonda, extendida a las 10 alergias × 6 países, encontró la misma forma de hueco en otros sitios:
  · alias del catálogo que el resolutor acepta y el escáner no ve («cajuil» → Merey, «pistaches» → Pistachos, «pacanas»,
    «harina de todo uso» → Harina de trigo, «gouda», «cheddar», «cajeta» → Arequipe, «ajoaceite» → Alioli, «humus»…);
  · el plural de los términos compuestos: «2 tortillas de harina» y «2 tortillas integrales» (200 líneas en el corpus)
    pasaban limpias para un celíaco — el plural sólo se aceptaba en la ÚLTIMA palabra;
  · el filtro del catálogo de los países beta dejaba Gambas, Almejas, Percebes, Vieira (mariscos), Anchoas, Boquerones,
    Trucha y Bacalaítos (pescado) en el pool del alérgico — la paridad con el escáner sólo se medía en DO;
  · el registry de platos (día determinista y CandidateSet del prompt) excluía por igualdad EXACTA con el chip: «maní»,
    «lácteos», «nueces», «soja»… escritos a mano no excluían nada, y 7 plantillas tenían etiquetas de alérgenos viejas
    (granola sin gluten, natilla sin huevo).
tooltip-anchor: P1-PLAN-LOTE-796-ALERGIA-FRUTOS-SECOS
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

CHIPS = ("Lacteos", "Gluten", "Huevo", "Mariscos", "Pescado", "Frutos Secos", "Mani", "Soya", "Sesamo", "Lactosa")
PAISES = ("DO", "ES", "US", "MX", "PR", "CO")


def _plan(*ings):
    return {"days": [{"day": 1, "meals": [{"meal": "Merienda", "name": "Plato", "ingredients": list(ings)}]}]}


def _viola(ing, alergias):
    return bool(go._scan_allergen_violations(_plan(ing), list(alergias)))


def _esqueleto():
    return {"brief_concept": "Día variado", "assigned_technique": "A la plancha",
            "protein_pool": ["Pollo"], "carb_pool": ["Avena", "Batata", "Yuca"], "fruit_pool": ["Mango"],
            "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"]}


# ─────────────── (1) el defecto reportado: el pool y la línea dura ───────────────

@pytest.mark.parametrize("decl", ["Frutos Secos", "frutos secos", "nueces", "tree nuts"])
@pytest.mark.parametrize("item", ["frutos secos", "Frutos secos mixtos", "10 g de frutos secos mixtos sin sal",
                                  "1 puñado de fruto seco"])
def test_el_pool_veta_los_frutos_secos_por_su_nombre_generico(decl, item):
    assert go._allergen_pool_item_banned(item, [decl]), (decl, item)


def test_la_linea_dura_no_ofrece_frutos_secos_al_alergico():
    from prompts.day_generator import allergy_hard_line
    linea = allergy_hard_line(["lácteos", "frutos secos"])
    alternativas = linea.split("Sin lácteos:", 1)[1]
    assert "frutos secos" not in alternativas and "casabe con aguacate" in alternativas, alternativas
    # [revisión] al alérgico al MANÍ tampoco: la mezcla lleva maní («Nueces mixtas» = mixed nuts WITH peanuts); se le
    # ofrece el fruto seco suelto. Sin alergia al maní ni a frutos secos, la sugerencia de siempre.
    assert "almendras con fruta" in allergy_hard_line(["Lacteos", "Maní"])
    assert "frutos secos con fruta" not in allergy_hard_line(["Lacteos", "Maní"])
    assert "frutos secos con fruta" in allergy_hard_line(["Lacteos", "Huevo"])


def _sugerencias(ctx):
    ok = re.search(r"Para diversificar desayuno/merienda usa: (.+?) \(estas son OK siempre", ctx).group(1)
    alts = re.search(r"la merienda NO lleva avena \(usa (.+?)\)", ctx).group(1)
    return [x.strip() for x in re.split(r",| o ", ok + "," + alts) if x.strip()]


def test_al_alergico_a_frutos_secos_la_asignacion_no_se_los_sugiere():
    """El oráculo del test de abajo es el escáner; éste lo fija por su nombre, sin depender de él."""
    from prompts.day_generator import build_day_assignment_context
    for decl in ("Frutos Secos", "nueces"):
        ctx = build_day_assignment_context(_esqueleto(), 1, allergies=[decl], dislikes=["Ninguno"])
        assert "frutos secos" not in _sugerencias(ctx), (decl, _sugerencias(ctx))
    assert "frutos secos" not in _sugerencias(build_day_assignment_context(_esqueleto(), 1, allergies=["Mani"]))
    assert "frutos secos" in _sugerencias(build_day_assignment_context(_esqueleto(), 1, allergies=["Huevo"]))


@pytest.mark.parametrize("chip", CHIPS)
def test_ninguna_sugerencia_de_la_asignacion_del_dia_contradice_la_alergia(chip):
    """Para cada chip, lo que el prompt ofrece «OK siempre» y como alternativa ligera pasa el escáner de esa alergia."""
    from prompts.day_generator import build_day_assignment_context, allergy_hard_line
    ctx = build_day_assignment_context(_esqueleto(), 1, allergies=[chip], dislikes=["Ninguno"])
    malas = [s for s in _sugerencias(ctx) if _viola(s, [chip])]
    assert not malas, (chip, malas)
    linea = allergy_hard_line([chip, "Lacteos"])
    if "Sin lácteos:" in linea:
        alts = re.search(r"meriendas y desayunos con (.+?)\.$", linea).group(1).split(", ")
        assert not [a for a in alts if _viola(a, [chip, "Lacteos"])], (chip, alts)


def test_la_cuota_por_comida_no_pide_el_alergeno_como_portador_de_grasa():
    from prompts.day_generator import build_slot_targets_block
    objetivos = {"calories": 2000, "protein": 120, "carbs": 220, "fats": 80}
    comidas = ["Desayuno", "Almuerzo", "Merienda", "Cena"]
    base = build_slot_targets_block(objetivos, comidas)
    assert "frutos secos" in base and "mantequilla de maní" in base and "queso" in base
    assert build_slot_targets_block(objetivos, comidas, vetos=[]) == base      # sin vetos: byte-idéntico
    con = build_slot_targets_block(objetivos, comidas, vetos=["Frutos Secos", "Mani", "Lacteos"])
    portador = re.search(r"fuente real \((.+?)\)", con).group(1)
    assert "frutos secos" not in portador and "maní" not in portador and "queso" not in portador, portador
    assert "aceite" in portador and "aguacate" in portador


# ─────────────── (2) el vocabulario: nombres por país ───────────────

@pytest.mark.parametrize("ing", [
    "10 g de frutos secos mixtos sin sal", "1 puñado de fruto seco", "20 g de frutos secos",
    "15 g de cajuil", "semillas de cajuil tostadas", "1 cda de mantequilla de cajuil",   # DO
    "15 g de pajuil",                                                                     # PR
    "castañas de cajú", "20 g de pistaches", "1 pistache",                                # MX
    "chiles en nogada", "20 g de pacanas", "2 cdas de salsa romesco", "1 taza de ajoblanco",  # MX · ES
    "30 g de nueces pecanas", "anacardos", "marañones", "merey", "avellanas", "piñones",
])
def test_el_escaner_ve_cada_fruto_seco_por_su_nombre_local(ing):
    assert _viola(ing, ["Frutos Secos"]), ing


@pytest.mark.parametrize("decl", ["Frutos Secos", "frutos secos", "nueces", "almendras", "avellanas", "anacardos",
                                  "merey", "marañón", "cajuil", "pistachos", "pistaches", "pecanas", "tree nuts",
                                  "nuts", "noix", "frutta secca"])
def test_cada_forma_de_declarar_frutos_secos_cubre_el_generico(decl):
    assert _viola("10 g de frutos secos mixtos sin sal", [decl]), decl
    assert _viola("20 g de almendras", [decl]), decl


@pytest.mark.parametrize("decl", ["Mani", "maní", "Maní", "cacahuate", "cacahuete", "cacahuates", "peanut",
                                  "arachide", "amendoim"])
@pytest.mark.parametrize("ing", ["1 cda de mantequilla de maní", "crema de cacahuete", "20 g de cacahuates tostados",
                                 "salsa de maní", "15 g de maní tostado"])
def test_mani_por_pais(decl, ing):
    assert _viola(ing, [decl]), (decl, ing)


@pytest.mark.parametrize("ing,chip", [
    ("2 tortillas de harina", "Gluten"), ("2 tortillas integrales", "Gluten"),
    ("3 tortillas de harina de trigo", "Gluten"), ("2 ensaladas rusas", "Huevo"),
    ("2 porciones de tortilla española", "Huevo"), ("1 tortilla de papas", "Huevo"),
    ("1 porción de tortilla de patatas", "Huevo"), ("1 cda de ajoaceite", "Huevo"),
    ("100 g de harina de todo uso", "Gluten"), ("1 taza de harina blanca", "Gluten"),
    ("harina multiusos", "Gluten"), ("2 muffins ingleses", "Gluten"), ("1 hotcake", "Gluten"),
    ("50 g de burgol", "Gluten"), ("1 plato de cuchuco", "Gluten"), ("pan sobao", "Gluten"), ("sobao", "Gluten"),
    ("4 crackers", "Gluten"), ("frituras de bacalao", "Gluten"), ("1 biscuit", "Gluten"),
    ("2 quipes horneados", "Gluten"), ("1 kipe", "Gluten"), ("1 taza de tipilí", "Gluten"),
    ("30 g de gouda", "Lacteos"), ("20 g de cheddar rallado", "Lacteos"), ("provolone", "Lacteos"),
    ("edam", "Lacteos"), ("30 g de manchego", "Lacteos"), ("1 cda de cajeta", "Lacteos"),
    ("manjar blanco", "Lacteos"), ("1 quesito", "Lacteos"), ("suero atollabuey", "Lacteos"),
    ("100 g de cottage", "Lactosa"), ("30 g de gouda", "Lactosa"), ("1 cda de cajeta", "Lactosa"),
    ("concha de abanico", "Mariscos"), ("2 cdas de humus", "Sesamo"),
])
def test_alias_y_plurales_que_el_escaner_no_veia(ing, chip):
    assert _viola(ing, [chip]), (ing, chip)


@pytest.mark.parametrize("ing,chip", [
    ("30 g de frutas secas", "Frutos Secos"), ("2 tortillas de maíz", "Gluten"), ("1 cuchuco de maíz", "Gluten"),
    ("4 crackers de arroz", "Gluten"), ("100 g de harina de maíz precocida", "Gluten"), ("1 casabe", "Gluten"),
    ("1 cda de panela rallada", "Lacteos"), ("1 taza de leche de coco", "Lacteos"), ("1 papa", "Huevo"),
    ("1 tortilla de maíz", "Huevo"), ("200 g de papas hervidas", "Huevo"), ("1 cda de aceite de coco", "Frutos Secos"),
])
def test_sin_falsos_positivos_nuevos(ing, chip):
    assert not _viola(ing, [chip]), (ing, chip)


def test_el_plural_compuesto_no_arrastra_otra_clase():
    """El plural de las palabras internas sólo AÑADE formas del mismo término: las declaraciones siguen resolviendo
    a las mismas clases (el contrato de `test_p1_plan_lote_252::test_una_declaracion_no_arrastra_otra_clase`)."""
    from constants import strip_accents
    def clases(decl):
        exp = go._expand_allergy_declarations([decl])
        return {c for c, syns in go._ALLERGEN_SYNONYMS.items()
                if syns and {strip_accents(str(s)).lower() for s in syns} <= exp}
    assert clases("papas") == set() and clases("papa") == set() and clases("tortilla") == {"gluten"}
    assert clases("frutas") == set() and clases("fruta") == set() and clases("bacalao") == {"pescado"}
    assert clases("concha") == {"gluten"} and clases("frutos secos") == {"frutos secos"}
    assert clases("conchas") == {"gluten"}


# ─────────────── alias del catálogo: si la FILA es alérgeno, su alias también ───────────────

# El plato se escribe en español (frontera de P1-I18N-DASHBOARD: los nombres de comida no se traducen) y el escáner
# busca sinónimos españoles; los alias en inglés del catálogo son para ENTENDER lo que el usuario escribe, no para el
# plato. Si mañana entra un alias en español que el escáner no ve, este test lo nombra.
_ALIAS_EN_INGLES = {
    "scallops", "nougat", "buttermilk", "canned sardines", "sardine", "sausage gravy", "provolone",
    "gouda cheese", "string cheese", "cream cheese", "cheddar cheese", "octopus", "pine nuts", "pistachio",
    "pistachios", "goose barnacles", "english muffins", "buttermilk biscuits", "breadcrumbs", "cornbread", "walnuts",
    "english walnut", "pecans", "mixed nuts", "heavy cream", "grouper", "cashew", "mussels",
    "blue mussels", "pie crust", "almond butter", "butter", "evaporated milk", "soy milk", "goat milk powder",
    "oat milk", "oatmilk", "almond milk", "deviled eggs", "all-purpose flour", "prawns", "crackers", "saltines",
    "soda crackers", "graham crackers", "macaroni salad", "half and half", "sour cream", "elbow macaroni",
    "barley", "pearled barley", "crab", "blue crab", "squid", "cod fish", "oats", "herring", "kippered herring",
    "marcona almonds", "clams", "sesame seeds", "sesame", "creamy wheat", "cream of wheat", "pie crust refrigerada",
}
# Alias que NO nombran el alérgeno de su fila a propósito: el alias es de otra cosa que la fila también es.
_ALIAS_NO_ALERGENO = {("Gluten", "soya"),        # «Salsa de soya» (trigo) también se busca por «soya», que no es trigo
                      ("Gluten", "harina"),      # «harina» sola: 46 líneas reales son «harina de maíz» (lote 252)
                      ("Lactosa", "parmesano"),  # queso curado sin lactosa; la clase lactosa es estrecha a propósito
                      # [revisión] «Nueces mixtas» lleva maní (la mezcla), pero «nueces»/«nuez» nombran el fruto seco suelto
                      ("Mani", "nueces"), ("Mani", "nuez")}


def test_cada_alias_espanol_del_catalogo_hereda_la_alergia_de_su_fila():
    filas = json.loads((_BACKEND / "scripts/data/catalogo_nutricion_2026_09_12.json").read_text(encoding="utf-8"))["filas"]
    faltan = []
    for r in filas:
        for chip in CHIPS:
            if not _viola(r["name"], [chip]):
                continue
            for a in r.get("aliases") or []:
                if a.lower() in _ALIAS_EN_INGLES or (chip, a.lower()) in _ALIAS_NO_ALERGENO:
                    continue
                if not _viola(a, [chip]):
                    faltan.append((chip, r["name"], a))
    assert not faltan, faltan


# ─────────────── (3) el filtro del catálogo, en los 6 países ───────────────

@pytest.mark.parametrize("pais", PAISES)
@pytest.mark.parametrize("chip", CHIPS)
def test_el_catalogo_del_pais_no_ofrece_lo_que_el_escaner_prohibe(chip, pais):
    from constants import _get_fast_filtered_catalogs
    pools = _get_fast_filtered_catalogs((chip,), (), "balanced", country=pais, market_extras=True,
                                        culture_country=pais)
    quedan = [x for pool in pools for x in pool]
    malos = [x for x in quedan if _viola(str(x), [chip])]
    assert not malos, (chip, pais, malos)


# ─────────────── (4) el registry de platos: día determinista y CandidateSet del prompt ───────────────

@pytest.mark.parametrize("decl", ["Mani", "maní", "cacahuete", "Frutos Secos", "nueces", "almendras", "Lacteos",
                                  "lácteos", "leche", "Gluten", "trigo", "celiaco", "Huevo", "huevos", "Sesamo",
                                  "sésamo", "ajonjolí", "Soya", "soja", "Mariscos", "camarones", "Pescado", "atún",
                                  "Lactosa", "frutos de cáscara", "APLV", "celiaquía", "crustáceos"])
def test_el_registry_excluye_la_alergia_escrita_a_mano(decl):
    import dish_registry as dr
    terminos = go._expand_allergy_declarations([decl])
    malos, vistos = [], set()
    for pais in PAISES:
        por_id = dr.templates_by_id(pais) or {}
        for slot in ("desayuno", "almuerzo", "merienda", "cena"):
            for c in dr.template_candidates(pais, slot, None, k=500, exclude_allergens=[decl]):
                if (pais, c.get("template_id")) in vistos:
                    continue
                vistos.add((pais, c.get("template_id")))
                t = por_id.get(c.get("template_id")) or {}
                nombres = [x.get("name") or "" for x in t.get("constituents") or []]
                if go._scan_allergen_violations(_plan(*nombres), [decl], terminos=terminos):
                    malos.append((pais, t.get("name")))
    assert not malos, (decl, sorted(set(malos))[:8])


def test_el_registry_no_excluye_de_mas_sin_alergia():
    import dish_registry as dr
    for pais in PAISES:
        for slot in ("desayuno", "almuerzo", "merienda", "cena"):
            assert (len(dr.template_candidates(pais, slot, None, k=500))
                    == len(dr.template_candidates(pais, slot, None, k=500, exclude_allergens=["Ninguna"])))
