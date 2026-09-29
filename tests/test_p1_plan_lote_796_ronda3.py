# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-796 · 2026-09-28] Ronda 3 de la revisión del lote: lo que la ronda 2 rompió y lo que seguía abierto.

  · La línea dura del prompt corta en 20 términos por alergia y los nombres nuevos (quesos sueltos, alias locales,
    platos que esconden el alérgeno) desplazaban a los de siempre: Huevo perdía «mayonesa», Frutos secos «pistacho».
    Lo frecuente va primero y lo oculto/alias al final; cada alergia conserva lo que la línea nombraba antes.
  · El capuchino, el latte o la bechamel hechos con leche vegetal marcaban lácteo (vegano y alérgico).
  · El escáner sólo miraba la PRIMERA aparición de cada término: si ésa estaba excusada, la segunda pasaba
    («leche de almendras o leche descremada», «salsa macha y machas», «mole de olla y mole negro»).
  · Nombres que el escáner no veía: mezclas de nueces (maní), masas con huevo, la tortilla de verduras, albóndigas,
    ranch, pan de maíz, César en el nombre de la plantilla y los de la lista 11.
tooltip-anchor: P1-PLAN-LOTE-796-RONDA-3
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


def _plan(*ings):
    return {"days": [{"day": 1, "meals": [{"meal": "Merienda", "name": "Plato", "ingredients": list(ings)}]}]}


def _viola(ing, alergias):
    return bool(go._scan_allergen_violations(_plan(ing), list(alergias)))


def _vegano_viola(ing):
    return bool(go._scan_diet_violations(_plan(ing), "vegan"))


def _incluye(alergias):
    from prompts.day_generator import allergy_hard_line
    m = re.search(r"\(incluye: ([^)]*)\)", allergy_hard_line(list(alergias)))
    return m.group(1).split(", ") if m else []


# ─────────────── 1 · la línea dura del prompt no pierde lo que nombraba ───────────────

# Lo que la línea nombraba en main (merge-base 94d0425a), medido con `allergy_hard_line` antes del lote.
LINEA_DE_MAIN = {
    ("Huevo",): "clara, huevo, claras, huevos, flan, yema, aioli, cesar, yemas, alioli, mousse, omelet, ponche, huevito, "
                "natilla, frittata, mayonesa, merengue, omelette, holandesa",
    ("Lácteos",): "crema, leche, queso, yogur, yogurt, cottage, ricotta, mantequilla, flan, ghee, nata, whey, cesar, kefir, "
                  "helado, lacteo, caseina, cuajada, natilla, arequipe",
    ("Lactosa",): "crema, leche, queso, yogur, yogurt, ricotta, mantequilla, flan, nata, whey, kefir, helado, cuajada, "
                  "natilla, arequipe, requeson, mantecado, mozzarella, queso crema, suero costeno",
    ("Gluten",): "pan, pasta, trigo, harina de trigo, coca, pita, wrap, avena, bagel, crepa, crepe, farro, fideo, kamut, "
                 "malta, migas, noqui, penne, wafle, wheat",
    ("Maní",): "mani, mantequilla de mani, peanut, cacahuate, cacahuete, cacahuetes, crema de mani, salsa de mani, "
               "crema de cacahuete, mantequilla de cacahuete",
    ("Frutos secos",): "merey, nueces, almendra, almendras, nuez, pesto, pinon, pecana, turron, castana, maranon, mazapan, "
                       "nutella, pinones, praline, anacardo, avellana, marzipan, pistacho, macadamia",
    ("Sésamo",): "halva, tahin, hummus, sesamo, tahina, tahini, zaatar, gomasio, ajonjoli, aceite de sesamo, "
                 "semillas de sesamo",
    ("Mariscos",): "lambi, pulpo, calamar, camaron, cangrejo, langosta, mejillon, camarones, choro, erizo, gamba, jaiba, "
                   "macha, ostra, sepia, almeja, cigala, gambas, navaja, necora",
    ("Pescado",): "atun, bacalao, pescado, tilapia, mero, rape, bagre, carpa, cazon, cesar, fumet, hueva, jurel, melva, "
                  "pargo, perca, picua, sargo, anchoa, angula",
    ("Soya",): "soya, tofu, tvp, miso, soja, natto, shoyu, tamari, tempeh, edamame, teriyaki, salsa de soya, "
               "salsa teriyaki, lecitina de soya, proteina de soya, proteina vegetal texturizada",
    ("Lácteos", "Huevo"): "clara, crema, huevo, leche, queso, yogur, claras, huevos, yogurt, cottage, ricotta, mantequilla, "
                          "flan, ghee, nata, whey, yema, aioli, cesar, kefir",
    ("Gluten", "Huevo", "Lácteos"): "pan, clara, crema, huevo, leche, pasta, queso, trigo, yogur, claras, huevos, yogurt, "
                                    "cottage, ricotta, mantequilla, harina de trigo, coca, flan, ghee, nata",
    ("Maní", "Frutos secos"): "mani, merey, nueces, almendra, almendras, mantequilla de mani, nuez, pesto, pinon, peanut, "
                              "pecana, turron, castana, maranon, mazapan, nutella, pinones, praline, anacardo, avellana",
    ("Gluten", "Lácteos"): "pan, crema, leche, pasta, queso, trigo, yogur, yogurt, cottage, ricotta, mantequilla, "
                           "harina de trigo, coca, flan, ghee, nata, pita, whey, wrap, avena",
    ("Huevo", "Maní", "Soya", "Sésamo"): "mani, soya, tofu, clara, huevo, claras, huevos, mantequilla de mani, tvp, flan, "
                                        "miso, soja, yema, aioli, cesar, halva, natto, shoyu, tahin, yemas",
}


@pytest.mark.parametrize("alergias", list(LINEA_DE_MAIN))
def test_la_linea_dura_conserva_lo_que_nombraba_antes(alergias):
    ahora = set(_incluye(alergias))
    perdidos = [t for t in LINEA_DE_MAIN[alergias].split(", ") if t not in ahora]
    assert not perdidos, (alergias, perdidos)


def test_la_linea_dura_nombra_primero_lo_frecuente():
    huevo, frutos = _incluye(["Huevo"]), _incluye(["Frutos secos"])
    assert "mayonesa" in huevo and "pistacho" in frutos and "panqueque" in huevo
    assert {"helado", "cuajada", "arequipe", "requeson", "mozzarella", "queso crema"} <= set(_incluye(["Lactosa"]))
    assert {"helado", "caseina", "cuajada"} <= set(_incluye(["Lácteos"]))
    assert {"penne", "wafle", "wheat"} <= set(_incluye(["Gluten"]))
    # lo frecuente, antes que lo oculto: la mayonesa va delante de la croqueta y el pistacho delante del cajú
    for linea, frecuente, oculto in ((huevo, "mayonesa", "omelet"), (frutos, "pistacho", "turron")):
        assert linea.index(frecuente) < linea.index(oculto), (frecuente, oculto, linea)


def test_los_platos_ocultos_van_al_final_de_su_alergia():
    """Para el maní la línea tiene hueco: los nombres de base y, detrás, la mezcla de frutos secos y el mole."""
    mani = _incluye(["Maní"])
    assert "frutos secos" in mani and "mole" in mani
    assert mani.index("mantequilla de cacahuete") < mani.index("mole"), mani
    huevo = _incluye(["Huevo"])
    for oculto in ("croqueta", "apanado", "pancake", "tortita"):
        if oculto in huevo:
            assert huevo.index("holandesa") < huevo.index(oculto), (oculto, huevo)


# ─────────────── 2 · el preparado lácteo hecho con leche vegetal ───────────────

@pytest.mark.parametrize("ing", ["1 capuchino con leche de avena", "1 latte con bebida de almendras",
                                 "bechamel de coliflor", "1 cappuccino preparado con leche de soya",
                                 "1 latte frío con leche vegetal", "1/2 taza de bechamel con leche de avena",
                                 "1 capuchino (con leche de coco)"])
def test_el_preparado_con_leche_vegetal_no_es_lacteo(ing):
    for chip in ("Lacteos", "Lactosa", "APLV"):
        assert not _viola(ing, [chip]), (ing, chip)
    assert not _vegano_viola(ing), ing


@pytest.mark.parametrize("ing", ["1 capuchino", "1 latte con leche entera", "2 cdas de bechamel",
                                 "1 capuchino con leche de avena y crema batida",
                                 "1 latte con leche de avena o leche entera",
                                 "1 capuchino y 1 vaso de bebida de almendras con queso"])
def test_el_preparado_lacteo_sigue_marcado(ing):
    assert _viola(ing, ["Lacteos"]), ing
    assert _vegano_viola(ing), ing


# ─────────────── 3-4 · cada aparición del término, no sólo la primera ───────────────

@pytest.mark.parametrize("ing,chip", [
    ("1 taza de leche de almendras o leche descremada", "Lacteos"),
    ("yogur de coco o yogur griego", "Lacteos"),
    ("1 taza de leche de soya o leche entera", "Lacteos"),
    ("1 taza de leche de soya o leche entera", "Lactosa"),
    ("2 tostadas de casabe o tostadas integrales", "Gluten"),
    ("caldo de mole de olla y mole negro", "Mani"),
    ("chocolate para mole y 50 g de mole poblano", "Mani"),
    ("chocolate para mole y 50 g de mole poblano", "Sesamo"),
    ("chocolate para mole y 50 g de mole poblano", "Gluten"),
    ("pisto manchego con 30 g de manchego", "Lacteos"),
    ("1 pastelito de yuca y 1 pastelito de pollo", "Gluten"),
    ("salsa macha y machas", "Mariscos"),
    ("15 g de almendras tostadas y 2 tostadas integrales", "Gluten"),
])
def test_la_segunda_aparicion_no_se_excusa_por_la_primera(ing, chip):
    assert _viola(ing, [chip]), (ing, chip)


@pytest.mark.parametrize("ing", ["1 taza de leche de almendras o leche descremada", "yogur de coco o yogur griego",
                                 "1 taza de leche de soya o leche entera", "pisto manchego con 30 g de manchego"])
def test_la_dieta_vegana_tambien_mira_cada_aparicion(ing):
    assert _vegano_viola(ing), ing


@pytest.mark.parametrize("ing,chip", [
    ("leche de almendras y leche de coco", "Lacteos"), ("tostadas de casabe y tostadas de yuca", "Gluten"),
    ("1 plato de mole de olla y otro de mole de caderas", "Mani"), ("salsa macha y más salsa macha", "Mariscos"),
    ("almendras tostadas y nueces tostadas", "Gluten"), ("yogur de coco y yogur de almendras", "Lacteos"),
    ("2 pastelitos de yuca y 1 pastelito de plátano", "Gluten"),
])
def test_si_todas_las_apariciones_estan_excusadas_no_viola(ing, chip):
    assert not _viola(ing, [chip]), (ing, chip)


def test_una_sola_violacion_por_ingrediente():
    """Recorrer todas las apariciones no multiplica el informe: un ingrediente, una violación."""
    v = go._scan_allergen_violations(_plan("leche de almendras o leche entera con queso"), ["Lacteos"])
    assert len(v) == 1, v


# ─────────────── 5 · nombres que el escáner no veía ───────────────

@pytest.mark.parametrize("ing,chip", [
    # mezclas de nueces para el maní (normalize_ingredient_for_tracking las resuelve a «nueces/almendras», con maní)
    ("15 g de mezcla de nueces", "Mani"), ("20 g de mix de nueces", "Mani"), ("nueces variadas", "Mani"),
    ("1 puñado de nueces y semillas", "Mani"), ("30 g de trail mix", "Mani"),
    # huevo en masas
    ("2 pancakes de avena y plátano", "Huevo"), ("2 hot cakes", "Huevo"), ("2 hotcakes", "Huevo"),
    ("2 waffles", "Huevo"), ("2 wafles", "Huevo"), ("1 muffin", "Huevo"), ("2 crepas", "Huevo"),
    ("2 crepes", "Huevo"), ("1 bizcocho", "Huevo"), ("1 magdalena", "Huevo"), ("2 tortitas", "Huevo"),
    ("2 panqueques", "Huevo"), ("tortitas de avena y plátano", "Huevo"), ("tortitas de papa", "Huevo"),
    # la tortilla de verduras (DO/ES/PR/CO) es de huevo
    ("1 tortilla de vegetales", "Huevo"), ("1 tortilla de espinaca", "Huevo"), ("1 tortilla de queso", "Huevo"),
    ("1 tortilla de yuca", "Huevo"), ("1 tortilla de papa", "Huevo"), ("1 tortilla de verduras", "Huevo"),
    ("1 porción de tortilla de espinacas", "Huevo"),
    # albóndigas, ranch, pan de maíz
    ("4 albóndigas de res", "Huevo"), ("4 albóndigas de res", "Gluten"), ("2 cdas de aderezo ranch", "Lacteos"),
    ("2 cdas de aderezo ranch", "Huevo"), ("2 cdas de aderezo ranch", "Lactosa"),
    ("1 rebanada de pan de maíz", "Huevo"), ("1 rebanada de pan de maíz", "Lacteos"), ("1 cornbread", "Huevo"),
    ("1 cornbread", "Lacteos"),
    # lista 11 de la revisión
    ("150 g de sierra en escabeche", "Pescado"), ("escabeche de sierra", "Pescado"),
    ("1 burrito de pollo", "Gluten"), ("1 sándwich de pavo", "Gluten"), ("1 sandwich de atún", "Gluten"),
    ("1 sándwich integral de pollo", "Gluten"), ("2 tostadas de maíz (casabe fino o tostada integral)", "Gluten"),
    ("1 vaso de habichuelas con dulce (leche de coco, azúcar, leche)", "Lacteos"),
    ("2 cdas de panko", "Gluten"), ("1 hoja de hojaldre", "Gluten"), ("2 empanadillas de atún", "Gluten"),
    ("2 pastelillos de carne", "Gluten"), ("1 magdalena", "Gluten"), ("4 galletitas", "Gluten"),
    ("1 quesadilla de pollo", "Lacteos"), ("2 pandebonos", "Lacteos"), ("1 almojábana", "Lacteos"),
    ("2 buñuelos", "Lacteos"), ("1 taza de ensaladilla rusa", "Huevo"), ("2 cdas de salsa rosada", "Huevo"),
    ("1 cda de salsa tártara", "Huevo"),
])
def test_nombres_nuevos_del_escaner(ing, chip):
    assert _viola(ing, [chip]), (ing, chip)


@pytest.mark.parametrize("ing,chip", [
    ("2 pancakes de avena y plátano", "Mani"), ("1 tortilla de maíz", "Huevo"), ("2 tortillas de harina", "Huevo"),
    ("1 tortilla integral", "Huevo"), ("2 tortitas de arroz", "Huevo"), ("2 tortitas de maíz inflado", "Huevo"),
    ("1 english muffin", "Huevo"), ("1 muffin inglés", "Huevo"), ("2 muffins ingleses", "Huevo"),
    ("60 g de jamón de sándwich", "Gluten"), ("2 lonjas de jamón de sandwich", "Gluten"),
    ("1 porción de casabe (tortilla de yuca)", "Huevo"), ("2 tostadas de maíz (tostada integral de maíz)", "Gluten"),
    ("yogurt griego con nueces y semillas de linaza", "Mani"),
    ("30 g de almendras", "Mani"), ("20 g de nueces", "Mani"), ("huevos rancheros", "Lacteos"),
    ("2 cdas de salsa ranchera", "Lacteos"), ("1 tortilla de maíz con queso", "Huevo"),
    ("1 porción de ensalada de pollo", "Huevo"),
])
def test_los_nombres_nuevos_no_bloquean_de_mas(ing, chip):
    assert not _viola(ing, [chip]), (ing, chip)


def test_la_bechamel_con_leche_vegetal_sigue_llevando_harina():
    """La leche vegetal excusa el lácteo de la bechamel, no la harina del roux; la «de coliflor» no lleva ninguna."""
    for ing in ("1/2 taza de bechamel con leche de almendras", "bechamel con bebida de soya"):
        assert _viola(ing, ["Gluten"]), ing
        assert _viola(ing, ["Gluten", "Lacteos"]), ing
        assert not _viola(ing, ["Lacteos"]), ing
    assert not _viola("bechamel de coliflor", ["Gluten"])


@pytest.mark.parametrize("ing,no_viola,viola", [
    ("2 pancakes veganos", ["Huevo", "Lacteos"], ["Gluten", ["Huevo", "Gluten"]]),
    ("2 pancakes sin huevo", ["Huevo"], ["Gluten"]),
    ("1 porción de pizza vegana", ["Lacteos"], ["Gluten", ["Lacteos", "Gluten"]]),
    ("4 croquetas veganas", ["Huevo", "Lacteos"], ["Gluten"]),
    ("2 cdas de mole vegano", [], ["Mani", "Sesamo"]),
])
def test_la_excusa_vegana_es_de_las_clases_animales(ing, no_viola, viola):
    for chip in no_viola:
        assert not _viola(ing, [chip]), (ing, chip)
    for chip in viola:
        assert _viola(ing, chip if isinstance(chip, list) else [chip]), (ing, chip)


@pytest.mark.parametrize("ing,chip", [
    ("2 pancakes de avena", "Huevo"), ("1 muffin de almendras", "Huevo"), ("4 croquetas de arroz", "Huevo"),
    ("4 croquetas de arroz", "Gluten"), ("2 crepas de coco", "Huevo"), ("2 cdas de mole de almendra", "Mani"),
])
def test_el_adjetivo_vegetal_no_excusa_un_plato(ing, chip):
    """«de avena», «de arroz», «de coco» tras un PLATO nombran su harina o su relleno, no un análogo sin el alérgeno."""
    assert _viola(ing, [chip]), (ing, chip)
    assert not _viola("1 taza de leche de coco", ["Lacteos"]) and not _viola("2 cdas de mantequilla de maní", ["Lacteos"])


def test_el_muffin_ingles_sigue_siendo_trigo():
    """La excusa del muffin inglés es del HUEVO: para el celíaco sigue siendo pan."""
    for ing in ("1 english muffin", "1 muffin inglés"):
        assert _viola(ing, ["Gluten"]), ing
        assert _viola(ing, ["Gluten", "Huevo"]), ing


def test_los_nombres_nuevos_no_arrastran_clases():
    """Los platos se BUSCAN con la clase declarada; no resuelven una declaración a otra clase."""
    from constants import strip_accents
    for decl, esperado in (("pancakes", {"gluten"}), ("albóndigas", set()), ("ranch", set()), ("quesadilla", set()),
                           ("mezcla de nueces", {"frutos secos"}), ("sierra", set()),
                           ("tortilla de espinaca", set())):
        exp = go._expand_allergy_declarations([decl])
        clases = {c for c, syns in go._ALLERGEN_SYNONYMS.items()
                  if syns and {strip_accents(str(s)).lower() for s in syns} <= exp}
        assert clases == esperado, (decl, clases)


def test_la_dieta_vegana_no_hereda_los_platos_nuevos():
    """Los platos que esconden huevo o lácteo son de la ALERGIA: el vegano los recibe con su versión vegetal y el
    escáner de dieta no los mira (no hay un «pancake» vegano que marcar)."""
    for ing in ("2 pancakes de avena y plátano", "1 muffin de plátano", "1 tortilla de vegetales", "1 quesadilla",
                "4 albóndigas de lentejas"):
        assert not _vegano_viola(ing), ing


# ─────────────── 5 · el registry (día determinista, coach, CandidateSet) ───────────────

def _registry(pais, alergias=()):
    import dish_registry as dr
    out = set()
    for slot in ("desayuno", "almuerzo", "cena", "merienda"):
        out |= {c.get("name") for c in dr.template_candidates(pais, slot, None, k=500,
                                                              exclude_allergens=list(alergias))}
    return out


@pytest.mark.parametrize("pais,plato,alergias", [
    ("US", "Pollo con aderezo ranch y vegetales", ("Huevo", "Lácteos", "APLV")),
    ("US", "Wafles con jarabe de arce", ("Huevo", "Lácteos", "APLV")),
    ("US", "Chili con carne con pan de maíz", ("Huevo", "Lácteos", "APLV")),
    ("US", "Ensalada César con pollo", ("Huevo", "Pescado")),
    ("ES", "Albóndigas en salsa con puré de patata", ("Huevo", "Gluten")),
])
def test_el_registry_no_nombra_el_plato_al_alergico(pais, plato, alergias):
    assert plato in _registry(pais), (pais, plato)
    for alergia in alergias:
        assert plato not in _registry(pais, [alergia]), (pais, plato, alergia)


# ─────────────── 6 · P3 ───────────────

def test_cacahuate_saca_la_mezcla_del_catalogo():
    """«Nueces/Almendras» lleva maní en su alias: sale del pool con «Maní», «cacahuete» o «peanut», y con «cacahuate»
    (la grafía mexicana) también."""
    from constants import _get_fast_filtered_catalogs
    for decl in ("Maní", "cacahuete", "peanut", "cacahuate", "Cacahuates"):
        vivos = {str(x) for p in _get_fast_filtered_catalogs((decl,), (), "balanced") for x in p}
        assert "Nueces/Almendras" not in vivos, decl
        assert {"Almendras fileteadas", "Merey", "Pistachos"} <= vivos, decl


def test_termino_compartido_normaliza_los_prohibidos():
    import termino_compartido as tc
    base = {s for s in go._ALLERGEN_SYNONYMS["mani"]}
    prohibidos = {s.upper() for s in base} | {"MOLE"}
    assert tc.otra_clase_lo_prohibe("mole", prohibidos)
    assert tc.otra_clase_lo_prohibe("mole", {"Maní", *(s.title() for s in base)})


@pytest.mark.parametrize("chip", ["Mani", "Sesamo", "Frutos Secos", "Gluten"])
def test_la_pasta_de_chiles_para_mole_es_mole(chip):
    for ing in ("2 cdas de pasta de chiles para mole", "salsa de chiles para mole", "1 cda de pasta de chile para mole"):
        assert _viola(ing, [chip]), (ing, chip)
    assert not _viola("2 chiles anchos para mole", [chip]) and not _viola("15 g de chocolate para mole", [chip])


def test_anclas():
    for fichero, ancla in (("vocabulario_alergenos.py", "P1-PLAN-LOTE-796-ALIAS-AL-FINAL"),
                           ("prompts/day_generator.py", "P1-PLAN-LOTE-796-LINEA-DURA"),
                           ("excusas_vegetales.py", "P1-PLAN-LOTE-796-PREPARADO-VEGETAL"),
                           ("termino_compartido.py", "P1-PLAN-LOTE-796-EXCUSA-DE-SU-CLASE")):
        assert ancla in (_BACKEND / fichero).read_text(encoding="utf-8"), (fichero, ancla)
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "_re.finditer(_patron_termino_alergeno(f), ing_low)" in src
    assert 'excusa_de_su_clase(f, ing_low, _m_al.start(), _m_al.end(), forbidden)' in src
    assert 'excusa_contextual(term, ing_low, m.start(), m.end(), dieta=True)' in src
