# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-796 · 2026-09-28] Revisión del lote: los huecos que quedaban dentro del mismo tema.

  · «frutos de cáscara» —el nombre LEGAL de la clase en España (Reglamento UE 1169/2011), el de cada etiqueta— no
    declaraba nada: ni pool, ni backstop, ni prompt, ni registry, ni catálogo. Lo mismo «APLV»/«PLV» para lácteos.
  · Al alérgico al MANÍ se le ofrecían mezclas: «10 g de frutos secos mixtos sin sal» es línea real de la batería y
    alias de la fila «Nueces mixtas», que se dio de alta como «mixed nuts, dry roasted, with peanuts». El prompt se la
    sugería en la lista «OK siempre» y en la línea dura.
  · «tortilla francesa» (huevo), préstamos de trigo que se escriben así en DO/MX/CO/PR («waffles», «spaghetti»,
    «pancakes», «hot cakes»), el mole (maní, ajonjolí, almendra y pan/galleta en la pasta), «manchego» marcando el pisto.
  · El filtro del catálogo sólo disparaba sus catch-alls con el chip EXACTO: «celiaquía», «leche», «camarones»,
    «atún» o «almendras» escritos a mano dejaban en el pool lo que el escáner prohíbe.
tooltip-anchor: P1-PLAN-LOTE-796-REVISION
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

PAISES = ("DO", "ES", "US", "MX", "PR", "CO")


def _plan(*ings):
    return {"days": [{"day": 1, "meals": [{"meal": "Merienda", "name": "Plato", "ingredients": list(ings)}]}]}


def _viola(ing, alergias):
    return bool(go._scan_allergen_violations(_plan(ing), list(alergias)))


def _clases(decl):
    from constants import strip_accents
    exp = go._expand_allergy_declarations([decl])
    return {c for c, syns in go._ALLERGEN_SYNONYMS.items()
            if syns and {strip_accents(str(s)).lower() for s in syns} <= exp}


def _esqueleto():
    return {"brief_concept": "Día variado", "assigned_technique": "A la plancha",
            "protein_pool": ["Pollo"], "carb_pool": ["Avena", "Batata", "Yuca"], "fruit_pool": ["Mango"],
            "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"]}


def _sugerencias(ctx):
    ok = re.search(r"Para diversificar desayuno/merienda usa: (.+?) \(estas son OK siempre", ctx).group(1)
    alts = re.search(r"la merienda NO lleva avena \(usa (.+?)\)", ctx).group(1)
    return [x.strip() for x in re.split(r",| o ", ok + "," + alts) if x.strip()]


def _alternativas(linea):
    return linea.split("Sin lácteos:", 1)[1]


# ─────────────── P1-1 · «frutos de cáscara» (y «APLV») declaran su clase ───────────────

DECL_CASCARA = ("frutos de cáscara", "Frutos de cascara", "fruto de cáscara", "Alergia a los frutos de cáscara",
                "frutos de casca rija")


@pytest.mark.parametrize("decl", DECL_CASCARA)
def test_frutos_de_cascara_declara_la_clase(decl):
    assert _clases(decl) == {"frutos secos"}, (decl, _clases(decl))
    assert go._allergen_pool_item_banned("Almendras", [decl])
    assert go._allergen_pool_item_banned("frutos secos", [decl])
    for ing in ("30 g de almendras", "20 g de nueces", "10 g de frutos secos mixtos sin sal", "15 g de alfóncigos"):
        assert _viola(ing, [decl]), (decl, ing)
    comida = {"name": "Merienda", "ingredients": ["30 g de almendras", "20 g de nueces"]}
    assert go.clinical_backstop_for_meal(comida, allergies=[decl])


def test_frutos_de_cascara_en_el_prompt():
    from prompts.day_generator import allergy_hard_line, build_day_assignment_context
    alts = _alternativas(allergy_hard_line(["lácteos", "frutos de cáscara"]))
    assert "frutos secos" not in alts and "almendras" not in alts and "casabe con aguacate" in alts, alts
    sug = _sugerencias(build_day_assignment_context(_esqueleto(), 1, allergies=["frutos de cáscara"]))
    assert "frutos secos" not in sug and "almendras" not in sug, sug


@pytest.mark.parametrize("decl", ["APLV", "aplv", "PLV", "CMPA", "alergia a la proteína de la leche de vaca"])
def test_aplv_declara_lacteos(decl):
    assert "lacteos" in _clases(decl), (decl, _clases(decl))
    for ing in ("1 taza de leche", "30 g de queso", "1 yogurt griego"):
        assert _viola(ing, [decl]), (decl, ing)
    assert go.clinical_backstop_for_meal({"name": "D", "ingredients": ["1 taza de leche", "30 g de queso"]},
                                         allergies=[decl])


# ─────────────── P1-2 · la mezcla de frutos secos lleva maní ───────────────

MEZCLAS = ("10 g de frutos secos mixtos sin sal", "20 g de frutos secos", "1 puñado de fruto seco", "Nueces mixtas",
           "30 g de nueces mixtas", "nueces surtidas", "mixed nuts", "1 mix de frutos secos",
           "cóctel de frutos secos")


@pytest.mark.parametrize("decl", ["Mani", "maní", "cacahuete", "peanut", "arachide"])
@pytest.mark.parametrize("ing", MEZCLAS)
def test_al_alergico_al_mani_la_mezcla_de_frutos_secos_le_viola(decl, ing):
    assert _viola(ing, [decl]), (decl, ing)


@pytest.mark.parametrize("ing", ["20 g de almendras", "15 g de nueces", "merey", "20 g de pistachos", "avellanas"])
def test_al_alergico_al_mani_el_fruto_seco_suelto_si_se_le_sirve(ing):
    assert not _viola(ing, ["Mani"]), ing


def test_el_pool_y_el_prompt_no_le_ofrecen_mezclas_al_alergico_al_mani():
    from prompts.day_generator import allergy_hard_line, build_day_assignment_context
    assert go._allergen_pool_item_banned("frutos secos", ["Mani"])
    alts = _alternativas(allergy_hard_line(["Lacteos", "Maní"]))
    assert "frutos secos" not in alts and "almendras con fruta" in alts, alts
    sug = _sugerencias(build_day_assignment_context(_esqueleto(), 1, allergies=["Mani"]))
    assert "frutos secos" not in sug and "almendras" in sug, sug


def test_sin_esa_alergia_el_prompt_no_cambia():
    """El relevo sólo aparece cuando la sugerencia genérica cae: sin alergia al maní el texto es el de siempre."""
    from prompts.day_generator import allergy_hard_line, build_day_assignment_context
    alts = _alternativas(allergy_hard_line(["Lacteos"]))
    assert "frutos secos con fruta" in alts and "almendras con fruta" not in alts, alts
    sug = _sugerencias(build_day_assignment_context(_esqueleto(), 1))
    assert "frutos secos" in sug and "almendras" not in sug, sug


# ─────────────── P2-3 · la tortilla francesa es de huevo ───────────────

@pytest.mark.parametrize("ing", ["1 tortilla francesa", "2 tortillas francesas", "tortilla francesa de 2 huevos"])
def test_tortilla_francesa_es_huevo(ing):
    assert _viola(ing, ["Huevo"]), ing


def test_tortilla_francesa_no_arrastra_clases():
    assert not _viola("1 tortilla francesa", ["Gluten"])
    assert _clases("tortilla") == {"gluten"} and _clases("francesa") == set()


# ─────────────── P2-4 · préstamos de trigo que se escriben así en español ───────────────

@pytest.mark.parametrize("ing", ["2 waffles", "1 waffle integral", "1 plato de spaghetti", "3 pancakes",
                                 "2 hot cakes", "1 hot cake", "2 hot-cakes", "3 panquecas"])
def test_prestamos_de_trigo(ing):
    assert _viola(ing, ["Gluten"]), ing


def test_el_prestamo_sin_gluten_sigue_excusado():
    assert not _viola("2 waffles sin gluten", ["Gluten"])


# ─────────────── P2-5 · el catálogo resuelve la ALERGIA escrita a mano por su clase ───────────────

DECL_A_MANO = ("celiaquía", "celíaco", "intolerancia a la lactosa", "leche", "milk", "camarones", "crustáceos",
               "moluscos", "shellfish", "atún", "almendras", "frutos de cáscara", "APLV", "maní", "cacahuete",
               "ajonjolí", "soja", "huevos", "trigo")


@pytest.mark.parametrize("pais", PAISES)
@pytest.mark.parametrize("decl", DECL_A_MANO)
def test_el_catalogo_no_ofrece_lo_que_el_escaner_prohibe_a_la_alergia_escrita(decl, pais):
    from constants import _get_fast_filtered_catalogs
    pools = _get_fast_filtered_catalogs((decl,), (), "balanced", country=pais, market_extras=True,
                                        culture_country=pais)
    malos = [x for pool in pools for x in pool if _viola(str(x), [decl])]
    assert not malos, (decl, pais, malos)


@pytest.mark.parametrize("decl", ["Mani", "maní", "cacahuete"])
def test_al_alergico_al_mani_el_catalogo_le_deja_los_frutos_secos_sueltos(decl):
    """Un TÉRMINO de la alergia no dispara el catch-all de otra clase: «frutos secos» (la mezcla, término del maní) no
    puede quitarle al alérgico al maní las almendras, el merey ni los pistachos."""
    from constants import _get_fast_filtered_catalogs
    vivos = [x for p in _get_fast_filtered_catalogs((decl,), (), "balanced") for x in p]
    assert {"Almendras fileteadas", "Merey", "Pistachos", "Granola"} <= set(vivos), decl


def test_el_disgusto_no_se_expande_por_clase():
    """Quien no quiere camarones puede querer pulpo: sólo la ALERGIA se resuelve a su clase."""
    from constants import _get_fast_filtered_catalogs
    disgusto = [x for p in _get_fast_filtered_catalogs((), ("camarones",), "balanced") for x in p]
    alergia = [x for p in _get_fast_filtered_catalogs(("camarones",), (), "balanced") for x in p]
    assert "Pulpo" in disgusto and "Pulpo" not in alergia


def test_el_catalogo_sin_alergia_no_cambia():
    from constants import _get_fast_filtered_catalogs
    for pais in PAISES:
        a = _get_fast_filtered_catalogs((), (), "balanced", country=pais, market_extras=True, culture_country=pais)
        b = _get_fast_filtered_catalogs(("Ninguna",), (), "balanced", country=pais, market_extras=True,
                                        culture_country=pais)
        assert a == b, pais


# ─────────────── P2-6 · el mole (pasta con maní, ajonjolí, almendra y pan) ───────────────

@pytest.mark.parametrize("chip", ["Mani", "Sesamo", "Frutos Secos", "Gluten"])
@pytest.mark.parametrize("ing", ["2 cdas de pasta de mole", "1/2 taza de salsa de mole", "mole poblano",
                                 "3 cdas de mole", "mole negro", "pollo en mole rojo"])
def test_el_mole_lleva_los_alergenos_de_su_pasta(ing, chip):
    assert _viola(ing, [chip]), (ing, chip)


@pytest.mark.parametrize("chip", ["Mani", "Sesamo", "Frutos Secos", "Gluten"])
def test_el_mole_de_olla_no(chip):
    assert not _viola("1 plato de mole de olla", [chip])
    assert not _viola("1/2 taza de guacamole", [chip])
    # el chocolate o el chile PARA el mole son ingredientes suyos (alias de «Chocolate de mesa»), no la pasta
    assert not _viola("15 g de chocolate para mole", [chip]) and not _viola("2 chiles anchos para mole", [chip])
    assert _viola("2 cdas de pasta para mole", [chip]) and _viola("chocolate y pasta para mole", [chip])


@pytest.mark.parametrize("ing,chip", [("2 cdas de salsa macha", "Mani"), ("2 cdas de salsa macha", "Sesamo"),
                                      ("pollo encacahuatado", "Mani"), ("150 g de pollo encacahuatado", "Mani")])
def test_salsas_de_cacahuate(ing, chip):
    assert _viola(ing, [chip]), (ing, chip)


@pytest.mark.parametrize("decl", ["Mani", "Sesamo", "Frutos Secos", "Gluten", "maní", "ajonjolí"])
def test_el_registry_no_nombra_el_mole_al_alergico(decl):
    import dish_registry as dr
    nombres = {c.get("name") for c in dr.template_candidates("MX", "almuerzo", None, k=500,
                                                              exclude_allergens=[decl])}
    assert "Mole ligero de pollo con arroz" not in nombres, decl
    assert "Mole ligero de pollo con arroz" in {c.get("name") for c in dr.template_candidates(
        "MX", "almuerzo", None, k=500)}


# ─────────────── P3-7 · «pisto manchego» no lleva queso ───────────────

def test_el_pisto_manchego_no_es_lacteo():
    for chip in ("Lacteos", "Lactosa"):
        assert not _viola("1 taza de pisto manchego", [chip]), chip
        assert _viola("30 g de queso manchego", [chip]) and _viola("30 g de manchego", [chip]), chip
        assert _viola("pisto manchego con queso", [chip]), chip
    assert not go._scan_diet_violations(_plan("1 taza de pisto manchego"), "vegan")
    assert go._scan_diet_violations(_plan("30 g de manchego"), "vegan")


# ─────────────── P3-8 · los demás nombres que el escáner no veía ───────────────

@pytest.mark.parametrize("ing,chip", [
    ("1 cda de za'atar", "Sesamo"), ("1 cda de za’atar", "Sesamo"), ("20 g de manises", "Mani"),
    ("1 porción de quesillo", "Lacteos"), ("1 porción de quesillo", "Huevo"), ("1 vaso de kumis", "Lacteos"),
    ("kumis", "Lactosa"), ("1 capuchino", "Lacteos"), ("1 cappuccino", "Lactosa"), ("1 café latte", "Lactosa"),
    ("salsa bechamel", "Lacteos"), ("salsa bechamel", "Gluten"), ("4 croquetas de pollo", "Gluten"),
    ("4 croquetas de jamón", "Huevo"), ("4 croquetas de jamón", "Lacteos"), ("1 milanesa de res", "Gluten"),
    ("1 milanesa de pollo", "Huevo"), ("pollo empanado", "Gluten"), ("pescado apanado", "Gluten"),
    ("calamares rebozados", "Gluten"), ("calamares rebozados", "Huevo"), ("pechuga empanizada", "Gluten"),
    ("3 pastelitos de carne", "Gluten"), ("1 porción de pizza", "Gluten"), ("1 porción de pizza", "Lacteos"),
    ("30 g de granola", "Frutos Secos"),
])
def test_nombres_que_el_escaner_no_veia(ing, chip):
    assert _viola(ing, [chip]), (ing, chip)


@pytest.mark.parametrize("ing,chip", [
    ("1 latte de avena", "Lacteos"), ("2 pastelitos de yuca", "Gluten"), ("1 taza de chocolate caliente", "Lacteos"),
    ("30 g de granola", "Mani"), ("1 taza de leche de coco", "Lacteos"), ("1 tortilla de maíz", "Huevo"),
    ("2 cdas de salsa macha", "Mariscos"),  # la macha es una almeja chilena; la salsa macha, aceite de chile
])
def test_sin_falsos_positivos_nuevos(ing, chip):
    assert not _viola(ing, [chip]), (ing, chip)


def test_la_almeja_macha_sigue_siendo_marisco():
    assert _viola("200 g de machas a la parmesana", ["Mariscos"])


def test_la_dieta_vegana_ve_los_lacteos_nuevos():
    assert go._scan_diet_violations(_plan("salsa bechamel"), "vegan")
    assert go._scan_diet_violations(_plan("1 capuchino"), "vegan")
    assert not go._scan_diet_violations(_plan("1 latte de avena"), "vegan")


@pytest.mark.parametrize("decl,esperado", [
    ("frutos secos", {"frutos secos"}), ("maní", {"mani"}), ("mole", set()),
    ("granola", {"gluten"}), ("croquetas", set()), ("pizza", set()), ("bechamel", {"lacteos", "lactosa"}),
    ("quesillo", {"lacteos", "lactosa"}), ("latte", {"lacteos", "lactosa"}), ("waffles", {"gluten"}),
])
def test_los_terminos_nuevos_no_arrastran_otra_clase(decl, esperado):
    assert _clases(decl) == esperado, (decl, _clases(decl))


# ─────────────── «… sin gluten» sólo absuelve al gluten ───────────────

@pytest.mark.parametrize("ing,chip", [
    ("2 cdas de mole sin gluten", "Mani"), ("2 cdas de mole sin gluten", "Sesamo"),
    ("15 g de granola sin gluten", "Frutos Secos"), ("1 biscuit sin gluten", "Lacteos"),
    ("1 porción de pizza sin gluten", "Lacteos"), ("4 croquetas sin gluten", "Huevo"),
    ("salsa bechamel sin gluten", "Lactosa"), ("pollo empanizado sin gluten", "Huevo"),
])
def test_la_excusa_sin_gluten_no_absuelve_a_otra_clase(ing, chip):
    """La excusa FORWARD de gluten («avena certificada sin gluten») era ciega a la clase: un término que es también de
    otra clase (el mole del maní, la granola de los frutos secos, el biscuit de los lácteos) quedaba absuelto para esa
    otra alergia. La declaración «sin gluten» sólo habla del gluten."""
    assert _viola(ing, [chip]), (ing, chip)
    assert _viola(ing, [chip, "Gluten"]), (ing, chip)


@pytest.mark.parametrize("ing", ["15 g de granola sin gluten", "1 porción de pizza sin gluten", "2 waffles sin gluten",
                                 "1 cda de salsa de soya sin gluten", "4 croquetas sin gluten"])
def test_la_excusa_sin_gluten_sigue_valiendo_para_el_celiaco(ing):
    assert not _viola(ing, ["Gluten"]), ing
