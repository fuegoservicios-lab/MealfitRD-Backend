# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-68 · 2026-09-16] Los defectos que el dueño vio en el plato «Ñame Guisado…» del plan real del 16-sep.

(A) **La concordancia la manda lo que se VE.** La línea cruda «222.75 g de pechuga de pollo» salía como
    «1 pechugas de pollo»: el número se redondea a cuartos para mostrarlo (1,11 → «1») pero el singular/plural se
    decidía con el valor sin redondear (1,11 > 1 ⇒ plural), y de paso se perdía el «(porción)» que lleva la
    etiqueta singular.

(B) **Declinar el piso de IDENTIDAD no puede saltarse el suelo cocinable.** El plato «Ñame Guisado en Salsa
    Criolla…» se entregó con 5,27 g de ñame. El piso protagonista de carbos (60 g) se declina cuando no hay
    headroom kcal —correcto, para no oscilar contra el reconciliador de banda— pero su `continue` se llevaba por
    delante el suelo genérico de 15 g, así que la línea salía en 5 g. Medido sobre el plato real: con headroom
    sube a 60; sin headroom se quedaba en 5 y ahora queda en 15, sin dropearse nunca (la identidad del plato).
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ─────────────────────────────── (A) la concordancia sigue al número mostrado

def test_el_plural_sigue_al_numero_mostrado():
    from humanize_ingredients import humanize_ingredient
    assert humanize_ingredient("222.75 g de pechuga de pollo") == "1 pechuga de pollo (porción)"
    assert humanize_ingredient("200 g de pechuga de pollo") == "1 pechuga de pollo (porción)"


def test_por_encima_de_una_porcion_sigue_en_plural():
    from humanize_ingredients import humanize_ingredient
    for crudo, esperado in (("260 g de pechuga de pollo", "1¼ pechugas de pollo"),
                            ("300 g de pechuga de pollo", "1½ pechugas de pollo"),
                            ("450 g de pechuga de pollo", "2¼ pechugas de pollo")):
        assert humanize_ingredient(crudo) == esperado, crudo


def test_por_debajo_de_una_porcion_sigue_en_singular():
    from humanize_ingredients import humanize_ingredient
    assert humanize_ingredient("150 g de pechuga de pollo") == "¾ pechuga de pollo (porción)"


def test_el_ancla_del_arreglo_sigue_en_el_fuente():
    """Parser-based: si alguien vuelve a decidir el plural con el valor sin redondear, este test lo acusa."""
    src = _src(_BACKEND / "humanize_ingredients.py")
    assert "P1-PLAN-LOTE-68-DISPLAY-CONCORDANCIA" in src
    assert "units_mostradas = round(units * 4) / 4.0" in src
    assert 'measure["singular"] if units_mostradas <= 1.0' in src
    assert 'measure["singular"] if units <= 1.0' not in src, "volvió a decidir con el valor crudo"


# ─────────────────────────────── (B) el piso de identidad y el suelo cocinable

class _DbFalso:
    """Doble del `IngredientNutritionDB`: kcal por gramo fijo, para decidir headroom sin base."""

    def __init__(self, kcal_por_g=1.2):
        self.kcal_por_g = kcal_por_g

    def macros_from_ingredient_string(self, s):
        import re
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s))
        if not m:
            return None
        g = float(m.group(1).replace(",", "."))
        return {"kcal": g * self.kcal_por_g, "protein": 0.0, "carbs": g * 0.2, "fats": 0.0}


def _dia_con_name(gramos, kcal_comida=500):
    return {"day": 1, "meals": [{
        "meal": "Cena", "name": "Ñame Guisado en Salsa Criolla con Espinacas",
        "cals": kcal_comida, "protein": 30, "carbs": 40, "fats": 10,
        "ingredients": [f"{gramos} g de ñame pelado en cubos", "2 tazas de espinacas frescas",
                        "1 cdta de aceite de oliva", "100 g de pechuga de pollo"],
        "ingredients_raw": [f"{gramos} g de ñame pelado en cubos", "2 tazas de espinacas frescas",
                            "1 cdta de aceite de oliva", "100 g de pechuga de pollo"],
        "recipe": ["Mise en place: pela el ñame.", "El Toque de Fuego: cocina 20 minutos.", "Montaje: sirve."]}]}


def _gramos_de_name(dia):
    import re
    linea = [str(x) for x in dia["meals"][0]["ingredients"] if "ñame pelado" in str(x)]
    assert linea, "la línea del ñame NO puede desaparecer: es la identidad del plato"
    return float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", linea[0]).group(1).replace(",", "."))


def test_con_headroom_sube_al_piso_de_identidad():
    """El piso de IDENTIDAD (60 g) necesita el catálogo para saber que el ñame es el carbo del plato; sin base
    esta rama declina por diseño (P1-PROTAGONIST-CONTEXT-GATE) y la cubre el test de abajo. Medido con catálogo
    real sobre el plato del 16-sep: 5 g → 60 g."""
    from shopping_calculator import get_master_ingredients
    if not (get_master_ingredients() or []):
        import pytest as _pt
        _pt.skip("sin catálogo: el piso de identidad no puede clasificar el alimento (se mide en la pata con base)")
    from graph_orchestrator import _floor_subservible_portions
    d = _dia_con_name(5, kcal_comida=500)
    _floor_subservible_portions([d], day_kcal_target=2000, db=_DbFalso())
    assert _gramos_de_name(d) >= 60.0, d["meals"][0]["ingredients"]


def test_sin_headroom_cae_al_suelo_cocinable_y_no_se_dropea():
    """El defecto del lote: el `continue` de la rama declinada dejaba la línea en 5 g."""
    from graph_orchestrator import _floor_subservible_portions, PORTION_SHRINK_FLOOR_G
    d = _dia_con_name(5, kcal_comida=2000)          # la comida YA agota el objetivo del día
    _floor_subservible_portions([d], day_kcal_target=2000, db=_DbFalso())
    g = _gramos_de_name(d)
    assert g >= float(PORTION_SHRINK_FLOOR_G), f"quedó en {g} g: el piso declinado se comió el suelo"
    assert len(d["meals"][0]["ingredients"]) == 4, "la identidad del plato jamás se dropea"


def test_sin_contexto_kcal_tambien_cae_al_suelo():
    from graph_orchestrator import _floor_subservible_portions, PORTION_SHRINK_FLOOR_G

    class _DbMuda(_DbFalso):
        def macros_from_ingredient_string(self, s):
            return None

    d = _dia_con_name(5)
    _floor_subservible_portions([d], day_kcal_target=None, db=_DbMuda())
    assert _gramos_de_name(d) >= float(PORTION_SHRINK_FLOOR_G)


def test_una_linea_ya_servible_no_se_toca():
    from graph_orchestrator import _floor_subservible_portions
    d = _dia_con_name(120, kcal_comida=2000)
    _floor_subservible_portions([d], day_kcal_target=2000, db=_DbFalso())
    assert _gramos_de_name(d) == 120.0


def test_el_ancla_del_suelo_sigue_en_el_fuente():
    src = _src(_BACKEND / "graph_orchestrator.py")
    assert src.count("P1-PLAN-LOTE-68-IDENTIDAD-NO-SALTA-EL-SUELO") == 4,         "las cuatro ramas que declinan un piso protagonista deben caer al suelo cocinable"


# ─────────────────────────────── (C) lo que la lista compra, alguien tiene que cocinarlo

def _comida_con_name_crudo():
    return {"meal": "Cena", "name": "Ñame Guisado en Salsa Criolla",
            "ingredients": ["150 g de ñame pelado en cubos", "1 cdta de aceite de oliva"],
            "recipe": ["Mise en place: pela el ñame y córtalo en cubos.",
                       "El Toque de Fuego: calienta el aceite y sofríe la cebolla 4 minutos.",
                       "Montaje: sirve el guiso en un plato hondo."]}


def _catalogo():
    import json
    return json.loads((_BACKEND / "scripts" / "data" / "culinary_corpus_2026_09_12.json")
                      .read_text(encoding="utf-8"))["catalogo_filas"]


def test_el_helper_ve_lo_que_ve_v7f():
    import culinary_coherence as cc
    idx = cc.build_culinary_index(_catalogo())
    foods = cc.alimentos_sin_coccion(_comida_con_name_crudo(), idx)
    assert [f for f, _ in foods] == ["Ñame"], foods
    assert foods[0][1] == "viver"


def test_si_un_paso_lo_cuece_el_helper_calla():
    import culinary_coherence as cc
    idx = cc.build_culinary_index(_catalogo())
    m = _comida_con_name_crudo()
    m["recipe"][1] = "El Toque de Fuego: hierve el ñame 20 minutos hasta que esté tierno."
    assert cc.alimentos_sin_coccion(m, idx) == []


def test_el_reparador_inserta_el_paso_y_v7f_calla():
    import culinary_coherence as cc
    from graph_orchestrator import _auto_patch_uncooked_foods
    plan = {"days": [{"day": 1, "meals": [_comida_con_name_crudo()]}]}
    filas = _catalogo()
    assert [v["check"] for v in (cc.culinary_contract_scan(plan, filas) or []) if v["check"] == "V7f"] == ["V7f"]
    assert _auto_patch_uncooked_foods(plan, filas) == 1
    pasos = plan["days"][0]["meals"][0]["recipe"]
    assert any("cocínalo" in str(s) for s in pasos), pasos
    assert str(pasos[-1]).startswith("Montaje"), "el paso de cocción va ANTES del emplatado"
    assert [v for v in (cc.culinary_contract_scan(plan, filas) or []) if v["check"] == "V7f"] == []


def test_el_reparador_es_idempotente():
    from graph_orchestrator import _auto_patch_uncooked_foods
    plan = {"days": [{"day": 1, "meals": [_comida_con_name_crudo()]}]}
    filas = _catalogo()
    assert _auto_patch_uncooked_foods(plan, filas) == 1
    assert _auto_patch_uncooked_foods(plan, filas) == 0, "re-correrlo no puede volver a insertar"


def test_el_reparador_corre_antes_del_scan_que_mide():
    """El contrato del nodo es reparar → medir; si alguien lo mueve debajo del scan, el residuo medido mentiría."""
    src = _src(_BACKEND / "graph_orchestrator.py")
    i_rep = src.index("_auto_patch_uncooked_foods(plan, _gmi_uc()")    # [lote 220] + form_data=...
    i_scan = src.index("culinary_contract_scan_status(plan, _cul_cat")
    assert i_rep < i_scan, "el reparador quedó DESPUÉS del scan"


# ─────────────────────────────── (D) el ratchet: ningún `\b` convertido en retroceso

def test_ningun_fichero_del_backend_tiene_caracteres_de_control():
    """Escribir código con heredocs convirtió `\b` en 0x08 TRES veces el 16-sep (dos regex de producción y el
    doble de un test, que pasaba por la razón equivocada), y el escaneo destapó una CUARTA preexistente en
    `db_inventory.py`. Un backspace dentro de una regex no rompe nada ruidosamente: la hace no casar JAMÁS."""
    malos = []
    for carpeta in ("", "tests", "routers", "scripts", "prompts"):
        base = _BACKEND / carpeta if carpeta else _BACKEND
        if not base.is_dir():
            continue
        for f in base.glob("*.py"):
            for n, l in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if any(ord(c) < 32 and c != "	" for c in l):
                    malos.append(f"{f.relative_to(_BACKEND)}:{n}")
    assert not malos, f"caracteres de control (probable `\b` convertido en 0x08) en: {malos}"


def test_el_envase_que_declara_uds_vuelve_a_medirse():
    """La cuarta víctima del mismo `\b`, y preexistente: sin ella, «cartón 30 uds.» de Huevo se quedaba sin
    gramos y la deducción de la Nevera caía a otra rama."""
    import db_inventory
    maestro = {"density_g_per_unit": 50.0, "default_unit": "unidad",
               "market_packages": [{"unit": "carton", "units": 30, "label": "cartón 30 uds."}]}
    assert db_inventory._container_grams(maestro, "carton") == 1500.0

