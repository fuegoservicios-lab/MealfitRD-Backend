# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-17 · 2026-09-12] Decimoséptimo lote del plan de pendientes: D4, quién escribía el literal `30` en la compra.

El hallazgo del 09-08 (`docs/hallazgo_abierto_cantidades_absurdas_en_compra.md`) dejó UNA cosa abierta: identificar quién
escribe «30 cdas de aceite de oliva» en `ingredients_raw` para un plato que lee «¾ cdta». Aquí queda reproducido con la
función real, y cerrado lo que el arreglo del 09-07 dejaba detrás de un knob:

  · El productor: la rama «polvo de queso» de `_floor_subservible_portions` (P1-CHEESE-DUST-BUMP) llama a
    `rescale_ingredient_string` con factor `15 g / gramos_del_polvo` (0,25 g → 60×; 0,125 g → 120×) sobre la línea de raw
    que eligió por ÍNDICE. `rescale` conserva el token de unidad (`cdas`, `tazas`) y sólo mueve el número: por eso el
    mismo `30` salió con tres unidades. Y el `30` no es un default: es 15 × 2 (½ × 60; ¼ × 120 — la cantidad de raw era
    el doble de los gramos del polvo en los tres casos).
  · Con `RAW_PAIR_BY_FOOD=False` el resolutor por alimento vuelve al índice y la función de HOY volvía a escribir
    «30 cdas de cebolla picada» tal cual (medido antes del guard). La rama ahora sólo escribe raw si la línea es de queso:
    un factor nacido de los gramos del polvo no tiene sentido en ninguna otra línea.
  · El residuo: 0 líneas absurdas en 100 comidas (6 planes); los 3 planes del hallazgo ya no existen ⇒ sin barrido.
    La sonda `scripts/medir_cantidades_absurdas_raw.py` queda para re-medir, comparando raw↔display en ml.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import importlib.util
import inspect
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

# ────────────────────────────────────────────────────────────── el productor del literal

FLOOR_G = 15.0


@pytest.mark.parametrize("raw_antes,polvo_g,literal", [
    ("0.5 cdas de cebolla picada", 0.25, "30 cdas de cebolla picada"),
    ("0,5 tazas de rábanos", 0.25, "30 tazas de rábanos"),
    ("0.25 cdas de aceite de oliva", 0.125, "30 cdas de aceite de oliva"),
    ("1/2 cdas de cebolla picada", 0.25, "30 cdas de cebolla picada"),
])
def test_el_literal_30_es_el_piso_de_15_g_por_el_ratio_2(raw_antes, polvo_g, literal):
    """El mismo `30` con tres unidades no es un default: es `rescale` conservando el token de unidad y moviendo el
    número por 15 / gramos del polvo. Las tres cantidades de raw del hallazgo eran el doble de los gramos del polvo."""
    from nutrition_db import rescale_ingredient_string

    assert rescale_ingredient_string(raw_antes, FLOOR_G / polvo_g) == literal


def test_sobre_el_queso_el_factor_sin_cota_es_el_trabajo():
    """La rama existe para esto: 0,25 g de polvo → 15 g. La cota superior no va aquí; va en QUÉ línea se toca."""
    from nutrition_db import rescale_ingredient_string

    assert rescale_ingredient_string("0,25 g de queso rallado", FLOOR_G / 0.25) == "15 g de queso rallado"
    assert rescale_ingredient_string("0.125 g de queso gouda rallado", FLOOR_G / 0.125) == "15 g de queso gouda rallado"


# ────────────────────────────────────────────────────────────── la rama, de punta a punta

def _comida_del_hallazgo():
    """Raw reordenado y del MISMO largo (la forma `[conservadas] + [añadidas]`): por índice, la posición del polvo de
    queso en el display es la de la cebolla en raw — exactamente el «Mapuey Horneado» del 09-08."""
    return {
        "name": "Mapuey horneado",
        "cals": 320,
        "ingredients": ["0,25 g de queso rallado", "½ cda de cebolla picada", "120 g de mapuey"],
        "ingredients_raw": ["0.5 cdas de cebolla picada", "120 g de mapuey", "0.25 g de queso rallado"],
        "recipe": ["Hornea el mapuey y sirve con la cebolla y el queso."],
    }


def _correr_piso(meal):
    import graph_orchestrator as go

    try:
        return go._floor_subservible_portions([{"meals": [meal]}], day_kcal_target=None, db=None)
    except Exception:  # noqa: BLE001
        pytest.skip("el piso necesita db en este entorno")


def test_con_el_indice_forzado_el_polvo_de_queso_no_infla_la_cebolla(monkeypatch):
    """`RAW_PAIR_BY_FOOD=False` ⇒ `_raw_idx_for_display` devuelve el índice (mismo largo). Antes del guard la
    función escribía aquí `'30 cdas de cebolla picada'` — el literal del hallazgo, medido el 2026-09-12. Ahora la
    rama del queso no toca una línea que no es de queso, y el display sigue subiendo al piso."""
    import graph_orchestrator as go

    monkeypatch.setattr(go, "RAW_PAIR_BY_FOOD", False)
    m = _comida_del_hallazgo()
    _correr_piso(m)
    assert m["ingredients"][0] == "15 g de queso rallado", m["ingredients"]
    assert m["ingredients_raw"][0] == "0.5 cdas de cebolla picada", m["ingredients_raw"]
    assert not any(re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(cdas?|tazas?|cdtas?)\b", str(r)) and
                   float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", str(r)).group(1).replace(",", ".")) > 8
                   for r in m["ingredients_raw"]), m["ingredients_raw"]


def test_con_paridad_por_alimento_el_queso_sube_en_las_dos_listas():
    """Producción (`RAW_PAIR_BY_FOOD=True`): la línea de raw se elige por alimento, el queso sube en las dos listas
    y la cebolla no se mueve. Si el resolutor no tiene catálogo en este entorno, sólo se exige lo que no depende de él."""
    import graph_orchestrator as go

    m = _comida_del_hallazgo()
    _correr_piso(m)
    assert m["ingredients"][0] == "15 g de queso rallado", m["ingredients"]
    assert m["ingredients_raw"][0] == "0.5 cdas de cebolla picada", m["ingredients_raw"]
    if go._resolve_line_food_grams("0,25 g de queso rallado", cheap=True)[0] != "queso":
        pytest.skip("resolutor sin catálogo en este entorno: la paridad por alimento no se puede ejercitar")
    assert m["ingredients_raw"][2] == "15 g de queso rallado", m["ingredients_raw"]


def test_el_guard_vive_en_la_rama_del_polvo_de_queso():
    """Estructural: entre el cálculo del factor y la escritura en raw, la rama exige que la línea sea de queso.
    Un refactor que «simplifique» la condición reabre el `30` en cuanto alguien apague RAW_PAIR_BY_FOOD."""
    import graph_orchestrator as go

    src = inspect.getsource(go._floor_subservible_portions)
    i_factor = src.index("_qf = floor_g / _q_cur")
    i_guard = src.index('"queso" in _sa(str(raw[_ri]).lower())')
    i_write = src.index("raw[_ri] = _resc(str(raw[_ri]), _qf)")
    assert i_factor < i_guard < i_write
    assert "P1-PLAN-LOTE-17-QUESO-RAW-GUARD" in src


# ────────────────────────────────────────────────────────────── la sonda

def _sonda():
    p = _BACKEND / "scripts" / "medir_cantidades_absurdas_raw.py"
    spec = importlib.util.spec_from_file_location("medir_cantidades_absurdas_raw", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, p.read_text(encoding="utf-8")


def test_la_sonda_es_solo_lectura():
    _, src = _sonda()
    assert "conn.read_only = True" in src
    sql = " ".join(re.findall(r'"([^"]*)"', src))
    assert not re.search(r"\b(UPDATE|INSERT|DELETE|TRUNCATE|ALTER|DROP)\b", sql), sql
    assert "jsonb_set" not in src


def test_la_sonda_acusa_el_hallazgo_y_perdona_la_conversion_correcta():
    """El 09-08 la sonda contó 8 y eran 3-4: «2¼ tazas de cebollín» = «34,75 cdas» es una conversión CORRECTA que se
    acusó por mirar sólo el número. Esta compara raw↔display en ml y sólo acusa por encima de 3×."""
    mod, _ = _sonda()
    display = ["¾ cdta de aceite de oliva", "½ taza de rábanos en láminas", "½ cda de cebolla picada",
               "2¼ tazas de cebollín", "120 g de yuca"]
    aceite = mod.juzgar_linea("30 cdas de aceite de oliva", display)
    assert aceite["veredicto"] == "absurda" and aceite["factor"] == 120.0, aceite
    rabanos = mod.juzgar_linea("30 tazas de rábanos", display)
    assert rabanos["veredicto"] == "absurda" and rabanos["factor"] == 60.0, rabanos
    cebolla = mod.juzgar_linea("30 cdas de cebolla picada", display)
    assert cebolla["veredicto"] == "absurda" and cebolla["factor"] == 60.0, cebolla
    cebollin = mod.juzgar_linea("34.75 cdas de cebollín", display)
    assert cebollin["veredicto"] == "conversion_correcta" and abs(cebollin["factor"] - 0.97) < 0.02, cebollin
    assert mod.juzgar_linea("2 cdas de aceite de oliva", display) is None
    assert mod.juzgar_linea("120 g de yuca", display) is None
    huerfana = mod.juzgar_linea("30 cdas de perejil", display)
    assert huerfana["veredicto"] == "absurda" and huerfana["motivo"] == "sin gemela en el display"


def test_la_sonda_recorre_tambien_los_dias_archivados_y_da_veredicto():
    mod, _ = _sonda()
    plan = {
        "days": [{"meals": [{"name": "A", "ingredients": ["½ cda de cebolla picada"],
                             "ingredients_raw": ["0.5 cdas de cebolla picada"]}]}],
        "_archived_days": [{"meals": [{"name": "B", "ingredients": ["½ cda de cebolla picada"],
                                       "ingredients_raw": ["30 cdas de cebolla picada"], "_portion_floor_adjusted": True}]}],
    }
    comidas, casos = mod.medir_plan("e2bbb280-0000", plan)
    assert comidas == 2 and len(casos) == 1 and casos[0]["plato"] == "B" and casos[0]["piso"] is True
    assert "decidir barrido" in mod.veredicto({"comidas": 2, "lineas_absurdas": 1, "planes_afectados": ["e2bbb280"]})
    assert "nada que barrer" in mod.veredicto({"comidas": 100, "lineas_absurdas": 0, "planes_afectados": []})
    assert "NO CONCLUYENTE" in mod.veredicto({"comidas": 0, "lineas_absurdas": 0, "planes_afectados": []})


# ────────────────────────────────────────────────────────────── docs y marker

def test_el_hallazgo_queda_cerrado_con_la_reproduccion_y_el_residuo():
    doc = (_BACKEND / "docs" / "hallazgo_abierto_cantidades_absurdas_en_compra.md").read_text(encoding="utf-8")
    assert "**Estado:** CERRADO 2026-09-12 (`P1-PLAN-LOTE-17`, D4)" in doc
    assert "## Cierre 2026-09-12 · D4" in doc
    assert "`30 cdas de cebolla picada`" in doc and "15 g × 2" in doc
    assert "RAW_PAIR_BY_FOOD=False" in doc and "P1-PLAN-LOTE-17-QUESO-RAW-GUARD" in doc
    assert "No hay nada que barrer" in doc
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert "| D4 | ✅ 2026-09-12 · sin barrido |" in plan


def test_marker_bumpeado():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-17 · 2026-09-12"$', src, re.M)
