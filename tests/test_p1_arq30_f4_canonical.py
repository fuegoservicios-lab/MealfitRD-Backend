# -*- coding: utf-8 -*-
"""[P1-ARQ30-F4-CANONICAL · 2026-09-06] `IngredientLine`: compras y macros, NO el texto (ARQ30-P1-01).

`ARQ30-P1-01` propone una autoridad única sobre la composición de la que se deriven **texto y
compras**. Antes de mover esa autoridad se midió si la representación aguanta lo que el motor escribe
hoy, sobre **11.073 líneas de 96 planes vivos**:

    roundtrip exacto ....... 0,5 %   (50)
    roundtrip equivalente .. 17,8 %  (1.973)

Y lo que se pierde no es formato:

    3 huevos                  →  3 unidad de Huevo
    45 g de melón en cubos    →  45 g de Melón        ← «en cubos» desaparece
    3 dientes de ajo          →  3 diente de Ajo
    1¼ cucharadas de cilantro →  1.25 cda de Cilantro

El CORTE es información culinaria de la que dependen los pasos. **Decisión del dueño (2026-09-06): la
representación sirve para COMPRAS y MACROS; el texto sigue siendo del LLM.** En ese alcance sí
aguanta:

    resuelve al catálogo .... 99,5 %  (los 60 que no: agua, hielo — no son alimentos)
    gramos derivables ....... 82,4 %
    cantidad conservada ..... 96,3 %

**Los gramos subieron del 19,9 % al 82,4 % sin escribir una regla nueva**, solo preguntando a
`nutrition_db.to_grams` en vez de parar en la conversión de unidades. Una taza de espinacas necesita
la densidad del catálogo, y esa conversión ya tenía dueño — que además lleva dentro la lección de
`P1-UNKNOWN-UNIT-NOT-WHOLE` (una unidad desconocida no es una unidad entera: para una hierba eso es
el mazo).

Esto es el `expand` de una migración expand → canary → promoción. **Nada de producción escribe a
través de esta representación todavía.**
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

from canonical_recipe import (  # noqa: E402
    ESTADO_DESCONOCIDO, IngredientLine, parse_line, render_line, shopping_view,
)

_SRC = (_BACKEND / "canonical_recipe.py").read_text(encoding="utf-8")


# ── el alcance, escrito ───────────────────────────────────────────────────────────────────────
def test_el_alcance_esta_declarado_en_el_modulo():
    """Sin esto escrito, alguien «completaría» el roundtrip dentro de seis meses creyendo que es una
    tarea pendiente — y degradaría todas las recetas."""
    assert "COMPRAS y MACROS" in _SRC
    assert "0,5 %" in _SRC and "17,8 %" in _SRC, "la cifra que motivó el alcance vive con el código"


def test_render_line_se_declara_de_diagnostico():
    i = _SRC.index("def render_line(")
    cuerpo = _SRC[i:i + 1200]
    assert "No es la línea del usuario" in cuerpo
    assert "Nada de producción debe escribir con esto" in cuerpo


def test_nadie_en_produccion_escribe_con_esta_representacion():
    """El `expand` de la migración: la vía anterior sigue siendo la única que entrega.

    Se buscan IMPORTS en ficheros `.py`, no menciones. La primera versión usaba `git grep` sobre
    cualquier aparición del nombre y se rompió sola en cuanto se commiteó el doc que la describe —
    además de depender del ÍNDICE de git: mientras el fichero estaba sin rastrear, el guard no lo
    veía. Un centinela cuyo veredicto depende de si algo está commiteado no mide el código.
    """
    import re

    rx = re.compile(r"^\s*(?:from\s+canonical_recipe\s+import|import\s+canonical_recipe)", re.M)
    tocan = set()
    for _p in _BACKEND.rglob("*.py"):
        rel = _p.relative_to(_BACKEND).as_posix()
        if rel.startswith(("tests/", "venv", "scripts/")) or "__pycache__" in rel:
            continue
        try:
            if rx.search(_p.read_text(encoding="utf-8")):
                tocan.add(rel)
        except Exception:
            continue
    assert not tocan, f"producción ya importa la representación canónica: {tocan}"


def test_el_guard_del_expand_sabe_buscar():
    """Centinela del centinela: si el patrón dejara de encontrar el import que SÍ existe (el de la
    sonda), el guard de arriba pasaría en vacío."""
    import re

    rx = re.compile(r"^\s*(?:from\s+canonical_recipe\s+import|import\s+canonical_recipe)", re.M)
    sonda = (_BACKEND / "scripts" / "canonical_roundtrip.py").read_text(encoding="utf-8")
    assert rx.search(sonda), "el patrón ya no reconoce un import real: el guard sería vacuo"


# ── lo que la representación SÍ tiene que hacer ───────────────────────────────────────────────
@pytest.mark.parametrize("raw,raiz,qty", [
    ("55 g de queso", "queso", 55.0),
    ("35 g de soya texturizada cocida", "soya", 35.0),
    ("280 ml de leche descremada", "leche", 280.0),
])
def test_la_cantidad_y_la_identidad_sobreviven(raw, raiz, qty):
    """La CANTIDAD es exacta —un plan es una promesa de cantidades— y la identidad conserva la raíz
    del alimento. El id completo no se fija aquí a propósito: con el catálogo accesible el parser
    canonicaliza («queso» → `queso_blanco`, «leche descremada» → `leche_descremada`) y sin él se
    queda en el literal. Anclar el id exacto ataría el test a si la DB responde."""
    l = parse_line(raw)
    assert l.qty == qty
    assert l.ingredient_id.startswith(raiz), l.ingredient_id


def test_la_pista_de_gramos_no_se_queda_en_el_nombre():
    """«1 batata mediana cocida, 250 g» daba el id `batata_250_g`: dos líneas del mismo alimento con
    pistas distintas producían identidades distintas, y la lista de compras las contaría aparte."""
    assert parse_line("1 batata mediana cocida, 250 g").ingredient_id == "batata"
    # «½ pedazo de ñame (≈150 g)» resuelve a Ñame, y su id sale `name` porque `strip_accents`
    # convierte la ñ en n. Es una colisión latente con cualquier alimento inglés llamado «name»; no
    # hay ninguno hoy en el catálogo y no se toca aquí, pero queda anotado.
    _name = parse_line("½ pedazo de ñame (≈150 g)")
    assert _name is not None and "150" not in _name.ingredient_id


@pytest.mark.parametrize("raw,estado", [
    ("35 g de soya texturizada cocida", "cocido"),
    ("45 g de pasta integral seca", "seco"),
    ("100 g de garbanzos escurridos", "escurrido"),
    ("55 g de queso", ESTADO_DESCONOCIDO),
])
def test_el_estado_sale_del_texto_y_ausente_no_es_crudo(raw, estado):
    """`desconocido` NO es `crudo`. El rendimiento cocido↔seco es 0,35× para legumbres: asumir crudo
    donde el texto calla triplicaría la compra."""
    assert parse_line(raw).state == estado


def test_una_linea_sin_cantidad_no_se_inventa():
    """Tratar «Sal al gusto» como cero gramos sería inventar un dato."""
    l = parse_line("Sal al gusto")
    assert l is None or l.grams is None


def test_sin_catalogo_los_gramos_son_None_no_cero():
    """La invariante I20 otra vez: ausente ≠ cero. Sin `nutrition_db`, una taza no da gramos."""
    l = parse_line("2 tazas de espinacas")
    assert l is not None and l.grams is None
    assert shopping_view(l)["grams"] is None


def test_los_gramos_declaran_quien_los_dio():
    """Cuando ARQ30-P1-04 unifique la autoridad de cantidades, cambia el proveedor y no el contrato —
    y `grams_source` es lo que permite ver desde fuera cuál respondió."""
    l = parse_line("55 g de queso")
    assert l.grams == 55.0 and l.grams_source == "canonical_units.to_base_amount"


def test_shopping_view_es_el_contrato_del_alcance():
    v = shopping_view(parse_line("55 g de queso"))
    assert set(v) == {"ingredient_id", "name", "qty", "unit", "grams", "state", "grams_source"}
    assert "raw" not in v, "el texto crudo no es asunto de compras ni de macros"


# ── no se reimplementa lo que ya tiene dueño ──────────────────────────────────────────────────
@pytest.mark.parametrize("autoridad", [
    "_parse_quantity",                 # cantidad/unidad/nombre
    "_reconcile_qty_with_gram_hint",   # la pista «, 250 g»
    "canonicalize_unit",               # unidad canónica
    "nutrition_db.to_grams",           # densidad del catálogo
    "ingredient_id_for",               # identidad
])
def test_compone_las_autoridades_que_ya_existen(autoridad):
    """Un quinto parser aquí sería la deriva que este repo ya pagó con `canonicalize_diet_type` y
    con `pantry_names_match`."""
    assert autoridad.split(".")[-1] in _SRC, autoridad


def test_no_nace_una_tabla_de_densidades_ni_de_unidades():
    for inventada in ("DENSIDADES", "GRAMOS_POR_TAZA", "_UNIT_TO_G", "PESOS_POR_UNIDAD"):
        assert inventada not in _SRC, f"{inventada} duplica a canonical_units/nutrition_db"


def test_el_render_no_duplica_el_estado_que_ya_esta_en_el_nombre():
    """`_parse_quantity` deja el calificativo dentro del nombre, así que anexarlo producía «pasta
    integral seca seco» — el render inventaba una palabra que el original no traía."""
    salida = render_line(parse_line("45 g de pasta integral seca"))
    assert salida.lower().count("sec") == 1, salida


def test_render_de_basura_no_revienta():
    assert render_line(None) == ""
    assert render_line(IngredientLine(raw="", name="Queso", ingredient_id="queso")) == "Queso"
