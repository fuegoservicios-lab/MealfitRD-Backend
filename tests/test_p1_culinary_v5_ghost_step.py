# -*- coding: utf-8 -*-
"""[P1-CULINARY-V5-GHOST-STEP · 2026-09-06] V5: el paso usa algo que la lista no trae.

V3 pregunta «¿hay un ingrediente que ningún paso menciona?». **Nadie preguntaba lo contrario**, y es
la categoría más frecuente del juez culinario: de 227 comidas que señala, 96 son `paso_incoherente`.
El daño es directo — el usuario compra la lista y la receta le manda usar algo que no tiene:

    «Montaje: … coloca el cilantro por encima»      lista: orégano, ajo, cebolla… sin cilantro
    «Montaje: … añade la piña»                       lista: cottage, manzana, semillas de calabaza
    «Mise en place: … pela y trocea el plátano»      lista: yogurt, lechosa, fresas, leche

## Ocho rondas contra 1.186 comidas vivas

El detector ingenuo daba **460** acusaciones. Cada filtro nació de un falso positivo medido, no de
una hipótesis:

    460 → 364  el índice devuelve el alias corto Y el largo («yogurt griego» casaba también `Yogur`)
    364 → 287  las notas de seguridad hablan de CLASES en abstracto («el pollo/cerdo debe cocinarse»)
    287 → 241  lista y paso nombran el mismo alimento con alias distintos
    241 → 100  el índice no resuelve «1½ filetes de pescado», y eso NO significa que no esté
    100 →  21  «chuleta de cerdo» cuando la lista dice «chuleta»: el paso es más específico
     21 →  11  un paso que USA lo nombra tras un verbo de entrada; uno que lo PRODUCE, no

Juzgadas a mano las 11: **10 reales, 1 falso** (`ají morrón`, cuya lista dice «0.5 ají»).

## Las dos lecciones que costaron más

**El cuarto filtro.** Sin él, el detector medía el recall del CATÁLOGO y acusaba al plan de su propia
ceguera: un ceviche con «1½ filetes de pescado» en la lista salía acusado de no llevar pescado porque
el índice no resolvía esa línea. *Un detector que confunde «no lo encuentro» con «no está» es inútil.*

**Los tres intentos de matar el último 30 %.** Bajé el umbral de palabra a 3 letras, ensanché la
ventana y luego la estreché a las 3 palabras previas. Las tres veces el detector cayó a **CERO**,
llevándose los hallazgos reales — porque «con», «las» o el ingrediente vecino están en toda lista.
*Un filtro que descarta todo no es preciso, es ciego, y se parece muchísimo a uno que funciona si
solo miras el número.* Lo que salvó la ronda fue tener hallazgos ya juzgados a mano con los que
comparar.

## Por qué `warn` y no `block`

`severity='minor'`, `repairable=False`: es telemetría. Escalar a bloqueo con un 91 % de precisión
castigaría un plan de cada once sin motivo — y calibrarlo contra la tasa del propio juez LLM sería el
overfitting que este repo ya pagó en agosto. El siguiente paso es un golden set humano.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

from culinary_coherence import (  # noqa: E402
    build_culinary_index, culinary_contract_scan, _v5_paso_usa_lo_que_no_esta,
)

_SRC = (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")

_CAT = [
    {"name": "Cilantro", "aliases": [], "category": "Vegetales", "ready_to_eat": True, "prep_methods": []},
    {"name": "Orégano dominicano", "aliases": ["oregano"], "category": "Despensa", "ready_to_eat": True, "prep_methods": []},
    {"name": "Piña", "aliases": ["pina"], "category": "Frutas", "ready_to_eat": True, "prep_methods": []},
    {"name": "Manzana", "aliases": [], "category": "Frutas", "ready_to_eat": True, "prep_methods": []},
    {"name": "Queso cottage", "aliases": [], "category": "Lácteos", "ready_to_eat": True, "prep_methods": []},
    {"name": "Granola", "aliases": [], "category": "Despensa", "ready_to_eat": True, "prep_methods": []},
    {"name": "Yogurt griego sin azúcar", "aliases": ["yogurt griego"], "category": "Lácteos", "ready_to_eat": True, "prep_methods": []},
    {"name": "Yogur", "aliases": [], "category": "Lácteos", "ready_to_eat": True, "prep_methods": []},
    {"name": "Chuleta", "aliases": [], "category": "Proteínas", "ready_to_eat": False, "prep_methods": ["plancha"]},
    {"name": "Cerdo", "aliases": [], "category": "Proteínas", "ready_to_eat": False, "prep_methods": ["plancha"]},
    {"name": "Sofrito", "aliases": [], "category": "Despensa", "ready_to_eat": True, "prep_methods": []},
    {"name": "Cebolla", "aliases": [], "category": "Vegetales", "ready_to_eat": False, "prep_methods": ["saltear"]},
]


@pytest.fixture(scope="module")
def index():
    return build_culinary_index(_CAT)


def _v5(meal, index):
    return [v["food"] for v in _v5_paso_usa_lo_que_no_esta(1, meal, index)]


# ── lo que V5 tiene que cazar ─────────────────────────────────────────────────────────────────
def test_el_paso_manda_usar_algo_que_nadie_compro(index):
    """El caso vivo: la receta remata con cilantro y la lista trae orégano."""
    meal = {"meal": "Cena", "name": "Lentejas a la plancha",
            "ingredients": ["½ taza de lentejas secas", "¼ taza de orégano dominicano"],
            "recipe": ["Montaje: forma una cama y termina con el cilantro por encima."]}
    assert "cilantro" in _v5(meal, index)


def test_otro_caso_vivo_la_pina_del_cottage(index):
    meal = {"meal": "Merienda", "name": "Cottage frío con manzana",
            "ingredients": ["½ taza de queso cottage", "1⅔ tazas de manzana"],
            "recipe": ["Montaje: coloca el queso cottage en un recipiente y añade la piña."]}
    assert "pina" in _v5(meal, index)


# ── los cinco filtros, uno a uno ──────────────────────────────────────────────────────────────
def test_el_alias_corto_no_es_un_fantasma(index):
    """«yogurt griego» casa `Yogur` Y `Yogurt griego sin azúcar`. La lista resuelve al largo, así que
    el corto se convertía en fantasma: 128 de las 460 acusaciones de la primera versión."""
    meal = {"meal": "Desayuno", "name": "Vaso",
            "ingredients": ["½ taza de yogurt griego sin azúcar"],
            "recipe": ["Mise en place: mide el yogurt griego sin azúcar."]}
    assert _v5(meal, index) == []


def test_la_nota_de_seguridad_no_acusa(index):
    """«el pollo/cerdo debe cocinarse por completo» habla de una CLASE en abstracto, no de los
    ingredientes de este plato. Era 30 de los 38 «cerdo» de la ronda 2."""
    meal = {"meal": "Almuerzo", "name": "Plato",
            "ingredients": ["1 cebolla"],
            "recipe": ["⚠️ Seguridad alimentaria: el pollo/cerdo debe cocinarse por completo."]}
    assert _v5(meal, index) == []


def test_un_paso_que_NIEGA_no_esta_usando(index):
    meal = {"meal": "Desayuno", "name": "Batido",
            "ingredients": ["1 cebolla"],
            "recipe": ["Se reemplazó el huevo crudo del batido por yogur griego."]}
    assert _v5(meal, index) == []


def test_si_el_indice_no_lo_resuelve_NO_significa_que_no_este(index):
    """El filtro que más enseñó: sin él, un ceviche con «1½ filetes de pescado» en la lista salía
    acusado de no llevar pescado, porque el índice no resolvía esa línea. El detector medía el recall
    del catálogo y acusaba al plan de su propia ceguera."""
    meal = {"meal": "Almuerzo", "name": "Plato",
            "ingredients": ["1½ cilantros frescos"],
            "recipe": ["Montaje: coloca el cilantro por encima."]}
    assert _v5(meal, index) == []


def test_el_paso_puede_ser_mas_especifico_que_la_lista(index):
    """«pica la chuleta de cerdo» cuando la lista dice «½ chuleta»: el mismo alimento."""
    meal = {"meal": "Almuerzo", "name": "Croquetas",
            "ingredients": ["½ chuleta", "1 cebolla"],
            "recipe": ["Mise en place: pica la chuleta de cerdo en trozos pequeños."]}
    assert "cerdo" not in _v5(meal, index)


def test_lo_que_la_receta_PRODUCE_no_es_un_ingrediente_que_falte(index):
    """«sofríe la cebolla hasta formar el sofrito»: el sofrito se HACE con lo que sí está en la
    lista. Un paso que consume nombra su objeto tras un verbo de entrada; uno que produce, no."""
    meal = {"meal": "Almuerzo", "name": "Guiso",
            "ingredients": ["1 cebolla"],
            "recipe": ["El Toque de Fuego: sofríe la cebolla hasta formar el sofrito."]}
    assert "sofrito" not in _v5(meal, index)


# ── contratos del check ───────────────────────────────────────────────────────────────────────
def test_v5_corre_dentro_del_scan():
    assert "_v5_paso_usa_lo_que_no_esta(day, meal, index)" in _SRC
    i = _SRC.index("out.extend(_v4_cantidad_inconsistente")
    assert "_v5_paso_usa_lo_que_no_esta" in _SRC[i:i + 220], "V5 no está encadenado tras V4"


def test_v5_es_warn_no_block():
    """91 % de precisión no autoriza a bloquear: castigaría un plan de cada once sin motivo. Y
    calibrarlo contra la tasa del propio juez LLM sería el overfitting que ya se pagó en agosto."""
    i = _SRC.index("def _v5_paso_usa_lo_que_no_esta")
    cuerpo = _SRC[i:i + 3000]
    assert '"minor", False' in cuerpo, "V5 tiene que ser minor y no reparable"
    assert '"high"' not in cuerpo


def test_v5_es_fail_open():
    """Un check de coherencia jamás puede tumbar una generación."""
    assert _v5_paso_usa_lo_que_no_esta(1, {"ingredients": None, "recipe": None}, {}) == []
    assert _v5_paso_usa_lo_que_no_esta(1, {}, None) == []


def test_el_scan_completo_sigue_sin_lanzar(index):
    meal = {"meal": "Cena", "name": "X", "ingredients": ["1 cebolla"],
            "recipe": ["Montaje: coloca el cilantro."]}
    out = culinary_contract_scan({"days": [{"meals": [meal]}]}, _CAT)
    assert any(v["check"] == "V5" for v in out)


def test_la_medicion_que_lo_calibro_vive_con_el_codigo():
    """Sin las cifras, «V5 tiene falsos positivos» dentro de seis meses es una opinión. Con ellas,
    quien lo toque sabe qué rompió cada filtro."""
    i = _SRC.index("V5 — el espejo de V3")
    cabecera = _SRC[i:i + 2200]
    for cifra in ("460", "364", "287", "241", "100", "11"):
        assert cifra in cabecera, cifra
