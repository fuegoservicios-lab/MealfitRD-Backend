# -*- coding: utf-8 -*-
"""[P1-ARQ27-F2-IDENTIDAD · 2026-09-06] La identidad de un alimento no es su categoría de tienda.

**Continuación de [`test_p1_arq27_f2_identidad.py`](test_p1_arq27_f2_identidad.py)**, que ya cerró la
mayor parte de `ARQ27-P1-05` el mismo día: midió las 347 filas, encontró que el motor ya resolvía por
nombre y no por pasillo, arregló la única discrepancia viva (`Yogur de coco` se declaraba
`['lacteos','lactosa']`) y ancló la paridad entre lo que el registry DECLARA y lo que el guard DECIDE.
Aquello no se repite aquí.

Lo que quedaba, medido:

| criterio del gap | estado |
|---|---|
| «Leche de coco conserva identidad vegetal…» | ya cerrado por el fichero hermano |
| «La proyección pública no contiene certificación dietaria» (la evidencia del gap) | **abierto**: `/api/catalog` mandaba `category` y nada más |
| «Quesos/preparados/fortificados conservan incertidumbres» | **abierto**: 16 plantillas con queso o embutido, cero la declaraban |
| «…histórico sigue resolviendo» | **abierto**: 1.137 alias producen ids que no casan con ningún nombre |

La evidencia del gap era exacta: un consumidor que quisiera saber si algo es vegano tenía justo el
campo que dice `Lácteos` para la leche de coco.

**Dónde NO estaba el defecto**, comprobado uno a uno para no arreglar lo que no está roto: el guard
de embarazo lee `category=='lacteos'` pero ya excluye las bebidas vegetales con su propia regex; las
dos ramas de `db_inventory` son tasas de consumo (200 ml/día), una heurística de despensa; y el
frontend usa «Lácteos» para el chip de alergia (una CLASE, no la categoría) y para el estante de la
Nevera, que es presentación — justo lo que el gap permite.

**Cero tablas nuevas.** `food_identity` compone `_diet_pool_item_banned` (SSOT de dieta desde
P1-DIET-CANON-SSOT) y `allergen_classes_for`. Una tercera tabla de «qué es vegano» divergiría de las
otras dos en cuanto alguien la editara — la lección que este repo ya pagó dos veces.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import food_identity as fi  # noqa: E402
from plan_policy import canonical_name_for, ingredient_id_for  # noqa: E402

_ROUTER = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")


def _fila(nombre, categoria="Lácteos", fuente="usda", fdc="1"):
    return {"name": nombre, "category": categoria, "nutrition_source": fuente, "fdc_id": fdc}


# ── lo que el fichero hermano ya cubre, no se repite ──────────────────────────────────────────
def test_el_hermano_cubre_la_identidad_vegetal():
    """Las cinco leches vegetales y los lácteos reales ya están anclados en
    `test_p1_arq27_f2_identidad.py`, con la paridad entre las dos capas. Repetir sus asserts aquí
    sería una segunda fuente que puede divergir de la primera — el defecto que ese mismo fichero
    existe para impedir."""
    hermano = (_BACKEND / "tests" / "test_p1_arq27_f2_identidad.py").read_text(encoding="utf-8")
    assert "VEGETALES_EN_LACTEOS" in hermano
    assert "def test_las_dos_capas_de_lacteo_coinciden" in hermano


def test_la_identidad_que_publica_el_catalogo_sale_del_mismo_SSOT():
    """Lo único que hace falta comprobar aquí: que `food_identity` no decide por su cuenta, sino que
    pregunta a los mismos guards que el hermano ancla."""
    d = fi.identidad(_fila("Leche de coco"))
    assert d["vegan_ok"] is True and d["allergen_classes"] == []
    d2 = fi.identidad(_fila("Leche"))
    assert d2["vegan_ok"] is False and "lacteos" in (d2["allergen_classes"] or [])


def test_un_indeterminado_no_es_un_no():
    """`None` significa «no se pudo determinar» y NUNCA se colapsa a False. Es la invariante que este
    repo ya pagó dos veces: `int(x or -1)` con `attempts=0`, y el nutriente ausente leído como cero.
    Un consumidor que reciba `null` debe preguntar, no dar por hecho que se puede comer."""
    import food_identity
    orig = food_identity._prohibido_para
    food_identity._prohibido_para = lambda *a, **k: None
    try:
        d = fi.identidad(_fila("Lo que sea"))
        assert d["vegan_ok"] is None and d["vegetarian_ok"] is None
    finally:
        food_identity._prohibido_para = orig


def test_una_fila_sin_nombre_no_inventa_identidad():
    assert fi.identidad({"name": ""}) == {}
    assert fi.identidad({}) == {}


# ── la confianza del dato ─────────────────────────────────────────────────────────────────────
def test_lo_curado_no_se_presenta_como_medido():
    """42 filas del catálogo son `manual`: una estimación del equipo. Un «queso de hoja» varía por
    marca y por receta; decir que su cifra es una medición sería afirmar de más."""
    assert fi.confianza_nutricional(_fila("Queso de hoja", fuente="manual", fdc=None)) == fi.CONFIANZA_CURADA
    assert fi.confianza_nutricional(_fila("Queso blanco", fuente="usda", fdc="172223")) == fi.CONFIANZA_REFERENCIADA


def test_un_fdc_id_compartido_es_un_proxy():
    """La cifra describe OTRO alimento. Hoy no hay ninguno —288 filas con id, las 288 distintas— así
    que esto es un guard contra regresión: el 19-ago sí eran 47 de 347."""
    rows = [_fila("Sobrasada", fdc="9999"), _fila("Chorizo", fdc="9999"), _fila("Pollo", fdc="111")]
    comp = fi.fdc_ids_compartidos(rows)
    assert comp == {"9999"}
    assert fi.confianza_nutricional(rows[0], comp) == fi.CONFIANZA_PROXY
    assert fi.confianza_nutricional(rows[2], comp) == fi.CONFIANZA_REFERENCIADA


def test_el_proxy_gana_a_la_fuente():
    """Una fila `usda` cuyo id apunta a otro alimento es exactamente el caso que P1-BEDCA-DEPROXY-ES
    encontró: la fuente es impecable y el número es de otro."""
    rows = [_fila("A", fuente="usda", fdc="7"), _fila("B", fuente="usda", fdc="7")]
    assert fi.confianza_nutricional(rows[0], fi.fdc_ids_compartidos(rows)) == fi.CONFIANZA_PROXY


def test_sin_fuente_reconocible_es_desconocido_no_referenciado():
    assert fi.confianza_nutricional(_fila("X", fuente="", fdc=None)) == fi.CONFIANZA_DESCONOCIDA


def test_los_ids_nulos_no_se_agrupan_entre_si():
    """Si `None` contara como valor compartido, TODAS las filas sin id serían proxies unas de otras."""
    rows = [_fila("A", fdc=None), _fila("B", fdc=None), _fila("C", fdc="")]
    assert fi.fdc_ids_compartidos(rows) == frozenset()


# ── la anotación del catálogo ─────────────────────────────────────────────────────────────────
def test_anotar_es_aditivo():
    rows = [_fila("Leche de coco"), _fila("Leche")]
    fi.anotar_catalogo(rows)
    assert all("diet" in r and r["category"] == "Lácteos" for r in rows), "no debe tocar lo que ya iba"


def test_el_catalogo_publica_la_identidad():
    assert "from food_identity import anotar_catalogo" in _ROUTER


def test_el_invitado_tambien_la_recibe():
    """El paso 15 del wizard es público y ahí es donde un vegano elige sus básicos: mandarle
    `category` a secas le da «Lácteos» para la leche de coco y nada con que corregirlo."""
    i = _ROUTER.index("_CATALOG_CAMPOS_INVITADO = (")
    assert '"diet"' in _ROUTER[i:i + 400]


def test_no_nace_una_tercera_tabla_de_dieta():
    """La lección de `canonicalize_diet_type` y `pantry_names_match`: una tabla nueva que derive de
    las otras dos diverge en cuanto alguien la edita."""
    src = (_BACKEND / "food_identity.py").read_text(encoding="utf-8")
    assert "_diet_pool_item_banned" in src and "allergen_classes_for" in src
    for inventada in ("VEGAN_FOODS", "DAIRY_NAMES", "_ES_VEGANO", "PLANT_MILKS"):
        assert inventada not in src, f"{inventada} es una taxonomía nueva; usa los SSOT que ya deciden"


# ── el puente de alias ────────────────────────────────────────────────────────────────────────
def test_el_alias_resuelve_al_nombre_canonico():
    """1.137 alias del catálogo producen ids que no casan con ningún nombre. Antes no resolvían."""
    rows = [{"name": "Kétchup", "aliases": ["catsup", "salsa de tomate estilo ketchup"]}]
    assert canonical_name_for("catsup", ["Kétchup"], rows) == "Kétchup"
    assert canonical_name_for("catsup", ["Kétchup"]) is None, "sin filas, el comportamiento es el de antes"


def test_varios_alias_resuelven_al_MISMO_alimento():
    """«Alias no aumenta alimentos únicos» — el criterio, literal."""
    rows = [{"name": "Pique", "aliases": ["vinagre pique", "pique criollo", "pique boricua"]}]
    nombres = {canonical_name_for(ingredient_id_for(a), ["Pique"], rows)
               for a in ("vinagre pique", "pique criollo", "pique boricua")}
    assert nombres == {"Pique"}


def test_el_nombre_canonico_sigue_ganando_al_alias():
    """El orden es load-bearing: si un alias de OTRA fila coincidiera con un nombre canónico, el
    nombre manda. Un puente no puede secuestrar una identidad que ya existe."""
    rows = [{"name": "Otro", "aliases": ["pique"]}]
    assert canonical_name_for("pique", ["Pique"], rows) == "Pique"


def test_el_puente_es_aditivo():
    """Rollback declarado en el gap: «columnas/estructuras aditivas y puente de alias, sin fusiones
    destructivas». `ingredient_id_for` no cambia — los ids ya acuñados siguen siendo los mismos."""
    assert ingredient_id_for("Leche de coco") == "leche_de_coco"
    assert ingredient_id_for("Kétchup") == "ketchup"
