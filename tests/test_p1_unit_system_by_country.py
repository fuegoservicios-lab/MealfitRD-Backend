"""[P1-UNIT-SYSTEM-BY-COUNTRY · 2026-08-21] La lista de compras del español venía en libras.

Medido sobre los 2 planes beta vivos: **14 de 25 y 26 de 48** ítems traen unidades imperiales.
«1 lb · alcanza ~5 de 7 días», «3 lbs», «1 Ud. (~3.3 lbs)». En España, México y Colombia la carne
se vende por kilos y la báscula del súper da gramos: hay que convertir a mano cada línea, y ¼ lb =
113 g no es un número redondo que nadie pida. La misma decisión **acierta para DO/US/PR** —ahí la
libra es la unidad real de compra— y falla para ES/MX/CO.

DOS MEDICIONES ACOTARON EL ARREGLO ANTES DE ESCRIBIRLO, y las dos lo hicieron más pequeño:

1. **Las recetas ya son métricas: 0 de 96 líneas** del plan español usan libras. El LLM escribe
   gramos. O sea que el problema NO está en la generación: vive entero en el agregador
   determinista, `apply_smart_market_units`, cuyo docstring dice literalmente «motor determinista
   de unidades de mercado **dominicano**». Un solo sitio, no el motor entero.

2. **La mitad de los «lb» que se ven NO son una instrucción de pesar**, sino la etiqueta de un
   envase real: `1 botella (5 Oz · Genérico)`, `1 funda (Selecto 1 Lb · Wala)`. Eso es el rótulo de
   un producto que existe; convertirlo sería falsificar una etiqueta. Sólo se convierte cuando la
   unidad de mercado ES el peso — cuando la lista le está diciendo al usuario cuánto pesar.

DISPLAY-ONLY, Y NO ES UN DETALLE. Se reescriben `display_qty` y `display_string`; NO se tocan
`market_unit`, `market_qty_numeric`, `base_qty` ni `base_unit`. La razón es concreta: `/restock`
(«ya compré la lista») construye las filas de la Nevera con `market_qty_numeric` + `market_unit`,
así que convertir el dato metería gramos donde la deducción espera libras — la misma clase de
trampa que hizo descartar el arreglo propuesto para P1-5, donde el nombre resultó ser un
identificador de punta a punta.

Y hay UN camino por el que el display sí llega al dato: `Dashboard.jsx:4398` cae a
`parseMarketQty(display_qty)` **cuando `resolveShopQty(ing)` devuelve 0**. Por eso la conversión se
niega a actuar sobre un ítem sin cantidad numérica: es exactamente el caso en que ese fallback
dispara. La guarda no es defensiva por gusto, cubre el único hueco medido.

Cubre:
  A. `unit_system` es un campo del SSOT de países, no una tabla nueva.
  B. Se convierte lo que es una instrucción de pesar.
  C. NO se toca la etiqueta de un envase, ni el dato que consume la Nevera.
  D. La guarda del fallback de la Nevera.
  E. Knob de rollback.
  F. Byte-identidad para DO/US/PR.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def sc():
    import shopping_calculator as _sc
    return _sc


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


# ── A. El SSOT ──────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc,sistema", [
    ("DO", "imperial"), ("US", "imperial"), ("PR", "imperial"),
    ("ES", "metric"), ("MX", "metric"), ("CO", "metric"),
])
def test_el_sistema_de_unidades_vive_en_country_profiles(cc, sistema):
    """DO/US/PR en imperial no es un olvido: en los tres la libra es la unidad real de compra."""
    from constants import COUNTRY_PROFILES, unit_system_for_country
    assert COUNTRY_PROFILES[cc]["unit_system"] == sistema
    assert unit_system_for_country(cc) == sistema


@pytest.mark.parametrize("basura", [None, "", "ZZ", "basura", 42])
def test_un_pais_desconocido_cae_a_imperial(basura):
    """Fail-safe hacia la conducta de hoy: lo desconocido se comporta como República Dominicana,
    igual que `canonicalize_country`."""
    from constants import unit_system_for_country
    assert unit_system_for_country(basura) == "imperial"


def test_no_nace_una_segunda_tabla_de_paises():
    """El campo vive DENTRO de `COUNTRY_PROFILES`. Un dict aparte `{'ES': 'metric', ...}` sería la
    cuarta tabla de la lección P1-DIET-CANON-SSOT — la que drifteó y sirvió Pollo a vegetarianas."""
    from constants import COUNTRY_PROFILES
    for cc, perfil in COUNTRY_PROFILES.items():
        assert "unit_system" in perfil, f"{cc} sin unit_system"


# ── B. Se convierte la instrucción de pesar ─────────────────────────────────────────────────────

def _obj(**kw):
    base = {"name": "Cerdo", "market_unit": "lb", "market_qty_numeric": 1.0,
            "display_qty": "1 lb", "display_string": "1 lb de Cerdo"}
    base.update(kw)
    return base


@pytest.mark.parametrize("qty,unidad,esperado_en", [
    (1.0, "lb", "454 g"),
    (3.0, "lbs", "1,4 kg"),
    (0.25, "lb", "113 g"),
    (8.0, "oz", "227 g"),
])
def test_el_peso_se_convierte_para_un_pais_metrico(sc, knob_on, qty, unidad, esperado_en):
    o = _obj(market_qty_numeric=qty, market_unit=unidad,
             display_qty=f"{qty} {unidad}", display_string=f"{qty} {unidad} de Cerdo")
    assert sc._project_display_units_for_country(o, "ES") is True
    assert esperado_en in o["display_qty"], o["display_qty"]
    assert esperado_en in o["display_string"], o["display_string"]


def test_la_nota_de_cobertura_sobrevive(sc, knob_on):
    """«alcanza ~5 de 7 días — recompra» es información que el usuario necesita; la conversión
    reescribe la cantidad, no borra el resto de la línea."""
    o = _obj(display_qty="1 lb · alcanza ~5 de 7 días — recompra",
             display_string="1 lb de Queso blanco · alcanza ~5 de 7 días — recompra",
             name="Queso blanco")
    assert sc._project_display_units_for_country(o, "ES") is True
    assert "alcanza ~5 de 7 días — recompra" in o["display_qty"]
    assert "454 g" in o["display_qty"]


def test_el_nombre_del_alimento_no_se_toca(sc, knob_on):
    """Un nombre de alimento es un identificador (P1-PANTRY-NAME-RESOLUTION, y el re-diagnóstico
    de P1-5). Esto convierte cantidades, jamás nombres."""
    o = _obj(name="Cerdo", display_string="1 lb de Cerdo")
    sc._project_display_units_for_country(o, "ES")
    assert o["name"] == "Cerdo"
    assert "Cerdo" in o["display_string"]


# ── C. Lo que NO se toca ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("unidad,dq", [
    ("botella", "1 botella (5 Oz · Genérico)"),
    ("funda", "1 funda (Selecto 1 Lb · Wala)"),
    ("pote", "1 pote (Cremosa 16 Oz · Zerca)"),
    ("sobre", "1 sobre (0.5 Oz · Badia)"),
])
def test_la_etiqueta_de_un_envase_real_no_se_convierte(sc, knob_on, unidad, dq):
    """«5 Oz · Genérico» es el rótulo de un producto que existe en el estante, no una orden de
    pesar. Convertirlo sería falsificar una etiqueta — y el usuario no encontraría el envase."""
    o = _obj(market_unit=unidad, display_qty=dq, display_string=f"{dq} de X")
    antes = dict(o)
    assert sc._project_display_units_for_country(o, "ES") is False
    assert o["display_qty"] == antes["display_qty"]
    assert o["display_string"] == antes["display_string"]


def test_el_filtro_por_unidad_protege_lo_que_la_regex_sola_dejaria_pasar(sc, knob_on):
    """Este caso existe porque la mutación lo pidió: quitar el filtro `market_unit` NO rompía
    ningún test, o sea que el filtro era un guard sin prueba — y un guard que no puede fallar se
    lo lleva el próximo refactor.

    La forma peligrosa: `display_qty` EMPIEZA con un peso pero `market_qty_numeric` cuenta otra
    cosa. La construye el propio código (`f"{weight_lbl} (~{units_count} Uds.)"`): ahí el peso y
    el conteo son DOS números distintos, así que convertir usando el numérico escribiría un peso
    que nadie calculó — «907 g» donde la receta pedía 3 lbs. La regex anclada no lo ve: para ella
    la línea empieza con un peso y punto.

    Honestidad sobre su alcance: medido contra los 1064 ítems de los 12 planes vivos, **0** tienen
    hoy esta forma. El guard es PREVENTIVO, no correctivo — y por eso necesitaba su test."""
    o = _obj(market_unit="Ud.", market_qty_numeric=2.0,
             display_qty="3 lbs (~2 Uds.)", display_string="3 lbs (~2 Uds.) de Coliflor",
             name="Coliflor")
    assert sc._project_display_units_for_country(o, "ES") is False
    assert o["display_qty"] == "3 lbs (~2 Uds.)", (
        "convirtió usando un numérico que contaba UNIDADES, no libras"
    )


def test_los_campos_que_consume_la_nevera_quedan_intactos(sc, knob_on):
    """`/restock` construye las filas de `user_inventory` con `market_qty_numeric` + `market_unit`.
    Convertir el DATO metería gramos donde la deducción espera libras: la Nevera descontaría mal y
    en silencio. Por eso esto es display-only, sin excepción."""
    o = _obj(market_qty_numeric=3.0, market_unit="lbs", display_qty="3 lbs",
             display_string="3 lbs de Cerdo", base_qty=1360.0, base_unit="g")
    assert sc._project_display_units_for_country(o, "ES") is True
    assert o["market_qty_numeric"] == 3.0
    assert o["market_unit"] == "lbs"
    assert o["base_qty"] == 1360.0 and o["base_unit"] == "g"


# ── D. La guarda del fallback de la Nevera ──────────────────────────────────────────────────────

@pytest.mark.parametrize("qty", [0, 0.0, None, "", "basura"])
def test_sin_cantidad_numerica_no_se_convierte(sc, knob_on, qty):
    """`Dashboard.jsx:4398` cae a `parseMarketQty(display_qty)` cuando `resolveShopQty(ing)` da 0 —
    ése es el ÚNICO camino medido por el que el display llega al dato. Convertir ahí escribiría
    «1,4» en la Nevera con la unidad `lbs` al lado. La guarda cubre exactamente ese hueco."""
    o = _obj(market_qty_numeric=qty)
    assert sc._project_display_units_for_country(o, "ES") is False
    assert o["display_qty"] == "1 lb"


def test_no_revienta_con_un_objeto_deforme(sc, knob_on):
    """Corre por cada ítem de la lista: una excepción aquí rompe la lista entera."""
    for deforme in ({}, {"market_unit": "lb"}, {"display_qty": None}, None, "texto"):
        assert sc._project_display_units_for_country(deforme, "ES") is False


# ── E. Knob ─────────────────────────────────────────────────────────────────────────────────────

def test_el_knob_permite_revertir(sc, knob_on, monkeypatch):
    o = _obj()
    monkeypatch.setenv("MEALFIT_UNIT_SYSTEM_BY_COUNTRY", "false")
    assert sc._unit_system_by_country_enabled() is False
    assert sc._project_display_units_for_country(o, "ES") is False
    assert o["display_qty"] == "1 lb"
    monkeypatch.delenv("MEALFIT_UNIT_SYSTEM_BY_COUNTRY", raising=False)
    assert sc._unit_system_by_country_enabled() is True


# ── F. Byte-identidad de los países imperiales ──────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["DO", "US", "PR", None, "basura"])
def test_los_paises_imperiales_no_cambian(sc, knob_on, cc):
    """En DO, US y PR la libra ES la unidad de compra: convertirla sería el mismo defecto al revés.
    Y lo desconocido se comporta como DO."""
    o = _obj()
    assert sc._project_display_units_for_country(o, cc) is False
    assert o["display_qty"] == "1 lb"
    assert o["display_string"] == "1 lb de Cerdo"


@pytest.mark.parametrize("forma", ["lista", "categorias"])
def test_el_recorrido_cubre_las_dos_formas_del_resultado(sc, knob_on, forma):
    """El agregador devuelve una lista plana o un dict por categoría según `categorize`. Cubrir
    sólo una dejaría media lista en libras según por dónde entrara el caller."""
    items = [_obj(), _obj(name="Res", market_qty_numeric=2.0, market_unit="lbs",
                  display_qty="2 lbs", display_string="2 lbs de Res")]
    res = items if forma == "lista" else {"🥩 Proteínas": items}
    assert sc._project_units_over_result(res, "ES") == 2
    assert "454 g" in items[0]["display_qty"]
    assert "907 g" in items[1]["display_qty"]


def test_la_proyeccion_corre_DESPUES_de_calcular_el_costo(sc):
    """La posición es load-bearing y por eso se ancla. `_cost_from_market` calcula el costo
    PARSEANDO el display redondeado (P3-PRICE-MARKET-COVERAGE: «costo desde el DISPLAY, no desde
    weight_in_lbs crudo»). Convertir antes le daría gramos a un parser que espera libras y el
    precio saldría mal — un fallo silencioso que contaminaría el banner de presupuesto.

    Se mide sobre el cuerpo de `get_shopping_list_delta`, que tiene UN solo punto de salida."""
    import inspect
    src = inspect.getsource(sc.get_shopping_list_delta)
    i_proy = src.find("_project_units_over_result")
    i_ret = src.rfind("return res")
    assert i_proy > 0, "la proyección no está cableada en el punto de salida de la lista"
    assert i_proy < i_ret, "la proyección quedó después del return: nace inerte"
    assert "_cost_from_market" not in src[i_proy:], (
        "hay cálculo de costo DESPUÉS de la proyección: parsearía gramos esperando libras"
    )


def test_el_pais_lo_deriva_la_lista_desde_el_sello_del_plan(sc):
    """`get_shopping_list_delta` ya recibe `plan_result`, así que deriva el país del sello
    `_country` (P1-PLAN-STAMPS-COUNTRY) sin tocar sus 26 call sites. Un `country=` nuevo en cada
    uno habría sido 26 sitios donde olvidarse de pasarlo."""
    import inspect
    src = inspect.getsource(sc.get_shopping_list_delta)
    assert "country_for_plan" in src, (
        "la lista no deriva el país del sello del plan: o lo re-canonicaliza por su cuenta "
        "(segunda tabla) o no lo sabe"
    )
