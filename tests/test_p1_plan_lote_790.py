"""[P1-PLAN-LOTE-790 · 2026-09-28] G58 (código) + fuga de catálogo entre países.

LO QUE SE MIDIÓ (replay determinista del agregador sobre la copia de `master_ingredients`, sin DB
ni IA — `replay_g58.py` de la auditoría G58/G13 del 28-sep):

(a) EL RÓTULO DEL ENVASE. `_sku_size_label` convertía gramos a la etiqueta del mercado DOMINICANO
    y sólo a ése: 100 g salía «¼ lb», 1 kg «2.2 lbs», y `int(size_g)` convertía el sobre de azafrán
    de 0,4 g en «0g». La proyección métrica de P1-UNIT-SYSTEM-BY-COUNTRY no toca estos rótulos a
    propósito, así que aun curando los datos (lote 791) un español leería «1 paquete (¼ lb)». Ahora
    el rótulo derivado del peso del envase habla el sistema del PAÍS de la lista (ES/MX/CO en g/kg/ml;
    DO/US/PR idénticos a hoy) y muestra decimales bajo 1 g en todos. La etiqueta de un envase REAL
    (`market_packages.label`, «5 Oz · Genérico») sigue sin tocarse: es el rótulo de un producto.

(b) LA FUGA. El agregador conservaba el alimento de catálogo-país de CUALQUIERA de los seis: el
    plan español rd587 llevó «Tortilla de maíz» (sólo MX) a su lista como si fuera de su mercado.
    La DECISIÓN sobre el ítem ajeno: se QUEDA en la lista (la receta lo pide y el fallo caro es la
    lista incompleta en silencio) pero deja de pasar por comida de su país: se sella
    `catalogo_de_otro_pais=[<países>]` y se avisa con un WARN grep-able. Dropearlo dejaría la receta pidiendo
    tortilla y la lista sin ella; además abriría una divergencia en el guard de coherencia, cuyo
    espejo (`_survives_shopping_list`) no sabe de países. Y el replay lo confirmó con un caso peor:
    el mismo plan lleva «Trucha», que sólo reclama el bloque de CO — dropear por país le quitaba a
    un español el pescado de la semana. El bloque de un país son sus altas SIN PRECIO, no lo que
    se vende allí.

Y un hallazgo en el camino: «1 paquete de Harina de yuca» salía «… de De Harina de yuca» en
«Otros»: el pass-through de `resolve_preparation_distinct` devolvía el texto crudo con su «de».
"""
from __future__ import annotations

import inspect
import logging
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sc():
    import shopping_calculator as _sc
    return _sc


@pytest.fixture(scope="module")
def ep():
    import envase_pais as _ep
    return _ep


# ── Oráculo: la función ANTES de este lote, copiada tal cual ────────────────────────────────────
def _legacy_label(size_g, unit_hint=None):
    if size_g is None:
        return ""
    size_g = float(size_g)
    if unit_hint and unit_hint.lower() in ['cartón', 'carton', 'botella', 'ml', 'l', 'galón', 'envase', 'lata']:
        VOLUME_LABELS = {250: "250ml", 473: "473ml", 946: "946ml", 1000: "1L", 1892: "1/2 Galón"}
        for vol_g, label in VOLUME_LABELS.items():
            if abs(size_g - vol_g) < 10:
                return label
        if unit_hint.lower() in ['botella', 'ml', 'l', 'galón']:
            if size_g >= 1000:
                liters = size_g / 1000
                if abs(liters - round(liters)) < 0.05:
                    return f"{round(liters):d}L"
                return f"{liters:.1f}L"
            return f"{int(round(size_g))}ml"
    if unit_hint and unit_hint.lower() in ['pote', 'frasco']:
        if abs(size_g - 453.592) < 15: return "16 oz"
        if abs(size_g - 226.796) < 15: return "8 oz"
        if abs(size_g - 340.194) < 15: return "12 oz"
    lbs = size_g / 453.592
    if abs(lbs - round(lbs)) < 0.05 and round(lbs) >= 1:
        return f"{round(lbs)} lb" if round(lbs) == 1 else f"{round(lbs)} lbs"
    if abs(lbs - 0.5) < 0.05:
        return "½ lb"
    if abs(lbs - 0.25) < 0.05:
        return "¼ lb"
    if lbs > 1.2:
        return f"{round(lbs, 1):g} lbs"
    return f"{int(size_g)}g"


_SIZES = [1, 5, 14, 28, 45, 71, 100, 113, 150, 200, 226.8, 250, 283, 340.2, 425, 453.6, 473, 500,
          567, 600, 680, 750, 907.2, 946, 1000, 1360, 1500, 1892, 2000, 2268, 3000]
_HINTS = [None, "paquete", "sobre", "pote", "frasco", "botella", "lata", "cartón", "envase",
          "funda", "galón", "caja", "Botella", "PAQUETE"]


# ── A. DO/US/PR (y lo desconocido) rotulan EXACTAMENTE como antes ───────────────────────────────

@pytest.mark.parametrize("pais", [None, "DO", "US", "PR", "basura"])
def test_los_paises_imperiales_rotulan_igual_que_antes(sc, ep, pais):
    with ep.lista_de_pais(pais):
        for h in _HINTS:
            for s in _SIZES:
                assert sc._sku_size_label(s, h) == _legacy_label(s, h), (pais, s, h)
    assert sc._sku_size_label(None, "sobre") == ""


def test_sin_pais_en_contexto_rotula_como_antes(sc):
    """Los 26 call sites que no pasan por la lista de un plan (chat, swap) no cambian."""
    for h in _HINTS:
        for s in _SIZES:
            assert sc._sku_size_label(s, h) == _legacy_label(s, h)


# ── B. Bajo 1 g, decimales ──────────────────────────────────────────────────────────────────────

def test_el_sobre_de_azafran_ya_no_pesa_0g(sc, ep):
    assert _legacy_label(0.4, "sobre") == "0g", "el oráculo debe reproducir el defecto medido"
    with ep.lista_de_pais("DO"):
        assert sc._sku_size_label(0.4, "sobre") == "0.4g"
        assert sc._sku_size_label(0.38, "sobre") == "0.38g"
    with ep.lista_de_pais("ES"):
        assert sc._sku_size_label(0.4, "sobre") == "0,4 g"
        assert sc._sku_size_label(0.38, "sobre") == "0,38 g"
        assert sc._sku_size_label(0.5, None) == "0,5 g"


# ── C. ES/MX/CO en el sistema métrico ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("pais", ["ES", "MX", "CO"])
@pytest.mark.parametrize("size,hint,esperado", [
    (100, "paquete", "100 g"),
    (113, None, "113 g"),
    (226.8, "paquete", "227 g"),
    (453.6, "pote", "454 g"),
    (226.8, "frasco", "227 g"),
    (340.2, "frasco", "340 g"),
    (1000, "paquete", "1 kg"),
    (2268, "funda", "2,3 kg"),
    (250, "lata", "250 ml"),
    (425, "lata", "425 g"),
    (946, "cartón", "946 ml"),
    (1000, "cartón", "1 L"),
    (1892, "cartón", "1,9 L"),
    (500, "botella", "500 ml"),
    (1500, "botella", "1,5 L"),
    (2000, "botella", "2 L"),
    (5, "sobre", "5 g"),
])
def test_los_paises_metricos_rotulan_en_gramos_y_litros(sc, ep, pais, size, hint, esperado):
    with ep.lista_de_pais(pais):
        assert sc._sku_size_label(size, hint) == esperado


@pytest.mark.parametrize("pais", ["ES", "MX", "CO"])
def test_ningun_rotulo_metrico_habla_en_libras(sc, ep, pais):
    with ep.lista_de_pais(pais):
        for h in _HINTS:
            for s in _SIZES:
                lab = sc._sku_size_label(s, h)
                assert not any(x in lab for x in ("lb", "oz", "Galón")), (pais, s, h, lab)


def test_el_knob_del_sistema_de_unidades_revierte_tambien_el_rotulo(sc, ep, monkeypatch):
    """Una palanca, no dos: apagar P1-UNIT-SYSTEM-BY-COUNTRY devuelve la lista entera a libras."""
    monkeypatch.setenv("MEALFIT_UNIT_SYSTEM_BY_COUNTRY", "false")
    with ep.lista_de_pais("ES"):
        assert sc._sku_size_label(100, "paquete") == "¼ lb"
        assert sc._sku_size_label(0.4, "sobre") == "0.4g"


# ── D. El país viaja por contexto, sin tocar las firmas ─────────────────────────────────────────

def test_el_contexto_se_restaura_incluso_si_la_lista_revienta(ep):
    assert ep.pais_de_la_lista() is None
    with pytest.raises(RuntimeError):
        with ep.lista_de_pais("ES"):
            assert ep.pais_de_la_lista() == "ES"
            with ep.lista_de_pais("MX"):
                assert ep.pais_de_la_lista() == "MX"
            assert ep.pais_de_la_lista() == "ES"
            raise RuntimeError("x")
    assert ep.pais_de_la_lista() is None


def test_las_dos_puertas_de_la_lista_de_un_plan_llevan_el_pais(sc):
    """Ancla parser-based: el decorador va pegado al `def` de las dos funciones que reciben el plan."""
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    for fn in ("get_shopping_list_delta", "get_realtime_pantry"):
        i = src.find(f"\ndef {fn}(")
        assert i > 0, fn
        antes = src[max(0, i - 200):i]
        assert "@_con_pais_del_plan" in antes.splitlines()[-1], f"{fn} sin el país del plan"
    assert "def _sku_size_label(" not in src, "el rótulo vive en envase_pais.py (tope de líneas)"


def test_el_decorador_saca_el_pais_del_sello_del_plan(ep, monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    vistos = []

    @ep.con_pais_del_plan
    def f(user_id, plan_result, otra=1):
        vistos.append(ep.pais_de_la_lista())
        return otra

    assert f(None, {"_country": "ES"}) == 1
    assert f(None, plan_result={"_country": "MX"}, otra=2) == 2
    assert f(None, {}) == 1
    assert f(None, None) == 1
    assert vistos == ["ES", "MX", "DO", "DO"], vistos
    assert ep.pais_de_la_lista() is None


# ── E. Agregador real sobre un catálogo mínimo ──────────────────────────────────────────────────

def _fila(nombre, cat, du, **kw):
    base = {"name": nombre, "category": cat, "aliases": [], "default_unit": du, "price_per_lb": 0.0,
            "price_per_unit": 0.0, "market_container": None, "container_weight_g": None,
            "available_sizes_g": None, "market_packages": None, "density_g_per_unit": None,
            "density_g_per_cup": None, "shelf_life_days": 180, "name_en": None}
    base.update(kw)
    return base


_CATALOGO = [
    _fila("Azafrán", "Despensa", "sobre", aliases=["azafran", "saffron"], name_en="Saffron",
          market_container="sobre", container_weight_g=0.4, available_sizes_g=[0.4]),
    _fila("Achiote", "Despensa", "sobre", aliases=["annatto", "bija"], name_en="Achiote",
          market_container="paquete", container_weight_g=100.0, available_sizes_g=[100]),
    _fila("Tortilla de maíz", "Despensa", "paquete", aliases=["tortillas de maíz", "tortilla de maiz"],
          name_en="Corn tortilla", shelf_life_days=30),
    _fila("Duraznos", "Frutas", "libra", aliases=["peaches"], name_en="Peaches", shelf_life_days=7),
    _fila("Harina de yuca", "Despensa", "paquete", aliases=["cassava flour", "harina de mandioca"],
          name_en="Cassava flour"),
    _fila("Trucha", "Proteínas", "lb", aliases=["trout"], name_en="Trout", shelf_life_days=3),
    _fila("Chile en polvo", "Despensa", "frasco", aliases=["chili powder"], name_en="Chili powder",
          market_container="frasco", container_weight_g=71.0, available_sizes_g=[71]),
    _fila("Pretzels", "Despensa", "funda", aliases=["pretzel"], name_en="Pretzels",
          market_container="funda", container_weight_g=340.0, available_sizes_g=[340]),
    _fila("Pique", "Despensa", "botella", aliases=["pique boricua"], name_en="Pique",
          market_container="botella", container_weight_g=148.0, available_sizes_g=[148]),
    _fila("Pechuga de pollo", "Proteínas", "lb", aliases=["pollo", "pechuga"], price_per_lb=150.0,
          name_en="Chicken breast", shelf_life_days=3),
]


@pytest.fixture
def catalogo(sc, monkeypatch):
    import envase_pais as _ep
    monkeypatch.setattr(_ep, "_SELLOS_AVISADOS", {}, raising=False)  # el WARN deduplicado no cruza tests
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    monkeypatch.setattr(sc, "_master_cache", [dict(r) for r in _CATALOGO])
    monkeypatch.setattr(sc, "_master_cache_ts", time.time() + 10 ** 6)
    monkeypatch.setattr(sc, "_VERIFIED_SHOPPING_NAMES", None, raising=False)
    return sc


def _items(sc, lineas, **kw):
    res = sc.aggregate_and_deduct_shopping_list(list(lineas), structured=True, **kw)
    return res.get("items") if isinstance(res, dict) else res


def _item(sc, lineas, nombre, **kw):
    return next((i for i in _items(sc, lineas, **kw) if i.get("name") == nombre), None)


def test_el_azafran_curado_se_lee_en_su_sistema(catalogo, ep):
    with ep.lista_de_pais("ES"):
        it = _item(catalogo, ["1 sobre de Azafrán"], "Azafrán")
    assert it is not None and "(0,4 g)" in it["display_string"], it
    with ep.lista_de_pais("DO"):
        it = _item(catalogo, ["1 sobre de Azafrán"], "Azafrán")
    assert it is not None and "(0.4g)" in it["display_string"], it


def test_el_achiote_mexicano_ya_no_sale_en_cuartos_de_libra(catalogo, ep):
    with ep.lista_de_pais("MX"):
        it = _item(catalogo, ["1 paquete de Achiote"], "Achiote")
    assert it is not None and "(100 g)" in it["display_string"], it
    assert "lb" not in it["display_string"]


def test_la_lista_de_un_plan_espanol_rotula_en_gramos(catalogo):
    """Por la puerta real (`get_realtime_pantry`), con el país sacado del sello del plan."""
    plan = {"_country": "ES", "days": [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Arroz", "ingredients": ["1 sobre de Azafrán"]}]}]}
    res = catalogo.get_realtime_pantry(plan, [])
    assert any("Azafrán" in s and "(0,4 g)" in s for s in res), res
    plan_do = dict(plan, _country="DO")
    res = catalogo.get_realtime_pantry(plan_do, [])
    assert any("Azafrán" in s and "(0.4g)" in s for s in res), res


# ── F. La fuga: el alimento de OTRO país se queda, pero sellado ─────────────────────────────────

def test_la_tortilla_mexicana_en_una_lista_espanola_se_queda_sellada(catalogo, ep, caplog):
    caplog.set_level(logging.WARNING)
    with ep.lista_de_pais("ES"):
        it = _item(catalogo, ["80 g de Tortilla de maíz"], "Tortilla de maíz")
    assert it is not None, "el ítem ajeno se dropeó: la receta lo pide y la lista quedaría incompleta"
    assert it.get("catalogo_de_otro_pais") == ["MX"], it
    assert any("P1-PLAN-LOTE-790" in r.getMessage() and "Tortilla de maíz" in r.getMessage()
               for r in caplog.records), "la fuga tiene que ser grep-able"


def test_el_warn_de_la_fuga_sale_una_vez_por_lista_no_por_llamada(catalogo, ep, caplog):
    """[revisión ronda 1, defecto 8] Un recálculo de una lista ES llama al agregador 3-9 veces (lista
    semanal, quincenal, mensual, delta…): el mismo WARN salía otras tantas. Se avisa una vez por
    (país, alimentos sellados); lo repetido baja a DEBUG. Otra fuga distinta sí vuelve a avisar."""
    caplog.set_level(logging.DEBUG, logger="envase_pais")

    def _warns():
        return [r for r in caplog.records
                if r.levelno == logging.WARNING and "P1-PLAN-LOTE-790" in r.getMessage()]

    with ep.lista_de_pais("ES"):
        for _ in range(3):
            it = _item(catalogo, ["80 g de Tortilla de maíz"], "Tortilla de maíz")
            assert it.get("catalogo_de_otro_pais") == ["MX"], "el sello no se deduplica: sólo el log"
    assert len(_warns()) == 1, [r.getMessage() for r in _warns()]
    with ep.lista_de_pais("ES"):
        _item(catalogo, ["80 g de Tortilla de maíz", "100 g de Pretzels"], "Pretzels")
    assert len(_warns()) == 2, "una fuga distinta tiene que volver a avisar"
    with ep.lista_de_pais("MX"):
        _item(catalogo, ["100 g de Pretzels"], "Pretzels")
    assert len(_warns()) == 3, "otro país, otro aviso"


def test_la_trucha_tambien_es_de_espana_y_no_se_sella(catalogo, ep):
    """El caso del replay: dropear por país borraba el pescado de la semana de un plan español.

    [revisión ronda 1, defecto 2] Y sellarla era un falso positivo: en España la trucha se vende en
    cualquier pescadería. Se añade a su bloque (mismo patrón que «duraznos», P1-PLAN-LOTE-624)."""
    assert catalogo.is_country_catalog_unpriced_item("Trucha", country="ES") is True
    with ep.lista_de_pais("ES"):
        it = _item(catalogo, ["340 g de Trucha"], "Trucha")
    assert it is not None, "la trucha se dropeó de una lista española"
    assert "catalogo_de_otro_pais" not in it, it


def test_el_chile_en_polvo_tambien_es_de_mexico(catalogo, ep):
    """[revisión ronda 1, defecto 2] Cuatro planes mexicanos llevaban «Chile en polvo», que sólo
    reclamaba el bloque de US: en México se vende en cualquier súper."""
    assert catalogo.is_country_catalog_unpriced_item("Chile en polvo", country="MX") is True
    with ep.lista_de_pais("MX"):
        it = _item(catalogo, ["5 g de Chile en polvo"], "Chile en polvo")
    assert it is not None and "catalogo_de_otro_pais" not in it, it


@pytest.mark.parametrize("pais,linea,nombre", [
    ("PR", "100 g de Pretzels", "Pretzels"),   # sólo lo reclama US
    ("US", "30 g de Pique", "Pique"),          # sólo lo reclama PR
])
def test_puerto_rico_y_estados_unidos_son_el_mismo_mercado(catalogo, ep, pais, linea, nombre):
    """[revisión ronda 1, defecto 2] «PR usa US declarado» (lote 791): el súper de Puerto Rico es el
    de Estados Unidos. Sellar como ajeno lo que sólo reclama el otro era un falso positivo."""
    with ep.lista_de_pais(pais):
        it = _item(catalogo, [linea], nombre)
    assert it is not None and "catalogo_de_otro_pais" not in it, it


def test_el_mismo_mercado_no_desella_a_los_demas(catalogo, ep):
    with ep.lista_de_pais("ES"):
        it = _item(catalogo, ["100 g de Pretzels"], "Pretzels")
    assert it is not None and it.get("catalogo_de_otro_pais") == ["US"], it


@pytest.mark.parametrize("pais", ["MX", "DO", None])
def test_en_su_mercado_o_en_DO_no_se_sella(catalogo, ep, pais):
    """MX: es suyo. DO: conserva el predicado de siempre (byte-identidad, mismo criterio que el
    catálogo verificado del generador). Sin país: la conducta histórica."""
    with ep.lista_de_pais(pais):
        it = _item(catalogo, ["80 g de Tortilla de maíz"], "Tortilla de maíz")
    assert it is not None
    assert "catalogo_de_otro_pais" not in it


def test_un_alimento_de_varios_paises_incluido_el_suyo_no_se_sella(catalogo, ep):
    with ep.lista_de_pais("ES"):
        it = _item(catalogo, ["300 g de Duraznos"], "Duraznos")
    assert it is not None and "catalogo_de_otro_pais" not in it


def test_la_comida_con_precio_no_se_sella(catalogo, ep):
    with ep.lista_de_pais("ES"):
        it = _item(catalogo, ["300 g de Pechuga de pollo"], "Pechuga de pollo")
    assert it is not None and "catalogo_de_otro_pais" not in it


def test_el_knob_apaga_el_sello(catalogo, ep, monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_CATALOG_FOREIGN_FLAG", "false")
    with ep.lista_de_pais("ES"):
        it = _item(catalogo, ["80 g de Tortilla de maíz"], "Tortilla de maíz")
    assert it is not None and "catalogo_de_otro_pais" not in it


def test_el_espejo_del_guard_sigue_viendo_el_item(catalogo):
    """El ítem ajeno sobrevive en los DOS lados del guard de coherencia: ninguna divergencia nueva."""
    assert catalogo._survives_shopping_list("Tortilla de maíz") is True


def test_el_generador_no_le_ofrece_la_tortilla_mexicana_a_un_espanol(sc, monkeypatch):
    """La fuga nace aguas arriba: el catálogo verificado del generador YA pregunta por país
    (P1-COUNTRY-CATALOG-BY-COUNTRY); el modelo la escribió igual. La lista es la última red.

    [revisión ronda 1, defecto 9] No basta el predicado: se mide el bloque «USA EXCLUSIVAMENTE» que
    de verdad se renderiza para el generador (`_vc_comprable` → `_get_verified_catalog_instruction`)."""
    import graph_orchestrator as go
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    assert sc.is_country_catalog_unpriced_item("Tortilla de maíz", country="ES") is False
    assert sc.is_country_catalog_unpriced_item("Tortilla de maíz", country="MX") is True
    filas = [{"name": n, "price_per_lb": 0, "price_per_unit": 0}
             for n in ("Tortilla de maíz", "Trucha", "Chile en polvo", "Azafrán")]
    filas.append({"name": "Arroz blanco", "price_per_lb": 35.0, "price_per_unit": 0})
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: [dict(r) for r in filas])
    monkeypatch.setattr(sc, "_verified_ingredients_only_enabled", lambda *a, **k: True)
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    try:
        es = go._get_verified_catalog_instruction({"country": "ES"})
        mx = go._get_verified_catalog_instruction({"country": "MX"})
    finally:
        go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    assert "Tortilla de maíz" not in es, "al generador español se le ofrece la tortilla mexicana"
    assert "Azafrán" in es and "Trucha" in es and "Arroz blanco" in es
    assert "Tortilla de maíz" in mx and "Chile en polvo" in mx
    assert "Azafrán" not in mx


# ── G. «… de De Harina de yuca» ─────────────────────────────────────────────────────────────────

def test_la_harina_de_yuca_no_arrastra_su_de(catalogo):
    assert catalogo.normalize_name("de Harina de yuca") == "Harina de yuca"
    assert catalogo.normalize_name("paquete de harina de yuca") == "Harina de yuca"
    it = _item(catalogo, ["1 paquete de Harina de yuca"], "Harina de yuca")
    assert it is not None, [i.get("name") for i in _items(catalogo, ["1 paquete de Harina de yuca"])]
    assert "De Harina" not in it["display_string"]
    assert it.get("category") == "Despensa"


def test_el_pass_through_sigue_sin_colapsar_la_harina_al_tuberculo(sc):
    """El guard P1-PREP-COLLAPSE-GUARD no se afloja: la harina de plátano no es plátano."""
    assert sc.resolve_preparation_distinct("de harina de platano") == (True, None)
    assert sc.normalize_name("de Harina de plátano") == "Harina de plátano"


# ── H. Anclas ───────────────────────────────────────────────────────────────────────────────────

def test_tooltip_anchor(ep):
    src = inspect.getsource(ep)
    assert "tooltip-anchor: P1-PLAN-LOTE-790" in src
    assert "[P1-PLAN-LOTE-790 · 2026-09-28]" in src
