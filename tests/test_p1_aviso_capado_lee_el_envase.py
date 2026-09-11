"""[P1-AVISO-CAPADO-LEE-EL-ENVASE · 2026-09-11] El aviso del tope no miraba el envase que se compra.

## Lo que salía (catálogo de producción, 2026-09-10)

    aggregate_and_deduct_shopping_list(["1 sobre de sazón con culantro y achiote"], structured=True)
    → «1 caja (8 sobres · 1,41 oz) · alcanza ~5 de 7 días — recompra»   RD$99

Medido: la necesidad de la semana llega en 40 g, el tope de especias (P6-SPICE-CAP) la recorta a
28 g (`capped_pre=40`, `capped_post=28`) y la lista compra UNA caja de 40 g. La caja cubre la semana;
el «recompra» era falso. El número salía de `post / pre` = 28/40 = 0,7 → «~5 de 7»: la fracción que
sobrevive al TOPE, no la que cubre lo COMPRADO (el envase redondea hacia arriba).

## Por qué el arreglo que ya existía no lo veía

P1-COVERAGE-VS-PURCHASE (2026-07-27) cazó esta misma clase y puso `_purchase_covers_need` para callar
el aviso cuando lo comprado cubre la necesidad. Pero se evaluaba sobre `result` ANTES de que existiera
`result["package_grams"]` (P1-BRAND-SIZE-FILTER lo adjunta ~30 líneas después), así que con cualquier
envase —caja, pote, frasco, sobre, botella— caía en su «ante la duda, False». Sólo funcionaba con
unidades de peso (lb/kg/g). Sus tests pasaban `package_grams` a mano en el dict: probaban la regla,
no que el aviso la recibiera.

Barrido de las especias del catálogo bajo el tope (1 persona-semana ⇒ tope 28 g): con 40 g de
necesidad, 6 de las 7 que se venden en envase decían «~5 de 7 — recompra» con un envase que cubre la
semana (sólo el comino, cuyo pote ES de 28 g, avisaba con razón). Con 100 g el número mentía a la baja:
el pimentón decía «~2 de 7» sobre un frasco de 85 g que cubre ~6.

## La regla

Lo comprado en gramos contra la necesidad ANTES del tope: si cubre, calla; si no, los días salen de lo
comprado. Sólo cuando el tope habla en gramos (su `post` es lo que llega a `apply_smart_market_units`);
si registró sobres o latas, o el envase se recontó por unidades (P2-PACK-UNITS-MATCH), se conserva la
fracción del tope — el comportamiento anterior. La decisión de compra del tope NO cambia.
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc

_LB = 453.592

# Fila de producción (P1-DO-DESPENSA-DE-SU-MERCADO): caja de 8 sobres, 1,41 oz, RD$99. La DB entrega
# `container_weight_g` y `density_g_per_unit` como texto.
SAZON = {
    "name": "Sazón con culantro y achiote", "category": "Despensa",
    "market_container": "caja", "container_weight_g": "40", "density_g_per_unit": "5",
    "market_packages": [{"unit": "caja", "grams": 40, "label": "8 sobres · 1,41 oz", "price": 99}],
    "price_per_lb": 1122.64, "shelf_life_days": 14, "aliases": [],
}

# Envases y precios de producción de las especias bajo P6-SPICE-CAP que se venden en envase.
# (nombre, fila, envase que elige la lista, días que cubre ese envase de una necesidad de 100 g)
_ESPECIAS = [
    ("Ajo en polvo", {"market_container": "frasco", "container_weight_g": "85",
                      "market_packages": [{"unit": "botella", "grams": 85.05, "label": "3 Oz", "price": 115}]},
     ("botella", 85.05), 6),
    ("Canela en polvo", {"market_container": "sobre", "container_weight_g": "14.2",
                         "market_packages": [{"unit": "sobre", "grams": 14.2, "price": 55},
                                             {"unit": "pote", "grams": 56.7, "price": 105}]},
     ("pote", 56.7), 4),
    ("Pimienta negra", {"market_container": "sobre", "container_weight_g": "14.2",
                        "market_packages": [{"unit": "sobre", "grams": 14.2, "price": 59},
                                            {"unit": "frasco", "grams": 42.5, "price": 89}]},
     ("frasco", 42.5), 3),
    ("Orégano dominicano", {"market_container": "frasco", "container_weight_g": "90.7",
                            "market_packages": [{"unit": "sobre", "grams": 45, "price": 39},
                                                {"unit": "pote", "grams": 90.7, "price": 81}]},
     ("sobre", 45.0), 3),
    ("Pimentón", {"market_container": "frasco", "container_weight_g": "85",
                  "market_packages": [{"unit": "frasco", "grams": 85, "price": 149}]},
     ("frasco", 85.0), 6),
    ("Sazón con culantro y achiote", SAZON, ("caja", 40.0), 3),
    # Control: el pote del comino ES del tamaño del tope, así que lo comprado y lo capado coinciden.
    ("Comino", {"market_container": "pote", "container_weight_g": "28",
                "market_packages": [{"unit": "pote", "grams": 28, "price": 55}]},
     ("pote", 28.0), 2),
]


@pytest.fixture(autouse=True)
def _limpio():
    sc.reset_caps_applied_last_run()
    yield
    sc.reset_caps_applied_last_run()


def _por_peso(name, gramos, master, cycle_days=7):
    """Como la llama el agregador en su ruta de peso (`has_weight`): libras, unidad 'lb', raw 0."""
    return sc.apply_smart_market_units(name, gramos / _LB, "lb", 0.0, master, cycle_days=cycle_days)


# ───────────── 1. el caso del dueño ─────────────

def test_la_caja_que_cubre_la_semana_no_manda_recomprar():
    sc._record_cap_applied(SAZON["name"], 40.0, 28.0, "P6-SPICE-CAP")
    item = _por_peso(SAZON["name"], 28.0, SAZON)
    assert (item["market_unit"], item["market_qty_numeric"], item.get("package_grams")) == ("caja", 1.0, 40.0)
    # El tope se sigue declarando para tooling: se corrige lo que DICE la nota, no la metadata.
    assert item.get("capped_by") == "P6-SPICE-CAP"
    assert (item.get("capped_pre"), item.get("capped_post")) == (40.0, 28.0)
    assert "recompra" not in item["display_qty"], item["display_qty"]
    assert "alcanza" not in item["display_string"], item["display_string"]
    assert item.get("coverage_ok_by_package") is True


@pytest.fixture()
def catalogo_sazon(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: [dict(SAZON)])
    sc.invalidate_master_cache()
    yield
    sc.invalidate_master_cache()


def _sazon_de(res):
    return next(i for i in res if isinstance(i, dict) and "sazón" in str(i.get("name", "")).lower())


def test_el_agregador_de_punta_a_punta(catalogo_sazon):
    """El repro literal. Aquí el tope lo registra el AGREGADOR, no el test: prueba que la comparación
    en gramos se abre en el camino real (el `post` del tope es lo que llega a la función)."""
    item = _sazon_de(sc.aggregate_and_deduct_shopping_list(
        ["1 sobre de sazón con culantro y achiote"], structured=True))
    assert item.get("capped_by") == "P6-SPICE-CAP", item
    assert (item["market_unit"], item["market_qty_numeric"]) == ("caja", 1.0)
    assert item.get("estimated_cost_rd") == 99.0
    assert "recompra" not in item["display_qty"], item["display_qty"]


def test_cuando_de_verdad_falta_el_aviso_sigue(catalogo_sazon):
    """Diez sobres: la necesidad llega en 400 g y la caja de 40 g cubre ~0,7 días de 7. Aviso cierto."""
    item = _sazon_de(sc.aggregate_and_deduct_shopping_list(
        ["10 sobres de sazón con culantro y achiote"], structured=True))
    assert (item["market_unit"], item["market_qty_numeric"]) == ("caja", 1.0)
    assert "alcanza ~1 de 7 días — recompra" in item["display_qty"], item["display_qty"]
    assert "alcanza ~1 de 7 días — recompra" in item["display_string"], item["display_string"]


# ───────────── 2. las otras especias bajo el tope ─────────────

@pytest.mark.parametrize("nombre,fila,envase,dias", _ESPECIAS, ids=[e[0] for e in _ESPECIAS])
def test_con_la_semana_cubierta_calla(nombre, fila, envase, dias):
    """Necesidad 40 g, tope 28 g. Antes: «~5 de 7 — recompra» en las siete."""
    sc._record_cap_applied(nombre, 40.0, 28.0, "P6-SPICE-CAP")
    item = _por_peso(nombre, 28.0, fila)
    assert (item["market_unit"], item.get("package_grams")) == envase
    if envase[1] >= 40.0:
        assert "alcanza" not in item["display_qty"], item["display_qty"]
        assert item.get("coverage_ok_by_package") is True
    else:  # el comino compra 28 g de 40: el aviso es cierto y se queda
        assert "alcanza ~5 de 7 días — recompra" in item["display_qty"], item["display_qty"]


@pytest.mark.parametrize("nombre,fila,envase,dias", _ESPECIAS, ids=[e[0] for e in _ESPECIAS])
def test_los_dias_salen_de_lo_comprado(nombre, fila, envase, dias):
    """Necesidad 100 g, tope 28 g. Antes todas decían «~2 de 7» (28/100), también el pimentón, cuyo
    frasco de 85 g cubre ~6. Sólo el comino conserva el 2: su pote es el tope."""
    sc._record_cap_applied(nombre, 100.0, 28.0, "P6-SPICE-CAP")
    item = _por_peso(nombre, 28.0, fila)
    assert (item["market_unit"], item.get("package_grams"), item["market_qty_numeric"]) == (*envase, 1.0)
    assert f"alcanza ~{dias} de 7 días — recompra" in item["display_qty"], item["display_qty"]
    assert f"alcanza ~{dias} de 7 días — recompra" in item["display_string"], item["display_string"]


# ───────────── 3. donde no hay gramos comparables, lo de antes ─────────────

def test_tope_contado_en_sobres_conserva_la_fraccion_del_tope():
    """La rama de conteo de P6-SPICE-CAP registra SOBRES («10 → 1»): ese `pre` no se compara con
    gramos. Sin gramos comparables, la nota sigue siendo post/pre."""
    sc._record_cap_applied("Comino", 10.0, 1.0, "P6-SPICE-CAP")
    item = sc.apply_smart_market_units("Comino", 0.0, "sobre", 1.0, {})
    assert item.get("capped_by") == "P6-SPICE-CAP"
    assert "alcanza ~1 de 7 días — recompra" in item["display_qty"], item["display_qty"]
    assert item.get("coverage_ok_by_package") is None


def test_envase_recontado_por_unidades_conserva_la_fraccion_del_tope():
    """P2-PACK-UNITS-MATCH recuenta por UNIDADES reales del envase: sus gramos (del SKU) y la necesidad
    (densidad del master) dejan de medir lo mismo. 4 paquetes × 356 g = 1.424 g «cubren» 1.440 g, pero
    son 20 tortillas de las 30 que pide el plan: el aviso tiene que salir."""
    fila = {"market_container": "paquete", "container_weight_g": "356", "density_g_per_unit": "48",
            "market_packages": [{"unit": "paquete", "grams": 356, "label": "Burrito 5 unid 356 gr",
                                 "price": 120}]}
    sc._record_cap_applied("Tortilla de trigo", 1440.0, 960.0, "TEST-CAP")
    item = _por_peso("Tortilla de trigo", 960.0, fila)
    assert item["market_qty_numeric"] == 4.0, item
    assert "alcanza ~5 de 7 días — recompra" in item["display_qty"], item["display_qty"]


# ───────────── 4. la compra no cambia ─────────────

def test_la_compra_del_tope_no_cambia(monkeypatch):
    """Se corrige lo que la lista DICE. Con el aviso apagado por su knob la compra es idéntica."""
    sc._record_cap_applied(SAZON["name"], 400.0, 28.0, "P6-SPICE-CAP")
    con = _por_peso(SAZON["name"], 28.0, SAZON)
    monkeypatch.setattr(sc, "CAPPED_STAPLE_HONESTY", False)
    sin = _por_peso(SAZON["name"], 28.0, SAZON)
    for campo in ("market_qty_numeric", "market_unit", "package_grams", "market_pkg_price_rd"):
        assert con.get(campo) == sin.get(campo), campo


# ───────────── 5. las dos piezas puras ─────────────

@pytest.mark.parametrize("item,esperado", [
    ({"market_qty_numeric": 1.0, "package_grams": "40"}, 40.0),           # la DB lo trae como texto
    ({"market_qty_numeric": 2.0, "market_unit": "pote", "package_grams": 453.6}, 907.2),
    ({"market_qty_numeric": 0.25, "market_unit": "lb"}, 0.25 * _LB),
    ({"market_qty_numeric": 7.0, "market_unit": "Ud."}, None),             # contable sin envase
    ({"market_qty_numeric": "x", "market_unit": "lb"}, None),
    ({}, None),
])
def test_gramos_comprados(item, esperado):
    got = sc._purchased_grams(item)
    assert (got == pytest.approx(esperado)) if esperado is not None else (got is None)


@pytest.mark.parametrize("post,base,esperado", [
    (28.0, {"base_qty": 28.0, "base_unit": "g"}, True),
    (453.592, {"base_qty": 1.0, "base_unit": "lb"}, True),
    (1.0, {"base_qty": 1.0, "base_unit": "sobre"}, False),    # tope contado en sobres
    (28.0, {"base_qty": 18.0, "base_unit": "g"}, False),      # llega otra cantidad que la del tope
    (28.0, {}, False),
])
def test_el_tope_habla_en_gramos(post, base, esperado):
    assert sc._cap_in_grams(post, base) is esperado
