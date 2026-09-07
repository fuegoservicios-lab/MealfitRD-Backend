# -*- coding: utf-8 -*-
"""[P1-YOGURT-PRECIO-REAL · 2026-09-07] Los tres yogurts tenían el MISMO precio, y era el del griego.

Las tres filas lácteas de `master_ingredients` llevaban `price_per_lb = 0` y
`price_per_unit = 100`, las únicas 3 de 348 con ese valor y las tres marcadas
`price_confidence = 'high'`. Ese 100 **no era inventado**: es el precio real del Yoplait
GRIEGO (RD$100 / 150 g), copiado a las tres. El motor acababa tasando el yogurt normal como
un griego.

Y la diferencia del 12 % que el motor «veía» entre ambos no era un precio: era el DIVISOR.
`_budget_build_master_price_map` calcula `price_per_unit × 453,592 / (density_g_per_unit or
container_weight_g)`, y las filas de griego declaran `density=170` junto a `container=150`.
Con el MISMO RD$100, el normal dividía por 150 y el griego por 170:

    normal  100 × 453,592 / 150 = 302,39 RD$/lb
    griego  100 × 453,592 / 170 = 266,82 RD$/lb   <- «más barato» por una columna de densidad

Un placeholder con sello `high` es peor que un NULL: pasa cualquier auditoría de «¿qué precios
no me fío?» y encima produce una ORDENACIÓN falsa que `_apply_budget_cheapen_pass` obedece.

Este fichero ancla el MÉTODO con el que se derivaron los precios nuevos desde
`supermarket_products` (`scripts/seed_yogurt_precios_2026_09_07.py`), no los números en la DB:
un test que consulte la fila viva mide el entorno, no el contrato — es el error que
`test_p1_measurement_integrity` ya tuvo que corregir hoy.
"""
import importlib.util
from pathlib import Path

import pytest

_SEED = Path(__file__).resolve().parents[1] / "scripts" / "seed_yogurt_precios_2026_09_07.py"


@pytest.fixture(scope="module")
def seed():
    """Carga el script como módulo. No ejecuta nada: `main()` sólo corre bajo `__main__`."""
    spec = importlib.util.spec_from_file_location("seed_yogurt_precios", _SEED)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────────────────
# El gramaje estaba escrito en el texto, no en la columna.
# ─────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("texto,esperado", [
    ("Pote 6 Oz", 170.097),
    ("Vaso Natural 0% 150 gr", 150.0),
    ("Tarro Light Vainilla 32 Oz", 907.184),
    ("Botella Yopsi Fresa 1 Lt", 1000.0),
    ("Botella Yopsi Natural 1/2 Gl", 1892.705),
    ("Galón Ciruela Pasa (3.4 Kg)", 3400.0),
    ("Bebible Piña Colada 250 Ml", 250.0),
])
def test_el_gramaje_se_lee_del_texto_de_la_presentacion(texto, esperado, seed):
    """`size_grams` sólo lo tenían 43 de 152 productos; el resto lo dice la presentación.

    Sin leerlo, la mediana se calculaba sobre un tercio de la muestra."""
    g, origen = seed.gramos(None, texto)
    assert g == pytest.approx(esperado, rel=1e-3)
    assert origen == "texto"


def test_la_columna_manda_sobre_el_texto(seed):
    """Si `size_grams` viene poblado es el dato bueno; el texto es el respaldo."""
    g, origen = seed.gramos(150.0, "Pote 6 Oz")
    assert (g, origen) == (150.0, "columna")


def test_sin_medida_devuelve_none_nunca_cero(seed):
    """«Galón Natural» no lleva cifra y «Pack 4 Vasos» no lleva tamaño.

    Un cero aquí sería una división por cero o un precio infinito; None los deja fuera de la
    mediana, que es lo honesto — un dato ausente no es un dato de valor cero."""
    for texto in ("Galón Natural", "Pack 4 Vasos Vainilla", "", None):
        assert seed.gramos(None, texto) == (None, None)


# ─────────────────────────────────────────────────────────────────────────────────────────
# Las dos decisiones de método.
# ─────────────────────────────────────────────────────────────────────────────────────────
def _p(marca, presentacion, precio, gramos=None):
    return {"brand": marca, "presentation": presentacion, "price_rd": precio,
            "size_grams": gramos}


def test_lo_bebible_no_entra_en_la_mediana(seed):
    """81 de los 115 «Yogurt Regular» son botellas y galones. Un plan no come eso con cuchara,
    y mezclarlos hundía la mediana de 141 a 84 RD$/lb: el galón barato arrastrando al vasito."""
    filas = [
        _p("Yoka", "Vaso Light Vainilla 6 Oz", 50),         # 133,3 RD$/lb
        _p("Yoka", "Botella Yopsi Fresa 1 Gl", 365),        # bebible, fuera
        _p("Yoplait", "Bebible Piña Colada 250 Ml", 45),    # bebible, fuera
    ]
    valor, det = seed.precio_por_lb_por_marca(filas)
    assert det["bebible"] == 2
    assert valor == pytest.approx(133.3, rel=1e-2)


def test_la_mediana_es_por_marca_no_por_sku(seed):
    """Contar productos pesa a cada marca por cuántos SABORES stockea, no por su precio.

    Chobani tiene 16 vasos de griego y Asturiana 3 de regular: una mediana por SKU es una
    encuesta de surtido disfrazada de encuesta de precio. Aquí la marca cara tiene 4 SKUs y la
    barata 1; por SKU saldría ~400, por marca sale el punto medio entre las dos marcas."""
    filas = [_p("Cara", f"Vaso Sabor{i} 150 gr", 132.3, 150) for i in range(4)]
    filas.append(_p("Barata", "Vaso Natural 150 gr", 33.1, 150))
    valor, det = seed.precio_por_lb_por_marca(filas)
    assert set(det["marcas"]) == {"Cara", "Barata"}
    assert valor == pytest.approx((400.0 + 100.0) / 2, rel=1e-2)


def test_sin_ninguna_presentacion_medible_devuelve_none(seed):
    """Un alimento cuyo texto no dice tamaño no recibe un precio inventado."""
    valor, det = seed.precio_por_lb_por_marca([_p("X", "Pack 4 Vasos", 368)])
    assert valor is None
    assert det["sin_medida"] == 1


# ─────────────────────────────────────────────────────────────────────────────────────────
# La invariante que el dueño pidió: el griego tiene que salir MÁS CARO que el normal.
# ─────────────────────────────────────────────────────────────────────────────────────────
def test_el_griego_sale_mas_caro_que_el_normal_con_los_precios_de_la_tienda(seed):
    """Es la razón entera del P-fix: sin esto, `_apply_budget_cheapen_pass` no puede dar
    yogurt normal a un presupuesto bajo, porque cree que el griego es más barato.

    Los datos son los reales de `supermarket_products` reducidos a una marca por bando."""
    regular = [_p("Asturiana", "Vaso Fresa 125 gr", 39, 125),
               _p("Yoka", "Vaso Light Vainilla 6 Oz", 50)]
    griego = [_p("Yoplait", "Vaso Natural 0% 150 gr", 100, 150),
              _p("Chobani", "Vaso Non Fat Plain 5.3 Oz", 185)]
    v_reg, _ = seed.precio_por_lb_por_marca(regular)
    v_gri, _ = seed.precio_por_lb_por_marca(griego)
    assert v_gri > v_reg, "el griego no puede salir más barato que el yogurt normal"
    assert v_gri / v_reg > 1.5, "la brecha tiene que ser lo bastante ancha para mover el pase"


# ─────────────────────────────────────────────────────────────────────────────────────────
# Procedencia del yogurt de cabra.
# ─────────────────────────────────────────────────────────────────────────────────────────
def test_los_micros_de_cabra_no_son_los_de_vaca(seed):
    """USDA no tiene yogurt de cabra genérico, así que los 8 micros que el esquema exige NOT
    NULL salen de la LECHE de cabra (usda:171278) — no de la fila `Yogurt` de vaca.

    Ahí está justo la diferencia que importa: la fila de vaca lleva folato 7,0 µg y B12
    0,37 µg; la leche de cabra, 1,0 y 0,07. Copiar los de vaca habría inflado el folato 7× y
    la B12 5× en una app de nutrición."""
    assert seed.MICROS_CABRA["folate_mcg_dfe_per_100g"] == 1.0
    assert seed.MICROS_CABRA["vitamin_b12_mcg_per_100g"] == 0.07


def test_el_yogurt_de_cabra_no_estampa_un_fdc_id_ajeno(seed):
    """Un `fdc_id` afirma que la fila ES ese alimento, y Redwood Hill no es Deliciel.

    Estampar uno ajeno es exactamente lo que `P1-BEDCA-DEPROXY-ES` encontró costando 47 filas
    (un id sustituyendo a SIETE embutidos). La procedencia va en `nutrition_source_ref`, que es
    prosa y no se confunde con una identidad."""
    src = _SEED.read_text(encoding="utf-8")
    assert '"fdc_id"' not in src, "el INSERT de Yogurt de cabra no debe declarar fdc_id"
    assert "nutrition_source_ref" in src
    assert "2422160" in src and "171278" in src, "las dos fuentes deben quedar declaradas"


def test_el_alias_de_cabra_no_secuestra_al_queso_de_cabra(seed):
    """Frases completas, nunca «cabra» a secas: casaría con «queso de cabra», que es otro
    alimento. Es la clase de colisión por subcadena que este repo lleva 19 veces documentada."""
    src = _SEED.read_text(encoding="utf-8")
    inicio = src.index('"aliases": [')
    alias_txt = src[inicio:src.index("]", inicio)]
    for suelta in ('"cabra"', "'cabra'"):
        assert suelta not in alias_txt, "«cabra» a secas colisionaría con «queso de cabra»"
