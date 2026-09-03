"""[P1-BETA-PRICE-LEAKS · 2026-08-21] Tres fugas de dinero dominicano a un usuario beta, con el
mismo mecanismo detrás: el modo `beta_no_prices` anulaba el NÚMERO y dejaba pasar todo lo demás.

Fase 1 T7 montó el strip de precios y funciona: los 2 planes beta vivos tienen 0 ítems con
`estimated_cost_rd`. Lo que no cubrió es que un precio no viaja sólo como número.

  1. LA MARCA Y EL ENVASE. `_strip_prices_for_beta_pricing_mode` anula `estimated_cost_rd` y
     `estimated_cost`, nunca `display_qty` — y `display_qty` lleva la etiqueta del SKU del que
     salió ese precio. Medido en producción: **30 de 48 ítems** de la lista del plano ES y 15 de
     25 del US llevan marca. El usuario imprime su PDF y se lo lleva al súper con «1 cartón
     (1 Lt · Wala) de Leche descremada», «1 pote (Cremosa 16 Oz · Zerca) de Mantequilla de maní»,
     «1 funda (Organics 12 Oz · Goya) de Semillas de chía». Wala, Zerca, Sosua, Jazma y Rica son
     marcas de casa de supermercados DOMINICANOS que no existen en España, y «funda» es el
     dominicanismo de bolsa. No las eligió —son los defaults más baratos del catálogo RD— y no
     puede quitarlas, porque el panel que las gestionaría está oculto por ser beta.

     Por qué sobrevivió a la auditoría de frontend: esa dimensión verificó —correctamente— que el
     panel «Marcas» está oculto. Las dos cosas son ciertas a la vez: **se auditó el panel, no la
     cadena de texto**.

  2. LA NEVERA. `POST /api/supermarket/match` devuelve marca + precio y tres superficies lo
     pintan con «RD$» sin condición alguna (`BrandSelect`, `Pantry`, el paso «Prepara tu Nevera»).
     `grep -n "_pricing_mode" frontend/src` da exactamente 3 consumidores, los 3 en Dashboard:
     ninguna de esas pantallas lo lee. Ocurre en la MISMA sesión en la que su PDF le dice «España
     está en beta — pronto añadiremos los precios nativos de tu súper». Y no es teórico: 21 de 25
     y 42 de 48 ítems de los planes beta hacen match con SKUs dominicanos.

     Se arregla SERVER-SIDE, en el endpoint, no en las tres pantallas: es el mismo argumento de
     choke point que justificó `_strip_prices_for_beta_pricing_mode` para el agregador — cubre las
     tres y las que vengan. Se suprime el PRECIO, no la marca: elegir marca sigue teniendo sentido.

  3. EL PROMPT — **la auditoría se equivocó aquí, y queda anclado como control**. Decía que el
     prompt le mete «RD$X (pesos dominicanos)» a un plan beta. Ejecutando el builder: el bloque de
     precios en RD$ ya está gateado por país desde F1-T7, y `build_budget_context` renderiza
     FIELMENTE la moneda que recibe («300 EUR» con EUR). Si un usuario beta ve «RD$» es porque su
     `budgetCurrency` es DOP, y eso pasa porque el wizard pregunta el presupuesto DIEZ pasos antes
     que el país. Se arregla ahí, no aquí. Ver la sección C.

Cubre:
  A. La marca desaparece del `display_qty` en beta y sobrevive en DO.
  B. El endpoint de match no devuelve precio para un usuario beta, y sí la marca.
  C. Controles de lo que la auditoría creía roto y NO lo está (el prompt).
  D. Parser-based: las dos superficies tocadas leen el literal SSOT.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SC_PATH = _BACKEND_ROOT / "shopping_calculator.py"
_SUPERMARKET_PATH = _BACKEND_ROOT / "routers" / "supermarket.py"
_PG_PATH = _BACKEND_ROOT / "prompts" / "plan_generator.py"


@pytest.fixture(scope="module")
def sc():
    import shopping_calculator as _sc
    return _sc


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


# ── A. La marca y el envase ─────────────────────────────────────────────────────────────────────

def test_el_strip_beta_quita_la_marca_del_display(sc):
    """RED pre-fix: `_strip_prices_for_beta_pricing_mode` sólo tocaba los dos campos de coste.
    El SKU del que salió ese coste seguía impreso en `display_qty`, que es lo que el usuario LEE
    y lo que viaja al PDF."""
    res = [{"name": "Leche descremada", "display_qty": "1 cartón (1 Lt · Wala)",
            "estimated_cost_rd": 95.0},
           {"name": "Mantequilla de maní", "display_qty": "1 pote (Cremosa 16 Oz · Zerca)",
            "estimated_cost_rd": 210.0}]
    sc._strip_prices_for_beta_pricing_mode(res)
    for it in res:
        assert it["estimated_cost_rd"] is None
        assert "·" not in it["display_qty"], (
            f"la marca sigue impresa: {it['display_qty']!r}"
        )


def test_el_strip_beta_conserva_la_presentacion_util(sc):
    """No se borra el envase entero: «1 cartón (1 Lt)» le sigue diciendo al usuario qué comprar.
    Lo que sobra es la marca de casa de un supermercado que no es el suyo."""
    res = [{"name": "Leche descremada", "display_qty": "1 cartón (1 Lt · Wala)"}]
    sc._strip_prices_for_beta_pricing_mode(res)
    d = res[0]["display_qty"]
    assert "cartón" in d and "1 Lt" in d, f"se perdió la presentación: {d!r}"


def test_el_strip_beta_no_rompe_un_display_sin_marca(sc):
    """La mayoría de los ítems no llevan etiqueta de marca: no deben cambiar."""
    res = [{"name": "Acelgas", "display_qty": "½ lb"}]
    sc._strip_prices_for_beta_pricing_mode(res)
    assert res[0]["display_qty"] == "½ lb"


def test_el_strip_beta_funciona_sobre_el_dict_categorizado(sc):
    """El agregador devuelve `dict[categoría, list]` cuando `categorize=True`: el strip ya
    recorría las dos formas y la de la marca tiene que hacerlo igual."""
    res = {"PROTEÍNAS": [{"name": "Pollo", "display_qty": "2 lb (Bandeja · Rica)",
                          "estimated_cost_rd": 300.0}]}
    sc._strip_prices_for_beta_pricing_mode(res)
    assert "·" not in res["PROTEÍNAS"][0]["display_qty"]


# ── B. La Nevera / el endpoint de match ─────────────────────────────────────────────────────────

def test_el_endpoint_de_match_suprime_el_precio_para_un_pais_beta():
    """El fix va en el endpoint —choke point— y no en las tres pantallas que lo pintan: el mismo
    argumento que justificó el strip del agregador. Aquí se ancla que el helper exista y sea
    país-consciente; las pantallas se cubren por construcción."""
    import routers.supermarket as sup
    assert hasattr(sup, "_strip_prices_for_beta_match"), (
        "no existe el strip del endpoint de match"
    )
    # La forma REAL del payload es `matches: dict[nombre_pedido, list[variantes]]` — la primera
    # versión de este test lo modeló como una lista y pasó a verde contra un helper que no la
    # tocaba: un fake que no modela la estructura real no prueba nada sobre ella.
    payload = {"matches": {"pollo": [{"food_name": "Pollo", "brand": "Rica", "price_rd": 185.0,
                                      "presentation": "2 lb"}]}}
    sup._strip_prices_for_beta_match(payload, "ES")
    m = payload["matches"]["pollo"][0]
    assert m["price_rd"] is None, "el precio en RD$ sigue viajando a un usuario beta"
    assert m["brand"] == "Rica", "se suprime el PRECIO, no la marca: elegir marca sigue sirviendo"


def test_el_endpoint_de_match_conserva_el_precio_para_do():
    import routers.supermarket as sup
    payload = {"matches": {"pollo": [{"food_name": "Pollo", "brand": "Rica", "price_rd": 185.0}]}}
    sup._strip_prices_for_beta_match(payload, "DO")
    assert payload["matches"]["pollo"][0]["price_rd"] == 185.0


# ── C. El prompt: VERIFICADO, y NO es un gap ────────────────────────────────────────────────────
#
# La auditoría listó como P1 que «el prompt del LLM recibe RD$X (pesos dominicanos) al generar un
# plan de país beta». Lo comprobé ejecutando el builder y NO es un defecto de este código:
#
#   · `build_prices_context` —la inyección del catálogo entero con precios en RD$— YA está
#     gateada por país desde F1-T7 (`has_native_prices` en `_build_shared_context`), con su
#     propio guard vivo. Un usuario beta no la recibe.
#   · `build_budget_context` renderiza FIELMENTE la moneda que le dan: con `budgetCurrency='EUR'`
#     emite «300 EUR», no «RD$300». Medido.
#
# Lo que sí ocurre hoy es que un usuario beta LLEGA con `budgetCurrency='DOP'`, porque el wizard
# le pregunta el presupuesto DIEZ pasos antes que el país (P1-QCOUNTRY-BEFORE-BUDGET, gap
# abierto). Entonces el prompt dice «RD$» — correctamente, para la moneda que tiene.
#
# Arreglarlo AQUÍ sería tapar el síntoma: el prompt pasaría a mentir en la otra dirección,
# renombrando a euros un monto que el usuario tecleó creyendo que eran pesos. La corrección vive
# en el orden del wizard. Estos dos controles anclan el estado correcto para que nadie «arregle»
# este archivo por el camino equivocado.

def test_el_bloque_de_precios_en_rd_sigue_gateado_por_pais():
    """Control: la inyección del catálogo con precios en RD$ NO llega a un país beta (F1-T7)."""
    src = (_BACKEND_ROOT / "graph_orchestrator.py").read_text(encoding="utf-8", errors="replace")
    i = src.index('"prices_context":')
    ventana = src[i:i + 400]
    # [P3-PRICING-MODE-SSOT-BLANKET · 2026-08-22] Este caso anclaba el literal `has_native_prices`,
    # o sea la consulta A MANO del flag. El comentario de `pricing_mode_for_country` prohíbe por
    # escrito ese «2º chequeo» —es la 2ª tabla de `P1-DIET-CANON-SSOT`— así que el guard exigía
    # justo lo que había que quitar. La propiedad que defiende no cambia: el bloque de precios en
    # RD$ sigue gateado por país; lo que cambia es que ahora pasa por la ÚNICA puerta.
    assert "pricing_mode_for_country" in ventana and "beta_no_prices" in ventana, (
        "el bloque de precios en RD$ dejó de gatearse por país: un usuario beta recibiría el "
        "catálogo dominicano entero con importes"
    )


def test_el_prompt_de_presupuesto_renderiza_la_moneda_que_recibe(knob_on):
    """Control: el builder es fiel. Con EUR dice EUR; con DOP dice RD$. Si un usuario beta ve
    «RD$» es porque su `budgetCurrency` es DOP — y eso se arregla en el wizard, no aquí."""
    from prompts.plan_generator import build_budget_context
    base = {"budget": "custom", "budgetAmount": "300", "householdSize": 1}
    eur = build_budget_context({**base, "country": "ES", "budgetCurrency": "EUR"})
    dop = build_budget_context({**base, "country": "DO", "budgetCurrency": "DOP"})
    assert "EUR" in eur and "RD$" not in eur
    assert "RD$" in dop


# ── E. Parser-based: el literal SSOT, no un segundo chequeo a mano ──────────────────────────────

def test_las_tres_superficies_leen_el_ssot_del_modo_beta():
    """El comentario de `pricing_mode_for_country` lo dice sin ambigüedad: un 2º chequeo
    `has_native_prices` a mano en cualquiera de estos sitios sería la segunda tabla que
    P1-DIET-CANON-SSOT ya pagó una vez."""
    for ruta in (_SC_PATH, _SUPERMARKET_PATH):
        src = ruta.read_text(encoding="utf-8", errors="replace")
        assert "P1-BETA-PRICE-LEAKS" in src, f"{ruta.name} no declara el marker"
    sup = _SUPERMARKET_PATH.read_text(encoding="utf-8", errors="replace")
    assert "pricing_mode_for_country" in sup, (
        "el endpoint de match decide el modo beta sin el literal SSOT"
    )
