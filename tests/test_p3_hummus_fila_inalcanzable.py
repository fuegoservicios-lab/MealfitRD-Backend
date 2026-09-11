"""[P3-HUMMUS-FILA-INALCANZABLE · 2026-08-23 · CERRADO por P1-DO-DESPENSA-DE-SU-MERCADO · 2026-09-10]

El pool `'DO'` del catálogo sin precio tenía un único token que el bloque del generador nunca podía
usar — y el arreglo que el audit proponía habría borrado el alimento de la lista de la compra.

LO MEDIDO EL 2026-08-23, que confirmó el diagnóstico del audit:

    is_country_catalog_unpriced_item('Hummus', country=c) → DO True · ES/US/MX/PR/CO False
    is_country_catalog_unpriced_item('Hummus')            → True   (sin argumento de país)
    tamaños de pool: ES 32 · US 43 · MX 28 · PR 19 · CO 18 · DO 1
    De las 141 filas sin precio, 140 las reclamaba al menos un país beta; exactamente 1 no la
    reclamaba ninguno: Hummus.

Y las TRES puertas que preguntan por país cortocircuitan antes para DO (`_vc_beta = _vc_country
!= 'DO'` en el catálogo del generador; `cc == 'DO' → return ()` en los condimentos; `_sug_country
!= 'DO'` en la tool de sugerencias), así que la respuesta `country='DO'` no la consumía NADIE en
producción: el único lector del token era la vista PLANA.

POR QUÉ NO SE APLICÓ ENTONCES EL ARREGLO PROPUESTO («sacar su token del pool 'DO'»). Se ejecutó el
agregador REAL contra el catálogo REAL, antes y después de podar:

    con el token:  ['1 paquete (1 lb) de Pechuga de pollo', '¼ lb de Hummus']
    sin el token:  ['1 paquete (1 lb) de Pechuga de pollo']
                   + WARNING [VERIFIED-ONLY-DROP] 'Hummus' excluido de la lista

O sea que la poda no movía una asimetría de sitio: BORRABA el alimento de la lista de la compra en
silencio, el fallo más caro de la doctrina de este repo. La conclusión de aquel día fue que la
asimetría era el DISEÑO (agregador fail-open, generador por-país) y que lo que faltaba era dejar la
decisión escrita y medida — con una condición de reapertura explícita: «si algún día OTRO
mecanismo pasa a conservar esta fila (un precio real, un keep nuevo), la refutación habría dejado
de ser cierta y la decisión se puede reabrir».

EPÍLOGO [P1-DO-DESPENSA-DE-SU-MERCADO · 2026-09-10]: la condición se cumplió. El dueño trajo el
precio de su supermercado (Dietz & Watson 10 oz RD$299 ⇒ `price_source='owner_verified'`,
RD$478,40/lb), el hummus pasó a ser un alimento VERIFICADO y sobrevive a la lista POR PRECIO, no
por el keep. Con eso la poda dejó de borrar nada y se ejecutó: el pool `'DO'` queda VACÍO a
propósito, y un bloque vacío ya no cae a la unión de los seis países (`is not None`, no
truthiness). La asimetría se cerró de verdad. Los tests de abajo anclan el estado NUEVO; la
medición del audit queda arriba como historia — y como la razón por la que la poda NO se hizo
antes de tener el precio.
"""
from __future__ import annotations

import pytest

#: La fila que ningún país beta reclamaba y que el pool nativo conservaba a solas — hasta que tuvo precio.
_HUERFANA = "Hummus"
_BETA = ("ES", "US", "MX", "PR", "CO")


@pytest.fixture(scope="module")
def sc():
    import shopping_calculator as _sc
    return _sc


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setenv("MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP", "true")
    # `tests/conftest.py` apaga VERIFIED-ONLY para preservar el baseline histórico de la suite; sin
    # él no hay drop que medir y los tests del agregador pasarían por la razón equivocada (el
    # default de CÓDIGO es ON desde P1-VERIFIED-ONLY-DEFAULT-ON, que es lo que corre en producción).
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")


# ── A. La asimetría, cerrada: ya nadie reclama la fila como catálogo sin precio ─────────────────

def test_ningun_pais_reclama_ya_la_fila_que_era_huerfana(sc):
    """[P1-DO-DESPENSA-DE-SU-MERCADO] Antes: DO True · beta False. Ahora: False para los seis. Si
    DO vuelve a True es que alguien re-añadió el token a un alimento CON precio — la colisión
    priced+unpriced que caza `test_i2_registry_collision_sweep_extendido_a_aliases`."""
    assert sc.is_country_catalog_unpriced_item(_HUERFANA, country="DO") is False
    for cc in _BETA:
        assert sc.is_country_catalog_unpriced_item(_HUERFANA, country=cc) is False, (
            f"{cc} pasó a reclamar {_HUERFANA}: si eso es intencional, muévelo de pool a sabiendas")


def test_sin_pais_la_fila_ya_no_necesita_el_keep(sc):
    """La vista PLANA era el único lector real del token. Con precio, el alimento sobrevive a la
    lista por la vía normal (verificado) y el keep no tiene nada que rescatar."""
    assert sc.is_country_catalog_unpriced_item(_HUERFANA) is False


def test_el_pool_nativo_esta_vacio_a_proposito(sc):
    """El estado que el audit propuso y que sólo se pudo ejecutar con el precio en la mano. El
    bloque existe (RD es un país conocido) y está vacío (no tiene altas sin precio) — dos cosas
    distintas, y la segunda no puede degradar a «no sé qué país es»."""
    assert "DO" in sc._COUNTRY_CATALOG_UNPRICED_BY_COUNTRY
    assert sc._COUNTRY_CATALOG_UNPRICED_BY_COUNTRY["DO"] == ()
    assert sc.is_country_catalog_unpriced_item("Jamón serrano", country="DO") is False, (
        "el bloque vacío cayó a la unión de los seis países")


# ── B. Nadie pregunta por país siendo DO ────────────────────────────────────────────────────────

def test_la_puerta_de_condimentos_cortocircuita_para_el_pais_nativo():
    """Una de las tres puertas que sí pasan país; su respuesta para DO es vacía POR CONTRATO, no
    por lo que haya en el pool."""
    from constants import _country_catalog_condiment_patterns
    assert _country_catalog_condiment_patterns("DO") == ()
    assert _country_catalog_condiment_patterns("do") == (), "y por la puerta canónica, no por la grafía"


# ── C. La refutación, invertida: ahora sobrevive por precio ─────────────────────────────────────

@pytest.mark.e2e
def test_el_alimento_que_era_huerfano_sigue_llegando_a_la_lista_de_la_compra(sc):
    """El guard de verdad, que sigue valiendo con el pool podado: el alimento NO desaparece de la
    lista. Si esto se pone rojo, el hummus perdió el precio que trajo el dueño."""
    lista = sc.aggregate_and_deduct_shopping_list([f"120 g de {_HUERFANA}",
                                                   "150 g de Pechuga de pollo"])
    if not lista:
        pytest.skip("agregador sin catálogo (¿pool de Neon sin abrir?)")
    assert any(_HUERFANA.lower() in str(k).lower() for k in lista), (
        f"{_HUERFANA} desapareció de la lista de la compra: {sorted(lista)}")


@pytest.mark.e2e
def test_sobrevive_por_precio_y_se_cotiza(sc):
    """La medición del 2026-08-23, repetida tras el precio: la poda YA está hecha (pool 'DO' vacío)
    y el alimento sigue en la lista — y además con coste, que es lo que el keep nunca pudo darle.
    Es la refutación de entonces, invertida a sabiendas."""
    assert sc._is_verified_for_shopping(_HUERFANA) is True, "el hummus perdió el precio del dueño"
    result = sc.aggregate_and_deduct_shopping_list([f"120 g de {_HUERFANA}",
                                                    "150 g de Pechuga de pollo"], structured=True)
    items = result.get("items") if isinstance(result, dict) else result
    if not items:
        pytest.skip("agregador sin catálogo (¿pool de Neon sin abrir?)")
    fila = next((it for it in items if _HUERFANA.lower() in str(it.get("name", "")).lower()), None)
    assert fila is not None, f"{_HUERFANA} desapareció de la lista: {items}"
    assert (fila.get("estimated_cost_rd") or 0) > 0, "el hummus volvió a salir sin coste"
