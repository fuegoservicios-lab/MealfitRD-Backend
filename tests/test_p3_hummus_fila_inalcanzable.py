"""[P3-HUMMUS-FILA-INALCANZABLE · 2026-08-23] El pool `'DO'` del catálogo sin precio tiene un único
token que el bloque del generador nunca puede usar — y el arreglo que el audit proponía habría
borrado el alimento de la lista de la compra.

LO MEDIDO, que confirma el diagnóstico del audit:

    is_country_catalog_unpriced_item('Hummus', country=c) → DO True · ES/US/MX/PR/CO False
    is_country_catalog_unpriced_item('Hummus')            → True   (sin argumento de país)
    tamaños de pool: ES 32 · US 43 · MX 28 · PR 19 · CO 18 · DO 1
    De las 141 filas sin precio, 140 las reclama al menos un país beta; exactamente 1 no la
    reclama ninguno: Hummus.

Y las TRES puertas que preguntan por país cortocircuitan antes para DO (`_vc_beta = _vc_country
!= 'DO'` en el catálogo del generador; `cc == 'DO' → return ()` en los condimentos; `_sug_country
!= 'DO'` en la tool de sugerencias), así que la respuesta `country='DO'` no la consume NADIE en
producción: el único lector del token es la vista PLANA.

POR QUÉ NO SE APLICA EL ARREGLO PROPUESTO («sacar su token del pool 'DO'»). Se ejecutó el
agregador REAL contra el catálogo REAL, antes y después de podar:

    con el token:  ['1 paquete (1 lb) de Pechuga de pollo', '¼ lb de Hummus']
    sin el token:  ['1 paquete (1 lb) de Pechuga de pollo']
                   + WARNING [VERIFIED-ONLY-DROP] 'Hummus' excluido de la lista

O sea que la poda no mueve una asimetría de sitio: BORRA el alimento de la lista de la compra en
silencio, que es el fallo más caro de la doctrina de este repo, y además convierte la fila en
huérfana para `test_p1_country_catalog_by_country.py::test_ninguna_fila_del_catalogo_se_queda_sin_pais`.

La otra dirección —ofrecerle Hummus al generador dominicano— rompe la byte-identidad DO y
contradice la frase del propio bloque («SOLO puede comprar alimentos con precio verificado»).

CONCLUSIÓN: la asimetría es el DISEÑO, no el defecto. El agregador es fail-open («si un alimento
acaba en la lista, consérvalo venga de donde venga») y el generador es por-país («qué OFRECER»).
Lo que faltaba no era un arreglo: era que la decisión estuviera escrita y medida. Este fichero la
escribe, y el test del final se pone rojo el día que alguien intente la poda.
"""
from __future__ import annotations

import pytest

#: La fila que ningún país beta reclama. El audit la identificó midiendo las 141 filas sin precio.
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
    # él no hay drop que medir y los dos tests del agregador pasarían por la razón equivocada (el
    # default de CÓDIGO es ON desde P1-VERIFIED-ONLY-DEFAULT-ON, que es lo que corre en producción).
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")


# ── A. La asimetría, tal cual la midió el audit ─────────────────────────────────────────────────

def test_la_fila_huerfana_la_reclama_solo_el_pool_nativo(sc):
    assert sc.is_country_catalog_unpriced_item(_HUERFANA, country="DO") is True
    for cc in _BETA:
        assert sc.is_country_catalog_unpriced_item(_HUERFANA, country=cc) is False, (
            f"{cc} pasó a reclamar {_HUERFANA}: si eso es intencional, muévelo de pool a sabiendas")


def test_sin_pais_la_fila_sobrevive_al_keep(sc):
    """La vista PLANA es el único lector real del token, y es la que decide si el alimento
    sobrevive a la lista de la compra."""
    assert sc.is_country_catalog_unpriced_item(_HUERFANA) is True


# ── B. Nadie pregunta por país siendo DO ────────────────────────────────────────────────────────

def test_la_puerta_de_condimentos_cortocircuita_para_el_pais_nativo():
    """Una de las tres puertas que sí pasan país; su respuesta para DO es vacía POR CONTRATO, no
    por lo que haya en el pool."""
    from constants import _country_catalog_condiment_patterns
    assert _country_catalog_condiment_patterns("DO") == ()
    assert _country_catalog_condiment_patterns("do") == (), "y por la puerta canónica, no por la grafía"


# ── C. La refutación, ejecutada ─────────────────────────────────────────────────────────────────

@pytest.mark.e2e
def test_el_alimento_huerfano_sigue_llegando_a_la_lista_de_la_compra(sc):
    """El guard de verdad: si alguien poda el pool 'DO', esto se pone rojo porque el alimento
    desaparece de la lista."""
    lista = sc.aggregate_and_deduct_shopping_list([f"120 g de {_HUERFANA}",
                                                   "150 g de Pechuga de pollo"])
    if not lista:
        pytest.skip("agregador sin catálogo (¿pool de Neon sin abrir?)")
    assert any(_HUERFANA.lower() in str(k).lower() for k in lista), (
        f"{_HUERFANA} desapareció de la lista de la compra: {sorted(lista)}")


@pytest.mark.e2e
def test_podar_el_pool_nativo_borraria_el_alimento_de_la_lista(sc, monkeypatch):
    """La medición que refuta el arreglo propuesto por el audit, hecha reproducible. Si algún día
    OTRO mecanismo pasa a conservar esta fila (un precio real, un keep nuevo), este test avisa: la
    refutación habría dejado de ser cierta y la decisión se puede reabrir."""
    podado = dict(sc._COUNTRY_CATALOG_UNPRICED_BY_COUNTRY)
    podado["DO"] = ()
    monkeypatch.setattr(sc, "_COUNTRY_CATALOG_UNPRICED_BY_COUNTRY", podado)
    monkeypatch.setattr(sc, "_COUNTRY_CATALOG_UNPRICED_TOKENS", tuple(dict.fromkeys(
        t for ts in podado.values() for t in ts)))
    assert sc.is_country_catalog_unpriced_item(_HUERFANA) is False
    lista = sc.aggregate_and_deduct_shopping_list([f"120 g de {_HUERFANA}",
                                                   "150 g de Pechuga de pollo"])
    if not lista:
        pytest.skip("agregador sin catálogo (¿pool de Neon sin abrir?)")
    assert not any(_HUERFANA.lower() in str(k).lower() for k in lista), (
        "la poda ya NO borra el alimento de la lista: la refutación de P3-HUMMUS-FILA-INALCANZABLE "
        "dejó de ser cierta y la asimetría se puede cerrar de verdad")
