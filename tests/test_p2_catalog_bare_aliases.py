"""[P2-CATALOG-BARE-ALIASES · 2026-08-21] «Frijoles», «chorizo», «tahini» se caen de la lista — y
la mitad de las veces eso es lo CORRECTO.

La auditoría lo listó como «caen en silencio: ninguna fila reclama el término desnudo», con la
lección de método por delante: *el harness decía 0 drops porque comparaba su propia lista curada
contra el catálogo* — un 0/0 no prueba que no haya drops en producción, prueba que la lista y el
catálogo están de acuerdo entre sí.

MEDIDO, Y CORRIGE EL DIAGNÓSTICO EN DOS PUNTOS:

**1. No caen en silencio.** `record_verified_only_drop` los cuenta y el cron `_creativity_kpi_job`
emite el top-N a `pipeline_metrics`; además `_filter_expected_to_shopping_survivors` emite el WARN
`VERIFIED-ONLY-GUARD-BLIND`. Probado de punta a punta con 12 términos: **12 de 12 registrados,
cero sin rastro**. La señal existe; lo que falta es el DATO al que apunta.

**2. Para los términos DESNUDOS, caerse es la conducta correcta.** El catálogo tiene seis chorizos
(español, mexicano, verde, santarrosano, chistorra, sobrasada) y nueve chiles. Darle a uno de ellos
el alias bare «chorizo» haría que CUALQUIER mención colapsara a esa fila: un español pidiendo
chorizo se llevaría el santarrosano colombiano. Es «un alias bare es un arma», que en Fase 2 costó
cuatro veces. Entre comprar el alimento equivocado y no comprarlo con telemetría, lo segundo es
estrictamente mejor — y es lo que pasa hoy.

Así que el gap real NO es «el resolvedor está roto». Son dos cosas distintas, y sólo la segunda es
deuda de datos:
  · **Ambiguo por diseño** (`chorizo`, `chile`, `frijoles`, `pavo`, `guisante`): resolver sería el
    bug. La palanca está aguas arriba — el prompt no debería emitir el término desnudo.
  · **Sin fila** (`tahini`, `merluza`, `albaricoque`, `alubias`, `ternera`, `frutilla`…): el
    catálogo no tiene el alimento. Alta con procedencia verificable, o sea curación.

QUÉ HACE ESTE FICHERO. Rompe la circularidad del harness: la lista de términos se escribe
**independientemente del catálogo** —son palabras que un plan de ES/MX/CO/PR puede emitir— y se
fija el veredicto actual de cada una. Si mañana una que hoy resuelve deja de resolver, falla. Si
una que hoy cae empieza a resolver, también falla — y entonces alguien tiene que mirar SI resolvió
a la fila correcta o a una arbitraria. Un caracterización, no un umbral: un umbral dejaría pasar
justo el caso peligroso (que `chorizo` empiece a resolver a un chorizo cualquiera).
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def sc():
    import shopping_calculator as _sc
    if not (_sc.get_master_ingredients() or []):
        pytest.skip("catálogo no disponible (sin DB)")
    return _sc


@pytest.fixture(scope="module")
def nombres(sc):
    return {r["name"] for r in (sc.get_master_ingredients() or [])}


# Términos escritos SIN mirar el catálogo: son las palabras que un plan de ES/MX/CO/PR puede
# emitir. Ésa es toda la gracia — medir con la lista que el catálogo ya satisface es medir el
# acuerdo de la lista consigo misma.
_RESUELVEN = [
    "calabacin", "pimiento", "aguacate", "elote", "maiz", "cilantro", "perejil", "papa",
    "patata", "camote", "yuca", "calabaza", "champinon", "yogur", "garbanzo", "ejotes",
    "judias verdes", "palta", "melocoton", "lenteja", "requeson",
]

# Caen hoy. La columna dice POR QUÉ, y es la que separa deuda de datos de conducta correcta.
_CAEN = {
    # ── Ambiguo por diseño: resolver sería el bug ────────────────────────────────────────────
    "chorizo": "6 filas de chorizo compiten; el bare se llevaría una arbitraria",
    "chile": "9 filas de chile compiten",
    "frijoles": "5 filas de frijol/habichuela compiten",
    "pavo": "sólo hay 'Pavo molido' y 'Pavochón', no pavo genérico",
    "guisantes": "sólo hay 'Guisantes secos', otro alimento (341 kcal)",
    "chicharos": "ídem, y el singular resolvía a Chicharrón — ver P2-CHICHARO-CHICHARRON",
    "manteca": "en ES es grasa de cerdo, en AR mantequilla: ambiguo ENTRE PAÍSES",
    # ── Sin fila: deuda de datos, alta con procedencia ───────────────────────────────────────
    "tahini": "sin fila (hay Ajonjolí, que es la semilla, no la pasta)",
    "merluza": "sin fila",
    "albaricoque": "sin fila",
    "damasco": "sin fila (mismo alimento que albaricoque)",
    "alubias": "sin fila propia",
    "arvejas": "sin fila",
    "ternera": "sin fila (hay Carne de res, que no es lo mismo)",
    "puerco": "sin fila bare (hay Cerdo, Lomo de cerdo…)",
    "chancho": "sin fila",
    "guajolote": "sin fila",
    "frutilla": "sin fila (hay Fresa; es el nombre del Cono Sur)",
    "zumo": "sin fila (y es una categoría, no un alimento)",
}


def _resuelve(sc, nombres, termino: str) -> bool:
    return str(sc.normalize_name(termino) or "") in nombres


@pytest.mark.parametrize("termino", _RESUELVEN)
def test_los_terminos_regionales_que_resuelven_siguen_resolviendo(sc, nombres, termino):
    """La mitad que YA funciona. Sin esto, un arreglo de la otra mitad podría romperla sin que
    nadie lo note — y perder un alimento que hoy sí se compra es peor que no ganar uno nuevo."""
    assert _resuelve(sc, nombres, termino), (
        f"{termino!r} dejó de resolver a una fila del catálogo → se cae de la lista de la compra"
    )


@pytest.mark.parametrize("termino,motivo", sorted(_CAEN.items()))
def test_los_terminos_que_caen_siguen_cayendo_por_su_motivo(sc, nombres, termino, motivo):
    """Caracterización, no umbral. Un umbral («que no caigan más de N») dejaría pasar justo el caso
    peligroso: que `chorizo` EMPIECE a resolver, a una de las seis filas, arbitrariamente.

    Si este test falla, no lo relajes: mira si resolvió a la fila CORRECTA. Para los ambiguos, la
    palanca está aguas arriba (que el prompt no emita el término desnudo), no aquí."""
    assert not _resuelve(sc, nombres, termino), (
        f"{termino!r} ahora resuelve a {sc.normalize_name(termino)!r}. Motivo por el que NO debía: "
        f"{motivo}. Comprueba que la fila elegida es la correcta y no una arbitraria antes de "
        f"actualizar esta tabla."
    )


def test_lo_que_cae_queda_registrado(sc, monkeypatch):
    """El otro punto que la auditoría daba por perdido: «caen en silencio». No es cierto —
    `record_verified_only_drop` los cuenta y el cron los emite. Medido: 12 de 12. Se ancla porque
    esa telemetría es la ÚNICA señal de «la lista salió incompleta», y perderla dejaría el defecto
    de verdad invisible."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e")
    db_core.connection_pool.open()
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    muestra = ["tahini", "merluza", "albaricoque", "frutilla"]
    sc.snapshot_and_reset_verified_only_drops()
    sc.aggregate_and_deduct_shopping_list([f"100 g de {t}" for t in muestra], structured=True)
    registrados = sc.snapshot_and_reset_verified_only_drops() or {}
    sin_rastro = [t for t in muestra if t not in registrados]
    assert not sin_rastro, (
        f"cayeron del carrito SIN registrarse en el contador de drops: {sin_rastro}. Esa telemetría "
        f"es la única señal de que la lista salió incompleta"
    )


def test_la_lista_de_terminos_no_sale_del_catalogo():
    """La lección de método de la auditoría, anclada. El harness anterior decía «0 drops» porque
    comparaba su lista curada CONTRA el catálogo: un 0/0 que sólo prueba que ambos coinciden.

    Estas dos listas se escriben desde el idioma —lo que un plan de ES/MX/CO puede emitir—, no
    desde `master_ingredients`. Este test lo hace explícito para que nadie las «arregle» generándolas
    del catálogo y devuelva la medición a su círculo."""
    import inspect

    import tests.test_p2_catalog_bare_aliases as mod
    src = inspect.getsource(mod)
    # Sólo el bloque de las DOS listas: los fixtures de arriba sí leen el catálogo, y deben —
    # es contra él contra lo que se mide. Lo que no puede salir de ahí son los TÉRMINOS.
    ini = src.index("_RESUELVEN = [")
    fin = src.index("def _resuelve")
    bloque = src[ini:fin]
    for prohibido in ("get_master_ingredients", "aliases", "for r in ", "normalize_name"):
        assert prohibido not in bloque, (
            f"las listas de términos se están generando del catálogo ({prohibido!r}): eso devuelve "
            f"la medición a la circularidad que este fichero existe para romper"
        )
    assert len(_RESUELVEN) >= 15 and len(_CAEN) >= 15, (
        "las listas encogieron: una muestra pequeña deja de ser una medición"
    )
