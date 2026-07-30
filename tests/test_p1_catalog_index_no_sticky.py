"""[P1-CATALOG-INDEX-NO-STICKY · 2026-07-29] Los índices del catálogo cacheaban su propio FALLO
para siempre: un blip transitorio mataba el phantom-repair durante toda la vida del worker.

`_phantom_catalog_index()` es fail-open a `{}` — razonable. Lo que no era razonable es que ese `{}`
se guardara en `_PHANTOM_CATALOG_INDEX_CACHE`, cuyo guard de entrada es
`if _PHANTOM_CATALOG_INDEX_CACHE is not None: return`. `{}` no es None ⇒ **el fallo quedaba pegado
al proceso**, incluso cuando el catálogo real volvía un segundo después. Bastaba UN fallo en la
PRIMERA construcción (blip de Neon, pool aún sin abrir, TTL expirando en mal momento) para dejar
inactivos el phantom-repair, el dedupe de líneas duplicadas, el reconciliador display↔raw y el pareo
raw-by-food — o sea, justo los pases que evitan que la lista de compras compre el alimento
equivocado. Señal total del apagado: un `logger.warning`. Es la clase ya anotada en memoria: *un
motor que no recibe datos es indistinguible de uno apagado, y no lo dice ningún log.*

Cómo se destapó: 8 tests de 6 ficheros que pasaban en aislamiento y fallaban en la corrida completa.
Un test ~800 ficheros antes parchea `db_core.connection_pool` con un MagicMock;
`get_master_ingredients()` lo cacheaba como catálogo (`res or []`, sin validar tipo — un MagicMock es
truthy) y el índice phantom lo iteraba, lanzaba, y cacheaba su fallo. El mock fue el mensajero; el
agujero era de producción por los dos lados.

Medido: mismo comando, 8 failed antes / 232 passed después. Y la corrida envenenada tardaba 9m21s
frente a 2m05s — el estado degradado además cuesta 4,5× en tiempo.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import graph_orchestrator as go
import shopping_calculator as sc

_BACKEND = Path(__file__).resolve().parents[1]
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_SC = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
_CONFTEST = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _limpio():
    go._PHANTOM_CATALOG_INDEX_CACHE = None
    go._CATALOG_DENSITY_INDEX_CACHE = None
    go._CATALOG_INDEX_NEG_AT.clear()
    sc._master_cache = None
    yield
    go._PHANTOM_CATALOG_INDEX_CACHE = None
    go._CATALOG_DENSITY_INDEX_CACHE = None
    go._CATALOG_INDEX_NEG_AT.clear()
    sc._master_cache = None


# ------------------------------------------------------- el fallo no se pega para siempre

def test_failed_build_does_not_stick_forever(monkeypatch):
    """Lo esencial: tras un fallo, pasado el TTL corto, la siguiente llamada REINTENTA."""
    def _boom():
        raise RuntimeError("catálogo caído (blip)")
    monkeypatch.setattr(sc, "get_master_ingredients", _boom)

    assert go._phantom_catalog_index() == {}, "fail-open: devuelve vacío, no explota"
    monkeypatch.setattr(go, "_CATALOG_INDEX_NEG_TTL_S", 0.0)   # como si el TTL ya pasara

    _llamadas = {"n": 0}

    def _ok():
        _llamadas["n"] += 1
        return [{"name": "Pavo molido"}]
    monkeypatch.setattr(sc, "get_master_ingredients", _ok)

    idx = go._phantom_catalog_index()
    assert _llamadas["n"] >= 1, (
        "tras el fallo NO reintentó contra el catálogo — el `{}` se quedó pegado y el "
        "phantom-repair muere durante toda la vida del worker")
    assert idx, "con el catálogo de vuelta el índice debe reconstruirse"


def test_within_ttl_it_does_not_hammer_the_catalog(monkeypatch):
    """El otro extremo tampoco vale: sin negative-cache se reconstruiría el índice en CADA línea
    (un scan del catálogo por línea) mientras el catálogo esté caído."""
    _n = {"v": 0}

    def _boom():
        _n["v"] += 1
        raise RuntimeError("caído")
    monkeypatch.setattr(sc, "get_master_ingredients", _boom)
    monkeypatch.setattr(go, "_CATALOG_INDEX_NEG_TTL_S", 300.0)

    go._phantom_catalog_index()
    for _ in range(5):
        assert go._phantom_catalog_index() == {}
    assert _n["v"] == 1, f"dentro del TTL debe servir el vacío cacheado, consultó {_n['v']} veces"


def test_success_is_cached_normally(monkeypatch):
    """Un build BUENO sí se cachea (la optimización original sigue en pie)."""
    _n = {"v": 0}

    def _ok():
        _n["v"] += 1
        return [{"name": "Arroz blanco"}]
    monkeypatch.setattr(sc, "get_master_ingredients", _ok)
    a = go._phantom_catalog_index()
    b = go._phantom_catalog_index()
    assert a is b and _n["v"] == 1, "el éxito debe cachearse una sola vez"


def test_density_index_gets_the_same_treatment(monkeypatch):
    def _boom():
        raise RuntimeError("caído")
    monkeypatch.setattr(sc, "get_master_ingredients", _boom)
    monkeypatch.setattr(go, "_CATALOG_INDEX_NEG_TTL_S", 0.0)
    go._catalog_density_index()
    assert go._catalog_index_should_rebuild(
        "_catalog_density_index", go._CATALOG_DENSITY_INDEX_CACHE), (
        "el índice de densidades debe quedar RE-INTENTABLE tras el TTL, no pegado en {} para "
        "siempre")


# ------------------------------------------------------- el catálogo valida el TIPO

def test_master_cache_rejects_non_list(monkeypatch):
    """`res or []` aceptaba cualquier objeto truthy y le sellaba 5 min de TTL. Un MagicMock, un
    cursor a medio consumir o cualquier retorno raro del driver entraban como catálogo verificado."""
    class _Mock:
        def __bool__(self):
            return True
        def __iter__(self):
            raise TypeError("no soy iterable de filas")

    monkeypatch.setattr(sc, "execute_sql_query", lambda *a, **k: _Mock())
    monkeypatch.setattr(sc, "connection_pool", object())
    sc._master_cache = None
    out = sc.get_master_ingredients()
    assert isinstance(out, list), f"debe degradar a lista, devolvió {type(out).__name__}"
    assert not isinstance(sc._master_cache, _Mock), (
        "un objeto no-lista quedó cacheado como catálogo — vuelven 5 min sirviendo basura")


def test_master_cache_accepts_a_real_list(monkeypatch):
    monkeypatch.setattr(sc, "execute_sql_query", lambda *a, **k: [{"name": "Pavo"}])
    monkeypatch.setattr(sc, "connection_pool", object())
    sc._master_cache = None
    assert sc.get_master_ingredients() == [{"name": "Pavo"}]


# ------------------------------------------------------- anclajes estructurales

def test_no_unconditional_cache_assignment_left():
    """La regresión concreta: asignar el índice al caché SIN comprobar que salió no-vacío."""
    for fn, cache in (("def _phantom_catalog_index", "_PHANTOM_CATALOG_INDEX_CACHE"),
                      ("def _catalog_density_index", "_CATALOG_DENSITY_INDEX_CACHE")):
        i = _GO.index(fn)
        body = _GO[i:_GO.index("\ndef ", i + 10)]
        # El contrato NO es "en qué indentación se asigna" (mi primera versión de este test
        # asertaba eso y se volvió falsa en cuanto rediseñé el helper — otra ancla frágil, y encima
        # con match por subcadena: `"    X = idx"` está dentro de `"        X = idx"`).
        # Lo que importa es que el guard de ENTRADA sepa distinguir "índice vacío por fallo" de
        # "índice bueno", y que el fallo quede sellado con su instante para medir el TTL.
        assert f"    if {cache} is not None:" not in body, (
            f"{cache} vuelve al guard `is not None`, que trata `{{}}` como un índice válido ⇒ el "
            f"fallo transitorio se pega a la vida entera del worker")
        assert "_catalog_index_should_rebuild(" in body, (
            f"el guard de entrada de {cache} debe pasar por el helper que distingue vacío de bueno")
        assert "_catalog_index_note_failure(" in body, (
            f"el fallo de {cache} debe sellar su instante o el TTL no tiene desde dónde medir")


def test_master_cache_validates_type_in_source():
    i = _SC.index("def get_master_ingredients")
    body = _SC[i:_SC.index("\ndef ", i + 10)]
    assert "isinstance(res, list)" in body, "vuelve el `res or []` que acepta cualquier truthy"
    assert "_master_cache = res or []" not in body


def test_ttl_knob_is_registered():
    from knobs import get_knobs_registry_snapshot
    assert "MEALFIT_CATALOG_INDEX_NEG_TTL_S" in get_knobs_registry_snapshot()


def test_conftest_resets_the_poisonable_caches():
    """La otra mitad: ~25 ficheros parchean `db_core.connection_pool`, así que el reset vive en un
    punto ÚNICO del conftest en vez de en cada víctima."""
    assert "_limpiar_caches_de_catalogo" in _CONFTEST
    for attr in ("_master_cache", "_PHANTOM_CATALOG_INDEX_CACHE",
                 "_CATALOG_DENSITY_INDEX_CACHE", "_LINE_FOOD_GRAMS_CACHE"):
        assert attr in _CONFTEST, f"el reset del conftest no cubre {attr}"
