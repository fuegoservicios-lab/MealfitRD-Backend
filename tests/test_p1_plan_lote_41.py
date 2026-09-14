# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-41 · 2026-09-14] La tormenta de reintentos del catálogo con la base caída.

Medido (2026-09-13 en la CI sin base, reproducido en local el 14-sep con un perfil del landing, 14 días):
`horizon.build_blueprint` hace **32.676** llamadas a `shopping_calculator.get_master_ingredients` (32.564 con
el `_EFF` de `test_p1_arq27_f3_candidateset`) y, sin pool, CADA una entraba en la rama `else`, registraba
`logging.error("No connection_pool available…")` y devolvía `[]` sin sellar nada: **32.676 líneas de error**
en un blueprint. Con un pool que falla (Neon caído en un arranque en frío) eran **32.676 intentos de
conexión**, cada uno con el timeout del pool. `catalog_capability` sólo cacheaba snapshots NO vacíos: con
`rows == []` devolvía `None` sin cachear y volvía a preguntar en la siguiente ancla, constituyente y plantilla.

Lo que estos tests fijan:
  · sin pool, 1.000 llamadas ⇒ UN error de log y CERO accesos al pool dentro de la ventana (las demás, DEBUG
    con el conteo); al vencer la ventana, reintenta;
  · si el pool aparece a mitad de ventana, la siguiente llamada lee la tabla (el pool se mira ANTES que la
    ventana; `shopping_calculator` importa `connection_pool` por valor, así que el que «aparece» es el suyo);
  · con un pool que falla, sella igual y no reintenta hasta vencer o cambiar de objeto de pool;
  · el sello negativo NUNCA escribe `_master_cache_ts` (P1-CATALOG-INDEX-NO-STICKY: cinco minutos de vacío
    servidos como catálogo);
  · `invalidate_master_cache` y `catalog_capability.reset_cache` limpian el sello y el «no se sabe» por país;
  · `catalog_capability` cachea el `None` por país con la MISMA ventana y `None` sigue siendo «desconocido»;
  · el knob `MEALFIT_CATALOG_NEGATIVE_CACHE_S` (30 s, clamp [1, 300]) está registrado y documentado;
  · un blueprint entero sin base: ≤ 2 líneas de error y 0 intentos al pool (después: 1 y 0).
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import catalog_capability as CC  # noqa: E402
import db_core  # noqa: E402
import shopping_calculator as sc  # noqa: E402

_MSG_SIN_POOL = "No connection_pool available to fetch master_ingredients"
_MSG_EXCEPCION = "Error fetching master_ingredients via pool"


def _errores_catalogo(caplog) -> list:
    return [r for r in caplog.records if r.levelno == logging.ERROR and "master_ingredients" in r.getMessage()]


def _debugs_ventana(caplog) -> list:
    return [r for r in caplog.records if r.levelno == logging.DEBUG and "ventana negativa" in r.getMessage()]


@pytest.fixture(autouse=True)
def _estado_limpio():
    sc.invalidate_master_cache()
    CC.reset_cache()
    yield
    sc.invalidate_master_cache()
    CC.reset_cache()


@pytest.fixture
def sin_pool(monkeypatch):
    """La CI sin base: ni `db_core` ni `shopping_calculator` (que lo importó por valor) tienen pool."""
    monkeypatch.setattr(db_core, "connection_pool", None)
    monkeypatch.setattr(sc, "connection_pool", None)
    accesos = {"n": 0}

    def _jamas(*a, **k):
        accesos["n"] += 1
        raise AssertionError("sin pool no se debe ejecutar SQL")

    monkeypatch.setattr(sc, "execute_sql_query", _jamas)
    return accesos


# ── la ventana negativa en `get_master_ingredients` ─────────────────────────────────────────────
def test_sin_pool_mil_llamadas_un_solo_error_y_cero_accesos_al_pool(sin_pool, caplog):
    caplog.set_level(logging.DEBUG)
    for _ in range(1000):
        assert sc.get_master_ingredients() == []
    assert sin_pool["n"] == 0
    errores = _errores_catalogo(caplog)
    assert len(errores) == 1, [r.getMessage() for r in errores][:3]
    assert _MSG_SIN_POOL in errores[0].getMessage() and "P1-PLAN-LOTE-41" in errores[0].getMessage()
    assert sc._master_cache_neg_count == 999
    assert len(_debugs_ventana(caplog)) == 999
    assert sc._master_cache_neg_until > time.time()
    assert sc._master_cache_neg_pool_id is None


def test_al_vencer_la_ventana_reintenta_y_vuelve_a_registrar_una_vez(sin_pool, caplog):
    caplog.set_level(logging.DEBUG)
    for _ in range(10):
        sc.get_master_ingredients()
    assert len(_errores_catalogo(caplog)) == 1
    sc._master_cache_neg_until = time.time() - 1  # la ventana vence
    for _ in range(10):
        sc.get_master_ingredients()
    assert len(_errores_catalogo(caplog)) == 2, "al vencer, reintenta y sella una ventana nueva"
    assert sc._master_cache_neg_count == 9 and sc._master_cache_neg_until > time.time()


def test_la_ventana_dura_lo_que_dice_el_knob(sin_pool, caplog, monkeypatch):
    monkeypatch.setenv("MEALFIT_CATALOG_NEGATIVE_CACHE_S", "1")
    caplog.set_level(logging.DEBUG)
    t0 = time.time()
    sc.get_master_ingredients()
    assert 0.5 <= sc._master_cache_neg_until - t0 <= 1.5
    time.sleep(1.1)
    sc.get_master_ingredients()
    assert len(_errores_catalogo(caplog)) == 2


def test_si_el_pool_aparece_a_mitad_de_ventana_lee_la_tabla_en_la_siguiente_llamada(sin_pool, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    for _ in range(5):
        assert sc.get_master_ingredients() == []
    assert sc._master_cache_neg_until > time.time()
    filas = [{"name": "Pollo", "aliases": []}, {"name": "Arroz blanco", "aliases": ["arroz"]}]
    lecturas = {"n": 0}

    def _tabla(query, *a, **k):
        lecturas["n"] += 1
        assert "master_ingredients" in query
        return list(filas)

    monkeypatch.setattr(sc, "connection_pool", object())  # aparece: truthy y distinto de «ninguno»
    monkeypatch.setattr(sc, "execute_sql_query", _tabla)
    assert sc.get_master_ingredients() == filas, "la ventana no aplica al pool que acaba de aparecer"
    assert lecturas["n"] == 1
    assert sc._master_cache_ts > 0 and sc._master_cache_neg_until == 0.0 and sc._master_cache_neg_count == 0
    assert sc.get_master_ingredients() == filas and lecturas["n"] == 1, "y ahora sirve el TTL normal de 5 min"


def test_con_pool_que_falla_sella_y_no_reintenta_hasta_vencer_o_cambiar_de_pool(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    intentos = {"n": 0}

    def _neon_caido(*a, **k):
        intentos["n"] += 1
        raise OSError("Neon caído (simulado)")

    pool_a = object()
    monkeypatch.setattr(sc, "connection_pool", pool_a)
    monkeypatch.setattr(sc, "execute_sql_query", _neon_caido)
    for _ in range(1000):
        assert sc.get_master_ingredients() == []
    assert intentos["n"] == 1, "un intento por ventana, no uno por llamada"
    errores = _errores_catalogo(caplog)
    assert len(errores) == 1 and _MSG_EXCEPCION in errores[0].getMessage()
    assert sc._master_cache_neg_pool_id == id(pool_a) and sc._master_cache_neg_count == 999
    # Otro objeto de pool ⇒ la ventana no es suya: reintenta en la siguiente llamada.
    monkeypatch.setattr(sc, "connection_pool", object())
    sc.get_master_ingredients()
    assert intentos["n"] == 2 and len(_errores_catalogo(caplog)) == 2
    # El mismo pool otra vez, dentro de la ventana nueva ⇒ absorbido.
    for _ in range(50):
        sc.get_master_ingredients()
    assert intentos["n"] == 2


def test_el_sello_negativo_nunca_escribe_master_cache_ts(sin_pool, monkeypatch):
    assert sc._master_cache_ts == 0
    for _ in range(50):
        sc.get_master_ingredients()
    assert sc._master_cache_ts == 0, "rama «sin pool»: el vacío no se sella como catálogo"
    assert sc._master_cache_neg_until > time.time()
    sc.invalidate_master_cache()

    def _caido(*a, **k):
        raise OSError("caído")

    monkeypatch.setattr(sc, "connection_pool", object())
    monkeypatch.setattr(sc, "execute_sql_query", _caido)
    for _ in range(50):
        sc.get_master_ingredients()
    assert sc._master_cache_ts == 0, "rama «excepción»: tampoco"
    assert sc._master_cache_neg_until > time.time()
    # Y con la ventana puesta, una caché vieja se sirve tal cual, sin renovar su sello de verificado.
    sc._master_cache = [{"name": "Yuca"}]
    assert sc.get_master_ingredients() == [{"name": "Yuca"}] and sc._master_cache_ts == 0


def test_el_reset_limpia_el_sello_y_el_no_se_sabe_por_pais(sin_pool, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    sc.get_master_ingredients()
    assert sc._master_cache_neg_until > 0
    CC.catalog_capability("DO")
    assert "DO" in CC._CACHE_NEG
    sc.invalidate_master_cache()
    assert sc._master_cache_neg_until == 0.0 and sc._master_cache_neg_pool_id is None and sc._master_cache_neg_count == 0
    assert CC._CACHE_NEG == {}, "invalidar el catálogo olvida también el «no se sabe» por país"
    sc.get_master_ingredients()
    assert len(_errores_catalogo(caplog)) == 2, "tras el reset se reintenta de inmediato"
    CC.catalog_capability("DO")
    assert "DO" in CC._CACHE_NEG
    CC.reset_cache()
    assert CC._CACHE_NEG == {} and CC._CACHE == {} and CC._AVISADOS == set()


# ── `catalog_capability`: el `None` por país con la misma ventana ────────────────────────────────
def test_catalog_capability_cachea_el_none_por_pais_con_la_misma_ventana(monkeypatch):
    lecturas = {"n": 0}

    def _vacio(*a, **k):
        lecturas["n"] += 1
        return []

    monkeypatch.setattr(sc, "get_master_ingredients", _vacio)
    for _ in range(100):
        assert CC.catalog_capability("DO") is None
    assert lecturas["n"] == 1, "cien anclas, una lectura"
    assert CC.known_ingredient_names("DO") is None and CC.is_available("Pollo", "DO") is None
    assert CC.template_buyable_in(["Pollo", "Yuca"], "DO") is True, "desconocido ⇒ no se recorta nada"
    assert lecturas["n"] == 1
    # Otro país es otra entrada; el mismo país, vencida su ventana, vuelve a leer.
    assert CC.catalog_capability("ES") is None and lecturas["n"] == 2
    CC._CACHE_NEG["DO"] = time.time() - 1
    assert CC.catalog_capability("DO") is None and lecturas["n"] == 3
    # Y si el catálogo vuelve, el `None` cacheado no lo tapa más allá de su ventana.
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: [{"name": "Pollo", "aliases": []}])
    CC._CACHE_NEG["DO"] = time.time() - 1
    s = CC.catalog_capability("DO")
    assert s and s["count"] == 1 and "DO" not in CC._CACHE_NEG


def test_catalog_capability_que_revienta_tambien_sella(monkeypatch):
    llamadas = {"n": 0}

    def _boom(*a, **k):
        llamadas["n"] += 1
        raise RuntimeError("db connection_pool is not available.")

    monkeypatch.setattr(sc, "get_master_ingredients", _boom)
    for _ in range(100):
        assert CC.catalog_capability("DO") is None
    assert llamadas["n"] == 1


# ── knob ───────────────────────────────────────────────────────────────────────────────────────
def test_el_knob_se_registra_con_clamp_y_esta_documentado(monkeypatch):
    from knobs import _KNOBS_REGISTRY
    assert "MEALFIT_CATALOG_NEGATIVE_CACHE_S" in _KNOBS_REGISTRY, "se registra al importar shopping_calculator"
    monkeypatch.delenv("MEALFIT_CATALOG_NEGATIVE_CACHE_S", raising=False)
    assert sc._catalog_negative_cache_s() == 30
    assert _KNOBS_REGISTRY["MEALFIT_CATALOG_NEGATIVE_CACHE_S"]["default"] == 30
    for raw, esperado in (("0", 1), ("-5", 1), ("1", 1), ("300", 300), ("9999", 300), ("abc", 30), ("45", 45)):
        monkeypatch.setenv("MEALFIT_CATALOG_NEGATIVE_CACHE_S", raw)
        assert sc._catalog_negative_cache_s() == esperado, raw
    doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "MEALFIT_CATALOG_NEGATIVE_CACHE_S" in doc and "32.676" in doc


# ── el blueprint entero, sin base ──────────────────────────────────────────────────────────────
def test_un_blueprint_sin_base_ya_no_es_una_tormenta(sin_pool, caplog, monkeypatch):
    import dish_registry as dr
    import horizon as H
    if not dr.registry_hash("DO"):
        pytest.skip("sin snapshot compilado del registry")
    caplog.set_level(logging.DEBUG)
    llamadas = {"n": 0}
    orig = sc.get_master_ingredients

    def _contada(*a, **k):
        llamadas["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(sc, "get_master_ingredients", _contada)
    eff = {
        "diet": {"type": "omnivora", "allergies": []},
        "culture_weights": [{"profile_id": "dominican_criolla", "weight": 1.0}],
        "market_country": "DO",
        "shopping": {"main_cycle_days": 7, "freezer_mode": "limited"},
        "recurrence": {"global_mode": "balanced"},
    }
    bp = H.build_blueprint(eff, total_days=14, meals_per_day=4)
    assert len(bp.get("days") or []) == 14
    assert llamadas["n"] >= 1
    assert sin_pool["n"] == 0, "cero accesos al pool"
    errores = _errores_catalogo(caplog)
    assert len(errores) <= 2, f"antes eran 32.564 por blueprint; ahora {len(errores)}"
    assert llamadas["n"] < 1000, f"antes 32.564 llamadas; con el `None` por país cacheado, {llamadas['n']}"


# ── docs · plan · marker · anclas ───────────────────────────────────────────────────────────────
def test_docs_plan_marker_y_anclas():
    for doc in ("knobs_reference.md", "plan_pendientes_2026_09_11.md"):
        assert "P1-PLAN-LOTE-41" in (_BACKEND / "docs" / doc).read_text(encoding="utf-8"), doc
    for src in ("shopping_calculator.py", "catalog_capability.py", "tests/conftest.py"):
        assert "P1-PLAN-LOTE-41" in (_BACKEND / src).read_text(encoding="utf-8"), src
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 41
    # El pool se comprueba ANTES que la ventana: si alguien invierte el orden, el pool que aparece queda tapado.
    fuente = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    fn = re.search(r"def get_master_ingredients\(\):(.*?)\n    return _master_cache\n", fuente, re.DOTALL)
    assert fn, "get_master_ingredients no se encontró"
    cuerpo = fn.group(1)
    assert cuerpo.index("_pool_id = id(connection_pool) if connection_pool else None") < cuerpo.index("_master_cache_neg_until and")
    assert cuerpo.count("_master_cache_ts = now") == 1, "sólo la lectura buena sella el ts"
