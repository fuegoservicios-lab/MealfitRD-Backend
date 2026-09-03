"""[P2-BACKEND-SUPERMARKET-CACHE · 2026-08-14] La invalidación de la caché del
catálogo corría ANTES de la escritura que la motivaba.

EL DEFECTO. Los tres handlers admin del supermercado hacen, en este orden:

    _invalidate_catalog_cache()          # vacía la caché
    ...
    await asyncio.to_thread(_insert)     # ← punto de cesión GARANTIZADO

Ese `await` deja correr a otra tarea. `/match` puede entrar entera en medio,
encontrar la caché vacía, releer las filas **pre-escritura** y repoblarla con un
`at = time.time()` fresco. A partir de ahí, hasta 5 minutos (el TTL por defecto)
de precios obsoletos alimentando el costeo de marca del Dashboard y de la Nevera
— justo después de que un admin los corrigiera, que es cuando más se confía en
que están bien.

⚠️ LO QUE **NO** LO ARREGLA: mover `_invalidate_catalog_cache()` a después de la
escritura. Deja exactamente la misma ventana, sólo que desplazada — el lector
sigue pudiendo colarse entre la escritura y la invalidación. Es la corrección que
parece obvia y no sirve.

LO QUE SÍ: un contador de generación. El rellenador captura `gen` ANTES de leer y
sólo publica si sigue igual. Si cambió, tira lo que trajo. El coste de descartar
es una consulta más en la siguiente petición; el coste de publicar es servir
precios viejos durante todo el TTL.

Tooltip-anchor: P2-BACKEND-SUPERMARKET-CACHE
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ROUTER = _REPO_ROOT / "routers" / "supermarket.py"


@pytest.fixture()
def sm():
    import routers.supermarket as modulo

    modulo._CATALOG_CACHE.update({"at": 0.0, "rows": None, "master": None, "gen": 0})
    return modulo


# ---------------------------------------------------------------------------
# Comportamiento: la carrera, reproducida
# ---------------------------------------------------------------------------

def test_un_relleno_limpio_se_publica(sm):
    gen = sm._catalog_generation()
    assert sm._publish_catalog_cache(["fila"], ["maestro"], gen) is True
    assert sm._CATALOG_CACHE["rows"] == ["fila"]


def test_un_relleno_adelantado_por_una_escritura_se_DESCARTA(sm):
    """El caso exacto del bug: leer, que un admin escriba, y luego publicar."""
    gen = sm._catalog_generation()          # el lector captura la generación…
    sm._invalidate_catalog_cache()          # …un admin escribe mientras él leía…
    publicado = sm._publish_catalog_cache(["filas viejas"], ["maestro viejo"], gen)

    assert publicado is False, (
        "El relleno se publicó pese a que hubo una mutación admin en medio: son "
        "filas PRE-escritura y quedarían vivas hasta que expire el TTL."
    )
    assert sm._CATALOG_CACHE["rows"] is None, (
        "La caché quedó poblada con datos obsoletos. Debe seguir vacía para que "
        "la siguiente petición vuelva a la DB."
    )


def test_tras_descartar_el_siguiente_lector_si_puede_publicar(sm):
    """Descartar no puede dejar la caché envenenada para siempre."""
    gen_viejo = sm._catalog_generation()
    sm._invalidate_catalog_cache()
    sm._publish_catalog_cache(["viejas"], [], gen_viejo)          # descartado

    gen_nuevo = sm._catalog_generation()                          # lector nuevo
    assert sm._publish_catalog_cache(["frescas"], [], gen_nuevo) is True
    assert sm._CATALOG_CACHE["rows"] == ["frescas"]


def test_cada_invalidacion_avanza_la_generacion(sm):
    antes = sm._catalog_generation()
    sm._invalidate_catalog_cache()
    sm._invalidate_catalog_cache()
    assert sm._catalog_generation() == antes + 2, (
        "Dos escrituras seguidas tienen que dejar dos generaciones: si el contador "
        "no avanza por cada una, un lector que capturó entre ambas creería estar al día."
    )


def test_invalidar_sigue_vaciando_la_cache(sm):
    """La razón original de P1-SUPERMARKET-CATALOG-CACHE no se toca."""
    sm._CATALOG_CACHE.update({"rows": ["algo"], "master": ["x"], "at": 1.0})
    sm._invalidate_catalog_cache()
    assert sm._CATALOG_CACHE["rows"] is None and sm._CATALOG_CACHE["at"] == 0.0


# ---------------------------------------------------------------------------
# Estructura: el lector captura la generación ANTES de la consulta
# ---------------------------------------------------------------------------

def test_match_captura_la_generacion_antes_de_leer_la_db():
    """Capturarla después la haría siempre coincidir: el guard sería decorativo."""
    src = _ROUTER.read_text(encoding="utf-8")
    i_captura = src.find("_gen_al_empezar = _catalog_generation()")
    assert i_captura != -1, (
        "[P2-BACKEND-SUPERMARKET-CACHE] `/match` ya no captura la generación."
    )
    i_select = src.find("SELECT id::text AS id, food_name", i_captura - 4000)
    assert i_select > i_captura, (
        "[P2-BACKEND-SUPERMARKET-CACHE] La generación se captura DESPUÉS de la "
        "consulta. Así siempre coincidirá consigo misma y el guard no puede "
        "detectar nada — sería un adorno."
    )
    assert src.find("_publish_catalog_cache(", i_select) > i_select, (
        "[P2-BACKEND-SUPERMARKET-CACHE] El relleno no pasa por `_publish_catalog_cache`."
    )


def test_ninguna_escritura_directa_de_la_cache_esquiva_el_guard():
    """Si alguien vuelve a escribir `_CATALOG_CACHE['rows']` a mano, el contador sobra."""
    src = _ROUTER.read_text(encoding="utf-8")
    # Fuera de los dos helpers, nadie debe asignar `rows`/`at` de la caché.
    cuerpo = src.split("def _publish_catalog_cache", 1)[1]
    cuerpo = cuerpo.split("\n\n\n", 1)[1] if "\n\n\n" in cuerpo else cuerpo
    directas = re.findall(r'_CATALOG_CACHE\["(?:rows|at|master)"\]\s*=', cuerpo)
    assert not directas, (
        "[P2-BACKEND-SUPERMARKET-CACHE] Hay escrituras directas de la caché fuera "
        f"de los helpers ({len(directas)}): esquivan la comprobación de generación "
        "y reabren la ventana de precios obsoletos."
    )
