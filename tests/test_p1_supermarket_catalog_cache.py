"""[P1-SUPERMARKET-CATALOG-CACHE · 2026-08-05] `/match` no recarga el catálogo entero
en cada llamada.

MEDIDO EN PRODUCCIÓN (2026-08-05). El dueño reportó que «Marcas del súper» tarda en
cargar en cada refresco. El endpoint hacía DOS `SELECT` completos sobre
`supermarket_products` —sin caché— y el Dashboard lo pide en cada carga de página:

    1.739 filas activas · 251 alimentos distintos · 236-382 ms solo de DB por llamada

El trabajo en Python es despreciable al lado: el parser de presentaciones tarda 2 ms
sobre las 597 filas que lo necesitan. O sea, el tiempo que el usuario ve esperando era
casi todo latencia de dos consultas que devuelven SIEMPRE lo mismo.

El catálogo solo cambia cuando un admin lo edita, así que las tres rutas de mutación
invalidan la caché explícitamente: el TTL es una red de seguridad, no el mecanismo.

tooltip-anchor: P1-SUPERMARKET-CATALOG-CACHE
"""
import io
import re
from pathlib import Path

import pytest

_ROUTER = Path(__file__).resolve().parents[1] / "routers" / "supermarket.py"


def _src() -> str:
    return io.open(_ROUTER, encoding="utf-8").read()


def _sin_comentarios(bloque: str) -> str:
    """Un comentario que cite el símbolo buscado haría pasar el test con el arreglo
    borrado — ya ocurrió con `test_p1_upcoming_fetchall`."""
    return "\n".join(
        l for l in bloque.splitlines() if not l.lstrip().startswith("#")
    )


def test_existe_la_cache_y_su_knob():
    src = _sin_comentarios(_src())
    assert "_CATALOG_CACHE" in src
    assert "MEALFIT_SUPERMARKET_CATALOG_CACHE_TTL_S" in src, (
        "Sin knob no hay rollback sin redeploy si la caché diera problemas."
    )


def test_ttl_cero_desactiva_la_cache():
    """El rollback tiene que ser real, no cosmético."""
    import os
    from routers.supermarket import _catalog_cache_ttl_s
    prev = os.environ.get("MEALFIT_SUPERMARKET_CATALOG_CACHE_TTL_S")
    try:
        os.environ["MEALFIT_SUPERMARKET_CATALOG_CACHE_TTL_S"] = "0"
        assert _catalog_cache_ttl_s() == 0
        os.environ["MEALFIT_SUPERMARKET_CATALOG_CACHE_TTL_S"] = "300"
        assert _catalog_cache_ttl_s() == 300
        # Clamp por arriba y valor inválido → default.
        os.environ["MEALFIT_SUPERMARKET_CATALOG_CACHE_TTL_S"] = "99999"
        assert _catalog_cache_ttl_s() == 3600
        os.environ["MEALFIT_SUPERMARKET_CATALOG_CACHE_TTL_S"] = "no-es-un-numero"
        assert _catalog_cache_ttl_s() == 300
    finally:
        if prev is None:
            os.environ.pop("MEALFIT_SUPERMARKET_CATALOG_CACHE_TTL_S", None)
        else:
            os.environ["MEALFIT_SUPERMARKET_CATALOG_CACHE_TTL_S"] = prev


def test_invalidar_deja_la_cache_fria():
    from routers.supermarket import _CATALOG_CACHE, _invalidate_catalog_cache
    _CATALOG_CACHE["rows"] = [{"x": 1}]
    _CATALOG_CACHE["master"] = [{"y": 2}]
    _CATALOG_CACHE["at"] = 1.0
    _invalidate_catalog_cache()
    assert _CATALOG_CACHE["rows"] is None
    assert _CATALOG_CACHE["master"] is None
    assert _CATALOG_CACHE["at"] == 0.0


@pytest.mark.parametrize("fn", ["_insert", "_update", "_delete"])
def test_las_tres_mutaciones_invalidan(fn):
    """El editor vive en la MISMA página que consume el catálogo: sin invalidar,
    el admin cambia un precio y no lo ve hasta que expira el TTL."""
    src = _src()
    i = src.index(f"    def {fn}():")
    # La invalidación va ANTES de la función interna que escribe.
    ventana = _sin_comentarios(src[max(0, i - 900):i])
    assert "_invalidate_catalog_cache()" in ventana, (
        f"La mutación de `{fn}` no invalida la caché del catálogo."
    )


def test_la_segunda_query_tambien_se_cachea():
    """Cachear solo la grande dejaría la mitad del coste intacto."""
    src = _sin_comentarios(_src())
    i = src.index("SELECT DISTINCT master_food_name")
    ventana = src[max(0, i - 600):i + 900]
    assert "master_rows_cached" in ventana, (
        "La query de `master_food_name` no consulta la caché: seguiría pegándole "
        "a la DB en cada llamada."
    )
    # [P2-BACKEND-SUPERMARKET-CACHE · 2026-08-14] Antes se exigía la asignación
    # LITERAL `_CATALOG_CACHE["master"] = …`. Ese guard estaba atado al mecanismo,
    # y el mecanismo cambió a propósito: las escrituras directas de la caché
    # esquivaban la comprobación de generación que cierra la carrera
    # invalidación-antes-del-`await`, así que ahora TODAS pasan por
    # `_publish_catalog_cache(...)`. El contrato que importa —que la segunda
    # consulta también se cachee— no se toca.
    assert "_publish_catalog_cache(" in ventana, (
        "La query de `master_food_name` no se guarda en la caché."
    )
