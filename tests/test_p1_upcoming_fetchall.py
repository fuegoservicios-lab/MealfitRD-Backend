"""[P1-UPCOMING-FETCHALL · 2026-08-05] La query de `upcoming_chunks` declara
`fetch_all=True`.

POR QUÉ EXISTE ESTE TEST. `db_core.execute_sql_query` emite un WARNING cuando el
caller no declara `fetch_one`/`fetch_all` y la query devuelve varias filas
("Retornando filas como default seguro; el caller debería marcar fetch_all=True
explícito"). El endpoint `/chunk-status` lo pollea el Dashboard cada pocos
segundos por usuario activo, así que el aviso se emitía en bucle.

Medido en el VPS el 2026-08-05, ventana de 45 minutos con UN usuario generando
un plan: 50 de los 70 WARNING del backend salían de esta única llamada — el 71%
del ruido del log. Eso no es cosmético: un log dominado por un aviso benigno es
un log en el que las señales reales no se ven.

El default seguro seguiría funcionando si alguien quita el flag, así que nada
se rompería de forma visible — por eso hace falta un test y no basta con el
comentario.

tooltip-anchor: P1-UPCOMING-FETCHALL
"""
import io
import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_PLANS = _BACKEND_ROOT / "routers" / "plans.py"
_CRON = _BACKEND_ROOT / "cron_tasks.py"


def _plans_source() -> str:
    return io.open(_PLANS, encoding="utf-8").read()


def test_upcoming_chunks_query_exists():
    """Ancla del sitio: si alguien renombra la query, este test avisa antes de
    que el de abajo empiece a verificar el vacío."""
    src = _plans_source()
    assert "SELECT id::text AS chunk_id, week_number, days_offset, days_count" in src, (
        "No se encontró la query de `upcoming_chunks` en routers/plans.py. "
        "¿Se renombró o se movió? Este test dejaría de vigilar en silencio."
    )


def test_upcoming_chunks_declares_fetch_all():
    """La llamada que construye `upcoming_chunks` pasa `fetch_all=True`.

    ⚠️ Las líneas de COMENTARIO se descartan antes de buscar. La primera versión
    de este test no lo hacía y pasaba con el flag borrado: el comentario que
    explica el arreglo cita literalmente `fetch_all=True`, así que el regex se
    encontraba a sí mismo. Un test que certifica TEXTO en vez de la DECISIÓN no
    prueba nada — verificado por mutación en ambos sentidos.
    """
    src = _plans_source()
    start = src.index("SELECT id::text AS chunk_id, week_number, days_offset, days_count")
    # La llamada se cierra en el `) or []` que sigue a los parámetros.
    end = src.index(") or []", start)
    bloque = "\n".join(
        linea for linea in src[start:end].splitlines()
        if not linea.lstrip().startswith("#")
    )
    assert re.search(r"fetch_all\s*=\s*True", bloque), (
        "La query de `upcoming_chunks` no declara `fetch_all=True`. Sin él, "
        "db_core emite un WARNING por CADA llamada y el Dashboard pollea este "
        "endpoint cada pocos segundos: medido, el 71% del ruido del log."
    )


def _sin_comentarios(bloque: str) -> str:
    return "\n".join(
        linea for linea in bloque.splitlines()
        if not linea.lstrip().startswith("#")
    )


def test_cron_chunk_overdue_declara_fetch_all():
    """El cron horario tiene la query GEMELA y el mismo defecto.

    La primera version de este fichero solo vigilaba `/chunk-status`. Revisando
    los logs de produccion aparecio el segundo sitio: `corr=cron:chunk_overdue_alert`
    emitiendo el mismo aviso de `db_core`. Un guard que cubre un sitio de dos da
    una falsa sensacion de cierre — la misma clase de agujero que ya obligo a
    rehacer el ancla de la renovacion como blanket sobre dos ficheros.
    """
    src = io.open(_CRON, encoding="utf-8").read()
    start = src.index("SELECT DISTINCT ON (user_id)")
    end = src.index(") or []", start)
    bloque = _sin_comentarios(src[start:end])
    assert re.search(r"fetch_all\s*=\s*True", bloque), (
        "La query del cron `_chunk_overdue_alert_job` no declara `fetch_all=True`. "
        "Sin el, db_core avisa en cada corrida horaria."
    )
