"""[P1-LIVE-1 . 2026-05-11] La query del cron
`_alert_chunk_pantry_snapshots_stale` debe filtrar solo chunks con
`_pantry_captured_at` realmente capturado.

Bug observado en produccion (mpoodlmnzaeuuazsazbj, 2026-05-11 03:01 UTC):
    Una alerta `chunk_pantry_snapshots_stale` (warning) quedo abierta
    por horas con `stale_chunks=11` aunque la condicion accionable
    habia desaparecido. La query original usaba:

        EXTRACT(EPOCH FROM (
            NOW() - COALESCE(
                (q.pipeline_snapshot->'form_data'->>'_pantry_captured_at')::timestamptz,
                q.created_at
            )
        ))

    Cuando `_pantry_captured_at IS NULL` (snapshot nunca capturado
    porque el worker todavia no levanto el chunk), el COALESCE caia
    a `q.created_at`, que es trivialmente viejo para chunks creados
    hace >20h. Esos chunks NO tienen "snapshot stale" — tienen un
    problema distinto (worker atras / scheduler saturado), cubierto
    por `_alert_chunks_paused_indefinitely` y el force-flexible a 24h
    del worker. Mezclar ambos en una sola alerta:
      1. Producia falsos positivos permanentes (counter nunca bajaba
         porque el snapshot que nunca existio jamas se "refrescaba").
      2. Bloqueaba el auto-resolve P0-AUDIT-2 (la alerta no se cerraba
         aunque la condicion accionable ya estuviera resuelta).
      3. Entrenaba al operador a ignorar el dashboard.

Fix:
    Quitar el COALESCE. Filtrar `_pantry_captured_at IS NOT NULL`
    en la WHERE y comparar directamente el timestamp del snapshot
    contra `NOW() - threshold`. Chunks sin snapshot capturado son
    out-of-scope para esta alerta especifica.

Estrategia del test (parser estatico sobre cron_tasks.py):
    1. La query SELECT dentro de `_alert_chunk_pantry_snapshots_stale`
       contiene la condicion `_pantry_captured_at` IS NOT NULL en la
       WHERE clause.
    2. La query NO usa COALESCE para fallback a `q.created_at` en la
       computacion de `snapshot_age_seconds` ni en el filtro de edad.
    3. El docstring contiene el marker `P1-LIVE-1` para trazabilidad
       post-mortem.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def _extract_function_body(src: str, name: str) -> str:
    """Cuerpo de una funcion top-level hasta el siguiente `def`."""
    m = re.search(
        rf"^def\s+{re.escape(name)}\s*\([^)]*\)[^:]*:(.*?)(?=^def\s)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, f"No se encontro funcion top-level `{name}`."
    return m.group(1)


def test_function_exists(cron_src: str):
    """Sanity: la funcion productora del alert debe existir."""
    assert re.search(
        r"^def\s+_alert_chunk_pantry_snapshots_stale\s*\(",
        cron_src,
        re.MULTILINE,
    ), "P1-LIVE-1: la funcion productora desaparecio."


def test_query_filters_captured_at_not_null(cron_src: str):
    """La WHERE clause de la query SELECT debe filtrar
    `_pantry_captured_at` IS NOT NULL para no contar chunks
    sin snapshot capturado.
    """
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")

    # Buscamos el predicate especifico que filtra captured_at NOT NULL.
    # Aceptamos variantes con o sin parentesis externos, pero exigimos
    # que el JSONB path al campo este presente y termine en `IS NOT NULL`.
    pattern = re.compile(
        r"pipeline_snapshot\s*->\s*'form_data'\s*->>\s*'_pantry_captured_at'\s*\)\s*IS\s+NOT\s+NULL",
        re.IGNORECASE,
    )
    assert pattern.search(body), (
        "P1-LIVE-1 regresion: la query no filtra "
        "`(pipeline_snapshot->'form_data'->>'_pantry_captured_at') IS NOT NULL`. "
        "Sin este predicate, chunks con `captured_at=NULL` se cuentan como "
        "stale via COALESCE a `created_at`, produciendo falsos positivos "
        "permanentes y bloqueando el auto-resolve P0-AUDIT-2."
    )


def test_query_does_not_coalesce_to_created_at(cron_src: str):
    """La query NO debe usar `COALESCE(...captured_at..., q.created_at)`
    para fallback. Chunks sin snapshot son out-of-scope.
    """
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")

    # Extraer el bloque SELECT hasta su ORDER BY (delimitador inequivoco
    # del fin de la query SQL).
    select_block_match = re.search(
        r"SELECT[\s\S]+?ORDER\s+BY\s+snapshot_age_seconds\s+DESC",
        body,
        re.IGNORECASE,
    )
    assert select_block_match, (
        "P1-LIVE-1 setup: no se encontro el bloque SELECT...ORDER BY de la "
        "query del alert. Confirmar que la query sigue inline en el cuerpo."
    )
    select_block = select_block_match.group(0)

    # No debe existir un COALESCE que tenga `q.created_at` adentro,
    # porque eso es exactamente el patron que producia el falso positivo.
    coalesce_with_created = re.compile(
        r"COALESCE\s*\([\s\S]*?q\.created_at[\s\S]*?\)",
        re.IGNORECASE,
    )
    assert not coalesce_with_created.search(select_block), (
        "P1-LIVE-1 regresion: la query usa `COALESCE(..., q.created_at)` "
        "como fallback de `_pantry_captured_at`. Eso resucita el bug "
        "original: chunks sin snapshot capturado se cuentan como stale "
        "via el created_at del chunk."
    )


def test_docstring_documents_p1_live_1(cron_src: str):
    """El docstring debe contener el marker `P1-LIVE-1` para trazabilidad
    post-mortem del cambio."""
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")
    assert "P1-LIVE-1" in body, (
        "P1-LIVE-1 regresion: marker `P1-LIVE-1` ausente del docstring. "
        "Un futuro auditor no puede confirmar que el filtro "
        "`captured_at IS NOT NULL` corresponde a este P-fix."
    )
