"""[P2-PARTIAL-PLAN-1 · 2026-05-11] `_save_plan_and_track_background` siempre
usa `save_new_meal_plan_atomic`; el path con CANCEL out-of-tx fue eliminado.

Bug original (audit 2026-05-11):
    `_save_plan_and_track_background(... additional_db_queries: list = None)`
    tenía dos branches:
      - Si `additional_db_queries` truthy: `execute_sql_write("UPDATE
        plan_chunk_queue SET status='cancelled' ...")` FUERA de la
        transacción del INSERT, seguido de `save_new_meal_plan_robust`.
        Comentario inline admitía: "Riesgo residual es mínimo dado que
        el P0-1/TOCTOU guard en el worker actúa como red de seguridad."
      - Si None: `save_new_meal_plan_atomic` (atómico).

    Ningún caller en producción pasaba `additional_db_queries` (verificado
    vía grep cross-codebase: el único caller en `routers/plans.py:1391`
    solo pasa `actual_user_id, result, selected_techniques`). El branch
    no-atómico era código muerto + vector TOCTOU latente.

Cierre:
    1. Removido el parámetro `additional_db_queries` de la signature.
    2. Eliminado el branch que hacía CANCEL out-of-tx.
    3. Single path: `save_new_meal_plan_atomic(user_id, insert_data)`.
    4. Removidos imports muertos de `save_new_meal_plan_robust` (top-level
       en services.py + lazy en `save_partial_plan_get_id`).

Tests parser-based: scan de services.py para asegurar que el path muerto
no reaparezca + que el único helper usado para INSERT desde
`_save_plan_and_track_background` sea `save_new_meal_plan_atomic`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICES_FP = _REPO_ROOT / "backend" / "services.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _SERVICES_FP.read_text(encoding="utf-8")


def _extract_function_body(src: str, signature_prefix: str) -> str:
    """Extrae body desde `def <name>(...)` hasta el siguiente `def ` top-level."""
    start = src.find(signature_prefix)
    assert start > 0, f"`{signature_prefix}` no encontrado en services.py"
    rest = src[start + len(signature_prefix):]
    nx = re.search(r"\ndef\s+\w+\(", rest)
    end_offset = nx.start() if nx else len(rest)
    return src[start: start + len(signature_prefix) + end_offset]


def test_signature_has_no_additional_db_queries(src: str):
    """`_save_plan_and_track_background` ya no acepta `additional_db_queries`."""
    sig_line = re.search(
        r"def _save_plan_and_track_background\([^)]*\)",
        src,
        re.DOTALL,
    )
    assert sig_line, (
        "P2-PARTIAL-PLAN-1 regresión: signature de "
        "`_save_plan_and_track_background` no parsea. ¿se eliminó la función?"
    )
    assert "additional_db_queries" not in sig_line.group(0), (
        "P2-PARTIAL-PLAN-1 regresión: el parámetro `additional_db_queries` "
        "reaparecio en la signature. Esto reabre el branch out-of-tx que el "
        "fix eliminó. Si genuinamente necesitas queries adicionales atómicas "
        "con el INSERT, extender `save_new_meal_plan_atomic` (db_plans.py) "
        "con un parámetro `additional_queries` que ejecute DENTRO del mismo "
        "`conn.transaction()`, NO restaurar el patrón pre-fix."
    )


def test_body_uses_atomic_only(src: str):
    """El body invoca `save_new_meal_plan_atomic` y NO `save_new_meal_plan_robust`."""
    body = _extract_function_body(src, "def _save_plan_and_track_background(")
    assert "save_new_meal_plan_atomic(user_id, insert_data" in body, (
        "P2-PARTIAL-PLAN-1 regresión: el body no llama a "
        "`save_new_meal_plan_atomic(user_id, insert_data, ...)`. Sin esto "
        "los chunks no se cancelan atómicamente con el INSERT y vuelve "
        "el TOCTOU."
    )
    assert "save_new_meal_plan_robust(" not in body, (
        "P2-PARTIAL-PLAN-1 regresión: el body llama a "
        "`save_new_meal_plan_robust(...)` directamente. Ese helper hace "
        "INSERT con additional_queries pero NO cancela chunks atómicamente "
        "(execute_sql_transaction sin SELECT/UPDATE chunk_queue). Usar "
        "`save_new_meal_plan_atomic` que sí lo hace."
    )


def test_no_manual_cancel_out_of_tx(src: str):
    """No queda ningún `UPDATE plan_chunk_queue SET status = 'cancelled'`
    inline en el body — ese era el patrón out-of-tx pre-fix."""
    body = _extract_function_body(src, "def _save_plan_and_track_background(")
    cancel_inline = re.search(
        r"UPDATE\s+plan_chunk_queue\s+SET\s+status\s*=\s*'cancelled'",
        body,
        re.IGNORECASE,
    )
    assert cancel_inline is None, (
        "P2-PARTIAL-PLAN-1 regresión CRÍTICA: encontrado un "
        "`UPDATE plan_chunk_queue SET status = 'cancelled'` inline en "
        "`_save_plan_and_track_background`. Eso ES el bug que cerramos: "
        "CANCEL out-of-tx separado del INSERT abre ventana TOCTOU. "
        "Mover la lógica DENTRO de `save_new_meal_plan_atomic` (donde "
        "ya vive con `release_chunk_reservations` + UPDATE atómico)."
    )


def test_top_level_import_of_robust_removed(src: str):
    """El import top-level de `save_new_meal_plan_robust` desde db (líneas
    10-24 originalmente) fue removido — sólo `save_new_meal_plan_atomic`
    queda en imports top-level."""
    # Buscar el bloque de imports `from db import (...)` y verificar que
    # `save_new_meal_plan_robust` NO esté listado adentro.
    m = re.search(r"from\s+db\s+import\s+\(([^)]*)\)", src, re.DOTALL)
    assert m, "bloque `from db import (...)` no encontrado en services.py"
    imports_block = m.group(1)
    assert "save_new_meal_plan_robust" not in imports_block, (
        "P2-PARTIAL-PLAN-1 regresión: `save_new_meal_plan_robust` reaparecio "
        "en el bloque `from db import (...)`. Era código muerto tras el fix "
        "(no hay caller en services.py). Si lo necesitas para una función "
        "nueva, importa lazy dentro del scope que lo usa, no top-level."
    )


def test_save_partial_plan_get_id_no_dead_lazy_import(src: str):
    """El lazy import `from db_plans import save_new_meal_plan_robust` dentro
    de `save_partial_plan_get_id` también era dead code — el body real
    usa `save_new_meal_plan_atomic`."""
    body = _extract_function_body(src, "def save_partial_plan_get_id(")
    assert "from db_plans import save_new_meal_plan_robust" not in body, (
        "P2-PARTIAL-PLAN-1 regresión: el lazy import de "
        "`save_new_meal_plan_robust` dentro de `save_partial_plan_get_id` "
        "reaparecio. Era dead code (el body real llama a "
        "`save_new_meal_plan_atomic`). Confunde a revisores futuros."
    )
    # Y el body sigue usando atomic
    assert "save_new_meal_plan_atomic(user_id, insert_data" in body, (
        "P2-PARTIAL-PLAN-1: `save_partial_plan_get_id` no llama a "
        "`save_new_meal_plan_atomic`. Si refactoreaste el body, asegurar "
        "que la cancelación atómica de chunks se preserve."
    )
