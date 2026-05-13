"""[P1-EMIT-LOCK-SWEEP-NORAISE · 2026-05-13] Regresión contra escape de
excepción desde `_sweep_stale_emit_locks_kv` que dispare `EVENT_JOB_ERROR`.

Bug original (audit 2026-05-13):
    El alert `scheduler_error_sweep_stale_emit_locks_kv` aparecía en
    `system_alerts` cada vez que el cron ejecutaba (cada 6h), aunque
    pipeline_metrics confirmaba que el sweep completaba EXITOSAMENTE
    (`sweep_failed=False`, tick row presente).

    Causa raíz: línea huérfana `return reset_count` quedó al final del
    cuerpo de `_sweep_stale_emit_locks_kv` (copy-paste residual de la
    función hermana `_sweep_stale_llm_circuit_breakers` que sí define
    `reset_count`). El sweep completaba DELETE + tick, luego intentaba
    `return reset_count` → `NameError: name 'reset_count' is not defined`
    → APScheduler captura → listener dispara `scheduler_error_*`.

    Side effect: alert fatigue + ruido en SRE; cada 6h se inyectaba
    una alert "critical" sobre un cron sin trabajo real (0 keys).

Estrategia del test:
    1. Parser-based: confirmar que el cuerpo de la función NO contiene
       `return reset_count` ni referencias huérfanas a `reset_count`
       (variable definida en función hermana `_sweep_stale_llm_circuit_breakers`).
    2. Verificar que la firma sigue siendo `-> None` (no retorna nada).
    3. La función NO debe tener ningún `return <name>` que referencie un
       símbolo no definido localmente.

Tooltip-anchor: P1-EMIT-LOCK-SWEEP-NORAISE-START
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRON_FP = _REPO_ROOT / "backend" / "cron_tasks.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _CRON_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(src: str) -> ast.Module:
    return ast.parse(src)


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(
        f"[P1-EMIT-LOCK-SWEEP-NORAISE] No se encontró la función `{name}` "
        "en cron_tasks.py."
    )


def test_no_orphan_reset_count_return(tree: ast.Module):
    """El cuerpo de `_sweep_stale_emit_locks_kv` NO debe referenciar
    `reset_count` (variable de la función hermana)."""
    func = _find_function(tree, "_sweep_stale_emit_locks_kv")

    locally_assigned: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    locally_assigned.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            locally_assigned.add(node.target.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            locally_assigned.add(node.target.id)

    references = {
        n.id
        for n in ast.walk(func)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    orphan = references - locally_assigned - set(dir(__builtins__)) - {
        "execute_sql_write", "logger", "json", "_env_int", "Exception",
    }
    assert "reset_count" not in orphan, (
        "P1-EMIT-LOCK-SWEEP-NORAISE regresión: la función "
        "`_sweep_stale_emit_locks_kv` vuelve a referenciar `reset_count` "
        "(variable de `_sweep_stale_llm_circuit_breakers`). Bug clase "
        "copy-paste residual: dispara `NameError` post-tick, escapa al "
        "scheduler, listener inserta `scheduler_error_*` cada 6h. "
        "Eliminar la línea huérfana `return reset_count` o similar."
    )


def test_function_returns_none(tree: ast.Module):
    """`_sweep_stale_emit_locks_kv` declara `-> None`; ningún return debe
    retornar un nombre que escape al type checker."""
    func = _find_function(tree, "_sweep_stale_emit_locks_kv")
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            assert False, (
                "P1-EMIT-LOCK-SWEEP-NORAISE regresión: "
                "`_sweep_stale_emit_locks_kv` debe declarar `-> None` y "
                f"sus returns deben ser vacíos. Encontrado: return con valor "
                f"`{ast.dump(node.value)}`. Si necesitas devolver algo, "
                "actualiza la signature y este test."
            )


def test_signature_unchanged(src: str):
    """Anchor: la firma sigue siendo `-> None`. Si cambia, este test
    debe actualizarse junto."""
    assert re.search(
        r"def\s+_sweep_stale_emit_locks_kv\s*\(\s*\)\s*->\s*None\s*:",
        src,
    ), (
        "P1-EMIT-LOCK-SWEEP-NORAISE regresión: la firma de "
        "`_sweep_stale_emit_locks_kv` ya no es `() -> None`. Si era "
        "intencional, ajusta `test_function_returns_none` y la doc del "
        "P-fix. Si no, alguien introdujo un parámetro o tipo de retorno "
        "no esperado."
    )
