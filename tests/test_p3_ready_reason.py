"""[P3-READY-REASON · 2026-05-12] Anchor + regression guard.

`/ready` debe incluir un `reason` granular en el body del 503 cuando
`is_plan_graph_ready()` retorna False. Pre-fix devolvía solo
`{status: not_ready, plan_graph: not_compiled}` sin pista del modo de
fallo; el operador debía abrir logs CRITICAL del worker separadamente.

Defensas que el test enforza:
  1. Anchor `P3-READY-REASON` en graph_orchestrator.py + app.py.
  2. `is_plan_graph_ready_with_reason()` existe y retorna tuple `(bool, str|None)`.
  3. `_PLAN_GRAPH_LAST_BUILD_ERROR` global declarada.
  4. `_get_plan_graph()` actualiza el snapshot del error en el `except` path
     y lo resetea al `None` tras un build exitoso.
  5. Endpoint `/ready` en app.py incluye `reason` en el body del 503.

Test parser-based — escanea source con regex.
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRAPH = _REPO_ROOT / "backend" / "graph_orchestrator.py"
_APP = _REPO_ROOT / "backend" / "app.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_anchor_present_in_graph_orchestrator():
    src = _read(_GRAPH)
    assert "P3-READY-REASON" in src, (
        "Falta anchor `P3-READY-REASON` en backend/graph_orchestrator.py."
    )


def test_anchor_present_in_app():
    src = _read(_APP)
    assert "P3-READY-REASON" in src


def test_last_build_error_global_declared():
    src = _read(_GRAPH)
    pat = re.compile(r"_PLAN_GRAPH_LAST_BUILD_ERROR\s*:\s*dict\s*\|\s*None\s*=\s*None")
    assert pat.search(src), (
        "Falta declaración `_PLAN_GRAPH_LAST_BUILD_ERROR: dict | None = None` "
        "global en graph_orchestrator. Sin ella, el snapshot del error no "
        "tiene dónde vivir entre el except del build y el reader del probe."
    )


def test_build_except_path_writes_snapshot():
    """`_get_plan_graph()` debe escribir un dict en `_PLAN_GRAPH_LAST_BUILD_ERROR`
    dentro del `except Exception as e:` (con keys type/message/timestamp/failures_total)."""
    src = _read(_GRAPH)
    # Extract _get_plan_graph function body
    m = re.search(
        r"def\s+_get_plan_graph\s*\([^)]*\)\s*:\s*\n(.*?)(?=\n\ndef\s|\Z)",
        src,
        re.DOTALL,
    )
    assert m is not None, "No se encontró función `_get_plan_graph` en graph_orchestrator."
    body = m.group(1)
    # Must declare _PLAN_GRAPH_LAST_BUILD_ERROR in global statement
    assert "_PLAN_GRAPH_LAST_BUILD_ERROR" in body, (
        "_get_plan_graph() debe referenciar `_PLAN_GRAPH_LAST_BUILD_ERROR` "
        "(via `global ...` + asignación en el except path)."
    )
    # Verify the snapshot dict structure with required keys
    snap_pat = re.compile(
        r"_PLAN_GRAPH_LAST_BUILD_ERROR\s*=\s*\{[^}]*['\"]type['\"]\s*:.*?['\"]message['\"]\s*:.*?['\"]timestamp['\"]\s*:.*?['\"]failures_total['\"]",
        re.DOTALL,
    )
    assert snap_pat.search(body), (
        "El except path debe escribir un dict con keys type / message / "
        "timestamp / failures_total en _PLAN_GRAPH_LAST_BUILD_ERROR."
    )


def test_message_truncated_to_avoid_leak():
    """El `message` del snapshot debe truncar `str(e)` para evitar leak de
    paths/SQL/stack en el body del 503. Verifica que hay slicing `[:240]`
    (o similar) cerca del campo message."""
    src = _read(_GRAPH)
    # Pattern: "message": str(e)[:240]  (o similar)
    pat = re.compile(r"['\"]message['\"]\s*:\s*str\(e\)\[:\s*\d+\s*\]")
    assert pat.search(src), (
        "El `message` del snapshot debe truncar str(e) (ej. `str(e)[:240]`) "
        "para evitar leak de paths internos o SQL en el body público del 503."
    )


def test_success_path_resets_snapshot():
    """Tras un build exitoso, el snapshot debe quedar en None — sino el
    operador ve `ready=true` con un `reason` apuntando a un error viejo."""
    src = _read(_GRAPH)
    # En el path post-success de _get_plan_graph, debe haber asignación
    # _PLAN_GRAPH_LAST_BUILD_ERROR = None
    m = re.search(
        r"_PLAN_GRAPH\s*=\s*graph[^\n]*\n[^\n]*\n[^\n]*_PLAN_GRAPH_LAST_BUILD_ERROR\s*=\s*None",
        src,
        re.DOTALL,
    )
    # Be more permissive
    has_reset = re.search(
        r"_PLAN_GRAPH_LAST_BUILD_ERROR\s*=\s*None.*?logger\.info",
        src,
        re.DOTALL,
    )
    assert m is not None or has_reset is not None, (
        "Tras build exitoso, `_PLAN_GRAPH_LAST_BUILD_ERROR` debe resetearse "
        "a None. Sin reset, el operador ve un error viejo en /ready aunque "
        "el grafo ya esté operativo."
    )


def test_is_plan_graph_ready_with_reason_signature():
    src = _read(_GRAPH)
    pat = re.compile(
        r"def\s+is_plan_graph_ready_with_reason\s*\(\s*\)\s*->\s*tuple\[\s*bool\s*,\s*str\s*\|\s*None\s*\]\s*:",
    )
    assert pat.search(src), (
        "Falta función `is_plan_graph_ready_with_reason() -> tuple[bool, str | None]` "
        "en graph_orchestrator."
    )


def test_function_returns_uninitialized_state():
    """Cuando _PLAN_GRAPH es None Y no hay error registrado, reason debe
    ser `'uninitialized'` (estado raro pero posible si warm_plan_graph no
    corrió aún)."""
    src = _read(_GRAPH)
    assert '"uninitialized"' in src or "'uninitialized'" in src, (
        "Función debe retornar `uninitialized` cuando nunca se intentó build. "
        "Sin este estado el operador no distingue 'crash' de 'pending'."
    )


def test_function_returns_build_failed_with_type_msg_n():
    """Formato del reason de error debe ser `build_failed:<ExcType>:<msg>:<n>`."""
    src = _read(_GRAPH)
    assert "build_failed:" in src, (
        "El reason del fallo debe usar prefijo `build_failed:` para que "
        "orquestadores puedan dispatchear por tipo."
    )


def test_app_imports_with_reason_variant():
    src = _read(_APP)
    pat = re.compile(
        r"from\s+graph_orchestrator\s+import\s+\([^)]*is_plan_graph_ready_with_reason",
        re.DOTALL,
    )
    assert pat.search(src), (
        "`app.py` debe importar `is_plan_graph_ready_with_reason` desde "
        "graph_orchestrator."
    )


def test_ready_endpoint_uses_with_reason_variant():
    """El handler de `/ready` debe llamar `is_plan_graph_ready_with_reason()`
    (no solo `is_plan_graph_ready()`) y propagar el reason al body del 503."""
    src = _read(_APP)
    # Aislar el handler readiness_check
    m = re.search(
        r"def\s+readiness_check\s*\([^)]*\)[^:]*:(.*?)(?=\n@app\.|\n\ndef\s|\Z)",
        src,
        re.DOTALL,
    )
    assert m is not None, "No se encontró handler `readiness_check`."
    body = m.group(1)
    assert "is_plan_graph_ready_with_reason()" in body, (
        "`/ready` handler debe llamar `is_plan_graph_ready_with_reason()` "
        "(la variante con reason), no `is_plan_graph_ready()` plain."
    )
    # El body del 503 debe incluir `"reason": <variable>`
    assert re.search(r'["\']reason["\']\s*:\s*reason', body), (
        "El body del 503 debe incluir `\"reason\": reason` (la variable "
        "obtenida del helper) para que orquestadores lo loguen."
    )


def test_anchor_present_in_test_file():
    """Cross-link guard P2-HIST-AUDIT-14."""
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P3-READY-REASON" in src
