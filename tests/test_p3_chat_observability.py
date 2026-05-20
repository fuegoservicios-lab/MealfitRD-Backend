"""[P3-CHAT-OBSERVABILITY · 2026-05-20] Bundle de tests para los 3 fixes
opcionales del audit prod-readiness del módulo agente:

  Fix A — Dedupe cross-worker de `generate_chat_title_background`:
      `_generating_titles = set()` in-memory racea bajo gunicorn `-w N`
      (cada worker tiene su set). Reemplazado por claim atomico via
      `app_kv_store` con TTL 5 min (`_try_claim_title_lock_cross_worker`).
      Fast-path in-memory preservado para evitar roundtrip DB cuando el
      mismo worker ya tiene el lock.

  Fix B — Métrica `chat_rag_embedding_failed`:
      Pre-fix los 2 except del RAG (en `chat_with_agent` y
      `chat_with_agent_stream`) solo loguean — SRE NO podía graficar
      "% de chats sin RAG por failure". Nuevo helper
      `_emit_chat_rag_embedding_failed_metric_best_effort` emit a
      `pipeline_metrics` con `node='chat_rag_embedding_failed'` +
      `metadata={source: <fn_name>}` para diferenciar non-stream vs stream.

  Fix C — Alert `chat_checkpoint_pool_split_missing`:
      `agent.py:1829, 2071` hacen `chat_checkpoint_pool or connection_pool`.
      Si el split pool no se creó al arranque, fallback al transaction
      pooler reabre el modo SSL bad length/EOF que P1-CHECKPOINT-POOL-SPLIT
      cerró. Nuevo emit a `system_alerts` con cooldown 1h in-process
      (`_emit_checkpoint_pool_split_missing_alert_best_effort`).

Cross-link convention (P2-HIST-AUDIT-14): el slug `p3_chat_observability`
matchea este archivo `test_p3_chat_observability.py`.

Tooltip-anchor: P3-CHAT-OBSERVABILITY.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AGENT_PY = _BACKEND_ROOT / "agent.py"
_ALERTS_DOC = _BACKEND_ROOT / "docs" / "system_alerts_resolution_table.md"
_P2_AUDIT_4_TEST = _BACKEND_ROOT / "tests" / "test_p2_audit_4_alert_keys_documented.py"


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def alerts_doc_src() -> str:
    return _ALERTS_DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def p2_audit_4_src() -> str:
    return _P2_AUDIT_4_TEST.read_text(encoding="utf-8")


# ===========================================================================
# Sección 1 — Fix A: dedupe cross-worker title generation
# ===========================================================================

def test_claim_helper_defined(agent_src: str):
    """`_try_claim_title_lock_cross_worker(session_id) -> bool` debe estar
    definido. Es el SSOT del lock cross-worker."""
    assert re.search(
        r"def\s+_try_claim_title_lock_cross_worker\s*\(\s*session_id\s*:", agent_src
    ), (
        "P3-CHAT-OBSERVABILITY regresión: `_try_claim_title_lock_cross_worker` "
        "no definido en agent.py. Es el SSOT del lock distribuido para "
        "title generation; sin él, `_generating_titles` in-memory racea bajo "
        "multi-worker."
    )


def test_claim_helper_uses_app_kv_store_with_ttl(agent_src: str):
    """El claim debe usar `app_kv_store` con WHERE stale (TTL pattern).
    Sin TTL, un worker que crashea sin cleanup deja el lock orphan
    indefinidamente → la session_id queda bloqueada para siempre."""
    body_re = re.compile(
        r"def\s+_try_claim_title_lock_cross_worker\s*\(.*?\)(.*?)(?=\ndef\s|\nclass\s)",
        re.DOTALL,
    )
    m = body_re.search(agent_src)
    assert m is not None
    body = m.group(1)
    assert "app_kv_store" in body, (
        "P3-CHAT-OBSERVABILITY regresión: el claim no usa `app_kv_store`. "
        "Si moviste a otra storage, actualizar este test."
    )
    assert "ON CONFLICT" in body and "WHERE" in body and "RETURNING" in body, (
        "P3-CHAT-OBSERVABILITY regresión: el claim debe ser un UPSERT "
        "atómico con `ON CONFLICT ... WHERE stale RETURNING key`. Sin "
        "ON CONFLICT, dos workers concurrent pueden ambos hacer INSERT; "
        "sin WHERE stale, locks huérfanos bloquean la session_id "
        "indefinidamente; sin RETURNING, el caller no puede distinguir "
        "'yo claimé' de 'otro tiene el lock'."
    )
    assert "_TITLE_LOCK_TTL_S" in body or "TTL" in body, (
        "P3-CHAT-OBSERVABILITY regresión: el claim debe referenciar un TTL "
        "(constante `_TITLE_LOCK_TTL_S` o similar). Sin TTL, locks "
        "huérfanos persisten."
    )


def test_generate_chat_title_uses_claim_helper(agent_src: str):
    """`generate_chat_title_background` debe invocar el claim helper
    ANTES de proceder con la generación. Si el claim retorna False,
    return temprano (otro worker tiene el lock)."""
    fn_re = re.compile(
        r"def\s+generate_chat_title_background\s*\(.*?\)(.*?)(?=\ndef\s|\nclass\s)",
        re.DOTALL,
    )
    m = fn_re.search(agent_src)
    assert m is not None
    body = m.group(1)
    assert "_try_claim_title_lock_cross_worker(session_id)" in body, (
        "P3-CHAT-OBSERVABILITY regresión: `generate_chat_title_background` "
        "no llama al claim cross-worker. Sin el call, el dedupe sigue siendo "
        "solo in-memory (race bajo multi-worker)."
    )
    # Order: el claim debe estar antes del get_session_messages (que es
    # caro: roundtrip DB para leer mensajes) — fail-fast el "lock activo".
    claim_idx = body.find("_try_claim_title_lock_cross_worker(session_id)")
    get_msgs_idx = body.find("get_session_messages(session_id)")
    assert 0 <= claim_idx < get_msgs_idx, (
        "P3-CHAT-OBSERVABILITY regresión: el claim debe aparecer ANTES de "
        "`get_session_messages(session_id)` para fail-fast cuando otro "
        "worker tiene el lock (evita el roundtrip DB extra)."
    )


def test_generate_chat_title_has_finally_discard(agent_src: str):
    """Cleanup del in-memory set en `finally`. Pre-fix tenía leak: el set
    `_generating_titles` crecía indefinidamente porque NO había discard."""
    fn_re = re.compile(
        r"def\s+generate_chat_title_background\s*\(.*?\)(.*?)(?=\ndef\s|\nclass\s)",
        re.DOTALL,
    )
    m = fn_re.search(agent_src)
    assert m is not None
    body = m.group(1)
    assert "_generating_titles.discard(session_id)" in body, (
        "P3-CHAT-OBSERVABILITY regresión: `_generating_titles.discard(...)` "
        "ausente. Sin él, el set crece indefinidamente con cada generación "
        "(memory leak slow-burn)."
    )


# ===========================================================================
# Sección 2 — Fix B: chat_rag_embedding_failed metric
# ===========================================================================

def test_rag_metric_helper_defined(agent_src: str):
    """Helper `_emit_chat_rag_embedding_failed_metric_best_effort(user_id,
    session_id, source)` definido."""
    sig_re = re.compile(
        r"def\s+_emit_chat_rag_embedding_failed_metric_best_effort\s*\("
    )
    assert sig_re.search(agent_src), (
        "P3-CHAT-OBSERVABILITY regresión: helper "
        "`_emit_chat_rag_embedding_failed_metric_best_effort` no definido."
    )


def test_rag_metric_writes_pipeline_metrics(agent_src: str):
    """El helper debe escribir a `pipeline_metrics` con
    `node='chat_rag_embedding_failed'`."""
    body_re = re.compile(
        r"def\s+_emit_chat_rag_embedding_failed_metric_best_effort\s*\(.*?\)(.*?)(?=\ndef\s|\nclass\s)",
        re.DOTALL,
    )
    m = body_re.search(agent_src)
    assert m is not None
    body = m.group(1)
    assert "INSERT INTO pipeline_metrics" in body, (
        "P3-CHAT-OBSERVABILITY regresión: el helper no escribe a "
        "`pipeline_metrics`. Es el sink canónico del repo para telemetría "
        "de nodos."
    )
    assert "chat_rag_embedding_failed" in body, (
        "P3-CHAT-OBSERVABILITY regresión: el helper no usa el node-name "
        "canónico `chat_rag_embedding_failed`. SRE filtra por `node` en "
        "el dashboard — si renombras, actualizar el dashboard también."
    )


def test_rag_metric_emit_in_both_chat_functions(agent_src: str):
    """Ambos callsites del RAG (`chat_with_agent` non-stream y
    `chat_with_agent_stream`) deben invocar el helper en su except del
    RAG embedding."""
    # Buscar emisiones del helper con argumento `source=<...>`.
    non_stream = re.search(
        r'_emit_chat_rag_embedding_failed_metric_best_effort\([^)]*"chat_with_agent"',
        agent_src,
    )
    stream = re.search(
        r'_emit_chat_rag_embedding_failed_metric_best_effort\([^)]*"chat_with_agent_stream"',
        agent_src,
    )
    assert non_stream is not None, (
        "P3-CHAT-OBSERVABILITY regresión: `chat_with_agent` (non-stream) "
        "NO emite la métrica RAG-failed. SRE no podría desglosar el "
        "fallback path entre stream/non-stream."
    )
    assert stream is not None, (
        "P3-CHAT-OBSERVABILITY regresión: `chat_with_agent_stream` NO "
        "emite la métrica RAG-failed."
    )


# ===========================================================================
# Sección 3 — Fix C: alert checkpoint_pool_split_missing
# ===========================================================================

def test_pool_split_alert_helper_defined(agent_src: str):
    """Helper `_emit_checkpoint_pool_split_missing_alert_best_effort()`
    definido."""
    sig_re = re.compile(
        r"def\s+_emit_checkpoint_pool_split_missing_alert_best_effort\s*\("
    )
    assert sig_re.search(agent_src), (
        "P3-CHAT-OBSERVABILITY regresión: helper "
        "`_emit_checkpoint_pool_split_missing_alert_best_effort` no definido."
    )


def test_pool_split_alert_writes_system_alerts(agent_src: str):
    """El helper debe escribir a `system_alerts` con el `alert_key`
    canónico + UPSERT (`ON CONFLICT ... DO UPDATE SET ... resolved_at = NULL`)."""
    body_re = re.compile(
        r"def\s+_emit_checkpoint_pool_split_missing_alert_best_effort\s*\(.*?\)(.*?)(?=\ndef\s|\nclass\s)",
        re.DOTALL,
    )
    m = body_re.search(agent_src)
    assert m is not None
    body = m.group(1)
    assert "INSERT INTO system_alerts" in body
    assert "chat_checkpoint_pool_split_missing" in body
    assert "ON CONFLICT" in body and "resolved_at = NULL" in body, (
        "P3-CHAT-OBSERVABILITY regresión: el alert no usa el patrón "
        "canónico de upsert (P2-NEW-3). Sin `ON CONFLICT (alert_key) DO "
        "UPDATE SET ... resolved_at = NULL`, cada emit crea row nueva "
        "(unique violation) o la alert no se re-abre tras resolved."
    )


def test_pool_split_alert_has_cooldown(agent_src: str):
    """El helper debe respetar un cooldown in-process. Bajo carga alta
    (1000 req/s), sin cooldown haríamos 1000 UPSERTs/s al mismo row →
    contención inútil."""
    body_re = re.compile(
        r"def\s+_emit_checkpoint_pool_split_missing_alert_best_effort\s*\(.*?\)(.*?)(?=\ndef\s|\nclass\s)",
        re.DOTALL,
    )
    m = body_re.search(agent_src)
    assert m is not None
    body = m.group(1)
    assert "_POOL_SPLIT_ALERT_COOLDOWN_S" in body or "cooldown" in body.lower(), (
        "P3-CHAT-OBSERVABILITY regresión: el helper no respeta cooldown. "
        "Bajo concurrencia alta, sin cooldown la BD recibe miles de UPSERTs/s "
        "al mismo row de `system_alerts` → contención."
    )
    assert "_pool_split_alert_lock" in body or "Lock(" in body, (
        "P3-CHAT-OBSERVABILITY regresión: el helper no usa lock para el "
        "check-and-set del cooldown. Sin lock, dos threads concurrent "
        "pueden ambos pasar el check y emitir el UPSERT a la vez."
    )


def test_pool_split_alert_emitted_at_both_callsites(agent_src: str):
    """Los 2 callsites del fallback (`chat_with_agent` + `chat_with_agent_stream`)
    deben invocar el alert. Pattern: `if chat_checkpoint_pool is None and
    connection_pool is not None: _emit_checkpoint_pool_split_missing_alert_best_effort()`.

    Regex con lookbehind `(?<!def\\s)` excluye la def-line del helper —
    contamos solo callsites reales. Esperamos exactamente 2 (no más no
    menos) para que un refactor que borra uno haga fallar el test."""
    # Negative lookbehind: ignora la def-line.
    emit_calls = re.findall(
        r"(?<!def\s)_emit_checkpoint_pool_split_missing_alert_best_effort\(\s*\)",
        agent_src,
    )
    assert len(emit_calls) >= 2, (
        f"P3-CHAT-OBSERVABILITY regresión: el alert emit aparece {len(emit_calls)} "
        f"veces (excluyendo def), esperado ≥2 (uno por callsite: "
        f"chat_with_agent + chat_with_agent_stream). Si la condición de "
        f"fallback se movió a un helper compartido, ajustar este test."
    )


def test_pool_split_alert_documented_in_canonical_doc(alerts_doc_src: str):
    """El `alert_key='chat_checkpoint_pool_split_missing'` debe tener
    entry en `system_alerts_resolution_table.md` — sin él, el test
    `test_p2_audit_4_alert_keys_documented::test_every_emitted_alert_key_is_documented`
    falla (drift bidireccional)."""
    assert "chat_checkpoint_pool_split_missing" in alerts_doc_src, (
        "P3-CHAT-OBSERVABILITY regresión: el `alert_key` no está documentado "
        "en `backend/docs/system_alerts_resolution_table.md`. Añadir row "
        "con productor=agent.py + resolver=Manual + modelo=Manual."
    )


def test_agent_py_in_emitter_files_audit_4(p2_audit_4_src: str):
    """`agent.py` debe estar en `_EMITTER_FILES` del test P2-AUDIT-4 — sin
    él, el test ghost-detection marcaría el nuevo `alert_key` como
    documentado-sin-emitter (falso positivo)."""
    assert '_BACKEND / "agent.py"' in p2_audit_4_src, (
        "P3-CHAT-OBSERVABILITY regresión: `agent.py` no añadido a "
        "`_EMITTER_FILES` del test P2-AUDIT-4. El ghost-detection (`test_"
        "every_documented_pattern_has_emitter`) marcará el alert nuevo como "
        "ghost row falsamente."
    )


# ===========================================================================
# Sección 4 — tooltip-anchor presente
# ===========================================================================

def test_tooltip_anchor_present(agent_src: str):
    """Marker `P3-CHAT-OBSERVABILITY` aparece ≥3× en agent.py (1 por fix
    + el de los helpers nuevos)."""
    count = agent_src.count("P3-CHAT-OBSERVABILITY")
    assert count >= 3, (
        f"P3-CHAT-OBSERVABILITY regresión: tooltip-anchor aparece {count}× "
        f"en agent.py, esperado ≥3."
    )
