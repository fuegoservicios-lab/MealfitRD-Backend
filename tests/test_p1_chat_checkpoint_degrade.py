"""[P1-CHAT-CHECKPOINT-DEGRADE · 2026-05-20] Degradación silenciosa cuando
el `PostgresSaver.put_writes` final muere por SSL bad length / EOF detected
después de que el stream ya entregó contenido al frontend.

Bug observado en runtime el 2026-05-20 (tras P1-CHECKPOINT-POOL-SPLIT y
P1-CHAT-CHECKPOINT-FIX del mismo día):
    Banner rojo "El asistente tuvo un problema procesando tu mensaje"
    aparece DESPUÉS de que la respuesta completa del LLM se renderizó en
    la conversación. La causa raíz residual: incluso con
    `chat_checkpoint_pool` apuntando a `:5432` (session mode), Supavisor
    session-pooler también mata conns idle (~60-70s threshold), solo que
    menos agresivamente que el transaction pooler. Cuando LangGraph
    mantiene una conexión checkout durante el LLM call (10-30s), una conn
    que ya envejeció ~40s en el pool muere mid-pipeline al `put_writes`
    final → SSL bad length / EOF detected.

Fix defense-in-depth (dos capas):
    1. **Pool recycling agresivo** (`db_core.py`): `chat_checkpoint_pool`
       baja `min_size=1 → 0` y `max_idle=300 → 30s`. Sin pre-warming las
       conns nunca envejecen idle más de 30s, casi eliminando el modo de
       fallo.
    2. **Silent degrade** (`agent.py:chat_with_agent_stream`): si la
       excepción menciona markers SSL/EOF Y `_chunks_yielded > 0`, log
       a WARN, setea `_stream_outcome = "checkpoint_lost"`, y retorna
       SIN yield 'error' al frontend. El user ya vio la respuesta;
       perder el checkpoint solo afecta el próximo turn (que recarga
       history desde db_chat, no-op visible).

Trade-off conocido:
    - El degrade silencioso pierde el checkpoint LangGraph del turn. El
      próximo chat re-carga history desde `db_chat.get_session_messages`
      (que SÍ se persiste correctamente, ese write no depende del pool
      problemático). Cero impacto visible al user.
    - Si `_chunks_yielded == 0` el degrade NO aplica → banner sale, porque
      indica fallo real (conn dead antes del primer token LLM).
    - Telemetría `pipeline_metrics` registra outcome=checkpoint_lost para
      que SRE pueda graficar frecuencia. Si >5% sostenido, escalar a
      MemorySaver completo (deprecar chat_checkpoint_pool).

Cross-link convention (P2-HIST-AUDIT-14): slug `p1_chat_checkpoint_degrade`
matchea este archivo `test_p1_chat_checkpoint_degrade.py`.

Tooltip-anchor: P1-CHAT-CHECKPOINT-DEGRADE.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_CORE_PY = _BACKEND_ROOT / "db_core.py"
_AGENT_PY = _BACKEND_ROOT / "agent.py"


@pytest.fixture(scope="module")
def db_core_src() -> str:
    return _DB_CORE_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


# ============================================================
# Capa 1: pool recycling agresivo en db_core.py
# ============================================================

def test_chat_checkpoint_pool_min_size_zero(db_core_src: str):
    """El `chat_checkpoint_pool` debe construirse con `min_size=0` — sin
    pre-warming. Pre-fix `min_size=1` mantenía una conn idle desde startup
    que envejecía ~40-60s antes del primer chat, llegando al kill threshold
    de Supavisor mid-stream."""
    pool_block = re.search(
        r"chat_checkpoint_pool\s*=\s*ConnectionPool\(\s*"
        r"conninfo\s*=\s*original_session_url\s*,\s*"
        r"min_size\s*=\s*(\d+)",
        db_core_src,
        re.DOTALL,
    )
    assert pool_block, (
        "Bloque `chat_checkpoint_pool = ConnectionPool(conninfo=original_session_url, min_size=...)` "
        "no encontrado en db_core.py — refactor inesperado."
    )
    min_size = int(pool_block.group(1))
    assert min_size == 0, (
        f"P1-CHAT-CHECKPOINT-DEGRADE regresión: chat_checkpoint_pool tiene "
        f"min_size={min_size}, esperado 0. Sin esto, el pool pre-warma 1 "
        f"conn al startup que muere idle antes del primer chat → SSL bad "
        f"length / EOF detected mid-stream."
    )


def test_chat_checkpoint_pool_max_idle_short(db_core_src: str):
    """El `max_idle` del `chat_checkpoint_pool` debe ser ≤ 60s. Pre-fix
    estaba en 300s (5 min) — más allá del threshold de Supavisor (~60-70s)
    para matar conns idle. Con max_idle=30s las conns se reciclan antes
    del kill."""
    pool_block = re.search(
        r"chat_checkpoint_pool\s*=\s*ConnectionPool\([^)]*?max_idle\s*=\s*([\d.]+)",
        db_core_src,
        re.DOTALL,
    )
    assert pool_block, (
        "max_idle no encontrado en bloque chat_checkpoint_pool — refactor inesperado."
    )
    max_idle = float(pool_block.group(1))
    assert 0 < max_idle <= 60.0, (
        f"P1-CHAT-CHECKPOINT-DEGRADE regresión: chat_checkpoint_pool tiene "
        f"max_idle={max_idle}s, esperado ≤60s. Supavisor session pooler mata "
        f"conns idle ~60-70s; max_idle más alto deja conns rancias en el "
        f"pool que mueren mid-pipeline al `put_writes`."
    )


# ============================================================
# Capa 2: silent degrade en agent.py:chat_with_agent_stream
# ============================================================

def test_chunks_yielded_counter_initialized(agent_src: str):
    """`_chunks_yielded = 0` debe declararse cerca del top de
    chat_with_agent_stream, junto con los otros budget trackers. Sin él
    el degrade no puede distinguir "stream produjo contenido" de "stream
    murió antes del primer token"."""
    # Buscamos `_chunks_yielded = 0` con comment marker.
    pattern = re.compile(
        r"_chunks_yielded\s*=\s*0",
    )
    assert pattern.search(agent_src), (
        "P1-CHAT-CHECKPOINT-DEGRADE regresión: contador `_chunks_yielded = 0` "
        "ausente del setup de chat_with_agent_stream. Sin él, el degrade "
        "silencioso no puede gatear sobre 'stream entregó contenido'."
    )


def test_chunks_yielded_increment_after_yield(agent_src: str):
    """`_chunks_yielded += 1` debe aparecer DESPUÉS de un yield de tipo
    'chunk' al wire SSE. Es el único punto donde sabemos que el frontend
    recibió contenido visible. Sin el increment, el contador queda en 0
    forever y el degrade NO se activa."""
    # Buscamos el patrón yield 'chunk' inmediatamente seguido (en <=4 líneas)
    # del increment.
    lines = agent_src.splitlines()
    yield_lineno = next(
        (i for i, ln in enumerate(lines)
         if "'type': 'chunk'" in ln and "yield" in ln),
        None,
    )
    assert yield_lineno is not None, (
        "yield de 'type': 'chunk' no encontrado en agent.py — refactor inesperado del stream."
    )
    # Buscar `_chunks_yielded += 1` en las siguientes 5 líneas.
    follow_block = "\n".join(lines[yield_lineno:yield_lineno + 6])
    assert "_chunks_yielded += 1" in follow_block, (
        f"P1-CHAT-CHECKPOINT-DEGRADE regresión: `_chunks_yielded += 1` no "
        f"aparece dentro de las 5 líneas posteriores al yield 'chunk' (línea "
        f"{yield_lineno + 1}). Sin el increment el contador no sube nunca, "
        f"el degrade no se gatea, y el banner vuelve a mostrarse."
    )


def test_ssl_markers_classified_in_except(agent_src: str):
    """El `except Exception` del stream loop debe clasificar la excepción
    contra una lista de markers SSL/EOF antes de yield 'error'. La lista
    incluye 'SSL error: bad length' (psycopg signature) y 'EOF detected'
    (SSL SYSCALL signature)."""
    required_markers = (
        "SSL error: bad length",
        "EOF detected",
        "flush request failed",
        "connection is lost",
    )
    for marker in required_markers:
        assert marker in agent_src, (
            f"P1-CHAT-CHECKPOINT-DEGRADE regresión: marker '{marker}' "
            f"ausente del classifier en agent.py. Sin él, la excepción "
            f"correspondiente caería al yield 'error' y el banner volvería "
            f"a mostrarse."
        )


def test_degrade_gated_by_chunks_yielded(agent_src: str):
    """El degrade silencioso DEBE estar gateado por `_chunks_yielded > 0`.
    Si la conn muere ANTES del primer token (chunks_yielded==0), el fallo
    es real y el user debe ver el banner para reintentar."""
    # Buscamos el if que combina `_is_checkpoint_ssl_death` y `_chunks_yielded > 0`.
    pattern = re.search(
        r"if\s+_is_checkpoint_ssl_death\s+and\s+_chunks_yielded\s*>\s*0\s*:",
        agent_src,
    )
    assert pattern, (
        "P1-CHAT-CHECKPOINT-DEGRADE regresión: el gate "
        "`if _is_checkpoint_ssl_death and _chunks_yielded > 0:` ausente. "
        "Sin el gate, una conn que muere antes del primer token también "
        "se degradaría silencio — el user no vería ningún mensaje ni "
        "banner, parecería que el chat 'no respondió'."
    )


def test_degrade_branch_returns_without_yielding_error(agent_src: str):
    """El branch del degrade DEBE retornar sin yield 'error'. Si yieldase
    error, el frontend mostraría el banner aunque la respuesta ya esté
    visible — peor que pre-fix porque el banner aparece sobre contenido
    completo."""
    # Buscamos el bloque del degrade y verificamos que NO contiene yield 'error'
    # entre el if del classifier y el return.
    branch_match = re.search(
        r"if\s+_is_checkpoint_ssl_death\s+and\s+_chunks_yielded\s*>\s*0\s*:"
        r"(.*?)return",
        agent_src,
        re.DOTALL,
    )
    assert branch_match, "Branch del degrade no encontrado."
    branch_body = branch_match.group(1)
    assert "yield" not in branch_body or "'type': 'error'" not in branch_body, (
        "P1-CHAT-CHECKPOINT-DEGRADE regresión: el branch del degrade "
        "contiene yield 'type': 'error' — el degrade NO debe emitir el "
        "banner. Si necesitas un yield informativo, usa 'type': 'info' "
        "o 'type': 'warning' (no 'error')."
    )


def test_degrade_sets_outcome_checkpoint_lost(agent_src: str):
    """El degrade debe setear `_stream_outcome = "checkpoint_lost"` para
    que la telemetría a `pipeline_metrics` distinga este caso de
    ok/error/timeout. SRE puede graficar frecuencia y escalar a
    MemorySaver completo si pasa de 5%."""
    branch_match = re.search(
        r"if\s+_is_checkpoint_ssl_death\s+and\s+_chunks_yielded\s*>\s*0\s*:"
        r"(.*?)return",
        agent_src,
        re.DOTALL,
    )
    assert branch_match, "Branch del degrade no encontrado."
    assert 'checkpoint_lost' in branch_match.group(1), (
        "P1-CHAT-CHECKPOINT-DEGRADE regresión: el branch del degrade no "
        "setea `_stream_outcome = \"checkpoint_lost\"`. Sin esto la "
        "telemetría agrupa estos casos como 'ok' y SRE no puede ver "
        "frecuencia para decidir escalado."
    )


def test_tooltip_anchor_in_db_core(db_core_src: str):
    """Marker `P1-CHAT-CHECKPOINT-DEGRADE` aparece ≥1× en db_core.py."""
    assert db_core_src.count("P1-CHAT-CHECKPOINT-DEGRADE") >= 1, (
        "P1-CHAT-CHECKPOINT-DEGRADE: tooltip-anchor ausente del bloque "
        "del pool recycling en db_core.py. Sin él un refactor del bloque "
        "no link-back-a este test."
    )


def test_tooltip_anchor_in_agent(agent_src: str):
    """Marker `P1-CHAT-CHECKPOINT-DEGRADE` aparece ≥2× en agent.py
    (al menos: setup del counter + bloque del classifier)."""
    count = agent_src.count("P1-CHAT-CHECKPOINT-DEGRADE")
    assert count >= 2, (
        f"P1-CHAT-CHECKPOINT-DEGRADE: tooltip-anchor aparece {count}× "
        f"en agent.py, esperado ≥2 (setup del counter + bloque classifier)."
    )
