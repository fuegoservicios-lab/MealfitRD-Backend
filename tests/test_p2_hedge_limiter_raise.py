"""[P2-HEDGE-LIMITER-RAISE + P2-HEDGE-EXC-DETAIL · 2026-05-16] Dos defensas
contra el modo de fallo observado en plan bf6f1383 (2026-05-16):

  1. Día 1 quedó sin hedge porque `HEDGE_MAX_CONCURRENT = max(1, 4 // 2) = 2`
     ya estaba saturado por Días 2-3. Sin hedge, el único retry secuencial
     del primary falló con `Exception` opaca → CB OPEN → self_critique
     bloqueado → plan emergency rechazado por el fallback guard.

     Fix: knob `MEALFIT_HEDGE_MAX_CONCURRENT` (default 3) reemplaza el cálculo
     hardcoded. 3 garantiza que los 3 day_generators paralelos del chunk
     inicial tengan protección simétrica. Clamp superior `LLM_MAX_CONCURRENT-1`
     en runtime preserva headroom para primaries.

  2. El log decía solo `Falló definitivamente tras hedging: Exception` — no
     se podía distinguir 503 Gemini vs rate-limit vs CB-OPEN vs TimeoutError
     (ceiling). 8 min de diagnóstico perdidos en el incidente.

     Fix: incluir `type(err).__name__ + str(err)[:300]` + `exc_info=...` para
     traceback completo. Branch defensivo si `err is None` (no debería ocurrir
     dado `_safe_gen` siempre captura, pero el contrato queda explícito).

Tests parser-based porque el código no es ejecutable en aislamiento (depende
de LLM_SEMAPHORE distribuido + Redis + asyncio loop). Lo importante es que
nadie revierta accidentalmente el comportamiento.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND_ROOT / "graph_orchestrator.py").read_text(encoding="utf-8")
_ENV = (_BACKEND_ROOT / ".env").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fix #3: HEDGE_MAX_CONCURRENT knob
# ---------------------------------------------------------------------------


def test_hedge_max_concurrent_knob_declared_with_default_3():
    """`HEDGE_MAX_CONCURRENT_KNOB` debe leerse desde
    `MEALFIT_HEDGE_MAX_CONCURRENT` con default 3.

    Default 3 cubre el caso real (3 day_generators paralelos en el chunk
    inicial). Bajar a 2 revertiría el fix; subir sin cuidado puede saturar
    el LLM_SEMAPHORE global (LLM_MAX_CONCURRENT default 4).
    """
    pattern = (
        r'HEDGE_MAX_CONCURRENT_KNOB\s*=\s*_env_int\s*\(\s*'
        r'"MEALFIT_HEDGE_MAX_CONCURRENT"\s*,\s*(\d+)\s*\)'
    )
    m = re.search(pattern, _GRAPH)
    assert m, (
        "HEDGE_MAX_CONCURRENT_KNOB no declarado vía _env_int en "
        "graph_orchestrator.py. Reemplazo de hardcoded `// 2` perdido — "
        "revisar P2-HEDGE-LIMITER-RAISE · 2026-05-16."
    )
    default = int(m.group(1))
    assert default == 3, (
        f"Default del knob `MEALFIT_HEDGE_MAX_CONCURRENT`={default}, esperado 3. "
        "Subir a >3 satura LLM_SEMAPHORE (default 4); bajar revierte el fix."
    )


def test_hedge_max_concurrent_uses_knob_with_clamp():
    """Dentro del nodo `generate_days_parallel_node`, el cálculo final de
    `HEDGE_MAX_CONCURRENT` debe (a) usar el knob, (b) tener floor 1, (c)
    tener clamp superior `LLM_SEMAPHORE.max_concurrent - 1` para preservar
    headroom para primaries.
    """
    # Capturar el bloque de cálculo (5-10 líneas después del knob ref).
    block_pattern = (
        r"_hedge_cap_ceiling\s*=\s*max\(\s*1\s*,\s*LLM_SEMAPHORE\.max_concurrent\s*-\s*1\s*\)"
        r"\s*\n\s*HEDGE_MAX_CONCURRENT\s*=\s*max\(\s*1\s*,\s*min\(\s*HEDGE_MAX_CONCURRENT_KNOB\s*,"
        r"\s*_hedge_cap_ceiling\s*\)\s*\)"
    )
    assert re.search(block_pattern, _GRAPH), (
        "Cálculo de HEDGE_MAX_CONCURRENT NO usa el knob con clamp esperado. "
        "Patrón requerido:\n"
        "  _hedge_cap_ceiling = max(1, LLM_SEMAPHORE.max_concurrent - 1)\n"
        "  HEDGE_MAX_CONCURRENT = max(1, min(HEDGE_MAX_CONCURRENT_KNOB, _hedge_cap_ceiling))\n"
        "Esto enforza floor 1 + ceiling = max_concurrent - 1 (headroom para "
        "primaries). Si modificas la fórmula, actualizar este test."
    )


def test_hedge_max_concurrent_no_hardcoded_div_2():
    """Defensa anti-regresión: la fórmula vieja
    `max(1, LLM_SEMAPHORE.max_concurrent // 2)` NO debe reaparecer en
    el cálculo activo del nodo (el comentario histórico está permitido).
    """
    # Buscamos asignaciones reales, no comentarios:
    bad_pattern = re.compile(
        r"^\s*HEDGE_MAX_CONCURRENT\s*=\s*max\(\s*1\s*,\s*LLM_SEMAPHORE\.max_concurrent\s*//\s*2\s*\)",
        re.MULTILINE,
    )
    assert not bad_pattern.search(_GRAPH), (
        "Asignación vieja `HEDGE_MAX_CONCURRENT = max(1, LLM_SEMAPHORE.max_concurrent // 2)` "
        "detectada — revierte P2-HEDGE-LIMITER-RAISE. Usar el knob."
    )


def test_env_sets_hedge_max_concurrent_at_3():
    """El `.env` debe setear el knob explícitamente a 3 para que un operador
    sepa qué valor está activo sin tener que mirar el código del default."""
    m = re.search(
        r"^MEALFIT_HEDGE_MAX_CONCURRENT\s*=\s*(\d+)\s*$",
        _ENV,
        re.MULTILINE,
    )
    assert m, (
        "Falta `MEALFIT_HEDGE_MAX_CONCURRENT=` en backend/.env. Sin esto, "
        "el valor activo depende del default código (frágil ante refactor)."
    )
    val = int(m.group(1))
    assert val == 3, (
        f"MEALFIT_HEDGE_MAX_CONCURRENT={val} en .env, esperado 3. "
        "Bajar a 2 revertiría el fix del incidente bf6f1383."
    )


def test_env_comment_references_pfix_marker():
    """El bloque de comentario en .env debe nombrar el marker P-fix para que
    un mantenedor futuro pueda buscar la historia."""
    assert "P2-HEDGE-LIMITER-RAISE" in _ENV, (
        "Falta marker `P2-HEDGE-LIMITER-RAISE` en comentario inline de .env. "
        "Sin él, un refactor cosmético borra el contexto del por qué."
    )


# ---------------------------------------------------------------------------
# Fix #2: log exception detail
#
# Helper: extraer un slice del archivo alrededor del literal único del log,
# evitando regex multilinea con backtracking exponencial sobre 10k+ líneas.
# ---------------------------------------------------------------------------


def _hedge_failure_block() -> str:
    """Devuelve un slice de ~1500 chars alrededor del `failed_days.append`
    que sigue al log del fallback hedging. Suficiente para verificar el
    contenido del else-branch sin invocar regex pesados."""
    marker = "failed_days.append(day_num)"
    idx = _GRAPH.find(marker)
    assert idx > 0, "No se encontró `failed_days.append(day_num)` en graph_orchestrator.py."
    start = max(0, idx - 1500)
    end = min(len(_GRAPH), idx + len(marker) + 100)
    return _GRAPH[start:end]


def test_hedge_failure_log_includes_type_and_message():
    """El log de `Falló definitivamente tras hedging` debe incluir:
      (a) `type(err).__name__`
      (b) `str(err)` (con truncate)
      (c) `exc_info=...` para traceback en el handler logging
    """
    body = _hedge_failure_block()
    # El literal del log debe estar en el slice (sanity):
    assert "Falló definitivamente tras hedging" in body, (
        "El literal del log no está en la región esperada — algo movió el "
        "código y el helper necesita ajuste."
    )
    assert "type(err).__name__" in body, (
        "El log no incluye `type(err).__name__` — pierde el tipo de excepción."
    )
    assert "str(err)" in body, (
        "El log no incluye `str(err)` — pierde el mensaje del exception."
    )
    assert "exc_info=" in body, (
        "El log no pasa `exc_info=...` — pierde el traceback completo, "
        "imposibilita root-cause via Sentry/stdout."
    )
    assert "P2-HEDGE-EXC-DETAIL" in body, (
        "Falta marker `P2-HEDGE-EXC-DETAIL` cerca del log — un refactor "
        "cosmético podría borrar el motivo de pasar `exc_info`."
    )


def test_hedge_failure_log_truncates_message():
    """`str(err)` debe truncarse a un cap razonable (≤500 chars). Sin esto,
    una excepción con payload gigante (e.g. stack-trace pegado en el mensaje)
    contaminaría los logs / Sentry events."""
    body = _hedge_failure_block()
    m = re.search(r"str\(err\)[^\n]*?\[\s*:\s*(\d+)\s*\]", body)
    assert m, "No se detectó truncate `[:N]` sobre str(err)."
    cap = int(m.group(1))
    assert 50 <= cap <= 500, (
        f"Truncate de str(err) = {cap}, fuera de [50, 500]. Muy bajo "
        "pierde info útil; muy alto contamina logs en errores patológicos."
    )


def test_hedge_failure_log_defends_against_none_err():
    """Defensa simétrica: si `err is None` (no debería pasar con _safe_gen
    pero el contrato queda explícito), el log debe explicarlo, no crashear
    con AttributeError al hacer `None.__traceback__`."""
    body = _hedge_failure_block()
    has_guard = (
        "err is not None" in body
        or "if err:" in body
        or "if err is None" in body
    )
    assert has_guard, (
        "No hay guard explícito para `err is None`. Si en futuro alguien "
        "cambia `_safe_gen` y pierde la captura del exception, este código "
        "crashearía con AttributeError sobre `err.__traceback__`."
    )
