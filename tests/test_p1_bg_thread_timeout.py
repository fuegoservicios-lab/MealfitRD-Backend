"""[P1-BG-THREAD-TIMEOUT · 2026-05-15] Anchor + regression guard.

Pre-fix `backend/routers/chat.py` lanzaba `threading.Thread(target=fn,
daemon=True).start()` para 2 background tasks (título del chat + bg_tasks
del SSE). Sin timeout, si Gemini/Supabase se cuelgan upstream el thread
daemon vive hasta que el proceso reinicia → memory + GIL pressure
acumulativo → degradación gradual del worker.

Fix: `backend/bg_executor.py::submit_bg_task` con pool compartido bounded +
watcher thread que llama `future.result(timeout=...)` y emite alert
`bg_task_timeout:<name>` si timeout exceeds.

Defensas que este test enforza:
  1. Anchor `P1-BG-THREAD-TIMEOUT` presente en `backend/bg_executor.py`.
  2. Knobs `MEALFIT_BG_TASK_MAX_WORKERS` y `MEALFIT_BG_TASK_TIMEOUT_S`
     resueltos via `_env_int` (auto-registro en `_KNOBS_REGISTRY`).
  3. `submit_bg_task` tiene signature canónica `(fn, *args, task_name, timeout_s=None, **kwargs)`.
  4. `routers/chat.py` NO contiene `threading.Thread(target=` para los
     callsites flagged (título + bg_tasks). DEBE usar `submit_bg_task` en
     su lugar.
  5. Funcional: un task que excede su timeout emite la alerta esperada
     vía monkeypatch del callback de persistencia.
  6. Funcional: un task que termina dentro del timeout NO emite alerta.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BG_EXECUTOR = _REPO_ROOT / "backend" / "bg_executor.py"
_CHAT_ROUTER = _REPO_ROOT / "backend" / "routers" / "chat.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Anchor + structure del fix
# ---------------------------------------------------------------------------
def test_anchor_present_in_bg_executor():
    src = _read(_BG_EXECUTOR)
    assert "P1-BG-THREAD-TIMEOUT" in src, (
        "Falta anchor `P1-BG-THREAD-TIMEOUT` en backend/bg_executor.py."
    )


def test_submit_bg_task_signature():
    """La signature debe ser estable: callers usan kwargs `task_name`
    obligatorio + `timeout_s` opcional. Un rename rompería todos los
    callsites silenciosamente.

    Verificación robusta: aislar el header de la función (desde `def` hasta
    el primer `:` que cierra la firma), normalizar whitespace, y validar el
    orden de los parámetros — sin pelearse con type hints que contengan
    comas internas (`Callable[..., Any]`).
    """
    src = _read(_BG_EXECUTOR)
    header_match = re.search(
        r"def\s+submit_bg_task\s*\((.*?)\)\s*->.*?:",
        src,
        re.DOTALL,
    )
    assert header_match is not None, (
        "No se encontró `def submit_bg_task(...)` con return annotation."
    )
    params_blob = re.sub(r"\s+", " ", header_match.group(1))
    # Orden obligatorio (parámetro por parámetro, sin pelearse con type hints).
    must_have_in_order = [
        r"\bfn\b",
        r"\*args\b",
        r"\btask_name\b",
        r"\btimeout_s\b",
        r"\*\*kwargs\b",
    ]
    last_pos = -1
    for pat in must_have_in_order:
        m = re.search(pat, params_blob)
        assert m is not None, (
            f"`submit_bg_task` no tiene el parámetro `{pat}`. Params: {params_blob!r}"
        )
        assert m.start() > last_pos, (
            f"Parámetros fuera de orden en `submit_bg_task` (`{pat}` antes de "
            f"otro requerido). Params: {params_blob!r}"
        )
        last_pos = m.start()
    # timeout_s debe tener default None (kwarg opcional).
    assert re.search(r"timeout_s\s*:\s*[^=]+=\s*None|timeout_s\s*=\s*None", params_blob), (
        "`timeout_s` debe tener default `None` para que sea kwarg opcional."
    )


def test_knobs_use_env_int_with_clamps():
    src = _read(_BG_EXECUTOR)
    assert "_env_int(\"MEALFIT_BG_TASK_MAX_WORKERS\"" in src, (
        "Knob `MEALFIT_BG_TASK_MAX_WORKERS` debe resolverse via `_env_int` "
        "para auto-registrarse en `_KNOBS_REGISTRY`."
    )
    assert "_env_int(\"MEALFIT_BG_TASK_TIMEOUT_S\"" in src, (
        "Knob `MEALFIT_BG_TASK_TIMEOUT_S` debe resolverse via `_env_int`."
    )


def test_pool_executor_bounded():
    """No queremos `ThreadPoolExecutor()` sin `max_workers` (default es
    `min(32, cpu+4)` — irrelevante; queremos cap explícito)."""
    src = _read(_BG_EXECUTOR)
    pat = re.compile(r"ThreadPoolExecutor\s*\(\s*max_workers\s*=", re.DOTALL)
    assert pat.search(src), (
        "ThreadPoolExecutor debe instanciarse con `max_workers=` explícito."
    )


# ---------------------------------------------------------------------------
# 2. Migration enforcement en chat.py (los 2 callsites legacy migrados)
# ---------------------------------------------------------------------------
def test_chat_router_imports_submit_bg_task():
    src = _read(_CHAT_ROUTER)
    assert re.search(r"from\s+bg_executor\s+import\s+submit_bg_task", src), (
        "`routers/chat.py` debe importar `submit_bg_task` desde `bg_executor`."
    )


def test_chat_router_no_legacy_threading_thread_calls():
    """Los 2 callsites flagged en P1-BG-THREAD-TIMEOUT no deben re-aparecer.
    `threading.Thread(target=` es el patrón sintáctico canónico de daemon
    fire-and-forget que abandonamos.
    """
    src = _read(_CHAT_ROUTER)
    # Encontrar TODAS las ocurrencias en código (no en comentarios) y
    # rechazar si encuentra. Permite el patrón en docstrings/historical
    # comments porque marker `[P1-BG-THREAD-TIMEOUT` documenta la migración.
    code_lines = []
    in_block_comment = False
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # comentario
        if '"""' in stripped:
            in_block_comment = not in_block_comment
            continue
        if in_block_comment:
            continue
        code_lines.append(line)
    code_only = "\n".join(code_lines)
    pat = re.compile(r"threading\.Thread\s*\(\s*target\s*=")
    matches = pat.findall(code_only)
    assert not matches, (
        f"P1-BG-THREAD-TIMEOUT regresión: `routers/chat.py` aún contiene "
        f"{len(matches)} `threading.Thread(target=...)` callsite(s) en "
        f"código activo. Reemplazar por `submit_bg_task(..., task_name=...)`."
    )


def test_chat_router_uses_submit_bg_task_for_known_tasks():
    """Los 2 task_names canónicos deben estar presentes — anchor para que
    un refactor que renombre el callsite también renombre la alerta key."""
    src = _read(_CHAT_ROUTER)
    assert "chat_title_generation" in src, (
        "Esperaba `task_name=\"chat_title_generation\"` para el thread del "
        "título. Si renombraste, actualiza este test y la docs."
    )
    assert "chat_sse_bg_tasks" in src, (
        "Esperaba `task_name=\"chat_sse_bg_tasks\"` para el thread de "
        "summarize + facts del SSE."
    )


# ---------------------------------------------------------------------------
# 3. Funcional: timeout dispara la alerta; éxito NO
# ---------------------------------------------------------------------------
def test_clamp_respects_floor():
    """`_clamp(value, lo, hi)` debe respetar el floor (timeout_s=5 → 10)."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT / "backend"))
    import bg_executor

    assert bg_executor._clamp(5, 10, 1800) == 10
    assert bg_executor._clamp(50, 10, 1800) == 50
    assert bg_executor._clamp(99999, 10, 1800) == 1800


def test_timeout_alert_fires_with_short_internal_timeout(monkeypatch):
    """Test funcional usando el `_clamp` interno para forzar timeout=10s
    (mínimo) — pero como 10s es demasiado para CI, parcheamos `_clamp`
    para permitir timeout=0.3s solo en este test."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT / "backend"))
    import bg_executor

    captured: list = []

    def fake_persist(task_name: str, timeout_s: int) -> None:
        captured.append((task_name, timeout_s))

    monkeypatch.setattr(bg_executor, "_persist_bg_task_timeout_alert", fake_persist)
    # Bypass del clamp solo para este test — `_clamp` se valida en otro test.
    monkeypatch.setattr(bg_executor, "_clamp", lambda v, lo, hi: v)

    def slow_task():
        time.sleep(1.5)

    bg_executor.submit_bg_task(slow_task, task_name="test_slow_timeout", timeout_s=0.3)
    # Esperar a que el watcher dispare timeout + alerta.
    time.sleep(1.0)
    assert any(name == "test_slow_timeout" for name, _ in captured), (
        f"Esperaba alerta `test_slow_timeout`; captured={captured}"
    )


def test_fast_task_does_not_trigger_alert(monkeypatch):
    """Task que termina dentro del timeout NO debe emitir alerta."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT / "backend"))
    import bg_executor

    captured: list = []

    def fake_persist(task_name: str, timeout_s: int) -> None:
        captured.append((task_name, timeout_s))

    monkeypatch.setattr(bg_executor, "_persist_bg_task_timeout_alert", fake_persist)
    monkeypatch.setattr(bg_executor, "_clamp", lambda v, lo, hi: v)

    def fast_task():
        time.sleep(0.05)

    bg_executor.submit_bg_task(fast_task, task_name="test_fast", timeout_s=2.0)
    time.sleep(0.5)
    assert not any(name == "test_fast" for name, _ in captured), (
        f"Task rápido NO debe disparar alerta; captured={captured}"
    )


# ---------------------------------------------------------------------------
# 4. Cross-link guard (P2-HIST-AUDIT-14): el slug del marker tiene archivo.
#    El marker actual (P1-SENTRY-PII-SCRUBBING-BACKEND) anchorea al test
#    correspondiente; este test es para P1-BG-THREAD-TIMEOUT, no satisface
#    el cross-link del marker. Solo dejamos anchor textual.
# ---------------------------------------------------------------------------
def test_anchor_present_in_test_file():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P1-BG-THREAD-TIMEOUT" in src
