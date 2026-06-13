"""[P1-DIARY-UPLOAD-GUEST-IDOR · 2026-05-30] Regresión parser-based del cierre
del IDOR cross-user en la rama guest de `POST /api/diary/upload`, más el
hermano P2-DIARY-ASYNC-SYNC-DB (offload de DB síncrona del event loop en el
mismo handler `async def`).

Contexto del bug (audit del agente 2026-05-30):
    P1-PROD-AUDIT-3 endureció SOLO la rama autenticada del guard de ownership
    (eliminó el escape `and user_id != session_id`), pero la rama guest
    (`actual_user_id = session_id`) seguía confiando en un `session_id`
    arbitrario SIN lookup de ownership. Un atacante NO autenticado que enviara
    `user_id="guest"` + `session_id=<UUID real de la víctima>` lograba que el
    background task escribiera en el `visual_diary` de la víctima (la tabla NO
    tiene FK a auth.users; SERVICE_ROLE bypassea RLS) → contaminación
    cross-user del RAG/memoria (prompt-injection persistente).

Estos tests parsean `routers/diary.py` y fallan si el guard o el gate de la
escritura desaparecen (un renombre/refactor rompe el test ANTES que producción).
Son parser-based (sin DB) para correr en el venv DB-less del repo.
"""
from __future__ import annotations

import re
from pathlib import Path

_DIARY_PY = Path(__file__).resolve().parent.parent / "routers" / "diary.py"


def _src() -> str:
    return _DIARY_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# P1: IDOR guest-path guard
# ---------------------------------------------------------------------------
def test_guest_path_idor_guard_present():
    """La rama guest debe anular `actual_user_id` cuando no hay token
    verificado que coincida con la identidad reclamada en session_id."""
    src = _src()
    assert "P1-DIARY-UPLOAD-GUEST-IDOR" in src, (
        "Falta el tooltip-anchor P1-DIARY-UPLOAD-GUEST-IDOR — ¿se removió el guard?"
    )
    # El guard concreto: gatea sobre user_id == "guest" y exige token coincidente.
    assert re.search(
        r'if\s+user_id\s*==\s*"guest"\s+and\s+actual_user_id\s*:',
        src,
    ), "Falta el branch `if user_id == \"guest\" and actual_user_id:`"
    assert re.search(
        r'if\s+not\s+verified_user_id\s+or\s+verified_user_id\s*!=\s*actual_user_id\s*:',
        src,
    ), "Falta el check de token coincidente en la rama guest"
    # Debe anular actual_user_id (no persistir bajo identidad ajena).
    assert "actual_user_id = None" in src, (
        "El guard debe setear actual_user_id = None para omitir la persistencia"
    )


def test_background_visual_entry_write_is_gated_on_actual_user_id():
    """El background write a visual_diary debe estar gateado en
    `if actual_user_id:` — si la rama guest lo anuló, NO se persiste."""
    src = _src()
    # Buscar el bloque que gatea el add_task de _save_visual_entry_background.
    m = re.search(
        r'if\s+actual_user_id\s*:\s*\n\s*background_tasks\.add_task\(\s*\n?\s*_save_visual_entry_background',
        src,
    )
    assert m is not None, (
        "El `background_tasks.add_task(_save_visual_entry_background, ...)` debe "
        "estar dentro de un `if actual_user_id:` para evitar escribir bajo "
        "user_id None o una identidad ajena."
    )


# ---------------------------------------------------------------------------
# P2: event-loop blocking — sync DB offloaded to asyncio.to_thread
# ---------------------------------------------------------------------------
def test_sync_db_calls_offloaded_to_thread():
    """Las 3 llamadas DB síncronas del handler async `api_diary_upload`
    deben ir vía `asyncio.to_thread` (no bloquear el event loop)."""
    src = _src()
    assert "P2-DIARY-ASYNC-SYNC-DB" in src, "Falta el tooltip-anchor P2-DIARY-ASYNC-SYNC-DB"
    assert "await asyncio.to_thread(get_user_profile, user_id)" in src, (
        "get_user_profile (chrono) debe ir por asyncio.to_thread"
    )
    assert 'await asyncio.to_thread(log_api_usage, actual_user_id, "llm_vision")' in src, (
        "log_api_usage (llm_vision) debe ir por asyncio.to_thread"
    )
    assert re.search(
        r'await\s+asyncio\.to_thread\(\s*\n\s*execute_sql_write\s*,',
        src,
    ), "El INSERT de pipeline_metrics (chrono) debe ir por asyncio.to_thread"


def test_handler_still_async_def():
    """Sanity: `api_diary_upload` sigue siendo `async def` (el offload solo
    tiene sentido en un handler async)."""
    src = _src()
    assert re.search(r'async\s+def\s+api_diary_upload\s*\(', src), (
        "api_diary_upload debe seguir siendo async def"
    )
