"""[P1-NEW-3 · 2026-05-10] Regression guard: la release de
`chunk_user_locks` en `_chunk_worker` se ejecuta en TODOS los exit
paths (success, exception, early return por plan fallido, early
return por plan no encontrado).

Bug temido (audit 2026-05-10 — descartado tras verificación):
    Si el `_chunk_worker` hace `return` antes del bloque `finally`
    que ejecuta el `DELETE FROM chunk_user_locks`, el usuario queda
    bloqueado de cualquier chunk hasta que el cron de housekeeping
    libere el lock por `CHUNK_LOCK_STALE_MINUTES`.

Verificación post-audit (cron_tasks.py snapshot 2026-05-10):
    1. El `try:` principal del body arranca en cron_tasks.py:17046.
    2. El `finally:` con `DELETE FROM chunk_user_locks` está en
       cron_tasks.py:~22389-22401.
    3. Los 2 early returns (cron_tasks.py:17056 plan-no-encontrado y
       cron_tasks.py:17062 plan-fallido) están DENTRO del try → Python
       garantiza que el finally corre antes del unwind del frame.
    4. El otro early return (~L17042 tras
       `_handle_heartbeat_start_failure`) está FUERA del try principal
       pero `_handle_heartbeat_start_failure` ya libera el lock (acción
       5 de su docstring: "DELETE del chunk_user_locks") +
       reservaciones + UPDATE pending → no necesita el finally.

Este test bloquea regresión del contrato:
    - El `try:` principal debe abrir antes del primer `return`
      condicional del worker (excepto el lock-acquire failure que
      cleanup self-managed).
    - El `finally:` correspondiente debe contener el DELETE.
    - `_handle_heartbeat_start_failure` debe seguir incluyendo el
      DELETE como parte de su cleanup explícito.

Si en el futuro alguien:
    - Mueve los early returns FUERA del try → este test falla.
    - Borra el `finally:` con el DELETE → este test falla.
    - Cambia `_handle_heartbeat_start_failure` para no liberar el
      lock → este test falla.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


def _read_source() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def _extract_chunk_worker_body(src: str) -> str:
    """Devuelve el cuerpo de `_chunk_worker` desde `def _chunk_worker(task):`
    hasta el `import concurrent.futures` que cierra el wrapper."""
    start = re.search(r"\n    def _chunk_worker\(task\):\n", src)
    assert start is not None, "No encuentro `def _chunk_worker(task):`"
    end = re.search(
        r"\n    import\s+concurrent\.futures\b",
        src[start.start():],
    )
    assert end is not None, (
        "No encuentro el cierre `import concurrent.futures` del wrapper de "
        "_chunk_worker. El layout cambió — re-verifica el contrato del lock."
    )
    return src[start.start():start.start() + end.start()]


# ---------------------------------------------------------------------------
# 1. El finally con DELETE existe en _chunk_worker
# ---------------------------------------------------------------------------
def test_chunk_worker_finally_releases_user_lock():
    """Existe un bloque `finally:` dentro de `_chunk_worker` que ejecuta
    `DELETE FROM chunk_user_locks WHERE locked_by_chunk_id = %s`. Este
    es el catch-all para success + exception + cualquier return dentro
    del try principal."""
    body = _extract_chunk_worker_body(_read_source())
    # Buscar patrón: finally:\n + ... + DELETE FROM chunk_user_locks
    pattern = re.compile(
        r"finally\s*:\s*\n"
        r"(?:.*\n){1,40}?"  # hasta 40 líneas de contexto
        r".*DELETE\s+FROM\s+chunk_user_locks\b",
        re.IGNORECASE,
    )
    assert pattern.search(body) is not None, (
        "El `finally:` con `DELETE FROM chunk_user_locks` desapareció del "
        "cuerpo de _chunk_worker. Sin él, cualquier return dentro del try "
        "principal deja el lock vivo hasta que el housekeeping cron lo "
        "libere por CHUNK_LOCK_STALE_MINUTES — bloqueando al usuario de "
        "todo chunk processing durante ese intervalo."
    )


# ---------------------------------------------------------------------------
# 2. Early returns de plan-no-encontrado y plan-fallido están dentro del try
# ---------------------------------------------------------------------------
def test_plan_check_early_returns_inside_main_try():
    """Los `return` tras `if not active_plan:` y
    `if active_plan.get('status') == 'failed':` están DENTRO del `try:`
    principal que tiene el `finally:` del lock release. Python garantiza
    que el finally corre antes del unwind del frame."""
    body = _extract_chunk_worker_body(_read_source())
    # Pattern: encontrar `try:` y luego verificar que las dos
    # condiciones tempranas con return ocurren después.
    # Localizar el try principal (heurística: el que precede al SELECT
    # de active_plan).
    main_try_match = re.search(
        r"\n        try:\s*\n"
        r"\s*# \[GAP 3 FIX:",
        body,
    )
    assert main_try_match is not None, (
        "No encuentro el `try:` principal del cuerpo (precedido por el "
        "comentario `# [GAP 3 FIX:`). El layout del worker cambió — "
        "re-verificar que los early returns siguen dentro del try del "
        "lock release."
    )

    # Después del try debe aparecer el patrón de plan-no-encontrado +
    # return Y el de plan-failed + return.
    body_after_try = body[main_try_match.start():]

    not_found_match = re.search(
        r"if\s+not\s+active_plan\s*:\s*\n"
        r"(?:.*\n){1,5}?"
        r"\s*return",
        body_after_try,
    )
    assert not_found_match is not None, (
        "El early return tras `if not active_plan:` no se encuentra DENTRO "
        "del try principal — un retorno fuera del try saltaría el finally "
        "del lock release."
    )

    failed_match = re.search(
        r"if\s+active_plan\.get\(['\"]status['\"]\)\s*==\s*['\"]failed['\"]\s*:\s*\n"
        r"(?:.*\n){1,5}?"
        r"\s*return",
        body_after_try,
    )
    assert failed_match is not None, (
        "El early return tras `if active_plan.get('status') == 'failed':` "
        "no se encuentra DENTRO del try principal — un retorno fuera del "
        "try saltaría el finally del lock release."
    )


# ---------------------------------------------------------------------------
# 3. _handle_heartbeat_start_failure libera el lock como parte de cleanup
# ---------------------------------------------------------------------------
def test_heartbeat_start_failure_releases_lock():
    """`_handle_heartbeat_start_failure` (cron_tasks.py:11995+) libera
    el lock de `chunk_user_locks` como parte de su cleanup. Esto cubre
    el early return ~L17042 que ocurre ANTES del try principal."""
    src = _read_source()
    # Localizar la función completa.
    fn_match = re.search(
        r"\ndef _handle_heartbeat_start_failure\(task_id,\s*user_id\)\s*->\s*None\s*:\s*\n",
        src,
    )
    assert fn_match is not None, (
        "Función `_handle_heartbeat_start_failure` desapareció. El early "
        "return tras `if not _heartbeat_thread.is_alive():` quedaría sin "
        "cleanup del lock."
    )
    # Body: hasta el siguiente `\ndef ` top-level.
    body_start = fn_match.start()
    next_fn = re.search(r"\ndef [A-Za-z_]", src[body_start + 1:])
    body_end = body_start + 1 + (next_fn.start() if next_fn else len(src))
    fn_body = src[body_start:body_end]

    assert re.search(
        r"DELETE\s+FROM\s+chunk_user_locks\b",
        fn_body,
        re.IGNORECASE,
    ) is not None, (
        "`_handle_heartbeat_start_failure` ya NO contiene "
        "`DELETE FROM chunk_user_locks`. Sin esa liberación, el early "
        "return tras fallo de heartbeat-start deja el lock vivo y el "
        "usuario queda bloqueado hasta el housekeeping cron."
    )
