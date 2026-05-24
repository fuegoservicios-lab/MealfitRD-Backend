"""[P1-PROD-AUDIT-1 · 2026-05-23] `SET LOCAL lock_timeout` debe ser strict
por default (raise si falla), no best-effort silencioso.

Gap original (audit production-readiness 2026-05-23, B-P1-10):
    Pre-fix `db_core.py:368-373` capturaba CUALQUIER excepción del
    `SET LOCAL lock_timeout` y la logueaba a `logger.debug(...)` — la query
    luego corría SIN el timeout, degradando a "esperar indefinido por el
    row lock".

    Modo de fallo silencioso:
      1. Otro worker tiene la fila row-locked (UPDATE en transaction).
      2. Nuestro SET LOCAL falla por razón rara (privilegio raro de Supabase,
         pool conn en estado inválido).
      3. La query del caller corre sin timeout, espera indefinido.
      4. Usuario reporta "el guardado se cuelga 5min" sin pista en logs
         (el debug log no se imprime en producción default).

Fix:
    Knob `MEALFIT_LOCK_TIMEOUT_SET_STRICT=true` (default post-audit):
      - Strict: si SET LOCAL falla, raise RuntimeError → caller maneja.
        Falla LOUD > deadlock silencioso.
      - Best-effort (false, legacy): logger.warning + continúa. Back-compat
        para casos raros donde la query SÍ debe correr.

Cobertura:
    A) Anchor `P1-PROD-AUDIT-1-LOCK-TIMEOUT-STRICT` presente.
    B) Knob `MEALFIT_LOCK_TIMEOUT_SET_STRICT` referenciado en db_core.py.
    C) Default es strict (`true`).
    D) Path strict raise RuntimeError con info diagnostic.
    E) Path best-effort emite logger.warning (no .debug — debug se silencia
       en prod). REGRESIÓN: pre-fix usaba .debug → silent.

Tooltip-anchor: P1-PROD-AUDIT-1-LOCK-TIMEOUT-STRICT | audit 2026-05-23.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_CORE = _BACKEND_ROOT / "db_core.py"


def _read_db_core() -> str:
    return _DB_CORE.read_text(encoding="utf-8")


def test_anchor_present():
    src = _read_db_core()
    assert "P1-PROD-AUDIT-1-LOCK-TIMEOUT-STRICT" in src, (
        "Anchor `P1-PROD-AUDIT-1-LOCK-TIMEOUT-STRICT` ausente en db_core.py. "
        "Sin anchor, futuro mantenedor revierte al silent best-effort sin contexto."
    )


def test_knob_referenced():
    src = _read_db_core()
    assert "MEALFIT_LOCK_TIMEOUT_SET_STRICT" in src, (
        "Knob `MEALFIT_LOCK_TIMEOUT_SET_STRICT` no leído del env en "
        "db_core.py. Sin el knob, no hay forma de revertir a best-effort "
        "si emerge un caso raro donde SET LOCAL no funciona pero la query "
        "SÍ debe correr."
    )


def test_default_is_strict():
    """El default del knob debe ser strict (`true`). Si alguien cambia el
    default a false, el silent-deadlock-mode reaparece.
    """
    src = _read_db_core()
    # Buscar el patrón del default — debe ser "true" en string.
    # Pattern típico: `os.environ.get("MEALFIT_LOCK_TIMEOUT_SET_STRICT", "true")`
    m = re.search(
        r'MEALFIT_LOCK_TIMEOUT_SET_STRICT["\']?\s*,\s*["\'](\w+)["\']',
        src,
    )
    assert m is not None, (
        "No se encontró default literal del knob MEALFIT_LOCK_TIMEOUT_SET_STRICT. "
        "El default debe estar inline en `os.environ.get(\"...\", \"true\")` "
        "para auditabilidad."
    )
    default_val = m.group(1).lower()
    assert default_val in ("true", "1", "yes", "on"), (
        f"Default del knob es `{default_val}` — DEBE ser truthy. "
        f"Si lo flippeas a false, el silent-deadlock-mode reaparece."
    )


def test_strict_path_raises_runtime_error():
    """En strict mode, el catch del SET LOCAL falla debe terminar en
    `raise RuntimeError(...)`. Si alguien quita el raise, el strict mode
    se degrada silenciosamente a best-effort.
    """
    src = _read_db_core()
    # Usar rfind para localizar la ÚLTIMA mención del knob (la línea de
    # código, no el comment). Then validar que `raise RuntimeError` está
    # cerca después.
    idx = src.rfind("MEALFIT_LOCK_TIMEOUT_SET_STRICT")
    assert idx != -1, "Knob no encontrado — cubierto por test_knob_referenced"
    # Ventana hacia ADELANTE — el raise viene después del check del knob.
    window = src[idx : idx + 2000]
    assert "raise RuntimeError" in window, (
        "Strict path NO termina en `raise RuntimeError(...)`. "
        "Sin el raise, strict mode se degrada a logger.error + continuar "
        "→ mismo silent-deadlock-mode pre-fix."
    )


def test_legacy_best_effort_uses_warning_not_debug():
    """Pre-fix usaba `logger.debug(...)` — silenciado en producción default.
    Post-fix DEBE usar `logger.warning(...)` para que el operador vea la
    señal incluso cuando esté en best-effort mode.
    """
    src = _read_db_core()
    idx = src.find("MEALFIT_LOCK_TIMEOUT_SET_STRICT")
    window = src[max(0, idx - 200) : idx + 2000]
    # Validar que hay un logger.warning en el bloque (best-effort path).
    assert "logger.warning" in window, (
        "Best-effort path NO usa `logger.warning(...)`. Si quedó como "
        "`logger.debug(...)`, el silent mode persiste — operador no ve "
        "la señal en producción default. Cambiar a logger.warning."
    )


def test_no_silent_logger_debug_for_set_local_fail():
    """Defensive: la línea `logger.debug(...)` con "SET LOCAL" del pre-fix
    NO debe seguir presente (regresión).
    """
    src = _read_db_core()
    # Buscar el patrón pre-fix exacto.
    legacy_pattern = re.compile(
        r'logger\.debug\(.*SET LOCAL lock_timeout falló', re.DOTALL
    )
    assert not legacy_pattern.search(src), (
        "REGRESIÓN: db_core.py tiene `logger.debug(...SET LOCAL lock_timeout "
        "falló...)` del pre-fix. Esa línea debe ser logger.error/warning + "
        "raise — silent debug log NO es defensa contra deadlock silencioso."
    )
