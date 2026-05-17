"""[P3-RESET-SINGLE-TXN + P3-RESET-BUTTON-LOADING-STATE · 2026-05-16] Fix
del delay observado por el usuario al clickear "Sí, empezar desde cero" en
Settings.

Síntoma reportado por usuario:
> "cuando le doy al boton dura unos segundos para reaccionar, por que ese delay?"

Causa raíz (2 problemas independientes):

  (1) **Backend**: `reset_user_account_preferences` (db_profiles.py:737)
      hacía 7 llamadas secuenciales a `execute_sql_write`, cada una con
      su propio `with connection_pool.connection() as conn:`. En free
      tier saturado (pool ~25, visible en logs `couldn't get a connection
      after 8.00 sec`), cada acquire podía tomar hasta 8s → wall-clock
      0.4s a 56s para el reset completo.

  (2) **Frontend**: el botón NO tenía loading state. Click → toast.loading
      arriba pero el botón mismo seguía idéntico → user clickeaba 2-3
      veces creyendo que falló.

Fix:

  (1) Backend: `with connection_pool.connection()` + `with conn.transaction()`
      UNA sola vez; los 7 statements (6 DELETEs + 1 UPDATE) ejecutan
      dentro de esa transacción. 1 acquire del pool + 7 statements + 1
      commit. Atomicidad bonus: si cualquier statement falla, ROLLBACK
      preserva la cuenta consistente.

  (2) Frontend: state `isResetting` + botón disabled + text "Borrando…" +
      color desaturado #FCA5A5 + cursor:wait. Feedback INMEDIATO sin
      depender de animations globales (mfSpin keyframe vive en Plan.jsx,
      no aplica cross-component).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_PROFILES = (_BACKEND_ROOT / "db_profiles.py").read_text(encoding="utf-8")
_FRONTEND_ROOT = _BACKEND_ROOT.parent / "frontend"
_SETTINGS = (_FRONTEND_ROOT / "src" / "pages" / "Settings.jsx").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fix #1: Backend single-transaction
# ---------------------------------------------------------------------------


def test_reset_uses_single_connection():
    """`reset_user_account_preferences` DEBE usar UNA sola
    `with connection_pool.connection()`, NO múltiples `execute_sql_write`
    secuenciales.

    Strippea comentarios + docstring para que el test mida CODE solamente
    (el docstring del fix menciona `execute_sql_write` históricamente)."""
    idx = _DB_PROFILES.find("def reset_user_account_preferences")
    assert idx > 0
    next_def = _DB_PROFILES.find("\ndef ", idx + 10)
    body = _DB_PROFILES[idx:next_def if next_def > 0 else idx + 5000]

    assert "P3-RESET-SINGLE-TXN" in body, (
        "Marker P3-RESET-SINGLE-TXN ausente — un refactor podría reintroducir "
        "el patrón viejo de 7 execute_sql_write secuenciales."
    )

    # Strippear docstring multiline y comentarios `#` para chequeos de
    # patrones de CODE.
    code_only = _strip_docstring_and_comments(body)

    # Debe haber EXACTAMENTE UN `with connection_pool.connection()` activo:
    pat = re.compile(r"^\s*with\s+connection_pool\.connection\(\)", re.MULTILINE)
    matches = pat.findall(code_only)
    assert len(matches) == 1, (
        f"`reset_user_account_preferences` tiene {len(matches)} sentencias "
        f"`with connection_pool.connection()` activas, esperaba 1. El patrón "
        f"viejo de 7 calls secuenciales reintroduciría el delay del usuario."
    )
    # NO debe haber llamadas activas a execute_sql_write:
    bad_pat = re.compile(r"^\s*execute_sql_write\s*\(", re.MULTILINE)
    bad_matches = bad_pat.findall(code_only)
    assert not bad_matches, (
        "Detectado llamado activo a `execute_sql_write(...)` — el refactor "
        "debe usar `cursor.execute(...)` dentro de UN connection_pool "
        "context, no execute_sql_write (que abre su propia conexión por call)."
    )


def _strip_docstring_and_comments(src: str) -> str:
    """Helper test-local: remueve docstrings triple-quoted y comentarios
    line-style `#...` para que los pattern checks solo vean CODE
    (no menciones históricas en docstrings)."""
    # Strip triple-quoted strings (docstrings):
    no_doc = re.sub(r'"' * 3 + r'[\s\S]*?' + r'"' * 3, '', src)
    # Strip line comments:
    no_comments = re.sub(r'#[^\n]*', '', no_doc)
    return no_comments


def test_reset_uses_transaction_for_atomicity():
    """`with conn.transaction()` debe envolver los 7 statements. Sin esto,
    en autocommit mode cada statement se commitea individualmente — si el
    statement #5 falla, los #1-4 ya están commiteados → cuenta en estado
    parcial."""
    idx = _DB_PROFILES.find("def reset_user_account_preferences")
    next_def = _DB_PROFILES.find("\ndef ", idx + 10)
    body = _DB_PROFILES[idx:next_def if next_def > 0 else idx + 5000]

    assert "conn.transaction()" in body, (
        "Sin `with conn.transaction():`, el rollback automático en error "
        "no aplica. Un fallo parcial dejaría la cuenta inconsistente."
    )


def test_reset_still_covers_7_tables():
    """Anti-regresión: el refactor debe seguir borrando las 7 cosas
    originales (meal_likes, meal_rejections, user_inventory, user_facts,
    ingredient_frequencies, meal_plans, user_profiles.health_profile)."""
    idx = _DB_PROFILES.find("def reset_user_account_preferences")
    next_def = _DB_PROFILES.find("\ndef ", idx + 10)
    body = _DB_PROFILES[idx:next_def if next_def > 0 else idx + 5000]

    required_tables = [
        ("meal_likes", "DELETE FROM meal_likes"),
        ("meal_rejections", "DELETE FROM meal_rejections"),
        ("user_inventory", "DELETE FROM user_inventory"),
        ("user_facts", "DELETE FROM user_facts"),
        ("ingredient_frequencies", "DELETE FROM ingredient_frequencies"),
        ("meal_plans", "DELETE FROM meal_plans"),
        ("user_profiles", "UPDATE user_profiles"),
    ]
    for label, statement in required_tables:
        assert statement in body, (
            f"Operación sobre `{label}` ausente del refactor — perdió "
            f"cobertura. Esperaba: `{statement}`."
        )


# ---------------------------------------------------------------------------
# Fix #2: Frontend loading state
# ---------------------------------------------------------------------------


def test_isResetting_state_declared():
    """`useState` para `isResetting` debe estar declarado."""
    assert "const [isResetting, setIsResetting] = useState(false)" in _SETTINGS, (
        "Estado `isResetting` no declarado. Sin él, el botón no tiene "
        "loading visual y user clickea múltiples veces creyendo que falló."
    )
    assert "P3-RESET-BUTTON-LOADING-STATE" in _SETTINGS, (
        "Marker P3-RESET-BUTTON-LOADING-STATE ausente — un refactor podría "
        "remover el loading state sin signal."
    )


def test_button_uses_isResetting_for_visual_feedback():
    """El botón debe leer `isResetting` para: (a) disabled, (b) cambiar
    el texto a 'Borrando…', (c) color desaturado."""
    # Localizar el bloque del botón "Sí, empezar desde cero"
    # Anchor único del fix nuevo (la JSX expression con `isResetting`).
    # Evita matches en comentarios cercanos que mencionan el botón.
    idx = _SETTINGS.find("isResetting ? 'Borrando")
    assert idx > 0, "Expresión JSX `isResetting ? 'Borrando` no encontrada."
    # Slice de ~2500 chars ANTES (donde está la definición del button)
    block = _SETTINGS[max(0, idx - 12000):idx + 200]

    assert "disabled={isResetting}" in block, (
        "Botón no tiene `disabled={isResetting}` — user podría clickear "
        "múltiples veces durante el await."
    )
    assert "'Borrando…'" in block or "'Borrando...'" in block or "Borrando…" in block, (
        "Botón no cambia texto a 'Borrando…' cuando isResetting=true. "
        "Feedback visual perdido."
    )
    # Color cambia a desaturado (#FCA5A5 vs #EF4444):
    assert "#FCA5A5" in block, (
        "Botón no usa color desaturado (#FCA5A5) cuando isResetting=true. "
        "Sin esto, sigue viéndose 100% activo."
    )
    # cursor:wait:
    assert "'wait'" in block, (
        "Botón no setea `cursor: 'wait'` cuando isResetting=true."
    )


def test_setIsResetting_true_at_start_of_handler():
    """El handler onClick debe llamar `setIsResetting(true)` ANTES del
    await del backend, para feedback inmediato."""
    # Anchor único del fix nuevo (la JSX expression con `isResetting`).
    # Evita matches en comentarios cercanos que mencionan el botón.
    idx = _SETTINGS.find("isResetting ? 'Borrando")
    block = _SETTINGS[max(0, idx - 12000):idx + 200]

    # El setIsResetting(true) debe estar ANTES del await fetchWithAuth
    set_true_idx = block.find("setIsResetting(true)")
    await_idx = block.find("await fetchWithAuth")
    assert set_true_idx > 0, "setIsResetting(true) no encontrado."
    assert await_idx > 0, "await fetchWithAuth no encontrado."
    assert set_true_idx < await_idx, (
        "setIsResetting(true) debe estar ANTES del `await fetchWithAuth` "
        "para feedback INMEDIATO. Si está después, el botón solo cambia "
        "tras 5-8s de espera — sin diferencia para el user."
    )


def test_isResetting_re_enabled_on_error():
    """En error path, `setIsResetting(false)` debe re-habilitar el botón.
    En happy path NO es necesario (navigate desmonta el componente)."""
    # Anchor único del fix nuevo (la JSX expression con `isResetting`).
    # Evita matches en comentarios cercanos que mencionan el botón.
    idx = _SETTINGS.find("isResetting ? 'Borrando")
    block = _SETTINGS[max(0, idx - 12000):idx + 200]

    # Debe haber setIsResetting(false) dentro de un catch:
    catch_idx = block.find("catch (error)")
    if catch_idx < 0:
        catch_idx = block.find("catch(error)")
    assert catch_idx > 0, "Catch block no encontrado."
    catch_block = block[catch_idx:catch_idx + 1000]
    assert "setIsResetting(false)" in catch_block, (
        "En error path, `setIsResetting(false)` NO se llama — el botón "
        "queda permanentemente disabled tras un error transient. User "
        "no puede retry sin recargar."
    )
