"""[P2-AUDIT-6 · 2026-05-15] Test parser-based: las 3 páginas de auth
(`Login.jsx`, `Register.jsx`, `ResetPassword.jsx`) tienen accessibility
attributes que screen readers necesitan: `htmlFor`/`id` pairs en
inputs + `role="alert"`/`role="status"` con `aria-live` en error/success
containers.

Por qué este test:
    Pre-fix: cero `htmlFor` en los 3 archivos. Las `<label>` quedaban
    asociadas a inputs solo por proximidad DOM (no semánticamente). Screen
    readers (NVDA, VoiceOver, TalkBack) no anunciaban el label cuando el
    user enfocaba el input → onboarding inaccesible.

    Adicional: `errorBox` y `successBox` sin `role`/`aria-live` → screen
    readers no anuncian errores de validación al renderizarse. Crítico para
    el flujo de auth donde error feedback es central.

Fix esperado:
    - Cada `<input>` user-facing tiene `id="..."` matched con un
      `<label htmlFor="...">` que lo describe.
    - Error containers tienen `role="alert"` + `aria-live="assertive"`.
    - Success containers tienen `role="status"` + `aria-live="polite"`.
    - Iconos decorativos dentro de `inputIcon`/`errorBox`/`successBox`
      tienen `aria-hidden="true"`.

Drift detection:
    - Login.jsx: ≥3 pairs htmlFor/id (email login, password, email reset).
    - Register.jsx: ≥4 pairs (name, email, password, confirm).
    - ResetPassword.jsx: ≥2 pairs (new password, confirm).
    - Los 3 archivos: errorBox con role="alert"; success container con
      role="status" donde aplica.

Cross-link convention (P2-HIST-AUDIT-14): slug `p2_audit_6` matchea este
archivo. Bundle marker `P2-AUDIT-6` (último alfabético del bundle P2).

Tooltip-anchor: P2-AUDIT-6-START | gap audit 2026-05-15
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PAGES_DIR = _REPO_ROOT / "frontend" / "src" / "pages"
_LOGIN_JSX = _PAGES_DIR / "Login.jsx"
_REGISTER_JSX = _PAGES_DIR / "Register.jsx"
_RESET_JSX = _PAGES_DIR / "ResetPassword.jsx"


@pytest.fixture(scope="module")
def login_src() -> str:
    return _LOGIN_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def register_src() -> str:
    return _REGISTER_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def reset_src() -> str:
    return _RESET_JSX.read_text(encoding="utf-8")


def _count_htmlfor_id_pairs(src: str) -> int:
    """Cuenta `htmlFor="X"` que tenga un `id="X"` correspondiente en el
    mismo archivo. Patrón canónico de label↔input association."""
    htmlfor_re = re.compile(r'htmlFor\s*=\s*["\']([^"\']+)["\']')
    id_re = re.compile(r'\bid\s*=\s*["\']([^"\']+)["\']')
    htmlfor_ids = set(htmlfor_re.findall(src))
    page_ids = set(id_re.findall(src))
    return len(htmlfor_ids & page_ids)


# ---------------------------------------------------------------------------
# 1. Login.jsx: ≥3 pairs (email login, password, email reset)
# ---------------------------------------------------------------------------
def test_login_has_min_htmlfor_id_pairs(login_src: str):
    n = _count_htmlfor_id_pairs(login_src)
    assert n >= 3, (
        f"P2-AUDIT-6 regresión: Login.jsx tiene {n} pairs htmlFor/id (esperado ≥3 "
        f"— email-login, password, email-reset). Sin pairs, screen readers no "
        f"asocian label con input."
    )


def test_login_error_box_has_alert_role(login_src: str):
    assert re.search(
        r'errorBox\}?\s*role\s*=\s*["\']alert["\']',
        login_src,
    ) or re.search(
        r'role\s*=\s*["\']alert["\'][^>]*errorBox',
        login_src,
    ) or re.search(
        # className puede aparecer antes o después de role; aceptamos cualquier
        # orden siempre que el className includes errorBox y role="alert"
        # estén en el mismo elemento JSX.
        r'<div[^>]*className=\{styles\.errorBox\}[^>]*role\s*=\s*["\']alert["\']',
        login_src,
    ), (
        "P2-AUDIT-6 regresión: `<div className={styles.errorBox}>` en Login.jsx "
        "no tiene `role=\"alert\"`. Screen readers no anuncian errores de "
        "validación. Añadir `role=\"alert\" aria-live=\"assertive\"`."
    )


def test_login_success_box_has_status_role(login_src: str):
    assert re.search(
        r'<div[^>]*className=\{styles\.successBox\}[^>]*role\s*=\s*["\']status["\']',
        login_src,
    ), (
        "P2-AUDIT-6 regresión: `<div className={styles.successBox}>` en "
        "Login.jsx no tiene `role=\"status\"`. Añadir `role=\"status\" "
        "aria-live=\"polite\"` para que screen readers anuncien sin "
        "interrumpir."
    )


# ---------------------------------------------------------------------------
# 2. Register.jsx: ≥4 pairs (name, email, password, confirm)
# ---------------------------------------------------------------------------
def test_register_has_min_htmlfor_id_pairs(register_src: str):
    n = _count_htmlfor_id_pairs(register_src)
    assert n >= 4, (
        f"P2-AUDIT-6 regresión: Register.jsx tiene {n} pairs htmlFor/id "
        f"(esperado ≥4 — name, email, password, confirm)."
    )


def test_register_error_box_has_alert_role(register_src: str):
    assert re.search(
        r'<div[^>]*className=\{styles\.errorBox\}[^>]*role\s*=\s*["\']alert["\']',
        register_src,
    ), (
        "P2-AUDIT-6 regresión: errorBox en Register.jsx no tiene "
        "`role=\"alert\"`."
    )


# ---------------------------------------------------------------------------
# 3. ResetPassword.jsx: ≥2 pairs (new, confirm)
# ---------------------------------------------------------------------------
def test_reset_has_min_htmlfor_id_pairs(reset_src: str):
    n = _count_htmlfor_id_pairs(reset_src)
    assert n >= 2, (
        f"P2-AUDIT-6 regresión: ResetPassword.jsx tiene {n} pairs htmlFor/id "
        f"(esperado ≥2 — new password, confirm)."
    )


def test_reset_error_box_has_alert_role(reset_src: str):
    assert re.search(
        r'<div[^>]*className=\{styles\.errorBox\}[^>]*role\s*=\s*["\']alert["\']',
        reset_src,
    ), (
        "P2-AUDIT-6 regresión: errorBox en ResetPassword.jsx no tiene "
        "`role=\"alert\"`."
    )


def test_reset_success_box_has_status_role(reset_src: str):
    assert re.search(
        r'<div[^>]*className=\{styles\.successBox\}[^>]*role\s*=\s*["\']status["\']',
        reset_src,
    ), (
        "P2-AUDIT-6 regresión: successBox en ResetPassword.jsx no tiene "
        "`role=\"status\"`."
    )


# ---------------------------------------------------------------------------
# 4. Anchor textual P2-AUDIT-6 presente en los 3 archivos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path_label, getter", [
    ("Login.jsx", lambda fs: fs.read_text(encoding="utf-8")),
])
def test_anchor_present_in_login(path_label, getter):
    src = getter(_LOGIN_JSX)
    assert "P2-AUDIT-6" in src, (
        f"P2-AUDIT-6 regresión: anchor textual perdido en {path_label}."
    )


def test_anchor_present_in_register(register_src: str):
    assert "P2-AUDIT-6" in register_src, (
        "P2-AUDIT-6 regresión: anchor perdido en Register.jsx."
    )


def test_anchor_present_in_reset(reset_src: str):
    assert "P2-AUDIT-6" in reset_src, (
        "P2-AUDIT-6 regresión: anchor perdido en ResetPassword.jsx."
    )
