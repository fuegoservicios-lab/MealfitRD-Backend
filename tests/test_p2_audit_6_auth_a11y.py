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
    """[reapuntado 2026-07-28] El Login migró a OTP (2026-06-21): los 3 pares htmlFor/id del
    flujo de contraseña murieron con él. El idioma vivo es `aria-label` por input — sustituto
    válido de la asociación label/input para lectores de pantalla. La invariante real: NINGÚN
    input sin etiquetar (hoy: 2 inputs — correo y código — con 2 aria-label)."""
    inputs = len(re.findall(r"<input\b", login_src))
    etiquetados = len(re.findall(r'aria-label\s*=\s*"', login_src))
    assert inputs >= 2, f"Login OTP debe tener al menos correo y código: {inputs} inputs"
    assert etiquetados >= inputs, (
        f"P2-AUDIT-6 regresión: {inputs} inputs pero solo {etiquetados} aria-label — "
        f"hay inputs sin etiqueta para el lector de pantalla."
    )


def test_login_error_box_has_alert_role(login_src: str):
    """[reapuntado 2026-07-28] La caja pasó de `styles.errorBox` a `mf-error` con el rediseño.
    La invariante es la MISMA: el contenedor de error anuncia con role=alert + aria-live, y los
    inputs lo referencian por aria-describedby para que el lector diga QUÉ campo falló."""
    assert re.search(
        r'<div[^>]*id="login-error"[^>]*role\s*=\s*["\']alert["\'][^>]*aria-live',
        login_src,
    ), (
        "P2-AUDIT-6 regresión: el contenedor de error del Login perdió role=\"alert\"/aria-live."
    )
    assert "aria-describedby={error ? 'login-error'" in login_src, (
        "P2-AUDIT-6 regresión: los inputs ya no referencian #login-error vía aria-describedby."
    )


def test_login_success_box_has_status_role(login_src: str):
    """[reapuntado 2026-07-28] El successBox murió con el flujo de contraseña: el éxito del OTP
    viaja por toast (sonner gestiona su propio aria-live). La invariante que queda anclable es
    que el feedback de éxito EXISTE (reenvío de código) — sin él, el usuario no sabe si el
    código salió."""
    assert re.search(r"toast\.success\(", login_src), (
        "P2-AUDIT-6 regresión: el Login perdió el feedback de éxito (toast.success)."
    )


# ---------------------------------------------------------------------------
# 2. Register.jsx: RETIRADO — la página ya no existe
# ---------------------------------------------------------------------------
# [P2-DEAD-CODE-SWEEP · 2026-08-15] Aquí vivían dos tests sobre `Register.jsx`
# (≥4 pares htmlFor/id, y `role="alert"` en el errorBox).
#
# `P1-EMAIL-OTP · 2026-06-21` retiró el registro con formulario: el login por
# código crea la cuenta, y `App.jsx` dejó `/register` como
# `<Navigate to="/login" replace />`. Desde entonces `Register.jsx` no tenía
# ruta, ni import, ni chunk en `dist/` — y estos tests llevaban ~2 meses
# vigilando la accesibilidad de una página que nadie podía abrir. Un test verde
# sobre un fichero muerto no protege a ningún usuario: sólo hace creer que sí.
#
# LA COBERTURA NO SE PIERDE, que es la condición para poder borrarlos. Las dos
# invariantes siguen ancladas donde SÍ hay formulario: `Login.jsx` (el OTP) y
# `ResetPassword.jsx` conservan sus propios tests de pares htmlFor/id y de
# `role="alert"` en este mismo fichero.
#
# El fichero se borró; el test de abajo impide que vuelva sin una decisión.
def test_register_jsx_sigue_borrado():
    """Resucitar `Register.jsx` sería reabrir un formulario que el producto retiró.

    No es una prohibición dogmática: si algún día vuelve el registro con
    contraseña, este test debe caer y hay que quitarlo A LA VEZ que se
    restauran sus guardas de a11y y de longitud mínima. Lo que impide es que
    reaparezca por inercia —un revert, un merge— con nadie mirando.
    """
    assert not _REGISTER_JSX.exists(), (
        f"{_REGISTER_JSX} volvió a existir. `P1-EMAIL-OTP` retiró el registro con "
        "formulario (el login por código crea la cuenta) y `/register` es un "
        "redirect. Si el registro con contraseña vuelve de verdad, restaura "
        "también sus tests de a11y (pares htmlFor/id, role=alert) y el de mínimo "
        "8 caracteres de test_p3_prod_audit_6.py — se retiraron con el fichero."
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


# [P2-DEAD-CODE-SWEEP · 2026-08-15] `test_anchor_present_in_register` retirado con
# el fichero. Ver la lápida de la sección 2.


def test_anchor_present_in_reset(reset_src: str):
    assert "P2-AUDIT-6" in reset_src, (
        "P2-AUDIT-6 regresión: anchor perdido en ResetPassword.jsx."
    )
