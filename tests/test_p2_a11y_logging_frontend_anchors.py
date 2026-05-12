"""[P2-A11Y-LOGGING · 2026-05-13] Anchors a11y en componentes frontend
identificados con gaps reales (parte del bundle P2 junto con silent
degradation logging).

Contexto:
    Audit production-readiness 2026-05-12 detectó 7 a11y gaps reales
    (no inventados) en componentes con icon-only `<button>` y toggle
    sin estado anunciable:

      1. `IOSInstallPrompt.jsx` — dismiss `<X>` button sin aria-label.
      2. `Header.jsx` — mobile toggle (hamburger ↔ X) sin aria-label
         ni aria-expanded.
      3. `ChatWidget.jsx` — 4 icon-only buttons:
            - Back from history (ArrowLeft)
            - Open history (History icon)
            - New chat (Plus)
            - FAB toggle (MessageSquare ↔ ×)
      4. `Pricing.jsx` — toggle Mensual/Anual sin aria-pressed (estado
         activo visible vía className pero invisible a lectores de
         pantalla).

    Resto del codebase (Modal, BottomTabBar, Footer, InteractiveQuestions,
    HowItWorks, Hero) ya tiene ARIA correcta — el agente de exploración
    confirmó esto en el audit. NO se inventan gaps.

Fix:
    Cada button icon-only recibe `aria-label="<verbo + objeto>"`.
    Los toggles binarios (menu open/close, FAB) reciben `aria-expanded`.
    El toggle de billing recibe `aria-pressed` por opción + `role="group"`
    en el contenedor con `aria-label="Periodo de facturación"`.

Lo que este test enforza:
  A) Anchor `[P2-A11Y-LOGGING` presente en los 4 archivos arreglados
     (mínimo 1 ocurrencia cada uno — la del comment inline).
  B) Conteo mínimo de `aria-label=` y `aria-expanded=` / `aria-pressed=`
     por archivo (cubre los gaps específicos identificados, no exhaustive).
  C) Sentinel anti-regresión: el `<button>` específico identificado
     (dismissPrompt en IOSInstallPrompt, mobileToggle en Header,
     setIsOpen FAB en ChatWidget) NO puede aparecer sin `aria-label`
     en su atributo list — si alguien remueve el aria-label en un
     refactor, el test falla con copy explicativo.

Tooltip-anchor: P2-A11Y-LOGGING / P2-A11Y-FRONTEND.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"

_IOS_PROMPT = _FRONTEND_SRC / "components" / "IOSInstallPrompt.jsx"
_HEADER = _FRONTEND_SRC / "components" / "layout" / "Header.jsx"
_CHAT_WIDGET = _FRONTEND_SRC / "components" / "dashboard" / "ChatWidget.jsx"
_PRICING = _FRONTEND_SRC / "components" / "home" / "Pricing.jsx"

# (archivo, min_anchors, min_aria_attrs)
# min_aria_attrs cuenta: aria-label= + aria-expanded= + aria-pressed=
_PATCHED: tuple[tuple[Path, int, int], ...] = (
    (_IOS_PROMPT, 1, 1),    # 1 dismiss button
    (_HEADER, 1, 2),        # mobile toggle: aria-label + aria-expanded
    (_CHAT_WIDGET, 3, 5),   # 3 comment blocks (back+history/plus+FAB), 5 aria attrs (4 aria-label + 1 aria-expanded)
    (_PRICING, 1, 3),       # 2 aria-pressed + 1 aria-label en role="group"
)


def _find_button_block_by_marker(src: str, marker: str, window_lines: int = 18) -> str:
    """Localiza un `<button` cerca de una línea que contiene `marker`
    (busca primero hacia atrás N líneas, luego hacia adelante N líneas).
    Retorna las `window_lines` líneas desde el `<button`. Robusto frente
    a `=>` arrow functions dentro del JSX (que rompen el regex de
    `<button ... >`).

    Buscar BACKWARDS primero: el marker suele ser un atributo (className,
    onClick, comment) DENTRO del tag — el `<button` está N líneas arriba.
    """
    lines = src.splitlines()
    marker_indices = [i for i, ln in enumerate(lines) if marker in ln]
    assert marker_indices, f"marker {marker!r} no encontrado en source"
    for mi in marker_indices:
        # Backwards hasta window_lines líneas.
        for j in range(mi, max(-1, mi - window_lines), -1):
            if "<button" in lines[j]:
                return "\n".join(lines[j : j + window_lines])
        # Forwards si no se halló atrás.
        for j in range(mi, min(len(lines), mi + window_lines)):
            if "<button" in lines[j]:
                return "\n".join(lines[j : j + window_lines])
    raise AssertionError(
        f"`<button` no encontrado dentro de ±{window_lines} líneas del "
        f"marker {marker!r}"
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# A) Anchor count.
@pytest.mark.parametrize("path,min_anchors,_aria", _PATCHED)
def test_a_anchor_present(path: Path, min_anchors: int, _aria: int):
    src = _read(path)
    count = src.count("[P2-A11Y-LOGGING")
    assert count >= min_anchors, (
        f"P2-A11Y-LOGGING: {path.name} tiene {count} anchors "
        f"`[P2-A11Y-LOGGING`, esperaba >= {min_anchors}. "
        f"Si removiste el comment inline en un refactor cosmético, "
        f"restaurar — el anchor es la traza del POR QUÉ del ARIA "
        f"a un futuro mantenedor."
    )


# B) Conteo mínimo de atributos ARIA por archivo.
@pytest.mark.parametrize("path,_min_anchors,min_aria", _PATCHED)
def test_b_aria_attrs_count(path: Path, _min_anchors: int, min_aria: int):
    src = _read(path)
    aria_count = (
        src.count("aria-label=")
        + src.count("aria-expanded=")
        + src.count("aria-pressed=")
    )
    assert aria_count >= min_aria, (
        f"P2-A11Y-LOGGING: {path.name} tiene {aria_count} atributos "
        f"`aria-label= | aria-expanded= | aria-pressed=`, esperaba "
        f">= {min_aria}. Verificar que no se removieron en refactor."
    )


# C) Sentinel anti-regresión: callsites específicos no pueden perder aria-label.
def test_c_ios_prompt_dismiss_button_has_aria_label():
    src = _read(_IOS_PROMPT)
    # El <button> de dismiss usa `onClick={dismissPrompt}`. Buscar el
    # bloque del botón y verificar que tiene aria-label.
    block_match = re.search(
        r"<button\s+[^>]*onClick=\{dismissPrompt\}[^>]*>",
        src,
        re.DOTALL,
    )
    assert block_match, (
        "P2-A11Y-LOGGING: el `<button onClick={dismissPrompt}>` en "
        "IOSInstallPrompt.jsx no fue localizado. Si renombraste el "
        "handler, ajustar este sentinel."
    )
    block = block_match.group(0)
    assert "aria-label=" in block, (
        "P2-A11Y-LOGGING: el dismiss `<button>` en IOSInstallPrompt.jsx "
        "perdió `aria-label`. Es icon-only (<X size=18/>) — sin label "
        "los lectores de pantalla narran 'botón' sin contexto. "
        "Restaurar `aria-label=\"Cerrar aviso de instalación\"` "
        "(o equivalente)."
    )


def test_d_header_mobile_toggle_has_aria_label_and_expanded():
    src = _read(_HEADER)
    # Localizar el `<button>` que sigue al marker `styles.mobileToggle`.
    block = _find_button_block_by_marker(src, "styles.mobileToggle", window_lines=12)
    assert "aria-label=" in block, (
        "P2-A11Y-LOGGING: el mobile toggle en Header.jsx perdió "
        "`aria-label`. Sin label, el hamburger ↔ X no se anuncia."
    )
    assert "aria-expanded=" in block, (
        "P2-A11Y-LOGGING: el mobile toggle en Header.jsx perdió "
        "`aria-expanded`. Lectores de pantalla necesitan saber si "
        "el menú está abierto o cerrado."
    )


def test_e_chatwidget_fab_has_aria_label_and_expanded():
    src = _read(_CHAT_WIDGET)
    # El FAB tiene comment `Fab Button` justo antes.
    block = _find_button_block_by_marker(src, "Fab Button", window_lines=18)
    assert "aria-label=" in block, (
        "P2-A11Y-LOGGING: el FAB en ChatWidget.jsx perdió `aria-label`."
    )
    assert "aria-expanded=" in block, (
        "P2-A11Y-LOGGING: el FAB en ChatWidget.jsx perdió `aria-expanded`."
    )


def test_f_pricing_billing_toggle_uses_aria_pressed():
    src = _read(_PRICING)
    # El toggle tiene className styles.billingToggle.
    block_match = re.search(
        r"<div\s+className=\{styles\.billingToggle\}[^>]*>([\s\S]*?)</div>",
        src,
    )
    assert block_match, (
        "P2-A11Y-LOGGING: el contenedor `<div className={styles."
        "billingToggle}>` en Pricing.jsx no fue localizado."
    )
    block = block_match.group(0)
    assert 'role="group"' in block, (
        "P2-A11Y-LOGGING: el toggle Mensual/Anual en Pricing.jsx "
        "perdió `role=\"group\"` en el contenedor. Sin role, los "
        "buttons hijos no se asocian semánticamente."
    )
    # 2 ocurrencias esperadas (1 por cada button hijo).
    assert block.count("aria-pressed=") >= 2, (
        "P2-A11Y-LOGGING: el toggle Mensual/Anual en Pricing.jsx "
        "perdió `aria-pressed=` en sus 2 buttons. Sin aria-pressed, "
        "el estado activo (visible vía className) es invisible a "
        "lectores de pantalla."
    )
