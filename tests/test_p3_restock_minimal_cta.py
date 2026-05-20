"""[P3-RESTOCK-MINIMAL-CTA · 2026-05-20] Rediseño visual del botón
"Ya compré todo" + modal de confirmación. Decisión de producto post-
P3-RESTOCK-NO-BAR del mismo día: "mejora el diseño del botón y su
interfaz cuando se entra, quiero algo minimalista pero más original,
y que vaya de acorde con la paleta de colores necesaria de la web".

Direction elegida (vía AskUserQuestion): **Outline + accent dot**.

PRE-FIX

  Botón "Ya compré todo":
    - bg: gradient `linear-gradient(135deg, #10B981 0%, #059669 100%)` (verde saturado)
    - shadow: `0 10px 20px -5px rgba(16, 185, 129, 0.4)` (sombra colorida fuerte)
    - icon: CheckCircle (size 18)
    - className: `new-plan-btn` (estilos compartidos con CTAs principales)

  Modal "¿Confirmar Compra?":
    - Icon box 64×64 con gradient verde + sombra fuerte
    - Título con signos interrogativos
    - Botón principal: gradient verde + sombra verde + hover scale 1.02
    - Cancelar: botón con padding, no link

POST-FIX

  Botón "Ya compré todo" (className `restock-cta-minimal`):
    - bg: white (#FFFFFF)
    - border: 1px slate-200 (#E2E8F0), hover slate-900 (#0F172A)
    - dot emerald-500 pulsante (animation `restock-cta-pulse`)
    - text: slate-900 medium weight (no bold loud)
    - shadow: subtle gray, no colorida
    - microinteracción: pulse del dot, dot ring se acelera en hover,
      translateY(-1px) en hover, focus-visible ring indigo

  Modal "Confirmar compra":
    - Icon outline 56×56 con border slate-200, bg white, ShoppingCart slate-900
    - Status dot emerald-500 (14×14) en esquina inferior derecha del icon
    - Título "Confirmar compra" sin signos
    - Botón principal slate-900 con ArrowRight que se desliza translateX(4px)
      en hover (className `restock-modal-confirm`)
    - Cancelar como text-link slate-400 (className `restock-modal-cancel`)
    - prefers-reduced-motion respetado en todas las microinteracciones

Coherencia con paleta:
    - `--text-main: #0F172A` (slate-900) — color de marca para CTAs
    - `--text-muted: #64748B` (slate-500) — para descripción
    - `--text-light: #94A3B8` (slate-400) — para cancel link
    - Borders: slate-200 (#E2E8F0) — borde sutil del web design system
    - Accent: emerald-500 (#10B981) — solo en dot (semántica "success/ready")
    - Focus: indigo-600 (`--primary: #4F46E5`) — anillo de focus a11y

Tooltip-anchor: P3-RESTOCK-MINIMAL-CTA.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DASHBOARD_FP = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


@pytest.fixture(scope="module")
def src() -> str:
    return _DASHBOARD_FP.read_text(encoding="utf-8")


# ===========================================================================
# Botón "Ya compré todo" — outline + accent dot
# ===========================================================================

def test_button_uses_minimal_classname(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] el botón usa `className="restock-cta-minimal"`,
    NO `className="new-plan-btn"` legacy. Si vuelve a new-plan-btn, hereda
    estilos compartidos del CTA principal — pierde el rediseño minimalista."""
    # El botón en el header del Dashboard, dentro del `{hasPendingShoppingItems && (` block.
    handler_idx = src.find("setShowRestockModal(true)")
    assert handler_idx >= 0, "onClick={() => setShowRestockModal(true)} no encontrado"
    # Cuerpo del botón: ~500 chars circundantes.
    body = src[max(0, handler_idx - 200):handler_idx + 1500]
    assert 'className="restock-cta-minimal"' in body, (
        "[P3-RESTOCK-MINIMAL-CTA] el botón 'Ya compré todo' debe usar "
        "`className=\"restock-cta-minimal\"`. Si reverte a `new-plan-btn`, "
        "regresa al gradient verde saturado legacy."
    )


def test_button_bg_is_white(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] el botón ya no usa gradient verde — su
    background inline es `#FFFFFF`."""
    # El botón pre-fix tenía `background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)'`.
    # Verificamos que esa string específica NO está en la zona del botón.
    handler_idx = src.find("setShowRestockModal(true)")
    body = src[max(0, handler_idx - 200):handler_idx + 1500]
    assert "background: '#FFFFFF'" in body, (
        "[P3-RESTOCK-MINIMAL-CTA] el botón 'Ya compré todo' debe tener "
        "`background: '#FFFFFF'` (blanco minimal). Si tiene gradient verde, "
        "rompió la decisión de producto."
    )
    assert "linear-gradient(135deg, #10B981 0%, #059669 100%)" not in body, (
        "[P3-RESTOCK-MINIMAL-CTA] el gradient verde legacy NO debe estar "
        "en el botón 'Ya compré todo'. Si reapareció, regresa al diseño "
        "saturado."
    )


def test_button_has_pulsing_dot(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] el botón incluye un `<span className="restock-cta-dot">`
    que renderiza el dot emerald pulsante (único acento de color)."""
    handler_idx = src.find("setShowRestockModal(true)")
    body = src[max(0, handler_idx - 200):handler_idx + 1500]
    assert 'className="restock-cta-dot"' in body, (
        "[P3-RESTOCK-MINIMAL-CTA] el botón debe contener "
        "`<span className=\"restock-cta-dot\" />` — es el único acento de "
        "color del diseño minimal."
    )


def test_button_dot_pulse_css_defined(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] keyframes + .restock-cta-dot definidos en
    el bloque `<style>` del Dashboard."""
    assert ".restock-cta-dot {" in src, (
        "[P3-RESTOCK-MINIMAL-CTA] CSS class `.restock-cta-dot {...}` no encontrada."
    )
    assert "@keyframes restock-cta-pulse" in src, (
        "[P3-RESTOCK-MINIMAL-CTA] keyframes `restock-cta-pulse` no definidos. "
        "Sin animation el dot es estático y pierde el feedback 'ready to act'."
    )
    # El pulse usa box-shadow expand pattern (no scale — más sutil).
    assert "box-shadow: 0 0 0 7px rgba(16, 185, 129, 0)" in src, (
        "[P3-RESTOCK-MINIMAL-CTA] el pulse debe usar box-shadow expand pattern "
        "(ring grows + fades), no scale. Más sutil y respeta paleta."
    )


def test_button_hover_uses_slate_border(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] hover oscurece el borde a slate-900 (no
    a verde saturado). Coherente con paleta del sitio."""
    assert ".restock-cta-minimal:hover:not(:disabled)" in src
    assert "border-color: #0F172A" in src, (
        "[P3-RESTOCK-MINIMAL-CTA] hover del botón debe usar "
        "`border-color: #0F172A` (slate-900 = --text-main). Si usa verde, "
        "rompe la coherencia con paleta."
    )


def test_button_focus_visible_uses_indigo(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] focus-visible usa indigo-600 (color de
    marca `--primary`). Accesibilidad + coherencia visual."""
    assert ".restock-cta-minimal:focus-visible" in src
    assert "outline: 2px solid #4F46E5" in src, (
        "[P3-RESTOCK-MINIMAL-CTA] focus-visible debe usar `outline: 2px "
        "solid #4F46E5` (--primary indigo-600). Coherencia con design system."
    )


def test_button_respects_reduced_motion(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] `@media (prefers-reduced-motion: reduce)`
    desactiva el pulse del dot + el translateY del hover. A11y obligatorio."""
    # Reducción se aplica al menos al dot (animation: none).
    assert "@media (prefers-reduced-motion: reduce)" in src
    reduced_block_pattern = re.compile(
        r"@media\s*\(\s*prefers-reduced-motion:\s*reduce\s*\)\s*\{[^}]*?\.restock-cta-dot\s*\{[^}]*?animation:\s*none",
        re.DOTALL,
    )
    assert reduced_block_pattern.search(src), (
        "[P3-RESTOCK-MINIMAL-CTA] `@media (prefers-reduced-motion: reduce)` "
        "debe desactivar el pulse del `.restock-cta-dot` (animation: none). "
        "Sin esto, users con sensibilidad a movimiento sufren el pulse infinito."
    )


# ===========================================================================
# Modal de confirmación — icon outline + slate CTA + arrow microinteracción
# ===========================================================================

def test_modal_title_without_question_marks(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] el título es "Confirmar compra" (limpio),
    NO "¿Confirmar Compra?" (legacy con signos interrogativos)."""
    assert ">\n                                            Confirmar compra\n" in src, (
        "[P3-RESTOCK-MINIMAL-CTA] el título del modal debe ser exactamente "
        "'Confirmar compra' (sin signos interrogativos, sin capitalización "
        "'Compra'). Estilo más minimal."
    )
    assert "¿Confirmar Compra?" not in src, (
        "[P3-RESTOCK-MINIMAL-CTA] el título legacy '¿Confirmar Compra?' "
        "debe estar removido."
    )


def test_modal_icon_outline_no_heavy_bg(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] el icon container del modal usa border
    slate-200 + bg blanco (outline minimal), NO el gradient verde 64×64
    con sombra fuerte legacy."""
    # El cuadro verde legacy: `linear-gradient(135deg, #10B981 0%, #059669 100%)` con `boxShadow: '0 8px 16px rgba(16, 185, 129, 0.3)'`
    legacy_icon_box = re.compile(
        r"width:\s*'64px',\s*height:\s*'64px'[^}]*?borderRadius:\s*'20px'[^}]*?linear-gradient\(135deg,\s*#10B981",
        re.DOTALL,
    )
    assert not legacy_icon_box.search(src), (
        "[P3-RESTOCK-MINIMAL-CTA] el icon container legacy 64×64 con gradient "
        "verde debe estar removido."
    )
    # Nuevo: container 56×56 con border slate-200.
    new_icon_pattern = re.compile(
        r"width:\s*'56px',\s*height:\s*'56px'[^}]*?border:\s*'1\.5px solid #E2E8F0'",
        re.DOTALL,
    )
    assert new_icon_pattern.search(src), (
        "[P3-RESTOCK-MINIMAL-CTA] el icon container nuevo debe ser 56×56 con "
        "border slate-200 (#E2E8F0) y background blanco — outline minimal."
    )


def test_modal_status_dot_present(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] el icon container tiene un status dot
    emerald (14×14) en la esquina — preserva semántica 'ready' sin saturar."""
    status_dot_pattern = re.compile(
        r"width:\s*'14px',\s*height:\s*'14px'[^}]*?background:\s*'#10B981'",
        re.DOTALL,
    )
    assert status_dot_pattern.search(src), (
        "[P3-RESTOCK-MINIMAL-CTA] el status dot 14×14 emerald-500 en la "
        "esquina del icon debe estar presente (el ÚNICO acento de color "
        "del modal post-rediseño)."
    )


def test_modal_confirm_button_uses_slate_900(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] el botón principal del modal usa
    `className="restock-modal-confirm"` con `background: #0F172A`
    (slate-900 = --text-main). NO gradient verde legacy."""
    assert 'className="restock-modal-confirm"' in src, (
        "[P3-RESTOCK-MINIMAL-CTA] el botón principal del modal debe usar "
        "`className=\"restock-modal-confirm\"`."
    )
    assert ".restock-modal-confirm {" in src
    assert "background: #0F172A;" in src, (
        "[P3-RESTOCK-MINIMAL-CTA] el botón principal debe tener "
        "`background: #0F172A` (slate-900, --text-main del design system)."
    )


def test_modal_confirm_button_has_animated_arrow(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] el botón principal tiene `<ArrowRight ...
    className="restock-modal-arrow" />` que se desliza horizontalmente
    en hover via `translateX(4px)`. Microinteracción minimal."""
    assert '<ArrowRight size={17}' in src, (
        "[P3-RESTOCK-MINIMAL-CTA] `<ArrowRight size={17} />` debe estar "
        "presente en el botón confirm del modal. La flecha es el visual "
        "que comunica 'siguiente acción'."
    )
    assert 'className="restock-modal-arrow"' in src, (
        "[P3-RESTOCK-MINIMAL-CTA] el `<ArrowRight>` debe tener "
        "`className=\"restock-modal-arrow\"` para targeting del hover transition."
    )
    # Hover translate definido.
    hover_translate = re.compile(
        r"\.restock-modal-confirm:hover:not\(:disabled\)\s+\.restock-modal-arrow\s*\{[^}]*?transform:\s*translateX\(4px\)",
        re.DOTALL,
    )
    assert hover_translate.search(src), (
        "[P3-RESTOCK-MINIMAL-CTA] el hover del CTA debe trasladar la flecha "
        "`translateX(4px)`. Es la microinteracción que comunica acción minimal."
    )


def test_modal_cancel_is_text_link_not_button(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] el cancelar usa `className="restock-modal-cancel"`
    con background transparent, sin padding pesado — es un text-link
    (NO compite visualmente con el CTA principal)."""
    assert 'className="restock-modal-cancel"' in src
    assert ".restock-modal-cancel {" in src
    # El CSS debe tener background transparent.
    cancel_block = re.search(
        r"\.restock-modal-cancel\s*\{(?P<body>[^}]+)\}",
        src,
        re.DOTALL,
    )
    assert cancel_block, "CSS class .restock-modal-cancel no encontrada"
    body = cancel_block.group("body")
    assert "background: transparent;" in body, (
        "[P3-RESTOCK-MINIMAL-CTA] `.restock-modal-cancel` debe tener "
        "`background: transparent` (text-link, no botón colorido)."
    )
    assert "color: #94A3B8;" in body, (
        "[P3-RESTOCK-MINIMAL-CTA] `.restock-modal-cancel` debe usar "
        "`color: #94A3B8` (--text-light, gris discreto). Pre-fix también "
        "lo usaba, mantener para consistencia con el resto del sitio."
    )


def test_modal_no_legacy_green_gradients(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] los gradients verdes del CTA y del icon
    box del modal NO deben estar en el JSX inline. Solo el status dot
    14×14 puede usar verde puro (#10B981) como acento."""
    # El gradient legacy del CTA principal del modal.
    legacy_cta_gradient_in_modal = re.compile(
        r"onClick=\{handleRestock\}[^>]*?linear-gradient\(135deg,\s*#10B981",
        re.DOTALL,
    )
    assert not legacy_cta_gradient_in_modal.search(src), (
        "[P3-RESTOCK-MINIMAL-CTA] el gradient verde del CTA `onClick={handleRestock}` "
        "debe estar removido. Pre-fix tenía `linear-gradient(135deg, #10B981 0%, "
        "#059669 100%)` inline — ahora vive en `.restock-modal-confirm` con bg "
        "slate-900."
    )


# ===========================================================================
# Tooltip-anchor preservado
# ===========================================================================

def test_tooltip_anchor_present(src: str) -> None:
    """[P3-RESTOCK-MINIMAL-CTA] marker textual aparece ≥4 veces (botón
    comment + botón CSS comment + modal comment JSX + modal CSS comment)."""
    count = src.count("P3-RESTOCK-MINIMAL-CTA")
    assert count >= 4, (
        f"[P3-RESTOCK-MINIMAL-CTA] esperaba ≥4 menciones del marker en "
        f"Dashboard.jsx. Encontradas: {count}."
    )
