"""[P2-NO-CREDITS-CTA · 2026-09-02] Ancla backend del cambio de UX del botón de créditos.

Captura del dueño: el botón «Límite» brillaba en índigo al pasar el ratón (hover de un estado
apagado: llevaba aria-disabled, no disabled) y al hacer clic navegaba a la Nevera (la rama de
«Nevera vacía» iba ANTES que la de créditos) o abría y cerraba el modal. Ahora: créditos
primero, un aviso «Sin créditos este mes · Se renuevan el {fecha}» con «Mejorar plan» (oculto
en nativo), sin hover en ese estado; y el CTA «Sí, ya compré» de la Nevera con el mismo hover
que el resto (uniformidad pedida por el dueño).

Tooltip-anchor: P2-NO-CREDITS-CTA | _noCreditsToast
"""
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def _src(rel: str) -> str:
    p = FRONTEND / rel
    if not p.exists():
        pytest.skip("frontend no visible desde este checkout")
    return p.read_text(encoding="utf-8")


def test_credits_checked_before_pantry_and_single_toast():
    src = _src("src/pages/Dashboard.jsx")
    m = src.index("Créditos ANTES que Nevera")
    i = src.index("_noCreditsToast();", m)
    j = src.index("navigate('/dashboard/pantry');", i)
    assert i < j and "if (planFinished) {" in src[i:j]
    assert "toast.error(t('Sin créditos este mes'), {" in src
    assert "timeZone: 'UTC'" in src


def test_no_hover_glow_when_limit_reached():
    src = _src("src/pages/Dashboard.jsx")
    assert '.new-plan-btn:hover:not(:disabled):not([aria-disabled="true"])' in src
    assert "? t('Sin créditos')" in src and "t('Límite')" not in src


def test_restock_cta_has_uniform_hover():
    src = _src("src/components/dashboard/RestockNudge.jsx")
    assert ".restock-nudge-cta:hover:not(:disabled) {" in src and ".restock-nudge-cta:active:not(:disabled) {" in src
