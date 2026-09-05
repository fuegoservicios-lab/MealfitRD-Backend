"""[P2-RESTOCK-COPY-COVERAGE · 2026-09-02] El copy de «Ya compré la lista» sigue al estado
de la Nevera.

Vivo (plan 82da3be1, renovado con 48 ítems en la Nevera): el header decía «36 ítems de la
lista ya en tu Nevera» y el banner de abajo «Tu Nevera está vacía para este plan», con el
botón «Ya compré la lista» cuando lo que faltaba eran 5. El banner solo miraba
«no restocked + hay pendientes» (is_restocked se resetea al renovar, a propósito).

Tres estados, con los números que YA calcula `shoppingDeltaMeta` (sin lógica nueva):
  - cobertura cero  → copy de siempre (anclas de tests históricos intactas);
  - cobertura parcial → botón «Ya compré lo que faltaba (n)», banner «ya cubre X de N»;
  - cobertura total → el botón ya no se renderiza (hasPendingShoppingItems=false).
Parser-based sobre Dashboard.jsx + RestockNudge.jsx + los 4 catálogos. La conducta
la cubre `frontend/src/__tests__/RestockNudge.p2_copy_coverage.test.jsx`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_FRONT = _ROOT / "frontend" / "src"
_DASH_PATH = _FRONT / "pages" / "Dashboard.jsx"
_NUDGE_PATH = _FRONT / "components" / "dashboard" / "RestockNudge.jsx"
_DASH = ""
_NUDGE = ""


@pytest.fixture(scope="module", autouse=True)
def _load_frontend_sibling_sources(frontend_repo_path):
    # [P2-CI-BACKEND-CERO-TESTS] la fixture compartida salta el módulo si falta el hermano;
    # la lectura ya no ocurre al importar (el checkout del backend no trae ../frontend).
    _ = frontend_repo_path
    global _DASH, _NUDGE
    _DASH = _DASH_PATH.read_text(encoding="utf-8")
    _NUDGE = _NUDGE_PATH.read_text(encoding="utf-8")

_KEYS = (
    "Ya compré lo que faltaba ({n})",
    "Tu Nevera ya cubre {cubiertos} de {total} ítems de este plan",
    "¿Ya compraste lo que falta ({n})? Márcalo con un toque para que tu plan use lo que tienes.",
    "Sí, ya los compré",
)


def test_button_label_follows_coverage_and_keeps_legacy_copy():
    i = _DASH.find('className="restock-cta-dot"')
    assert i != -1
    win = _DASH[i:i + 900]
    assert "(shoppingDeltaMeta?.itemsRemoved || 0) > 0" in win, "la cobertura es la del delta YA calculado"
    assert "t('Ya compré lo que faltaba ({n})', { n: shoppingDeltaMeta?.deltaCount || 0 })" in win
    assert "t('Ya compré la lista')" in win, "cobertura cero conserva el copy de siempre"


def test_nudge_receives_real_coverage_from_dashboard():
    i = _DASH.find("<RestockNudge")
    assert i != -1
    win = _DASH[i:i + 1500]
    assert "coveredCount={shoppingDeltaMeta?.itemsRemoved || 0}" in win
    assert "pendingCount={shoppingDeltaMeta?.deltaCount || 0}" in win


def test_nudge_copy_is_conditional_with_safe_defaults():
    assert "coveredCount = 0," in _NUDGE and "pendingCount = 0," in _NUDGE, \
        "llamadores viejos sin props ⇒ copy de cobertura cero"
    assert "const partialCoverage = coveredCount > 0 && pendingCount > 0;" in _NUDGE
    assert "t('Tu Nevera está vacía para este plan')" in _NUDGE, "ancla histórica (P5-RESTOCK-PRESERVE)"
    assert "t('Sí, ya compré')" in _NUDGE and "t('Sí, ya los compré')" in _NUDGE
    for k in _KEYS[1:]:
        assert f"t('{k}'" in _NUDGE, k


@pytest.mark.parametrize("locale", ["en-US", "fr-FR", "it-IT", "pt-BR"])
def test_catalogs_carry_the_four_keys_with_placeholders(locale):
    cat = (_FRONT / "i18n" / "locales" / f"{locale}.json").read_text(encoding="utf-8")
    for k in _KEYS:
        head = f'"{k}": "'
        assert head in cat, (locale, k)
        tail = cat.split(head, 1)[1][:400]
        for ph in ("{n}", "{cubiertos}", "{total}"):
            if ph in k:
                assert ph in tail, (locale, k, ph)
