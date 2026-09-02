"""[P3-CREDITS-LAST-ONE · 2026-09-02] Ancla backend de la decisión de diseño del medidor de
créditos: el color NO cambia con el último crédito (ámbar = «aún puedes»; el rojo queda para
el 0 real, o pierde significado). La urgencia va en una señal secundaria: etiqueta «Último
crédito» + latido más vivo del bloom, apagado bajo prefers-reduced-motion.

Tooltip-anchor: P3-CREDITS-LAST-ONE | isLastCredit = state === 'low' && remaining === 1
"""
import json
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def _src(rel: str) -> str:
    p = FRONTEND / rel
    if not p.exists():
        pytest.skip(f"frontend no visible desde este checkout: {rel}")
    return p.read_text(encoding="utf-8")


def test_last_credit_is_secondary_signal_not_a_color_change():
    jsx = _src("src/components/dashboard/CreditsMeter.jsx")
    assert "const isLastCredit = state === 'low' && remaining === 1;" in jsx
    assert "t('Último crédito')" in jsx
    # el rojo sigue reservado al 0 real
    assert "else if (remaining <= 0 || isLimitReached) state = 'depleted';" in jsx


def test_pulse_has_reduced_motion_off_switch():
    css = _src("src/components/dashboard/CreditsMeter.module.css")
    assert ".lastCredit .core {" in css and ".lastCredit .gauge::before {" in css
    assert ".lastCredit .core { animation: none; }" in css


@pytest.mark.parametrize("loc", ["en-US", "fr-FR", "it-IT", "pt-BR"])
def test_label_translated(loc):
    cat = json.loads(_src(f"src/i18n/locales/{loc}.json"))
    assert cat.get("Último crédito"), f"{loc}: falta la traducción de «Último crédito»"


def test_vitest_exists():
    assert (FRONTEND / "src/__tests__/CreditsMeter.last_credit.test.jsx").exists()
