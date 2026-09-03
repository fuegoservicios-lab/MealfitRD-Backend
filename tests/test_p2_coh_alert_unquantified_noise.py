"""[P2-COH-ALERT-UNQUANTIFIED-NOISE · 2026-09-03] La alerta diaria de coherencia mide SEÑAL.

Medido en la alerta del 03-sep (cron `_shopping_coherence_alert_job`): 31/32 planes «con
divergencias» (96,9 %), pero 101 de 113 divergencias eran `recipe_unquantified` — condimentos «al
gusto» sin gramos en la receta (Pimienta negra en 29 planes, Sal en 27). El umbral
`MEALFIT_COH_ALERT_PLAN_FRACTION` (10 %) disparaba un ERROR cada día: una alerta que siempre dispara
no es una alerta. Ahora el umbral se calcula sobre los planes con al menos una divergencia cuya
hipótesis NO está en `MEALFIT_COH_ALERT_IGNORE_HYPOTHESES` (default `recipe_unquantified`); el conteo
bruto sigue en el resumen y en las flags del tick.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
SRC = (BACKEND / "cron_tasks.py").read_text(encoding="utf-8")


def _cron_module():
    return pytest.importorskip("cron_tasks")


def test_helper_signal_ignores_only_the_configured_hypotheses(monkeypatch):
    ct = _cron_module()
    monkeypatch.delenv("MEALFIT_COH_ALERT_IGNORE_HYPOTHESES", raising=False)
    ign = ct._coh_alert_ignored_hypotheses()
    assert ign == frozenset({"recipe_unquantified"})
    only_noise = [{"hypothesis": "recipe_unquantified", "food": "Sal"}, {"hypothesis": "recipe_unquantified", "food": "Pimienta negra"}]
    assert ct._coh_plan_has_signal(only_noise, ign) is False
    assert ct._coh_plan_has_signal(only_noise + [{"hypothesis": "cap_swallowed_modifier", "food": "Pollo"}], ign) is True
    assert ct._coh_plan_has_signal([{"food": "x"}], ign) is True          # sin hipótesis ⇒ `unknown` ⇒ señal
    assert ct._coh_plan_has_signal([], ign) is False and ct._coh_plan_has_signal(None, ign) is False
    monkeypatch.setenv("MEALFIT_COH_ALERT_IGNORE_HYPOTHESES", "recipe_unquantified, unit_mismatch")
    ign2 = ct._coh_alert_ignored_hypotheses()
    assert ign2 == frozenset({"recipe_unquantified", "unit_mismatch"})
    assert ct._coh_plan_has_signal([{"hypothesis": "unit_mismatch"}], ign2) is False
    # vacío ⇒ default (contrato de `_env_str`); para «no ignorar nada» se pone una hipótesis inexistente
    monkeypatch.setenv("MEALFIT_COH_ALERT_IGNORE_HYPOTHESES", "ninguna")
    assert ct._coh_plan_has_signal(only_noise, ct._coh_alert_ignored_hypotheses()) is True


def test_cron_threshold_uses_plans_with_signal_and_keeps_raw_count():
    """Parser: el umbral se calcula con `plans_with_signal`; el bruto `plans_with_div` sigue vivo
    (resumen + flags del tick P2-LIVE-9)."""
    assert "plan_fraction = plans_with_signal / n if n else 0.0" in SRC
    assert "plan_fraction = plans_with_div / n if n else 0.0" not in SRC
    assert "if _coh_plan_has_signal(divs, _ignored_hyp):" in SRC
    assert "_tick_plans_with_div = plans_with_div" in SRC
    assert re.search(r'f"\{plans_with_div\} con divergencias, \{plans_with_signal\} con señal', SRC)
    # el helper se declara ANTES del job y el contador se inicializa junto al bruto
    assert SRC.index("def _coh_plan_has_signal(") < SRC.index("def _shopping_coherence_alert_job(")
    assert SRC.index("plans_with_div = 0") < SRC.index("plans_with_signal = 0") < SRC.index("if divs:")


def test_marker_bumped():
    assert "P2-COH-ALERT-UNQUANTIFIED-NOISE" in (BACKEND / "app.py").read_text(encoding="utf-8")
