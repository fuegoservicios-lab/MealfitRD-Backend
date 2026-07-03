"""[P1-GATES-FLIP-ON · 2026-07-03] (audit v6 · P1-4) Flip OFF→ON de los 3 gates con serie.

Playbook medir→actuar COMPLETADO: gym baseline de 20 perfiles held-out (scripts/plan_gym.py,
LLM real, 2026-07-03; evidencia en docs/gym_baseline_2026_07_03.json):
  - RECIPE_CONTRACT:   contract_ratio 0.0 en 19/20 planes, 0.333 en 1 (< umbral 0.5)
                       → el gate habría disparado retry en 0/20. Cero riesgo de retry-storm.
  - SODIUM_EXCESS:     worst-day ceiling flag en 4/20; el trigger real del gate (promedio
                       > techo×1.5) es más raro + advisory en intento final → seguro.
  - MICRO_CLOSER_PERDAY: 9/20 planes con worst-day floor flaggeado (micros = eje más débil,
                       mean 57.5) → el closer per-día tiene trabajo real y NO es gate.

Patrón P1-VERIFIED-ONLY-DEFAULT-ON: default True en CÓDIGO + baseline OFF en conftest (los
fixtures históricos construyen planes sintéticos que dispararían los gates a propósito).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_CONFTEST = (_BACKEND / "tests" / "conftest.py").read_text(encoding="utf-8")


def test_marker_bumped():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-GATES-FLIP-ON" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-03"


def test_code_defaults_on():
    """Los 3 defaults de CÓDIGO son True — un revert accidental falla aquí primero."""
    assert re.search(r'SODIUM_EXCESS_GATE_ENABLED\s*=\s*_env_bool\("MEALFIT_SODIUM_EXCESS_GATE",\s*True\)', _GO)
    assert re.search(r'RECIPE_CONTRACT_GATE_ENABLED\s*=\s*_env_bool\("MEALFIT_RECIPE_CONTRACT_GATE",\s*True\)', _GO)
    assert re.search(r'MICRO_CLOSER_PERDAY_ENABLED\s*=\s*_env_bool\("MEALFIT_MICRO_CLOSER_PERDAY",\s*True\)', _GO)


def test_conftest_baseline_off():
    """El baseline de la suite se fija OFF (setdefault: env real del operador gana)."""
    for knob in ("MEALFIT_SODIUM_EXCESS_GATE", "MEALFIT_RECIPE_CONTRACT_GATE", "MEALFIT_MICRO_CLOSER_PERDAY"):
        assert f'setdefault("{knob}", "false")' in _CONFTEST, f"falta baseline OFF de {knob}"
        assert os.environ.get(knob, "").lower() == "false", f"{knob} debe estar OFF en runtime de tests"


def test_evidence_file_present():
    """La serie que justificó el flip queda versionada (auditable sin re-correr el gym)."""
    ev = _BACKEND / "docs" / "gym_baseline_2026_07_03.json"
    assert ev.exists(), "falta docs/gym_baseline_2026_07_03.json (evidencia del flip)"
    import json
    d = json.loads(ev.read_text(encoding="utf-8"))
    assert (d.get("aggregate") or {}).get("n") == 20, "el baseline debe ser de los 20 perfiles held-out"


def test_gates_still_degrade_to_advisory_on_final_attempt():
    """El flip NO cambia el contrato nunca-cero-plan: ambos gates degradan a advisory al final."""
    assert "_recipe_contract_advisory_final" in _GO
    # el gate de sodio usa el patrón advisory-en-intento-final del review — buscar desde el
    # ENFORCEMENT (última ocurrencia del knob, en review_plan_node), no desde la definición.
    idx = _GO.rfind("SODIUM_EXCESS_GATE_ENABLED")
    assert "advisory" in _GO[idx:idx + 6000].lower()
