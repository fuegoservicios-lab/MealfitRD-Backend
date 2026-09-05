"""[P1-ARQ25-F7-CULTURE · 2026-09-05] Fase 7 (subfase B): contrato frontend↔backend del paso «Cocinas que
te representan». `frontend/src/config/cultures.js` repite los perfiles (id, etiqueta, país), las intensidades y
el tope de secundarias del motor `cultural_profiles.py`; el wizard escribe `cultureProfiles` con la forma que
`weights_from_form_field` lee; la política llega al panel con `culture_weights`.
"""
import re
from pathlib import Path

import pytest

import cultural_profiles as cp

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend" / "src"
_CFG = _FRONT / "config" / "cultures.js"


def _front_list(js: str, name: str) -> list:
    m = re.search(r"export const " + name + r"\s*=\s*\[([^\]]*)\]", js)
    assert m, name
    return re.findall(r"'([a-z_]+)'", m.group(1))


@pytest.mark.skipif(not _CFG.exists(), reason="frontend no disponible en este checkout")
def test_a_el_frontend_repite_los_perfiles_intensidades_y_tope():
    js = _CFG.read_text(encoding="utf-8")
    ids = re.findall(r"\{ id: '([a-z_]+)', labelKey: i18nKey\('([^']+)'\), marketDefault: '([A-Z]{2})' \}", js)
    assert [i[0] for i in ids] == list(cp.PROFILES), "mismos ids y mismo orden que PROFILES"
    for pid, label, market in ids:
        assert cp.PROFILES[pid]["name_es"] == label, pid
        assert cp.PROFILES[pid]["market_default"] == market, pid
    assert _front_list(js, "CULTURE_INTENSITIES") == list(cp.INTENSITY_WEIGHT)
    for k, v in cp.INTENSITY_WEIGHT.items():
        assert f"{k}: {int(round(v * 100))}" in js, k
    assert f"MAX_SECONDARY_CULTURES = {cp.MAX_SECONDARY};" in js
    assert f"DEFAULT_CULTURE = '{cp.DEFAULT_PROFILE}';" in js


@pytest.mark.skipif(not _CFG.exists(), reason="frontend no disponible en este checkout")
def test_b_el_wizard_escribe_la_forma_que_el_motor_lee():
    ctx = (_FRONT / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
    assert re.search(r"cultureProfiles\s*:\s*null,", ctx), "nace null: la sugerencia no se siembra"
    q = (_FRONT / "components" / "assessment" / "questions" / "QCulture.jsx").read_text(encoding="utf-8")
    assert "updateData('cultureProfiles', { main: id, secondary:" in q
    assert "{ profile_id: id, intensity: DEFAULT_INTENSITY }" in q
    # la forma del wizard es exactamente la que lee weights_from_form_field
    ws = cp.weights_from_form_field({"main": "dominican_criolla", "secondary": [{"profile_id": "us_everyday", "intensity": "frecuente"}]})
    assert [w["profile_id"] for w in ws] == ["dominican_criolla", "us_everyday"]
    flow = (_FRONT / "components" / "assessment" / "InteractiveAssessmentFlow.jsx").read_text(encoding="utf-8")
    assert "...(COUNTRY_SYSTEM_UI && CULTURE_PROFILES_UI ? [{" in flow
    assert "<QCulture onManualAdvance={nextStep} />" in flow
    panel = (_FRONT / "components" / "dashboard" / "PlanPolicyPanel.jsx").read_text(encoding="utf-8")
    assert "effective.culture_weights" in panel


def test_c_el_efectivo_lleva_culture_weights_para_el_panel():
    import plan_policy as pp
    form = {"country": "US", "cultureProfiles": {"main": "dominican_criolla", "secondary": [{"profile_id": "us_everyday", "intensity": "ocasional"}]}}
    eff = pp.compile_from_form(form)["effective"]
    assert eff["culture_weights"][0] == {"profile_id": "dominican_criolla", "weight": 0.85}
    assert eff["culture_weights"][1]["profile_id"] == "us_everyday" and abs(eff["culture_weights"][1]["weight"] - 0.15) < 1e-9
