# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-559 · 2026-09-27] Cambiar/actualizar platos con el horario, los básicos y el país del plan.

Auditoría del formulario: `swap_meal` inyecta `horizon.schedule_rule(form_data)` pero `scheduleType` no llegaba (el
frontend no lo manda y `_enrich_clinical_from_profile` no lo hidrataba); el `meal_form` de «Actualizar platos» descartaba
`staple_foods` y `scheduleType`, y tomaba el país del perfil vivo en vez del sello del plan.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


def _cuerpo(fn):
    m = re.search(rf"def\s+{re.escape(fn)}\s*\(", _PLANS)
    nxt = re.search(r"\n(?:@router\.|@app\.|def\s)", _PLANS[m.start() + 1:])
    return _PLANS[m.start():m.start() + 1 + nxt.start()] if nxt else _PLANS[m.start():]


def test_el_swap_recibe_el_horario_del_perfil(monkeypatch):
    import routers.plans as rp
    monkeypatch.setattr(rp, "get_user_profile", lambda uid: {"health_profile": {"scheduleType": "night_shift"}},
                        raising=False)
    import db
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {"scheduleType": "night_shift"}})
    data = {"user_id": "u-559"}
    rp._enrich_clinical_from_profile(data, "u-559")   # hidrata `data` en su sitio (devuelve el perfil)
    assert data.get("scheduleType") == "night_shift"
    import horizon
    assert "NOCTURNO" in horizon.schedule_rule(data)


def test_actualizar_platos_lleva_horario_basicos_y_pais_del_plan():
    b = _cuerpo("api_regenerate_day")
    i = b.find("meal_form = {")
    bloque = b[i:i + 9000]
    assert '"scheduleType": data.get("scheduleType"),' in bloque
    assert '"staple_foods": data.get("staple_foods") or data.get("stapleFoods"),' in bloque
    assert 'country_for_plan(plan_data, {"country": data.get("country")})' in bloque
    assert '"country": data.get("country"),' not in bloque


def test_el_sello_del_plan_gana_al_perfil_vivo(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    from constants import country_for_plan
    assert country_for_plan({"_country": "DO"}, {"country": "ES"}) == "DO"
    assert country_for_plan({}, {"country": "ES"}) == "ES"
