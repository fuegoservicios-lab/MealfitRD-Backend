# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-241 · 2026-09-25] El horario del formulario (turno nocturno / rotativo) llega al plan.

Auditoría del 25-sep: el generador no leía `scheduleType` en ningún sitio; al turno nocturno le salía «Desayuno 06:30 ·
Almuerzo 13:00 · Cena 19:30» (almuerzo a la hora en que duerme) y el día 2 «Flexible».
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import horizon  # noqa: E402


def test_el_generador_lee_el_horario_en_claro():
    out = horizon.explain_form_codes_for_prompt({"scheduleType": "night_shift", "cookingTime": "30min"})
    assert "duerme de día" in out["scheduleType"] and "SIN café" in out["scheduleType"]
    assert "30 minutos" in out["cookingTime"], "el tiempo de cocina se sigue explicando"
    assert horizon.explain_form_codes_for_prompt({"scheduleType": "standard"})["scheduleType"] == "standard"
    assert "ROTATIVO" in horizon.explain_form_codes_for_prompt({"scheduleType": "variable"})["scheduleType"]


def test_los_reescritores_reciben_la_regla():
    assert horizon.schedule_rule({"scheduleType": "night_shift"})
    assert horizon.schedule_rule({"scheduleType": "standard"}) == "" and horizon.schedule_rule({}) == ""
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert go.count('__import__("horizon").schedule_rule(') == 2, "corrector del self-critique y regen quirúrgico"
    assert "HORARIO DEL USUARIO (obligatorio, también al corregir): {_sch_rule}" in go
    assert "HORARIO DEL USUARIO (obligatorio, también al corregir): {_sch_rule_sg}" in go
    ag = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert '__import__("horizon").schedule_rule(form_data)' in ag and "🌙 HORARIO DEL USUARIO" in ag


def _plan():
    return {"days": [{"day": d, "meals": [{"meal": s, "time": t, "name": f"{s} {d}"} for s, t in (
        ("Desayuno", "06:30"), ("Almuerzo", "13:00"), ("Merienda", "16:30"), ("Cena", "19:30"))]} for d in (1, 2)]}


def test_las_horas_del_turno_nocturno():
    import horario_comidas as hc
    p = _plan()
    assert hc.asignar_horas(p, {"scheduleType": "night_shift"}) == 8
    horas = {m["meal"]: m["time"] for m in p["days"][0]["meals"]}
    assert horas == {"Desayuno": "15:30", "Almuerzo": "19:30", "Merienda": "01:00", "Cena": "07:00"}, horas
    assert hc.asignar_horas(p, {"scheduleType": "night_shift"}) == 0, "idempotente"


def test_rotativo_flexible_y_estandar_intacto():
    import horario_comidas as hc
    p = _plan()
    hc.asignar_horas(p, {"scheduleType": "variable"})
    assert {m["time"] for d in p["days"] for m in d["meals"]} == {"Flexible"}
    p2, p3 = _plan(), _plan()
    assert hc.asignar_horas(p2, {"scheduleType": "standard"}) == 0 and p2 == p3


def test_cableado_en_el_escudo():
    dp = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    assert dp.index("_hcom.asignar_horas(_pd, _clin_ctx)") < dp.index("_rfin.retirar_prohibidos(_pd, _clin_ctx")
    assert '"scheduleType": _hp.get("scheduleType")' in (_BACKEND / "db_profiles.py").read_text(encoding="utf-8")


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 241
