# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-244 · 2026-09-25] El camino degradado (sin IA) tamiza también el plan de respaldo y mira rechazos y
embarazo.

Auditoría del 25-sep: los días y comidas de `emergency_backup_plan` saltaban el tamiz (entraban cuando se agotaba el
pool), y el escáner del camino degradado no recibía el perfil (ni rechazos ni mercurio en embarazo).
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import cron_tasks as ct  # noqa: E402


def _dia(nombre, ings):
    return {"meals": [{"meal": "Almuerzo", "name": nombre, "ingredients": ings}]}


def test_el_escaner_ve_los_rechazos_con_el_perfil():
    d = _dia("Bowl de atún", ["120 g de atún en agua", "½ taza de arroz"])
    assert ct._degraded_safety_violations(d, [], "balanced") == []
    v = ct._degraded_safety_violations(d, [], "balanced", form_data={"dislikes": ["Pescado"]})
    assert v and "rechazo" in v[0], v


def test_el_escaner_ve_el_mercurio_en_embarazo():
    d = _dia("Pez espada a la plancha", ["150 g de pez espada"])
    v = ct._degraded_safety_violations(d, [], "balanced",
                                        form_data={"medicalConditions": ["Embarazo"], "gender": "female"})
    assert v, "el embarazo sin mercurio también en el camino degradado"


def test_el_respaldo_pasa_por_el_tamiz():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    i = src.index("backup_days = health_profile.get('emergency_backup_plan', []) or []")
    bloque = src[i:i + 1400]
    assert "_degraded_safety_violations(d, current_allergies, current_diet," in bloque
    assert "tooltip-anchor: P1-PLAN-LOTE-244-RESPALDO-TAMIZADO" in bloque
    assert "form_data=locals().get(\"_hp_union\")" in src


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 244
