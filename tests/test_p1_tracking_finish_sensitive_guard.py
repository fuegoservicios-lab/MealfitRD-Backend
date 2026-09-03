# -*- coding: utf-8 -*-
"""[P1-TRACKING-FINISH-SENSITIVE-GUARD · 2026-08-12] Cierre del hallazgo más
serio del audit del formulario: pulsar «Empezar a contar» durante la ventana
de descifrado asíncrono PATCHeaba `medicalConditions: []` sobre el
health_profile REAL (merge jsonb key-level = destructivo) y, con las
condiciones «vacías», el gate de embarazo/lactancia del calculador quedaba
puenteado — metas CON DÉFICIT y ok:true, sin missing_fields que delatara nada.

Tres capas (las dos frontend se anclan en PlanMode.contract.test.jsx):
  1. Frontend: guard loadingSensitive en QTrackingFinish + arrays vacíos NO se
     persisten (omitir la clave conserva lo que el servidor sabe).
  2. Backend targets: medicalConditions REQUERIDA — [] es falsy ⇒ missing.
  3. Backend generación: appMode fuera del health_profile (ruteo, no salud).
"""
import asyncio
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
FRONTEND_SRC = BACKEND.parent / "frontend" / "src"

_PERFIL_BASE = {
    "age": "23", "gender": "male", "height": "168", "weight": "123",
    "weightUnit": "lb", "activityLevel": "sedentary", "mainGoal": "gain_muscle",
}


def _targets(monkeypatch, hp):
    import db
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": hp})
    from routers.user_data import api_nutrition_targets
    return asyncio.run(api_nutrition_targets(verified_user_id="u-guard"))


def test_condiciones_vacias_es_fail_closed_no_deficit_silencioso(monkeypatch):
    """El bypass del incidente: perfil sin medicalConditions (o []) YA NO
    devuelve metas ok:true — declara el faltante."""
    for hp in (dict(_PERFIL_BASE), {**_PERFIL_BASE, "medicalConditions": []}):
        out = _targets(monkeypatch, hp)
        assert out.get("ok") is False, out
        assert "medicalConditions" in out.get("missing_fields", []), out


def test_con_condiciones_declaradas_sigue_ok(monkeypatch):
    out = _targets(monkeypatch, {**_PERFIL_BASE, "medicalConditions": ["Ninguna"]})
    assert out.get("ok") is True, out
    assert out["calories"] > 0


def test_hp_data_de_generacion_excluye_appmode():
    """appMode es ruteo del wizard: el modo real vive en la COLUMNA plan_mode.
    Dejarlo colarse al jsonb creaba una segunda verdad."""
    src = (BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    m = re.search(r"hp_data = \{k: v for k, v in data\.items\(\) if k not in \(([^)]*)\)\}", src)
    assert m, "el filtro de hp_data cambió de forma — actualizar este ancla"
    assert "'appMode'" in m.group(1), (
        "appMode volvió a colarse al health_profile — es ruteo del wizard, "
        "no dato de salud (la columna plan_mode es la verdad del modo)."
    )


def test_frontend_guard_y_omision_de_arrays_vacios():
    """Anclas de la capa 1 (el archivo vive en el repo hermano frontend)."""
    src = (FRONTEND_SRC / "components" / "assessment" / "questions" / "QTrackingFinish.jsx").read_text(encoding="utf-8")
    # guard ANTES de encender saving (si vive después, la ventana sigue abierta)
    g = src.index("if (loadingSensitive)")
    s = src.index("setSaving(true)")
    assert g < s, "el guard loadingSensitive debe correr ANTES de setSaving"
    assert "Array.isArray(v) && v.length === 0) continue" in src, (
        "la omisión de arrays vacíos murió: un [] de hidratación incompleta "
        "volvería a machacar allergies/medicalConditions reales del perfil."
    )
