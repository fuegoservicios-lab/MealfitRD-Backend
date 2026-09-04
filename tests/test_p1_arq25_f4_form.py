"""[P1-ARQ25-F4-FORM · 2026-09-03] Fase 4 del roadmap 2.5: formulario progresivo y UX de explicación.

Contrato que este test ancla (roadmap §6.7, Fase 4):
  1. El adapter del formulario entiende las preguntas nuevas — `mealOrganization` (obligatoria en el
     wizard, default `balanced` en el backend para clientes viejos), `freezerMode`, `freshTopup`,
     `batchCooking` — y «Mis básicos» como EDITOR DE ANCLAS (`stapleAnchors`: franja, frecuencia
     semanal min/max por 7 días, misma/variada preparación). `source.form_version` distingue v1/v2.
  2. El sello de fidelidad (`_fidelity_report`) dice `mode` (enforce|shadow): la pantalla
     «solicitaste / aplicamos / por qué» solo afirma «aplicamos» cuando el motor obedeció.
  3. Sink del embudo del wizard (`POST /api/plans/telemetry/wizard` → `pipeline_metrics.wizard_funnel`):
     la línea base de conversión que el gate exige medir, invitados incluidos, sid hasheado.
  4. Contrato versionado frontend↔backend: `frontend/src/config/planPolicy.js` repite los enums y
     los reason codes del compilador; si divergen, este test falla antes que el usuario.
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import pytest

import plan_policy as pp

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT_CFG = _BACKEND.parent / "frontend" / "src" / "config" / "planPolicy.js"


def _base_form(**extra) -> dict:
    form = {
        "dietType": "balanced", "allergies": [], "dislikes": [], "medicalConditions": [],
        "stapleFoods": ["Huevo", "Arroz"], "groceryDuration": "biweekly", "cookingTime": "30min",
        "budget": "medium", "householdSize": 1, "country": "DO",
    }
    form.update(extra)
    return form


# ── 1. adapter ───────────────────────────────────────────────────────────────
def test_stapleAnchors_enriquece_las_anclas_sin_mover_los_nombres():
    pol = pp.policy_from_form(_base_form(stapleAnchors=[
        {"name": "Huevo", "slots": ["desayuno", "Breakfast", "merienda"], "min_per_7d": 5, "max_per_7d": 7, "preparation_mode": "same_preparation"},
    ]))
    by = {a["name"]: a for a in pol["food_anchors"]}
    assert set(by) == {"Huevo", "Arroz"}, "los nombres siguen siendo los de stapleFoods"
    assert by["Huevo"]["slots"] == ["breakfast", "snack"], "franjas canónicas, sin duplicados"
    assert (by["Huevo"]["min_per_7d"], by["Huevo"]["max_per_7d"]) == (5, 7)
    assert by["Huevo"]["preparation_mode"] == "same_preparation"
    assert (by["Arroz"]["min_per_7d"], by["Arroz"]["max_per_7d"], by["Arroz"]["preparation_mode"]) == (2, 7, "vary_preparation")
    assert pol["source"]["form_version"] == "v2"


def test_anclas_con_detalle_basura_caen_a_los_defaults_y_se_acotan():
    pol = pp.policy_from_form(_base_form(stapleAnchors=[
        {"name": "Huevo", "slots": ["brunch"], "min_per_7d": "9", "max_per_7d": "abc", "preparation_mode": "raw"},
        {"name": "Pollo", "min_per_7d": 6, "max_per_7d": 2},   # invertidos ⇒ se ordenan
        "no-es-un-dict", {"slots": ["cena"]},                    # sin nombre ⇒ se ignora
    ]))
    by = {a["name"]: a for a in pol["food_anchors"]}
    assert by["Huevo"]["slots"] == [] and by["Huevo"]["preparation_mode"] == "vary_preparation"
    assert (by["Huevo"]["min_per_7d"], by["Huevo"]["max_per_7d"]) == (7, 7)      # 9 → 7; 'abc' → default 7
    assert (by["Pollo"]["min_per_7d"], by["Pollo"]["max_per_7d"]) == (2, 6)
    assert "Pollo" in by, "un ancla con detalle que no está en stapleFoods también cuenta"


def test_sin_campos_nuevos_la_politica_es_la_de_ayer():
    pol = pp.policy_from_form(_base_form())
    assert pol["source"]["form_version"] == "v1"
    assert pol["recurrence"]["global_mode"] == "balanced"
    assert pol["shopping"]["freezer_mode"] == "limited"
    assert pol["shopping"]["batch_cooking"] == "sometimes"
    assert pol["shopping"]["fresh_topup_days"] == 7, "quincenal sin respuesta ⇒ reposición semanal (default previo)"


def test_las_preguntas_4_a_6_llegan_al_compilador():
    pol = pp.policy_from_form(_base_form(mealOrganization="routine", freezerMode="none", freshTopup="no", batchCooking="often"))
    assert pol["recurrence"]["global_mode"] == "routine"
    assert pol["recurrence"]["slot_modes"] == {s: "routine" for s in pp.SLOTS}
    assert pol["shopping"] == {"main_cycle_days": 15, "fresh_topup_days": None, "freezer_mode": "none", "batch_cooking": "often"}
    assert pol["source"]["form_version"] == "v2"
    # y el compilador sigue aplicando la precedencia: sin congelador ni reposición, el ciclo baja a 7
    effective, rels = pp.compile_policy(pol, context={})
    assert effective["shopping"]["main_cycle_days"] == 7
    assert any(r["reason_code"] == "cycle_shortened_no_freezer_no_topup" for r in rels)


def test_valor_invalido_de_mealOrganization_no_rompe():
    pol = pp.policy_from_form(_base_form(mealOrganization="chaos"))
    assert pol["recurrence"]["global_mode"] == "balanced"


# ── 2. sello de fidelidad con `mode` ─────────────────────────────────────────
def test_el_sello_de_fidelidad_dice_si_el_motor_obedecio():
    src = (_BACKEND / "horizon.py").read_text(encoding="utf-8")
    i = src.index("def review_fidelity_gate(")
    body = src[i:i + 3500]
    assert 'report["mode"] = mode' in body
    assert body.index('mode = "enforce" if enforced else "shadow"') < body.index("plan[FIDELITY_REPORT_KEY] = report")


# ── 3. sink del embudo ───────────────────────────────────────────────────────
def test_wizard_funnel_persiste_solo_eventos_conocidos_con_sid_hasheado(monkeypatch):
    import routers.plans as rp
    import db_core
    rows = []
    monkeypatch.setattr(db_core, "execute_sql_write", lambda sql, params: rows.append((sql, params)))
    out = rp.api_wizard_funnel_telemetry(
        {"sid": "abc-123", "events": [
            {"event": "wizard_start", "index": 0, "total": 24, "app_mode": "plan", "policy_form": True},
            {"event": "hack_me", "index": 1},
            {"event": "step_view", "step_id": "mealOrganization", "field": "mealOrganization", "index": 12.0, "total": 24},
            "basura",
            {"event": "wizard_submit", "locale": "es-DO", "form_version": "v2", "secret": "no-debe-pasar"},
        ]},
        verified_user_id=None,
    )
    assert out == {"accepted": 3}
    assert len(rows) == 3
    sql, params = rows[0]
    assert "wizard_funnel" in sql and params[0] is None
    assert params[1] is not None and params[1] != "abc-123" and len(params[1]) == 16, "sid hasheado, nunca crudo"
    import json
    metas = [json.loads(p[2]) for _, p in rows]
    assert metas[0]["event"] == "wizard_start" and metas[0]["total"] == 24
    assert metas[1]["index"] == 12 and metas[1]["field"] == "mealOrganization"
    assert "secret" not in metas[2] and metas[2]["form_version"] == "v2"


def test_wizard_funnel_es_best_effort_y_sin_paywall(monkeypatch):
    import routers.plans as rp
    import db_core

    def _boom(sql, params):
        raise RuntimeError("db down")
    monkeypatch.setattr(db_core, "execute_sql_write", _boom)
    assert rp.api_wizard_funnel_telemetry({"sid": "x", "events": [{"event": "wizard_start"}]}, verified_user_id="u1") == {"accepted": 0}
    assert rp.api_wizard_funnel_telemetry({"events": "no-lista"}, verified_user_id=None) == {"accepted": 0}
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    i = src.index('@router.post("/telemetry/wizard")')
    assert "Depends(_WIZARD_TELEMETRY_LIMITER)" in src[i:i + 400]
    assert "verify_api_quota" not in src[i:i + 400]
    assert "_WIZARD_TELEMETRY_LIMITER = RateLimiter(" in src


# ── 4. contrato frontend↔backend ─────────────────────────────────────────────
def _front_list(js: str, name: str) -> list:
    m = re.search(r"export const " + name + r"\s*=\s*\[([^\]]*)\]", js)
    assert m, name
    return re.findall(r"'([a-z_]+)'", m.group(1))


@pytest.mark.skipif(not _FRONT_CFG.exists(), reason="frontend no disponible en este checkout")
def test_el_frontend_repite_los_enums_del_compilador():
    js = _FRONT_CFG.read_text(encoding="utf-8")
    assert _front_list(js, "MEAL_ORGANIZATION_MODES") == list(pp.RECURRENCE_MODES)
    assert _front_list(js, "FREEZER_MODES") == list(pp.FREEZER_MODES)
    assert _front_list(js, "BATCH_MODES") == list(pp.BATCH_MODES)
    assert _front_list(js, "ANCHOR_SLOTS") == list(pp.SLOTS)
    assert _front_list(js, "PREPARATION_MODES") == list(pp.PREPARATION_MODES)
    assert f"POLICY_SCHEMA_VERSION = {pp.POLICY_SCHEMA_VERSION};" in js
    assert set(_front_list(js, "RELAXATION_REASON_CODES")) == set(pp._REASON_COPY), "cada reason code tiene copy en el frontend"
    for code in pp._REASON_COPY:
        assert f"case '{code}':" in js, code
    # las franjas que el frontend ofrece son alias que el backend resuelve
    for slot in _front_list(js, "ANCHOR_SLOTS"):
        assert pp._SLOT_ALIASES.get(slot) == slot


@pytest.mark.skipif(not _FRONT_CFG.exists(), reason="frontend no disponible en este checkout")
def test_el_wizard_escribe_los_campos_que_el_adapter_lee():
    front = _BACKEND.parent / "frontend" / "src"
    flow = (front / "components" / "assessment" / "InteractiveAssessmentFlow.jsx").read_text(encoding="utf-8")
    ctx = (front / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
    for field in pp.FORM_V2_FIELDS:
        assert re.search(field + r"\s*:", ctx), f"{field} sin default en initialFormData"
    assert "fields: ['mealOrganization']" in flow, "la pregunta 1 es obligatoria en el wizard"
    assert "<QShoppingHabits" in flow and "<QMealOrganization" in flow
    assert "PLAN_POLICY_FORM_UI" in flow, "las preguntas nuevas van detrás del knob VITE_PLAN_POLICY_FORM"
    staples = (front / "components" / "assessment" / "questions" / "QStapleFoods.jsx").read_text(encoding="utf-8")
    assert "stapleAnchors" in staples, "«Mis básicos» es el editor de anclas"


def test_marker_bumpeado():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert '_LAST_KNOWN_PFIX = "P1-ARQ25-F4-FORM · 2026-09-03"' in app or "P1-ARQ25-F4-FORM · 2026-09-03" in app
