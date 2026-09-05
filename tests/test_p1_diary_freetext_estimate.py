"""[P1-DIARY-FREETEXT-ESTIMATE · 2026-09-04] Escribir el plato y que la app estime; y que los
desvíos declarados («comí otra cosa» / «todavía no») los lean el coach y el siguiente bloque.

Tres huecos del mismo día: (1) el componedor aceptaba lo que el catálogo no conoce SOLO si la
persona tecleaba las cuatro macros — nadie sabe cuánta proteína tiene «un mangú con huevo
frito»; (2) los desvíos que P1-EAT-PLAN-MEAL-TRUTH empezó a escribir en `pipeline_metrics` no
los leía nadie: el coach afirmaba que comiste el almuerzo del plan; (3) el aprendizaje del
siguiente bloque contaba ese día como «0 registros ⇒ 100 % consumido».

Invariantes: la estimación NO persiste (vuelve como borrador; el registro sigue por la vía
`custom` de `/consumed/manual`, misma clamp, sin resta de Nevera); su costo va a
`llm_usage_events` (node=diary_freetext_estimate), nunca a `api_usage`; filas propias (I2).
"""
from __future__ import annotations

import asyncio

import re
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_DIARY = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")


def _handler(src: str, name: str) -> str:
    start = src.index(f"def {name}(")
    end = src.find("\n@router.", start)
    return src[start:end if end != -1 else len(src)]


# ------------------------------------------------------------------ estimador

def test_estimate_endpoint_is_quota_exempt_flash_and_soft_fail():
    body = _handler(_DIARY, "api_estimate_macros")
    assert '@router.post("/consumed/estimate-macros")\nasync def api_estimate_macros(' in _DIARY
    assert "verify_api_quota" not in body and "log_api_usage" not in body
    assert "_ESTIMATE_MACROS_LIMITER" in body
    # flash por knob (no hardcode), estructurado como el resto del motor, costo por el mixin async
    assert "_plan_flash_model_name()" in body
    assert "with_structured_output(\n        MacroEstimateModel, method=\"json_mode\"\n    )" in body
    assert '_current_node_var.set("diary_freetext_estimate")' in body
    assert "await asyncio.wait_for(" in body and "llm.ainvoke(" in body
    # soft-fail 200 + error_code (P3-SWAP-SOFT-FAIL-200): que el modelo no conteste es normal
    assert '"error_code": "estimate_unavailable"' in body
    # misma clamp que la vía custom del manual: el estimador no abre una segunda frontera
    assert "from food_search import _clamp_macros" in body
    # y NO persiste: cero INSERT
    assert "INSERT" not in body and "_persist_consumed_meal" not in body


def test_estimate_request_bounds_and_output_model():
    from routers.diary import EstimateMacrosRequest, MacroEstimateModel
    with pytest.raises(Exception):
        EstimateMacrosRequest(text="ab")
    with pytest.raises(Exception):
        EstimateMacrosRequest(text="x" * 201)
    req = EstimateMacrosRequest(text="mangú con huevo frito", meal_type="desayuno")
    assert req.meal_type == "desayuno"
    m = MacroEstimateModel(name="Mangú con huevo", calories=520, protein=18, carbs=70, healthy_fats=18)
    assert m.portion_note == ""
    with pytest.raises(Exception):
        MacroEstimateModel(name="x", calories=-1, protein=0, carbs=0, healthy_fats=0)


# [P2-ASYNC-TESTS-SIN-PLUGIN · 2026-09-05] Estos tests eran `async def` con `@pytest.mark.asyncio`,
# pero `pytest-asyncio` NO está instalado ni declarado en `requirements.txt` — ni aquí ni en el VPS ni
# en el workflow de CI. pytest no los saltaba: los marcaba FAILED («async def functions are not
# natively supported»), así que el guard llevaba vivo lo justo para ensuciar el resultado y nada más.
# Se conducen con `asyncio.run` desde un test síncrono: cero dependencias nuevas (y añadir una de test
# a `requirements.txt`, que es el fichero de PRODUCCIÓN, sería peor).
def test_estimate_returns_clamped_draft_marked_estimated(monkeypatch):
    asyncio.run(_test_estimate_returns_clamped_draft_marked_estimated(monkeypatch))


async def _test_estimate_returns_clamped_draft_marked_estimated(monkeypatch):
    import routers.diary as diary
    import graph_orchestrator as go

    class _FakeLLM:
        def with_structured_output(self, model, method=None):
            assert model is diary.MacroEstimateModel and method == "json_mode"
            return self

        async def ainvoke(self, msgs):
            assert go._current_node_var.get() == "diary_freetext_estimate"
            return diary.MacroEstimateModel(name="  Mangú con huevo frito ", calories=4900.4, protein=20, carbs=70,
                                            healthy_fats=18, portion_note="1 plato (~350 g)")

    monkeypatch.setattr(go, "ChatGLM", lambda **kw: _FakeLLM())
    monkeypatch.setattr(go, "_plan_flash_model_name", lambda: "glm-test-flash")
    out = await diary.api_estimate_macros(diary.EstimateMacrosRequest(text="mangú  con huevo   frito"), verified_user_id="u" * 36)
    assert out["estimated"] is True and out["model"] == "glm-test-flash"
    assert out["name"] == "Mangú con huevo frito"
    # el modelo acota (le=5000) y food_search clampa (misma frontera que la vía custom)
    assert out["macros"] == {"kcal": 4900.4, "protein": 20.0, "carbs": 70.0, "fats": 18.0}
    assert out["portion_note"] == "1 plato (~350 g)"
    assert go._current_node_var.get() is None  # contexto restaurado


def test_estimate_soft_fails_when_model_raises(monkeypatch):
    asyncio.run(_test_estimate_soft_fails_when_model_raises(monkeypatch))


async def _test_estimate_soft_fails_when_model_raises(monkeypatch):
    import routers.diary as diary
    import graph_orchestrator as go

    class _Boom:
        def with_structured_output(self, *a, **k):
            return self

        async def ainvoke(self, msgs):
            raise RuntimeError("timeout")

    monkeypatch.setattr(go, "ChatGLM", lambda **kw: _Boom())
    out = await diary.api_estimate_macros(diary.EstimateMacrosRequest(text="pica pollo"), verified_user_id="u" * 36)
    assert out["operation_failed"] is True and out["error_code"] == "estimate_unavailable"


# ------------------------------------------------------------------ desvíos → coach

def test_deviations_block_includes_today_and_never_affirms():
    from chat_history_context import build_plan_deviations_block
    today = date(2026, 9, 4)
    rows = [
        {"created_at": datetime(2026, 9, 4, 19, 58, tzinfo=timezone.utc), "reason": "ate_other",
         "meal_type": "almuerzo", "meal_name": "Tortitas de calamar", "local_hour": 15},
        {"created_at": datetime(2026, 9, 3, 13, 5, tzinfo=timezone.utc), "reason": "not_yet",
         "meal_type": "desayuno", "meal_name": "Avena con guineo", "local_hour": 9},
        {"created_at": datetime(2026, 8, 20, 13, 5, tzinfo=timezone.utc), "reason": "ate_other",
         "meal_type": "cena", "meal_name": "Viejo", "local_hour": 20},
    ]
    block = build_plan_deviations_block(rows, today, days_back=3, tz_offset_mins=240)
    assert "NO HABER COMIDO" in block
    assert "- HOY: almuerzo: «Tortitas de calamar» → comió OTRA COSA" in block
    assert "desayuno: «Avena con guineo» → todavía no lo había comido (a las 09h)" in block
    assert "Viejo" not in block  # fuera de la ventana
    assert build_plan_deviations_block([], today) == ""


def test_agent_reads_deviations_after_the_diary_block():
    i_diary = _AGENT.index("out += build_past_diary_block(rows, today, days_back=days_back, tz_offset_mins=tz_offset_mins)")
    i_dev = _AGENT.index("out += build_plan_deviations_block(devs, today, days_back=days_back, tz_offset_mins=tz_offset_mins)")
    assert i_diary < i_dev
    assert "get_plan_meal_deviations_since(user_id, since)" in _AGENT


def test_get_plan_meal_deviations_is_user_scoped_and_fail_open(monkeypatch):
    import db_facts
    src = (_BACKEND / "db_facts.py").read_text(encoding="utf-8")
    body = src[src.index("def get_plan_meal_deviations_since("):src.index("def get_consumed_meals_since(")]
    assert "WHERE user_id = %s AND node = 'plan_meal_deviation'" in body
    monkeypatch.setattr(db_facts, "execute_sql_query", lambda *a, **k: [
        {"created_at": "2026-09-04T19:58:19Z", "metadata": '{"reason": "ate_other", "meal_type": "almuerzo", "meal_name": "Tortitas"}'},
    ])
    rows = db_facts.get_plan_meal_deviations_since("u", "2026-09-01")
    assert rows == [{"created_at": "2026-09-04T19:58:19Z", "reason": "ate_other", "meal_type": "almuerzo",
                     "meal_name": "Tortitas", "local_hour": None, "day_index": None, "plan_id": None}]
    monkeypatch.setattr(db_facts, "execute_sql_query", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db")))
    assert db_facts.get_plan_meal_deviations_since("u", "2026-09-01") == []


# ------------------------------------------------------------------ desvíos → aprendizaje

def test_learning_ratio_counts_deviations_as_explicit_signal():
    from cron_tasks import _calculate_chunk_consumption_ratio
    days = [{"meals": [{"name": "Mangú"}, {"name": "Moro"}, {"name": "Sancocho"}, {"name": "Avena"}]}]
    # sin registros ni desvíos: proxy «100 % consumido»
    base = _calculate_chunk_consumption_ratio(days, [], 0)
    assert base["zero_log_proxy"] is True
    # un desvío declarado deja de ser «cero registros»
    out = _calculate_chunk_consumption_ratio(days, [], 0, deviation_count=1)
    assert out["zero_log_proxy"] is False
    assert out["explicit_logged_meals"] == 1 and out["declared_deviations"] == 1
    assert out["explicit_matched_meals"] == 0
    # y el gate pasa el conteo real
    assert "deviation_count=_deviation_count)" in _CRON
    assert "_deviation_count = len(get_plan_meal_deviations_since(user_id, prev_start_iso) or [])" in _CRON


def test_marker_and_claudemd_row():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P1-DIARY-FREETEXT-ESTIMATE · 2026-09-04" in app
