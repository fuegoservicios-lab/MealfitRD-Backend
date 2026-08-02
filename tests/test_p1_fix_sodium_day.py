"""[P1-FIX-SODIUM-DAY · 2026-08-02] Botón "Arreglar este día" — puente de un clic entre el
banner de aviso de sodio (`micro_worst_day_ceiling`, P2-PANEL-SOFT-REJECT) y el swap
sodio-consciente ya desplegado (P1-SODIUM-AWARE-PLACEMENT).

Caso real que lo motiva: banner "1 de 3 días se pasa del techo (peor: Día 1)" con
ricotta+camarones; el usuario tuvo que ADIVINAR qué plato cambiar (y cambió el de otro día).

Endpoint: `POST /api/plans/{plan_id}/fix-sodium-day` (routers/plans.py::api_fix_sodium_day).

Cobertura:
  (a) ownership/I2 — parser sobre el SELECT/persist del handler.
  (b) funcional con mocks — día sobre techo → elige el meal de MÁS sodio → swap mock OK →
      persiste vía `api_swap_meal_persist` IN-PROCESS → responde sodio antes/después
      RELEÍDO post-persist (no calculado).
  (c) funcional — ningún día sobre techo → `no_day_over_ceiling` sin tocar nada (cero
      llamadas a swap_meal/persist/log_api_usage).
  (d) funcional — swap agota intentos → `fixed=false` + `error_code` del chef, plan
      intacto (persist/log_api_usage NUNCA se llaman).
  (e) gating espejado a `/swap-meal` (parser) — mismo par `verify_api_quota` + RateLimiter
      propio.
  (f) marker anchor.
  (g) [P1-FIX-SODIUM-DAY-HONEST] `micro_worst_day_ceiling` NO es sodio-exclusivo — si el
      `high` del peor día persistido (`micronutrient_report.per_day_ceilings`) excluye
      sodio → `ceiling_not_sodium` sin tocar nada (ni swap, ni persist, ni cobro); si lo
      incluye, el flujo de arreglo sigue igual; si el SSOT no marca nada, cae al chequeo
      de sodio existente sin bloquear.

Harness: los mismos boundaries que el resto de la suite de swap monkeypatchea
(`db_core.execute_sql_query`, `db.get_user_profile`, `routers.plans.swap_meal`,
`routers.plans.api_swap_meal_persist`, `routers.plans.log_api_usage`) — `_meal_sodium_mg`
y `_sodium_day_ceiling_mg_for_banner` (graph_orchestrator) se sustituyen por un estimador
sintético basado en tabla de sodio por ingrediente (evita depender del catálogo real /
Postgres, igual que el resto de tests de swap del repo).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import db
import db_core
import graph_orchestrator as go
import routers.plans as _rp

_BACKEND = Path(__file__).resolve().parent.parent
_PLANS_SRC = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_APP_SRC = (_BACKEND / "app.py").read_text(encoding="utf-8")


def _extract_function_body(src: str, fn_name: str) -> str:
    m = re.search(rf"def\s+{re.escape(fn_name)}\s*\(", src)
    assert m, f"No se encontró `def {fn_name}(` en plans.py"
    start = m.start()
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def endpoint_body() -> str:
    return _extract_function_body(_PLANS_SRC, "api_fix_sodium_day")


# ---------------------------------------------------------------------------
# (a) Ownership / I2 — parser
# ---------------------------------------------------------------------------
def test_i2_select_filters_user_id(endpoint_body):
    assert endpoint_body.count("WHERE id = %s AND user_id = %s") >= 2, (
        "el SELECT inicial Y el re-read post-persist deben filtrar AND user_id = %s (I2)"
    )


def test_guest_rejected_before_any_db_read(endpoint_body):
    i_guard = endpoint_body.find("if not verified_user_id:")
    i_select = endpoint_body.find("SELECT plan_data FROM meal_plans")
    assert -1 not in (i_guard, i_select)
    assert i_guard < i_select, "el 401 de guest debe preceder cualquier lectura de DB"
    assert "status_code=401" in endpoint_body[i_guard:i_guard + 200]


def test_persist_call_passes_verified_user_id(endpoint_body):
    i = endpoint_body.find("api_swap_meal_persist(")
    assert i != -1, "debe reusar api_swap_meal_persist IN-PROCESS (no duplicar el mutator)"
    window = endpoint_body[i:i + 300]
    assert "verified_user_id=verified_user_id" in window


# ---------------------------------------------------------------------------
# (e) Gating espejado a /swap-meal — parser
# ---------------------------------------------------------------------------
def test_gating_mirrors_swap_meal():
    assert "_FIX_SODIUM_DAY_LIMITER = RateLimiter(max_calls=12, period_seconds=60)" in _PLANS_SRC
    i = _PLANS_SRC.find('@router.post("/{plan_id}/fix-sodium-day")')
    assert i != -1
    sig = _PLANS_SRC[i:i + 400]
    assert "Depends(verify_api_quota)" in sig, "mismo paywall que /swap-meal (cobrado post-éxito)"
    assert "Depends(_FIX_SODIUM_DAY_LIMITER)" in sig


def test_charges_credit_only_after_swap_success(endpoint_body):
    """Espejo de P2-SWAP-CHARGE-ON-SUCCESS: log_api_usage debe ocurrir DESPUÉS de
    `swap_meal(meal_form)`, nunca antes (si el LLM falla, no se cobra)."""
    i_swap = endpoint_body.find("new_meal = swap_meal(meal_form)")
    i_charge = endpoint_body.find('log_api_usage(verified_user_id, "llm_fix_sodium_day")')
    assert -1 not in (i_swap, i_charge)
    assert i_swap < i_charge


# ---------------------------------------------------------------------------
# (f) Marker anchor
# ---------------------------------------------------------------------------
def test_marker_anchored():
    assert _PLANS_SRC.count("P1-FIX-SODIUM-DAY") >= 3
    assert '_LAST_KNOWN_PFIX = "P1-FIX-SODIUM-DAY · 2026-08-02"' in _APP_SRC


# ---------------------------------------------------------------------------
# Helper puro `_worst_sodium_day_and_meal` — sin mocks de infraestructura
# ---------------------------------------------------------------------------
def test_worst_day_and_meal_picks_max_sodium_meal_in_worst_day():
    sodium_map = {"A": 100, "B": 900, "C": 50, "D": 1200}
    days = [
        {"meals": [{"name": "A"}, {"name": "C"}]},              # total 150 — bajo el techo
        {"meals": [{"name": "A"}, {"name": "B"}]},               # total 1000 — sobre el techo (500)
        {"meals": [{"name": "D"}]},                               # total 1200 — PEOR día
    ]
    found = _rp._worst_sodium_day_and_meal(days, 500, lambda m: sodium_map[m["name"]])
    assert found == {
        "day_index": 2, "meal_index": 0, "day_sodium_mg": 1200.0, "meal_sodium_mg": 1200.0,
    }


def test_worst_day_and_meal_none_when_all_under_ceiling():
    days = [{"meals": [{"name": "A"}]}, {"meals": [{"name": "B"}]}]
    assert _rp._worst_sodium_day_and_meal(days, 2000, lambda m: 100.0) is None


def test_worst_day_and_meal_none_for_empty_days():
    assert _rp._worst_sodium_day_and_meal([], 2000, lambda m: 999.0) is None
    assert _rp._worst_sodium_day_and_meal(None, 2000, lambda m: 999.0) is None


def test_worst_meal_picks_highest_sodium_within_the_worst_day():
    """Dentro del peor día, la comida elegida es la de MÁS sodio — no la primera,
    no la de más kcal (caso real: ricotta+camarones era la CENA, no el almuerzo)."""
    sodium_map = {"Almuerzo": 300, "Cena": 1100}
    days = [{"meals": [{"name": "Almuerzo"}, {"name": "Cena"}]}]
    found = _rp._worst_sodium_day_and_meal(days, 1000, lambda m: sodium_map[m["name"]])
    assert found["meal_index"] == 1
    assert found["meal_sodium_mg"] == 1100.0


# ---------------------------------------------------------------------------
# Estimador sintético de sodio (tabla por ingrediente) — evita depender del
# catálogo real / Postgres, mismo criterio que `_meal_sodium_mg` (prefiere
# `ingredients_raw`, cae a `ingredients`).
# ---------------------------------------------------------------------------
_NA_TABLE = {
    "avena": 50.0, "pollo": 120.0, "carne": 300.0,
    "camaron": 600.0, "ricotta": 500.0,
}


def _fake_meal_sodium(meal, _db):
    if not isinstance(meal, dict):
        return 0.0
    total = 0.0
    ings = meal.get("ingredients_raw") if isinstance(meal.get("ingredients_raw"), list) else meal.get("ingredients")
    for s in (ings or []):
        low = str(s).lower()
        for token, mg in _NA_TABLE.items():
            if token in low:
                total += mg
    return total


def _build_plan_data():
    return {
        "days": [
            {"meals": [
                {"name": "Avena con Fruta", "cals": 300, "protein": 10, "carbs": 40, "fats": 8,
                 "meal": "Desayuno", "ingredients": ["100g Avena"], "ingredients_raw": ["100g Avena"]},
                {"name": "Pollo al Horno", "cals": 400, "protein": 35, "carbs": 20, "fats": 10,
                 "meal": "Almuerzo", "ingredients": ["150g Pollo"], "ingredients_raw": ["150g Pollo"]},
            ]},  # total 170 — bajo el techo (1000)
            {"meals": [
                {"name": "Sancocho de Carne", "cals": 500, "protein": 30, "carbs": 45, "fats": 18,
                 "meal": "Almuerzo", "ingredients": ["300g Carne de res"],
                 "ingredients_raw": ["300g Carne de res"]},
                {"name": "Ricotta con Camarones", "cals": 480, "protein": 32, "carbs": 15, "fats": 22,
                 "meal": "Cena", "ingredients": ["150g Camarones", "100g Ricotta"],
                 "ingredients_raw": ["150g Camarones", "100g Ricotta"]},
            ]},  # total 1400 — PEOR día (300 + 1100)
        ],
    }


class _FakeSqlState:
    """Backing store mutable compartido entre el fake `execute_sql_query` y el fake
    `api_swap_meal_persist` — simula el UPDATE atómico sin tocar Postgres real."""
    def __init__(self, plan_data):
        self.plan_data = plan_data
        self.select_calls = 0


@pytest.fixture
def sql_state():
    return _FakeSqlState(_build_plan_data())


@pytest.fixture
def patched_infra(monkeypatch, sql_state):
    """Monkeypatchea los mismos boundaries que el resto de la suite de swap:
    DB (`db_core.execute_sql_query`, `db.get_user_profile`), el estimador de sodio
    (`graph_orchestrator._meal_sodium_mg` / `_sodium_day_ceiling_mg_for_banner`) y
    la contabilidad (`routers.plans.log_api_usage`). `swap_meal` / `api_swap_meal_persist`
    se parchean POR TEST (varían según el escenario)."""
    import copy as _copy

    def _fake_execute_sql_query(query, params=None, fetch_one=False, fetch_all=False):
        assert "meal_plans" in query
        assert params[0] == "plan-fix-sodium-1" and params[1] == "user-1"
        sql_state.select_calls += 1
        return {"plan_data": _copy.deepcopy(sql_state.plan_data)}

    monkeypatch.setattr(db_core, "execute_sql_query", _fake_execute_sql_query)
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {
        "health_profile": {"gender": "female", "age": 30, "allergies": [], "dietType": "balanced"}
    })
    monkeypatch.setattr(go, "_meal_sodium_mg", _fake_meal_sodium)
    monkeypatch.setattr(go, "_sodium_day_ceiling_mg_for_banner", lambda form_data=None: 1000.0)
    monkeypatch.setattr(_rp, "log_api_usage", lambda *a, **k: None)
    return sql_state


def _call_endpoint(**kwargs):
    return _rp.api_fix_sodium_day(
        "plan-fix-sodium-1", data={}, verified_user_id="user-1", _rl=None, **kwargs
    )


# ---------------------------------------------------------------------------
# (b) Funcional: día sobre techo → elige el meal de más sodio → swap OK → persiste
# ---------------------------------------------------------------------------
def test_happy_path_fixes_worst_meal_of_worst_day(monkeypatch, patched_infra):
    captured_meal_form = {}
    persist_calls = []

    def _fake_swap_meal(meal_form):
        captured_meal_form.update(meal_form)
        return {
            "name": "Camarones al Ajillo con Vegetales Frescos",
            "desc": "Camarones salteados con ajo y vegetales.",
            "cals": 450, "prep_time": 20,
            "recipe": ["Saltea los camarones con ajo y aceite de oliva."],
            "ingredients": ["150g Camarones frescos", "1 cda Aceite de oliva"],
            "ingredients_raw": ["150g Camarones frescos", "1 cda Aceite de oliva"],
        }

    def _fake_persist(plan_id, body, verified_user_id=None):
        persist_calls.append((plan_id, body, verified_user_id))
        d = patched_infra.plan_data["days"][body["day_index"]]
        d["meals"][body["meal_index"]] = body["new_meal"]
        return {"success": True}

    monkeypatch.setattr(_rp, "swap_meal", _fake_swap_meal)
    monkeypatch.setattr(_rp, "api_swap_meal_persist", _fake_persist)

    result = _call_endpoint()

    # Eligió el DÍA correcto (1, el de 1400mg) y el MEAL correcto (1, Ricotta+Camarones
    # con 1100mg > Sancocho con 300mg) — no el primer día, no el primer meal.
    assert captured_meal_form["rejected_meal"] == "Ricotta con Camarones"
    assert captured_meal_form["meal_type"] == "Cena"
    assert captured_meal_form["swap_reason"] == "high_sodium"
    # sodium_resto_override_mg = sodio del día (1400) menos el meal reemplazado (1100) = 300
    # (el Sancocho de Carne, la OTRA comida del día) — activa P1-SODIUM-AWARE-PLACEMENT.
    assert captured_meal_form["sodium_resto_override_mg"] == 300.0

    assert len(persist_calls) == 1
    plan_id, body, uid = persist_calls[0]
    assert plan_id == "plan-fix-sodium-1"
    assert body["day_index"] == 1 and body["meal_index"] == 1
    assert uid == "user-1"
    assert body["new_meal"]["name"] == "Camarones al Ajillo con Vegetales Frescos"
    assert body["new_meal"]["isExpanded"] is False

    assert result["fixed"] is True
    assert result["day"] == 1
    assert result["old_meal"] == "Ricotta con Camarones"
    assert result["new_meal"] == "Camarones al Ajillo con Vegetales Frescos"
    assert result["sodio_antes_mg"] == 1400
    # post-persist: Sancocho (300) + el plato nuevo (solo "camaron" matchea → 600) = 900
    assert result["sodio_despues_mg"] == 900
    assert result["day_under_ceiling"] is True

    # Honestidad: el sodio "después" viene de RELEER el plan post-persist (2 SELECTs:
    # inicial + re-read), no de sumar el resultado crudo del swap.
    assert patched_infra.select_calls == 2


# ---------------------------------------------------------------------------
# (c) Funcional: ningún día sobre techo → no_day_over_ceiling, cero side-effects
# ---------------------------------------------------------------------------
def test_no_day_over_ceiling_touches_nothing(monkeypatch, patched_infra):
    monkeypatch.setattr(go, "_sodium_day_ceiling_mg_for_banner", lambda form_data=None: 5000.0)

    swap_called = []
    persist_called = []
    monkeypatch.setattr(_rp, "swap_meal", lambda mf: swap_called.append(mf))
    monkeypatch.setattr(_rp, "api_swap_meal_persist", lambda *a, **k: persist_called.append(1))

    result = _call_endpoint()

    assert result == {
        "fixed": False,
        "code": "no_day_over_ceiling",
        "message": (
            "Ningún día de tu plan está sobre el techo de sodio ahora mismo — quizá el "
            "panel está por refrescar."
        ),
    }
    assert swap_called == []
    assert persist_called == []
    # Un solo SELECT (la lectura inicial) — el soft no-op no re-lee ni persiste nada.
    assert patched_infra.select_calls == 1


# ---------------------------------------------------------------------------
# (d) Funcional: el chef agota intentos → fixed=false + error_code, plan INTACTO
# ---------------------------------------------------------------------------
def test_swap_retries_exhausted_leaves_plan_intact(monkeypatch, patched_infra):
    persist_called = []
    charge_called = []
    monkeypatch.setattr(_rp, "swap_meal", lambda mf: (_ for _ in ()).throw(
        ValueError("SWAP_LLM_RETRIES_EXHAUSTED: el chef IA no pudo generar una alternativa.")
    ))
    monkeypatch.setattr(_rp, "api_swap_meal_persist", lambda *a, **k: persist_called.append(1))
    monkeypatch.setattr(_rp, "log_api_usage", lambda *a, **k: charge_called.append(1))

    result = _call_endpoint()

    assert result["fixed"] is False
    assert result["day"] == 1
    assert result["old_meal"] == "Ricotta con Camarones"
    assert result["error_code"] == "swap_llm_retries_exhausted"
    assert "crédito" in result["error_message"]

    assert persist_called == [], "el plan debe quedar INTACTO — el persist nunca corre"
    assert charge_called == [], "no se descuenta crédito si el swap falló"
    # Solo el SELECT inicial — el fallo del chef ocurre ANTES de cualquier re-read.
    assert patched_infra.select_calls == 1
    # El plan_data compartido no fue mutado (nadie más que _fake_persist lo tocaría).
    assert patched_infra.plan_data["days"][1]["meals"][1]["name"] == "Ricotta con Camarones"


def test_swap_clinical_violation_soft_fails(monkeypatch, patched_infra):
    monkeypatch.setattr(_rp, "swap_meal", lambda mf: (_ for _ in ()).throw(
        ValueError("CLINICAL_VIOLATION: alérgeno detectado")
    ))
    persist_called = []
    monkeypatch.setattr(_rp, "api_swap_meal_persist", lambda *a, **k: persist_called.append(1))

    result = _call_endpoint()
    assert result["fixed"] is False
    assert result["error_code"] == "swap_clinical_violation"
    assert persist_called == []


def test_swap_ai_unavailable_soft_fails_without_charge(monkeypatch, patched_infra):
    from agent import LLMCircuitBreakerOpen

    def _raise_cb(mf):
        raise LLMCircuitBreakerOpen("breaker open")

    monkeypatch.setattr(_rp, "swap_meal", _raise_cb)
    charge_called = []
    monkeypatch.setattr(_rp, "log_api_usage", lambda *a, **k: charge_called.append(1))

    result = _call_endpoint()
    assert result["fixed"] is False
    assert result["error_code"] == "swap_ai_unavailable"
    assert charge_called == []


# ---------------------------------------------------------------------------
# [P1-FIX-SODIUM-DAY-HONEST · 2026-08-02] `micro_worst_day_ceiling` NO es sodio-exclusivo:
# `per_day_ceilings.worst_day.high` puede listar free_sugars_g/saturated_fat_g/potassium_mg
# (dyslipidemia/renal) SIN sodio. El botón solo debe "arreglar" cuando sodio es de verdad
# parte del problema del PEOR día — leído de la MISMA fuente persistida que ya leyó
# `_maybe_mark_panel_degraded` (`plan['micronutrient_report']`), no un recompute nuevo.
# ---------------------------------------------------------------------------
def test_ceiling_not_sodium_when_worst_day_high_excludes_sodium(monkeypatch, patched_infra):
    """Día 1 (índice 1) SÍ mide 1400mg de sodio con la tabla sintética (sobre el techo de
    1000) — sin el gate, el flujo actual lo 'arreglaría'. Pero el `micronutrient_report`
    persistido dice que el techo roto de ese día es `free_sugars_g`, NO sodio → debe
    responder `ceiling_not_sodium` sin tocar nada (ni swap, ni persist, ni cobro)."""
    patched_infra.plan_data["micronutrient_report"] = {
        "per_day_ceilings": {
            "flagged": True,
            "worst_day": {"day_index": 1, "high": ["free_sugars_g"]},
        },
    }
    swap_called = []
    persist_called = []
    charge_called = []
    monkeypatch.setattr(_rp, "swap_meal", lambda mf: swap_called.append(mf))
    monkeypatch.setattr(_rp, "api_swap_meal_persist", lambda *a, **k: persist_called.append(1))
    monkeypatch.setattr(_rp, "log_api_usage", lambda *a, **k: charge_called.append(1))

    result = _call_endpoint()

    assert result["fixed"] is False
    assert result["code"] == "ceiling_not_sodium"
    assert result["nutrients"] == ["free_sugars_g"]
    assert "Azúcares añadidos" in result["message"]
    assert "no por sodio" in result["message"]
    assert swap_called == [], "el chef NUNCA debe invocarse — el problema no es sodio"
    assert persist_called == [], "el plan debe quedar INTACTO"
    assert charge_called == [], "no se cobra crédito por un no-op"
    # Un solo SELECT — el gate honesto corta ANTES de cualquier re-lectura.
    assert patched_infra.select_calls == 1


def test_ceiling_not_sodium_translates_multiple_nutrients_es_do(monkeypatch, patched_infra):
    """Espejo del caso renal+dyslipidemia real: potasio (cap renal) y grasa saturada
    (dyslipidemia) pueden co-ocurrir sin sodio. Las etiquetas vienen del MISMO `_LABELS`
    que usa el panel — cero tabla de traducción duplicada."""
    patched_infra.plan_data["micronutrient_report"] = {
        "per_day_ceilings": {
            "flagged": True,
            "worst_day": {"day_index": 1, "high": ["potassium_mg", "saturated_fat_g"]},
        },
    }
    monkeypatch.setattr(_rp, "swap_meal", lambda mf: pytest.fail("no debe invocarse"))

    result = _call_endpoint()

    assert result["code"] == "ceiling_not_sodium"
    assert result["nutrients"] == ["potassium_mg", "saturated_fat_g"]
    assert "Potasio" in result["message"] and "Grasa saturada" in result["message"]


def test_sodium_in_high_list_still_proceeds_to_fix(monkeypatch, patched_infra):
    """Cuando sodio SÍ está en el `high` del peor día (solo o acompañado de otros
    nutrientes), el flujo de arreglo sigue exactamente igual que antes del gate honesto."""
    patched_infra.plan_data["micronutrient_report"] = {
        "per_day_ceilings": {
            "flagged": True,
            "worst_day": {"day_index": 1, "high": ["sodium_mg", "free_sugars_g"]},
        },
    }
    persist_calls = []

    def _fake_swap_meal(meal_form):
        return {
            "name": "Camarones al Ajillo con Vegetales Frescos",
            "cals": 450, "prep_time": 20,
            "recipe": ["Saltea los camarones con ajo y aceite de oliva."],
            "ingredients": ["150g Camarones frescos"],
            "ingredients_raw": ["150g Camarones frescos"],
        }

    def _fake_persist(plan_id, body, verified_user_id=None):
        persist_calls.append(body)
        d = patched_infra.plan_data["days"][body["day_index"]]
        d["meals"][body["meal_index"]] = body["new_meal"]
        return {"success": True}

    monkeypatch.setattr(_rp, "swap_meal", _fake_swap_meal)
    monkeypatch.setattr(_rp, "api_swap_meal_persist", _fake_persist)

    result = _call_endpoint()

    assert result["fixed"] is True
    assert result["day"] == 1
    assert len(persist_calls) == 1


def test_gate_is_noop_when_worst_day_not_flagged(monkeypatch, patched_infra):
    """`per_day_ceilings.flagged=False` (o ausente) → el gate honesto no bloquea nada; el
    flujo cae al chequeo de sodio existente (backward-compat con planes sin el reporte, o
    con el reporte diciendo que hoy no hay ningún techo roto)."""
    patched_infra.plan_data["micronutrient_report"] = {
        "per_day_ceilings": {"flagged": False, "worst_day": {"day_index": 0, "high": []}},
    }
    persist_calls = []
    monkeypatch.setattr(_rp, "swap_meal", lambda mf: {
        "name": "Nuevo plato", "cals": 450, "recipe": [], "ingredients": [], "ingredients_raw": [],
    })
    monkeypatch.setattr(_rp, "api_swap_meal_persist", lambda plan_id, body, verified_user_id=None: (
        persist_calls.append(body) or {"success": True}
    ))

    result = _call_endpoint()
    assert result["fixed"] is True, "el gate no debe bloquear cuando el SSOT no marca nada"
    assert len(persist_calls) == 1


def test_no_day_over_ceiling_message_scoped_to_sodium_now(monkeypatch, patched_infra):
    """El mensaje del no-op honesto (P1-FIX-SODIUM-DAY-HONEST) habla del techo de sodio
    AHORA MISMO ('ningún día... está sobre el techo de sodio ahora mismo'), sin implicar
    que sodio fuera la única causa posible del banner original."""
    monkeypatch.setattr(go, "_sodium_day_ceiling_mg_for_banner", lambda form_data=None: 5000.0)
    result = _call_endpoint()
    assert result["code"] == "no_day_over_ceiling"
    assert "techo de sodio ahora mismo" in result["message"]
