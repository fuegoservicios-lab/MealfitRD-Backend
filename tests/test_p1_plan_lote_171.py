# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-171 · 2026-09-23] Generación de prueba del dueño («¿está el generador al 100 % para producción?»).

El plan salió (3 días en ~3 min, revisor aprobado), pero el log destapó dos regresiones del 2-sep, el día en que el
Bloque 1 empezó a nacer como PLACEHOLDER (`generation_status='generating'`, fila creada ANTES del pipeline):

1. SEGURIDAD CLÍNICA. `fill_placeholder_meal_plan_atomic` sacaba el `user_id` ANTES del escudo pre-INSERT, que arma
   el contexto clínico desde el perfil por ese `user_id`: recibía `{}`, el motor de macros omitía el re-cap DM2 /
   bariátrico y el panel de micros salía sin techos. Aviso `P1-UPDATE-CLINICAL-RECAP … sin form_data`: 33 veces desde
   el 2-sep, ninguna antes.
2. «¿Generó un plan HOY?» miraba la última fila de `meal_plans`, que ya era la del plan en curso: todo plan nuevo
   parecía «regenerar hoy». Con Nevera → «usa SOLO la despensa» aunque el formulario dijera «desde cero»; sin ella →
   re-roll a temperatura 0,95. Además, una renovación («Quiero variedad») del mismo día recibía la orden de rotación
   estricta mientras la guarda de despensa, por diseño, ignoraba esa despensa.

Y el dueño pidió actualizar la Luna: GPT-6 Luna (`gpt-6-luna`), mitad de precio. Probada contra la API real con la
clave del VPS: JSON, esfuerzo low/medium/none y streaming con usage → 200; `temperature=0.1` → 400. langchain-openai
1.3.0 solo quita la temperatura a `gpt-5*`, así que el filtro vive en `llm_provider.ChatOpenAI`.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import db_plans  # noqa: E402


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ─────────────────────────── 1. el escudo del relleno recibe al usuario ───────────────────────────

class _Cursor:
    def __init__(self):
        self.sql = []
        self._last = None
        self.rowcount = 1

    def execute(self, q, params=None):
        self.sql.append(" ".join(str(q).split()))
        self._last = {"plan_data": {"generation_status": "generating"}} if "select plan_data from meal_plans" in self.sql[-1].lower() else None

    def fetchone(self):
        return self._last

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Conn:
    def __init__(self, cur):
        self._cur = cur

    def cursor(self, **kw):
        return self._cur

    def transaction(self):
        return self._cur

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Pool:
    def __init__(self, cur):
        self._cur = cur

    def connection(self):
        return _Conn(self._cur)


@pytest.fixture
def relleno(monkeypatch):
    cur = _Cursor()
    vistos = []
    monkeypatch.setattr(db_plans, "connection_pool", _Pool(cur))
    monkeypatch.setattr(db_plans, "set_meal_plan_for_update_timeouts", lambda c: None)
    monkeypatch.setattr(db_plans, "acquire_meal_plan_advisory_lock", lambda c, p, purpose=None: None)
    monkeypatch.setattr(db_plans, "_apply_inherited_lifetime_lessons", lambda u, d, cursor=None: None)
    monkeypatch.setattr(db_plans, "_finalize_plan_data_for_insert", lambda d, **kw: vistos.append(d.get("user_id")))
    return cur, vistos


@pytest.mark.parametrize("con_user_id_en_insert_data", [False, True])
def test_el_escudo_del_relleno_recibe_el_user_id(relleno, con_user_id_en_insert_data):
    """Sin `user_id`, `build_clinical_form_from_profile` devuelve `{}` («no sé») y el re-cap clínico se omite."""
    cur, vistos = relleno
    data = {"plan_data": {"generation_status": "partial", "days": [{"meals": []}]}, "name": "Plan"}
    if con_user_id_en_insert_data:
        data["user_id"] = "user-1"
    out = db_plans.fill_placeholder_meal_plan_atomic("plan-1", "user-1", data)
    assert out == "plan-1"
    assert vistos == ["user-1"], "el escudo pre-INSERT del relleno corrió sin saber de quién es el plan"


def test_el_update_del_relleno_no_reescribe_el_user_id(relleno):
    cur, _ = relleno
    db_plans.fill_placeholder_meal_plan_atomic(
        "plan-1", "user-1", {"plan_data": {"generation_status": "partial", "days": [{"meals": []}]}, "user_id": "user-1"})
    update = next(s for s in cur.sql if s.lower().startswith("update meal_plans set"))
    sets = update.split(" WHERE ")[0]
    assert "user_id" not in sets, update
    assert update.endswith("WHERE id = %s AND user_id = %s"), "I2: el UPDATE sigue filtrando por dueño"


# ─────────────────────────── 2. «¿generó un plan hoy?» solo cuenta planes entregados ───────────────────────────

def _fila(creado, status):
    return {"created_at": creado, "status": status}


@pytest.fixture
def tabla(monkeypatch):
    """`execute_sql_query` falso que aplica el WHERE del SQL REAL que recibe (no uno que el test supone)."""
    filas = []
    visto = {}

    def fake(query, params=None, fetch_one=False, **kw):
        visto["sql"] = " ".join(query.split())
        excluye = "not in ('generating', 'failed')" in visto["sql"].lower()
        vivas = [f for f in filas if not (excluye and f["status"] in ("generating", "failed"))]
        vivas.sort(key=lambda f: f["created_at"], reverse=True)
        return {"created_at": vivas[0]["created_at"]} if vivas else None

    monkeypatch.setattr(db_plans, "connection_pool", object())
    monkeypatch.setattr(db_plans, "execute_sql_query", fake)
    return filas, visto


def test_el_placeholder_en_curso_no_cuenta_como_plan_de_hoy(tabla):
    """El caso vivo del 23-sep: plan anterior del 15-sep y el placeholder de ESTE plan creado hace 1 s."""
    filas, _ = tabla
    ahora = datetime.now(timezone.utc)
    filas += [_fila(ahora - timedelta(days=8), "partial"), _fila(ahora, "generating")]
    assert db_plans.check_meal_plan_generated_today("u") is False


def test_un_placeholder_agotado_tampoco(tabla):
    filas, _ = tabla
    ahora = datetime.now(timezone.utc)
    filas += [_fila(ahora - timedelta(days=3), "complete"), _fila(ahora, "failed")]
    assert db_plans.check_meal_plan_generated_today("u") is False


def test_un_plan_entregado_hoy_si_cuenta(tabla):
    """La semántica original no cambia: regenerar tras un plan ENTREGADO hoy sigue siendo «mismo día»."""
    filas, _ = tabla
    ahora = datetime.now(timezone.utc)
    filas += [_fila(ahora - timedelta(minutes=30), "partial"), _fila(ahora, "generating")]
    assert db_plans.check_meal_plan_generated_today("u") is True


def test_la_consulta_filtra_por_dueno_y_estado(tabla):
    _, visto = tabla
    db_plans.check_meal_plan_generated_today("u")
    sql = visto["sql"]
    assert "WHERE user_id = %s" in sql and "ORDER BY created_at DESC LIMIT 1" in sql
    assert "COALESCE(plan_data->>'generation_status', '') NOT IN ('generating', 'failed')" in sql


def test_una_renovacion_no_recibe_la_orden_de_solo_despensa():
    """La guarda de despensa ignora la nevera en una renovación; el prompt no puede ordenar lo contrario."""
    go = _src("graph_orchestrator.py")
    i = go.index("is_rotation = (not _rot_fridge_empty) and bool(")
    bloque = go[i:i + 1600]
    assert "if is_rotation and _is_renewal_reason_rot(actual_form_data.get(\"update_reason\")):" in bloque
    assert "is_rotation = False" in bloque
    assert bloque.index("is_rotation = False") < bloque.index("ROTACIÓN DE PLATOS")
    from horizon import is_renewal_reason
    assert is_renewal_reason("variety") and is_renewal_reason("renewal.v1")
    assert not is_renewal_reason("dislike"), "«No me gustan estos platos» es la rotación de verdad"
    assert not is_renewal_reason(None)


# ─────────────────────────── 3. GPT-6 Luna ───────────────────────────

def test_la_luna_por_defecto_es_gpt6():
    import llm_provider as lp
    assert lp.GPT6_LUNA == "gpt-6-luna"
    assert lp.GPT56_LUNA == "gpt-5.6-luna", "la 5.6 sigue existiendo para quien la fije por knob"
    go = _src("graph_orchestrator.py")
    for anc in ("_REVIEWER_RISK_MODEL_FREE_DEFAULT = GPT6_LUNA", "_DAYGEN_TIER_MODEL_PLUS_DEFAULT = GPT6_LUNA",
                "_DAYGEN_TIER_MODEL_FREE_DEFAULT = GPT6_LUNA", '_env_str("MEALFIT_PRO_MODEL", GPT6_LUNA) or GPT6_LUNA'):
        assert anc in go, anc
    assert "_SWAP_MODEL_DEFAULT = GPT6_LUNA" in _src("agent.py")


def test_gpt6_luna_tiene_precio_antes_de_recibir_trafico():
    from db_profiles import _DEFAULT_LLM_PRICING_MICROS_PER_M as TABLA, compute_llm_cost_micros
    assert TABLA["gpt-6-luna"] == {"input": 100_000, "output": 500_000, "cached": 10_000}
    assert TABLA["gpt-6-luna"]["input"] * 2 == TABLA["gpt-5.6-luna"]["input"], "mitad de precio que la 5.6"
    assert compute_llm_cost_micros("gpt-6-luna", 1000, 100, 0) is not None


def test_la_temperatura_no_sale_hacia_gpt6():
    """langchain-openai 1.3.0 solo la quita a `gpt-5*`; la API de gpt-6-luna responde 400 a `temperature=0.1`."""
    from langchain_core.messages import HumanMessage
    import llm_provider as lp
    msgs = [HumanMessage("hola")]
    p6 = lp.ChatOpenAI(model="gpt-6-luna", temperature=0.95, api_key="sk-fake", max_tokens=300)._get_request_payload(msgs)
    assert "temperature" not in p6
    assert p6.get("max_completion_tokens") == 300, "el tope de salida sigue viajando con su nombre nuevo"
    assert lp.ChatOpenAI(model="gpt-6-luna", temperature=1, api_key="sk-fake")._get_request_payload(msgs)["temperature"] == 1
    glm = lp.ChatGLM(model="glm-5.3", temperature=0.7, api_key="sk-fake", base_url="https://api.z.ai/api/paas/v4")
    assert glm._get_request_payload(msgs)["temperature"] == 0.7, "a GLM/DeepSeek no se le toca"
    assert lp.openai_model_only_default_temperature("gpt-6-sol")
    assert not lp.openai_model_only_default_temperature("glm-5.3-flash")


def test_el_cliente_instrumentado_hereda_el_filtro():
    from langchain_core.messages import HumanMessage
    import graph_orchestrator as go
    llm = go.ChatOpenAIInstrumented(model="gpt-6-luna", temperature=0.1, api_key="sk-fake")
    assert "temperature" not in llm._get_request_payload([HumanMessage("hola")])


# ─────────────────────────── marcador ───────────────────────────

def test_marcador_del_lote():
    app = _src("app.py")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 171 and m.group(2) >= "2026-09-23"   # la serie sigue; el marker nunca baja
    assert "[P1-PLAN-LOTE-171 · 2026-09-23]" in app
