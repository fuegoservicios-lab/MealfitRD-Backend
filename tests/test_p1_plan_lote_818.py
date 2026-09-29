# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-818 · 2026-09-29] Menores de los revisores de 813 y 814.

1. La huella del 813 (`_cadena_salida_813`, y `_cadena_ctx_813` si quedara) es instrumento de la GENERACIÓN: la lee
   el escudo pre-INSERT para `input_equals_chain_out`. El pre-INSERT trabaja sobre una COPIA (deepcopy), así que el
   `result` original seguía llevándola al cliente por el SSE, a la KV del invitado y, de vuelta, a la fila por
   `restore-local` (que la caché semántica lee de `meal_plans`). Una sola lista (`CLAVES_PRIVADAS`) y un solo helper
   (`retirar_claves_privadas`) en los puntos de salida — DESPUÉS de persistir, para no dejar ciego al instrumento.
   Además `_duplicate_food_lines_merged` se reinicia al entrar a los mutadores (se acumulaba entre re-entradas).
2. 814: el UPDATE `stale` de `registry_dishes_unused` le da `last_evaluable_at` a la alerta abierta por v1
   (`triggered_at`: en v1, el último tick que pudo evaluar), como promete la tabla de alertas.
3. 814: el docstring de `recipe_library.dish_provenance` y el tick del cron dicen qué mide cada tasa.
"""
from __future__ import annotations

import inspect
import json

import pytest

import graph_orchestrator as go

_SALIDA = "_cadena_salida_813"
_CTX = "_cadena_ctx_813"


def _mdc():
    import mutadores_de_contenido as m
    return m


def _plan_con_huella():
    return {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Moro con pollo",
                                           "ingredients": ["150 g de Pollo"]}]}],
            _SALIDA: {"h": "0123456789abcdef", "surface": "assemble-tail"},
            _CTX: {"plan_id": None}}


# ─────────────── 1a. la lista y el helper únicos ───────────────
def test_claves_privadas_ssot_y_helper():
    """tooltip-anchor: P1-PLAN-LOTE-818-HUELLA-FUERA"""
    m = _mdc()
    assert set(m.CLAVES_PRIVADAS) == {_SALIDA, _CTX}
    pd = _plan_con_huella()
    assert m.retirar_claves_privadas(pd) == 2
    assert _SALIDA not in pd and _CTX not in pd and pd["days"], "sólo salen las claves privadas"
    assert m.retirar_claves_privadas(pd) == 0
    assert m.retirar_claves_privadas(None) == 0 and m.retirar_claves_privadas([1]) == 0


# ─────────────── 1b. el payload SSE / sync / cola: fuera, pero DESPUÉS de persistir ───────────────
@pytest.fixture
def _postprocess(monkeypatch):
    import plan_mode
    import plan_policy
    from routers import plans
    monkeypatch.setattr(plan_mode, "reencender_al_generar", lambda *a, **k: False)
    monkeypatch.setattr(plan_policy, "stamp_plan_policy", lambda *a, **k: None)
    monkeypatch.setattr(plans, "update_user_health_profile_atomic", lambda *a, **k: None)
    monkeypatch.setattr(plans, "reescritura_tras_generar", lambda hp, **k: (hp, None))
    monkeypatch.setattr(plans, "log_api_usage", lambda *a, **k: None)
    monkeypatch.setattr(plans, "get_user_profile", lambda *a, **k: {})
    vistos = []

    def _save(user_id, plan_data, selected_techniques=None, **k):
        vistos.append({c: (c in plan_data) for c in (_SALIDA, _CTX)})
        return "plan-818"

    monkeypatch.setattr(plans, "_save_plan_and_track_background", _save)

    def correr(user_id):
        from fastapi import BackgroundTasks
        return plans._postprocess_pipeline_result(
            result=_plan_con_huella(), actual_user_id=user_id, session_id=None, data={"mainGoal": "x"},
            taste_profile="", memory_ctx="", rejected_meal_names=[], total_days_requested=3, use_chunking=False,
            background_tasks=BackgroundTasks(), plan_start_date="2026-09-29", tz_offset_mins=-240,
            transport_label="sse")
    return correr, vistos


def test_el_payload_sse_no_lleva_la_huella_y_el_pre_insert_si_la_ve(_postprocess):
    correr, vistos = _postprocess
    out = correr("user-818")
    assert out.get("id") == "plan-818"
    assert vistos and vistos[0][_SALIDA] is True, "el pre-INSERT tiene que VER la huella: sin ella el instrumento " \
                                                  "del 813 (`input_equals_chain_out`) queda en None"
    assert _SALIDA not in out and _CTX not in out, "la huella viajaba al cliente en el payload SSE"


def test_invitado_tampoco_la_recibe(_postprocess):
    correr, vistos = _postprocess
    out = correr(None)
    assert not vistos and _SALIDA not in out and _CTX not in out


def test_se_retira_tras_las_dos_persistencias_y_antes_del_return():
    """tooltip-anchor: P1-PLAN-LOTE-818-HUELLA-FUERA (en `_postprocess_pipeline_result`)."""
    from routers import plans
    src = inspect.getsource(plans._postprocess_pipeline_result)
    i = src.index("retirar_claves_privadas(result)")
    assert src.index("save_partial_plan_get_id(") < i and src.index("_save_plan_and_track_background(") < i
    assert i < src.rindex("return result")


# ─────────────── 1c. restore-local no la acepta ───────────────
class _Cur:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append((" ".join(str(sql).split()), params))

    def fetchone(self):
        return {"id": "plan-818"}


class _Cm:
    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self.obj

    def __exit__(self, *a):
        return False


class _Conn:
    def __init__(self, log):
        self.log = log

    def transaction(self):
        return _Cm(self)

    def cursor(self, **k):
        return _Cur(self.log)


class _Pool:
    def __init__(self):
        self.log = []

    def connection(self):
        return _Cm(_Conn(self.log))


def test_restore_local_no_persiste_la_huella(monkeypatch):
    import db_core
    import db_plans
    import plan_jobs
    from routers import plans
    pool = _Pool()
    monkeypatch.setattr(db_core, "connection_pool", pool)
    monkeypatch.setattr(db_plans, "acquire_meal_plan_advisory_lock", lambda *a, **k: None)
    monkeypatch.setattr(db_plans, "set_meal_plan_for_update_timeouts", lambda *a, **k: None)
    monkeypatch.setattr(plan_jobs, "enqueue_shopping_reprojection", lambda *a, **k: None)
    body = {"plan_data": _plan_con_huella()}
    assert plans.api_restore_plan_local("plan-818", body, "user-818") == {"success": True}
    upd = [p for s, p in pool.log if s.startswith("UPDATE meal_plans SET plan_data")]
    assert upd, pool.log
    escrito = json.loads(upd[0][0])
    assert escrito["days"] and _SALIDA not in escrito and _CTX not in escrito, \
        "restore-local reescribía la huella que el cliente recibió por el SSE"


# ─────────────── 1d. la fila que lee la caché semántica y la KV del invitado ───────────────
def test_la_fila_de_meal_plans_no_la_lleva():
    """`match_similar_plan` (caché semántica) lee `meal_plans.plan_data`. `save_new_meal_plan_atomic` construye el
    INSERT con `skip_plan_data_finalize=True`: el constructor mismo es el último punto antes de la fila."""
    import db_plans
    sql, vals = db_plans._build_meal_plan_insert_sql({"plan_data": _plan_con_huella()},
                                                     skip_plan_data_finalize=True)
    pd = next(v.obj for v in vals if hasattr(v, "obj"))
    assert pd["days"] and _SALIDA not in pd and _CTX not in pd


def test_la_kv_del_invitado_no_la_lleva(monkeypatch):
    import db_plans
    escrito = []
    monkeypatch.setattr(db_plans, "execute_sql_write", lambda sql, params=None, **k: escrito.append(params))
    pd = _plan_con_huella()
    assert db_plans.upsert_guest_plan("sid-818", pd) is True
    guardado = json.loads(escrito[0][1])["plan_data"]
    assert guardado["days"] and _SALIDA not in guardado and _CTX not in guardado


# ─────────────── 1e. la telemetría de duplicados no se arrastra entre re-entradas ───────────────
def test_duplicados_se_reinicia_al_entrar_y_acumula_dentro_de_la_corrida(monkeypatch):
    m = _mdc()
    monkeypatch.setattr(go, "PHANTOM_INGREDIENT_REPAIR", False)
    monkeypatch.setattr(go, "NAME_PHANTOM_DAIRY_REPAIR", False)
    monkeypatch.setattr(go, "COOKED_GRAIN_DRY_REWRITE", False)
    monkeypatch.setattr(go._ccr, "nombrar_quesos_genericos", lambda days: None)
    tandas = iter([[{"day": 1, "food": "Pan", "into": "4 rebanadas de pan integral"}],
                   [{"day": 2, "food": "Huevo", "into": "3 huevos"}], [], []])
    monkeypatch.setattr(go, "_merge_duplicate_food_lines", lambda days: next(tandas))
    monkeypatch.setattr(m, "ASSEMBLE_MUTATORS_BEFORE_CHAIN", True)
    r = {"days": [], "_duplicate_food_lines_merged": [{"day": 9, "food": "viejo", "into": "de otra corrida"}]}
    m.en_posicion(r, "antes")
    m.en_posicion(r, "despues")
    assert [d["day"] for d in r["_duplicate_food_lines_merged"]] == [1, 2], r["_duplicate_food_lines_merged"]
    m.en_posicion(r, "antes")
    m.en_posicion(r, "despues")
    assert "_duplicate_food_lines_merged" not in r, "una re-entrada sin fusiones no puede heredar la anterior"


# ─────────────── 2. la alerta stale conserva last_evaluable_at (814) ───────────────
def test_stale_da_last_evaluable_at_a_la_alerta_de_v1(monkeypatch):
    """tooltip-anchor: P1-PLAN-LOTE-818-LAST-EVALUABLE. Simulado con SELECT sobre la fila real abierta el 26-sep:
    `last_evaluable_at` = «2026-09-26 11:28:03.224842+00», `verdict_counting` = «v1_filas»."""
    import registry_dish_alert as rda
    import admin_acceso
    monkeypatch.setattr(admin_acceso, "admin_ids", lambda: set())
    escr = []
    out = rda.run_v2("registry_dishes_unused", lambda *a, **k: [],
                     lambda sql, params=None, **k: escr.append((" ".join(str(sql).split()), params)),
                     lookback_h=168, min_samples=5, floor=0.5)
    assert out["skip"] and "insufficient_samples" in out["skip"]
    stale = [s for s, _ in escr if s.startswith("UPDATE system_alerts")]
    assert stale, escr
    assert "'last_evaluable_at', COALESCE(metadata->>'last_evaluable_at', triggered_at::text)" in stale[0], \
        "la alerta emitida por v1 no tiene last_evaluable_at: en v1 lo es su triggered_at"
    assert "triggered_at =" not in stale[0] and "resolved_at =" not in stale[0], "leerla no es tocarla"


# ─────────────── 3. qué mide cada tasa ───────────────
def test_docstring_de_dish_provenance_no_promete_la_costura_con_v2():
    import recipe_library as rl
    doc = rl.dish_provenance.__doc__ or ""
    assert "el rendimiento real de la costura—, no" not in doc
    assert "procedencia" in doc.lower() and "v2" in doc.lower()


def test_el_tick_dice_que_la_tasa_v2_es_de_procedencia(monkeypatch):
    import cron_tasks
    escr = []
    monkeypatch.setattr(cron_tasks, "execute_sql_query", lambda *a, **k: [], raising=False)
    monkeypatch.setattr(cron_tasks, "execute_sql_write",
                        lambda sql, params=None, **k: escr.append((str(sql), params)), raising=False)
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    for v2, base in (("1", "provenance"), ("0", "applicable")):
        escr.clear()
        monkeypatch.setenv("MEALFIT_REGISTRY_PROVENANCE_V2", v2)
        cron_tasks._registry_dish_rate_alert_job()
        tick = [p for s, p in escr if "_registry_dish_rate_alert_job_tick" in s]
        assert tick and json.loads(tick[-1][1])["registry_dish_rate_basis"] == base
