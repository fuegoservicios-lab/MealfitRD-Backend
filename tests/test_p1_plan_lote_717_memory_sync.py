# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-717 · 2026-09-28] Configuración que dice la verdad: la memoria pausada se pausa de verdad, las
lecturas no inventan, y la copia vieja de un dispositivo ya no pisa el perfil.

tooltip-anchor: P1-PLAN-LOTE-717

Cada bloque falla contra el código previo y pasa con el arreglo. Todo mockeado (el `.env` de un checkout apunta a
PRODUCCIÓN): ni una lectura/escritura real ni una llamada LLM.

  1. «Memoria a Largo Plazo» pausada ⇒ el coach NO consulta `user_facts` (los dos caminos de `agent.py`), la
     extracción no corre (chat, paneles de Configuración) y la cola de pendientes no procesa. Ilegible ⇒ pausada.
  2. Las lecturas de Configuración (memoria, entrenamiento IA, Nevera) responden 503 `preference_unavailable` si la base
     falla, en vez de inventar un valor con 200.
  3. `/api/diary/preferences/logging`: sin sesión ⇒ 401 (no un UPDATE `WHERE id = NULL` «exitoso»); cero filas ⇒ 404.
  4. `PATCH /api/profile`: las claves con dueño se ignoran; `health_profile_keys` limita el parche; la fusión va por el
     helper atómico y un cambio de alergias invalida los chunks pendientes.
  5. Generación: para las claves de los paneles manda el perfil (y la reescritura no las pisa); los básicos se
     escriben con las dos claves iguales.
  6. Plan mode: una pausa hecha MIENTRAS se generaba manda; la cola no reenciende; reanudar sin pausa es un no-op.
  7. Los tres PUT de los paneles tienen su limitador (par único).
"""
from __future__ import annotations

import asyncio
import contextlib
import re
from pathlib import Path

import pytest
from fastapi import BackgroundTasks, HTTPException

_BACKEND = Path(__file__).resolve().parents[1]
UID = "71771717-1717-4717-8717-171717171717"


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _sql_falla(*_a, **_k):
    raise RuntimeError("base caída (simulada)")


def _flag_de_memoria(valor):
    """Un `execute_sql_query` falso que solo sabe responder la lectura del interruptor de memoria. `valor=None` ⇒ la
    base falla en esa lectura. Cualquier otra consulta se comporta como sin pool (lanza), igual que en los arneses."""
    def _q(sql, params=None, **_k):
        if "long_term_memory_enabled" in sql:
            if valor is None:
                raise RuntimeError("base caída (simulada)")
            return {"long_term_memory_enabled": valor}
        raise RuntimeError("db connection_pool is not available.")
    return _q


# ═════════════════════════════════════════════════ 1. Memoria a largo plazo ══════════════════════════════════════════
def test_la_lectura_del_interruptor_falla_cerrada_y_sin_fila_es_el_default(monkeypatch):
    import db
    import memoria_largo_plazo as mlp

    monkeypatch.setattr(db, "execute_sql_query", _flag_de_memoria(False))
    assert mlp.memoria_activa(UID) is False
    monkeypatch.setattr(db, "execute_sql_query", _flag_de_memoria(True))
    assert mlp.memoria_activa(UID) is True
    monkeypatch.setattr(db, "execute_sql_query", _flag_de_memoria(None))
    assert mlp.memoria_activa(UID) is False, "ilegible ⇒ PAUSADA (privacidad), nunca «activada»"
    with pytest.raises(mlp.MemoriaIlegible):
        mlp.leer_memoria(UID)
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: None)
    assert mlp.leer_memoria(UID) is True, "sin fila = el DEFAULT TRUE de la columna, no un fallo"
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: pytest.fail("un invitado no se consulta"))
    assert mlp.memoria_activa("guest") is False and mlp.memoria_activa(None) is False


class _PromptCapturado(RuntimeError):
    pass


def _turno_del_coach(monkeypatch, path: str, memoria):
    """Un turno autenticado por el camino REAL del coach, cortado en el grafo (arnés de test_p1_nevera_opcional). El
    router RAG dice «busca», la búsqueda devuelve un hecho testigo. Devuelve (prompt, búsquedas, contextos de memoria)."""
    from types import SimpleNamespace

    import agent
    import db
    import db_core
    import db_inventory
    import db_plans
    from prompts.sentiment import PERSONALITY_PROFILES

    capturado, busquedas, routers, contextos = {}, [], [], []

    class _Grafo:
        def get_state(self, _config):
            return SimpleNamespace(values={})

        def invoke(self, inputs, **_k):
            capturado["prompt"] = inputs["sys_prompt"]
            raise _PromptCapturado

        stream = invoke

    class _Builder:
        def compile(self, **_k):
            return _Grafo()

    monkeypatch.setattr(db, "execute_sql_query", _flag_de_memoria(memoria))
    monkeypatch.setattr(db_core, "connection_pool", None)
    monkeypatch.setattr(db_inventory, "get_user_inventory", lambda uid: [])
    monkeypatch.setattr(db_plans, "get_latest_usable_meal_plan_with_id", lambda uid: None)
    monkeypatch.setattr(agent, "build_memory_context",
                        lambda *a: contextos.append(a) or {"recent_messages": [], "summary_context": ""})
    monkeypatch.setattr(agent, "classify_sentiment",
                        lambda _p: {**PERSONALITY_PROFILES["neutral"], "sentiment": "neutral"})
    monkeypatch.setattr(agent, "rag_query_router",
                        lambda p: routers.append(p) or {"skip": False, "query": p})
    monkeypatch.setattr(agent, "get_embedding", lambda *_a, **_k: [0.1, 0.2, 0.3])
    monkeypatch.setattr(agent, "get_multimodal_embedding", lambda *_a, **_k: None)
    monkeypatch.setattr(agent, "search_user_facts",
                        lambda *a, **k: busquedas.append(a) or [{"fact": "Odia el cilantro TESTIGO-717"}])
    monkeypatch.setattr(agent, "search_visual_diary", lambda *a, **k: [])
    monkeypatch.setattr(agent, "_emit_chat_stream_total_duration_best_effort", lambda *_a: None)
    monkeypatch.setattr(agent, "chat_builder", _Builder())
    monkeypatch.setattr(agent, "chat_checkpoint_pool", None)
    monkeypatch.setattr(agent, "connection_pool", None)
    kwargs = dict(session_id="sesion-717", prompt="¿Qué ceno hoy?", user_id=UID, form_data={})
    with pytest.raises(_PromptCapturado):
        if path == "stream":
            list(agent.chat_with_agent_stream(**kwargs))
        else:
            agent.chat_with_agent(**kwargs)
    return capturado["prompt"], busquedas, routers, contextos


@pytest.mark.parametrize("path", ("nonstream", "stream"))
@pytest.mark.parametrize("memoria", (False, None), ids=("pausada", "ilegible"))
def test_con_la_memoria_pausada_o_ilegible_el_coach_no_consulta_lo_aprendido(monkeypatch, path, memoria):
    """EL CASO: el interruptor prometía «no consulta lo aprendido» y los dos caminos buscaban en `user_facts` en cada
    turno. Pausada (o sin poder leerla) ⇒ ni router RAG, ni búsqueda, ni bloque en el prompt, ni «modelo del usuario»."""
    prompt, busquedas, routers, contextos = _turno_del_coach(monkeypatch, path, memoria)
    assert busquedas == [], "con la memoria pausada no se busca en user_facts"
    assert routers == [], "ni siquiera se gasta la llamada del router RAG"
    assert "TESTIGO-717" not in prompt and "MEMORIA VECTORIAL" not in prompt
    assert contextos and all(len(a) > 1 and a[1] is None for a in contextos), \
        "build_memory_context sin user_id: sin «modelo del usuario» del Dreaming"
    # El control, en el mismo arnés: encendida SÍ consulta (si no, lo de arriba pasaría porque el arnés no busca nunca).
    prompt, busquedas, _routers, contextos = _turno_del_coach(monkeypatch, path, True)
    assert busquedas and "TESTIGO-717" in prompt and "MEMORIA VECTORIAL" in prompt
    assert contextos[0][1] == UID


def _api_chat(monkeypatch, memoria):
    """`POST /api/chat` (no-stream) por su función real, con todo lo de alrededor mockeado. Devuelve las tareas en
    segundo plano que dejó programadas."""
    import db
    import db_chat
    import routers.chat as ch

    monkeypatch.setattr(db, "execute_sql_query", _flag_de_memoria(memoria))
    # El camino previo leía el perfil con `get_user_profile`, que devuelve None tanto sin fila como con la base caída:
    monkeypatch.setattr(ch, "get_user_profile", lambda uid: None)
    monkeypatch.setattr(ch, "_resolve_chat_local_time", lambda d, t, u: (d, t))
    monkeypatch.setattr(db_chat, "get_session_owner", lambda sid: None)
    monkeypatch.setattr(ch, "get_or_create_session", lambda *a, **k: None)
    monkeypatch.setattr(ch, "save_message", lambda *a, **k: None)
    monkeypatch.setattr(ch, "merge_form_data_with_profile", lambda uid, fd: {})
    monkeypatch.setattr(ch, "get_latest_meal_plan", lambda uid: None)
    monkeypatch.setattr(ch, "chat_with_agent", lambda *a, **k: ("Respuesta", {}, None))
    monkeypatch.setattr(ch, "log_api_usage", lambda *a, **k: None)
    monkeypatch.setattr(ch, "get_session_messages", lambda sid: [])
    bt = BackgroundTasks()
    ch.api_chat(bt, {"session_id": "s-717", "prompt": "Soy alérgico al maní", "user_id": UID},
                verified_user_id=UID)
    return [t.func for t in bt.tasks]


def test_api_chat_no_extrae_si_la_memoria_es_ilegible(monkeypatch):
    """Antes: perfil ilegible ⇒ «activada» ⇒ se programaba la extracción. Ahora: fail-CLOSED."""
    import routers.chat as ch
    assert ch.async_extract_and_save_facts not in _api_chat(monkeypatch, None)
    assert ch.async_extract_and_save_facts not in _api_chat(monkeypatch, False)
    assert ch.async_extract_and_save_facts in _api_chat(monkeypatch, True), "control: encendida sí extrae"


def test_los_dos_caminos_del_chat_leen_el_interruptor_con_la_misma_puerta():
    """El camino streaming programa la extracción dentro del generador SSE (no se puede ejecutar aquí sin el
    transporte): se ancla que usa la MISMA lectura fail-closed y que el default «activada» no vuelve."""
    src = _src("routers/chat.py")
    assert src.count("memoria_activa(user_id, donde=") == 2
    assert "ltm_enabled = True" not in src


def _fe_aislado(monkeypatch, memoria):
    import db
    import fact_extractor as fe
    llamadas = []
    monkeypatch.setattr(db, "execute_sql_query", _flag_de_memoria(memoria))
    for nombre in ("should_extract_facts", "acquire_fact_lock", "enqueue_pending_fact", "extract_facts"):
        monkeypatch.setattr(fe, nombre, lambda *a, _n=nombre, **k: llamadas.append(_n) or False)
    monkeypatch.setattr(fe, "release_fact_lock", lambda *a, **k: llamadas.append("release_fact_lock"))
    return fe, llamadas


@pytest.mark.parametrize("memoria", (False, None), ids=("pausada", "ilegible"))
def test_la_extraccion_con_memoria_pausada_no_gasta_ni_encola(monkeypatch, memoria):
    """La guarda va en el DESTINO: ni router (LLM), ni lock, ni cola para después."""
    fe, llamadas = _fe_aislado(monkeypatch, memoria)
    fe.async_extract_and_save_facts(UID, "Soy alérgico al maní", "")
    assert llamadas == []
    fe, llamadas = _fe_aislado(monkeypatch, True)      # control: encendida sigue su camino (el router decide)
    fe.async_extract_and_save_facts(UID, "hola, gracias", "")
    assert llamadas == ["should_extract_facts"]


def _cola(monkeypatch, memoria, items):
    import db
    import fact_extractor as fe
    visto = {"procesados": [], "borrados": [], "liberado": [], "leida": 0}
    monkeypatch.setattr(db, "execute_sql_query", _flag_de_memoria(memoria))
    monkeypatch.setattr(fe, "acquire_fact_lock", lambda uid: "tok-717")
    monkeypatch.setattr(fe, "release_fact_lock", lambda *a, **k: visto["liberado"].append(a))

    def _dequeue(uid):
        visto["leida"] += 1
        return [dict(i) for i in items]
    monkeypatch.setattr(fe, "dequeue_pending_facts", _dequeue)
    monkeypatch.setattr(fe, "delete_pending_facts", lambda ids: visto["borrados"].append(list(ids)))
    monkeypatch.setattr(fe, "_process_single_extraction", lambda uid, msg, hist="": visto["procesados"].append(msg))
    fe.process_pending_queue_sync(UID)
    return visto


def test_la_cola_no_aprende_de_quien_pauso_la_memoria(monkeypatch):
    """Pausada: los pendientes se CONSERVAN sin procesar (si la reactiva dentro del tope, se aprenden entonces) y solo
    se descartan los que ya superaban el tope de antigüedad."""
    from datetime import datetime, timedelta, timezone
    import fact_extractor as fe
    ahora = datetime.now(timezone.utc)
    viejo = ahora - timedelta(hours=fe._pending_fact_max_age_h() + 1)
    items = [{"id": "fresco", "message": "soy alérgico al maní", "recent_history": "", "created_at": ahora},
             {"id": "viejo", "message": "me gusta el mangú", "recent_history": "", "created_at": viejo}]
    visto = _cola(monkeypatch, False, items)
    assert visto["procesados"] == []
    assert visto["borrados"] == [["viejo"]]
    assert visto["liberado"] == [(UID, "tok-717")]


def test_la_cola_con_memoria_ilegible_no_toca_nada(monkeypatch):
    visto = _cola(monkeypatch, None, [{"id": "x", "message": "m", "recent_history": "", "created_at": None}])
    assert visto["procesados"] == [] and visto["borrados"] == [] and visto["leida"] == 0
    assert visto["liberado"] == [(UID, "tok-717")], "el lock se libera igual"
    visto = _cola(monkeypatch, True, [{"id": "x", "message": "m", "recent_history": "", "created_at": None}])
    assert visto["procesados"] == ["m"] and visto["borrados"] == [["x"]], "control: encendida procesa y borra"


@pytest.mark.parametrize("panel", ("super", "clinico"))
def test_los_paneles_de_configuracion_no_aprenden_con_la_memoria_pausada(monkeypatch, panel):
    """El texto libre de los paneles programa `async_extract_and_save_facts`; con la memoria pausada la tarea corre y
    NO extrae (la guarda está en el destino)."""
    import db
    import routers.user_data as ud
    fe, llamadas = _fe_aislado(monkeypatch, False)
    monkeypatch.setattr(db, "update_user_health_profile_atomic", lambda _uid, mut: mut({}) or {})
    monkeypatch.setattr(ud, "_SUPERPERS_EXTRACT_FACTS", True)
    bt = BackgroundTasks()
    if panel == "super":
        asyncio.run(ud.api_put_super_personalization(
            bt, body=ud.SuperPersonalizationBody(freeText="Trabajo de noche y odio el cilantro"), verified_user_id=UID))
    else:
        asyncio.run(ud.api_put_clinical_profile(
            bt, body=ud.ClinicalProfileBody(freeText="Tengo reflujo por las noches"), verified_user_id=UID))
    assert bt.tasks, "el PUT programa la extracción (la decide el destino)"
    for t in bt.tasks:
        t.func(*t.args, **t.kwargs)
    assert llamadas == [], "con la memoria pausada el panel no gasta LLM ni encola"


# ═════════════════════════════════════════ 2. Lecturas de Configuración: 503, no inventar ══════════════════════════
def _503(coro):
    with pytest.raises(HTTPException) as e:
        asyncio.run(coro)
    assert e.value.status_code == 503 and e.value.detail == "preference_unavailable", (e.value.status_code,
                                                                                         e.value.detail)


def test_get_memoria_503_si_la_base_falla_y_401_sin_sesion(monkeypatch):
    import db
    from routers import preferences as pref
    monkeypatch.setattr(pref, "get_user_profile", lambda uid: None, raising=False)   # la lectura previa
    monkeypatch.setattr(db, "execute_sql_query", _sql_falla)
    _503(pref.api_get_long_term_memory(verified_user_id=UID))
    monkeypatch.setattr(db, "execute_sql_query", _flag_de_memoria(False))
    assert asyncio.run(pref.api_get_long_term_memory(verified_user_id=UID)) == {"long_term_memory_enabled": False}
    with pytest.raises(HTTPException) as e:
        asyncio.run(pref.api_get_long_term_memory(verified_user_id=None))
    assert e.value.status_code == 401


def test_get_entrenamiento_ia_503_si_la_base_falla(monkeypatch):
    import db
    from routers import preferences as pref
    monkeypatch.setattr(pref, "get_user_profile", lambda uid: None, raising=False)
    monkeypatch.setattr(db, "execute_sql_query", _sql_falla)
    _503(pref.api_get_ai_training_consent(verified_user_id=UID))
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: {"ai_training_consent": True})
    assert asyncio.run(pref.api_get_ai_training_consent(verified_user_id=UID)) == {"ai_training_consent": True}
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: None)
    assert asyncio.run(pref.api_get_ai_training_consent(verified_user_id=UID)) == {"ai_training_consent": False}


def test_nevera_503_si_la_base_falla_en_el_get_y_en_la_relectura_del_patch(monkeypatch):
    import nevera_opcional as no
    from routers import preferences as pref
    monkeypatch.setattr(no, "execute_sql_query", _sql_falla)
    _503(pref.api_get_nevera(verified_user_id=UID))
    monkeypatch.setattr(no, "fijar_nevera", lambda uid, en: True)
    _503(pref.api_set_nevera(body=pref.NeveraPreferenceBody(enabled=True), verified_user_id=UID))


# ═════════════════════════════════════════════ 3. Preferencia de registro del diario ═════════════════════════════
def test_logging_sin_sesion_es_401_y_no_toca_la_base(monkeypatch):
    import db_core
    from routers import diary
    monkeypatch.setattr(db_core, "execute_sql_query", lambda *a, **k: pytest.fail("sin sesión no se consulta"))
    monkeypatch.setattr(db_core, "execute_sql_write", lambda *a, **k: pytest.fail("sin sesión no se escribe"))
    for llamada in (lambda: diary.api_get_logging_preference(verified_user_id=None),
                    lambda: diary.api_set_logging_preference({"logging_preference": "manual"}, verified_user_id=None)):
        with pytest.raises(HTTPException) as e:
            llamada()
        assert e.value.status_code == 401


def test_logging_put_sin_filas_es_404_y_con_fila_exito(monkeypatch):
    import db_core
    from routers import diary
    monkeypatch.setattr(db_core, "execute_sql_write", lambda *a, **k: [])
    with pytest.raises(HTTPException) as e:
        diary.api_set_logging_preference({"logging_preference": "manual"}, verified_user_id=UID)
    assert e.value.status_code == 404
    monkeypatch.setattr(db_core, "execute_sql_write", lambda *a, **k: [{"id": UID}])
    assert diary.api_set_logging_preference({"logging_preference": "auto_proxy"}, verified_user_id=UID) == {
        "success": True, "logging_preference": "auto_proxy"}


def test_logging_get_con_la_base_caida_es_503(monkeypatch):
    import db_core
    from routers import diary
    monkeypatch.setattr(db_core, "execute_sql_query", _sql_falla)
    with pytest.raises(HTTPException) as e:
        diary.api_get_logging_preference(verified_user_id=UID)
    assert e.value.status_code == 503 and e.value.detail == "preference_unavailable"


# ═════════════════════════════════════════════════════ 4. PATCH /api/profile ═══════════════════════════════════════
_PERFIL_DB = {
    "allergies": ["maní"], "weight": 80, "country": "DO",
    "weight_history": [{"date": "2026-09-27", "weight": 80, "unit": "kg"}],
    "rejection_patterns": ["Mangú"], "frictions": ["tiempo"], "grocery_cycle": {"x": 1},
    "reflection_history": [1], "pipeline_score_history": [0.9],
    "clinical_profile": {"labs": {"tfg": 45.0}}, "super_personalization": {"freeText": "nuevo"},
    "staple_foods": ["Huevo", "Avena"], "stapleFoods": ["Huevo", "Avena"], "stapleAnchors": [{"name": "Huevo"}],
}


def _patch_perfil(monkeypatch, body_kwargs):
    import copy

    import db
    from routers.user_data import ProfilePatchBody, api_patch_profile
    estado = {"hp": copy.deepcopy(_PERFIL_DB), "atomico": 0, "sql": []}

    def _atomico(uid, mut):
        estado["atomico"] += 1
        hp = copy.deepcopy(estado["hp"])
        r = mut(hp)
        estado["hp"] = hp if r is None else r
        return estado["hp"]
    monkeypatch.setattr(db, "update_user_health_profile_atomic", _atomico)
    monkeypatch.setattr(db, "execute_sql_write", lambda sql, params=None, **k: estado["sql"].append(sql) or [{"id": UID}])
    out = asyncio.run(api_patch_profile(body=ProfilePatchBody(**body_kwargs), verified_user_id=UID))
    return out, estado


def test_el_patch_ignora_las_claves_con_dueno_y_aplica_el_resto(monkeypatch):
    """EL P0: el Dashboard manda el formulario ENTERO (hidratado una vez) en cada carga. Sus copias de las claves con
    dueño ya no pisan nada; lo editable (alergias, peso) se aplica como siempre."""
    viejo = {
        "weight_history": [], "rejection_patterns": [], "frictions": [], "grocery_cycle": {},
        "reflection_history": [], "pipeline_score_history": [],
        "clinical_profile": {"labs": {"tfg": 90.0}}, "super_personalization": {"freeText": "viejo"},
        "staple_foods": ["Huevo"], "stapleFoods": ["Huevo"], "stapleAnchors": [], "_interno": 1,
        "allergies": ["maní", "soya"], "weight": 81,
    }
    out, estado = _patch_perfil(monkeypatch, {"health_profile": viejo})
    hp = estado["hp"]
    assert estado["atomico"] == 1, "la fusión va por el helper atómico (invalidación post-commit)"
    assert hp["allergies"] == ["maní", "soya"] and hp["weight"] == 81
    for clave in ("weight_history", "rejection_patterns", "frictions", "grocery_cycle", "reflection_history",
                  "pipeline_score_history", "clinical_profile", "super_personalization", "staple_foods",
                  "stapleFoods", "stapleAnchors"):
        assert hp[clave] == _PERFIL_DB[clave], clave
    assert "_interno" not in hp
    assert out["success"] is True and "clinical_profile" in out["ignored_keys"] and "_interno" in out["ignored_keys"]


def test_health_profile_keys_limita_el_parche_a_lo_declarado(monkeypatch):
    """Un formulario entero con `allergies: []` (hidratación incompleta) ya no borra las alergias si el cliente declara
    que solo escribe el peso."""
    out, estado = _patch_perfil(monkeypatch, {
        "health_profile": {"allergies": [], "weight": 90, "country": "ES"}, "health_profile_keys": ["weight"]})
    assert estado["hp"]["weight"] == 90
    assert estado["hp"]["allergies"] == ["maní"] and estado["hp"]["country"] == "DO"
    assert out == {"success": True}


def test_un_parche_solo_de_claves_con_dueno_es_400_y_las_nombra(monkeypatch):
    """Un 200 haría creer al cliente que escribió su perfil clínico. Con escalares al lado, esos se escriben como
    siempre y la respuesta nombra lo ignorado."""
    with pytest.raises(HTTPException) as e:
        _patch_perfil(monkeypatch, {"health_profile": {"clinical_profile": {"labs": {}}}})
    assert e.value.status_code == 400 and "clinical_profile" in str(e.value.detail)
    out, estado = _patch_perfil(monkeypatch, {"health_profile": {"clinical_profile": {"labs": {}}},
                                              "fields": {"full_name": "Ana"}})
    assert out == {"success": True, "ignored_keys": ["clinical_profile"]} and estado["atomico"] == 0
    assert any("UPDATE user_profiles SET full_name" in s for s in estado["sql"])
    assert estado["hp"]["clinical_profile"] == _PERFIL_DB["clinical_profile"]


def test_el_contrato_health_profile_keys_tiene_techo():
    from pydantic import ValidationError
    from routers.user_data import ProfilePatchBody
    assert ProfilePatchBody(health_profile_keys=["weight"]).health_profile_keys == ["weight"]
    with pytest.raises(ValidationError):
        ProfilePatchBody(health_profile_keys=[f"k{i}" for i in range(201)])


class _Cursor:
    def __init__(self, store):
        self.store, self._fila = store, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        s = " ".join(str(sql).split())
        if s.startswith("SELECT") and "health_profile" in s:
            import copy
            self._fila = {"health_profile": copy.deepcopy(self.store["hp"])}
        elif s.startswith("UPDATE user_profiles SET health_profile"):
            self.store["hp"] = getattr(params[0], "obj", params[0])

    def fetchone(self):
        return self._fila


class _Conn:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def transaction(self):
        return contextlib.nullcontext()

    def cursor(self, row_factory=None):
        return _Cursor(self.store)


class _Pool:
    def __init__(self, store):
        self.store = store

    def connection(self):
        return _Conn(self.store)


@pytest.mark.parametrize("parche,invalida", (
    ({"allergies": ["huevo", "maní"]}, True),
    ({"notas": "cualquier cosa"}, False),
), ids=("alergia-nueva", "clave-inocua"))
def test_cambiar_las_alergias_por_el_patch_invalida_los_chunks_pendientes(monkeypatch, parche, invalida):
    """Lo que pidió la coordinación: el `||` crudo se saltaba la invalidación que el camino del coach sí hace
    (`update_user_health_profile_atomic` detecta `allergies_changed` y corre `_invalidate_stale_chunks`). Helper REAL,
    con un pool falso: la semana siguiente ya no se genera con la alergia vieja."""
    import db
    import db_profiles
    from routers.user_data import ProfilePatchBody, api_patch_profile
    store = {"hp": {"allergies": ["huevo"], "weight": 80}}
    invalidaciones = []
    monkeypatch.setattr(db_profiles, "connection_pool", _Pool(store))
    monkeypatch.setattr(db_profiles, "_invalidate_stale_chunks", lambda uid, motivo: invalidaciones.append((uid, motivo)))
    monkeypatch.setattr(db, "execute_sql_write", lambda *a, **k: [{"id": UID}])   # el camino previo «escribía» así
    out = asyncio.run(api_patch_profile(body=ProfilePatchBody(health_profile=dict(parche)), verified_user_id=UID))
    assert out == {"success": True}
    for k, v in parche.items():
        assert store["hp"][k] == v, "la fusión se aplicó sobre la fila (FOR UPDATE)"
    if invalida:
        assert invalidaciones == [(UID, "allergies_changed")]
    else:
        assert invalidaciones == []


# ═════════════════════════════════════════════ 5. Generación: el perfil manda en los paneles ════════════════════════
def test_perfil_servidor_lista_de_claves_con_dueno():
    import perfil_servidor as ps
    for clave in ("weight_history", "rejection_patterns", "frictions", "grocery_cycle", "reflection_history",
                  "pipeline_score_history"):
        assert clave in ps.CLAVES_DEL_SERVIDOR, clave
    assert ps.CLAVES_SOLO_DEL_PANEL == {"clinical_profile", "super_personalization"}
    assert ps.CLAVES_DE_BASICOS == {"staple_foods", "stapleFoods", "stapleAnchors"}
    for libre in ("allergies", "medicalConditions", "weight", "age", "gender", "height", "country", "dietType",
                  "avisos_comida", "avisos_agua", "avisos_por_comida", "otherAllergies", "medications"):
        assert not ps.es_clave_con_dueno(libre), f"{libre} se edita legítimamente por PATCH /api/profile"


def _hidratar(monkeypatch, data, hp):
    import db
    from routers.plans import _hydrate_country_from_profile_for_submit
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": hp})
    _hydrate_country_from_profile_for_submit(data, UID)
    return data


_HP_PANELES = {
    "country": "DO",
    "clinical_profile": {"labs": {"tfg": 45.0}}, "super_personalization": {"kitchenEquipment": ["estufa"]},
    "staple_foods": ["Huevo", "Avena"], "stapleFoods": ["Huevo"],
    "stapleAnchors": [{"name": "Huevo", "slots": ["desayuno"]}, {"name": "Pollo", "slots": ["cena"]}],
}


def test_renovacion_el_perfil_manda_en_paneles_y_basicos(monkeypatch):
    """«Renovar» desde un dispositivo con la copia vieja: el TFG 45 puesto en otro teléfono se usa (tope renal), y los
    básicos son los de Configuración — con las dos claves iguales y sin el ancla de un básico que ya no está."""
    data = _hidratar(monkeypatch, {
        "update_reason": "variety", "country": "DO",
        "clinical_profile": {"labs": {"tfg": 90.0}}, "super_personalization": {},
        "staple_foods": ["Pollo"], "stapleFoods": ["Pollo"], "stapleAnchors": [{"name": "Pollo"}],
    }, dict(_HP_PANELES))
    assert data["clinical_profile"] == {"labs": {"tfg": 45.0}}
    assert data["super_personalization"] == {"kitchenEquipment": ["estufa"]}
    assert data["staple_foods"] == data["stapleFoods"] == ["Huevo", "Avena"], "la canónica del panel"
    assert [a["name"] for a in data["stapleAnchors"]] == ["Huevo"], "el ancla de «Pollo» (ya no es básico) fuera"


def test_asistente_completo_manda_lo_contestado_en_basicos_pero_no_en_paneles(monkeypatch):
    """Sin `update_reason` el asistente acaba de preguntar los básicos (paso «Mis básicos"): mandan. Los paneles no
    los pregunta: su copia solo puede ser vieja."""
    data = _hidratar(monkeypatch, {
        "clinical_profile": {"labs": {"tfg": 90.0}}, "stapleFoods": ["Pollo"], "stapleAnchors": [{"name": "Pollo"}],
    }, dict(_HP_PANELES))
    assert data["clinical_profile"] == {"labs": {"tfg": 45.0}}
    assert data["stapleFoods"] == ["Pollo"] and "staple_foods" not in data
    assert data["stapleAnchors"] == [{"name": "Pollo"}]
    vacio = _hidratar(monkeypatch, {}, dict(_HP_PANELES))
    assert vacio["staple_foods"] == ["Huevo", "Avena"], "sin básicos en el formulario, se rellenan del perfil"


def _postprocesar(monkeypatch, data, *, hp_db, plan_mode_fila=None, transporte="sync", solicitado_en=None):
    """`_postprocess_pipeline_result` REAL por su rama no-chunked, con la base simulada: la reescritura del perfil se
    aplica sobre `hp_db` (como el FOR UPDATE) y `plan_mode` vive en `plan_mode_fila`."""
    import copy

    import plan_mode as pm
    import routers.plans as rp
    fila = plan_mode_fila if plan_mode_fila is not None else {"plan_mode": "plan", "plan_mode_changed_at": None}
    escrito = {"hp": copy.deepcopy(hp_db), "sql": []}

    def _atomico(uid, mut):
        hp = copy.deepcopy(escrito["hp"])
        r = mut(hp)
        escrito["hp"] = hp if r is None else r
        return escrito["hp"]

    def _pm_write(sql, params=None, returning=False, **_k):
        s = " ".join(sql.split())
        escrito["sql"].append(s)
        if s.startswith("UPDATE user_profiles SET plan_mode = 'plan'"):
            anterior = ("plan_mode_changed_at < %s::timestamptz" not in s or params[1] is None
                        or fila["plan_mode_changed_at"] is None or fila["plan_mode_changed_at"] < params[1])
            if fila["plan_mode"] == "tracking" and anterior:
                fila["plan_mode"] = "plan"
                return [{"id": UID}]
            return []
        if s.startswith("UPDATE user_profiles SET plan_mode = 'tracking'"):
            fila["plan_mode"] = "tracking"
        if "SET status = 'cancelled'" in s:
            return [{"id": "c1"}]
        return [] if returning else 1

    monkeypatch.setattr(pm, "PLAN_MODE_SWITCH_ENABLED", True)
    monkeypatch.setattr(pm, "execute_sql_write", _pm_write)
    monkeypatch.setattr(pm, "execute_sql_query", lambda sql, params=None, **k: {"plan_mode": fila["plan_mode"]})
    monkeypatch.setattr(rp, "update_user_health_profile_atomic", _atomico)
    monkeypatch.setattr(rp, "log_api_usage", lambda *a, **k: None)
    monkeypatch.setattr(rp, "_save_plan_and_track_background", lambda *a, **k: "plan-717")
    monkeypatch.setattr(rp, "get_user_profile", lambda uid: {"locale": "es-DO"})
    kwargs = dict(
        result={"days": [{"day": 1, "meals": [{"name": "Mangú"}]}]}, actual_user_id=UID, session_id=None,
        data=data, taste_profile="", memory_ctx="", rejected_meal_names=[], total_days_requested=3,
        use_chunking=False, background_tasks=BackgroundTasks(), plan_start_date="2026-09-28T04:00:00+00:00",
        tz_offset_mins=240, transport_label=transporte,
    )
    if solicitado_en is not None:
        kwargs["request_started_at"] = solicitado_en
    rp._postprocess_pipeline_result(**kwargs)
    return escrito, fila


def test_la_reescritura_tras_generar_no_pisa_las_claves_con_dueno(monkeypatch):
    """La generación volcaba el formulario entero al perfil: el `clinical_profile`, el historial de peso y lo
    aprendido del motor volvían a su copia vieja en cada plan. Ahora solo se reescribe lo que el formulario posee; en
    el asistente completo los básicos se escriben con las DOS claves iguales."""
    hp_db = {**_PERFIL_DB}
    data = {
        "allergies": ["maní", "soya"], "mainGoal": "lose_fat", "appMode": "plan", "user_id": UID,
        "clinical_profile": {"labs": {"tfg": 90.0}}, "super_personalization": {"freeText": "viejo"},
        "weight_history": [], "rejection_patterns": [], "staple_foods": ["Pollo"],
        "stapleAnchors": [{"name": "Pollo"}, {"name": "Huevo"}],
    }
    escrito, _ = _postprocesar(monkeypatch, dict(data), hp_db=hp_db)
    hp = escrito["hp"]
    assert hp["allergies"] == ["maní", "soya"] and hp["mainGoal"] == "lose_fat" and hp["tz_offset_minutes"] == 240
    for clave in ("clinical_profile", "super_personalization", "weight_history", "rejection_patterns"):
        assert hp[clave] == _PERFIL_DB[clave], clave
    assert hp["staple_foods"] == hp["stapleFoods"] == ["Pollo"], "asistente: lo contestado, en las dos claves"
    assert hp["stapleAnchors"] == [{"name": "Pollo"}], "sin anclas de básicos que ya no están"
    assert "appMode" not in hp and "user_id" not in hp

    renovado, _ = _postprocesar(monkeypatch, {**data, "update_reason": "variety"}, hp_db=hp_db)
    for clave in ("staple_foods", "stapleFoods", "stapleAnchors"):
        assert renovado["hp"][clave] == _PERFIL_DB[clave], f"renovación: {clave} del perfil"


# ═════════════════════════════════════════════════════════ 6. Plan mode ══════════════════════════════════════════════
def test_una_pausa_hecha_mientras_se_generaba_manda(monkeypatch):
    """EL CASO: el usuario pausa en Configuración mientras su plan se genera; al terminar, el postprocesado volvía a
    encender la generación en silencio. Ahora la pausa (posterior a la petición) se respeta y el plan recién
    persistido queda pausado como los demás (sus chunks cancelados con la firma que reanudar revive)."""
    from datetime import datetime, timedelta, timezone
    solicitado = datetime.now(timezone.utc) - timedelta(minutes=5)
    pausa = solicitado + timedelta(minutes=2)
    escrito, fila = _postprocesar(monkeypatch, {"allergies": []}, hp_db={},
                                  plan_mode_fila={"plan_mode": "tracking", "plan_mode_changed_at": pausa},
                                  solicitado_en=solicitado)
    assert fila["plan_mode"] == "tracking", "la pausa posterior NO se deshace"
    assert any("SET status = 'cancelled'" in s for s in escrito["sql"]), "el plan nuevo se pausa también"


def test_generar_desde_una_pausa_anterior_sigue_siendo_consentimiento(monkeypatch):
    """El control (P1-PLAN-MODE): pausado AYER y pulsa «Generar plan» ⇒ se enciende, como siempre."""
    from datetime import datetime, timedelta, timezone
    solicitado = datetime.now(timezone.utc)
    escrito, fila = _postprocesar(monkeypatch, {"allergies": []}, hp_db={},
                                  plan_mode_fila={"plan_mode": "tracking",
                                                  "plan_mode_changed_at": solicitado - timedelta(days=1)},
                                  solicitado_en=solicitado)
    assert fila["plan_mode"] == "plan"
    assert not any("SET status = 'cancelled'" in s for s in escrito["sql"])


def test_la_cola_no_reenciende_en_el_postprocesado(monkeypatch):
    """La vía `/generation-runs` encendió ANTES de encolar; dentro del worker solo podía deshacer una pausa."""
    escrito, fila = _postprocesar(monkeypatch, {"allergies": []}, hp_db={},
                                  plan_mode_fila={"plan_mode": "tracking", "plan_mode_changed_at": None},
                                  transporte="queue")
    assert fila["plan_mode"] == "tracking"
    assert not any("SET plan_mode = 'plan'" in s for s in escrito["sql"])


def test_los_tres_llamadores_sync_sse_pasan_la_hora_de_la_peticion():
    src = _src("routers/plans.py")
    assert src.count("request_started_at=now_utc,") == 3
    i = src.index("def _postprocess_pipeline_result(")
    cuerpo = src[i:src.index("\ndef _attach_pantry_degraded_response_meta")]
    assert "solicitado_en=request_started_at" in cuerpo and "repausar_tras_generar" in cuerpo


def test_reanudar_sin_estar_en_pausa_es_un_no_op_que_lo_dice(monkeypatch):
    """Reanudar con el plan corriendo hacía 40 días contestaba `plan_expired: true` (medía desde el último
    encendido) y re-estampaba la bandera. Ahora lo dice (`already_active`) sin tocar bandera ni reloj; solo cura restos
    (sello y cola firmada del plan vigente), que sin restos son no-op."""
    import plan_mode as pm
    escrituras, revividas = [], []
    monkeypatch.setattr(pm, "PLAN_MODE_SWITCH_ENABLED", True)
    monkeypatch.setattr(pm, "execute_sql_query", lambda *a, **k: {"plan_mode": "plan", "paused_days": 40})
    monkeypatch.setattr(pm, "execute_sql_write", lambda sql, *a, **k: escrituras.append(" ".join(sql.split())) or [])
    monkeypatch.setattr(pm, "_revive_paused_chunks", lambda uid: revividas.append(uid) or {"revived": 0, "plans": 0})
    out = pm.resume_plan_generation(UID)
    assert out["already_active"] is True and out["plan_expired"] is False and out["paused_days"] == 0
    assert out["chunks_revived"] == 0 and revividas == [UID]
    assert not any(s.startswith("UPDATE user_profiles") for s in escrituras), "la bandera no se toca"


def test_con_el_interruptor_operativo_apagado_la_respuesta_dice_que_no_cambio_nada(monkeypatch):
    import plan_mode as pm
    monkeypatch.setattr(pm, "PLAN_MODE_SWITCH_ENABLED", False)
    monkeypatch.setattr(pm, "execute_sql_write", lambda *a, **k: pytest.fail("apagado no escribe"))
    assert pm.resume_plan_generation(UID).get("skipped") == "switch_off"
    assert pm.pause_plan_generation(UID).get("skipped") == "switch_off"
    assert pm.reencender_al_generar(UID, transporte="sync", solicitado_en=None) is False


# ═════════════════════════════════════════════════ 7. Limitador de los paneles ═════════════════════════════════════
def test_los_tres_put_de_los_paneles_tienen_su_limitador_con_par_unico():
    src = _src("routers/user_data.py")
    for ruta in ("super-personalization", "clinical-profile", "staple-foods"):
        i = src.index(f'@router.put("/user/preferences/{ruta}")')
        firma = src[i:src.index("):", i)]
        assert "Depends(_PANEL_PREFERENCES_LIMITER)" in firma, ruta
    m = re.search(r"_PANEL_PREFERENCES_LIMITER = RateLimiter\(max_calls=(\d+), period_seconds=(\d+)\)", src)
    assert m, "el limitador de los paneles existe"
    par = f"RateLimiter(max_calls={m.group(1)}, period_seconds={m.group(2)})"
    usos = sum(p.read_text(encoding="utf-8", errors="replace").count(par)
               for p in _BACKEND.rglob("*.py") if "tests" not in p.parts and "scratch" not in p.parts)
    assert usos == 1, f"{par} compartiría ventana de Redis (rl:<max>:<periodo>:<uid>) con otro endpoint"


def test_el_limitador_de_los_paneles_corta_en_su_cupo(monkeypatch):
    from starlette.requests import Request

    import rate_limiter
    import routers.user_data as ud
    monkeypatch.setattr(rate_limiter, "redis_client", None)
    lim = ud._PANEL_PREFERENCES_LIMITER
    lim._hits.clear()
    req = Request({"type": "http", "client": ("127.0.0.1", 1), "headers": []})
    for _ in range(lim.max_calls):
        assert lim(req, verified_user_id="u-cupo-717") == "u-cupo-717"
    with pytest.raises(HTTPException) as e:
        lim(req, verified_user_id="u-cupo-717")
    assert e.value.status_code == 429
    lim._hits.clear()
