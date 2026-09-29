# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-843 · 2026-09-29] Sin petición delante: nada llama a un proveedor de IA para una cuenta sin permiso.

Review Focus 1 del plan: retirar el permiso con un plan a medio generar ⇒ no sale ninguna llamada más para esa cuenta
(recogida de bloques, coach proactivo, Dreaming…). Cada test fija el modo con monkeypatch y usa la base falsa.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import consentimientos as cs

_BACKEND = Path(__file__).resolve().parent.parent
UID = "11111111-2222-4333-8444-555555555555"


def _norm(sql) -> str:
    return " ".join(str(sql).split())


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


@pytest.fixture
def sin_permiso(monkeypatch):
    """`permite_ia` dice que no (como con `block` y la cuenta sin permiso o retirada)."""
    llamadas = []
    monkeypatch.setattr(cs, "permite_ia", lambda uid, donde="": llamadas.append((uid, donde)) or False)
    return llamadas


# ═════════════════════════════════════════════ 1. la recogida del chunk worker
@pytest.fixture
def recogida(monkeypatch):
    import cron_tasks as ct
    capturas = []

    def _w(query, params=None, returning=False, lock_timeout_ms=None):
        capturas.append(_norm(query))
        return [] if returning else True

    monkeypatch.setattr(ct, "execute_sql_write", _w)
    monkeypatch.setattr(ct, "execute_sql_query", lambda *a, **k: [] if k.get("fetch_all") else None)
    monkeypatch.setattr(ct, "_process_pending_shopping_lists", lambda: None)
    monkeypatch.setattr(ct, "_recover_pantry_paused_chunks", lambda: None)
    monkeypatch.setattr(ct, "_sync_chunk_queue_tz_offsets", lambda **k: 0)

    def _correr(target=None):
        capturas.clear()
        ct.process_plan_chunk_queue.__wrapped__(target)
        return [q for q in capturas if "FOR UPDATE SKIP LOCKED" in q and "RETURNING id, user_id, meal_plan_id" in q]
    return _correr


@pytest.mark.parametrize("target", [None, "22222222-3333-4444-8555-666666666666"])
def test_con_block_la_recogida_lleva_el_filtro_en_sus_dos_ramas(recogida, monkeypatch, target):
    monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", "block")
    pickups = recogida(target)
    assert len(pickups) == 1, "la sentencia de recogida no se ejecutó"
    q = pickups[0]
    assert "upc.id = q1.user_id" in q and "upc.ai_consent_version = 'ia-2026-10'" in q
    assert "upc.ai_consent_revoked_at IS NULL" in q, "retirar (revoked_at) corta la recogida"
    assert "up.plan_mode = 'tracking'" in q, "sigue junto a la pausa de P1-PLAN-MODE"
    assert "__PLAN_MODE_GATE__" not in q


def test_con_log_solo_la_retirada_y_con_off_nada(recogida, monkeypatch):
    monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", "log")
    q = recogida()[0]
    assert "upc.ai_consent_revoked_at IS NOT NULL" in q and "ai_consent_version" not in q
    monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", "off")
    assert "upc." not in recogida()[0]


def test_retirar_pone_la_bandera_que_la_recogida_lee_antes_de_tocar_la_cola():
    """La bandera que escribe `retirar` (`ai_consent_revoked_at`) es la columna que el filtro de la recogida exige
    NULL; y en `retirar` va antes que la pausa de la cola (la transacción se cierra primero)."""
    src = _src("consentimientos.py")
    cuerpo = src[src.index("def retirar("):src.index("def hash_de_sesion(")]
    assert cuerpo.index("ai_consent_revoked_at = now()") < cuerpo.index("pause_plan_generation(uid)")
    assert cuerpo.index("_en_transaccion(_tx)") < cuerpo.index("pause_plan_generation(uid)")


# ═════════════════════════════════════════════ 2. lo que avisa de esa cola
def test_el_escalado_y_la_alerta_de_zombies_saltan_a_quien_espera_el_permiso(monkeypatch):
    import cron_tasks as ct
    monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", "block")
    vistas = []
    monkeypatch.setattr(ct, "execute_sql_query", lambda q, *a, **k: vistas.append(_norm(q)) or [])
    monkeypatch.setattr(ct, "execute_sql_write", lambda *a, **k: True)
    ct._detect_and_escalate_stuck_chunks()
    ct._alert_stuck_chunks()
    escalado = next(q for q in vistas if "escalated_at IS NULL OR escalated_at" in q)
    zombies = next(q for q in vistas if "COUNT(*)::int AS chunk_count" in q)
    assert "upc.id = plan_chunk_queue.user_id" in escalado and "upc.ai_consent_revoked_at IS NULL" in escalado
    assert "upc.id = q.user_id" in zombies
    assert all("__AI_CONSENT_GATE__" not in q for q in vistas)


def test_el_update_del_escalado_y_la_rama_terminal_tambien_saltan_a_quien_espera(monkeypatch):
    """[ronda de arreglo 1] El UPDATE era masivo (escalaba también a los que el SELECT saltó) y la rama terminal no
    miraba el permiso: un bloque que espera a que la persona acepte acababa dado por perdido."""
    import cron_tasks as ct
    monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", "block")
    atascado = {"id": "44444444-5555-4666-8777-888888888888", "user_id": UID, "meal_plan_id": "p", "week_number": 2,
                "attempts": 0, "lag_seconds": 90000, "effective_lag_seconds": 90000, "escalated_at": None}
    lecturas, escrituras = [], []

    def _q(q, *a, **k):
        lecturas.append(_norm(q))
        return [atascado] if "escalated_at IS NULL OR escalated_at" in q else []

    monkeypatch.setattr(ct, "execute_sql_query", _q)
    monkeypatch.setattr(ct, "execute_sql_write", lambda q, params=None, **k: escrituras.append((_norm(q), params)))
    monkeypatch.setattr(ct, "_dispatch_push_notification", lambda **k: None)
    ct._detect_and_escalate_stuck_chunks()
    update, params = next((q, p) for q, p in escrituras if q.startswith("UPDATE plan_chunk_queue SET escalated_at"))
    assert "AND id = ANY(%s::uuid[])" in update and params == ([atascado["id"]],), "solo los que el SELECT eligió"
    terminal = next(q for q in lecturas if "escalated_at < NOW() - INTERVAL '72 hours'" in q)
    assert "upc.id = plan_chunk_queue.user_id" in terminal and "upc.ai_consent_revoked_at IS NULL" in terminal
    assert "__AI_CONSENT_GATE__" not in terminal


# ═════════════════════════════════════════════ 3. la traducción del plan (plan_jobs + motor)
def test_el_claim_de_plan_jobs_deja_en_cola_la_traduccion_sin_permiso(monkeypatch):
    import db
    import plan_jobs as pj
    capturas = []
    monkeypatch.setattr(db, "execute_sql_write", lambda q, params=None, **k: capturas.append(_norm(q)) or [])
    monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", "block")
    pj.claim_plan_jobs(5, "w1", ["display_i18n", "shopping_projection"])
    q = capturas[-1]
    assert "AND (j.job_type <> 'display_i18n' OR EXISTS (SELECT 1 FROM public.user_profiles upc WHERE upc.id = " \
           "j.user_id" in q
    monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", "off")
    pj.claim_plan_jobs(5, "w1", ["display_i18n"])
    assert "__AI_CONSENT_GATE__" not in capturas[-1] and "upc." not in capturas[-1]
    assert "__AI_CONSENT_GATE__" in pj.CLAIM_SQL and "j.job_type = ANY(%s)" in pj.CLAIM_SQL
    assert "ai_consent" in pj._RETRY_SKIPS and "ai_consent" not in pj._DONE_SKIPS


def test_el_motor_de_traduccion_no_llama_a_la_ia_sin_permiso(monkeypatch, sin_permiso):
    import plan_display_i18n as pdi
    monkeypatch.setattr(pdi, "_plan_display_i18n_enabled", lambda: True)
    monkeypatch.setattr(pdi, "_fetch_plan_data", lambda *a, **k: pytest.fail("ni siquiera se lee el plan"))
    monkeypatch.setattr(pdi, "build_chat_llm", lambda *a, **k: pytest.fail("sin permiso no se llama al modelo"))
    assert pdi.enrich_plan_display("plan-1", UID, "en-US") == {"enriched_meals": 0, "skipped": "ai_consent"}
    assert sin_permiso == [(UID, "traduccion_del_plan")]


# ═════════════════════════════════════════════ 4. Dreaming y el extractor de hechos
def test_dreaming_no_consolida_sin_permiso(monkeypatch, sin_permiso):
    import db_facts
    import dreaming
    import memoria_largo_plazo
    monkeypatch.setattr(memoria_largo_plazo, "memoria_activa", lambda uid, donde="": True)
    monkeypatch.setattr(db_facts, "acquire_fact_lock", lambda uid: pytest.fail("no se toca la memoria"))
    assert dreaming.consolidate_user(UID)["status"] == "skipped_ai_consent"


def test_el_extractor_no_llama_ni_encola_sin_permiso(monkeypatch, sin_permiso):
    import fact_extractor as fe
    monkeypatch.setattr(fe, "memoria_activa", lambda uid, donde="": True)
    monkeypatch.setattr(fe, "should_extract_facts", lambda *a, **k: pytest.fail("el router ya es IA"))
    monkeypatch.setattr(fe, "enqueue_pending_fact", lambda *a, **k: pytest.fail("tampoco se encola para después"))
    assert fe.async_extract_and_save_facts(UID, "tengo diabetes tipo 2") is None
    assert sin_permiso == [(UID, "extraccion_de_hechos")]


def test_la_cola_de_hechos_se_conserva_sin_permiso(monkeypatch, sin_permiso):
    import fact_extractor as fe
    borrados = []
    monkeypatch.setattr(fe, "acquire_fact_lock", lambda uid: "tok")
    monkeypatch.setattr(fe, "release_fact_lock", lambda uid, tok=None: None)
    monkeypatch.setattr(fe, "leer_memoria", lambda uid: True)
    monkeypatch.setattr(fe, "dequeue_pending_facts", lambda uid: [{"id": 1, "message": "soy celíaca"}])
    monkeypatch.setattr(fe, "_pending_item_too_old", lambda p: False)
    monkeypatch.setattr(fe, "_process_single_extraction", lambda *a, **k: pytest.fail("sin permiso no se procesa"))
    monkeypatch.setattr(fe, "delete_pending_facts", lambda ids: borrados.append(ids))
    fe.process_pending_queue_sync(UID)
    assert borrados == [], "el pendiente se conserva: si da el permiso dentro del plazo, se aprende entonces"


# ═════════════════════════════════════════════ 5. el coach proactivo y la respuesta a un aviso
def test_la_respuesta_a_un_aviso_no_se_clasifica_sin_permiso(monkeypatch, sin_permiso):
    import proactive_agent as pa
    monkeypatch.setattr(pa, "execute_sql_query", lambda *a, **k: {"id": 7, "nudge_type": "Cena"})
    monkeypatch.setattr(pa, "classify_nudge_sentiment", lambda *a, **k: pytest.fail("clasificar es IA"))
    monkeypatch.setattr(pa, "execute_sql_write", lambda *a, **k: pytest.fail("nada se marca"))
    pa.handle_nudge_response(UID, "no cené, estaba cansada")
    assert sin_permiso == [(UID, "respuesta_a_aviso")]


def test_el_coach_proactivo_sin_permiso_manda_el_aviso_fijo():
    """Sin permiso: ni embedding (Cohere) ni el prompt a la IA; el aviso fijo de su idioma (recordar no es IA)."""
    src = _src("proactive_agent.py")
    cuerpo = src[src.index("def run_proactive_checks("):]
    assert '_ia_ok = permite_ia(user_id, "coach_proactivo")' in cuerpo
    assert "context_embedding = get_embedding(context_summary) if _ia_ok else None" in cuerpo
    fijo = cuerpo.index("if not session_id or not _ia_ok:")
    assert fijo < cuerpo.index("chat_llm = ChatGLM(") < cuerpo.index("response = chat_llm.invoke(prompt)")
    assert cuerpo.index('_ia_ok = permite_ia(user_id, "coach_proactivo")') < cuerpo.index("get_embedding(")


def test_el_coach_proactivo_lee_el_permiso_despues_del_tope_diario():
    """[ronda de arreglo 1] La lectura va junto al embedding y al prompt, después de los `continue` del tope diario y de
    los demás filtros; hasta entonces `_ia_ok` es False (fail-closed). Las dos ramas que escriben un aviso la hacen."""
    src = _src("proactive_agent.py")
    cuerpo = src[src.index("def run_proactive_checks("):]
    lectura = '_ia_ok = permite_ia(user_id, "coach_proactivo")'
    assert cuerpo.count(lectura) == 2
    tope = cuerpo.index("if daily_nudges >= _tope_diario:")
    assert cuerpo.index("_ia_ok = False") < tope < cuerpo.index(lectura)
    resumen = cuerpo.index("no registró NADA. Generando nudge indulgente")
    comida = cuerpo.index("no registró {meal_to_check}. Generando mensaje")
    segunda = cuerpo.index(lectura, cuerpo.index(lectura) + 1)
    assert resumen < cuerpo.index(lectura) < comida < segunda < cuerpo.index("get_embedding(context_summary)")


def test_la_generacion_jit_muerta_no_revive_sin_permiso(monkeypatch, sin_permiso):
    import threading

    import proactive_agent as pa
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: pytest.fail("no arranca ningún hilo"))
    assert pa._trigger_week2_background_generation(UID, "plan-1", {}) is None


# ═════════════════════════════════════════════ 6. aprendizaje, retrospectiva y el título del plan
def test_el_aprendizaje_del_bloque_no_manda_lo_anotado_a_cohere_sin_permiso(monkeypatch):
    import cron_tasks as ct
    monkeypatch.setattr(ct, "get_embedding", lambda *a, **k: pytest.fail("sin permiso no hay embeddings"))
    dias = [{"meals": [{"name": "Mangú con huevo"}, {"name": "Arroz con habichuelas"}]}]
    consumo = [{"meal_name": "Pizza de la esquina"}, {"meal_name": "Tostadas"}]
    ct._calculate_chunk_consumption_ratio(dias, consumo, 0, usar_embeddings=False)
    src = _src("cron_tasks.py")
    llamada = src[src.index("ratio_info = _calculate_chunk_consumption_ratio("):][:400]
    assert 'usar_embeddings=__import__("consentimientos").permite_ia(user_id, "aprendizaje_del_bloque")' in llamada
    assert "if available_planned and usar_embeddings:" in src


def test_la_retrospectiva_semanal_pregunta_antes_de_la_ia():
    src = _src("cron_tasks.py")
    cuerpo = src[src.index("def _persist_nightly_learning_signals("):]
    gate = cuerpo.index('if run_retro and __import__("consentimientos").permite_ia(user_id, "retrospectiva_semanal"):')
    assert gate < cuerpo.index("llm_retro = generate_llm_retrospective(")


def test_el_titulo_del_plan_sin_permiso_es_el_determinista(monkeypatch, sin_permiso):
    import services
    monkeypatch.setattr(services, "generate_plan_title", lambda pd: pytest.fail("el título creativo es IA"))
    plan = {"calories": 1800, "days": [{"meals": [{"name": "Mangú con huevo"}]}]}
    assert services._titulo_del_plan(UID, plan) == "Mangú con huevo — 1800 kcal"
    monkeypatch.setattr(cs, "permite_ia", lambda uid, donde="": True)
    monkeypatch.setattr(services, "generate_plan_title", lambda pd: "Energía Serena")
    assert services._titulo_del_plan(UID, plan) == "Energía Serena"


def test_los_tres_caminos_del_titulo_pasan_por_el_gate():
    src = _src("services.py")
    assert src.count("_titulo_del_plan(user_id, plan_data)") == 3
    assert len(re.findall(r"(?<!def )generate_plan_title\(plan_data\)", src)) == 1, (
        "solo `_titulo_del_plan` llama directo al título creativo")
