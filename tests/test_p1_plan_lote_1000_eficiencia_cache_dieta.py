# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1000 · 2026-11-13] Eficiencia, latencia, dieta de contexto y guardarraíl de gasto LLM.

Cubre las 4 mejoras compatibles con la arquitectura activa (fotos en Gemini 4 Hydrogen, coach en gpt-6-luna):
  1. Ordenación cache-first del system prompt (`MEALFIT_CHAT_PROMPT_CACHE_ORDER`): Banda 1 estable -> Banda 2 del
     día -> Banda 3 volátil -> cola, preservando intactos el prefijo estático, el bloque clínico y la contabilidad de
     `prompt_desglose`.
  2. Fast-path de sentimiento en registros presentes objetivos (`MEALFIT_CHAT_SKIP_SENTIMENT_PRESENT_LOGS`) + nuevos
     disparadores naturales (`acabo de ...`, `anótame`, `apúntame`), manteniendo `classify_sentiment` cuando hay señales
     afectivas o consultas de planeación.
  3. Dieta de contexto por turno (`MEALFIT_CHAT_DIET_SUPLEMENTOS`, `MEALFIT_CHAT_DIET_PASADOS_INGREDIENTES`): compacta
     el manual fijo de suplementos cuando no hay potes/plan/foto/mención y recorta `(ingredientes: ...)` solo en
     `DIARIO REAL DE DÍAS ANTERIORES` cuando el turno no los necesita (`DIARIO DE HOY` intacto).
  4. Telemetría en nodos auxiliares (`chat_sentiment`, `chat_rag_router`, `chat_title`), resolución limpia a
     `gpt-6-luna` en `_chat_agent_model_name`, y cron `_llm_daily_spend_alert_job` con resolución automática.

tooltip-anchor: P1-PLAN-LOTE-1000
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent
import chat_prompt_opt as cpo
import cron_tasks
import prompt_desglose as pd
from knobs import _KNOBS_REGISTRY
from prompts.sentiment import PERSONALITY_PROFILES
import suplementos as sp

_BACKEND_DIR = Path(__file__).resolve().parent.parent


# ── 1. Cache-first prompt ordering + prompt_desglose ────────────────────────────────────────────────────────────────


def _armar_prompt_desordenado() -> tuple[str, int]:
    """Simula el orden histórico del prompt dinámico (volátil arriba, perfil/clínica en medio, pasado abajo)."""
    prefijo = "SYSTEM MANUAL ESTATICO BYTE A BYTE IDENTICO\n" + ("R" * 120)
    cut = len(prefijo)
    bloques = [
        "\n<!--MEALFIT_VOLATILE_SYS_SPLIT-->\nEl user_id actual es: u1",
        "\n\n🕒 CONTEXTO TEMPORAL ACTUAL: Hoy es jueves 2026-11-13, 14:22.",
        "\n⚠️ RITMO CIRCADIANO: Tarde activa.",
        "\n🌟 REGLA DE CONTINUIDAD TEMPORAL PROACTIVA: sugiere cena.",
        "\n\n🛒 ESTADO DE LA DESPENSA Y COMPRAS: pollo, arroz.",
        "\n--- SÚPER PERSONALIZACIÓN DEL USUARIO ---\nPrefiere respuestas directas.",
        "\n\n--- 👤 PERFIL DEL USUARIO (solo lectura) ---\nNombre: Angel.",
        "\n\n🛑 PERFIL CLÍNICO DEL USUARIO (PROTEGER CONTRA ALERGIAS): sin alergias.",
        "\n\n🎯 META DIARIA: 2200 kcal / 160 g proteína.",
        "\n\n🌐 RESPONSE LANGUAGE: Español.",
        "\n\nCONTEXTO CRÍTICO: El usuario tiene plan de definición activo.",
        sp.BLOQUE_CONOCIMIENTO,
        "\n\n<contexto_evolutivo_historico>\nTendencia estable.</contexto_evolutivo_historico>",
        "\n\nDIARIO DE HOY:\n- Desayuno: 2 huevos (ingredientes: huevo x2, aceite 5 ml) — 180 kcal",
        "\n\n💧 HIDRATACIÓN HOY: 1.5 L / 2.5 L.",
        "\n\n🧊 NEVERA: pechuga 500 g, brócoli 300 g.",
        (
            "\n\n🍽️ DIARIO REAL DE DÍAS ANTERIORES (SOLO LECTURA):\n"
            "  2026-11-12:\n"
            "  - Almuerzo: pollo guisado (ingredientes: pechuga 200 g, arroz 150 g, aceite 10 ml) — 540 kcal\n"
            "  - Cena: yogur griego (ingredientes: yogur 200 g, chía 10 g) — 210 kcal"
        ),
        "\n\nPERSONALIDAD ACTIVA: directa.",
        "\n--- MEMORIA VECTORIAL (RAG) ---\nLe gustan las arepas.",
        "\n\n📷 CONTEXTO DE FOTO: Plato detectado.",
        "\n\nRECORDATORIO FINAL — MODO VOZ: respuestas cortas.",
    ]
    return prefijo + "".join(bloques), cut


def test_reordenar_bloques_cache_mantiene_prefijo_y_ordena_tres_bandas(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_CACHE_ORDER", "1")
    original, cut = _armar_prompt_desordenado()

    rebuilt = cpo.reordenar_bloques_cache(original, prefijo_estatico_chars=cut)

    assert len(rebuilt) == len(original)
    assert rebuilt[:cut] == original[:cut]

    pos_super = rebuilt.index("SÚPER PERSONALIZACIÓN")
    pos_clinico = rebuilt.index("🛑 PERFIL CLÍNICO")
    pos_meta = rebuilt.index("🎯 META DIARIA")
    pos_supl = rebuilt.index("💊 SUPLEMENTOS (sabes esto")
    pos_mem = rebuilt.index("<contexto_evolutivo_historico>")
    pos_inv = rebuilt.index("🛒 ESTADO DE LA DESPENSA")
    pos_hoy = rebuilt.index("DIARIO DE HOY:")
    pos_pas = rebuilt.index("🍽️ DIARIO REAL DE DÍAS ANTERIORES")
    pos_tmp = rebuilt.index("🕒 CONTEXTO TEMPORAL ACTUAL")
    pos_circ = rebuilt.index("⚠️ RITMO CIRCADIANO")
    pos_sent = rebuilt.index("PERSONALIDAD ACTIVA:")
    pos_rag = rebuilt.index("--- MEMORIA VECTORIAL (RAG) ---")
    pos_vis = rebuilt.index("📷 CONTEXTO DE FOTO:")
    pos_voz = rebuilt.index("RECORDATORIO FINAL — MODO VOZ")

    # Banda 1 (perfil / clínica / metas / suplementos / memoria histórica) antes de Banda 2 (día)
    assert max(pos_super, pos_clinico, pos_meta, pos_supl, pos_mem) < min(pos_inv, pos_hoy, pos_pas)
    # Banda 2 (día / nevera / pasado) antes de Banda 3 (hora min-a-min / sentimiento / RAG)
    assert max(pos_inv, pos_hoy, pos_pas) < min(pos_tmp, pos_circ, pos_sent, pos_rag)
    # Cola de turno (visión / modo voz) al final
    assert max(pos_tmp, pos_circ, pos_sent, pos_rag) < min(pos_vis, pos_voz)


def test_reordenar_bloques_cache_apagado_o_sin_prefijo_devuelve_igual(monkeypatch):
    original, cut = _armar_prompt_desordenado()
    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_CACHE_ORDER", "0")
    assert cpo.reordenar_bloques_cache(original, prefijo_estatico_chars=cut) == original

    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_CACHE_ORDER", "1")
    assert cpo.reordenar_bloques_cache(original, prefijo_estatico_chars=0) == original


def test_desglose_integra_bloques_reordenados_y_dieta_sin_descuadre(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_CACHE_ORDER", "1")
    monkeypatch.setenv("MEALFIT_CHAT_DIET_SUPLEMENTOS", "1")
    monkeypatch.setenv("MEALFIT_CHAT_DIET_PASADOS_INGREDIENTES", "1")

    original, cut = _armar_prompt_desordenado()
    # Sin bloque foto para probar también dieta de suplementos
    sin_foto = original.replace("\n\n📷 CONTEXTO DE FOTO: Plato detectado.", "")
    optimizado = cpo.optimizar_prompt_dinamico(
        sin_foto,
        "me desayuné 2 huevos revueltos",
        prefijo_estatico_chars=cut,
        form_data={},
        con_foto=False,
    )

    bd = pd.desglosar_prompt(optimizado, prefijo_estatico_chars=cut)
    assert bd is not None
    bloques = bd["bloques"]
    assert bd["prefijo_estatico"] == cut
    assert bd["total_sys_chars"] == len(optimizado)
    assert sum(bloques.values()) <= len(optimizado)
    assert all(v > 0 for v in bloques.values())
    assert "clinico" in bloques
    assert "suplementos_alacena" in bloques
    assert "temporal" in bloques
    assert bloques["suplementos_alacena"] < len(sp.BLOQUE_CONOCIMIENTO)


# ── 2. Dieta de contexto por turno (suplementos e ingredientes históricos) ───────────────────────────────────────────


def test_dieta_suplementos_recorta_sin_contexto_y_preserva_cuando_aplica(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_DIET_SUPLEMENTOS", "1")
    base = "PREFIJO\n\n--- 👤 PERFIL DEL USUARIO ---\nAngel." + sp.BLOQUE_CONOCIMIENTO + "\n\nDIARIO DE HOY:\nNada."

    recortado = cpo.adelgazar_suplementos_en_prompt(base, "me cené un pollo con ensalada", form_data={}, con_foto=False)
    assert sp.BLOQUE_CONOCIMIENTO not in recortado
    assert "💊 SUPLEMENTOS (sabes esto)" in recortado
    assert len(base) - len(recortado) >= 900

    # Conserva completo cuando hay foto (puede mostrar etiqueta de suplemento)
    assert cpo.adelgazar_suplementos_en_prompt(base, "mira esto", form_data={}, con_foto=True) == base
    # Conserva completo cuando el prompt menciona un suplemento
    assert cpo.adelgazar_suplementos_en_prompt(base, "me tomé un scoop de creatina", form_data={}, con_foto=False) == base
    # Conserva completo cuando el plan ya tiene suplementos activos
    assert cpo.adelgazar_suplementos_en_prompt(
        base, "hola", form_data={"currentSupplements": ["creatina"]}, con_foto=False
    ) == base
    # Conserva completo cuando hay alacena de suplementos en el prompt
    con_alacena = base + "\n\nSU ALACENA: Creatina monohidrato (1 scoop = 5 g)."
    assert cpo.adelgazar_suplementos_en_prompt(con_alacena, "hola", form_data={}, con_foto=False) == con_alacena

    # Kill-switch apaga la dieta
    monkeypatch.setenv("MEALFIT_CHAT_DIET_SUPLEMENTOS", "0")
    assert cpo.adelgazar_suplementos_en_prompt(base, "me cené pollo", form_data={}, con_foto=False) == base


def test_dieta_ingredientes_pasados_preserva_diario_de_hoy_y_respeta_consultas(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_DIET_PASADOS_INGREDIENTES", "1")
    sys_prompt = (
        "PREFIJO\n\n"
        "DIARIO DE HOY:\n"
        "  - Desayuno: tortilla (ingredientes: huevo x3, espinaca 50 g) — 260 kcal\n\n"
        "🍽️ DIARIO REAL DE DÍAS ANTERIORES (SOLO LECTURA):\n"
        "  2026-11-12:\n"
        "  - Almuerzo: pollo guisado (ingredientes: pechuga 200 g, arroz 150 g) — 540 kcal\n"
        "  - Cena: yogur (ingredientes: yogur 200 g, chía 10 g) — 210 kcal\n\n"
        "🕒 CONTEXTO TEMPORAL ACTUAL: Hoy es jueves."
    )

    adelgazado = cpo.adelgazar_ingredientes_dias_pasados(sys_prompt, "me desayuné 2 huevos revueltos")
    # Hoy intacto
    assert "egg" in adelgazado or "huevo x3, espinaca 50 g" in adelgazado
    # Pasado conserva nombres, horarios y calorías, pero sin paréntesis de ingredientes
    assert "Almuerzo: pollo guisado — 540 kcal" in adelgazado
    assert "pechuga 200 g, arroz 150 g" not in adelgazado
    assert "🕒 CONTEXTO TEMPORAL ACTUAL: Hoy es jueves." in adelgazado

    # Turnos que preguntan por ayer / ingredientes / repetir / corregir reciben todo sin recorte
    for p_keep in (
        "¿qué ingredientes tenía el pollo de ayer?",
        "quiero repetir lo mismo del almuerzo pasado",
        "corrige mi cena del lunes",
    ):
        assert cpo.adelgazar_ingredientes_dias_pasados(sys_prompt, p_keep) == sys_prompt

    # Kill-switch
    monkeypatch.setenv("MEALFIT_CHAT_DIET_PASADOS_INGREDIENTES", "0")
    assert cpo.adelgazar_ingredientes_dias_pasados(sys_prompt, "me desayuné 2 huevos") == sys_prompt


# ── 3. Fast-path de sentimiento en registros objetivos + triggers naturales ────────────────────────────────────────


@pytest.mark.parametrize("texto", [
    "me desayuné 2 huevos revueltos",
    "acabo de cenar pollo con arroz",
    "me acabo de almorzar salmón con papa",
    "anótame un café con leche",
    "apúntame 200 ml de agua",
    "cuántas calorías me quedan hoy?",
])
def test_present_log_y_fast_path_sin_sentimiento(texto):
    assert agent._rag_es_presente_sin_rag(texto) is True
    assert cpo.es_turno_objetivo_sin_sentimiento(texto) is True


@pytest.mark.parametrize("texto", [
    "me desayuné 2 huevos pero estoy muy frustrado",
    "acabo de cenar, me siento cansado hoy",
    "anótame 1 taza de avena!!",
    "me comí una pizza 😔",
    "",
    "x " * 130,
])
def test_fast_path_conserva_sentimiento_si_hay_indicio_afectivo_o_largo(texto):
    assert cpo.es_turno_objetivo_sin_sentimiento(texto) is False


class _StreamPromptCapturado(Exception):
    pass


def _ejecutar_stream_con_mocks(monkeypatch, *, prompt: str):
    import db_core
    import db_inventory
    import memoria_largo_plazo as mlp

    calls = {"sentimiento": 0, "embedding": 0, "sys_prompt": None}

    class _Grafo:
        def get_state(self, _cfg):
            return SimpleNamespace(values={})

        def invoke(self, inputs, **_k):
            calls["sys_prompt"] = inputs["sys_prompt"]
            raise _StreamPromptCapturado

        stream = invoke

    class _Builder:
        def compile(self, **_k):
            return _Grafo()

    def _sent(_p):
        calls["sentimiento"] += 1
        return {**PERSONALITY_PROFILES["neutral"], "sentiment": "neutral"}

    def _emb(_q):
        calls["embedding"] += 1
        return None

    monkeypatch.setattr(agent, "chat_builder", _Builder())
    monkeypatch.setattr(agent, "chat_checkpoint_pool", None)
    monkeypatch.setattr(agent, "connection_pool", None)
    monkeypatch.setattr(db_core, "connection_pool", None)
    monkeypatch.setattr(agent, "classify_sentiment", _sent)
    monkeypatch.setattr(agent, "get_embedding", _emb)
    monkeypatch.setattr(agent, "get_multimodal_embedding", lambda _q: None)
    monkeypatch.setattr(agent, "build_user_identity_context", lambda *_a, **_k: "\n\n--- 👤 PERFIL DEL USUARIO ---\nAngel.")
    monkeypatch.setattr(agent, "build_clinical_guard_context", lambda *_a, **_k: "\n\n🛑 PERFIL CLÍNICO DEL USUARIO: ok.")
    monkeypatch.setattr(agent, "build_memory_context", lambda *_a, **_k: {"recent_messages": [], "summary_context": ""})
    monkeypatch.setattr(agent, "_safe_suplementos_bloque_para_chat", lambda *_a, **_k: sp.BLOQUE_CONOCIMIENTO)
    monkeypatch.setattr(mlp, "memoria_activa", lambda *_a, **_k: True)
    monkeypatch.setattr(db_inventory, "get_user_inventory", lambda *_a, **_k: {})

    try:
        list(agent.chat_with_agent_stream(
            prompt=prompt,
            user_id="user-test-1000",
            session_id="thread-test-1000",
            is_call_mode=False,
        ))
    except _StreamPromptCapturado:
        pass
    return calls


def test_stream_salta_sentimiento_en_registro_presente_y_lo_corre_si_hay_emocion(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_SKIP_SENTIMENT_PRESENT_LOGS", "1")
    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_CACHE_ORDER", "1")

    c_obj = _ejecutar_stream_con_mocks(monkeypatch, prompt="me desayuné 2 huevos revueltos")
    assert c_obj["sentimiento"] == 0
    assert c_obj["embedding"] == 0
    assert c_obj["sys_prompt"] is not None
    # Bloque clínico presente en role=system y suplementos compactados
    assert "🛑 PERFIL CLÍNICO" in c_obj["sys_prompt"]
    assert sp.BLOQUE_CONOCIMIENTO not in c_obj["sys_prompt"]
    assert "💊 SUPLEMENTOS (sabes esto)" in c_obj["sys_prompt"]

    # Con señal emocional sí corre classify_sentiment (pero sigue evitando RAG)
    c_emo = _ejecutar_stream_con_mocks(monkeypatch, prompt="me desayuné 2 huevos pero estoy muy frustrado")
    assert c_emo["sentimiento"] == 1
    assert c_emo["embedding"] == 0

    # Pregunta de planeación conserva sentimiento y búsqueda RAG (contrato Lote 684)
    c_plan = _ejecutar_stream_con_mocks(monkeypatch, prompt="¿qué puedo cenar hoy?")
    assert c_plan["sentimiento"] == 1
    assert c_plan["embedding"] == 1

    # Kill-switch devuelve el comportamiento previo para registros presentes
    monkeypatch.setenv("MEALFIT_CHAT_SKIP_SENTIMENT_PRESENT_LOGS", "0")
    c_off = _ejecutar_stream_con_mocks(monkeypatch, prompt="me desayuné 2 huevos revueltos")
    assert c_off["sentimiento"] == 1
    assert c_off["embedding"] == 0


# ── 4. Telemetría en nodos auxiliares, naming no-Hydrogen y cron de gasto diario ─────────────────────────────────


def test_registrar_uso_llm_auxiliar_emite_y_no_rompe_si_falla(monkeypatch):
    import llm_telemetry

    emitidos = []
    monkeypatch.setattr(
        llm_telemetry,
        "_emit_llm_usage_event_best_effort",
        lambda **kw: emitidos.append(kw),
    )

    llm_fake = object()
    res_fake = SimpleNamespace(content="ok", usage_metadata={"input_tokens": 50, "output_tokens": 10})
    cpo.registrar_uso_llm_auxiliar(llm=llm_fake, result=res_fake, duration_s=0.42, node="chat_sentiment")
    assert len(emitidos) == 1
    assert emitidos[0]["node"] == "chat_sentiment"
    assert emitidos[0]["duration_s"] == pytest.approx(0.42)

    # Best-effort: nunca propaga excepciones si falla el emisor
    monkeypatch.setattr(
        llm_telemetry,
        "_emit_llm_usage_event_best_effort",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("db down")),
    )
    cpo.registrar_uso_llm_auxiliar(llm=llm_fake, result=res_fake, duration_s=0.1, node="chat_title")


def test_nodos_auxiliares_cablean_telemetria_por_nombre_de_nodo():
    agent_src = (_BACKEND_DIR / "agent.py").read_text(encoding="utf-8")
    sent_src = (_BACKEND_DIR / "sentiment_classifier.py").read_text(encoding="utf-8")
    assert 'node="chat_title"' in agent_src
    assert 'node="chat_rag_router"' in agent_src
    assert 'node="chat_sentiment"' in sent_src


def test_chat_agent_model_name_resuelve_gpt6_luna_limpio_en_free_y_env_directo(monkeypatch):
    monkeypatch.delenv("MEALFIT_CHAT_AGENT_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1000")
    monkeypatch.setenv("MEALFIT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MEALFIT_CHAT_HYDROGEN_TIERS", "plus")
    monkeypatch.setattr("db_profiles.get_user_plan_tier", lambda _uid: "free")

    assert agent._chat_agent_model_name(user_id="user-free-1") == "gpt-6-luna"

    # Y un fix explícito de entorno sigue ganando a todo
    monkeypatch.setenv("MEALFIT_CHAT_AGENT_MODEL", "gpt-6-luna")
    monkeypatch.setattr("db_profiles.get_user_plan_tier", lambda _uid: "plus")
    assert agent._chat_agent_model_name(user_id="user-plus-1") == "gpt-6-luna"


def test_llm_daily_spend_alert_job_emite_warning_critical_y_autocierra(monkeypatch):
    monkeypatch.setenv("MEALFIT_LLM_DAILY_SPEND_ALERT_ENABLED", "1")
    monkeypatch.setenv("MEALFIT_LLM_DAILY_SPEND_ALERT_USD", "5.0")
    monkeypatch.setenv("MEALFIT_LLM_DAILY_SPEND_WINDOW_HOURS", "24")

    writes = []
    # Turno 1: Gemini en $6.25 (warning >= 5) y gpt-6-luna en $16.00 (critical >= 15)
    monkeypatch.setattr(
        cron_tasks,
        "execute_sql_query",
        lambda *_a, **_k: [
            {"model": "gemini-4-hydrogen", "spend_micros": 6_250_000, "calls": 180},
            {"model": "gpt-6-luna", "spend_micros": 16_000_000, "calls": 4000},
            {"model": "gpt-5-nano", "spend_micros": 50_000, "calls": 90},
        ],
    )
    monkeypatch.setattr(cron_tasks, "execute_sql_write", lambda sql, params=(): writes.append((sql, params)))

    cron_tasks._llm_daily_spend_alert_job()

    alert_inserts = [w for w in writes if "INSERT INTO system_alerts" in w[0]]
    assert len(alert_inserts) == 2
    by_key = {p[0]: p for _, p in alert_inserts}
    assert by_key["llm_daily_spend_high:gemini-4-hydrogen"][1] == "warning"
    assert by_key["llm_daily_spend_high:gpt-6-luna"][1] == "critical"

    # Debe limpiar los modelos que ya no exceden el umbral
    cleanups = [w for w in writes if "UPDATE system_alerts SET resolved_at = NOW()" in w[0]]
    assert len(cleanups) == 1
    assert set(cleanups[0][1][0]) == {
        "llm_daily_spend_high:gemini-4-hydrogen",
        "llm_daily_spend_high:gpt-6-luna",
    }

    # Debe registrar latido en pipeline_metrics
    beats = [w for w in writes if "llm_daily_spend_watch" in w[0]]
    assert len(beats) == 1
    assert beats[0][1][0] == 2.0

    # Turno 2: todo bajo el umbral -> auto-cierra alertas abiertas
    writes.clear()
    monkeypatch.setattr(cron_tasks, "execute_sql_query", lambda *_a, **_k: [])
    cron_tasks._llm_daily_spend_alert_job()
    assert not [w for w in writes if "INSERT INTO system_alerts" in w[0]]
    assert any("UPDATE system_alerts SET resolved_at = NOW()" in w[0] for w in writes)

    # Turno 3: fallo en DB -> fail-open (no propaga excepción) y deja rastro `job_failed: True`
    writes.clear()
    monkeypatch.setattr(
        cron_tasks,
        "execute_sql_query",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("pg down")),
    )
    cron_tasks._llm_daily_spend_alert_job()
    beats_err = [w for w in writes if "llm_daily_spend_watch" in w[0]]
    assert len(beats_err) == 1
    assert '"job_failed": true' in beats_err[0][1][1]


# ── 5. Marcador + registro de knobs P3-NEW-D ─────────────────────────────────────────────────────────────────────


def test_lote_1000_marker_y_knobs_registrados(monkeypatch):
    app_src = (_BACKEND_DIR / "app.py").read_text(encoding="utf-8")
    m = re.search(r'^_LAST_KNOWN_PFIX\s*=\s*"P1-PLAN-LOTE-(\d+)\b', app_src, re.MULTILINE)
    assert m is not None and int(m.group(1)) >= 1000

    # Evaluar todos los knobs nuevos para asegurar su auto-registro en _KNOBS_REGISTRY
    cpo.cache_order_activo()
    cpo.skip_sentiment_present_logs_activo()
    cpo.diet_suplementos_activo()
    cpo.diet_pasados_ingredientes_activo()
    monkeypatch.setattr(cron_tasks, "execute_sql_query", lambda *_a, **_k: [])
    monkeypatch.setattr(cron_tasks, "execute_sql_write", lambda *_a, **_k: None)
    cron_tasks._llm_daily_spend_alert_job()

    esperados = (
        "MEALFIT_CHAT_PROMPT_CACHE_ORDER",
        "MEALFIT_CHAT_SKIP_SENTIMENT_PRESENT_LOGS",
        "MEALFIT_CHAT_DIET_SUPLEMENTOS",
        "MEALFIT_CHAT_DIET_PASADOS_INGREDIENTES",
        "MEALFIT_LLM_DAILY_SPEND_ALERT_ENABLED",
        "MEALFIT_LLM_DAILY_SPEND_ALERT_USD",
        "MEALFIT_LLM_DAILY_SPEND_WINDOW_HOURS",
    )
    for k in esperados:
        assert k in _KNOBS_REGISTRY, f"{k} falta en _KNOBS_REGISTRY"

    cron_src = (_BACKEND_DIR / "cron_tasks.py").read_text(encoding="utf-8")
    assert '_env_int("MEALFIT_LLM_DAILY_SPEND_ALERT_INTERVAL_MIN"' in cron_src
