# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-719 · 2026-09-28] El texto libre del Perfil Clínico Avanzado no se cuela sin revisión clínica.

Configuración → Perfil Clínico Avanzado pide «Cirugías, diagnósticos en estudio, indicaciones de tu médico… La IA
extrae lo relevante», y ese texto viaja LITERAL al prompt del plan («Contexto clínico en sus palabras»). Era el único
dato clínico que no vetaba el bypass del reviewer (`review_plan_node` solo lo veta con `_clinical_flags`): «tengo ERC
estadio 4, tomo warfarina» escrito ahí daba un plan sin revisión clínica, justo lo que P1-MEDICAL-SCOPE-GATE existe
para impedir. `clinical_profile_active_flags` (SSOT del veto) marca ahora `contexto_clinico_libre`.

tooltip-anchor: P1-PLAN-LOTE-719
"""
from prompts.plan_generator import build_clinical_profile_context, clinical_profile_active_flags


def _fd(cp):
    return {"clinical_profile": cp}


def test_texto_libre_clinico_veta_el_bypass_del_reviewer():
    flags = clinical_profile_active_flags(_fd({"freeText": "Tengo ERC estadio 4 y tomo warfarina"}))
    assert "contexto_clinico_libre" in flags


def test_texto_libre_vacio_o_en_blanco_no_marca():
    assert clinical_profile_active_flags(_fd({"freeText": ""})) == []
    assert clinical_profile_active_flags(_fd({"freeText": "   "})) == []


def test_el_texto_que_marca_es_el_mismo_que_llega_al_prompt():
    # Paridad flag ⟺ texto: si el builder dejara de emitir el texto libre, el flag no tendría qué revisar.
    cp = {"freeText": "Me quitaron la vesícula en 2024"}
    assert "contexto_clinico_libre" in clinical_profile_active_flags(_fd(cp))
    assert "Contexto clínico en sus palabras" in build_clinical_profile_context(_fd(cp))


def test_llega_tambien_anidado_en_health_profile():
    fd = {"health_profile": {"clinical_profile": {"freeText": "Diagnóstico de gastritis en estudio"}}}
    assert "contexto_clinico_libre" in clinical_profile_active_flags(fd)


# ---------------------------------------------------------------------------
# Invalidación de bloques pendientes: el objetivo vivía en `mainGoal` y se comparaba `goal`; condiciones, dieta y
# rechazos no invalidaban nada (Configuración → «Alergias y dieta» los edita desde este lote).
# ---------------------------------------------------------------------------
from db_profiles import _razones_para_invalidar  # noqa: E402


def test_cambiar_el_objetivo_del_formulario_invalida():
    assert "goal_changed" in _razones_para_invalidar({"mainGoal": "lose_fat"}, {"mainGoal": "gain_muscle"})


def test_condiciones_medicas_medicamentos_y_dieta_invalidan():
    assert "medical_changed" in _razones_para_invalidar({"medicalConditions": ["Ninguna"]}, {"medicalConditions": ["Enfermedad Renal"]})
    assert "medical_changed" in _razones_para_invalidar({"medications": []}, {"medications": ["Warfarina"]})
    assert "diet_changed" in _razones_para_invalidar({"dietType": "balanced"}, {"dietType": "vegetarian"})
    assert "dislikes_changed" in _razones_para_invalidar({"dislikes": ["Cilantro"]}, {"dislikes": []})


def test_alergias_otra_en_texto_tambien_invalida():
    assert "allergies_changed" in _razones_para_invalidar({"allergies": ["Ninguna"]}, {"allergies": ["Ninguna"], "otherAllergies": "Fresa"})


def test_sin_cambios_reales_no_invalida_nada():
    hp = {"mainGoal": "lose_fat", "allergies": ["Maní", "Gluten"], "medicalConditions": ["Ninguna"], "dietType": "balanced", "weight": 80}
    reordenado = {**hp, "allergies": ["gluten", "maní"], "weight": 82}
    assert _razones_para_invalidar(hp, reordenado) == []


# ---------------------------------------------------------------------------
# Memoria pausada: dos lectores que seguían consultando (P1-PLAN-LOTE-717 cerró el chat; aquí la tool y Dreaming).
# ---------------------------------------------------------------------------
def test_search_deep_memory_no_consulta_con_la_memoria_pausada_o_ilegible(monkeypatch):
    import memoria_largo_plazo
    import tools
    monkeypatch.setattr(memoria_largo_plazo, "memoria_activa", lambda uid, donde="": False)
    llamadas = []
    monkeypatch.setattr(tools, "get_user_profile", lambda uid: llamadas.append(uid) or {"long_term_memory_enabled": True})
    fn = getattr(tools.search_deep_memory, "func", tools.search_deep_memory)
    salida = fn(user_id="u-719", query="qué desayuno")
    assert "pausada" in salida.lower()


def test_dreaming_salta_al_usuario_con_la_memoria_pausada(monkeypatch):
    import dreaming
    import memoria_largo_plazo
    monkeypatch.setattr(memoria_largo_plazo, "memoria_activa", lambda uid, donde="": False)
    import db_facts
    monkeypatch.setattr(db_facts, "acquire_fact_lock", lambda uid: (_ for _ in ()).throw(AssertionError("no debe tocar el lock")))
    res = dreaming.consolidate_user("u-719")
    assert res["status"] == "skipped_memory_paused"


def test_la_busqueda_semantica_de_hechos_respeta_la_pausa(monkeypatch):
    """La generación de planes (RAG del contexto) seguía usando lo aprendido con la memoria pausada."""
    import db_facts
    import memoria_largo_plazo
    monkeypatch.setattr(db_facts, "connection_pool", object())
    monkeypatch.setattr(memoria_largo_plazo, "memoria_activa", lambda uid, donde="": False)
    monkeypatch.setattr(db_facts, "delete_expired_temporal_facts",
                        lambda uid: (_ for _ in ()).throw(AssertionError("no debe llegar a la base")))
    assert db_facts.search_user_facts("u-719", [0.1] * 4, query_text="desayuno") == []
