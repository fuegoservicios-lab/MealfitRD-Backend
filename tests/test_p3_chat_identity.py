"""[P3-CHAT-IDENTITY · 2026-06-20] El chat coach conoce la IDENTIDAD + datos
corporales del usuario.

Antes, el system prompt del chat (agent.py) inyectaba plan/inventario/súper-
personalización/facts, pero NO el nombre/sexo/edad/peso/altura/objetivo del
perfil → el agente no te saludaba por tu nombre ni sabía que eres hombre/mujer
salvo que se lo dijeras en el chat. Este fix añade un bloque compacto de
identidad construido desde `form_data` (health_profile) + `full_name`
(user_profiles), inyectado en AMBOS paths del chat (stream + no-stream) justo
después de la súper-personalización.

Contrato verificado:
  - render completo de los 6 campos (nombre/sexo/edad/peso/altura/objetivo).
  - acepta strings (el form los manda como string).
  - no-op ("") cuando no hay datos accionables.
  - fallback legible para objetivos no mapeados.
  - bloque ADITIVO/NO clínico: declara que NO altera alergias/condiciones/macros.
  - anchor: inyectado en los dos paths del chat en agent.py.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prompts.chat_agent import build_user_identity_context


def test_empty_is_noop():
    assert build_user_identity_context(None, "") == ""
    assert build_user_identity_context({}, "") == ""
    # gender inválido + age vacío + sin nombre → nada accionable
    assert build_user_identity_context({"gender": "x", "age": ""}, "") == ""


def test_full_block_renders_all_fields():
    out = build_user_identity_context(
        {"gender": "male", "age": 21, "weight": 124, "weightUnit": "lb",
         "height": 175, "mainGoal": "gain_muscle"},
        "angelo brito",
    )
    assert "PERFIL DEL USUARIO" in out
    assert "angelo brito" in out
    assert "Hombre" in out
    assert "21 años" in out
    assert "124 lb" in out
    assert "175 cm" in out
    assert "ganar músculo" in out


def test_accepts_string_values_from_form():
    """El form persiste edad/peso/altura como strings → deben parsearse."""
    out = build_user_identity_context(
        {"gender": "male", "age": "30", "weight": "80.5", "weightUnit": "kg", "height": "180"},
        "Juan",
    )
    assert "30 años" in out
    assert "80.5 kg" in out
    assert "180 cm" in out


def test_female_and_goal_fallback():
    out = build_user_identity_context({"gender": "female", "goal": "tone_up"}, "")
    assert "Mujer" in out
    assert "tone up" in out  # objetivo no mapeado → underscores reemplazados


def test_name_only():
    out = build_user_identity_context({}, "Maria")
    assert "Maria" in out and "PERFIL DEL USUARIO" in out


def test_block_declares_non_clinical_contract():
    """El bloque debe declarar que NO altera alergias/condiciones/macros —
    invariante 'aditivo, no clínico'."""
    out = build_user_identity_context({"gender": "male"}, "X")
    low = out.lower()
    assert "no uses este bloque para alterar" in low
    assert "alergias" in low


def test_injected_in_both_chat_paths():
    """Anchor: el builder se inyecta en chat_with_agent Y chat_with_agent_stream
    (los dos surfaces del chat coach). Un renombre rompe el test antes que prod."""
    path = os.path.join(os.path.dirname(__file__), "..", "agent.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    n = src.count("build_user_identity_context(form_data or {}, _id_name)")
    assert n >= 2, (
        f"Esperaba ≥2 inyecciones de build_user_identity_context en agent.py "
        f"(stream + no-stream); hallé {n}."
    )
    assert "from prompts.chat_agent import" in src and "build_user_identity_context" in src
