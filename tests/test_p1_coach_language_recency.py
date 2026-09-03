"""[P1-COACH-LANGUAGE-RECENCY · 2026-08-18] La directiva de idioma del coach se refuerza
como ÚLTIMO bloque del system prompt en AMBAS copias del chat.

Incidente (primer usuario real con locale='en-US', 2026-08-18 23:14 UTC, session
b9f147ca): el PATCH del idioma llegó a las 23:13:22, el chat a las 23:14:17 — la cadena
de datos estaba SANA (columna en-US, `get_user_profile` la proyecta, la directiva de T3
se apendeó sin excepción) y aun así el coach respondió en español. Causa: la directiva
vivía a mitad de un system prompt gigante 100% español, seguida de ~40 bloques más
(plan JSON, culinary KB, tools, RAG), y el usuario escribió «hola» — la señal dominante
ganó a una instrucción enterrada. Recency manda: la MISMA directiva (SSOT
`build_language_directive`, cacheada por variante), repetida como último bloque antes
del invoke, es lo último que el modelo lee.

Parser-based sobre agent.py: si alguien mueve el refuerzo o inserta bloques después,
esto se pone rojo ANTES de que otro usuario en-US reciba español.
"""
import os
import re

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


def _fn_body(src: str, header_startswith: str) -> str:
    i = src.index(header_startswith)
    nxt = src.find("\ndef ", i + 1)
    return src[i:nxt if nxt != -1 else len(src)]


@pytest.mark.parametrize("header", [
    "def chat_with_agent(",
    "def chat_with_agent_stream(",
])
def test_directiva_aparece_dos_veces_y_la_ultima_es_el_ultimo_bloque(header):
    src = _read("agent.py")
    body = _fn_body(src, header)
    calls = [m.start() for m in re.finditer(r"build_language_directive\(_coach_locale\)", body)]
    assert len(calls) >= 2, (
        f"{header} esperaba ≥2 apéndices de la directiva (T3 temprano + refuerzo final "
        f"P1-COACH-LANGUAGE-RECENCY), hay {len(calls)} — sin el refuerzo final el modelo "
        f"desobedece la directiva enterrada (incidente en-US 2026-08-18)"
    )
    # El REFUERZO debe ser lo último apendeado antes del config/invoke: ningún
    # `system_prompt +=` puede aparecer entre el último apéndice de la directiva y
    # `config = {"configurable"`.
    cfg = body.index('config = {"configurable"')
    tail = body[calls[-1]:cfg]
    otros_appends = re.findall(r"system_prompt \+= (?!build_language_directive)", tail)
    assert not otros_appends, (
        f"{header}: hay {len(otros_appends)} bloque(s) apendeado(s) DESPUÉS del refuerzo "
        f"final de idioma — el refuerzo debe ser el ÚLTIMO bloque (recency es el fix)"
    )
    assert calls[-1] < cfg, "el refuerzo debe vivir antes del config del invoke"


def test_marker_del_refuerzo_presente():
    src = _read("agent.py")
    assert src.count("P1-COACH-LANGUAGE-RECENCY") >= 2, (
        "los DOS refuerzos (stream y no-stream) llevan el marker — si quitaste uno, "
        "la divergencia entre copias ya costó bugs (P1-CHAT-PAST-DAYS)"
    )
