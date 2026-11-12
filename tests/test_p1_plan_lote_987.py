"""[P1-PLAN-LOTE-987 · 2026-11-12] «Dos paquetes de las de ayer» no se pregunta: la porción es la del registro.

Caso vivo del dueño (02:11): ayer anotó «3 galletas Crakeñas CLUB (24 g)»; hoy «me metí dos paquete de galletas de
las de ayer» y el coach preguntó «¿paquetes individuales o grandes? ¿cuántas galletas?» por la regla de envasados.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def test_la_excepcion_llega_a_todos_los_prompts_del_chat():
    import prompts.chat_agent as c
    for nombre in ("CHAT_SYSTEM_PROMPT_BASE", "CHAT_STREAM_SYSTEM_PROMPT_BASE", "CHAT_AGENT_INLINE_PROMPT",
                   "CHAT_STREAM_INLINE_PROMPT", "CHAT_VOICE_MODE_PROMPT"):
        txt = getattr(c, nombre)
        assert "dos paquetes de las de ayer" in txt and "NO preguntes" in txt, nombre


def test_va_detras_de_la_regla_de_envasados():
    import prompts.chat_agent as c
    r = c._CHAT_RESOLVE_RULES
    assert r.index("ALIMENTOS ENVASADOS: distingue") < r.index("[P1-PLAN-LOTE-987]")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 987
