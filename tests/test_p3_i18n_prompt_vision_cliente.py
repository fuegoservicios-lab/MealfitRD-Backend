"""[P3-I18N-PROMPT-VISION-CLIENTE-ESPANOL · 2026-08-23] El cliente (`AgentPage.jsx`) metía
cuatro bloques de instrucciones en español («[Sistema: …] Instrucción: …») DENTRO del turno
del usuario cuando había una foto: el modelo los leía como si el usuario hablara español — la
señal más fuerte hacia el español, justo la que `build_language_directive` intenta vencer.

Cierre: el cliente manda `vision: {kind, description, reason, has_text}` y el servidor compone
el bloque (`prompts.chat_agent.build_vision_context`, en español como el resto del system
prompt, que es español entero por diseño) y lo añade al SYSTEM prompt en las dos ramas de
`chat_with_agent_stream`. El turno del usuario vuelve a ser sólo lo suyo.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FRONT = _BACKEND.parent / "frontend"
_MARKER = "P3-I18N-PROMPT-VISION-CLIENTE-ESPANOL"


@pytest.fixture(scope="module")
def build():
    from prompts.chat_agent import build_vision_context
    return build_vision_context


@pytest.mark.parametrize("vision,fragmento", [
    ({"kind": "unavailable", "reason": "busy", "has_text": True}, "procesando otra foto"),
    ({"kind": "unavailable", "reason": "down"}, "no está disponible"),
    ({"kind": "otro", "description": "un perro"}, "NO detectó comida"),
    ({"kind": "items", "description": "2 manzanas"}, "ALIMENTOS SUELTOS"),
    ({"kind": "items", "description": "2 manzanas", "has_text": True}, "Responde a su mensaje"),
    ({"kind": "plato", "description": "arroz con pollo"}, "DIARIO DE HOY"),
    ({"kind": "plato", "description": "arroz con pollo", "has_text": True}, "teniendo en cuenta la foto"),
])
def test_cada_modo_compone_su_bloque(build, vision, fragmento):
    out = build(vision)
    assert out.startswith("\n\n📷 CONTEXTO DE FOTO:"), out[:40]
    assert fragmento in out, out[:200]
    assert "[Sistema" not in out, "el prefijo [Sistema] era del turno del usuario; en el system prompt sobra"


def test_sin_foto_no_hay_bloque(build):
    assert build(None) == ""
    assert build({}) == ""
    assert build("plato") == ""


def test_la_descripcion_se_acota(build):
    out = build({"kind": "plato", "description": "x" * 5000})
    assert len(out) < 3200


def test_el_bloque_va_al_system_prompt_en_las_dos_ramas():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "vision: Optional[dict] = None" in src.split("def chat_with_agent_stream", 1)[1][:600]
    cuerpo = src.split("def chat_with_agent_stream", 1)[1]
    assert cuerpo.count("system_prompt += build_vision_context(vision)") == 2, (
        f"el bloque de la foto tiene que ir en la rama static-prefix Y en la legacy [{_MARKER}]")
    router = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")
    assert 'data.get("vision")' in router and "vision=vision," in router


def test_el_cliente_ya_no_compone_instrucciones_en_el_turno_del_usuario():
    p = _FRONT / "src" / "pages" / "AgentPage.jsx"
    if not p.exists():
        pytest.skip("frontend no está en este checkout")
    src = p.read_text(encoding="utf-8")
    codigo = "\n".join(l for l in src.splitlines() if not l.strip().startswith("//"))
    # La COMPOSICIÓN (un template que concatena un salto de línea y «[Sistema:» o «Instrucción:» al prompt), no
    # cualquier mención: el render del historial sigue LIMPIANDO ese andamiaje de las sesiones
    # persistidas antes del cierre, y esas regex deben quedarse.
    assert not re.search(r"\n\[Sistema:", codigo), f"el cliente vuelve a meter instrucciones en el turno del usuario [{_MARKER}]"
    assert not re.search(r"\nInstrucci[oó]n:", codigo)
    assert re.search(r"vision:\s*visionPayload", codigo), "el cliente no manda el contexto estructurado"
    assert "kind: 'multi'" in codigo and ": 'unavailable'" in codigo
    assert "has_text: !!userMsg" in codigo
