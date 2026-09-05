"""[P1-I18N-GLM-USER-TURN · 2026-09-04] GLM (Z.ai) rechaza un `messages` con SOLO un system message
(HTTP 400, code 1214 «The messages parameter is illegal»). La capa `_display` del plan invocaba así
desde siempre (con Gemini/Luna funcionaba) y tras la migración a GLM (2026-09-02) llevaba dos días
muerta en silencio para todo usuario no hispanohablante: el primer francés real (2026-09-04) recibió
su plan en español. Este test ancla que TODA invocación de producción lleve un turno de usuario.
"""
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

_BACKEND = Path(__file__).resolve().parents[1]
# Un array de UN solo SystemMessage: tras su paréntesis de cierre viene directamente el `]`
# (con `, HumanMessage(...)` detrás NO casa: esa es la forma correcta).
_SYSTEM_ONLY = re.compile(r"invoke\(\s*\[\s*SystemMessage\(content=[^)]*\)\s*\]\s*\)")


def _prod_py_files():
    for p in _BACKEND.rglob("*.py"):
        rel = p.relative_to(_BACKEND).as_posix()
        if rel.startswith(("tests/", "venv/", ".venv/", "node_modules/")):
            continue
        yield p


def test_a_ninguna_invocacion_de_produccion_va_solo_con_system():
    culpables = []
    for p in _prod_py_files():
        src = p.read_text(encoding="utf-8", errors="replace")
        for m in _SYSTEM_ONLY.finditer(src):
            line = src.count("\n", 0, m.start()) + 1
            culpables.append(f"{p.relative_to(_BACKEND).as_posix()}:{line}")
    assert not culpables, f"GLM devuelve 400/1214 con solo system message: {culpables}"


def test_b_la_capa_display_arma_system_mas_turno_de_usuario():
    import plan_display_i18n as mod

    msgs = mod._build_messages("PROMPT DE PRUEBA")
    assert len(msgs) == 2
    assert isinstance(msgs[0], SystemMessage) and msgs[0].content == "PROMPT DE PRUEBA"
    assert isinstance(msgs[1], HumanMessage) and msgs[1].content.strip()
    # el turno de usuario refuerza el formato, no cambia el idioma pedido por la directiva
    assert "JSON" in msgs[1].content


def test_c_el_call_site_real_usa_el_helper():
    src = (_BACKEND / "plan_display_i18n.py").read_text(encoding="utf-8")
    assert "response = llm.invoke(_build_messages(prompt))" in src
    assert "from langchain_core.messages import HumanMessage, SystemMessage" in src
