"""[P3-CHAT-SAFETY-OFF-DECISION · 2026-05-20] Anchor de la decisión de
producto "chat-agent safety_settings relajados" tomada tras el audit
"agente" 2026-05-20.

Contexto:
    El audit "agente" 2026-05-20 flageó `_safety_settings` en `agent.py`
    como configuración inusual (`HARM_CATEGORY_DANGEROUS_CONTENT: OFF`
    + resto en `BLOCK_ONLY_HIGH`, NO en defaults). Un auditor técnico
    sin contexto puede confundirlo con "filtros desactivados por descuido".
    La decisión de producto (2026-05-20) es **intencional**:

      - MealfitRD es app nutricional clínica: usuarios discuten déficit
        calórico, ayuno intermitente, restricciones médicas, "comí poco
        hoy", "estoy intentando saltarme la cena", etc. Defaults de
        Google bloquean estos mensajes como "dangerous content" →
        false-positives que rompen el flujo conversacional.
      - El BLOCK_ONLY_HIGH para HARASSMENT/HATE_SPEECH/SEXUALLY_EXPLICIT
        sigue cubriendo abuso real (umbral high = solo bloquea contenido
        manifestly tóxico). Solo DANGEROUS_CONTENT en OFF.

    Defensas en profundidad mitigan el riesgo:
      - P2-CHAT-SANITIZE (server-side): neutraliza tags HTML peligrosas
        + event handlers + URIs javascript: en wire SSE.
      - P0-AGENT-1 (chat-tools): force-override user_id en execute_tools
        antes de invocar cualquier tool — cubre prompt injection.
      - P1-CHAT-EMPTY-RESPONSE: detecta cuando filter server-side de
        Google bloquea internamente (PROHIBITED_CONTENT categoría no
        gobernada por safety_settings del SDK) y sustituye por copy
        fallback amigable.

Lo que este test enforza:
  A) La sección "Decisiones de producto" existe en CLAUDE.md.
  B) La sub-sección sobre safety_settings existe con el anchor
     `P3-CHAT-SAFETY-OFF-DECISION` Y los tokens canónicos del setting.
  C) El `_safety_settings` en `agent.py` ES exactamente el documentado
     (DANGEROUS_CONTENT.*OFF + 3 categorías en BLOCK_ONLY_HIGH). Si
     alguien lo cambia sin actualizar la decisión, el test falla con
     copy explicativo.

Cómo revertir la decisión (si producto decide endurecer filtros):
  1. Cambiar `_safety_settings` en `agent.py` (a defaults o más estricto).
  2. Eliminar la sub-sección de CLAUDE.md (test B fallará, esperado).
  3. Eliminar/actualizar este test.

Tooltip-anchor: P3-CHAT-SAFETY-OFF-DECISION.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_AGENT_PY = _BACKEND_ROOT / "agent.py"


def _read_claude_md() -> str:
    assert _CLAUDE_MD.exists(), f"CLAUDE.md no encontrado en {_CLAUDE_MD}"
    return _CLAUDE_MD.read_text(encoding="utf-8")


def _read_agent_py() -> str:
    assert _AGENT_PY.exists(), f"agent.py no encontrado en {_AGENT_PY}"
    return _AGENT_PY.read_text(encoding="utf-8")


# A) Sección "Decisiones de producto" existe.
def test_a_decisiones_de_producto_section_exists():
    src = _read_claude_md()
    assert "## Decisiones de producto" in src, (
        "P3-CHAT-SAFETY-OFF-DECISION: CLAUDE.md perdió la sección "
        "'## Decisiones de producto'. Esta sección es el SSOT de "
        "decisiones que parecen gaps técnicos pero son producto. "
        "Si la moviste, actualizar este test."
    )


# B) Sub-sección sobre safety_settings con anchor.
def test_b_safety_off_decision_subsection_with_anchor():
    src = _read_claude_md()
    assert "P3-CHAT-SAFETY-OFF-DECISION" in src, (
        "P3-CHAT-SAFETY-OFF-DECISION: CLAUDE.md perdió el anchor "
        "`P3-CHAT-SAFETY-OFF-DECISION`. Sin anchor, un futuro audit no "
        "sabe que la decisión está documentada y puede volver a flagear "
        "el setting como 'filtros desactivados por descuido'."
    )
    assert "safety_settings" in src, (
        "P3-CHAT-SAFETY-OFF-DECISION: CLAUDE.md no menciona "
        "'safety_settings' en su body."
    )
    assert "DANGEROUS_CONTENT" in src, (
        "P3-CHAT-SAFETY-OFF-DECISION: CLAUDE.md no menciona "
        "'DANGEROUS_CONTENT' — la categoría específica que está OFF."
    )


# C) [P0-DEEPSEEK-MIGRATION · 2026-06-12] La decisión quedó SUPERSEDED:
# los `safety_settings` (HarmCategory/HarmBlockThreshold) eran exclusivos
# del SDK de Gemini. DeepSeek no expone content-filters configurables
# client-side, así que la INTENCIÓN de la decisión (no bloquear charla
# nutricional legítima de déficit/ayuno) queda cubierta por el default del
# provider. Este test ancla ahora el estado superseded: el bloque NO debe
# reaparecer en agent.py, y CLAUDE.md debe marcar la sección como superseded.
def test_c_agent_safety_settings_match_decision():
    src = _read_agent_py()
    code_only = re.sub(r"#[^\n]*", "", src)
    assert "_safety_settings = {" not in code_only, (
        "P0-DEEPSEEK-MIGRATION: el dict `_safety_settings` (Gemini-only) "
        "reapareció en agent.py. El provider DeepSeek no acepta ese kwarg — "
        "el wrapper lo swallowearía en silencio y daría falsa sensación de "
        "filtro configurado. Si se migra a un provider con safety settings, "
        "re-documentar la decisión en CLAUDE.md antes de reintroducirlo."
    )
    assert "from google.genai" not in code_only, (
        "P0-DEEPSEEK-MIGRATION: import del SDK de Google reapareció en agent.py."
    )

    claude_md = _read_claude_md()
    assert "SUPERSEDED" in claude_md.upper(), (
        "CLAUDE.md debe marcar la decisión P3-CHAT-SAFETY-OFF-DECISION como "
        "superseded por P0-DEEPSEEK-MIGRATION."
    )
