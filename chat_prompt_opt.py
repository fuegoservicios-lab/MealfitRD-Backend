# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1000 · 2026-11-13] Optimizaciones de eficiencia, latencia y dieta de contexto del coach.

Operan 100 % sin pérdida de reglas ni seguridad clínica:
1. Reordenación cache-first del system prompt (`MEALFIT_CHAT_PROMPT_CACHE_ORDER`):
   Separa el bloque dinámico en 3 bandas después del prefijo estático:
   Banda 1 (perfil, clínica, metas, plan, suplementos, memoria histórica: estable por usuario/sesión)
   -> Banda 2 (inventario, diario hoy, hidratación, nevera, plan hoy, días pasados: del día)
   -> Banda 3 (temporal con HH:MM, circadiano, proactivo, sentimiento, RAG: volátil por turno)
   -> Cola (visión, recordatorio de voz) y, fuera, el refuerzo final de idioma.
   Nota clínica: `clinico` (`build_clinical_guard_context`) permanece SIEMPRE en `role=system` e incondicional.
2. Fast-path de sentimiento (`MEALFIT_CHAT_SKIP_SENTIMENT_PRESENT_LOGS`):
   Cuando `_rag_es_presente_sin_rag` ya determinó registro factual presente o duda de saldo de hoy y no hay
   indicios emocionales, salta también `classify_sentiment` (~1-1.5 s TTFT ahorrado).
3. Dieta de contexto por turno (`MEALFIT_CHAT_DIET_SUPLEMENTOS`, `MEALFIT_CHAT_DIET_PASADOS_INGREDIENTES`):
   - Suplementos: sin potes en Alacena, sin plan con suplementos, sin foto y sin mención en el turno, reemplaza
     el manual largo (~1 k chars) por un resumen de 1 línea (~155 chars) manteniendo el encabezado canónico.
   - Ingredientes históricos: en `DIARIO REAL DE DÍAS ANTERIORES` omite `(ingredientes: ...)` en turnos que
     no preguntan por ingredientes, recetas, clonación/repetición ni días anteriores. `DIARIO DE HOY` intacto.
4. Telemetría best-effort para nodos auxiliares (`chat_sentiment`, `chat_rag_router`, `chat_title`).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from constants import strip_accents
from llm_provider import _env_bool

logger = logging.getLogger("graph_orchestrator")


def cache_order_activo() -> bool:
    """Knob `MEALFIT_CHAT_PROMPT_CACHE_ORDER` (default True)."""
    return _env_bool("MEALFIT_CHAT_PROMPT_CACHE_ORDER", True)


def skip_sentiment_present_logs_activo() -> bool:
    """Knob `MEALFIT_CHAT_SKIP_SENTIMENT_PRESENT_LOGS` (default True)."""
    return _env_bool("MEALFIT_CHAT_SKIP_SENTIMENT_PRESENT_LOGS", True)


def diet_suplementos_activo() -> bool:
    """Knob `MEALFIT_CHAT_DIET_SUPLEMENTOS` (default True)."""
    return _env_bool("MEALFIT_CHAT_DIET_SUPLEMENTOS", True)


def diet_pasados_ingredientes_activo() -> bool:
    """Knob `MEALFIT_CHAT_DIET_PASADOS_INGREDIENTES` (default True)."""
    return _env_bool("MEALFIT_CHAT_DIET_PASADOS_INGREDIENTES", True)


# Prioridad cache-first (menor número = más cerca del frente del system prompt).
_CACHE_BLOCK_ORDER: dict[str, int] = {
    "split_marker": 0,
    # Banda 1: estable por usuario / sesión
    "super_personalizacion": 10,
    "identidad": 11,
    "clinico": 12,
    "meta_diaria": 13,
    "idioma_inicial": 14,
    "plan": 15,
    "prep_time": 16,
    "suplementos": 17,
    "suplementos_alacena": 18,
    "memoria_historica": 19,
    # Banda 2: estado del día, despensa e historial multi-día
    "inventario": 20,
    "diario_hoy": 21,
    "faltante_hoy": 22,
    "conv_dias_ant": 23,
    "sesion_previa": 24,
    "hidratacion": 25,
    "nevera": 26,
    "nevera_proteina": 27,
    "ref_registro": 28,
    "plan_hoy": 29,
    "dias_pasados_plan": 30,
    "dias_pasados_diario": 31,
    # Banda 3: cambia por minuto / turno / RAG / sentimiento
    "temporal": 40,
    "circadiano": 41,
    "proactivo": 42,
    "sentimiento": 43,
    "rag": 44,
    # Cola de turno (visión / modo voz) antes del refuerzo final de idioma
    "vision": 50,
    "modo_voz_rec": 51,
}


def reordenar_bloques_cache(
    system_prompt: str,
    *,
    prefijo_estatico_chars: int = 0,
) -> str:
    """Reordena únicamente los bloques dinámicos detrás de `prefijo_estatico_chars`.

    Nunca altera bytes del prefijo estático ni elimina contenido. Si la frontera no
    es positiva o el knob está apagado, devuelve `system_prompt` tal cual.
    """
    if not system_prompt or not cache_order_activo() or prefijo_estatico_chars <= 0:
        return system_prompt
    cut = min(prefijo_estatico_chars, len(system_prompt))
    if cut >= len(system_prompt):
        return system_prompt

    static_prefix = system_prompt[:cut]
    dynamic_tail = system_prompt[cut:]

    from prompt_desglose import _VOLATILE_BLOCK_SPECS

    # Excluimos `refuerzo_idioma` al escanear el tramo anterior al refuerzo final
    # para que `_LANG_HEADER_RE` no produzca dos hits duplicados por `idioma_inicial`.
    hits: list[tuple[int, str]] = []
    for name, pat in _VOLATILE_BLOCK_SPECS:
        if name == "refuerzo_idioma":
            continue
        m = pat.search(dynamic_tail)
        if m is not None:
            hits.append((m.start(), name))

    if len(hits) < 2:
        return system_prompt

    hits.sort(key=lambda item: item[0])
    # Deduplicar posiciones idénticas preservando el primer nombre encontrado.
    dedup: list[tuple[int, str]] = []
    seen_pos: set[int] = set()
    for pos, name in hits:
        if pos not in seen_pos:
            seen_pos.add(pos)
            dedup.append((pos, name))

    leading_orphan = dynamic_tail[: dedup[0][0]]
    slices: list[tuple[int, int, str]] = []
    for idx, (pos, name) in enumerate(dedup):
        end = dedup[idx + 1][0] if idx + 1 < len(dedup) else len(dynamic_tail)
        order = _CACHE_BLOCK_ORDER.get(name, 35)
        slices.append((order, idx, dynamic_tail[pos:end]))

    slices.sort(key=lambda item: (item[0], item[1]))
    rebuilt = static_prefix + leading_orphan + "".join(part for _, _, part in slices)
    if len(rebuilt) != len(system_prompt):
        return system_prompt
    return rebuilt


# ---------------------------------------------------------------------------
# Fast-path de sentimiento en registros objetivos presentes
# ---------------------------------------------------------------------------

_EMOTION_CUES_RE = re.compile(
    r"\b(?:cansad\w*|agotad\w*|frustrad\w*|triste\w*|deprim\w*|ansios\w*|ansiedad|"
    r"estres\w*|culp\w*|harto|harta|aburr\w*|no puedo|me rindo|rindo|odio|horrible|"
    r"pesimo|malisimo|terrible|fall[eo]|fracas\w*|verguenz\w*|miedo|preocup\w*|"
    r"dificil|imposible|desanim\w*|feliz|content[oa]|emocionad\w*|genial|increible|"
    r"logre|baje de peso|subi de peso|orgull\w*|por favor|ayuda|socorro|"
    r"no se que hacer|me siento|estoy mal|estoy bien)\b"
)
_EMOTION_EMOJI_RE = re.compile(r"[😢😔😞😤😡😭💔😰😓😩😫🤬🥺🎉🥳💪🔥❤🙌👏🙁☹]")


def es_turno_objetivo_sin_sentimiento(prompt: str) -> bool:
    """True cuando el turno es breve/factual y no muestra señales afectivas ni urgencia emocional."""
    raw = (prompt or "").strip()
    if not raw or len(raw) > 220:
        return False
    if "!!" in raw or _EMOTION_EMOJI_RE.search(raw):
        return False
    norm = re.sub(r"\s+", " ", strip_accents(raw).lower()).strip()
    if not norm:
        return False
    return _EMOTION_CUES_RE.search(norm) is None


# ---------------------------------------------------------------------------
# Dieta de contexto por turno (suplementos e ingredientes históricos)
# ---------------------------------------------------------------------------

_SUPLEMENTO_MENCION_RE = re.compile(
    r"\b(?:suplement\w*|creatina|whey|proteina en polvo|proteina\b|caseina|"
    r"colageno|magnesio|omega\s*3?|bcaa|glutamina|pre-?entreno|preworkout|"
    r"ganador(?:es)? de peso|mass gainer|multivitam\w*|vitamin\w*|pote\w*|"
    r"scoop\w*|pastilla\w*|capsula\w*|gomita\w*|polvo|batido\w*|alacena|etiqueta\w*|"
    r"zinc|hierro|calcio|electrolito\w*|melatonina)\b"
)

_SUPLEMENTOS_DIET_RESUMEN = (
    "\n\n💊 SUPLEMENTOS (sabes esto): sin potes en Alacena ni suplementos en plan. "
    "Usa `guardar_suplemento` o `log_consumed_meal(suplemento=...)` solo si menciona o envía foto de un pote."
)

_PASADOS_INGREDIENTES_NEEDED_RE = re.compile(
    r"\b(?:ingrediente\w*|receta\w*|prepar\w*|cocin\w*|con que|que llevaba|que tenia|"
    r"ayer|antier|anteayer|anoche|pasad\w*|semana|lunes|martes|miercoles|jueves|"
    r"viernes|sabado|domingo|repetir|clonar|lo mismo|mismo que|otra vez|"
    r"corrig\w*|cambial\w*|cambi\w*|quit\w*|agreg\w*|echale|extra\w*)\b"
)
_DIARIO_PASADO_MARKER = "\n\n🍽️ DIARIO REAL DE DÍAS ANTERIORES"
_PAST_LINE_INGREDIENTS_RE = re.compile(r" \(ingredientes: [^)\n]+\)")


def prompt_menciona_suplemento(prompt: str) -> bool:
    norm = re.sub(r"\s+", " ", strip_accents((prompt or "")).lower()).strip()
    return bool(norm and _SUPLEMENTO_MENCION_RE.search(norm))


def prompt_necesita_ingredientes_pasados(prompt: str) -> bool:
    norm = re.sub(r"\s+", " ", strip_accents((prompt or "")).lower()).strip()
    return bool(norm and _PASADOS_INGREDIENTES_NEEDED_RE.search(norm))


def adelgazar_suplementos_en_prompt(
    system_prompt: str,
    prompt: str,
    *,
    form_data: Any = None,
    con_foto: bool = False,
) -> str:
    """Reemplaza el bloque fijo de suplementos cuando el turno no los usa ni hay potes/foto."""
    if not system_prompt or not diet_suplementos_activo() or con_foto:
        return system_prompt
    if "SU ALACENA:" in system_prompt or prompt_menciona_suplemento(prompt):
        return system_prompt
    try:
        from suplementos import BLOQUE_CONOCIMIENTO, suplementos_activos

        if BLOQUE_CONOCIMIENTO not in system_prompt:
            return system_prompt
        if suplementos_activos(form_data):
            return system_prompt
        return system_prompt.replace(BLOQUE_CONOCIMIENTO, _SUPLEMENTOS_DIET_RESUMEN, 1)
    except Exception:
        return system_prompt


def adelgazar_ingredientes_dias_pasados(system_prompt: str, prompt: str) -> str:
    """Omite `(ingredientes: ...)` solo dentro de `DIARIO REAL DE DÍAS ANTERIORES` cuando no aplican."""
    if not system_prompt or not diet_pasados_ingredientes_activo():
        return system_prompt
    if prompt_necesita_ingredientes_pasados(prompt):
        return system_prompt
    idx = system_prompt.find(_DIARIO_PASADO_MARKER)
    if idx < 0:
        return system_prompt

    # Acotar al bloque del diario anterior hasta el siguiente bloque conocido del sistema
    start_after = idx + len(_DIARIO_PASADO_MARKER)
    end_idx = len(system_prompt)
    try:
        from prompt_desglose import _VOLATILE_BLOCK_SPECS

        for name, pat in _VOLATILE_BLOCK_SPECS:
            if name == "dias_pasados_diario":
                continue
            m = pat.search(system_prompt, pos=start_after)
            if m is not None and m.start() < end_idx:
                end_idx = m.start()
    except Exception:
        for sep in ("\n\n📷 ", "\n\nRECORDATORIO FINAL", "\n\n🌐 ", "\n\n📖 "):
            pos = system_prompt.find(sep, start_after)
            if pos != -1 and pos < end_idx:
                end_idx = pos

    head = system_prompt[:idx]
    target = system_prompt[idx:end_idx]
    tail = system_prompt[end_idx:]
    slim = _PAST_LINE_INGREDIENTS_RE.sub("", target)
    return head + slim + tail


def optimizar_prompt_dinamico(
    system_prompt: str,
    prompt: str,
    *,
    prefijo_estatico_chars: int = 0,
    form_data: Any = None,
    con_foto: bool = False,
) -> str:
    """Aplica dieta de contexto y luego ordena bloques dinámicos en sentido cache-first."""
    out = adelgazar_suplementos_en_prompt(
        system_prompt, prompt, form_data=form_data, con_foto=con_foto
    )
    out = adelgazar_ingredientes_dias_pasados(out, prompt)
    out = reordenar_bloques_cache(out, prefijo_estatico_chars=prefijo_estatico_chars)
    return out


# ---------------------------------------------------------------------------
# Telemetría best-effort para nodos auxiliares del flujo de chat
# ---------------------------------------------------------------------------

def registrar_uso_llm_auxiliar(
    *,
    llm: Any,
    result: Any,
    duration_s: float,
    node: str,
) -> None:
    """Emite evento en `llm_usage_events` para nodos pre/post chat sin poder romper el turno."""
    try:
        from llm_telemetry import _emit_llm_usage_event_best_effort

        _emit_llm_usage_event_best_effort(
            llm=llm,
            result=result,
            duration_s=duration_s,
            node=node,
        )
    except Exception as exc:
        logger.debug("[P1-PLAN-LOTE-1000] telemetria auxiliar (%s): %s", node, exc)
