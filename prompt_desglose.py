# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-999 · 2026-11-13] Fase 0: Desglose de caracteres por bloque del system prompt e historial del coach.

Observabilidad pura antes de tocar caché o recortes (Fases 2 y 3): mide cuántos caracteres aporta cada bloque del
`system_prompt` (manual estático, recorte lean, cada bloque volátil por nombre, RAG, memoria) y cuántos mensajes /
caracteres van en el historial enviado al modelo (`chat_call_model`), emitiéndolo en `llm_usage_events.metadata`.

Knob: `MEALFIT_CHAT_PROMPT_BREAKDOWN` (True). En False es no-op O(1) sin asignar diccionario.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from knobs import _env_bool

# Anchors ordenados de bloques que se concatenan al `system_prompt` en `chat_with_agent` y `chat_with_agent_stream`.
# Cada entrada es `(nombre_bloque, regex_inicio)`. El tramo entre un inicio y el inicio siguiente es el tamaño exacto
# del bloque, de forma que la suma de bloques + prefijo + unattributed == len(system_prompt).
_LANG_HEADER_RE = (
    r"\n\n🌐 (?:"
    r"RESPONSE LANGUAGE|IDIOMA DA RESPOSTA|LANGUE DE RÉPONSE|LINGUA DELLA RISPOSTA|"
    r"IDIOMA DE RESPUESTA|TITLE LANGUAGE|IDIOMA DO TÍTULO|LANGUE DU TITRE|LINGUA DEL TITOLO"
    r")"
)

_VOLATILE_BLOCK_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("split_marker", re.compile(r"\n<!--MEALFIT_VOLATILE_SYS_SPLIT-->\n(?:El user_id (?:del usuario )?actual es: [^\n]+)?")),
    ("temporal", re.compile(r"\n\n🕒 CONTEXTO TEMPORAL ACTUAL:")),
    ("circadiano", re.compile(r"\n⚠️ RITMO CIRCADIANO:")),
    ("proactivo", re.compile(r"\n🌟 REGLA DE CONTINUIDAD TEMPORAL PROACTIVA:")),
    ("sentimiento", re.compile(r"\n\nPERSONALIDAD ACTIVA:")),
    ("rag", re.compile(r"\n--- MEMORIA VECTORIAL \(RAG\) ---")),
    ("inventario", re.compile(r"\n\n🛒 ESTADO DE LA DESPENSA Y COMPRAS")),
    ("super_personalizacion", re.compile(r"\n--- SÚPER PERSONALIZACIÓN DEL USUARIO ---")),
    ("identidad", re.compile(r"\n\n--- 👤 PERFIL DEL USUARIO")),
    ("clinico", re.compile(r"\n\n🛑 PERFIL CLÍNICO DEL USUARIO")),
    ("meta_diaria", re.compile(r"\n\n🎯 META DIARIA:")),
    ("idioma_inicial", re.compile(_LANG_HEADER_RE)),
    ("plan", re.compile(r"\n\nCONTEXTO CRÍTICO: El usuario ")),
    ("prep_time", re.compile(r"\n\n⏱️ TIEMPO DE COCINA:")),
    ("suplementos", re.compile(r"\n?💊 SUPLEMENTOS (?:SELECCIONADOS|ACTIVOS):")),
    ("suplementos_alacena", re.compile(r"\n\n💊 SUPLEMENTOS \(sabes esto")),
    ("memoria_historica", re.compile(r"\n\n<contexto_evolutivo_historico>")),
    ("diario_hoy", re.compile(r"\n\nDIARIO DE HOY:")),
    ("faltante_hoy", re.compile(r"\n\n🎯 LO QUE LE FALTA HOY")),
    ("conv_dias_ant", re.compile(r"\n\n📅 ESTA CONVERSACIÓN EMPEZÓ ANTES DE HOY")),
    ("sesion_previa", re.compile(r"\n\n💬 SESIÓN ANTERIOR RECIENTE")),
    ("hidratacion", re.compile(r"\n\n💧 HIDRATACIÓN ")),
    ("nevera", re.compile(r"\n\n🧊 NEVERA")),
    ("nevera_proteina", re.compile(r"\n\n🥩 PROTEÍNA ")),
    ("ref_registro", re.compile(r"\n\n🔁 SE REFIERE A UN REGISTRO SUYO")),
    ("plan_hoy", re.compile(r"\n\n📅 HOY es ")),
    ("dias_pasados_plan", re.compile(r"\n\n📖 DÍAS QUE YA PASARON")),
    ("dias_pasados_diario", re.compile(r"\n\n🍽️ DIARIO REAL DE DÍAS ANTERIORES")),
    ("vision", re.compile(r"\n\n📷 CONTEXTO DE (?:VARIAS FOTOS|FOTO):")),
    ("modo_voz_rec", re.compile(r"\n\nRECORDATORIO FINAL — MODO VOZ")),
    ("refuerzo_idioma", re.compile(_LANG_HEADER_RE)),
)


def breakdown_activo() -> bool:
    """Knob `MEALFIT_CHAT_PROMPT_BREAKDOWN` (True): emite tamaño por bloque e historial en telemetría."""
    return _env_bool("MEALFIT_CHAT_PROMPT_BREAKDOWN", True)


def _find_all_block_hits(texto: str, start_at: int) -> list[tuple[int, str]]:
    """Localiza los límites de bloques ordenados dentro de `texto[start_at:]`.

    `idioma_inicial` y `refuerzo_idioma` comparten el mismo prefijo `🌐`: puede haber 1 o 2
    directivas seguidas en el punto inicial (perfil + idioma del mensaje) y 1 o 2 al final.
    Agrupamos cada racha contigua y asignamos la primera racha a `idioma_inicial` y la
    última (cuando está detrás de los bloques principales) a `refuerzo_idioma`.
    """
    n = len(texto)
    if start_at >= n:
        return []

    raw_hits: list[tuple[int, str]] = []
    seen_pos: set[int] = set()
    lang_positions: list[int] = []

    for name, pat in _VOLATILE_BLOCK_SPECS:
        if name in ("idioma_inicial", "refuerzo_idioma"):
            continue
        for m in pat.finditer(texto, start_at):
            pos = m.start()
            if pos not in seen_pos:
                seen_pos.add(pos)
                raw_hits.append((pos, name))
            break

    lang_pat = re.compile(_LANG_HEADER_RE)
    for m in lang_pat.finditer(texto, start_at):
        lang_positions.append(m.start())

    if lang_positions:
        # Separar ocurrencias antes de la cola final vs refuerzo final (después del último bloque no-idioma).
        last_non_lang = max((pos for pos, _ in raw_hits), default=start_at - 1)
        early_langs = [p for p in lang_positions if p < last_non_lang]
        tail_langs = [p for p in lang_positions if p > last_non_lang]
        if early_langs:
            p0 = min(early_langs)
            if p0 not in seen_pos:
                seen_pos.add(p0)
                raw_hits.append((p0, "idioma_inicial"))
        if tail_langs:
            p1 = min(tail_langs)
            if p1 not in seen_pos:
                seen_pos.add(p1)
                raw_hits.append((p1, "refuerzo_idioma"))

    raw_hits.sort(key=lambda item: item[0])
    return raw_hits


def desglosar_prompt(
    system_prompt: str,
    *,
    manual_chars: int = 0,
    pais_chars: int = 0,
    tools_instr_chars: int = 0,
    prefijo_estatico_chars: int = 0,
    lean_ahorro_chars: int = 0,
    rag_chars: int = 0,
    memoria_hist_chars: int = 0,
    stream: bool = False,
    modo_voz: bool = False,
) -> Optional[dict]:
    """Desglosa el `system_prompt` en tamaño por bloque (caracteres). Devuelve None si está apagado."""
    if not breakdown_activo():
        return None
    texto = system_prompt or ""
    total = len(texto)

    # Si no recibimos `prefijo_estatico_chars` (p.ej. llamada aislada en tests), inferimos el inicio
    # del primer bloque dinámico conocido.
    if prefijo_estatico_chars > 0:
        dyn_start = min(int(prefijo_estatico_chars), total)
    else:
        hits_all = _find_all_block_hits(texto, 0)
        dyn_start = hits_all[0][0] if hits_all else total

    bloques: dict[str, int] = {}
    if dyn_start > 0:
        bloques["manual_estatico"] = dyn_start

    hits = _find_all_block_hits(texto, dyn_start)
    if hits and hits[0][0] > dyn_start:
        bloques["sin_clasificar"] = hits[0][0] - dyn_start

    for idx, (pos, name) in enumerate(hits):
        end = hits[idx + 1][0] if (idx + 1) < len(hits) else total
        delta = max(0, end - pos)
        if delta <= 0:
            continue
        if name in ("dias_pasados_plan", "dias_pasados_diario"):
            bloques["dias_pasados"] = bloques.get("dias_pasados", 0) + delta
        elif name == "split_marker":
            bloques["split_marker"] = bloques.get("split_marker", 0) + delta
        else:
            bloques[name] = bloques.get(name, 0) + delta

    if not hits and dyn_start < total:
        bloques["sin_clasificar"] = bloques.get("sin_clasificar", 0) + (total - dyn_start)

    out: dict[str, Any] = {
        "total_sys_chars": total,
        "prefijo_estatico": dyn_start,
        "bloques": {k: v for k, v in bloques.items() if v > 0},
    }
    if manual_chars:
        out["manual_base"] = int(manual_chars)
    if pais_chars:
        out["pais_culinaria"] = int(pais_chars)
    if tools_instr_chars:
        out["tools_instr"] = int(tools_instr_chars)
    if lean_ahorro_chars:
        out["lean_ahorro"] = int(lean_ahorro_chars)
    if rag_chars and "rag" not in out["bloques"]:
        out["bloques"]["rag"] = int(rag_chars)
    if memoria_hist_chars and "memoria_historica" not in out["bloques"]:
        out["bloques"]["memoria_historica"] = int(memoria_hist_chars)
    out["stream"] = bool(stream)
    if modo_voz:
        out["modo_voz"] = True
    return out


def _msg_text_and_media_len(content: Any) -> tuple[int, int]:
    """Cuenta caracteres de texto y partes multimedia sin inflar base64 como si fuera texto."""
    if isinstance(content, str):
        return len(content), 0
    if isinstance(content, list):
        chars = 0
        media = 0
        for part in content:
            if isinstance(part, dict):
                ptype = str(part.get("type") or "")
                if ptype == "text":
                    chars += len(str(part.get("text") or ""))
                elif ptype in ("image_url", "input_image", "image", "audio", "input_audio"):
                    media += 1
                else:
                    chars += len(json.dumps(part, ensure_ascii=False, default=str))
            else:
                chars += len(str(part or ""))
        return chars, media
    return len(str(content or "")), 0


def resumen_historial(
    llm_messages: Optional[Iterable[Any]],
    active_tools: Optional[Iterable[Any]] = None,
) -> Optional[dict]:
    """Resume mensajes y caracteres de historial enviados al LLM en `call_model`."""
    if not breakdown_activo():
        return None
    sys_chars = 0
    hist_msgs = 0
    hist_chars = 0
    hist_media = 0
    tool_calls_n = 0
    for msg in llm_messages or ():
        mtype = getattr(msg, "type", None) or getattr(msg, "role", None)
        content = getattr(msg, "content", "")
        c_len, m_cnt = _msg_text_and_media_len(content)
        if mtype == "system":
            sys_chars += c_len
            continue
        hist_msgs += 1
        hist_chars += c_len
        hist_media += m_cnt
        tcalls = getattr(msg, "tool_calls", None) or []
        if tcalls:
            tool_calls_n += len(tcalls)
            hist_chars += len(json.dumps(tcalls, ensure_ascii=False, default=str))
    res: dict[str, Any] = {
        "sys_chars_enviados": sys_chars,
        "historial_msgs": hist_msgs,
        "historial_chars": hist_chars,
    }
    if hist_media:
        res["historial_media"] = hist_media
    if tool_calls_n:
        res["historial_tool_calls"] = tool_calls_n
    if active_tools is not None:
        res["tools_n"] = len(list(active_tools))
    return res


def metadata_para_evento(
    *,
    state: Optional[dict] = None,
    llm_messages: Optional[Iterable[Any]] = None,
    active_tools: Optional[Iterable[Any]] = None,
) -> Optional[dict]:
    """Construye el payload extra para `llm_usage_events.metadata` de `chat_call_model`."""
    extra: dict[str, Any] = {}
    st = state if isinstance(state, dict) else {}
    sid = st.get("session_id")
    if isinstance(sid, str) and sid.strip():
        extra["session_id"] = sid.strip()

    if breakdown_activo():
        base_bd = st.get("prompt_breakdown")
        merged: dict[str, Any] = dict(base_bd) if isinstance(base_bd, dict) else {}
        hist = resumen_historial(llm_messages, active_tools=active_tools)
        if hist:
            merged.update(hist)
        if merged:
            extra["prompt_breakdown"] = merged

    return extra or None
