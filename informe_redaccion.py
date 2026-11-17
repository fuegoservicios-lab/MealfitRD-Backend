# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1033 · 2026-11-17] Informe para el médico: la redacción con `gpt-6.1-terra` (plan D-B/D-C).

Una llamada, sin herramientas, salida JSON. El modelo recibe SOLO `data` (sin nombre, correo ni id). El servidor
VERIFICA la respuesta (`verificar`): cada evidencia es una ruta de `data`, cada número del texto está en `data` y no hay
verbos de diagnóstico ni de tratamiento. Con problemas, un reintento con la lista; si vuelve a fallar ⇒ `data_only`.

Nunca cae al modelo de respaldo (Hydrogen cuesta ~US$ 0,10 por llamada): sin terra ⇒ `data_only`. Tope duro de coste
por informe (estimado antes de llamar). Fail-closed: ante cualquier error, `data_only` (el informe con datos sigue
siendo útil). Nada de `data` ni de la redacción va a los logs.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Optional

from knobs import _env_float, _env_str

logger = logging.getLogger(__name__)

MAX_TOKENS_SALIDA = 1000
MAX_HALLAZGOS = 6
_PALABRAS_RESUMEN = 140   # 120 pedidas + holgura


def tope_usd() -> float:
    return _env_float("MEALFIT_HEALTH_REPORT_MAX_COST_USD", 0.03, validator=lambda v: 0.001 <= v <= 0.25)


def modelo() -> Optional[str]:
    """`gpt-6.1-terra` (o el que diga `MEALFIT_HEALTH_REPORT_MODEL`), o None si está apagado / sin clave. Sin respaldo."""
    raw = (_env_str("MEALFIT_HEALTH_REPORT_MODEL", "") or "").strip()
    if raw.lower() in ("off", "none", "0", "false"):
        return None
    try:
        from chat_hard_turn import hard_turn_model_name
        base = hard_turn_model_name()
    except Exception:  # noqa: BLE001
        base = None
    if raw:
        try:
            import os
            from llm_provider import is_openai_model
            if is_openai_model(raw) and not (os.environ.get("OPENAI_API_KEY") or "").strip():
                return None
        except Exception:  # noqa: BLE001
            return None
        return raw
    return base


# ─────────────────────────────────────────────────────────────── verificación (pura)
_VETADOS = [
    # es
    r"\bdiagn[oó]stic", r"\brecet", r"\bsuspend", r"\baument\w* la dosis", r"\bdebe tomar", r"\bdebe(?:r[ií]a)? dejar de tomar",
    r"\bprescrib", r"\bmedicaci[oó]n (?:recomendada|indicada)", r"\btratamiento (?:recomendado|indicado)", r"\bpadece\b",
    r"\bsufre de\b", r"\btiene (?:diabetes|hipertensi[oó]n|anemia|una enfermedad)",
    # en
    r"\bdiagnos", r"\bprescri", r"\bshould take\b", r"\bincrease the dose", r"\bstop taking\b", r"\bsuffers from\b",
    # fr
    r"\bdiagnostiqu", r"\bprescri", r"\bdoit prendre\b", r"\baugmenter la dose", r"\barr[eê]ter de prendre",
    # it
    r"\bdiagnostic", r"\bprescriv", r"\bdeve prendere\b", r"\baumentare la dose", r"\bsmettere di prendere",
    # pt
    r"\bdiagn[oó]stic", r"\bprescrev", r"\bdeve tomar\b", r"\baumentar a dose", r"\bparar de tomar",
]
_RE_VETADOS = re.compile("|".join(_VETADOS), re.IGNORECASE)
_RE_NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:[.,   ]\d{3})+|\d+(?:[.,]\d+)?)(?![\w])")


def _numeros_de_data(data: Any) -> set[float]:
    out: set[float] = set()

    def walk(v: Any, clave: str = "") -> None:
        for n in re.findall(r"\d+(?:\.\d+)?", clave):
            out.add(float(n))
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            out.add(round(float(v), 1))
            out.add(float(round(v)))
        elif isinstance(v, str):
            for n in re.findall(r"\d+(?:\.\d+)?", v):
                out.add(float(n))
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(x, str(k))
        elif isinstance(v, list):
            for x in v:
                walk(x, clave)

    walk(data)
    return out | {0.0, 1.0}


def _interpretaciones(token: str) -> set[float]:
    t = token.replace(" ", " ").replace(" ", " ")
    out: set[float] = set()
    if re.fullmatch(r"\d{1,3}(?:[., ]\d{3})+", t):
        out.add(float(re.sub(r"[., ]", "", t)))       # miles: 1.985 / 1,985 / 1 985
    if re.fullmatch(r"\d+[.,]\d+", t):
        out.add(round(float(t.replace(",", ".")), 1))  # decimal: 2,2 / 2.2
    if re.fullmatch(r"\d+", t):
        out.add(float(t))
    return out


def verificar(narrativa: Any, data: dict) -> list[str]:
    """Problemas de la redacción frente a `data` (lista vacía = válida)."""
    from informe_medico import rutas_de

    problemas: list[str] = []
    if not isinstance(narrativa, dict):
        return ["la respuesta no es un objeto JSON"]
    resumen = narrativa.get("resumen")
    hallazgos = narrativa.get("hallazgos")
    preguntas = narrativa.get("preguntas_para_la_consulta")
    if not isinstance(resumen, str) or not resumen.strip():
        problemas.append("falta `resumen`")
    elif len(resumen.split()) > _PALABRAS_RESUMEN:
        problemas.append("el resumen pasa de 120 palabras")
    if not isinstance(hallazgos, list):
        problemas.append("`hallazgos` no es una lista")
        hallazgos = []
    elif len(hallazgos) > MAX_HALLAZGOS:
        problemas.append(f"más de {MAX_HALLAZGOS} hallazgos")
    if not isinstance(preguntas, list) or not (3 <= len(preguntas) <= 5) or not all(isinstance(p, str) for p in preguntas):
        problemas.append("`preguntas_para_la_consulta` debe tener de 3 a 5 textos")
        preguntas = preguntas if isinstance(preguntas, list) else []

    rutas = set(rutas_de(data))
    textos = [resumen if isinstance(resumen, str) else ""]
    for i, h in enumerate(hallazgos):
        if not isinstance(h, dict) or not isinstance(h.get("texto"), str):
            problemas.append(f"hallazgo {i + 1} sin texto")
            continue
        textos.append(h["texto"])
        ev = h.get("evidencia")
        if not isinstance(ev, list) or not ev:
            problemas.append(f"hallazgo {i + 1} sin evidencia")
            continue
        for e in ev:
            if str(e) not in rutas:
                problemas.append(f"evidencia inexistente: {str(e)[:80]}")
    textos += [p for p in preguntas if isinstance(p, str)]

    numeros = _numeros_de_data(data)
    for t in textos:
        m = _RE_VETADOS.search(t)
        if m:
            problemas.append(f"lenguaje de diagnóstico o tratamiento: «{m.group(0)}»")
        for tok in _RE_NUM.findall(t):
            if not (_interpretaciones(tok) & numeros):
                problemas.append(f"cifra que no está en los datos: {tok}")
    return problemas


def _json_de(texto: Any) -> Optional[dict]:
    if isinstance(texto, list):
        texto = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in texto)
    t = str(texto or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        v = json.loads(t[a:b + 1])
        return v if isinstance(v, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _limpia(n: dict) -> dict:
    return {
        "resumen": str(n.get("resumen") or "").strip(),
        "hallazgos": [{"texto": str(h.get("texto")).strip(), "evidencia": [str(e) for e in h.get("evidencia") or []]}
                      for h in (n.get("hallazgos") or []) if isinstance(h, dict)],
        "preguntas_para_la_consulta": [str(p).strip() for p in (n.get("preguntas_para_la_consulta") or [])],
    }


# ─────────────────────────────────────────────────────────────── la llamada
def redactar(data: dict, locale: str = "es", *, con_permiso: bool = True,
             llm_factory: Optional[Callable[[str], Any]] = None) -> dict:
    """Devuelve `{status, narrative (dict|None), model, cost_usd_micros, motivo}`.

    status: 'ready' (verificada) | 'data_only'. `motivo` (si data_only): 'sin_permiso' | 'no_disponible' | 'coste' |
    'no_verificada' | 'error'. `llm_factory(modelo)` permite inyectar un modelo simulado en los tests.
    """
    vacio = {"status": "data_only", "narrative": None, "model": None, "cost_usd_micros": 0}
    if not con_permiso:
        return {**vacio, "motivo": "sin_permiso"}
    m = modelo()
    if not m:
        return {**vacio, "motivo": "no_disponible"}
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from prompts.informe_medico import instruccion
        from db_profiles import compute_llm_cost_micros
    except Exception:  # noqa: BLE001
        return {**vacio, "motivo": "error"}

    entrada = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    try:
        from chat_hard_turn import estimate_call_max_micros
        est = estimate_call_max_micros(m, [SystemMessage(content=instruccion(locale)), HumanMessage(content=entrada)],
                                       [], MAX_TOKENS_SALIDA)
    except Exception:  # noqa: BLE001
        est = None
    # dos llamadas como mucho (la segunda, con los problemas): el tope cubre las dos
    if est is not None and 2 * est > tope_usd() * 1_000_000:
        return {**vacio, "motivo": "coste"}

    if llm_factory is None:
        def llm_factory(nombre: str):  # noqa: E306
            import llm_provider
            kw = {"use_responses_api": True} if "terra" in nombre.lower() else {}
            return llm_provider.build_chat_llm(nombre, timeout=60, reasoning_effort="low",
                                               max_tokens=MAX_TOKENS_SALIDA, **kw)
    try:
        llm = llm_factory(m)
    except Exception:  # noqa: BLE001
        return {**vacio, "motivo": "no_disponible"}

    coste = 0
    problemas: list[str] = []
    for intento in range(2):
        try:
            t0 = time.monotonic()
            res = llm.invoke([SystemMessage(content=instruccion(locale, problemas if intento else None)),
                              HumanMessage(content=entrada)])
            dur = time.monotonic() - t0
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[LOTE-1033] redacción del informe falló ({type(exc).__name__})")
            return {**vacio, "model": m, "cost_usd_micros": coste, "motivo": "error"}
        try:
            u = getattr(res, "usage_metadata", None) or {}
            cached = int(((u.get("input_token_details") or {}).get("cache_read")) or 0)
            coste += int(compute_llm_cost_micros(m, int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0),
                                                 cached_tokens=cached) or 0)
        except Exception:  # noqa: BLE001
            pass
        try:
            from chat_prompt_opt import registrar_uso_llm_auxiliar
            registrar_uso_llm_auxiliar(llm=llm, result=res, duration_s=dur, node="health_report")
        except Exception:  # noqa: BLE001
            pass
        narr = _json_de(getattr(res, "content", res))
        problemas = verificar(narr, data) if narr is not None else ["la respuesta no es un objeto JSON"]
        if not problemas:
            return {"status": "ready", "narrative": _limpia(narr), "model": m, "cost_usd_micros": coste, "motivo": None}
        logger.info(f"[LOTE-1033] redacción rechazada (intento {intento + 1}): {len(problemas)} problema(s)")
    return {**vacio, "model": m, "cost_usd_micros": coste, "motivo": "no_verificada"}
