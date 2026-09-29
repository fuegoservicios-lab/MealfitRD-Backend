# backend/etiqueta_web.py
"""[P1-PLAN-LOTE-767 · 2026-09-29] La tabla nutricional de un suplemento, buscada en internet.

El dueño guardó su ganador de peso con la foto del FRENTE y el coach le pidió «una foto de atrás»: «¿el agente no puede
investigar la tabla nutricional de esa proteína? En la imagen se ve toda la info para buscar en internet». Probado
contra la API real el 29-sep (VPS), antes de escribir esto:

  · Gemini 3.8 Flash + búsqueda de Google, por el endpoint `/v1beta/interactions` (el de siempre,
    `models/…:generateContent` con `google_search`, contestaba SIN buscar: ni una consulta en la respuesta).
  · Sin tope, para un producto poco conocido, hizo 43 búsquedas (66 599 tokens de páginas leídas). Con «COMO MÁXIMO 3
    búsquedas» en la petición hace 3 (~1 500 tokens). Precio: 5 000 búsquedas gratis al mes, luego US$14 por mil.
  · El Atlas Gainer de Patriot Nutrition NO está publicado (3 y 43 búsquedas: no encontrado); el Gold Standard de
    Optimum sí (1 scoop de 31 g: 120 kcal, 24 g, 3 g, 1,5 g, 74 porciones).

Por eso el orden del coach (prompts/chat_agent.py `_POTE_DE_SUPLEMENTO`) es: la tabla de la foto; si no, las cifras
del FRENTE del envase; si no, esto; si no, la foto de la tabla. Lo encontrado se guarda por PRODUCTO (no por usuario:
no es un dato personal) para no pagar dos veces la misma búsqueda, y un «no encontrado» también, más corto.

Gemini solo BUSCA la etiqueta de un producto: no razona por el coach, no ve historial ni perfil. Es la tercera costura
permitida del blanket anti-Gemini (`test_p0_llm_provider_migration`), marcada `[P1-PLAN-LOTE-767-ETIQUETA-WEB]` en
cada línea que nombra el modelo.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from typing import Optional

from knobs import _env_bool, _env_float, _env_int, _env_str

import logging

logger = logging.getLogger(__name__)

_MODELO_DEFAULT = "gemini-3.8-flash"  # [P1-PLAN-LOTE-767-ETIQUETA-WEB]
_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
_NODO_USO = "etiqueta_web"
_PREFIJO_CACHE = "etiqueta_web:"
_DIAS_CACHE_ENCONTRADA = 90
_DIAS_CACHE_NO_ENCONTRADA = 14


def activa() -> bool:
    """Kill switch: `MEALFIT_ETIQUETA_WEB=0` y el coach vuelve a pedir la foto de la tabla, sin redeploy."""
    return _env_bool("MEALFIT_ETIQUETA_WEB", True)


def _modelo() -> str:
    return _env_str("MEALFIT_ETIQUETA_WEB_MODELO", _MODELO_DEFAULT)


def _max_busquedas() -> int:
    return _env_int("MEALFIT_ETIQUETA_WEB_MAX_BUSQUEDAS", 3, validator=lambda v: 1 <= v <= 6)


def _timeout_s() -> float:
    return _env_float("MEALFIT_ETIQUETA_WEB_TIMEOUT_S", 45.0, validator=lambda v: 5.0 <= v <= 120.0)


def max_por_usuario_al_dia() -> int:
    return _env_int("MEALFIT_ETIQUETA_WEB_MAX_POR_DIA", 5, validator=lambda v: 0 <= v <= 100)


def max_global_al_dia() -> int:
    """Tope de búsquedas nuevas al día para TODOS (con 3 búsquedas cada una, 150 caben de sobra en las 5 000 gratis del
    mes). Lo que ya está en la caché no cuenta."""
    return _env_int("MEALFIT_ETIQUETA_WEB_MAX_GLOBAL_DIA", 150, validator=lambda v: 0 <= v <= 5000)


def _clave() -> str:
    """La key de Gemini. En el VPS vive en `VISION_API_KEY` (el escáner ya va a Google con ella: misma cuenta, mismo
    saldo); si el escáner no apunta a Google, la de Gemini del entorno (`llm_provider._google_api_key`, fail-loud).
    NUNCA argumento del callsite."""
    base = os.environ.get("MEALFIT_VISION_BASE_URL", "")
    k = (os.environ.get("VISION_API_KEY") or "").strip()
    if k and "googleapis.com" in base:
        return k
    from llm_provider import _google_api_key
    return _google_api_key()


# ---------- clave de caché ----------

def _normal(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def clave_de_producto(marca, producto, sabor=None) -> str:
    """«Patriot Nutrition» + «Atlas Gainer Advanced Mass Vanilla» → una clave estable (sin acentos ni signos); el sabor,
    si no está ya en el nombre, se suma: cambia las cifras."""
    partes = [_normal(marca), _normal(producto)]
    s = _normal(sabor)
    if s and s not in partes[1]:
        partes.append(s)
    return _PREFIJO_CACHE + "|".join(p for p in partes if p)


# ---------- lo que Gemini contesta ----------

_RE_BLOQUE_JSON = re.compile(r"\{.*\}", re.S)


def texto_de_la_respuesta(datos: dict) -> str:
    """El texto final de una interacción: el último paso `model_output` (los anteriores son las búsquedas)."""
    textos = []
    for paso in (datos or {}).get("steps") or []:
        if isinstance(paso, dict) and paso.get("type") == "model_output":
            for c in paso.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "text":
                    textos.append(str(c.get("text") or ""))
    return textos[-1] if textos else ""


def leer_etiqueta(texto: str) -> Optional[dict]:
    """Del texto (un JSON, a veces entre ```json) a `{etiqueta, porciones, porcion_texto, fuente}`, o None si no la
    encontró o si lo que dice es imposible (misma validación que una etiqueta de foto: `suplementos.etiqueta_valida`)."""
    import suplementos
    m = _RE_BLOQUE_JSON.search(str(texto or ""))
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict) or not d.get("encontrado"):
        return None
    e = suplementos.etiqueta_valida({
        "gramos_porcion": d.get("gramos_porcion"), "kcal": d.get("kcal"), "protein_g": d.get("protein_g"),
        "carbs_g": d.get("carbs_g"), "fats_g": d.get("fats_g"),
    })
    if not e or not e.get("kcal"):
        return None
    try:
        porciones = float(d.get("porciones_por_envase") or 0)
    except (TypeError, ValueError):
        porciones = 0.0
    porciones = porciones if 1 <= porciones <= 1000 else None
    return {
        "etiqueta": e,
        "porciones": porciones,
        "porcion_texto": str(d.get("porcion_texto") or "")[:80] or None,
        "fuente": str(d.get("fuente") or "")[:200] or None,
    }


def _peticion(marca, producto, sabor) -> str:
    nombre = " ".join(x for x in (str(marca or "").strip(), str(producto or "").strip()) if x)
    if sabor and _normal(sabor) not in _normal(producto):
        nombre += f", sabor {str(sabor).strip()}"
    return (
        f"Haz COMO MÁXIMO {_max_busquedas()} búsquedas en Google y detente. Busca la tabla nutricional (nutrition "
        f"facts / información nutricional) del suplemento: {nombre}. Responde SOLO un JSON "
        "{\"encontrado\": bool, \"porcion_texto\": str, \"gramos_porcion\": num, \"kcal\": num, \"protein_g\": num, "
        "\"carbs_g\": num, \"fats_g\": num, \"porciones_por_envase\": num, \"fuente\": str}. Cifras POR PORCIÓN tal como "
        "las declara el fabricante para ESE producto y ese sabor (no las de otro producto de la marca). Si no la "
        "encuentras en esas búsquedas, encontrado=false y no inventes."
    )


# ---------- caché ----------

def _leer_cache(clave: str) -> Optional[dict]:
    from db import execute_sql_query
    fila = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s LIMIT 1", (clave,), fetch_one=True)
    valor = (fila or {}).get("value")
    if isinstance(valor, str):
        try:
            valor = json.loads(valor)
        except ValueError:
            return None
    if not isinstance(valor, dict):
        return None
    dias = _DIAS_CACHE_ENCONTRADA if valor.get("encontrado") else _DIAS_CACHE_NO_ENCONTRADA
    if time.time() - float(valor.get("buscado_en") or 0) > dias * 86400:
        return None
    return valor


def _escribir_cache(clave: str, valor: dict) -> None:
    from db import execute_sql_query
    execute_sql_query(
        "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s::jsonb, now()) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
        (clave, json.dumps(valor, ensure_ascii=False)),
    )


def _busquedas_de_hoy(user_id: Optional[str]) -> int:
    from db import execute_sql_query
    sql = ("SELECT COUNT(*) AS n FROM llm_usage_events WHERE node = %s "
           "AND created_at >= date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'")
    args = [_NODO_USO]
    if user_id:
        sql += " AND user_id = %s"
        args.append(user_id)
    fila = execute_sql_query(sql, tuple(args), fetch_one=True)
    return int((fila or {}).get("n") or 0)


# ---------- la búsqueda ----------

def buscar(user_id: str, marca, producto, sabor=None) -> dict:
    """`{"estado": "encontrada"|"no_encontrada"|"sin_cupo"|"apagada"|"error", ...}` con la etiqueta si la encontró.
    Primero la caché del producto; una búsqueda nueva cuenta contra los topes del día."""
    if not activa():
        return {"estado": "apagada"}
    if not str(producto or "").strip():
        return {"estado": "no_encontrada"}
    clave = clave_de_producto(marca, producto, sabor)
    try:
        guardado = _leer_cache(clave)
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-767] no se pudo leer la caché de etiquetas: {e}")
        guardado = None
    if guardado is not None:
        if guardado.get("encontrado"):
            return {"estado": "encontrada", "cache": True, **{k: guardado.get(k) for k in
                                                              ("etiqueta", "porciones", "porcion_texto", "fuente")}}
        return {"estado": "no_encontrada", "cache": True}
    try:
        if _busquedas_de_hoy(user_id) >= max_por_usuario_al_dia() or _busquedas_de_hoy(None) >= max_global_al_dia():
            return {"estado": "sin_cupo"}
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-767] no se pudo contar las búsquedas de hoy: {e}")
        return {"estado": "error"}
    import httpx
    t0 = time.monotonic()
    try:
        r = httpx.post(
            _URL,
            headers={"x-goog-api-key": _clave(), "Content-Type": "application/json"},
            json={"model": _modelo(), "input": _peticion(marca, producto, sabor), "tools": [{"type": "google_search"}]},
            timeout=_timeout_s(),
        )
        datos = r.json() if r.status_code == 200 else {}
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-767] la búsqueda de la etiqueta falló: {type(e).__name__}: {e}")
        return {"estado": "error"}
    if r.status_code != 200:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-767] la búsqueda de la etiqueta devolvió HTTP {r.status_code}")
        return {"estado": "error"}
    uso = datos.get("usage") or {}
    busquedas = sum(int(g.get("search_query_count") or 0) for g in (uso.get("grounding_tool_count") or [])
                    if isinstance(g, dict))
    hallazgo = leer_etiqueta(texto_de_la_respuesta(datos))
    try:
        from db import log_llm_usage_event
        log_llm_usage_event(
            user_id=user_id, model=_modelo(), node=_NODO_USO,
            input_tokens=uso.get("total_input_tokens"), output_tokens=uso.get("total_output_tokens"),
            cached_tokens=uso.get("total_cached_tokens"),
            metadata={"busquedas": busquedas, "encontrada": bool(hallazgo), "ms": int((time.monotonic() - t0) * 1000)},
        )
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-767] no se pudo anotar el uso de la búsqueda: {e}")
    valor = {"encontrado": bool(hallazgo), "buscado_en": time.time(), **(hallazgo or {})}
    try:
        _escribir_cache(clave, valor)
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-767] no se pudo guardar la etiqueta en la caché: {e}")
    if hallazgo:
        return {"estado": "encontrada", "cache": False, **hallazgo}
    return {"estado": "no_encontrada", "cache": False}
