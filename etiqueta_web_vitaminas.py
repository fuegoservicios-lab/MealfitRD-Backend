# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1103 · 2026-11-18] La tabla de VITAMINAS Y MINERALES (Supplement Facts) de un suplemento, buscada en
internet y completada en la Alacena EN SEGUNDO PLANO.

El dueño mandó la foto de su multivitamínico (frente: «Designs for Sport · MULTI+ PHYTO») con «Es mío, me metí dos
cápsulas», y el coach contestó «la tabla no se ve; mándame una foto clara de Supplement Facts» sin guardarlo. «Debió
buscar por internet y también guardarlo, ya que es mío.» La búsqueda del lote 767 (`etiqueta_web.py`) solo trae kcal y
macros: para un multivitamínico (0 kcal) siempre salía «no encontrada».

Probado en real el 18-nov (VPS, Gemini 3.8 Flash, endpoint `/v1beta/interactions`):
  · solo `google_search` (3 y 5 búsquedas): «no encontrado» — los resultados de búsqueda no traen la tabla, y con
    `url_context` en la MISMA petición no abre ninguna página;
  · búsqueda → URLs, y LUEGO `url_context` con esas URLs: la tabla completa (4 cápsulas por porción, 1500 mcg RAE de
    vitamina A, 400 mg de C…), de la ficha técnica en PDF y de Fullscript. Tarda 15 + 50-60 s.

Un minuto no cabe en un turno de chat, así que el coach guarda el pote YA (sin tabla) y esto corre en segundo plano:
busca, lee y, si la encuentra, la escribe en ese pote de la Alacena (solo si sigue sin tabla: una foto nunca se pisa).
La siguiente vez que pregunte, `analizar_suplemento` la lee de ahí.

Mismos interruptor, topes diarios (mismo nodo de uso: cuentan juntas), modelo y clave que `etiqueta_web`. Caché por
PRODUCTO (no es un dato personal) con su propio prefijo. Una cantidad sin la fila que la respalda (`label_text`) se
descarta (`supplement_labels.normalize`), igual que en una foto. Nada de datos de salud en los logs.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import logging

import etiqueta_web as _ew

logger = logging.getLogger(__name__)

_PREFIJO_CACHE = "etiqueta_web_vit:"
_TIMEOUT_BUSQUEDA_S = 90.0   # en segundo plano: el de etiqueta_web (45 s) se quedó corto en real
_TIMEOUT_LECTURA_S = 120.0
_MAX_URLS = 3
_RE_URL = re.compile(r"https?://[^\s\"'<>)\]\\]{8,300}")
_DOMINIOS_FUERA = ("google.", "vertexaisearch.", "gstatic.", "w3.org", "youtube.", "facebook.", "instagram.")


def clave_de_producto(marca, producto) -> str:
    return _PREFIJO_CACHE + _ew.clave_de_producto(marca, producto)[len(_ew._PREFIJO_CACHE):]


def _nombre(marca, producto) -> str:
    return " ".join(x for x in (str(marca or "").strip(), str(producto or "").strip()) if x)


def _peticion_busqueda(marca, producto) -> str:
    return (
        f"Haz COMO MÁXIMO {_ew._max_busquedas()} búsquedas en Google y detente. Busca páginas que muestren la tabla "
        f"«Supplement Facts» del suplemento: {_nombre(marca, producto)} (la ficha técnica del fabricante en PDF, la "
        "página del producto o una tienda como Fullscript o iHerb). Responde SOLO las URLs de los 4 resultados más "
        "relevantes para ESE producto, una por línea, las más prometedoras primero (aunque no estés seguro de que "
        "muestren la tabla completa: otra herramienta las va a abrir)."
    )


def _peticion_lectura(marca, producto, urls) -> str:
    return (
        f"Lee estas páginas y copia la tabla «Supplement Facts» (vitaminas y minerales por porción) del suplemento "
        f"{_nombre(marca, producto)}. Páginas: {' '.join(urls)}. Responde SOLO un JSON "
        "{\"encontrado\": bool, \"serving_text\": str, \"units_per_serving\": num, \"serving_unit\": str, "
        "\"porciones_por_envase\": num, \"nutrients\": [{\"name\": str, \"amount\": num, \"unit\": str, \"form\": str, "
        "\"label_text\": str}], \"fuente\": str}. serving_text tal como lo declara el fabricante (ej. «4 Capsules»); "
        "units_per_serving = cuántas cápsulas/tabletas son UNA porción; serving_unit = capsule, tablet, softgel, gummy o "
        "scoop. En nutrients, la cantidad ABSOLUTA declarada (no el % del valor diario) con su unidad (mg, mcg, g, IU, "
        "mcg DFE, mcg RAE), y en label_text la fila tal como aparece. Solo ESE producto (no otro de la marca). fuente = "
        "la URL de donde la copiaste. Si ninguna página la trae, encontrado=false y no inventes."
    )


_REDIRECCION = "vertexaisearch.cloud.google.com/grounding-api-redirect/"


def _resolver(url: str) -> Optional[str]:
    """La búsqueda devuelve enlaces de redirección de Google: su destino real (sin seguirlo), o None."""
    if _REDIRECCION not in url:
        return url
    try:
        import httpx
        r = httpx.get(url, follow_redirects=False, timeout=10.0)
        destino = r.headers.get("location") or ""
        return destino if destino.startswith("http") else None
    except Exception:
        return None


def urls_de_la_respuesta(texto: str, resolver=_resolver) -> list:
    """Las URLs que devolvió la búsqueda (las redirecciones de Google, resueltas), sin las de Google ni redes, sin
    repetir; las fichas en PDF primero."""
    vistas, urls = set(), []
    for u in _RE_URL.findall(str(texto or ""))[:8]:
        u = resolver(u.rstrip(".,;:")) or ""
        dominio = u.split("/")[2].lower() if u.count("/") >= 2 else ""
        if not dominio or any(d in dominio for d in _DOMINIOS_FUERA) or u in vistas:
            continue
        vistas.add(u)
        urls.append(u)
    urls.sort(key=lambda u: 0 if u.lower().split("?")[0].endswith(".pdf") else 1)
    return urls[:_MAX_URLS]


def leer_tabla(texto: str) -> Optional[dict]:
    """Del texto de Gemini a `{supplement_facts, porciones, fuente}`, o None si no la encontró o no trae ninguna
    cantidad respaldada por su fila."""
    from supplement_labels import normalize
    m = _ew._RE_BLOQUE_JSON.search(str(texto or ""))
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict) or not d.get("encontrado"):
        return None
    filas = d.get("nutrients") if isinstance(d.get("nutrients"), list) else []
    facts = normalize({
        "serving_text": d.get("serving_text"), "units_per_serving": d.get("units_per_serving"),
        "serving_unit": d.get("serving_unit"), "amount_basis": "per_serving", "table_readable": True,
        "nutrients": [dict(n, basis="declared") for n in filas if isinstance(n, dict)],
    })
    if not facts or not any(n.get("amount") is not None for n in facts["nutrients"]):
        return None
    try:
        porciones = float(d.get("porciones_por_envase") or 0)
    except (TypeError, ValueError):
        porciones = 0.0
    return {
        "supplement_facts": facts,
        "porciones": porciones if 1 <= porciones <= 1000 else None,
        "fuente": str(d.get("fuente") or "")[:200] or None,
    }


def _interaccion(entrada: str, herramienta: str, timeout: float) -> tuple:
    """(texto, uso) de una petición a Gemini con UNA herramienta (`google_search` o `url_context`)."""
    import httpx
    r = httpx.post(
        _ew._URL,
        headers={"x-goog-api-key": _ew._clave(), "Content-Type": "application/json"},
        json={"model": _ew._modelo(), "input": entrada, "tools": [{"type": herramienta}]},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    datos = r.json()
    return _ew.texto_de_la_respuesta(datos), (datos.get("usage") or {})


def _busquedas(uso: dict) -> int:
    return sum(int(g.get("count") or g.get("search_query_count") or 0)
               for g in (uso.get("grounding_tool_count") or []) if isinstance(g, dict))


def en_cache(marca, producto) -> Optional[dict]:
    """Lo que ya se buscó de este producto: `{"estado": "encontrada", ...}`, `{"estado": "no_encontrada"}` o None."""
    try:
        guardado = _ew._leer_cache(clave_de_producto(marca, producto))
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-1103] no se pudo leer la caché de tablas de vitaminas: {type(e).__name__}")
        return None
    if guardado is None:
        return None
    if guardado.get("encontrado"):
        return {"estado": "encontrada", "cache": True,
                **{k: guardado.get(k) for k in ("supplement_facts", "porciones", "fuente")}}
    return {"estado": "no_encontrada", "cache": True}


def hay_cupo(user_id: Optional[str]) -> bool:
    try:
        return (_ew._busquedas_de_hoy(user_id) < _ew.max_por_usuario_al_dia()
                and _ew._busquedas_de_hoy(None) < _ew.max_global_al_dia())
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-1103] no se pudo contar las búsquedas de hoy: {type(e).__name__}")
        return False


def buscar(user_id: Optional[str], marca, producto) -> dict:
    """La búsqueda en dos pasos (SIN mirar la caché ni los topes: eso lo hace quien la lanza). Escribe la caché."""
    t0 = time.monotonic()
    uso_total, n_busq, hallazgo, urls = {}, 0, None, []
    try:
        texto, uso = _interaccion(_peticion_busqueda(marca, producto), "google_search", _TIMEOUT_BUSQUEDA_S)
        n_busq = _busquedas(uso)
        uso_total = dict(uso)
        urls = urls_de_la_respuesta(texto)
        if urls:
            texto, uso2 = _interaccion(_peticion_lectura(marca, producto, urls), "url_context", _TIMEOUT_LECTURA_S)
            for k in ("total_input_tokens", "total_output_tokens", "total_cached_tokens"):
                uso_total[k] = int(uso_total.get(k) or 0) + int(uso2.get(k) or 0)
            hallazgo = leer_tabla(texto)
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-1103] la búsqueda de la tabla de vitaminas falló: {type(e).__name__}")
        return {"estado": "error"}
    try:
        from db import log_llm_usage_event
        log_llm_usage_event(
            user_id=user_id, model=_ew._modelo(), node=_ew._NODO_USO,
            input_tokens=uso_total.get("total_input_tokens"), output_tokens=uso_total.get("total_output_tokens"),
            cached_tokens=uso_total.get("total_cached_tokens"),
            metadata={"tipo": "vitaminas", "busquedas": n_busq, "paginas": len(urls), "encontrada": bool(hallazgo),
                      "ms": int((time.monotonic() - t0) * 1000)},
        )
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-1103] no se pudo anotar el uso de la búsqueda: {type(e).__name__}")
    try:
        _ew._escribir_cache(clave_de_producto(marca, producto),
                            {"encontrado": bool(hallazgo), "buscado_en": time.time(), **(hallazgo or {})})
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-1103] no se pudo guardar la tabla en la caché: {type(e).__name__}")
    return {"estado": "encontrada", "cache": False, **hallazgo} if hallazgo else {"estado": "no_encontrada"}


def etiqueta_de(hallazgo: dict) -> dict:
    """La etiqueta de un pote sin calorías con su tabla: lo que `suplementos.etiqueta_valida` acepta."""
    facts = hallazgo["supplement_facts"]
    e = {"gramos_porcion": 0, "kcal": 0, "protein_g": 0, "carbs_g": 0, "fats_g": 0, "supplement_facts": facts}
    if facts.get("units_per_serving"):
        e["unidades_por_toma"] = facts["units_per_serving"]
    return e


def completar_pote(user_id: str, nombre: str, hallazgo: dict) -> bool:
    """Escribe la tabla en el pote `nombre` de su Alacena si sigue SIN tabla de vitaminas (una foto nunca se pisa)."""
    import suplementos
    from db_core import execute_sql_write
    pote = suplementos.buscar(user_id, nombre)
    if not pote or not pote.get("id"):
        return False
    actual = pote.get("serving_label")
    if isinstance(actual, str):
        try:
            actual = json.loads(actual)
        except ValueError:
            actual = None
    _f = actual.get("supplement_facts") if isinstance(actual, dict) else None
    if isinstance(_f, dict) and _f.get("table_readable") and any(
            isinstance(n, dict) and n.get("amount") is not None for n in (_f.get("nutrients") or [])):
        return False      # [P1-PLAN-LOTE-1109] una tabla de foto LEGIBLE no se pisa; una ilegible sí se completa
    etiqueta = suplementos.etiqueta_valida(etiqueta_de(hallazgo))
    if not etiqueta:
        return False
    execute_sql_write(
        "UPDATE user_inventory SET serving_label = %s::jsonb, label_source = 'web', updated_at = now() "
        "WHERE id = %s AND user_id = %s AND kind = 'supplement'",
        (json.dumps(etiqueta), pote["id"], user_id),
    )
    return True


def _en_segundo_plano(user_id: str, nombre: str, marca, producto) -> None:
    r = buscar(user_id, marca, producto)
    if r.get("estado") != "encontrada":
        return
    # el coach guarda el pote en el MISMO turno que lanza esto; si la búsqueda le ganó, se espera un poco
    for espera in (0, 20, 60):
        if espera:
            time.sleep(espera)
        try:
            if completar_pote(user_id, nombre, r):
                logger.info("💊 [P1-PLAN-LOTE-1103] tabla de vitaminas de internet completada en la Alacena")
                return
        except Exception as e:
            logger.warning(f"⚠️ [P1-PLAN-LOTE-1103] no se pudo completar el pote: {type(e).__name__}")
            return


def lanzar(user_id: str, nombre: str, marca, producto) -> dict:
    """Lo que la herramienta hace con `vitaminas=true`: la caché si ya se buscó; si no, la búsqueda en segundo plano."""
    if not _ew.activa():
        return {"estado": "apagada"}
    if not str(producto or "").strip():
        return {"estado": "no_encontrada"}
    previo = en_cache(marca, producto)
    if previo is not None:
        return previo
    if not hay_cupo(user_id):
        return {"estado": "sin_cupo"}
    try:
        from bg_executor import submit_bg_task
        submit_bg_task(_en_segundo_plano, user_id, nombre or producto, marca, producto,
                       task_name="etiqueta_web_vitaminas", timeout_s=int(_TIMEOUT_BUSQUEDA_S + _TIMEOUT_LECTURA_S + 120))
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-1103] no se pudo lanzar la búsqueda: {type(e).__name__}")
        return {"estado": "error"}
    return {"estado": "buscando"}


def texto_para_el_coach(r: dict, nombre: str) -> str:
    estado = r.get("estado")
    if estado == "encontrada":
        txt = ("Tengo su tabla Supplement Facts (de internet): "
               + json.dumps(r.get("supplement_facts") or {}, ensure_ascii=False))
        if r.get("porciones"):
            txt += f"; {int(r['porciones'])} porciones por envase"
        _fuente = __import__("contenido_externo").fuente_segura(r.get("fuente"))
        if _fuente:
            txt += f" (fuente: {_fuente})"
        return txt + (". (Para el asistente: guárdalo EN ESTE TURNO con guardar_suplemento fuente='web', etiqueta="
                      "{kcal: 0, protein_g: 0, carbs_g: 0, fats_g: 0, supplement_facts: ESTA tabla} y las porciones; "
                      "si contó cuánto tomó, llama analizar_suplemento con etiqueta=ESTA tabla y esa cantidad. Dile en "
                      "una frase que la tabla salió de internet y que, si su pote dice otra cosa, te mande su foto.)")
    if estado == "buscando":
        return (f"Busco su tabla en internet en segundo plano (tarda alrededor de un minuto) y la completo sola en el "
                f"pote «{nombre}» de su Alacena. (Para el asistente: guárdalo YA con guardar_suplemento con ESE MISMO "
                "nombre, SIN etiqueta y sin pedirle foto; dile en una frase que lo guardaste y que estás buscando su "
                "tabla en internet, y que en un momento te puede preguntar qué le aportó lo que tomó. No inventes "
                "nutrientes ni dosis mientras tanto.)")
    if estado == "no_encontrada":
        return ("No encontré su tabla en internet. (Para el asistente: guárdalo SIN etiqueta y pídele UNA foto de la "
                "tabla Supplement Facts; no vuelvas a buscarlo.)")
    return ("Ahora no pude buscar en internet. (Para el asistente: guárdalo SIN etiqueta y pídele UNA foto de la tabla "
            "Supplement Facts; no digas que buscaste.)")
