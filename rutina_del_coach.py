# -*- coding: utf-8 -*-
"""[P1-COACH-RUTINA-GUSTOS · 2026-11-15] Aprendizaje continuo de rutinas y gustos del usuario.

Extrae de forma determinista (sin LLM) los platos frecuentes y recientes por franja
(`desayuno`, `almuerzo`, `cena`, `merienda`) junto con rellenos/combinaciones aprendidas
de `consumed_meals` (últimos 30 días) y `user_facts`, para que el coach recomiende comidas
de forma inteligente, en porciones caseras (sin gramos de balanza) y sin hacer restas
absurdas sobre el presupuesto de mañana.

Knob: `MEALFIT_COACH_RUTINA_GUSTOS` (True por defecto).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional
import unicodedata

from knobs import _env_bool

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 300.0
_CACHE_MAX_USERS = 512
_RUTINA_CACHE: Dict[str, tuple[float, tuple[list[dict], list[dict]]]] = {}
_CACHE_LOCK = threading.Lock()

_SLOTS_ORDEN = ("desayuno", "almuerzo", "cena", "merienda")
_SLOT_ETIQUETA = {
    "desayuno": "Desayuno",
    "almuerzo": "Almuerzo",
    "cena": "Cena",
    "merienda": "Merienda",
}

_SUPP_RE = re.compile(
    r"\b(?:creatina|creatine|capsula|pastilla|multivitamin\w*|vitamina\w*|"
    r"mineral\w*|magnesio|colageno|omega\s*3|bcaa|glutamina|suplement\w*|zinc)\b",
    re.IGNORECASE,
)

_STOPWORDS = frozenset({
    "con", "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "y", "e", "o", "en", "al", "para", "por", "sin", "mas", "sobre", "junto",
    "pequeno", "pequena", "pequenos", "pequenas", "grande", "grandes",
    "mediano", "mediana", "medianos", "medianas", "casero", "casera",
    "caseros", "caseras", "bowl", "tazon", "plato", "porcion", "porciones",
    "aprox", "aproximadamente", "kcal", "extra", "opcional",
})

_PLURAL_MAP = {
    "sandwiches": "sandwich",
    "huevos": "huevo",
    "galletas": "galleta",
    "tostadas": "tostada",
    "pancakes": "pancake",
    "vegetales": "vegetal",
    "frutas": "fruta",
    "panes": "pan",
    "panecillos": "panecillo",
    "tortitas": "tortita",
    "claras": "clara",
    "nueces": "nuez",
    "fresas": "fresa",
    "uvas": "uva",
    "maduros": "maduro",
    "tostones": "toston",
}

_GUSTO_FACT_RE = re.compile(
    r"\b(?:gusta|encanta|prefier|favorit|suel|siempre|acostumbr|cottage|mani|"
    r"avena|sandwich|desayun|almorz|cen|meriend|pollo|cerdo|res|pescado|fruta|miel)\b",
    re.IGNORECASE,
)


def activo() -> bool:
    """Knob `MEALFIT_COACH_RUTINA_GUSTOS` (default True)."""
    return _env_bool("MEALFIT_COACH_RUTINA_GUSTOS", True)


def invalidate_rutina_cache(user_id: Optional[str] = None) -> None:
    with _CACHE_LOCK:
        if user_id:
            _RUTINA_CACHE.pop(str(user_id), None)
        else:
            _RUTINA_CACHE.clear()


def _ascii(texto: Any) -> str:
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).strip()


def _es_registro_excluido(row: Dict[str, Any]) -> bool:
    mt = _ascii(row.get("meal_type") or "")
    if "suplement" in mt or row.get("suplemento") or "complement" in mt:
        return True
    try:
        kcal = float(row.get("calories") or 0.0)
    except (TypeError, ValueError):
        kcal = 0.0
    name_norm = _ascii(row.get("meal_name") or "")
    if kcal <= 50.0:
        return True
    if kcal < 100.0 and _SUPP_RE.search(name_norm):
        return True
    return False


def _parse_nutrition(row: Dict[str, Any]) -> Dict[str, Any]:
    nut = row.get("nutrition")
    if isinstance(nut, str):
        try:
            nut = json.loads(nut)
        except Exception:
            nut = {}
    return nut if isinstance(nut, dict) else {}


def _datos_base_registro(row: Dict[str, Any]) -> tuple[str, float, float]:
    """Devuelve `(nombre, kcal, protein)` limpiando extras circunstanciales si existe `base_meal`."""
    nut = _parse_nutrition(row)
    base = nut.get("base_meal")
    if isinstance(base, dict) and str(base.get("meal_name") or "").strip():
        nombre = str(base.get("meal_name")).strip()
        try:
            kcal = float(base.get("calories") if base.get("calories") is not None else (row.get("calories") or 0.0))
        except (TypeError, ValueError):
            kcal = 0.0
        try:
            prot = float(base.get("protein") if base.get("protein") is not None else (row.get("protein") or 0.0))
        except (TypeError, ValueError):
            prot = 0.0
        return nombre, kcal, prot

    nombre = str(row.get("meal_name") or "").strip()
    try:
        kcal = float(row.get("calories") or 0.0)
    except (TypeError, ValueError):
        kcal = 0.0
    try:
        prot = float(row.get("protein") or 0.0)
    except (TypeError, ValueError):
        prot = 0.0
    return nombre, kcal, prot


def _compactar_nombre(nombre: str) -> str:
    limpio = re.sub(r"\(\s*[≈~]?\s*\d+(?:[.,]\d+)?\s*(?:g|gr|ml|oz|kcal)\s*\)", "", str(nombre or ""), flags=re.I)
    limpio = re.sub(r"\s+", " ", limpio).strip(" -·,")
    if len(limpio) > 95:
        limpio = limpio[:92].rstrip() + "..."
    return limpio


def _tokens_plato(nombre: str) -> frozenset[str]:
    txt = _ascii(nombre)
    txt = re.sub(r"\([^)]*\)", " ", txt)
    txt = re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:g|gr|ml|oz|tazas?|cucharad\w*|pieza\w*|unidad\w*)?\b", " ", txt)
    txt = re.sub(r"[^a-z\s]+", " ", txt)
    tokens: set[str] = set()
    for tok in txt.split():
        if tok in _STOPWORDS or len(tok) <= 1:
            continue
        tokens.add(_PLURAL_MAP.get(tok, tok))
    return frozenset(tokens)


def _clave_plato(nombre: str) -> str:
    toks = _tokens_plato(nombre)
    if not toks:
        return _ascii(nombre)[:40]
    return " ".join(sorted(toks)[:7])


def _buscar_bucket_similar(buckets: Dict[str, Dict[str, Any]], clave: str, toks: frozenset[str]) -> Optional[str]:
    if clave in buckets:
        return clave
    if len(toks) < 2:
        return None
    mejor_clave: Optional[str] = None
    mejor_sim = 0.0
    for k, b in buckets.items():
        b_toks = b.get("tokens") or frozenset()
        if not b_toks:
            continue
        inter = len(toks & b_toks)
        if inter < 2:
            continue
        union = len(toks | b_toks)
        sim = inter / float(union) if union else 0.0
        if sim >= 0.60 and sim > mejor_sim:
            mejor_sim = sim
            mejor_clave = k
    return mejor_clave


def _extraer_gustos_facts(facts: Optional[List[Dict[str, Any]]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for f in facts or []:
        raw = str(f.get("fact") if isinstance(f, dict) else f or "").strip()
        if not raw or len(raw) < 6:
            continue
        norm = _ascii(raw)
        if not _GUSTO_FACT_RE.search(norm):
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(raw[:110].rstrip() + ("..." if len(raw) > 110 else ""))
        if len(out) >= 4:
            break
    return out


def extraer_rutina_desde_registros(
    meals: Optional[List[Dict[str, Any]]],
    facts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Función pura (sin I/O): agrupa los platos habituales y recientes por franja y extrae hábitos."""
    from chat_history_context import slot_canonico

    grupos: Dict[str, Dict[str, Dict[str, Any]]] = {s: {} for s in _SLOTS_ORDEN}

    for row in meals or []:
        if not isinstance(row, dict) or _es_registro_excluido(row):
            continue
        slot = slot_canonico(row.get("meal_type"))
        if slot not in grupos:
            continue
        nombre_raw, kcal, prot = _datos_base_registro(row)
        nombre = _compactar_nombre(nombre_raw)
        if not nombre:
            continue
        toks = _tokens_plato(nombre)
        clave = _clave_plato(nombre)
        if not clave:
            continue
        ts = str(row.get("consumed_at") or row.get("created_at") or "")
        match_k = _buscar_bucket_similar(grupos[slot], clave, toks)
        bucket = grupos[slot].get(match_k) if match_k else None
        if bucket is None:
            grupos[slot][clave] = {
                "nombre": nombre,
                "tokens": toks,
                "count": 1,
                "sum_kcal": kcal,
                "sum_prot": prot,
                "last_ts": ts,
            }
        else:
            bucket["count"] += 1
            bucket["sum_kcal"] += kcal
            bucket["sum_prot"] += prot
            if ts >= bucket["last_ts"]:
                bucket["last_ts"] = ts
                bucket["nombre"] = nombre
                bucket["tokens"] = bucket["tokens"] | toks

    por_franja: Dict[str, List[Dict[str, Any]]] = {}
    for slot in _SLOTS_ORDEN:
        items = sorted(
            grupos[slot].values(),
            key=lambda x: (x["count"], x["last_ts"]),
            reverse=True,
        )[:3]
        por_franja[slot] = [
            {
                "nombre": it["nombre"],
                "count": it["count"],
                "kcal": int(round(it["sum_kcal"] / max(1, it["count"]))),
                "protein": int(round(it["sum_prot"] / max(1, it["count"]))),
                "habitual": it["count"] >= 2,
            }
            for it in items
        ]

    habitos: Dict[str, Any] = {}
    try:
        from habitos_de_la_foto import extraer_habitos_desde_registros
        habitos = extraer_habitos_desde_registros(meals, facts) or {}
    except Exception:
        habitos = {}

    return {
        "por_franja": por_franja,
        "habitos": habitos,
        "gustos_facts": _extraer_gustos_facts(facts),
    }


def _cargar_historial_usuario(user_id: str) -> tuple[list[dict], list[dict]]:
    ahora = time.monotonic()
    uid = str(user_id)
    with _CACHE_LOCK:
        cached = _RUTINA_CACHE.get(uid)
        if cached and (ahora - cached[0]) < _CACHE_TTL_S:
            meals_c, facts_c = cached[1]
            return list(meals_c), list(facts_c)

    from db_facts import get_all_user_facts, get_consumed_meals_since

    since_iso = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    meals = get_consumed_meals_since(uid, since_iso, include_ingredients=True) or []
    facts = get_all_user_facts(uid) or []

    with _CACHE_LOCK:
        if len(_RUTINA_CACHE) >= _CACHE_MAX_USERS:
            oldest = min(_RUTINA_CACHE, key=lambda k: _RUTINA_CACHE[k][0])
            _RUTINA_CACHE.pop(oldest, None)
        _RUTINA_CACHE[uid] = (ahora, (list(meals), list(facts)))
    return list(meals), list(facts)


def _combinar_con_diario_hoy(meals: list[dict], consumed_today: Any) -> list[dict]:
    if not consumed_today or not isinstance(consumed_today, list):
        return meals
    seen_ids = {str(m.get("id")) for m in meals if isinstance(m, dict) and m.get("id")}
    merged = list(meals)
    for row in consumed_today:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        if rid and str(rid) in seen_ids:
            continue
        merged.append(row)
    return merged


def formatear_bloque_rutina(rutina: Dict[str, Any]) -> str:
    """Renderiza el bloque compacto para el system prompt a partir de `extraer_rutina_desde_registros`."""
    por_franja = rutina.get("por_franja") or {}
    habitos = rutina.get("habitos") or {}
    resumen_hab = str(habitos.get("resumen_prompt") or "").strip()
    gustos_facts = rutina.get("gustos_facts") or []

    lineas_franja: list[str] = []
    for slot in _SLOTS_ORDEN:
        platos = por_franja.get(slot) or []
        if not platos:
            continue
        partes = []
        for p in platos:
            marca = f" ({p['count']}x habitual)" if p.get("habitual") else ""
            macros = f"~{p['kcal']} kcal" + (f", {p['protein']}g prot" if p.get("protein") else "")
            partes.append(f"«{p['nombre']}»{marca} [{macros}]")
        lineas_franja.append(f"- {_SLOT_ETIQUETA[slot]}: " + " · ".join(partes))

    if not lineas_franja and not resumen_hab and not gustos_facts:
        return ""

    out = "\n\n🔄 RUTINA Y GUSTOS APRENDIDOS DEL USUARIO (aprendizaje continuo de su diario real):\n"
    if lineas_franja:
        out += "\n".join(lineas_franja) + "\n"
    if resumen_hab:
        out += f"- Combinaciones y porciones aprendidas: {resumen_hab}\n"
    if gustos_facts:
        out += "- Preferencias declaradas: " + " | ".join(gustos_facts) + "\n"

    out += (
        "REGLAS PARA RECOMENDAR CON INTELIGENCIA:\n"
        "1) PRIORIZA SU RUTINA REAL: cuando pida ideas para comer, retomar su rutina o planear hoy/mañana, "
        "reconoce y propón primero uno de SUS platos habituales/recientes de esa franja (puedes sugerirlo tal cual "
        "o ajustar ligeramente la porción para acercarse a su meta), más como mucho 1 alternativa compatible con sus "
        "gustos o su Nevera — NO recites platos genéricos ajenos si ya tiene hábitos reales en esa franja.\n"
        "2) PORCIONES CASERAS, NO BALANZA: describe la comida en unidades cotidianas (ej. «2 sándwiches de pan integral "
        "con queso cottage, mantequilla de maní y 3–4 huevos hervidos», «un tazón de lechoza, sandía y piña») con solo "
        "las ~kcal y ~g de proteína totales del plato. PROHIBIDO recitar gramos crudos ingrediente por ingrediente "
        "(«240 g de…, 35 g de…, 55 g de…») salvo que el usuario pida explícitamente la receta o el gramaje.\n"
        "3) PROHIBIDO RESTAR DEL PRESUPUESTO DE MAÑANA: si sugieres un plato para mañana, indica cuánto aporta "
        "(~kcal y ~g de proteína) y por qué le encaja, pero NUNCA digas «si lo comes te quedarían ~X kcal y ~Y g» "
        "haciendo restas contables sobre un día que ni ha empezado."
    )
    return out


def para_chat(
    user_id: Optional[str],
    form_data: Any = None,
    plan_vigente: Any = None,
    consumed_today: Any = None,
) -> str:
    """Builder fail-open para inyectar en `chat_with_agent` y `chat_with_agent_stream`."""
    if not activo() or not user_id or user_id == "guest":
        return ""
    try:
        meals, facts = _cargar_historial_usuario(str(user_id))
        meals = _combinar_con_diario_hoy(meals, consumed_today)
        if not meals and not facts:
            return ""
        rutina = extraer_rutina_desde_registros(meals, facts)
        return formatear_bloque_rutina(rutina)
    except Exception as err:
        logger.debug(f"[rutina_del_coach] omitido por error no bloqueante: {err!r}")
        return ""
