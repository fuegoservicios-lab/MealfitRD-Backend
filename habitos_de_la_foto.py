# backend/habitos_de_la_foto.py
# [ESCANER-HABITOS-Y-RELLENOS · 2026-11-14] Extrae hábitos alimentarios recientes del usuario
# (especialmente rellenos de pan/sándwich/tostada, porciones de huevo y tipo de pancakes) desde
# `consumed_meals` y `user_facts` para personalizar las dudas y supuestos del escáner de fotos.
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import contextvars
from datetime import datetime, timedelta, timezone
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional
import unicodedata

logger = logging.getLogger(__name__)

_CURRENT_HABITOS_FOTO: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "current_habitos_foto", default=None
)

_CACHE_TTL_S = 300.0
_CACHE_MAX_USERS = 512
_HABITOS_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()

_PATRON_PAN_BASE = re.compile(
    r"\b(?:sandwich(?:es)?|emparedados?|pan(?:es|ecillos?)?|tostadas?|wraps?|burritos?|arepas?|croissants?|bagels?)\b",
    re.IGNORECASE,
)

_PATRON_PAN_INICIO = re.compile(
    r"^(?:\d+\s+|(?:un|una|dos|tres|medio|media)\s+)?(?:sandwich(?:es)?|emparedados?|pan(?:es|ecillos?)?|tostadas?|wraps?|burritos?|"
    r"arepas?|croissants?|bagels?)\b",
    re.IGNORECASE,
)
# «tostadas de mantequilla de maní», «sándwich de cottage», «pan con aguacate» (no «pan de agua/molde…»: es el tipo de pan)
_PATRON_PAN_PLATO = re.compile(
    r"\b(?:sandwich(?:es)?|emparedados?|pan(?:es|ecillos?)?|tostadas?|wraps?|arepas?|croissants?|bagels?)\s+(?:de|con)\s+"
    r"(?!(?:agua|molde|sobao|centeno|maiz|ajo|yuca|batata|trigo|avena|masa\s+madre)\b)",
    re.IGNORECASE,
)

# Catálogo ordenado de rellenos frecuentes: los compuestos van primero para capturar combos reales
# (ej. "queso cottage y mantequilla de maní") antes de contar ingredientes por separado.
_CATALOGO_RELLENOS_PAN: tuple[dict[str, Any], ...] = (
    {
        "id": "cottage_mani",
        "texto": "Cottage y mantequilla de maní",
        "tokens": ("cottage", "mantequilla de mani"),
        "nombre_sufijo": "con queso cottage y mantequilla de maní",
        "delta": {"calories": 155, "protein": 8.5, "carbs": 4.0, "healthy_fats": 11.5},
    },
    {
        "id": "mani_queso",
        "texto": "Mantequilla de maní y queso",
        "tokens": ("mantequilla de mani", "queso"),
        "nombre_sufijo": "con mantequilla de maní y queso",
        "delta": {"calories": 175, "protein": 9.0, "carbs": 4.0, "healthy_fats": 13.5},
    },
    {
        "id": "jamon_queso",
        "texto": "Jamón y queso",
        "tokens": ("jamon", "queso"),
        "nombre_sufijo": "con jamón y queso",
        "delta": {"calories": 130, "protein": 11.0, "carbs": 2.0, "healthy_fats": 8.5},
    },
    {
        "id": "pavo_queso",
        "texto": "Pavo y queso",
        "tokens": ("pavo", "queso"),
        "nombre_sufijo": "con pavo y queso",
        "delta": {"calories": 125, "protein": 12.0, "carbs": 2.0, "healthy_fats": 7.5},
    },
    {
        "id": "huevo_queso",
        "texto": "Huevo y queso",
        "tokens": ("huevo", "queso"),
        "nombre_sufijo": "con huevo y queso",
        "delta": {"calories": 155, "protein": 12.0, "carbs": 1.5, "healthy_fats": 11.0},
    },
    {
        "id": "cottage",
        "texto": "Queso cottage",
        "tokens": ("cottage",),
        "nombre_sufijo": "con queso cottage",
        "delta": {"calories": 55, "protein": 6.0, "carbs": 2.0, "healthy_fats": 2.5},
    },
    {
        "id": "mani",
        "texto": "Mantequilla de maní",
        "tokens": ("mantequilla de mani",),
        "nombre_sufijo": "con mantequilla de maní",
        "delta": {"calories": 95, "protein": 4.0, "carbs": 3.0, "healthy_fats": 8.0},
    },
    {
        "id": "queso_crema",
        "texto": "Queso crema",
        "tokens": ("queso crema",),
        "nombre_sufijo": "con queso crema",
        "delta": {"calories": 95, "protein": 2.0, "carbs": 1.5, "healthy_fats": 9.0},
    },
    {
        "id": "aguacate",
        "texto": "Aguacate",
        "tokens": ("aguacate",),
        "nombre_sufijo": "con aguacate",
        "delta": {"calories": 85, "protein": 1.0, "carbs": 4.5, "healthy_fats": 7.5},
    },
    {
        "id": "atun",
        "texto": "Atún",
        "tokens": ("atun",),
        "nombre_sufijo": "con atún",
        "delta": {"calories": 100, "protein": 14.0, "carbs": 0.5, "healthy_fats": 4.5},
    },
    {
        "id": "pollo",
        "texto": "Pollo",
        "tokens": ("pollo",),
        "nombre_sufijo": "con pollo",
        "delta": {"calories": 110, "protein": 15.0, "carbs": 0.5, "healthy_fats": 4.5},
    },
    {
        "id": "jamon",
        "texto": "Solo jamón",
        "tokens": ("jamon",),
        "nombre_sufijo": "con jamón",
        "delta": {"calories": 55, "protein": 7.0, "carbs": 1.0, "healthy_fats": 2.5},
    },
    {
        "id": "queso",
        "texto": "Solo queso",
        "tokens": ("queso",),
        "nombre_sufijo": "con queso",
        "delta": {"calories": 85, "protein": 6.0, "carbs": 1.0, "healthy_fats": 6.5},
    },
)

RELLENOS_CATALOGO_POR_ID = {r["id"]: r for r in _CATALOGO_RELLENOS_PAN}

_PATRON_FRUTA_BOWL_BASE = re.compile(
    r"\b(?:frutas?|macedonia|ensalada\s+de\s+frutas?|lechosa|papaya|sandia|melon|pina|fresas?|mango|kiwi|arandanos?|uvas?|bowl|tazon|yogur(?:t)?|avena)\b",
    re.IGNORECASE,
)

_CATALOGO_EXTRAS_FRUTA_BOWL: tuple[dict[str, Any], ...] = (
    {
        "id": "miel_granola",
        "texto": "Con miel y granola",
        "tokens": ("miel", "granola"),
        "delta": {"calories": 175, "protein": 3.0, "carbs": 35.0, "healthy_fats": 3.0},
    },
    {
        "id": "miel",
        "texto": "Con miel",
        "tokens": ("miel",),
        "delta": {"calories": 65, "protein": 0.0, "carbs": 17.0, "healthy_fats": 0.0},
    },
    {
        "id": "yogur",
        "texto": "Con yogur griego",
        "tokens": ("yogur",),
        "delta": {"calories": 85, "protein": 8.0, "carbs": 7.0, "healthy_fats": 2.0},
    },
    {
        "id": "granola",
        "texto": "Con granola",
        "tokens": ("granola",),
        "delta": {"calories": 110, "protein": 3.0, "carbs": 18.0, "healthy_fats": 3.0},
    },
    {
        "id": "leche_condensada",
        "texto": "Con leche condensada",
        "tokens": ("leche condensada",),
        "delta": {"calories": 130, "protein": 3.0, "carbs": 22.0, "healthy_fats": 3.5},
    },
    {
        "id": "mani",
        "texto": "Con mantequilla de maní",
        "tokens": ("mantequilla de mani",),
        "delta": {"calories": 95, "protein": 4.0, "carbs": 3.0, "healthy_fats": 8.0},
    },
    {
        "id": "chia",
        "texto": "Con semillas de chía",
        "tokens": ("chia",),
        "delta": {"calories": 60, "protein": 2.0, "carbs": 5.0, "healthy_fats": 4.0},
    },
)

EXTRAS_FRUTA_CATALOGO_POR_ID = {r["id"]: r for r in _CATALOGO_EXTRAS_FRUTA_BOWL}

_PATRON_ENSALADA_BASE = re.compile(
    r"\b(?:ensalada|vegetales?|lechuga|tomate|pepino|remolacha|repollo|espinaca|rucula)\b",
    re.IGNORECASE,
)

_CATALOGO_ADEREZOS_ENSALADA: tuple[dict[str, Any], ...] = (
    {
        "id": "aceite_oliva",
        "texto": "Con aceite de oliva",
        "tokens": ("aceite",),
        "delta": {"calories": 90, "protein": 0.0, "carbs": 0.0, "healthy_fats": 10.0},
    },
    {
        "id": "vinagreta",
        "texto": "Con vinagreta / aderezo",
        "tokens": ("vinagreta",),
        "delta": {"calories": 80, "protein": 0.0, "carbs": 2.0, "healthy_fats": 8.0},
    },
    {
        "id": "mayonesa",
        "texto": "Con mayonesa / crema",
        "tokens": ("mayonesa",),
        "delta": {"calories": 100, "protein": 0.0, "carbs": 1.0, "healthy_fats": 11.0},
    },
    {
        "id": "aguacate",
        "texto": "Con aguacate y limón",
        "tokens": ("aguacate",),
        "delta": {"calories": 85, "protein": 1.0, "carbs": 4.5, "healthy_fats": 7.5},
    },
)

ADEREZOS_ENSALADA_POR_ID = {r["id"]: r for r in _CATALOGO_ADEREZOS_ENSALADA}


def _ascii(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("crema de mani", "mantequilla de mani").replace("peanut butter", "mantequilla de mani")
    t = t.replace("cottage cheese", "cottage").replace("yogurt", "yogur")
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    return " ".join(t.split())


def _texto_comida(row: Dict[str, Any]) -> str:
    partes = [str(row.get("meal_name") or "")]
    ings = row.get("ingredients")
    if isinstance(ings, list):
        partes.extend(str(x or "") for x in ings)
    elif isinstance(ings, str):
        partes.append(ings)
    return _ascii(" ".join(partes))


def _matches_tokens(texto_ascii: str, tokens: tuple[str, ...]) -> bool:
    for tok in tokens:
        if tok == "queso":
            # Evitar que "queso cottage" o "queso crema" cuenten como "queso" genérico si buscábamos combo distinto
            sin_especificos = texto_ascii.replace("queso cottage", "").replace("queso crema", "")
            if not re.search(r"\bqueso\b", sin_especificos):
                return False
        elif not re.search(r"\b" + re.escape(tok) + r"\b", texto_ascii):
            return False
    return True


def extraer_habitos_desde_registros(
    meals: Optional[List[Dict[str, Any]]],
    facts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Pura (sin I/O): extrae rellenos frecuentes de pan/sándwich y otros hábitos útiles para el escáner."""
    conteo_rellenos: Counter[str] = Counter()
    conteo_extras_fruta: Counter[str] = Counter()
    conteo_aderezos: Counter[str] = Counter()
    conteo_huevos: Counter[str] = Counter()
    conteo_pancakes: Counter[str] = Counter()

    for m in meals or []:
        if not isinstance(m, dict):
            continue
        txt = _texto_comida(m)
        if not txt:
            continue

        # [REVISION-ESCANER-14NOV] «Arroz con pollo y pan» no es un sándwich de pollo: el pan tiene que SER un plato en el nombre
        # (empieza por él, o «tostadas de/con…», «sándwich de…»). Antes cualquier comida con «pan» y «pollo» enseñaba al
        # escáner un relleno falso.
        nombre = _ascii(m.get("meal_name") or "")
        if _PATRON_PAN_INICIO.match(nombre) or _PATRON_PAN_PLATO.search(nombre):
            matched_ids = set()
            for entry in _CATALOGO_RELLENOS_PAN:
                # Si ya coincidió un combo que contiene "cottage" o "mantequilla de mani", no duplicar sus partes simples
                if entry["id"] in ("cottage", "mani", "queso", "jamon") and any(
                    tok in matched_ids for tok in ("cottage_mani", "mani_queso", "jamon_queso", "pavo_queso", "huevo_queso")
                ):
                    continue
                if _matches_tokens(txt, entry["tokens"]):
                    conteo_rellenos[entry["id"]] += 1
                    matched_ids.add(entry["id"])

        if _PATRON_FRUTA_BOWL_BASE.search(txt):
            matched_f = set()
            for entry in _CATALOGO_EXTRAS_FRUTA_BOWL:
                if entry["id"] in ("miel", "granola") and "miel_granola" in matched_f:
                    continue
                if _matches_tokens(txt, entry["tokens"]):
                    conteo_extras_fruta[entry["id"]] += 1
                    matched_f.add(entry["id"])

        if _PATRON_ENSALADA_BASE.search(txt):
            for entry in _CATALOGO_ADEREZOS_ENSALADA:
                if _matches_tokens(txt, entry["tokens"]):
                    conteo_aderezos[entry["id"]] += 1

        for h in re.findall(r"\b([1-6])\s+huevos?(?:\s+(?:hervidos?|revueltos?|fritos?|enteros?))?\b", txt):
            conteo_huevos[f"{h} huevos"] += 1

        if re.search(r"\bpancakes?\b", txt):
            if "avena" in txt:
                conteo_pancakes["pancakes de avena"] += 1
            elif "proteina" in txt:
                conteo_pancakes["pancakes de proteina"] += 1

    for f in facts or []:
        fact_txt = _ascii(f.get("fact") if isinstance(f, dict) else f)
        if not fact_txt:
            continue
        if _PATRON_PAN_BASE.search(fact_txt) or "cottage" in fact_txt or "mantequilla de mani" in fact_txt:
            for entry in _CATALOGO_RELLENOS_PAN:
                if _matches_tokens(fact_txt, entry["tokens"]):
                    conteo_rellenos[entry["id"]] += 2
                    break
        if _PATRON_FRUTA_BOWL_BASE.search(fact_txt) or "miel" in fact_txt or "granola" in fact_txt:
            for entry in _CATALOGO_EXTRAS_FRUTA_BOWL:
                if _matches_tokens(fact_txt, entry["tokens"]):
                    conteo_extras_fruta[entry["id"]] += 2
                    break
        if _PATRON_ENSALADA_BASE.search(fact_txt) or "aceite de oliva" in fact_txt or "vinagreta" in fact_txt:
            for entry in _CATALOGO_ADEREZOS_ENSALADA:
                if _matches_tokens(fact_txt, entry["tokens"]):
                    conteo_aderezos[entry["id"]] += 2
                    break

    rellenos_pan = [
        {
            "id": rid,
            "texto": RELLENOS_CATALOGO_POR_ID[rid]["texto"],
            "count": count,
        }
        for rid, count in conteo_rellenos.most_common(4)
        if rid in RELLENOS_CATALOGO_POR_ID
    ]
    extras_fruta = [
        {
            "id": eid,
            "texto": EXTRAS_FRUTA_CATALOGO_POR_ID[eid]["texto"],
            "count": count,
        }
        for eid, count in conteo_extras_fruta.most_common(4)
        if eid in EXTRAS_FRUTA_CATALOGO_POR_ID
    ]
    aderezos_ensalada = [
        {
            "id": aid,
            "texto": ADEREZOS_ENSALADA_POR_ID[aid]["texto"],
            "count": count,
        }
        for aid, count in conteo_aderezos.most_common(3)
        if aid in ADEREZOS_ENSALADA_POR_ID
    ]

    partes_resumen: List[str] = []
    if rellenos_pan:
        lista_r = ", ".join(f"{_ascii(r['texto'])} ({r['count']}x)" for r in rellenos_pan[:3])
        partes_resumen.append(f"En pan/sandwich/tostadas suele usar: {lista_r}.")
    if extras_fruta:
        lista_f = ", ".join(f"{_ascii(r['texto'])} ({r['count']}x)" for r in extras_fruta[:2])
        partes_resumen.append(f"En frutas/bowls suele usar: {lista_f}.")
    if aderezos_ensalada:
        lista_a = ", ".join(f"{_ascii(r['texto'])} ({r['count']}x)" for r in aderezos_ensalada[:2])
        partes_resumen.append(f"En ensaladas suele usar: {lista_a}.")
    if conteo_huevos:
        top_h = conteo_huevos.most_common(1)[0][0]
        partes_resumen.append(f"Porcion frecuente de huevos: {top_h}.")
    if conteo_pancakes:
        top_p = conteo_pancakes.most_common(1)[0][0]
        partes_resumen.append(f"Suele hacer: {top_p}.")

    resumen_prompt = " ".join(partes_resumen)[:360]
    return {
        "resumen_prompt": resumen_prompt,
        "rellenos_pan": rellenos_pan,
        "extras_fruta": extras_fruta,
        "aderezos_ensalada": aderezos_ensalada,
    }


def invalidate_habitos_cache(user_id: Optional[str] = None) -> None:
    with _CACHE_LOCK:
        if user_id:
            _HABITOS_CACHE.pop(user_id, None)
        else:
            _HABITOS_CACHE.clear()
    try:
        from rutina_del_coach import invalidate_rutina_cache
        invalidate_rutina_cache(user_id)
    except Exception:
        pass


def obtener_habitos_foto(user_id: Optional[str]) -> Dict[str, Any]:
    """Obtiene los hábitos recientes del usuario para el escáner de fotos (fail-open)."""
    vacio = {"resumen_prompt": "", "rellenos_pan": [], "extras_fruta": [], "aderezos_ensalada": []}
    if not user_id:
        return vacio

    ahora = time.monotonic()
    with _CACHE_LOCK:
        cached = _HABITOS_CACHE.get(user_id)
        if cached and (ahora - cached[0]) < _CACHE_TTL_S:
            return dict(cached[1])

    try:
        from db_facts import get_all_user_facts, get_consumed_meals_since
        since_iso = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        meals = get_consumed_meals_since(user_id, since_iso, include_ingredients=True) or []
        if len(meals) > 40:
            meals = sorted(
                meals,
                key=lambda r: str(r.get("consumed_at") or ""),
                reverse=True,
            )[:40]
        facts = get_all_user_facts(user_id) or []
        out = extraer_habitos_desde_registros(meals, facts)
    except Exception as e:
        logger.debug(f"[habitos_de_la_foto] omitido por error no bloqueante: {e}")
        return vacio

    with _CACHE_LOCK:
        if len(_HABITOS_CACHE) >= _CACHE_MAX_USERS:
            oldest_key = min(_HABITOS_CACHE, key=lambda k: _HABITOS_CACHE[k][0])
            _HABITOS_CACHE.pop(oldest_key, None)
        _HABITOS_CACHE[user_id] = (ahora, out)
    return dict(out)


def get_current_habitos_foto() -> Optional[Dict[str, Any]]:
    return _CURRENT_HABITOS_FOTO.get()


@contextmanager
def vision_habitos_scope(habitos: Optional[Dict[str, Any]]):
    token = _CURRENT_HABITOS_FOTO.set(habitos)
    try:
        yield
    finally:
        _CURRENT_HABITOS_FOTO.reset(token)
