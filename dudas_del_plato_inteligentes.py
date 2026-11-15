# backend/dudas_del_plato_inteligentes.py
# [ESCANER-DUDAS-INTELIGENTES · 2026-11-14] Post-procesamiento determinista de dudas del escáner:
# 1. Garantiza pregunta de relleno en alimentos cerrados/compuestos (sándwich, wrap, empanada, arepa).
# 2. Personaliza supuestos y chips con los rellenos habituales del usuario (`habitos_de_la_foto`).
# 3. Filtra preguntas tontas o visualmente obvias (ej. "¿Con o sin sirope?" cuando está seco y supuso "Sin sirope").
# 4. Prioriza dudas por impacto calórico real (rellenos ocultos > identidad > cantidades > grasa de cocción).
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import unicodedata

from habitos_de_la_foto import (
    ADEREZOS_ENSALADA_POR_ID,
    EXTRAS_FRUTA_CATALOGO_POR_ID,
    RELLENOS_CATALOGO_POR_ID,
    _matches_tokens,
)

_PATRON_CERRADO = re.compile(
    r"\b(sandwich(?:es)?|emparedados?|wraps?|burritos?|arepas?(?:\s+rellenas?)?|empanadas?|croissants?)\b",
    re.IGNORECASE,
)
_PATRON_PAN_SUELTO = re.compile(
    r"\b(tostadas?|bagels?|panecillos?|rebanadas?\s+de\s+pan(?:\s+(?:integral|de\s+molde|tostado|multigrano|centeno|sobao|blanco))?|"
    r"pan(?:\s+(?:integral|de\s+molde|tostado|multigrano|centeno|sobao|blanco))?)\b",
    re.IGNORECASE,
)
_PATRON_PLATO_ACOMPANANTE = re.compile(
    r"\b(?:arroz|moro|locrio|chofan|asopao|sancocho|mondongo|sopa|pasta|espaguetis|fettuccine|lasana|pizza|risotto|paella)\b",
    re.IGNORECASE,
)
_PATRON_PAN_CON_UNTABLE = re.compile(
    r"\b(?:pan(?:\s+(?:integral|de\s+molde|tostado|multigrano|centeno|sobao|blanco))?|tostadas?|bagels?|panecillos?)"
    r"\s+(?:de|con|untad[oa]s?\s+con)\s+([^,.;()]+)",
    re.IGNORECASE,
)

_PATRON_FRUTAS_MULTI = re.compile(
    r"\b(frutas?|macedonia|ensalada\s+de\s+frutas?|fruta\s+picada)\b",
    re.IGNORECASE,
)
_FRUTAS_DULCES_BOWL = (
    "lechosa", "papaya", "sandia", "melon", "pina", "fresa", "fresas",
    "mango", "kiwi", "arandano", "arandanos", "uva", "uvas", "melocoton", "guineo", "manzana",
)
_PATRON_FRUTA_UNITARIA = re.compile(
    r"\b(lechosa|papaya|sandia|melon|pina|fresas?|mango|kiwi|arandanos?|uvas?|melocoton)\b",
    re.IGNORECASE,
)
_PATRON_ENSALADA_VEGETALES = re.compile(
    r"\b(ensalada(?!\s+de\s+frutas?)|lechuga|tomate|pepino|remolacha|repollo|rucula)\b",
    re.IGNORECASE,
)
_PATRON_ENSALADA_EXPLICITA = re.compile(
    r"\b(ensalada(?!\s+de\s+frutas?)|lechuga|rucula)\b",
    re.IGNORECASE,
)
_PATRON_CALDO_GUISO = re.compile(
    r"\b(sopa|caldo|sancocho|asopao|consome|potaje|cocido|guisad[oa]s?|estofad[oa]|mondongo)\b",
    re.IGNORECASE,
)

_PATRON_DUDA_RELLENO = re.compile(
    r"\b(llevan?|tienen?|rellen\w*|dentro|unt\w*|ademas|interior|encima)\b",
    re.IGNORECASE,
)

_PATRON_TOPPING_TRIVIAL = re.compile(
    r"\b(sirope|maple|jarabe|miel|caramelo|ketchup|catsup|mayonesa|aderezo|salsa)\b",
    re.IGNORECASE,
)

# [REVISION-ESCANER-14NOV] tras normalizar, «té» y el pronombre «te» son la misma palabra: «¿Te pusiste sirope?» desactivaba el
# filtro de preguntas tontas. El té solo cuenta precedido de artículo/posesivo/preposición («un te», «el te», «de te», «tu te»).
_PATRON_BEBIDA_DUDA = re.compile(
    r"\b(cafe|(?:un|el|al|del|de|tu|su|mi)\s+te|jugo|batid[ao]|licuado|infusion|bebida|chocolate)\b",
    re.IGNORECASE,
)

_PATRON_HUMEDO_DUDA = re.compile(
    r"\b(fruta|frutas|lechosa|papaya|sandia|melon|pina|fresa|mango|kiwi|uva|arandano|manzana|macedonia|bowl|tazon|yogur|yogurt|avena|ensalada|vegetal|lechuga|tomate|remolacha|pepino)\b",
    re.IGNORECASE,
)

_PATRON_OPCION_NEGATIVA = re.compile(
    r"^\s*(?:sin\b|nada\b|ningun[ao]?\b|natural\b|solo\b(?!\s+(?:jamon|queso|claras|huevo|carne|pollo)))",
    re.IGNORECASE,
)


def _norm(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("yogurt", "yogur")
    return " ".join(re.sub(r"[^a-z0-9\s]+", " ", t).split())


def _redondeo_macro(v: float, k: str):
    return int(round(v)) if k == "calories" else round(v, 1)


def _es_duda_tonta(duda: Dict[str, Any], hay_otra_real: bool = False) -> bool:
    """Detecta preguntas de bajo valor o redundantes (ej. sirope en pancakes visualmente secos),
    pero protege aditivos invisibles o translúcidos sobre bebidas, frutas húmedas o ensaladas."""
    preg_n = _norm(duda.get("pregunta") or "")
    sobre_n = _norm(duda.get("sobre") or "")
    ops = duda.get("opciones") or []
    if not preg_n:
        return True

    contexto = f"{preg_n} {sobre_n}"
    # En bebidas (café, té, jugo) lo disuelto (ej. miel, azúcar) no se ve desde fuera
    if _PATRON_BEBIDA_DUDA.search(contexto):
        return False
    # En frutas húmedas, bowls o ensaladas: la miel, yogur, aceite o vinagreta no se distinguen del brillo del agua/jugo
    if _PATRON_HUMEDO_DUDA.search(contexto):
        return False

    # Preguntar "¿Con o sin sirope/ketchup/salsa?" en comida seca cuando la propia IA ya marcó "Sin ..." como supuesto
    if _PATRON_TOPPING_TRIVIAL.search(preg_n):
        sup = next((o for o in ops if o.get("supuesta")), None)
        if sup and _PATRON_OPCION_NEGATIVA.search(_norm(sup.get("texto") or "")):
            return True
        if hay_otra_real and "pan" not in preg_n and "sandwich" not in preg_n:
            return True

    return False


def _detectar_alimento_cerrado(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Devuelve {item_name, cantidad, es_sandwich} si el plato contiene un alimento cerrado."""
    items = result.get("items") or []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()
        m = _PATRON_CERRADO.search(_norm(name))
        if m:
            try:
                qty = max(1, min(6, int(round(float(it.get("quantity") or 1)))))
            except (TypeError, ValueError):
                qty = 1
            return {
                "sobre": name[:60],
                "tipo": m.group(1),
                "cantidad": qty,
                "es_sandwich": any(k in _norm(name) for k in ("sandwich", "emparedado", "wrap", "burrito", "croissant")),
            }

    texto = f"{result.get('meal_name') or ''} {result.get('description') or ''}"
    texto_n = _norm(texto)
    m = _PATRON_CERRADO.search(texto_n)
    if not m:
        return None
    qty = 1
    m_qty = re.search(r"\b(\d+)\s+(?:" + _PATRON_CERRADO.pattern + r")", texto_n, re.IGNORECASE)
    if m_qty:
        try:
            qty = max(1, min(6, int(m_qty.group(1))))
        except (TypeError, ValueError):
            qty = 1
    elif re.search(r"\bdos\s+(?:sandwich(?:es)?|emparedados?|wraps?|empanadas?|arepas?)\b", texto_n):
        qty = 2
    return {
        "sobre": m.group(1)[:60],
        "tipo": m.group(1),
        "cantidad": qty,
        "es_sandwich": any(k in m.group(1) for k in ("sandwich", "emparedado", "wrap", "burrito", "croissant")),
    }


def _duda_es_sobre_relleno(duda: Dict[str, Any], info_cerrado: Dict[str, Any]) -> bool:
    sobre_n = _norm(duda.get("sobre") or "")
    preg_n = _norm(duda.get("pregunta") or "")
    tipo_n = _norm(info_cerrado.get("tipo") or "")
    palabras_pan = ("pan", "panes", "rebanada", "rebanadas", "tostada", "tostadas", "bagel", "bagels", "panecillo", "panecillos")
    if _PATRON_CERRADO.search(sobre_n) or (tipo_n and tipo_n in sobre_n) or any(w in sobre_n.split() for w in palabras_pan):
        return True
    if _PATRON_CERRADO.search(preg_n) and _PATRON_DUDA_RELLENO.search(preg_n):
        return True
    if any(w in preg_n.split() for w in palabras_pan) and _PATRON_DUDA_RELLENO.search(preg_n):
        return True
    return False


def _aclaracion_menciona_relleno(aclaracion: Optional[str]) -> bool:
    acl_n = _norm(aclaracion or "")
    if not acl_n:
        return False
    pistas = ("cottage", "mantequilla", "jamon", "queso", "pavo", "atun", "pollo", "huevo", "aguacate", "sin nada", "sin untable", "pan solo")
    return any(p in acl_n for p in pistas) and (
        bool(_PATRON_CERRADO.search(acl_n))
        or "pan" in acl_n.split()
        or "panes" in acl_n.split()
        or "tostada" in acl_n.split()
        or "tostadas" in acl_n.split()
        or "rellen" in acl_n
        or "dentro" in acl_n
        or "lleva" in acl_n
        or "tiene" in acl_n
        or "untad" in acl_n
    )


def _relleno_asumido_en_pan(result: Dict[str, Any]) -> Optional[str]:
    """Detecta un relleno/untable declarado específicamente SOBRE el pan o tostada, sin
    confundirlo con otros acompañantes del plato (p. ej. jamón o aguacate aparte)."""
    frases_pan: List[str] = []
    for it in result.get("items") or []:
        if not isinstance(it, dict):
            continue
        it_n = _norm(it.get("name") or "")
        if _PATRON_PAN_SUELTO.search(it_n):
            frases_pan.append(it_n)
    # Conservar signos de puntuación (,.;()) al quitar tildes para no cruzar cláusulas de otros acompañantes
    raw_texto = f"{result.get('meal_name') or ''}, {result.get('description') or ''}"
    raw_sin_acentos = "".join(
        c for c in unicodedata.normalize("NFKD", raw_texto.lower())
        if not unicodedata.combining(c)
    )
    for m in _PATRON_PAN_CON_UNTABLE.finditer(raw_sin_acentos):
        clausula = _norm(m.group(1))
        if clausula and not re.match(r"^(?:agua|molde|sobao|centeno|maiz|ajo|yuca|batata|trigo|avena)\b", clausula):
            frases_pan.append(clausula)
    if not frases_pan:
        return None
    objetivo = " ".join(frases_pan)
    for rid in ("cottage_mani", "mani_queso", "jamon_queso", "pavo_queso", "huevo_queso", "cottage", "mani", "queso_crema", "aguacate", "atun", "pollo", "jamon", "queso"):
        entry = RELLENOS_CATALOGO_POR_ID[rid]
        if _matches_tokens(objetivo, entry["tokens"]):
            return rid
    return None


def _cantidad_pan_desde_texto(texto_n: str) -> int:
    m_num = re.search(r"\b(\d+)\s+(?:rebanadas?\s+de\s+)?(?:pan(?:es|ecillos?)?|tostadas?|bagels?)\b", texto_n)
    if m_num:
        try:
            return max(1, min(6, int(m_num.group(1))))
        except (TypeError, ValueError):
            pass
    for palabra, num in (("dos", 2), ("tres", 3), ("cuatro", 4), ("cinco", 5), ("seis", 6)):
        if re.search(rf"\b{palabra}\s+(?:rebanadas?\s+de\s+)?(?:pan(?:es|ecillos?)?|tostadas?|bagels?)\b", texto_n):
            return num
    return 1


def _detectar_pan_suelto_untable(
    result: Dict[str, Any],
    rellenos_usuario: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Detecta pan, rebanadas o tostadas como elemento principal del plato donde puede haber
    un untable oculto (por debajo o por costumbre del usuario), excluyendo pan acompañante."""
    texto_n = _norm(f"{result.get('meal_name') or ''} {result.get('description') or ''}")
    if _PATRON_PLATO_ACOMPANANTE.search(texto_n):
        return None
    if _relleno_asumido_en_pan(result) is not None:
        return None

    sobre = None
    qty = 1
    es_tostada_o_bagel = False
    for it in result.get("items") or []:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()
        name_n = _norm(name)
        m = _PATRON_PAN_SUELTO.search(name_n)
        if m:
            sobre = name[:60]
            try:
                qty = max(1, min(6, int(round(float(it.get("quantity") or 1)))))
            except (TypeError, ValueError):
                qty = 1
            if "tostad" in name_n or "bagel" in name_n:
                es_tostada_o_bagel = True
            break

    if not sobre:
        m = _PATRON_PAN_SUELTO.search(texto_n)
        if not m:
            return None
        sobre = m.group(1)[:60]
        qty = _cantidad_pan_desde_texto(texto_n)
        es_tostada_o_bagel = ("tostad" in m.group(1).lower() or "bagel" in m.group(1).lower())
    elif qty == 1:
        qty = max(qty, _cantidad_pan_desde_texto(texto_n))

    if not (rellenos_usuario or es_tostada_o_bagel or qty >= 2):
        return None

    return {
        "sobre": sobre,
        "tipo": "tostada" if es_tostada_o_bagel else "pan",
        "cantidad": qty,
        "es_sandwich": False,
        "es_pan_suelto": True,
    }


def _relleno_asumido_en_plato(result: Dict[str, Any]) -> Optional[str]:
    texto_n = _norm(f"{result.get('meal_name') or ''} {result.get('description') or ''}")
    for rid in ("cottage_mani", "mani_queso", "jamon_queso", "pavo_queso", "huevo_queso", "cottage", "mani", "queso_crema", "aguacate", "atun", "pollo", "jamon", "queso"):
        entry = RELLENOS_CATALOGO_POR_ID[rid]
        # [REVISION-ESCANER-14NOV] palabra entera: «sándwich de repollo» contenía «pollo» y el ajuste restaba macros de pollo
        if _matches_tokens(texto_n, entry["tokens"]):
            return rid
    return None


def _delta_para_opcion(entry: Dict[str, Any], base_id: Optional[str], qty: int) -> Dict[str, Any]:
    """Calcula el delta de macros respecto al relleno base asumido en la estimación."""
    es_untable = entry["id"] in ("cottage_mani", "mani_queso", "cottage", "mani", "queso_crema", "aguacate")
    if base_id in ("jamon", "pollo", "atun") and es_untable:
        base_delta = {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}
    else:
        base_delta = (RELLENOS_CATALOGO_POR_ID.get(base_id or "") or {}).get("delta") or {
            "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0
        }
    return {
        k: max(-2000, min(2000, _redondeo_macro((entry["delta"][k] - base_delta[k]) * qty, k)))
        for k in ("calories", "protein", "carbs", "healthy_fats")
    }


def _personalizar_opciones_duda_relleno(
    duda: Dict[str, Any],
    rellenos_usuario: List[Dict[str, Any]],
    qty: int,
    base_id: Optional[str],
) -> Dict[str, Any]:
    """Si ya existe una duda de relleno, inserta los rellenos habituales del usuario si faltaban."""
    if not rellenos_usuario:
        return duda
    ops = [dict(o) for o in (duda.get("opciones") or [])]
    if not ops:
        return duda

    textos_n = [_norm(o.get("texto") or "") for o in ops]

    for hab in rellenos_usuario[:2]:
        rid = hab.get("id")
        entry = RELLENOS_CATALOGO_POR_ID.get(rid or "")
        if not entry:
            continue
        if any(all(tok in t for tok in entry["tokens"]) for t in textos_n):
            continue
        nueva_op = {
            "texto": entry["texto"][:40],
            "supuesta": False,
            "ajuste": _delta_para_opcion(entry, base_id, qty),
        }
        if len(ops) < 5:
            ops.append(nueva_op)
        else:
            # Reemplaza la última opción no supuesta para no perder la supuesta
            idx_reemp = next((i for i in range(len(ops) - 1, -1, -1) if not ops[i].get("supuesta")), -1)
            if idx_reemp >= 0:
                ops[idx_reemp] = nueva_op
        textos_n = [_norm(o.get("texto") or "") for o in ops]

    out = dict(duda)
    out["opciones"] = ops[:5]
    return out


def _crear_duda_relleno_cerrado(
    result: Dict[str, Any],
    info_cerrado: Dict[str, Any],
    rellenos_usuario: List[Dict[str, Any]],
) -> Dict[str, Any]:
    qty = info_cerrado["cantidad"]
    es_pan_suelto = bool(info_cerrado.get("es_pan_suelto"))
    porciones = max(1, (qty + 1) // 2) if es_pan_suelto else qty
    base_id = _relleno_asumido_en_pan(result) if es_pan_suelto else _relleno_asumido_en_plato(result)
    base_entry = RELLENOS_CATALOGO_POR_ID.get(base_id or "")

    plural = qty > 1
    if es_pan_suelto:
        if info_cerrado.get("tipo") == "tostada":
            pregunta = "¿Las tostadas llevan algo untado o por dentro?" if plural else "¿La tostada lleva algo untado o por dentro?"
        else:
            pregunta = "¿Las rebanadas de pan llevan algo untado o por dentro?" if plural else "¿El pan lleva algo untado o por dentro?"
    else:
        sujeto = "los sándwiches" if (info_cerrado["es_sandwich"] and plural) else ("el sándwich" if info_cerrado["es_sandwich"] else f"el/la {info_cerrado['tipo']}")
        if base_entry:
            base_lbl = re.sub(r"^solo\s+", "", base_entry["texto"].lower()).strip()
            pregunta = f"¿Qué llevan {sujeto} además de {base_lbl} (o por dentro)?" if plural else f"¿Qué lleva {sujeto} además de {base_lbl} (o por dentro)?"
        else:
            pregunta = f"¿Qué llevan {sujeto} por dentro?" if plural else f"¿Qué lleva {sujeto} por dentro?"

    candidatos_ids: List[str] = []
    if base_id:
        candidatos_ids.append(base_id)
    for hab in rellenos_usuario:
        hid = hab.get("id")
        if hid and hid in RELLENOS_CATALOGO_POR_ID and hid not in candidatos_ids:
            candidatos_ids.append(hid)
    defaults = (
        ("mani", "cottage", "queso_crema", "aguacate", "jamon_queso")
        if es_pan_suelto
        else ("jamon_queso", "cottage_mani", "mani", "huevo_queso", "atun", "queso")
    )
    for default_id in defaults:
        if default_id not in candidatos_ids:
            candidatos_ids.append(default_id)
        if len(candidatos_ids) >= (5 if base_id else 4):
            break

    opciones: List[Dict[str, Any]] = []
    if not base_id:
        opciones.append({
            "texto": "Sin untable / pan solo" if es_pan_suelto else "Solo / sin relleno extra",
            "supuesta": True,
            "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0},
        })

    for cid in candidatos_ids[:5]:
        entry = RELLENOS_CATALOGO_POR_ID[cid]
        es_sup = (cid == base_id)
        ajuste = (
            {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}
            if es_sup
            else _delta_para_opcion(entry, base_id, porciones)
        )
        opciones.append({
            "texto": entry["texto"][:40],
            "supuesta": es_sup,
            "ajuste": ajuste,
        })
        if len(opciones) >= 5:
            break

    # Garantizar exactamente una opción supuesta=True
    if opciones and not any(o.get("supuesta") for o in opciones):
        opciones[0]["supuesta"] = True
        opciones[0]["ajuste"] = {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}

    return {
        "sobre": info_cerrado["sobre"],
        "pregunta": pregunta[:160],
        "opciones": opciones,
    }


def _detectar_fruta_bowl(result: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Detecta bowls de frutas picadas o mezclas de frutas húmedas donde la miel/yogur/granola son invisibles."""
    nombre_desc = _norm(f"{result.get('meal_name') or ''} {result.get('description') or ''}")
    sin_jugos = re.sub(r"\b(?:jugo|batid[ao]|licuado|smoothie)\s+(?:natural\s+)?de\s+\w+(?:\s+(?:y|con)\s+\w+)?", " ", nombre_desc)
    items = result.get("items") or []
    items_n = [_norm((it or {}).get("name") or "") for it in items if isinstance(it, dict)]
    todo_n = f"{sin_jugos} {' '.join(items_n)}".strip()

    frutas_presentes = [f for f in _FRUTAS_DULCES_BOWL if re.search(rf"\b{f}s?\b", todo_n)]
    sobre = next((it.get("name") for it in items if isinstance(it, dict) and (_PATRON_FRUTAS_MULTI.search(_norm(it.get("name"))) or _PATRON_FRUTA_UNITARIA.search(_norm(it.get("name"))))), None)

    es_bowl_frutas = (
        bool(_PATRON_FRUTAS_MULTI.search(sin_jugos))
        or len(set(frutas_presentes)) >= 2
        or (
            bool(_PATRON_FRUTA_UNITARIA.search(todo_n))
            and (
                len(items) <= 2
                or any(k in sin_jugos for k in ("bowl", "tazon", "picad", "trozo", "cubos", "porcion de fruta"))
            )
            and not re.search(r"\b(arroz|pollo|cerdo|res|carne|sandwich|mangu|huevo)\b", sin_jugos)
        )
    )
    if not es_bowl_frutas:
        return None
    return {"sobre": str(sobre or "frutas")[:60]}


def _duda_es_sobre_fruta_extras(duda: Dict[str, Any]) -> bool:
    txt = _norm(f"{duda.get('sobre') or ''} {duda.get('pregunta') or ''}")
    return bool(re.search(r"\b(miel|agave|granola|yogur|condensada|acompanamiento|encima|endulzante|extra|topping)\b", txt)) and (
        bool(_PATRON_FRUTAS_MULTI.search(txt)) or bool(_PATRON_FRUTA_UNITARIA.search(txt)) or "bowl" in txt or "fruta" in txt
    )


def _aclaracion_menciona_fruta_extras(aclaracion: Optional[str]) -> bool:
    acl_n = _norm(aclaracion or "")
    if not acl_n:
        return False
    pistas = ("sin miel", "con miel", "solo fruta", "sola la fruta", "sin nada encima", "sin extras", "con granola", "con yogur", "leche condensada", "sin azucar")
    return any(p in acl_n for p in pistas)


def _extra_fruta_asumido(result: Dict[str, Any]) -> Optional[str]:
    texto_n = _norm(f"{result.get('meal_name') or ''} {result.get('description') or ''}")
    for eid in ("miel_granola", "miel", "yogur", "granola", "leche_condensada", "mani", "chia"):
        entry = EXTRAS_FRUTA_CATALOGO_POR_ID[eid]
        # [REVISION-ESCANER-14NOV] palabra entera, como en los rellenos (subcadena: «mani» en «manicho», «chia» en «chiabatta»)
        if _matches_tokens(texto_n, entry["tokens"]):
            return eid
    return None


def _delta_extra_fruta(entry: Dict[str, Any], base_id: Optional[str]) -> Dict[str, Any]:
    base_delta = (EXTRAS_FRUTA_CATALOGO_POR_ID.get(base_id or "") or {}).get("delta") or {
        "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0
    }
    return {
        k: max(-2000, min(2000, _redondeo_macro(entry["delta"][k] - base_delta[k], k)))
        for k in ("calories", "protein", "carbs", "healthy_fats")
    }


def _personalizar_duda_fruta(
    duda: Dict[str, Any],
    extras_usuario: List[Dict[str, Any]],
    base_id: Optional[str],
) -> Dict[str, Any]:
    ops = [dict(o) for o in (duda.get("opciones") or [])]
    if not ops or not extras_usuario:
        return duda
    textos_n = [_norm(o.get("texto") or "") for o in ops]
    for hab in extras_usuario[:2]:
        eid = hab.get("id")
        entry = EXTRAS_FRUTA_CATALOGO_POR_ID.get(eid or "")
        if not entry:
            continue
        if any(all(tok in t for tok in entry["tokens"]) for t in textos_n):
            continue
        nueva_op = {
            "texto": entry["texto"][:40],
            "supuesta": False,
            "ajuste": _delta_extra_fruta(entry, base_id),
        }
        if len(ops) < 5:
            ops.append(nueva_op)
        else:
            idx_reemp = next((i for i in range(len(ops) - 1, -1, -1) if not ops[i].get("supuesta")), -1)
            if idx_reemp >= 0:
                ops[idx_reemp] = nueva_op
        textos_n = [_norm(o.get("texto") or "") for o in ops]
    out = dict(duda)
    out["opciones"] = ops[:5]
    return out


def _crear_duda_fruta_bowl(
    info_fruta: Dict[str, str],
    extras_usuario: List[Dict[str, Any]],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    base_id = _extra_fruta_asumido(result)
    base_delta = (EXTRAS_FRUTA_CATALOGO_POR_ID.get(base_id or "") or {}).get("delta") or {
        "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0
    }
    solo_fruta_ajuste = {
        k: _redondeo_macro(-base_delta[k], k) for k in ("calories", "protein", "carbs", "healthy_fats")
    } if base_id else {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}

    candidatos_ids: List[str] = []
    if base_id:
        candidatos_ids.append(base_id)
    for hab in extras_usuario:
        eid = hab.get("id")
        if eid and eid in EXTRAS_FRUTA_CATALOGO_POR_ID and eid not in candidatos_ids:
            candidatos_ids.append(eid)
    for default_id in ("miel", "yogur", "granola", "miel_granola"):
        if default_id not in candidatos_ids:
            candidatos_ids.append(default_id)

    opciones: List[Dict[str, Any]] = [{
        "texto": "Solo fruta",
        "supuesta": (base_id is None),
        "ajuste": solo_fruta_ajuste,
    }]
    for eid in candidatos_ids:
        entry = EXTRAS_FRUTA_CATALOGO_POR_ID[eid]
        es_sup = (eid == base_id)
        opciones.append({
            "texto": entry["texto"][:40],
            "supuesta": es_sup,
            "ajuste": (
                {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}
                if es_sup
                else _delta_extra_fruta(entry, base_id)
            ),
        })
        if len(opciones) >= 5:
            break

    return {
        "sobre": info_fruta["sobre"],
        "pregunta": "¿Las frutas llevan miel o algún acompañamiento encima?",
        "opciones": opciones,
    }


def _detectar_ensalada_vegetales(result: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Detecta ensaladas o vegetales frescos servidos donde el aceite de oliva o vinagreta es invisible.

    No confunde repollo, tomate o verduras cocidas dentro de una sopa/caldo/sancocho/guiso con una ensalada."""
    nombre_desc = _norm(f"{result.get('meal_name') or ''} {result.get('description') or ''}")
    if _PATRON_FRUTAS_MULTI.search(nombre_desc) and not re.search(r"\b(lechuga|tomate|pepino|remolacha|repollo)\b", nombre_desc):
        return None
    items = result.get("items") or []
    items_txt = " ".join(_norm((it or {}).get("name") or "") for it in items if isinstance(it, dict))
    todo = f"{nombre_desc} {items_txt}"
    # Si el plato es una sopa, caldo, sancocho o guiso sin mención explícita de ensalada/hojas frescas,
    # los vegetales como repollo, tomate o zanahoria van cocidos en el caldo (no llevan aderezo).
    tiene_ensalada_explicita = bool(_PATRON_ENSALADA_EXPLICITA.search(todo))
    if _PATRON_CALDO_GUISO.search(nombre_desc) and not tiene_ensalada_explicita:
        return None
    m = _PATRON_ENSALADA_VEGETALES.search(todo)
    if not m:
        return None
    sobre = next(
        (
            it.get("name")
            for it in items
            if isinstance(it, dict)
            and _PATRON_ENSALADA_EXPLICITA.search(_norm(it.get("name")))
        ),
        None,
    ) or next(
        (
            it.get("name")
            for it in items
            if isinstance(it, dict)
            and _PATRON_ENSALADA_VEGETALES.search(_norm(it.get("name")))
            and not _PATRON_CALDO_GUISO.search(_norm(it.get("name")))
        ),
        None,
    )
    return {"sobre": str(sobre or m.group(1))[:60]}


def _duda_es_sobre_aderezo(duda: Dict[str, Any]) -> bool:
    txt = _norm(f"{duda.get('sobre') or ''} {duda.get('pregunta') or ''}")
    return bool(re.search(r"\b(aceite|vinagreta|aderezo|mayonesa|salsa|limon)\b", txt)) and (
        bool(_PATRON_ENSALADA_VEGETALES.search(txt)) or "aderezo" in txt or "vinagreta" in txt
    )


def _aclaracion_menciona_aderezo(aclaracion: Optional[str]) -> bool:
    acl_n = _norm(aclaracion or "")
    if not acl_n:
        return False
    pistas = ("sin aceite", "sin aderezo", "solo limon", "con limon", "aceite de oliva", "vinagreta", "mayonesa", "sin salsa")
    return any(p in acl_n for p in pistas)


def _crear_duda_ensalada(
    info_ensalada: Dict[str, str],
    aderezos_usuario: List[Dict[str, Any]],
) -> Dict[str, Any]:
    candidatos_ids: List[str] = []
    for hab in aderezos_usuario:
        aid = hab.get("id")
        if aid and aid in ADEREZOS_ENSALADA_POR_ID and aid not in candidatos_ids:
            candidatos_ids.append(aid)
    for default_id in ("aceite_oliva", "vinagreta", "mayonesa"):
        if default_id not in candidatos_ids:
            candidatos_ids.append(default_id)

    opciones: List[Dict[str, Any]] = [{
        "texto": "Sin aceite / solo limón",
        "supuesta": True,
        "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0},
    }]
    for aid in candidatos_ids:
        entry = ADEREZOS_ENSALADA_POR_ID[aid]
        opciones.append({
            "texto": entry["texto"][:40],
            "supuesta": False,
            "ajuste": dict(entry["delta"]),
        })
        if len(opciones) >= 5:
            break

    return {
        "sobre": info_ensalada["sobre"],
        "pregunta": "¿La ensalada o vegetales llevan aceite o aderezo?",
        "opciones": opciones,
    }


def _prioridad_duda(duda: Dict[str, Any], info_cerrado: Optional[Dict[str, Any]]) -> int:
    preg_n = _norm(duda.get("pregunta") or "")
    if info_cerrado and _duda_es_sobre_relleno(duda, info_cerrado):
        return 0
    if _duda_es_sobre_fruta_extras(duda):
        return 0
    if _PATRON_DUDA_RELLENO.search(preg_n):
        return 0
    if re.search(r"\b(que\s+tipo|que\s+carne|que\s+bebida|de\s+que\s+es|verde\s+o\s+maduro|masa)\b", preg_n):
        return 1
    if re.search(r"\b(cuant\w*|huevo|claras|mitades|porcion|taza|piezas|lonjas)\b", preg_n):
        return 2
    if re.search(r"\b(grasa|aceite|manteca|mantequilla|freiste|cocin|aderezo|vinagreta)\b", preg_n):
        return 3
    return 4


def refinar_dudas_del_plato(
    result: Dict[str, Any],
    dudas_crudas: List[Dict[str, Any]],
    aclaracion: Optional[str] = None,
    habitos: Optional[Any] = None,
    max_dudas: int = 2,
) -> List[Dict[str, Any]]:
    """Refina las dudas de un plato sólido: elimina preguntas tontas, obliga duda de relleno en
    alimentos cerrados y aditivos invisibles en frutas/ensaladas, e inyecta hábitos del usuario."""
    if result.get("photo_kind") != "plato":
        return list(dudas_crudas or [])

    from bebidas_de_la_foto import es_bebida
    if es_bebida(result.get("meal_name", ""), result.get("description", ""), result.get("items")):
        return list(dudas_crudas or [])[:2]

    rellenos_usuario: List[Dict[str, Any]] = []
    extras_fruta_usuario: List[Dict[str, Any]] = []
    aderezos_usuario: List[Dict[str, Any]] = []
    if isinstance(habitos, dict):
        rellenos_usuario = list(habitos.get("rellenos_pan") or [])
        extras_fruta_usuario = list(habitos.get("extras_fruta") or [])
        aderezos_usuario = list(habitos.get("aderezos_ensalada") or [])

    info_cerrado = _detectar_alimento_cerrado(result)
    info_pan = _detectar_pan_suelto_untable(result, rellenos_usuario) if not info_cerrado else None
    info_fruta = _detectar_fruta_bowl(result)
    info_ensalada = _detectar_ensalada_vegetales(result)
    dudas = [dict(d) for d in (dudas_crudas or []) if isinstance(d, dict) and str(d.get("pregunta") or "").strip()]

    # ¿Hay otra duda real o elemento que merezca atención?
    hay_otra_real = bool(info_cerrado or info_pan or info_fruta or info_ensalada) or any(
        not _es_duda_tonta(d, hay_otra_real=False) for d in dudas
    )
    dudas = [d for d in dudas if not _es_duda_tonta(d, hay_otra_real=hay_otra_real)]

    # 1. Alimento cerrado o pan/tostadas con posible untable oculto: relleno + hábitos
    info_relleno = info_cerrado or info_pan
    if info_relleno and not _aclaracion_menciona_relleno(aclaracion):
        idx_rel = next((i for i, d in enumerate(dudas) if _duda_es_sobre_relleno(d, info_relleno)), None)
        if idx_rel is not None:
            porc = max(1, (info_relleno["cantidad"] + 1) // 2) if info_relleno.get("es_pan_suelto") else info_relleno["cantidad"]
            base_rel = _relleno_asumido_en_pan(result) if info_relleno.get("es_pan_suelto") else _relleno_asumido_en_plato(result)
            dudas[idx_rel] = _personalizar_opciones_duda_relleno(
                dudas[idx_rel], rellenos_usuario, porc, base_rel
            )
        else:
            duda_relleno = _crear_duda_relleno_cerrado(result, info_relleno, rellenos_usuario)
            dudas.insert(0, duda_relleno)

    # 2. Frutas picadas / bowl húmedo: miel, yogur, granola + hábitos
    if info_fruta and not _aclaracion_menciona_fruta_extras(aclaracion):
        idx_fr = next((i for i, d in enumerate(dudas) if _duda_es_sobre_fruta_extras(d)), None)
        if idx_fr is not None:
            dudas[idx_fr] = _personalizar_duda_fruta(
                dudas[idx_fr], extras_fruta_usuario, _extra_fruta_asumido(result)
            )
        else:
            dudas.insert(0, _crear_duda_fruta_bowl(info_fruta, extras_fruta_usuario, result))

    # 3. Ensalada o vegetales frescos: aceite de oliva / vinagreta / aderezo (no desplaza dudas de > impacto)
    if info_ensalada and not _aclaracion_menciona_aderezo(aclaracion):
        if not any(_duda_es_sobre_aderezo(d) for d in dudas):
            dudas.append(_crear_duda_ensalada(info_ensalada, aderezos_usuario))

    # Deduplicar preguntas iguales y ordenar por impacto calórico
    vistas = set()
    limpias: List[Dict[str, Any]] = []
    for d in sorted(dudas, key=lambda item: _prioridad_duda(item, info_relleno)):
        key = _norm(d.get("pregunta") or "")
        if not key or key in vistas:
            continue
        vistas.add(key)
        limpias.append(d)
        if len(limpias) >= max(2, min(6, max_dudas)):
            break

    return limpias
