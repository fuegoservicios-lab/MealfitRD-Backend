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

from habitos_de_la_foto import RELLENOS_CATALOGO_POR_ID

_PATRON_CERRADO = re.compile(
    r"\b(sandwich(?:es)?|emparedados?|wraps?|burritos?|arepas?(?:\s+rellenas?)?|empanadas?|croissants?)\b",
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

_PATRON_BEBIDA_DUDA = re.compile(
    r"\b(cafe|te|jugo|batid[ao]|licuado|infusion|bebida|chocolate)\b",
    re.IGNORECASE,
)

_PATRON_OPCION_NEGATIVA = re.compile(
    r"^\s*(?:sin\b|nada\b|ningun[ao]?\b|natural\b|solo\b(?!\s+(?:jamon|queso|claras|huevo|carne|pollo)))",
    re.IGNORECASE,
)


def _norm(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9\s]+", " ", t).split())


def _redondeo_macro(v: float, k: str):
    return int(round(v)) if k == "calories" else round(v, 1)


def _es_duda_tonta(duda: Dict[str, Any], hay_otra_real: bool = False) -> bool:
    """Detecta preguntas de bajo valor o redundantes (ej. sirope en pancakes visualmente secos)."""
    preg_n = _norm(duda.get("pregunta") or "")
    sobre_n = _norm(duda.get("sobre") or "")
    ops = duda.get("opciones") or []
    if not preg_n:
        return True

    # En bebidas (café, té, jugo) lo disuelto (ej. miel) no es visible desde fuera
    if _PATRON_BEBIDA_DUDA.search(f"{preg_n} {sobre_n}"):
        return False

    # Preguntar "¿Con o sin sirope/ketchup/salsa?" cuando la propia IA ya marcó "Sin ..." como supuesto
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
    if _PATRON_CERRADO.search(sobre_n) or (tipo_n and tipo_n in sobre_n) or "pan" in sobre_n.split():
        return True
    if _PATRON_CERRADO.search(preg_n) and _PATRON_DUDA_RELLENO.search(preg_n):
        return True
    if ("pan" in preg_n.split() or "panes" in preg_n.split()) and _PATRON_DUDA_RELLENO.search(preg_n):
        return True
    return False


def _aclaracion_menciona_relleno(aclaracion: Optional[str]) -> bool:
    acl_n = _norm(aclaracion or "")
    if not acl_n:
        return False
    pistas = ("cottage", "mantequilla de mani", "jamon", "queso", "pavo", "atun", "pollo", "huevo", "aguacate", "sin nada", "pan solo")
    return any(p in acl_n for p in pistas) and (
        bool(_PATRON_CERRADO.search(acl_n))
        or "pan" in acl_n.split()
        or "panes" in acl_n.split()
        or "rellen" in acl_n
        or "dentro" in acl_n
        or "lleva" in acl_n
    )


def _relleno_asumido_en_plato(result: Dict[str, Any]) -> Optional[str]:
    texto_n = _norm(f"{result.get('meal_name') or ''} {result.get('description') or ''}")
    for rid in ("cottage_mani", "mani_queso", "jamon_queso", "pavo_queso", "huevo_queso", "cottage", "mani", "queso_crema", "aguacate", "atun", "pollo", "jamon", "queso"):
        entry = RELLENOS_CATALOGO_POR_ID[rid]
        if all(tok in texto_n for tok in entry["tokens"]):
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
    base_id = _relleno_asumido_en_plato(result)
    base_entry = RELLENOS_CATALOGO_POR_ID.get(base_id or "")

    plural = qty > 1
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
    for default_id in ("jamon_queso", "cottage_mani", "mani", "huevo_queso", "atun", "queso"):
        if default_id not in candidatos_ids:
            candidatos_ids.append(default_id)
        if len(candidatos_ids) >= (5 if base_id else 4):
            break

    opciones: List[Dict[str, Any]] = []
    if not base_id:
        opciones.append({
            "texto": "Solo / sin relleno extra",
            "supuesta": True,
            "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0},
        })

    for cid in candidatos_ids[:5]:
        entry = RELLENOS_CATALOGO_POR_ID[cid]
        es_sup = (cid == base_id)
        ajuste = (
            {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}
            if es_sup
            else _delta_para_opcion(entry, base_id, qty)
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


def _prioridad_duda(duda: Dict[str, Any], info_cerrado: Optional[Dict[str, Any]]) -> int:
    preg_n = _norm(duda.get("pregunta") or "")
    if info_cerrado and _duda_es_sobre_relleno(duda, info_cerrado):
        return 0
    if _PATRON_DUDA_RELLENO.search(preg_n):
        return 0
    if re.search(r"\b(que\s+tipo|que\s+carne|que\s+bebida|de\s+que\s+es|verde\s+o\s+maduro|masa)\b", preg_n):
        return 1
    if re.search(r"\b(cuant\w*|huevo|claras|mitades|porcion|taza|piezas|lonjas)\b", preg_n):
        return 2
    if re.search(r"\b(grasa|aceite|manteca|mantequilla|freiste|cocin)\b", preg_n):
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
    alimentos cerrados e inyecta los hábitos del usuario dentro de las opciones."""
    if result.get("photo_kind") != "plato":
        return list(dudas_crudas or [])

    from bebidas_de_la_foto import es_bebida
    if es_bebida(result.get("meal_name", ""), result.get("description", ""), result.get("items")):
        return list(dudas_crudas or [])[:2]

    rellenos_usuario: List[Dict[str, Any]] = []
    if isinstance(habitos, dict):
        rellenos_usuario = list(habitos.get("rellenos_pan") or [])

    info_cerrado = _detectar_alimento_cerrado(result)
    dudas = [dict(d) for d in (dudas_crudas or []) if isinstance(d, dict) and str(d.get("pregunta") or "").strip()]

    # ¿Hay otra duda real o alimento cerrado que merezca atención?
    hay_otra_real = bool(info_cerrado) or any(not _es_duda_tonta(d, hay_otra_real=False) for d in dudas)
    dudas = [d for d in dudas if not _es_duda_tonta(d, hay_otra_real=hay_otra_real)]

    # Si hay sándwich/wrap/arepa/empanada cerrado y no lo aclararon ya: asegurar pregunta + hábitos
    if info_cerrado and not _aclaracion_menciona_relleno(aclaracion):
        idx_rel = next((i for i, d in enumerate(dudas) if _duda_es_sobre_relleno(d, info_cerrado)), None)
        if idx_rel is not None:
            dudas[idx_rel] = _personalizar_opciones_duda_relleno(
                dudas[idx_rel], rellenos_usuario, info_cerrado["cantidad"], _relleno_asumido_en_plato(result)
            )
        else:
            duda_relleno = _crear_duda_relleno_cerrado(result, info_cerrado, rellenos_usuario)
            dudas.insert(0, duda_relleno)

    # Deduplicar preguntas iguales y ordenar por impacto calórico
    vistas = set()
    limpias: List[Dict[str, Any]] = []
    for d in sorted(dudas, key=lambda item: _prioridad_duda(item, info_cerrado)):
        key = _norm(d.get("pregunta") or "")
        if not key or key in vistas:
            continue
        vistas.add(key)
        limpias.append(d)
        if len(limpias) >= max(2, min(6, max_dudas)):
            break

    return limpias
