# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-591 · 2026-09-27] Con sólo claras en la lista, el paso no pocha huevos enteros de yema cremosa.

Batería real (colesterol alto + atorvastatina): el tope de yemas dejó «5 claras de huevo» en la lista y el paso seguía
diciendo «pocha los 2 huevos en agua apenas hirviendo… hasta que la clara cuaje y la yema quede cremosa» — dos yemas que el
perfil no debía comer, crudas a medias, junto a la nota «cocina el huevo por completo… la clara firme» y la del
nutricionista «esta receta usa solo las claras». Corpus: 3 de 111 comidas con sólo claras. Los lotes 405/426 ya alinean el
huevo duro y las sustituciones; el pochado/escalfado no tenía quien lo siguiera. Aquí se cuajan las claras de la lista.
tooltip-anchor: P1-PLAN-LOTE-591
"""
from __future__ import annotations

import re
import unicodedata

_NOTAS = ("⚠", "🤰", "⚕", "💡", "🌱")
_POCHA_RE = re.compile(
    r"\b(?P<v>[Pp]ocha|[Ee]scalfa)\s+(?:(?:los|las|el)\s+)?(?:(?P<n>\d+)\s+)?huevos?\b(?P<medio>[^.;]*?)"
    r",?\s*hasta\s+que\s+la\s+clara\s+cuaje\s+y\s+la\s+yema\s+(?:quede|est[eé]|siga)\s+\w+")
_ADJ_RE = re.compile(r"\b(las|los)\s+(claras|huevos)\s+(pochad|escalfad)(?:os|as)\b", re.IGNORECASE)


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _solo_claras(meal) -> bool:
    lista = " | ".join(_sa(x) for x in (meal.get("ingredients") or []) if isinstance(x, str))
    return bool(re.search(r"\bclaras? de huevos?\b", lista)) and not re.search(
        r"\bhuevos?\b", re.sub(r"(?:claras?|yemas?) de huevos?", " ", lista))


def alinear(meal) -> int:
    """Nº de frases alineadas; 0 ante cualquier error o si la lista trae huevos enteros."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec or not _solo_claras(meal):
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or any(e in p for e in _NOTAS):
                continue
            q = _POCHA_RE.sub(lambda m: ("Cuaja" if m.group("v")[0].isupper() else "cuaja") + " las claras de huevo"
                              + m.group("medio") + ", hasta que estén firmes y opacas", p)
            q = _ADJ_RE.sub(lambda m: f"las claras {m.group(3).lower()}as", q)   # «las claras pochados» → «pochadas»
            if q != p:
                rec[i] = q
                n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


__all__ = ["alinear"]
