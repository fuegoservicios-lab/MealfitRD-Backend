# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-245 · 2026-09-25] Con «Nada» de tiempo, los cerradores de proteína eligen proteína LISTA.

Auditoría del 25-sep (escritores deterministas): los cerradores del piso de proteína no leían el tiempo de cocina y, con
«Nada» (≤10 min), añadían pechuga cruda con un paso «Cocina … a la plancha» o legumbre seca «hasta que ablande». El
prompt del generador ya dice qué cabe («proteínas listas o de cocción rápida como huevo, atún o sardina en lata, queso,
embutido magro, pollo ya cocido»); aquí se aplica lo mismo a los candidatos del cerrador con la metadata del catálogo
(`master_ingredients.ready_to_eat`) más el huevo (cocción rápida). Sin ningún candidato listo se conserva la lista
completa: un déficit de proteína lleva al plan de emergencia, que es peor que 10 minutos de más.
tooltip-anchor: P1-PLAN-LOTE-245-PROTEINA-LISTA
"""
from __future__ import annotations

import unicodedata


def _sa(s) -> str:
    return unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower().strip()


def _listos() -> set:
    try:
        from shopping_calculator import get_master_ingredients
        return {_sa(r.get("name")) for r in (get_master_ingredients() or [])
                if isinstance(r, dict) and r.get("ready_to_eat") is True and r.get("name")}
    except Exception:
        return set()


def filtrar_listas(cands, form_data, listos=None) -> list:
    """`cands` = [(densidad, nombre, info), …] de `_safe_high_density_proteins`. Con «Nada», solo los listos + huevo."""
    cands = list(cands or [])
    if str((form_data or {}).get("cookingTime") or "").strip().lower() != "none" or not cands:
        return cands
    listos = listos if listos is not None else _listos()
    out = []
    for c in cands:
        try:
            nombres = {_sa(c[1]), _sa(getattr(c[2], "name", "") or "")}
        except Exception:
            continue
        if nombres & listos or any("huevo" in n for n in nombres):
            out.append(c)
    return out or cands
