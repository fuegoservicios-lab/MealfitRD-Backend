# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-585 · 2026-09-27] La harina de maíz es un producto, no la «harina» del maíz.

Batería real (celíaco): «Wrap criollo de maíz…» y «Tortilla criolla de espinacas…» mandaban medir 15 g y 35 g de «harina
de maíz precocida» y la lista compraba «maíz dulce en granos» — con granos no se hace una tortilla. El reparador de
fantasmas (P1-PHANTOM-INGREDIENT: el paso declara con cantidad un alimento que la lista no trae) pelaba «harina de» como
si fuera una parte («pulpa de», «trozos de») y resolvía el NÚCLEO: «maíz» → Maíz dulce en granos. Corpus: 15 de 4.868
comidas, en ocho perfiles distintos. Una «harina de X» es un producto DISTINTO de X (P1-PREP-COLLAPSE-GUARD): antes de
pelar, la frase entera; si no está en el catálogo, la regla de las preparaciones decide (harina de maíz → Harina de maíz
precocida; harina de plátano → producto sin fila: no se compra plátano). tooltip-anchor: P1-PLAN-LOTE-585
"""
from __future__ import annotations

import unicodedata


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def resolver(words, idx):
    """`(clave, canónico)` si la frase «harina de …» es un producto del catálogo; `()` si es una preparación distinta sin
    fila (no se inserta su base); `None` si no aplica (sigue el pelado de siempre)."""
    try:
        if not words or len(words) < 3 or words[0] not in ("harina", "harinas") or words[1] != "de":
            return None
        frase = " ".join(words)
        hit = (idx or {}).get(frase)
        if hit:
            return frase, hit
        from shopping_calculator import resolve_preparation_distinct
        manejada, canon = resolve_preparation_distinct(frase)
        if not manejada:
            return None
        return (_sa(canon), canon) if canon else ()
    except Exception:
        return None


__all__ = ["resolver"]
