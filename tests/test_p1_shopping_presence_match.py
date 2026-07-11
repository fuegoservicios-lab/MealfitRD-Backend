"""[P1-SHOPPING-PRESENCE-MATCH · 2026-07-11] La presencia en Nevera también matchea
por contención (nombres parciales), no solo igualdad exacta.

Caso vivo (owner, PDF post-fix del forward-looking): tenía "Yogurt griego entero"
(1 pote, Yoka) en la Nevera y la lista seguía exigiendo "Yogurt · 1 pote"; ídem
"Plátano · 3 ud." con 3 "Plátano maduro" en casa. La igualdad exacta de claves
normalizadas no cruza el nombre canónico corto de la lista con el nombre largo del
inventario. Es el gemelo del bug P1-SHOPPING-NEEDS-MATCH (mismo patrón, otra cuchilla
del filtro).

Contrato:
1. `_lookupInventory(k1, k2)`: exact-match como fast-path + contención con límites
   de palabra en ambas direcciones (" yogurt " ⊂ " yogurt griego entero ").
2. Guards anti-falso-positivo: key consultada ≥4 chars y entrada padded ≥6 chars
   (el padding con espacios ya impide "sal" ⊂ "salsa").
3. El modelo de presencia (P5, spec del owner) queda intacto: presente en CUALQUIER
   cantidad → oculto; este fix solo corrige CÓMO se detecta la presencia.

tooltip-anchor: P1-SHOPPING-PRESENCE-MATCH
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_DASH_SRC = (_BACKEND.parent / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")


def test_lookup_helper_exists_with_containment():
    assert "P1-SHOPPING-PRESENCE-MATCH" in _DASH_SRC
    assert "const _invPadded = [...inventoryMap.entries()].map(([k, v]) => [` ${k} `, v]);" in _DASH_SRC, (
        "índice padded del inventario para contención con límites de palabra"
    )
    assert "pkey.includes(pk) || pk.includes(pkey)" in _DASH_SRC, (
        "contención en AMBAS direcciones — 'yogurt' (lista) ⊂ 'yogurt griego entero' "
        "(nevera) y viceversa"
    )


def test_presence_filter_uses_the_helper():
    assert "const invItem = _lookupInventory(nameKey1, nameKey2);" in _DASH_SRC, (
        "el filtro de presencia debe usar el lookup con contención — el exact-match "
        "dejaba 'Yogurt'/'Plátano' en la lista con el item YA en la Nevera (caso vivo)"
    )
    assert "const direct = inventoryMap.get(k1) || inventoryMap.get(k2);" in _DASH_SRC, (
        "exact-match como fast-path dentro del helper"
    )


def test_false_positive_guards():
    assert "key.length < 4" in _DASH_SRC, "key consultada mínima de 4 chars"
    assert "pkey.length >= 6" in _DASH_SRC, (
        "entrada padded mínima (4 chars + 2 espacios) — nombres ultracortos del "
        "inventario no deben tragarse ítems de la lista"
    )
