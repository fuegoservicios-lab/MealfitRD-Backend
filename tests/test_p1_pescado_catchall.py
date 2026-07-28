"""[P1-PESCADO-CATCHALL · 2026-07-28] Dislike "pescado" ahora excluye las especies.

## El hueco

Los catch-alls de `_get_fast_filtered_catalogs` (constants.py) expandían "mariscos"/
"carne"/"lácteos"/"frutos secos"/"huevo"/"gluten"/"soya" — pero NO "pescado" a secas,
que es el término que un usuario dominicano realmente escribe. `\\bpescado\\b` no
matchea 'Mero'/'Tilapia'/'Salmón' por nombre, así que un dislike "pescado" dejaba
mero, tilapia, salmón, bacalao, sardinas y arenque VIVOS en los pools de variedad y
el LLM podía servirlos (violación de preferencia; misma clase que el caso vegano de
P1-VARIETY-CATALOG-POOLS, un nivel abajo — preferencia, no seguridad).

Descubierto reapuntando el edge-recipe test (2026-07-28): al derivar los dislikes del
catálogo vivo quedó a la vista que "pescado" no filtraba especies.

## El contrato

- "pescado"/"pez"/"fish" → fuera TODAS las especies de pez del catálogo.
- Camarones/pulpo/calamar SOBREVIVEN: quien no come pescado puede comer mariscos
  ("mariscos" ya tenía su propio catch-all para el mar entero).
- Proteínas de tierra intactas.

tooltip-anchor: P1-PESCADO-CATCHALL
"""
from __future__ import annotations

import unicodedata

from constants import _get_fast_filtered_catalogs


def _sa(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()


_PECES = {"mero", "tilapia", "salmon", "bacalao", "arenque", "merluza"}


def _peces_en(proteins) -> list:
    return [p for p in proteins if _sa(p) in _PECES or "sardina" in _sa(p) or "atun" in _sa(p) or _sa(p) == "pescado"]


def test_dislike_pescado_excluye_especies():
    proteins, _, _, _ = _get_fast_filtered_catalogs((), ("pescado",), "")
    assert _peces_en(proteins) == [], (
        f"Dislike 'pescado' dejó especies vivas: {_peces_en(proteins)}"
    )


def test_dislike_pescado_conserva_mariscos_y_tierra():
    proteins, _, _, _ = _get_fast_filtered_catalogs((), ("pescado",), "")
    low = [_sa(p) for p in proteins]
    assert any("camaron" in p for p in low), "camarones no son pescado — deben sobrevivir"
    assert any("pollo" in p for p in low), "proteínas de tierra intactas"


def test_variantes_del_termino():
    for term in ("pescados", "pez", "fish", "PESCADO"):
        proteins, _, _, _ = _get_fast_filtered_catalogs((), (term,), "")
        assert _peces_en(proteins) == [], f"variante {term!r} no expandió el catch-all"


def test_vegetariano_sin_peces():
    """La dieta vegetariana añade 'pescado' a las restricciones — ahora la expansión
    de especies la cubre doblemente (antes dependía solo del catch-all de 'marisco')."""
    proteins, _, _, _ = _get_fast_filtered_catalogs((), (), "vegetariano")
    assert _peces_en(proteins) == []


def test_sin_dislike_no_filtra():
    proteins, _, _, _ = _get_fast_filtered_catalogs((), (), "")
    assert _peces_en(proteins), "sin restricciones el catálogo debe traer peces"
