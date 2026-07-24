"""[P1-SWEET-MARKER-SUFFIX · 2026-07-24] El marcador de plato dulce no puede matchear
un PREFIJO arbitrario: "mora" (la fruta) matcheaba dentro de "repollo **mora**do".

Hallazgo en vivo (plan a060108b, revisión de recetas del owner 2026-07-24): la receta
"Queso Blanco Glaseado con Batata Asada y Ensalada" lleva "1 taza de Repollo morado
(rallado)" → `_is_sweet_meal` devolvía **True** → el sweet-guard del cerrador filtraba
TODA proteína salada del pool → el piso de proteína de esa comida se saltaba en silencio.
Un plato salado clasificado como postre por una col morada.

Historia del bug (dos vueltas del mismo error):
    - Antes de 2026-06-28: substring naïve → "pina" (piña) matcheaba en "es-PINA-ca".
    - P1-SWEET-MARKER-WORDBOUNDARY puso frontera IZQUIERDA (`\\b<marker>`) → cerró espinaca.
    - Pero la derecha quedó abierta, así que cualquier marcador que sea PREFIJO de una
      palabra salada sigue dando falso positivo. "mora" ⊂ "morado" es el caso vivo.

Por qué no basta con `\\b` a la derecha:
    El propio comentario del fix anterior depende del prefijo para las variantes
    morfológicas: `\\byogur` tiene que seguir matcheando "yogurt", y "fresa" → "fresas".
    Un `\\b` duro las rompería a todas (regresión silenciosa del guard dulce).

Fix: frontera derecha que admite SOLO sufijos flexivos (`s`, `es`, `as`, `os`, `t`) —
"fresas"/"melones"/"yogurt" siguen entrando; "morado" (sufijo `do`) queda fuera.
"""
from __future__ import annotations

import graph_orchestrator as g
from constants import strip_accents as _sa


def _meal(name, ings=()):
    return {"name": name, "ingredients": list(ings)}


# ---------------------------------------------------------------------------
# 1. El falso positivo reportado
# ---------------------------------------------------------------------------
def test_repollo_morado_no_es_plato_dulce():
    """El caso exacto de la receta #4 del plan a060108b."""
    m = _meal("Queso Blanco Glaseado con Batata Asada y Ensalada",
              ["½ queso blanco (109g)", "1 taza de Repollo morado (rallado)", "½ zanahoria (rallada)"])
    assert g._is_sweet_meal(m, _sa) is False


def test_otros_prefijos_salados_no_matchean():
    for nm, ings in (
        ("Ensalada de Repollo Morado", []),
        ("Bowl con Repollo morado rallado", ["½ taza de Repollo morado rallado"]),
        ("Revoltillo con Espinaca", ["espinaca"]),          # el caso viejo (pina ⊂ espinaca)
    ):
        assert g._is_sweet_meal(_meal(nm, ings), _sa) is False, nm


# ---------------------------------------------------------------------------
# 2. Lo que NO se puede romper: variantes morfológicas legítimas
# ---------------------------------------------------------------------------
def test_plurales_y_variantes_siguen_siendo_dulces():
    for nm in (
        "Yogurt Griego con Fresas",        # fresa + s
        "Bowl de Yogurt con Moras",        # mora + s  (la fruta de verdad, en plural)
        "Ensalada de Melones y Uvas",      # melon + es
        "Avena con Manzanas",              # manzana + s
        "Batido de Guineos",               # guineo + s
        "Panqueques con Arandanos",        # arandano + s
    ):
        assert g._is_sweet_meal(_meal(nm), _sa) is True, nm


def test_mora_sola_sigue_siendo_dulce():
    assert g._is_sweet_meal(_meal("Yogurt con Mora"), _sa) is True


def test_marker_presente():
    import pathlib
    src = pathlib.Path(g.__file__).with_suffix(".py").read_text(encoding="utf-8", errors="replace")
    assert "[P1-SWEET-MARKER-SUFFIX · 2026-07-24]" in src
