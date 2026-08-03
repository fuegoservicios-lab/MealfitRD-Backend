"""[P3-OREGANO-DISPLAY-NAME · 2026-06-20] El nombre que la lista de compras MUESTRA para
el orégano sale de un LITERAL hardcodeado en el aggregator (shopping_calculator.py, la
consolidación de variantes de orégano), NO de master_ingredients.name. El owner pidió que
diga solo 'Orégano' (sin 'dominicano', redundante en es-DO).

Verificación adversaria (workflow 2026-06-20) confirmó que renombrar SOLO el master.name:
  (a) NO cambia el display — sigue saliendo del literal hardcodeado; y
  (b) HUERFANIZA el lookup de precio/envase: master_map.get('Orégano dominicano') falla
      tras el rename (price 81→null, container sobre→lb, category Despensa→Otros).
Por eso el fix correcto es editar el literal a 'Orégano' (que SÍ resuelve en master_map vía
el alias 'orégano'.title()='Orégano' del catálogo, slug='oregano'). Este test ancla el
literal; un revert a 'Orégano dominicano' lo falla antes de tocar producción.
"""
from __future__ import annotations

from pathlib import Path

_SC = Path(__file__).resolve().parent.parent / "shopping_calculator.py"


def test_oregano_consolidation_display_is_oregano():
    """[review final audit-v7-p1 · 2026-08-03] Dos cambios de anclaje, misma garantía:

    (1) Se ancla en `tooltip-anchor: P3-OREGANO-DISPLAY-NAME` y no en el marker suelto. El marker
        suelto aparece también en cualquier comentario que HABLE del fix (le pasó a este test al
        reponer el bloque perdido: un comentario vecino que lo citaba se llevó el `index` y la
        ventana leyó la región equivocada). El tooltip-anchor es el marcador de SITIO, uno solo.
    (2) La ventana fija de 700 chars se sustituye por el delimitador ESTRUCTURAL de la regla: la
        siguiente sentencia `if re.search(` después de la asignación. Así el test no caduca cuando
        alguien añade una línea de comentario, que es como murió su hermano de
        `test_p1_budget_convergence`.

    El bloque vive hoy en `canonicalize_shopping_food_name` (SSOT extraído del aggregator por
    P1-VEG-BACKFILL-HONESTY); la garantía —el literal MOSTRADO es 'Orégano', no 'Orégano
    dominicano', porque el segundo huerfaniza el lookup de precio en master_map— no cambió."""
    src = _SC.read_text(encoding="utf-8")
    idx = src.index("tooltip-anchor: P3-OREGANO-DISPLAY-NAME")
    asignacion = src.index("canonical_name = ", src.index("if re.search(", idx))
    fin_regla = src.index("if re.search(", asignacion)
    block = src[idx:fin_regla]
    assert "canonical_name = 'Orégano'" in block, (
        "el target de consolidación de orégano debe ser el display 'Orégano'"
    )
    assert "canonical_name = 'Orégano dominicano'" not in block, (
        "NO revertir a 'Orégano dominicano' — huerfaniza el lookup de precio en master_map"
    )
    assert "or[eé]gano" in block, (
        "el tooltip-anchor se separó de la regla que documenta: el test estaría vigilando otro sitio"
    )
