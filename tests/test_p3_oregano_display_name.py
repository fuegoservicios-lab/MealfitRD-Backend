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
    src = _SC.read_text(encoding="utf-8")
    idx = src.index("P3-OREGANO-DISPLAY-NAME")
    block = src[idx:idx + 700]  # cubre el comentario + las líneas del bloque de consolidación
    assert "canonical_name = 'Orégano'" in block, (
        "el target de consolidación de orégano debe ser el display 'Orégano'"
    )
    assert "canonical_name = 'Orégano dominicano'" not in block, (
        "NO revertir a 'Orégano dominicano' — huerfaniza el lookup de precio en master_map"
    )
