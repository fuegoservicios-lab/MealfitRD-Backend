"""[P3-SHOPPING-PROJECTION-PKG · 2026-09-05] Paquete `shopping/` (roadmap 2.5, §11 god files).

Primera extracción real fuera de `shopping_calculator.py`: `shopping.projection` (read model de las listas
7/15/30 por revisión, reproyección con huella y estado UI). La lista SÍNCRONA (agregación, packaging,
precios, shelf-life, presentación) sigue en `shopping_calculator.py` hasta la Fase 9 — moverla no cambia
ningún comportamiento y sí arriesga los ~200 tests parser-based que la anclan.
"""
