"""[P1-BROTH-NOT-MEAT · 2026-07-28] La lista compraba CARNE cuando la receta pedía caldo.

## Los dos defectos reales que destapó el clúster p6_lacteos en rojo

1. **Caldo → carne**: `normalize_name("caldo de pollo")` resolvía a **'Pechuga de pollo'** y
   `"caldo de res"` a **'Carne de res'** (subcadena sobre alimento español, la clase de siempre,
   ahora en el resolver maestro): 99 lbs de caldo inflaban la línea de pechuga a 838 kg. Fix en
   `resolve_preparation_distinct`: el caldo es producto DISTINTO — sin fila en catálogo, sigue el
   camino honesto del drop verified-only con WARN, jamás la carne.

2. **Caps con canon drift** ([P1-CAP-CANON-DRIFT]): los sets de nombres EXACTOS de
   `P6-CARBS-CAP` y `P6-CANNED-PROTEIN-CAP` quedaron obsoletos cuando la expansión del catálogo
   movió los canónicos ('pan integral' → 'Pan integral familiar'; 'sardinas' → 'Sardinas en
   lata'): **caps de pan y sardinas muertos en producción** sin que nada fallara. El atún seguía
   capando solo porque 'atun en agua' coincidía con su canon.

tooltip-anchor: P1-BROTH-NOT-MEAT
"""
from __future__ import annotations

import re
import pathlib

import shopping_calculator as sc

_SRC = pathlib.Path(sc.__file__).with_suffix(".py").read_text(encoding="utf-8")


def test_caldo_es_producto_distinto():
    assert sc.resolve_preparation_distinct("caldo de pollo") == (True, None)
    assert sc.resolve_preparation_distinct("caldo de res") == (True, None)
    assert sc.resolve_preparation_distinct("2 tazas de caldo de hueso") == (True, None)


def test_caldo_no_matchea_dentro_de_otras_palabras():
    """'respaldo'/'escaldado' no son caldos."""
    assert sc.resolve_preparation_distinct("pollo escaldado") == (False, None)
    assert sc.resolve_preparation_distinct("respaldo de pollo") == (False, None)


def test_los_sets_de_caps_cubren_los_canones_vivos():
    """Ancla del drift: si un futuro renombre del catálogo vuelve a mover los canónicos, estos
    literales deben actualizarse JUNTO con el set — el cap muere en silencio si divergen."""
    assert "'sardinas en lata'" in _SRC, "el cap de enlatados perdió el canon vivo de sardinas"
    assert "'pan integral familiar'" in _SRC, "el cap de carbos perdió el canon vivo del pan"
    # y los sets siguen siendo sets de nombres exactos (si alguien los vuelve substring-match,
    # revisar que 'pan' no capee 'panqueques')
    assert "_CANNED_PROTEIN_NAMES_FOR_CAP = {" in _SRC
    assert "_CARBS_PACKAGE_NAMES_FOR_CAP = {" in _SRC
