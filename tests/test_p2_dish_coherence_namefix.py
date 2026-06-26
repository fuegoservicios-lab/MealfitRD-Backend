"""[P2-DISH-COHERENCE-NAMEFIX · 2026-06-25] Pulido del reflejo de proteína en el nombre del plato.

Observado en el plan d21fe7de (renovación real): `_reflect_added_protein_in_name` producía nombres
rotos cuando el closer añadía una proteína multi-palabra:
  - "Yuca Rellena ... y Res Molida **y Carne De**"  (truncó 'carne de res' a 2 palabras + capitalizó 'de')
  - "Pimientos Rellenos de Res ... **con Carne De**" (duplicó: 'res' ya estaba en el nombre)

Fix: (a) chequea CUALQUIER token significativo (≥3 chars, sin stopwords) — si ya está en el nombre,
no duplica; (b) muestra la proteína COMPLETA con conectores ('de'/'la') en minúscula.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_orchestrator import _reflect_added_protein_in_name


def _sa(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def test_no_duplica_si_token_significativo_ya_esta():
    """'carne de res' NO se añade si 'res' ya está en el nombre (era el bug 'Res Molida y Carne De')."""
    m = {"name": "Yuca Rellena con Queso Blanco y Res Molida"}
    assert _reflect_added_protein_in_name(m, "carne de res", _sa) is False
    assert "Carne De" not in m["name"]
    assert m["name"] == "Yuca Rellena con Queso Blanco y Res Molida"


def test_proteina_multipalabra_se_muestra_completa_y_bien_formateada():
    """'carne de res' → 'Carne de Res' (completa, 'de' en minúscula), no el truncado 'Carne De'."""
    m = {"name": "Pimientos Rellenos de Arroz"}  # sin 'res'/'carne' en el nombre
    changed = _reflect_added_protein_in_name(m, "carne de res", _sa)
    assert changed is True
    assert "Carne de Res" in m["name"]
    assert "Carne De" not in m["name"]  # no truncado, no 'De' capitalizado


def test_yogur_griego_sigue_funcionando():
    m = {"name": "Batido de Frutas"}
    assert _reflect_added_protein_in_name(m, "yogur griego", _sa) is True
    assert m["name"] == "Batido de Frutas con Yogur Griego"


def test_queso_ya_presente_no_duplica():
    """'queso mozzarella' no se añade si 'queso' ya está (token significativo compartido)."""
    m = {"name": "Queso Blanco a la Parrilla con Ñame"}
    assert _reflect_added_protein_in_name(m, "queso mozzarella", _sa) is False


def test_conector_y_cuando_ya_hay_con():
    m = {"name": "Ensalada Fresca con Vegetales"}
    _reflect_added_protein_in_name(m, "pollo", _sa)
    assert m["name"] == "Ensalada Fresca con Vegetales y Pollo"
