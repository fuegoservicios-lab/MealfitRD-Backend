"""[P1-PANTRY-KEY-VULGAR-FRACTIONS · 2026-09-03] La clave canónica de la Nevera entiende fracciones unicode.

Las recetas escriben «⅓ taza de yogurt», «¾ cucharada de mantequilla de maní», «1½ tazas». La clase
de cantidad de `_QUANTITY_PATTERN` sólo conocía dígitos y «/», así que `pantry_names_match`
(SSOT de identidad de fila — P1-PANTRY-NAME-RESOLUTION) decía que «¾ cucharada de mantequilla
de maní» NO era mantequilla de maní. Medido en el primer plan del canary F3 (03-sep 09:33): ancla
marcada AUSENTE en un día que la tenía dos veces. Los casos vivos están en
`test_p1_pantry_name_resolution.py` (_SHOULD_MATCH) y `test_p1_arq25_f3_horizon.py`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent


def test_quantity_class_includes_unicode_fractions():
    src = (BACKEND / "constants.py").read_text(encoding="utf-8")
    assert "r'^[\\d\\s/.,¼½¾⅓⅔⅛⅜⅝⅞]+'" in src


@pytest.mark.parametrize("raw,expected", [
    ("⅓ taza de yogurt", "yogurt"),
    ("¾ cucharada de mantequilla de maní", "mantequilla de mani"),
    ("1½ tazas de avena", "avena"),
    ("½ kiwi en cubos", "kiwi en cubos"),
    ("2 huevos", "huevos"),
])
def test_canonical_key_strips_unicode_fractions(raw, expected):
    from constants import canonical_pantry_key
    assert canonical_pantry_key(raw) == expected


def test_marker_present():
    assert "P1-PANTRY-KEY-VULGAR-FRACTIONS" in (BACKEND / "constants.py").read_text(encoding="utf-8")
