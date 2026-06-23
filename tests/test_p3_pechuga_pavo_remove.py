"""[P3-PECHUGA-PAVO-REMOVE · 2026-06-23] El owner confirmó que La Sirena NO vende pechuga de
pavo FRESCA → eliminada del catálogo y del pool de proteínas. Revierte P3-PAVO-SKELETON-VERIFIED-
ALIGN/P3-CONDIMENT-CONSOLIDATION (que la habían hecho el pavo default).

- "Pechuga de pavo" NO está en DOMINICAN_PROTEINS (ni "Pavo" genérico).
- El day-gen prompt ya NO sugiere "pechuga de pavo FRESCA".
- "Jamón de pavo" (embutido procesado, vendido/verificado) SE CONSERVA: gateado por
  _RESTRICTED_PROTEIN_KEYS (máx 50g si el pool lo asigna) y normalize_name lo distingue.
- canonicalize_pavo NO se tocó (defensivo/inerte; la pechuga ya no aparece, y el espejo
  VERIFIED-ONLY dropea cualquier mención suelta de ambos lados → cero divergencia).
"""
from __future__ import annotations

from pathlib import Path

import constants as C

_BACKEND = Path(__file__).resolve().parent.parent


def test_pool_no_longer_offers_turkey():
    proteins = C.DOMINICAN_PROTEINS
    assert "Pechuga de pavo" not in proteins, "pechuga de pavo eliminada (no se vende en La Sirena)"
    assert "Pavo" not in proteins, "'Pavo' genérico tampoco se ofrece"


def test_daygen_prompt_no_longer_suggests_fresh_pechuga():
    src = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")
    assert "usa pechuga de pavo FRESCA" not in src, "el prompt aún sugiere pechuga de pavo fresca"
    assert "pechuga fresca / pavo guisado" not in src, "el label de pavo aún menciona pechuga fresca"


def test_jamon_de_pavo_still_distinguished_as_processed():
    """No regresión: el jamón de pavo (embutido procesado, vendido) sigue distinguido."""
    from shopping_calculator import normalize_name
    assert normalize_name("jamón de pavo en lonjas") == "Jamón de pavo"


def test_marker_present():
    src = (_BACKEND / "constants.py").read_text(encoding="utf-8")
    assert "P3-PECHUGA-PAVO-REMOVE" in src
