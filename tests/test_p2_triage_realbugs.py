"""[P2-TRIAGE-REALBUGS · 2026-06-16] Regresión de los 3 bugs reales (preexistentes)
que el triage de los ~269 tests rojos del full-run sacó a la superficie.

Los 3 eran genuinos (no test-drift): un bug de orden en `canonicalize_huevo`, un
drift de acentos entre el fallback y el canónico de proteína, y una violación de la
convención G18 en el tick de pipeline_metrics del cron de Dreaming. Ninguno venía del
audit clínico-de-precisión de la sesión; los arreglamos como cierre del triage.

Anchors de tooltip: P2-TRIAGE-REALBUGS.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


# ───────────── Bug 1: canonicalize_huevo — orden (plato compuesto antes que claras) ─────────────
def test_canonicalize_huevo_excludes_compound_dishes_first():
    """'Omelette de claras' es un PLATO → None, aunque contenga 'claras'. Antes
    el branch claras corría antes de la exclusión y devolvía 'Huevo'."""
    from shopping_calculator import canonicalize_huevo

    # Platos compuestos → None (la exclusión corre primero).
    assert canonicalize_huevo("Omelette de claras") is None
    assert canonicalize_huevo("Tortilla de huevo") is None
    assert canonicalize_huevo("Huevos endiablados") is None
    assert canonicalize_huevo("Omelette de yemas") is None
    # Ingredientes shopping reales → 'Huevo' (sin regresión).
    assert canonicalize_huevo("Claras de huevo") == "Huevo"
    assert canonicalize_huevo("Yema de huevo") == "Huevo"
    assert canonicalize_huevo("Huevos") == "Huevo"
    assert canonicalize_huevo("Brócoli") is None


# ───────────── Bug 2: drift de acentos fallback↔canónico de proteína ─────────────
def test_protein_canonical_has_unaccented_fish_variants():
    """El canónico debe contener las formas SIN tilde ('salmon'/'atun') que usa el
    fallback de nutrition_calculator, para que el fallback siga siendo subconjunto."""
    pytest.importorskip("langchain_google_genai", reason="constants.py requiere langchain")
    from constants import PROTEIN_SYNONYMS

    pescado = set(a.lower() for a in PROTEIN_SYNONYMS["pescado"])
    assert {"salmon", "atun"}.issubset(pescado), (
        "El canónico de 'pescado' perdió las variantes sin tilde — el fallback "
        "_FALLBACK_PROTEIN_SYNONYMS dejaría de ser subconjunto (drift de acentos)."
    )


# ───────────── Bug 3: G18 — cost NO va en el slot tokens_estimated del dreaming tick ─────────────
def test_dreaming_tick_respects_g18_tokens_estimated_slot():
    """El INSERT a pipeline_metrics del cron de Dreaming pone 0 en tokens_estimated
    (G18); el cost vive en metadata.cost_usd (agg)."""
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    # El nuevo INSERT usa la fila all-zeros para los slots numéricos.
    assert "VALUES (NULL, NULL, %s, 0, 0, 0, 0, %s::jsonb)" in src
    # El anti-patrón viejo (cost*1e6 en el slot) no debe quedar en el archivo.
    assert "cost_usd\", 0.0) or 0.0) * 1e6" not in src
    assert "P2-TRIAGE-REALBUGS" in src
