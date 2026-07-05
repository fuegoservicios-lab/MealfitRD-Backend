"""[P1-DAYGEN-FATS-BUDGET · 2026-07-05] Presupuesto cuantitativo de GRASAS per-día en el prompt
del day-generator (§18, espejo del §17 de sodio).

Causa raíz (medida en vivo 2026-07-04/05): el day-gen produce días grasos-extremos (141-166% del
target) que SATURAN el clamp del solver [0.3, 3.5] y el del reconcile [0.4, 1.8] — con el clamp
saturado ningún corrector determinista aguas abajo alcanza la banda [0.90, 1.12] → banner
low_band_macro:fats recurrente (fats 0.333 en las corridas 7f99b955 y previas). El prompt solo
tenía reglas de porción de aceite por staple, sin presupuesto per-día ni reglas de conteo de
grasas ocultas/densas.

El §18 es la palanca en la FUENTE (cero costo — prompt-only, estático a import-time → el
prompt-cache del SystemMessage queda intacto). P1-FATS-POSTCLOSER-RELEVEL es la palanca en el
MOTOR; ambas atacan el mismo banner por los dos lados.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DG_PATH = _REPO_ROOT / "backend" / "prompts" / "day_generator.py"
_DG = _DG_PATH.read_text(encoding="utf-8")


def test_marker_anchored_in_source():
    assert "P1-DAYGEN-FATS-BUDGET" in _DG


def test_section_18_content_anchors():
    i = _DG.index("18. PRESUPUESTO DE GRASAS")
    win = _DG[i:i + 2200]
    assert "POR DÍA" in win, "la banda se mide por día, no por promedio"
    assert "DOS fuentes grasas DENSAS" in win, "regla de conteo de fuentes densas"
    assert "1 cda (15 ml) por plato" in win, "cap de aceite por plato"
    assert "OCULTAS" in win, "grasas ocultas de la proteína (salmón/res 80/20/piel/huevo)"
    assert "UNA grasa protagonista por plato" in win


def test_static_import_time_prompt_cache_preserved():
    """El §18 se concatena a DAY_GENERATOR_SYSTEM_PROMPT a import-time (string estático) —
    mismo contrato del §17 (P1-PRECISION-LEVERS): el SystemMessage sigue siendo cacheable."""
    i17 = _DG.index("17. PRESUPUESTO DE SODIO")
    i18 = _DG.index("18. PRESUPUESTO DE GRASAS")
    assert i17 < i18, "el §18 vive después del §17 (numeración consistente)"
    # se appendea al MISMO prompt estático (no a un builder dinámico por-request)
    win = _DG[max(0, i18 - 1200):i18]
    assert "DAY_GENERATOR_SYSTEM_PROMPT = DAY_GENERATOR_SYSTEM_PROMPT + (" in win


def test_prompt_actually_contains_section_at_import():
    import importlib
    dg = importlib.import_module("prompts.day_generator")
    assert "18. PRESUPUESTO DE GRASAS" in dg.DAY_GENERATOR_SYSTEM_PROMPT
    assert "17. PRESUPUESTO DE SODIO" in dg.DAY_GENERATOR_SYSTEM_PROMPT
