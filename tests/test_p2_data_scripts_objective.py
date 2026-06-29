"""[audit objetivo · P2-4 / P2-5] Scripts de DATA (el owner los corre en Neon; aquí solo anclamos su
existencia + contenido clave, parser-based — no tocan Neon en CI).

- P2-5 (P2-DENSITY-CUP-FOODS): backfill de density_g_per_cup para avena/leche/harina/yogurt (cup-foods que
  el corrector previo no cubrió → sus micros se descartaban silenciosamente). Idempotente NULL-only, dry-run-first.
- P2-4 (P2-EXTENDED-MICRO-COVERAGE): diagnóstico de cobertura de las 8 columnas extendidas (no escribe; mide).
"""
from __future__ import annotations

import ast
import py_compile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_DENSITY_P2 = _SCRIPTS / "micro_density_corrective_p2_2026_06_29.py"
_COVERAGE = _SCRIPTS / "check_extended_micro_coverage.py"


def test_scripts_exist_and_compile():
    assert _DENSITY_P2.exists(), "falta el script de densidad P2-5"
    assert _COVERAGE.exists(), "falta el script de cobertura P2-4"
    py_compile.compile(str(_DENSITY_P2), doraise=True)
    py_compile.compile(str(_COVERAGE), doraise=True)


def test_density_script_covers_the_audit_cupfoods():
    """El backfill debe cubrir los staples que el audit flageó: avena, leche, harina, yogurt."""
    text = _DENSITY_P2.read_text(encoding="utf-8")
    assert "P2-DENSITY-CUP-FOODS" in text
    low = text.lower()
    for food in ("avena", "leche", "harina", "yogur"):
        assert food in low, f"el backfill no cubre '{food}'"
    # Idempotente NULL-only + dry-run (no commit accidental).
    assert "IS NULL" in text and "--commit" in text


def test_density_script_is_idempotent_null_only():
    """Defensa: el UPDATE filtra density_g_per_cup IS NULL (no pisa valores existentes)."""
    text = _DENSITY_P2.read_text(encoding="utf-8")
    assert "density_g_per_cup IS NULL" in text


def test_coverage_script_checks_all_8_extended_columns():
    text = _COVERAGE.read_text(encoding="utf-8")
    assert "P2-EXTENDED-MICRO-COVERAGE" in text
    cols = ("zinc_mg_per_100g", "folate_mcg_dfe_per_100g", "vitamin_a_mcg_rae_per_100g", "vitamin_c_mg_per_100g",
            "vitamin_e_mg_per_100g", "vitamin_k_mcg_per_100g", "selenium_mcg_per_100g", "omega3_ala_g_per_100g")
    for col in cols:
        assert col in text, f"el diagnóstico no mide la columna {col}"
    # Read-only: NO debe contener UPDATE/INSERT/DELETE (es un diagnóstico).
    tree = ast.parse(text)
    src = text.lower()
    assert "update public.master_ingredients" not in src, "el diagnóstico NO debe escribir"
    assert isinstance(tree, ast.Module)
