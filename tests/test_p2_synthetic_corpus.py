"""[P2-SYNTHETIC-CORPUS · 2026-07-04] Test ancla del corpus de training y del
generador de planes sintéticos.

Contratos:
  1. Migración SSOT en ambos dirs, idempotente, con CHECK de `source` y la
     coherencia source↔user_id (sintético sin user_id; 'user' lo requiere).
  2. El generador reusa el set held-out del benchmark (no perfiles ad-hoc
     cherry-picked), corre el pipeline REAL y aplica gate de calidad
     (review_passed + no-fallback + banda de macros + 7 días).
  3. Dry-run por default: el INSERT está gateado por --commit.
  4. Solo inserta source='synthetic' con user_id NULL — las filas 'user' son
     territorio del ETL futuro (gate get_ai_training_consented_user_ids).
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND.parent
_MIGRATION_NAME = "p2_synthetic_corpus_2026_07_04.sql"
_SCRIPT = _BACKEND / "scripts" / "generate_synthetic_corpus.py"


def _read(path: Path) -> str:
    assert path.exists(), f"No existe {path} — ¿se renombró sin actualizar el test?"
    return path.read_text(encoding="utf-8")


def test_migration_in_both_dirs_identical_and_constrained():
    backend_mig = _BACKEND / "migrations" / _MIGRATION_NAME
    root_mig = _REPO_ROOT / "migrations" / _MIGRATION_NAME
    src = _read(backend_mig)
    if root_mig.exists():
        assert src == root_mig.read_text(encoding="utf-8"), (
            "P3-MIGRATIONS-SSOT: la migración difiere entre ambos dirs."
        )
    assert "CREATE TABLE IF NOT EXISTS public.ai_training_corpus" in src
    assert "DROP CONSTRAINT IF EXISTS" in src, "Idempotencia: DROP IF EXISTS antes de ADD."
    assert re.search(r"CHECK \(source IN \('synthetic', 'user'\)\)", src)
    assert "ai_training_corpus_source_user_coherence" in src, (
        "Falta el CHECK de coherencia source↔user_id."
    )
    assert "RAISE EXCEPTION" in src, "Falta el sanity check DO $$."


def test_generator_uses_benchmark_profiles_and_real_pipeline():
    src = _read(_SCRIPT)
    assert "from scripts.benchmark_macro_compliance import PROFILES" in src, (
        "El generador debe reusar el set held-out del benchmark (no perfiles "
        "ad-hoc — el corpus de eval debe ser reproducible y no cherry-picked)."
    )
    assert "arun_plan_pipeline" in src, "Debe correr el pipeline REAL."


def test_generator_quality_gate():
    src = _read(_SCRIPT)
    # `>= 3`: el pipeline síncrono produce el chunk inicial de 3 días (los días
    # 4-7 llegan por chunk workers sobre planes persistidos, fuera de alcance).
    # `_review_failed_but_delivered`: contrato P3-NEW-8 del plan_result final
    # (`review_passed` vive solo en el state del grafo, NO en el resultado).
    for signal in ("_review_failed_but_delivered", "_is_fallback", "worst_dev", ">= 3"):
        assert signal in src, f"Falta la señal de calidad {signal!r} en el gate."


def test_generator_dry_run_default_and_synthetic_only():
    src = _read(_SCRIPT)
    # El INSERT solo corre bajo --commit (dry-run default).
    m = re.search(r"if commit:\s*\n\s+from db import execute_sql_write", src)
    assert m, "El INSERT debe estar gateado por --commit (dry-run default)."
    assert re.search(r"VALUES \('synthetic', NULL", src), (
        "El generador SOLO inserta source='synthetic' con user_id NULL — las "
        "filas 'user' pertenecen al ETL futuro con gate de consentimiento."
    )
