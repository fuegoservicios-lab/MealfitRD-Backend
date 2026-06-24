"""[P3-WATER-HALF-GLASS · 2026-06-24] Contrato de medios vasos (0.5) en el
tracker de hidratacion.

El rediseño del card de Hidratacion ofrece "Sorbo" (½ vaso) + un stepper de ½.
Para soportarlo end-to-end:
  - La columna `water_intake_log.glasses` pasa de INT a NUMERIC(5,1) (migracion).
  - El POST /water-intake acepta (int, float) + valida paso de 0.5.
  - El GET devuelve float (numeric → float).
  - La tool del chat `log_water_glass` acepta deltas fraccionarios (0.5).
  - El frontend (WaterTracker) suma/resta de a 0.5 (Sorbo + stepper).

Tests parser-based (anclan source de prod con regex; un renombre falla el test
antes de tumbar produccion).
"""
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"
_TOOLS_PY = _BACKEND_ROOT / "tools.py"
_WATER_TRACKER_JSX = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "WaterTracker.jsx"
_MIGRATION = _BACKEND_ROOT / "migrations" / "p3_water_half_glasses_2026_06_24.sql"
_MIGRATION_ROOT = _REPO_ROOT / "migrations" / "p3_water_half_glasses_2026_06_24.sql"


# ---------------------------------------------------------------------------
# 1. Migration
# ---------------------------------------------------------------------------
def test_migration_exists_in_both_ssot_dirs_and_idempotent():
    """La migracion vive en AMBOS dirs SSOT (P3-MIGRATIONS-SSOT) y es idempotente."""
    assert _MIGRATION.exists(), f"Migration ausente en backend/migrations: {_MIGRATION}"
    assert _MIGRATION_ROOT.exists(), f"Migration ausente en migrations/ (root): {_MIGRATION_ROOT}"
    sql = _MIGRATION.read_text(encoding="utf-8")
    assert sql == _MIGRATION_ROOT.read_text(encoding="utf-8"), (
        "Las dos copias de la migracion difieren — deben ser identicas (P3-MIGRATIONS-SSOT)."
    )
    # Cambia el tipo de columna a numeric.
    assert re.search(r"ALTER\s+COLUMN\s+glasses\s+TYPE\s+numeric", sql, re.IGNORECASE), (
        "La migracion debe cambiar glasses a NUMERIC."
    )
    # Guard idempotente por data_type.
    assert "data_type = 'integer'" in sql, "Guard idempotente (data_type='integer') ausente."
    # CHECK de paso 0.5.
    assert "DROP CONSTRAINT IF EXISTS water_intake_log_glasses_half_step" in sql
    assert "(glasses * 2) = floor(glasses * 2)" in sql, (
        "CHECK de paso 0.5 ausente — sin el, un INSERT directo podria meter 0.3 vasos."
    )
    # Sanity check (convencion P3-MIGRATION-IDEMPOTENCE-DOC).
    assert "RAISE EXCEPTION" in sql


# ---------------------------------------------------------------------------
# 2. Backend endpoint
# ---------------------------------------------------------------------------
def test_post_accepts_int_and_float_with_half_step_guard():
    body = _PLANS_PY.read_text(encoding="utf-8")
    post_block = re.search(r'@router\.post\("/water-intake"\).*?(?=@router\.)', body, re.DOTALL)
    assert post_block
    blk = post_block.group(0)
    assert "isinstance(raw_glasses, (int, float))" in blk, (
        "El POST debe aceptar int Y float (medios vasos)."
    )
    assert "isinstance(raw_glasses, bool)" in blk, "El guard de bool no debe perderse."
    assert "(raw_glasses * 2) != int(raw_glasses * 2)" in blk, "Guard de paso 0.5 ausente."
    # Persiste/retorna el valor float, no int().
    assert "glasses_val = float(raw_glasses)" in blk
    assert "int(raw_glasses)" not in blk, (
        "El POST ya no debe truncar a int(raw_glasses) — perderia el medio vaso."
    )


def test_get_returns_float_and_streak():
    body = _PLANS_PY.read_text(encoding="utf-8")
    get_block = re.search(
        r'@router\.get\("/water-intake"\).*?(?=\n@router\.|\nasync def |\ndef api_set_water_intake)',
        body,
        re.DOTALL,
    )
    assert get_block
    blk = get_block.group(0)
    assert "float(res.get(\"glasses\") or 0)" in blk, "El GET debe leer glasses como float."
    # Racha de hidratacion (para el card rediseñado).
    assert '"streak"' in blk and "_compute_water_streak" in blk, (
        "El GET debe incluir `streak` (racha de dias) para el card rediseñado."
    )
    assert "def _compute_water_streak(" in body, "Helper _compute_water_streak ausente."


# ---------------------------------------------------------------------------
# 3. Agent tool
# ---------------------------------------------------------------------------
def test_log_water_glass_accepts_fractional_delta():
    src = _TOOLS_PY.read_text(encoding="utf-8")
    block = re.search(r"def log_water_glass\(.*?(?=\n@tool|\ndef check_hydration)", src, re.DOTALL)
    assert block
    blk = block.group(0)
    assert "count_delta: float" in blk, "log_water_glass debe declarar count_delta: float."
    assert "isinstance(count_delta, (int, float))" in blk
    assert "(count_delta * 2) != int(count_delta * 2)" in blk, (
        "La tool debe validar paso de 0.5 (medio vaso)."
    )


# ---------------------------------------------------------------------------
# 4. Frontend
# ---------------------------------------------------------------------------
def test_frontend_supports_half_glass_and_keeps_backend_layer():
    src = _WATER_TRACKER_JSX.read_text(encoding="utf-8")
    # Suma de medio vaso (Sorbo + stepper de ½).
    assert "0.5" in src, "El frontend debe operar con medios vasos (0.5)."
    # Conserva la capa de backend (no se reemplazo por localStorage puro).
    assert "/api/plans/water-intake" in src, (
        "WaterTracker debe seguir persistiendo al backend (sync/reset/meta), "
        "no solo localStorage."
    )
    # Conserva el contrato de meta dinamica (test_p3_water_tracker lo exige tambien).
    assert "const [goal, setGoal]" in src and "const [goalBasis, setGoalBasis]" in src
