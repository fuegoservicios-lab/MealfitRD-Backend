"""[P2-PRICES-ENGINE-1 · 2026-06-16] Tests del motor de precios + inflación.

Cubre:
  - Helpers puros (clamp_factor, adjusted_price) sin DB.
  - Lógica de recompute (factor = base × índice, idempotencia, skip sin período base).
  - Validación de ingest_inflation_index e import_base_prices con DB mockeada.
  - Migración SSOT dual-dir (idéntica en migrations/ y backend/migrations/) + idempotencia.
  - Anchors del marker + registro del cron (parser-based; un renombre falla el test).
"""
import os
import re

import pytest

import price_engine as pe

_HERE = os.path.dirname(__file__)
_BACKEND_MIG = os.path.join(_HERE, "..", "migrations", "p2_prices_engine_1_2026_06_16.sql")
_ROOT_MIG = os.path.join(_HERE, "..", "..", "migrations", "p2_prices_engine_1_2026_06_16.sql")


# ── Helpers puros ────────────────────────────────────────────────────────────
def test_clamp_factor_within_range_passthrough():
    assert pe.clamp_factor(1.1, 3.0) == 1.1
    assert pe.clamp_factor(1.0, 3.0) == 1.0


def test_clamp_factor_caps_high_and_low():
    assert pe.clamp_factor(100.0, 3.0) == 3.0
    assert pe.clamp_factor(0.001, 3.0) == pytest.approx(1.0 / 3.0)


def test_adjusted_price_math_and_none_safe():
    assert pe.adjusted_price(100, 1.1) == 110.0
    assert pe.adjusted_price(38.5, 1.0) == 38.5
    assert pe.adjusted_price(None, 1.1) is None
    assert pe.adjusted_price("nan-ish", 1.1) is None


# ── recompute_adjusted_prices (DB mockeada) ──────────────────────────────────
def _install_fake_db(monkeypatch, master_rows, index_rows):
    """Mockea execute_sql_query/write de price_engine. Devuelve la lista de writes."""
    writes = []
    latest = max(index_rows, key=lambda r: r["period"]) if index_rows else None

    def fake_query(query, params=None, fetch_one=False, fetch_all=False):
        q = " ".join(query.split())
        if "ORDER BY period DESC LIMIT 1" in q:
            return dict(latest) if latest else None
        if "SELECT period, food_cpi FROM price_inflation_index" in q:
            return [dict(r) for r in index_rows]
        if "FROM master_ingredients" in q and "price_per_lb_base" in q:
            return [dict(r) for r in master_rows]
        return [] if fetch_all else None

    def fake_write(query, params=None, returning=False, lock_timeout_ms=None):
        writes.append({"query": " ".join(query.split()), "params": params})
        return [{"id": "x"}] if returning else True

    monkeypatch.setattr(pe, "execute_sql_query", fake_query)
    monkeypatch.setattr(pe, "execute_sql_write", fake_write)
    return writes


def test_recompute_applies_inflation_factor(monkeypatch):
    # Base 100 RD$/lb capturada en 2025-01 (cpi 100); índice actual 2026-01 (cpi 110) → ×1.1 → 110.
    master = [{"id": "a", "name": "Pollo", "price_per_lb": 100.0, "price_per_unit": None,
               "price_per_lb_base": 100.0, "price_per_unit_base": None, "price_base_period": "2025-01"}]
    index = [{"period": "2025-01", "food_cpi": 100.0}, {"period": "2026-01", "food_cpi": 110.0}]
    writes = _install_fake_db(monkeypatch, master, index)
    res = pe.recompute_adjusted_prices(force=True)
    assert res["status"] == "ok"
    assert res["updated"] == 1
    assert writes and writes[0]["params"][0] == 110.0  # nuevo price_per_lb


def test_recompute_idempotent_skips_unchanged(monkeypatch):
    # El vivo ya es 110 (= base 100 × 1.1) → no debe re-escribir.
    master = [{"id": "a", "name": "Pollo", "price_per_lb": 110.0, "price_per_unit": None,
               "price_per_lb_base": 100.0, "price_per_unit_base": None, "price_base_period": "2025-01"}]
    index = [{"period": "2025-01", "food_cpi": 100.0}, {"period": "2026-01", "food_cpi": 110.0}]
    writes = _install_fake_db(monkeypatch, master, index)
    res = pe.recompute_adjusted_prices(force=True)
    assert res["updated"] == 0
    assert not writes


def test_recompute_skips_when_base_period_index_missing(monkeypatch):
    # El período base 2099-12 no está en la serie → no se puede proyectar → skip.
    master = [{"id": "a", "name": "Pollo", "price_per_lb": 0.0, "price_per_unit": None,
               "price_per_lb_base": 100.0, "price_per_unit_base": None, "price_base_period": "2099-12"}]
    index = [{"period": "2026-01", "food_cpi": 110.0}]
    writes = _install_fake_db(monkeypatch, master, index)
    res = pe.recompute_adjusted_prices(force=True)
    assert res["skipped"] == 1
    assert res["updated"] == 0
    assert not writes


def test_recompute_no_index_is_noop(monkeypatch):
    _install_fake_db(monkeypatch, [], [])
    res = pe.recompute_adjusted_prices(force=True)
    assert res["status"] == "no_index"


def test_recompute_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MEALFIT_PRICES_ENABLED", raising=False)
    _install_fake_db(monkeypatch, [], [])
    res = pe.recompute_adjusted_prices()  # sin force → respeta el gate
    assert res["status"] == "disabled"


def test_prices_enabled_reads_env(monkeypatch):
    monkeypatch.setenv("MEALFIT_PRICES_ENABLED", "true")
    assert pe.prices_enabled() is True
    monkeypatch.setenv("MEALFIT_PRICES_ENABLED", "false")
    assert pe.prices_enabled() is False


# ── ingest_inflation_index ───────────────────────────────────────────────────
def test_ingest_index_rejects_bad_period(monkeypatch):
    monkeypatch.setattr(pe, "execute_sql_write", lambda *a, **k: True)
    with pytest.raises(ValueError):
        pe.ingest_inflation_index("2026/06", 110.0)
    with pytest.raises(ValueError):
        pe.ingest_inflation_index("2026-06", 0)


def test_ingest_index_happy_path(monkeypatch):
    captured = {}
    monkeypatch.setattr(pe, "execute_sql_write",
                        lambda q, p=None, **k: captured.update({"params": p}) or True)
    out = pe.ingest_inflation_index("2026-06", 121.4, source="bcrd")
    assert out == {"period": "2026-06", "food_cpi": 121.4}
    assert captured["params"][0] == "2026-06"


# ── import_base_prices ───────────────────────────────────────────────────────
def test_import_base_prices_match_and_unmatch(monkeypatch):
    calls = []

    def fake_write(query, params=None, returning=False, lock_timeout_ms=None):
        calls.append({"query": " ".join(query.split()), "params": params})
        # "Pollo" matchea (devuelve fila); "Inexistente" no.
        if params and params[-1] == "Inexistente":
            return []
        return [{"id": "x"}]

    monkeypatch.setattr(pe, "execute_sql_write", fake_write)
    rows = [
        {"name": "Pollo", "price_per_lb_base": "95", "price_confidence": "low"},
        {"name": "Inexistente", "price_per_lb_base": "10"},
        {"slug": "arroz", "price_per_lb_base": "38", "price_confidence": "high"},
    ]
    res = pe.import_base_prices(rows, default_period="2026-06")
    assert res["matched"] == 2
    assert res["unmatched"] == 1
    assert "Inexistente" in res["unmatched_keys"]
    # El row con slug usa WHERE slug = %s; el row con name usa lower(name).
    assert any("WHERE slug = %s" in c["query"] for c in calls)
    assert any("lower(name) = lower(%s)" in c["query"] for c in calls)


def test_import_rejects_invalid_confidence_silently(monkeypatch):
    captured = {}
    monkeypatch.setattr(pe, "execute_sql_write",
                        lambda q, p=None, **k: captured.update({"p": p}) or [{"id": "x"}])
    pe.import_base_prices([{"name": "Pollo", "price_confidence": "altísima"}])
    # price_confidence (5º param del SET) debe ser None (no la basura).
    assert captured["p"][4] is None


# ── Migración SSOT dual-dir + idempotencia ───────────────────────────────────
def test_migration_exists_in_both_dirs():
    assert os.path.exists(_BACKEND_MIG), "Falta backend/migrations/p2_prices_engine_1_*.sql"
    assert os.path.exists(_ROOT_MIG), "Falta migrations/p2_prices_engine_1_*.sql (SSOT dual-dir)"


def test_migration_dual_dir_identical():
    with open(_BACKEND_MIG, encoding="utf-8") as f:
        a = f.read()
    with open(_ROOT_MIG, encoding="utf-8") as f:
        b = f.read()
    assert a == b, "Las dos copias de la migración divergen (P3-MIGRATIONS-SSOT)"


def test_migration_is_idempotent_and_anchored():
    with open(_BACKEND_MIG, encoding="utf-8") as f:
        sql = f.read()
    assert "P2-PRICES-ENGINE-1" in sql
    assert "ADD COLUMN IF NOT EXISTS price_per_lb_base" in sql
    assert "CREATE TABLE IF NOT EXISTS public.price_inflation_index" in sql
    assert "DROP CONSTRAINT IF EXISTS" in sql
    assert "RAISE EXCEPTION" in sql  # sanity DO block
    assert re.search(r"food_cpi\s*>\s*0", sql)  # CHECK de positividad


# ── Anchors de código (parser-based) ─────────────────────────────────────────
def test_cron_job_registered():
    with open(os.path.join(_HERE, "..", "cron_tasks.py"), encoding="utf-8") as f:
        src = f.read()
    assert "def _price_inflation_adjust_job" in src
    assert 'id="price_inflation_adjust"' in src
    assert "P2-PRICES-ENGINE-1" in src


def test_engine_marker_present():
    with open(os.path.join(_HERE, "..", "price_engine.py"), encoding="utf-8") as f:
        src = f.read()
    assert "P2-PRICES-ENGINE-1" in src
