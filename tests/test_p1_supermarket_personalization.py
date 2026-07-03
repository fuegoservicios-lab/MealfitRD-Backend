"""[P1-SUPERMARKET-PERSONALIZATION · 2026-07-03] (audit v6 · P1-2) Conexión supermarket → generación.

Los ~2,000 supermarket_products alimentaban SOLO lista/costeo/sugerencias — 0% de influencia en la
CREACIÓN de platos. Cierra la fase 1: `brand_personalization.build_brand_pref_context` convierte
las marcas preferidas del usuario (user_brand_preferences ⋈ supermarket_products) en señal de
preferencia POSITIVA inyectada al planner/day-gen por el canal taste_profile, + gap-report de
familias sin master verificado (candidatos de expansión de catálogo).
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_BP = (_BACKEND / "brand_personalization.py").read_text(encoding="utf-8")
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_SCRIPT = _BACKEND / "scripts" / "supermarket_family_gap_report_2026_07_03.py"

_UUIDISH = "11111111-2222-3333-4444-555555555555"


def test_marker_bumped():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-SUPERMARKET-PERSONALIZATION" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-03"


def test_knob_default_on_in_source():
    assert re.search(
        r'BRAND_PREF_PERSONALIZATION_ENABLED\s*=\s*_env_bool\("MEALFIT_BRAND_PREF_PERSONALIZATION",\s*True\)',
        _BP,
    ), "el knob MEALFIT_BRAND_PREF_PERSONALIZATION debe nacer ON"


def test_guests_never_query_db(monkeypatch):
    import brand_personalization as bp

    def _boom(*a, **k):
        raise AssertionError("guest jamás debe consultar la DB")

    import db
    monkeypatch.setattr(db, "execute_sql_query", _boom)
    for uid in (None, "", "guest", "guest-abc123", "short-id"):
        assert bp.fetch_user_brand_foods(uid) == []
        assert bp.build_brand_pref_context(uid) == ""


def test_context_formats_foods_with_brands(monkeypatch):
    import brand_personalization as bp
    monkeypatch.setattr(bp, "BRAND_PREF_PERSONALIZATION_ENABLED", True)
    monkeypatch.setattr(bp, "fetch_user_brand_foods", lambda uid: [
        {"food": "yogurt griego", "brand": "La Sanjuanera"},
        {"food": "arroz blanco", "brand": ""},
    ])
    ctx = bp.build_brand_pref_context(_UUIDISH)
    assert "yogurt griego (La Sanjuanera)" in ctx
    assert "arroz blanco" in ctx
    assert "POSITIVA" in ctx
    # señal SUAVE: jamás obliga — el copy debe decirlo explícito
    assert "NO los fuerces" in ctx


def test_context_empty_without_prefs(monkeypatch):
    import brand_personalization as bp
    monkeypatch.setattr(bp, "BRAND_PREF_PERSONALIZATION_ENABLED", True)
    monkeypatch.setattr(bp, "fetch_user_brand_foods", lambda uid: [])
    assert bp.build_brand_pref_context(_UUIDISH) == "", \
        "sin preferencias → '' byte-equivalente (prompt-cache preservado)"


def test_context_empty_when_knob_off(monkeypatch):
    import brand_personalization as bp
    monkeypatch.setattr(bp, "BRAND_PREF_PERSONALIZATION_ENABLED", False)
    monkeypatch.setattr(bp, "fetch_user_brand_foods",
                        lambda uid: [{"food": "arroz", "brand": "X"}])
    assert bp.build_brand_pref_context(_UUIDISH) == ""


def test_fetch_caps_items(monkeypatch):
    import brand_personalization as bp
    import db
    rows = [{"food": f"alimento {i}", "brand": "M"} for i in range(30)]
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: rows)
    got = bp.fetch_user_brand_foods(_UUIDISH)
    assert len(got) <= int(bp.BRAND_PREF_MAX_ITEMS), \
        "el contexto debe caparse (MEALFIT_BRAND_PREF_MAX_ITEMS) — no inflar el prompt"


def test_fetch_failopen_on_db_error(monkeypatch):
    import brand_personalization as bp
    import db

    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "execute_sql_query", _boom)
    assert bp.fetch_user_brand_foods(_UUIDISH) == []
    assert bp.build_brand_pref_context(_UUIDISH) == ""


# ════════════════════════════════════════════════════════════════════════════
# Cableado en el context builder (canal taste_profile → planner + day-gen)
# ════════════════════════════════════════════════════════════════════════════
def test_wired_in_shared_context_after_learned_taste():
    assert "build_brand_pref_context as _bbc_prefs" in _GO, \
        "falta la inyección de marcas preferidas en _build_shared_context"
    idx_taste = _GO.index("build_taste_context as _btc_learned")
    idx_brand = _GO.index("build_brand_pref_context as _bbc_prefs")
    assert idx_taste < idx_brand, "mismo canal, después del taste aprendido"
    # ambos appendean a taste_profile (fluye a planner + day-gen sin tocar prompts)
    blk = _GO[idx_brand:idx_brand + 800]
    assert 'taste_profile = (taste_profile or "") + "\\n" + _brand_ctx' in blk


# ════════════════════════════════════════════════════════════════════════════
# Gap-report de familias (read-only, candidatos de expansión de catálogo)
# ════════════════════════════════════════════════════════════════════════════
def test_family_gap_report_exists_and_readonly():
    assert _SCRIPT.exists(), "falta scripts/supermarket_family_gap_report_2026_07_03.py"
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "master_ingredients" in src and "supermarket_products" in src
    assert "_norm_pref_food" in src, "debe reusar la normalización SSOT del engine"
    up = re.sub(r'""".*?"""', "", src, flags=re.S)  # sin docstring
    assert not re.search(r"\b(INSERT|UPDATE|DELETE|ALTER|DROP)\b", up), \
        "el gap-report es READ-ONLY — jamás escribe"
