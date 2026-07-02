"""[P3-VERIFIED-INGREDIENTS-ONLY · 2026-06-20] SOLO alimentos verificados con precio
La Sirena (los ~202 verificados de master_ingredients (era 119 pre-expansion 2026-06-26)) pueden aparecer en la lista de compras.
Decisión del owner: "no quiero que el LLM invente alimentos; solo los verificados del catalogo (~202)
deben estar en la lista". Un ingrediente inventado por el LLM (achiote, sazón en polvo,
clavo dulce...) no resuelve a un master con precio → se EXCLUYE de la lista.
[P3-AUDIT-V2-RESIDUALS · 2026-07-01] laurel/comino/cúrcuma ya NO sirven como fixtures de
"inventado": P1-SPICES-CATALOG-SYNC los promovió al catálogo con precio.

Diseño (verificado contra Neon prod 2026-06-20):
  - El drop (aggregate_and_deduct_shopping_list) y el espejo (run_shopping_coherence_guard,
    filtro de expected_raw) consumen la MISMA `_is_verified_for_shopping` → la simetría
    drop↔espejo está garantizada por construcción.
  - El drop usa una condición MÁS ESTRICTA (no-verificado Y sin precio en master) → es un
    SUBCONJUNTO del filtro del espejo (solo no-verificado) → es IMPOSIBLE dropear de la lista
    algo que siga en `expected_raw` → cero divergencias `expected_only` → cero retry forzado.
  - Medición real (test de 1 comida con laurel+comino): con el enforcement, divergencias
    9→7 (las de magnitud de laurel/comino desaparecen), `expected_only` vacío en ambos casos.
Knob `MEALFIT_VERIFIED_INGREDIENTS_ONLY` (default True). Flip a False revierte sin redeploy.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_SC = Path(__file__).resolve().parent.parent / "shopping_calculator.py"
_ANCHOR = "P3-VERIFIED-INGREDIENTS-ONLY"


# ── Parser anchors (siempre corren, sin DB) ──
def test_anchor_and_helper_present():
    src = _SC.read_text(encoding="utf-8")
    assert _ANCHOR in src, "falta el marker P3-VERIFIED-INGREDIENTS-ONLY"
    assert "def _is_verified_for_shopping" in src, "falta el helper SSOT"
    assert "def _verified_ingredients_only_enabled" in src, "falta el gate del knob"
    assert "MEALFIT_VERIFIED_INGREDIENTS_ONLY" in src


def test_default_on_in_code():
    """[P1-VERIFIED-ONLY-DEFAULT-ON · 2026-07-02] Default ON en CÓDIGO: el knob es load-bearing
    para la garantía coherente+costeable (prod lo corre ON desde 2026-06-20); OFF-en-código dejaba
    la regresión silenciosa ".env reseteado ⇒ enforcement apagado". El baseline de la suite se
    preserva vía tests/conftest.py (setdefault a false)."""
    src = _SC.read_text(encoding="utf-8")
    assert '"MEALFIT_VERIFIED_INGREDIENTS_ONLY", True' in src, (
        "el default EN CÓDIGO debe ser True (P1-VERIFIED-ONLY-DEFAULT-ON); "
        "rollback operacional: MEALFIT_VERIFIED_INGREDIENTS_ONLY=false en el .env"
    )
    # El baseline de la suite queda fijado explícito en conftest (los tests base asumen OFF).
    _conftest = _SC.parent / "tests" / "conftest.py"
    assert 'environ.setdefault("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "false")' in _conftest.read_text(encoding="utf-8"), (
        "tests/conftest.py debe fijar el baseline OFF para la suite histórica"
    )


def test_drop_and_mirror_share_the_same_gate_and_check():
    """El drop (aggregator) y el espejo (guard) deben usar la MISMA función gate
    y el MISMO check → nunca asimétricos → imposible generar expected_only."""
    src = _SC.read_text(encoding="utf-8")
    # def + drop + mirror = al menos 3 referencias a la función de check
    assert src.count("_is_verified_for_shopping") >= 3, (
        "drop y espejo deben compartir _is_verified_for_shopping"
    )
    # drop + mirror = al menos 2 referencias al gate del knob (más el def = 3)
    assert src.count("_verified_ingredients_only_enabled()") >= 2, (
        "drop y espejo deben gatearse por el mismo knob"
    )


# ── Integración (gated: requiere connection_pool a Neon) ──
def _db_ready() -> bool:
    try:
        import db_core
        if db_core.connection_pool is None:
            return False
        db_core.connection_pool.open()
        from shopping_calculator import get_master_ingredients
        return len(get_master_ingredients() or []) > 0
    except Exception:
        return False


_DB = _db_ready()


@pytest.fixture
def _force_enforcement(monkeypatch):
    """Fuerza el enforcement ON (el default EN CÓDIGO es OFF). Parchea el gate SSOT
    → cubre el drop del aggregator Y el espejo del guard (ambos consumen esta función)."""
    import shopping_calculator
    monkeypatch.setattr(shopping_calculator, "_verified_ingredients_only_enabled", lambda: True)


@pytest.mark.skipif(not _DB, reason="requiere connection_pool a Neon prod")
def test_helper_classifies_invented_vs_verified():
    from shopping_calculator import _is_verified_for_shopping
    # [P3-AUDIT-V2-RESIDUALS · 2026-07-01] fixtures actualizados: laurel/comino/cúrcuma fueron
    # PROMOVIDOS al catálogo con precio por P1-SPICES-CATALOG-SYNC (lote 2026-07-01) — este test
    # (que corre contra el dato VIVO) quedó stale usando esas especias como "inventadas". Los
    # off-catálogo estables son los que el prompt del day-gen prohíbe explícitamente.
    # invenciones del LLM (no en el catálogo verificado (~202)) → False
    assert _is_verified_for_shopping("achiote") is False
    assert _is_verified_for_shopping("sazon en polvo") is False
    assert _is_verified_for_shopping("clavo dulce") is False
    # verificados (resuelven a master con precio) → True
    assert _is_verified_for_shopping("oregano") is True
    assert _is_verified_for_shopping("pechuga de pollo") is True
    assert _is_verified_for_shopping("arroz blanco") is True
    assert _is_verified_for_shopping("cilantro") is True  # el sofrito SÍ está
    assert _is_verified_for_shopping("laurel") is True    # promovido P1-SPICES-CATALOG-SYNC
    assert _is_verified_for_shopping("comino") is True    # promovido P1-SPICES-CATALOG-SYNC


@pytest.mark.skipif(not _DB, reason="requiere connection_pool a Neon prod")
def test_aggregator_drops_unverified_keeps_verified(_force_enforcement):
    from shopping_calculator import aggregate_and_deduct_shopping_list
    res = aggregate_and_deduct_shopping_list(
        ["120g de pechuga de pollo", "70g de arroz blanco",
         "1 cdta de achiote", "1 cdta de sazon en polvo", "1 cdta de oregano"],
        structured=True,
    )
    names = [str(i.get("name")).lower() for i in res]
    assert not any("chiote" in n for n in names), "achiote (off-catálogo) debe dropearse"
    assert not any("sazon en polvo" in n for n in names), "sazón en polvo (off-catálogo) debe dropearse"
    assert any("ollo" in n for n in names), "pollo (verificado) debe quedar"
    assert any("rroz" in n for n in names), "arroz (verificado) debe quedar"
    assert any("gano" in n for n in names), "orégano (verificado) debe quedar"


@pytest.mark.skipif(not _DB, reason="requiere connection_pool a Neon prod")
def test_guard_mirror_no_expected_only_for_dropped(_force_enforcement):
    """El espejo del guard filtra expected_raw → dropear laurel/comino NO genera
    divergencia expected_only (la única que fuerza retry en modo=block)."""
    from shopping_calculator import aggregate_and_deduct_shopping_list, run_shopping_coherence_guard
    ings = ["120g de pechuga de pollo", "70g de arroz blanco",
            "1 cdta de achiote", "1 cdta de sazon en polvo"]
    agg = aggregate_and_deduct_shopping_list(ings, structured=True)
    plan_result = {
        "days": [{"meals": [{"meal": "Almuerzo", "ingredients": ings}]}],
        "aggregated_shopping_list": agg,
        "calc_household_multiplier": 1.0,
    }
    divs = run_shopping_coherence_guard(plan_result, mode_override="block", multiplier=1.0)
    expected_only = [str(d.get("food")).lower() for d in divs if d.get("side") == "expected_only"]
    assert not any("chiote" in f for f in expected_only), (
        "achiote NO debe ser expected_only — el espejo lo filtra de expected_raw"
    )
    assert not any("sazon en polvo" in f for f in expected_only), (
        "sazón en polvo NO debe ser expected_only — el espejo lo filtra de expected_raw"
    )
