"""[P1-REBALANCE-LINE-CLAMP · 2026-07-10] El rebalance del regen-day deja de ser
todo-o-nada frente a la Nevera: el plato cuya línea está topada conserva su porción y el
RESTO del día absorbe el gap completo.

Caso vivo (plan ff14f7cf, regen 02:04 UTC 2026-07-11): cerrar carbos/kcal pedía yogurt
~325g con 150g en Nevera → fracs 50%/25% también rompían (el gap se re-distribuía al
yogurt cada vez) → revert TOTAL → día 0.667 (carbs/kcal 0.0) + banner "limitado por tu
Nevera"… con 51 items y arroz/avena/batata de sobra para cerrar el mismo gap.

tooltip-anchor: P1-REBALANCE-LINE-CLAMP
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


class _FakeDB:
    """macros_from_ingredient_string mínimo: '<n> g de <Nombre>' → name/grams."""

    def macros_from_ingredient_string(self, s):
        try:
            parts = str(s).split(" g de ")
            if len(parts) != 2:
                return None
            return {"name": parts[1].strip(), "grams": float(parts[0].strip())}
        except Exception:
            return None


def _meal(name, protein, carbs, fats, ingredients):
    return {"name": name, "protein": protein, "carbs": carbs, "fats": fats,
            "ingredients": list(ingredients)}


def _mk_pre():
    # Día de 3 comidas: la merienda usa Yogurt (topado); las otras tienen headroom.
    return [
        _meal("Moro con Pollo", 40, 80, 15, ["150 g de Arroz blanco", "120 g de Pechuga de pollo"]),
        _meal("Yogur con Papaya", 15, 30, 8, ["150 g de Yogurt", "100 g de Lechosa"]),
        _meal("Pescado con Batata", 35, 60, 18, ["140 g de Filete de pescado blanco", "200 g de Batata"]),
    ]


def test_line_clamp_excludes_violator_and_rest_absorbs():
    from routers.plans import _rebalance_day_with_line_clamp
    calls = []

    def _fake_rebalance(meals, c, f, db, target_protein=0.0):
        calls.append({"n": len(meals), "c": c, "f": f, "p": target_protein,
                      "names": [m["name"] for m in meals]})
        for m in meals:  # simula el re-apunte
            m["carbs"] = float(m.get("carbs") or 0) * 1.2
        return True

    def _fake_exceeds(meals, ledger, db):
        return False, ""  # con el yogurt excluido, la Nevera alcanza

    ok, meals, excluded = _rebalance_day_with_line_clamp(
        _mk_pre(), {"carbs_g": 291.0, "fats_g": 58.0, "protein_g": 124.0},
        {"Yogurt": 150.0}, _FakeDB(), renal_capped=False,
        rebalance_fn=_fake_rebalance, exceeds_fn=_fake_exceeds,
        first_violator="Yogurt",
    )
    assert ok and meals is not None
    assert excluded == ["Yogurt"]
    # el subset excluye la merienda del yogurt
    assert calls and calls[0]["names"] == ["Moro con Pollo", "Pescado con Batata"]
    # targets del subset = target día − aporte del excluido (291−30, 58−8, 124−15)
    assert abs(calls[0]["c"] - 261.0) < 0.01
    assert abs(calls[0]["f"] - 50.0) < 0.01
    assert abs(calls[0]["p"] - 109.0) < 0.01
    # la comida excluida conserva su estado pre-rebalance (carbs sin escalar)
    _yog = next(m for m in meals if m["name"] == "Yogur con Papaya")
    assert _yog["carbs"] == 30


def test_line_clamp_iterates_on_second_violator():
    from routers.plans import _rebalance_day_with_line_clamp
    state = {"round": 0}

    def _fake_rebalance(meals, c, f, db, target_protein=0.0):
        return True

    def _fake_exceeds(meals, ledger, db):
        state["round"] += 1
        if state["round"] == 1:
            return True, "Batata: necesita ~500g pero hay ~200g"
        return False, ""

    ok, meals, excluded = _rebalance_day_with_line_clamp(
        _mk_pre(), {"carbs_g": 291.0, "fats_g": 58.0, "protein_g": 124.0},
        {"Yogurt": 150.0, "Batata": 200.0}, _FakeDB(), renal_capped=False,
        rebalance_fn=_fake_rebalance, exceeds_fn=_fake_exceeds,
        first_violator="Yogurt",
    )
    assert ok
    assert excluded == ["Yogurt", "Batata"]


def test_line_clamp_falls_back_when_exclusion_empties_day():
    from routers.plans import _rebalance_day_with_line_clamp

    def _fake_rebalance(meals, c, f, db, target_protein=0.0):
        return True

    def _fake_exceeds(meals, ledger, db):
        return False, ""

    # violador presente en TODAS las comidas → exclusión vaciaría el día → fallback
    pre = [
        _meal("A", 10, 20, 5, ["100 g de Arroz blanco"]),
        _meal("B", 10, 20, 5, ["120 g de Arroz blanco"]),
    ]
    ok, meals, _ = _rebalance_day_with_line_clamp(
        pre, {"carbs_g": 100.0, "fats_g": 20.0, "protein_g": 40.0},
        {"Arroz blanco": 100.0}, _FakeDB(), renal_capped=False,
        rebalance_fn=_fake_rebalance, exceeds_fn=_fake_exceeds,
        first_violator="Arroz blanco",
    )
    assert ok is False and meals is None


def test_callsite_ladder_order_lineclamp_then_fracs_then_revert():
    i_lc = _PLANS.find("Nivel 1 de la escalera: excluir el")
    assert i_lc > 0, "el nivel 1 (line-clamp) desapareció del callsite"
    blk = _PLANS[i_lc: i_lc + 3400]
    assert "_rebalance_day_with_line_clamp(" in blk
    assert 'first_violator=(_why.split(":")[0] if _why else "")' in blk, (
        "el primer violador (ya conocido del check inicial) siembra la exclusión"
    )
    # fracs gateados por _partial_ok (no pisan el resultado del line-clamp)
    assert '((0.5, 0.25) if not _partial_ok else ())' in blk
    # el revert total sigue siendo el último nivel
    i_rev = _PLANS.find("rebalance rompió pantry → revertido", i_lc)
    assert i_rev > 0, "el revert final (nivel 3) debe seguir vivo"


def test_marker_anchored_in_source():
    # Durable: SOLO anchors en el código fuente. NO asertar contra app.py — el
    # _LAST_KNOWN_PFIX se bumpea con cada P-fix posterior y el string sale del archivo
    # (4ª mordida de esta clase el 2026-07-10; el cross-link marker↔test lo cubre
    # test_p2_hist_audit_14_marker_test_link).
    assert _PLANS.count("P1-REBALANCE-LINE-CLAMP") >= 3
