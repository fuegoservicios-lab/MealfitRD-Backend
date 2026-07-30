"""[P1-GRAM-HINT-TRUMPS-QTY · 2026-07-30] 10 potes de mantequilla de maní (RD$1.170) para una
tostada que lleva 4 gramos.

Caso vivo del owner, leído de la base (plan 307395c7, desayuno del día 2):

    ingredients      →  '¼ cda de mantequilla de maní (4 g)'     ← lo que el usuario LEE
    ingredients_raw  →  '30 cdas de mantequilla de maní (4 g)'   ← lo que la lista COMPRA

La lista de compras lee `ingredients_raw`. 30 cdas ≈ 450 g por aparición × el multiplicador del
ciclo = `base_qty: 4515.0` en la lista guardada ⇒ 4515/453,6 = **10 potes**, la línea más cara del
ciclo de 30 días. Dos planes generados el mismo minuto dieron 244 g y 1 pote.

Lo que NO estaba roto (verificado antes de tocar nada): el parser de cantidad (el fix de
2026-07-27 aguanta en las 5 formas, incluida la corrupta), el parser de presentación
('Pote Cremosa 16 Oz' → 453,592 g) y la aritmética del empaque. El defecto es la CANTIDAD DE ORIGEN.

⚠️ Y ningún guard podía verlo: `expected_sum_from_recipes` lee
`meal.get("ingredients_raw") or meal.get("ingredients")` — la MISMA fuente que el agregador. Los dos
lados de su comparación salen de raw, coinciden perfectamente, y reporta cero divergencias. **El
guard de coherencia es estructuralmente ciego a una divergencia display↔raw**: defiende "la lista
cuadra con la receta" donde *receta* significa la línea raw, no la que el usuario lee.

El arreglo no necesita el display: **la línea raw trae dentro la prueba de que está mal.** Su propio
paréntesis dice `(4 g)` y `_parse_quantity` lo descartaba. Cuando la cantidad parseada convierte a
algo ≥5× distinto de lo que la línea declara pesar, gana la pista.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

import shopping_calculator as sc

_BACKEND = Path(__file__).resolve().parents[1]
_SC = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")


# ───────────────────────────── el caso vivo ─────────────────────────────

def test_el_caso_vivo_30_cdas_cae_a_la_pista():
    qty, unit, name = sc._parse_quantity("30 cdas de mantequilla de maní (4 g)")
    assert (qty, unit) == (4.0, "g"), (
        f"la línea que compró 10 potes debe resolverse por su pista de gramos, dio {qty} {unit}")
    assert "maní" in name.lower()


def test_la_linea_del_display_no_se_toca():
    """¼ cda ≈ 3,75 g contra una pista de 4 g: coherente. Si este test cae, el guard está
    reescribiendo líneas sanas."""
    assert sc._parse_quantity("¼ cda de mantequilla de maní (4 g)")[:2] == (0.25, "cda")


def test_el_guard_avisa_cuando_actua(caplog):
    with caplog.at_level(logging.WARNING):
        sc._parse_quantity("30 cdas de mantequilla de maní (4 g)")
    msgs = [r.getMessage() for r in caplog.records if "P1-GRAM-HINT-TRUMPS-QTY" in r.getMessage()]
    assert msgs, "actuar en silencio sobre una cantidad es tan malo como no actuar"
    assert "112" in msgs[0] or "×" in msgs[0]


# ─────────────────── lo que NO se puede tocar (falsos positivos) ───────────────────

@pytest.mark.parametrize("linea,qty,unit", [
    # La conversión gruesa (1 taza = 240 g) NO acierta en densidad, y por eso el umbral es 5×.
    ("2¼ tazas de fresas congeladas (338g)", 2.25, "taza"),      # 540 vs 338 = 1,6×
    ("1 taza de yogurt griego natural sin azúcar (198g)", 1.0, "taza"),   # 240 vs 198
    ("¼ taza de aguacate (39g, lavadas y cortadas en cuartos)", 0.25, "taza"),  # 60 vs 39
    ("⅓ taza de leche descremada (105ml)", 1 / 3, "taza"),       # sin pista en GRAMOS
    ("½ guineo (102g, pelado y cortado en cubos)", 0.5, "unidad"),   # unidad no convertible
    ("1 papa mediana (149g, lavadas)", 1.0, "unidad"),
    ("2 tazas de lechuga", 2.0, "taza"),                          # sin pista
    ("50 g de pierna de cerdo magra (sin grasa visible)", 50.0, "g"),
    # `hoja` es su unidad real (no `unidad`), y no está en la tabla de conversión gruesa: sin
    # veredicto posible, se deja como está. El paréntesis además no lleva número.
    ("3 hojas de menta fresca (picadas)", 3.0, "hoja"),
])
def test_lineas_sanas_intactas(linea, qty, unit):
    got_qty, got_unit, _ = sc._parse_quantity(linea)
    assert got_unit == unit, f"{linea!r}: unidad {got_unit!r}, esperaba {unit!r}"
    assert got_qty == pytest.approx(qty, rel=1e-6), f"{linea!r}: qty {got_qty}, esperaba {qty}"


def test_una_pista_de_volumen_no_cuenta_como_peso():
    """'(105ml)' no es una declaración de PESO. Tomarla como gramos metería un error de densidad
    por la puerta de atrás."""
    assert sc._parse_quantity("40 cdas de leche descremada (105ml)")[:2] == (40.0, "cda")


def test_unidad_no_convertible_nunca_dispara():
    """Sobre 'unidad'/'lata'/'pote' no hay conversión honesta a gramos, así que no hay veredicto:
    inventar uno sería peor que no mirar."""
    for linea in ("6 unidades de huevo (50 g)", "3 latas de atún (170 g)"):
        q, u, _ = sc._parse_quantity(linea)
        assert u not in ("g",), f"{linea!r} no debe resolverse por la pista (dio {q} {u})"


# ───────────────────────────── knobs y anclaje ─────────────────────────────

def test_knob_de_rollback(monkeypatch):
    monkeypatch.setattr(sc, "_HINT_TRUMPS_QTY", False)
    assert sc._parse_quantity("30 cdas de mantequilla de maní (4 g)")[:2] == (30.0, "cda")


def test_umbral_configurable(monkeypatch):
    """Con el umbral en 100× el caso vivo (112×) sigue disparando; en 200× ya no."""
    monkeypatch.setattr(sc, "_HINT_TRUMPS_RATIO", 100.0)
    assert sc._parse_quantity("30 cdas de mantequilla de maní (4 g)")[1] == "g"
    monkeypatch.setattr(sc, "_HINT_TRUMPS_RATIO", 200.0)
    assert sc._parse_quantity("30 cdas de mantequilla de maní (4 g)")[1] == "cda"


def test_knobs_registrados():
    from knobs import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()
    for k in ("MEALFIT_GRAM_HINT_TRUMPS_QTY", "MEALFIT_GRAM_HINT_TRUMPS_RATIO"):
        assert k in snap


def test_corre_dentro_del_parser_no_del_agregador():
    """⚠️ Load-bearing: `expected_sum_from_recipes` parsea las MISMAS líneas para el guard de
    coherencia. Si esto se moviera al agregador, los dos lados discreparían y el guard bloquearía
    el plan por culpa del arreglo — el modo de fallo que cerró P1-COHERENCE-INF-NOT-SEVERE."""
    i = _SC.index("def _parse_quantity(")
    body = _SC[i:_SC.index("\ndef ", i + 10)]
    assert "_reconcile_qty_with_gram_hint(s, qty, unit_str)" in body, (
        "la reconciliación salió del parser — el guard de coherencia dejaría de ver el mismo valor")
    i_agg = _SC.index("def aggregate_and_deduct_shopping_list")
    assert "_reconcile_qty_with_gram_hint" not in _SC[i_agg:i_agg + 6000], (
        "no duplicar la reconciliación en el agregador: una sola fuente")


def test_el_guard_de_coherencia_sigue_leyendo_raw():
    """Ancla del hallazgo, no del arreglo: mientras `expected_sum_from_recipes` lea `ingredients_raw`
    en su lado izquierdo, seguirá siendo CIEGO a una divergencia display↔raw. Este guard cierra el
    daño (la cantidad absurda), NO la ceguera. Si algún día el guard pasa a comparar display contra
    raw, este test cae y toca revisitar la nota."""
    i = _SC.index("def expected_sum_from_recipes")
    body = _SC[i:_SC.index("\ndef ", i + 10)]
    assert 'meal.get("ingredients_raw") or meal.get("ingredients")' in body
