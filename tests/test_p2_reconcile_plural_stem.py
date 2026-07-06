"""[P2-RECONCILE-PLURAL-STEM · 2026-07-06] Reconciliador display→raw sin duplicados por plural.

Validación integral del plan renovado cd4ae3c3 (monitoreo en vivo): boundary
reportó raw_misalign=9 con dirección INVERTIDA (raw MÁS largo que display) y el
forense mostró el actor: `_reconcile_display_missing_in_raw` apendeaba al raw
formas humanizadas que no matcheó contra sus gemelas crudas — el display viene
en PLURAL ('4 tortillas integrales', '1½ pepinos') y el raw crudo en singular
('4 tortilla integral', '1.5 pepino'); el matcher 'tortillas(?:s|es)?' JAMÁS
matchea 'tortilla' → duplicado → pepino/tortillas/claras contados DOBLE en la
lista de compras. Fix: candidatos singularizados + sufijo (es|e|s) — ambas
direcciones cubiertas. La otra mitad del misalign (pases que QUITAN líneas del
display sin tocar raw: sal/ajo del autofix de sodio) es el P1 crónico abierto
ya documentado en el propio boundary.
"""
import pytest

import graph_orchestrator as go


def _meal(ings, raw):
    return {"days": [{"day": 1, "meals": [
        {"name": "Wrap Test", "ingredients": list(ings), "ingredients_raw": list(raw),
         "recipe": ["Montaje: sirve."]},
    ]}]}


def test_plural_display_matches_singular_raw_no_duplicate():
    plan = _meal(
        ["4 tortillas integrales", "1½ pepinos", "3 claras de huevo"],
        ["4 tortilla integral", "1.5 pepino", "3 clara de huevo"],
    )
    days = plan["days"]
    added = go._reconcile_display_missing_in_raw(days)
    raw = days[0]["meals"][0]["ingredients_raw"]
    assert added == 0, f"plural↔singular es el MISMO alimento — nada que añadir: {raw}"
    assert len(raw) == 3, f"raw sin duplicados (pepino se contaba DOBLE en compras): {raw}"


def test_es_plural_direction_both_ways():
    plan = _meal(["2 limones", "3 tomates"], ["2 limón", "3 tomate"])
    days = plan["days"]
    assert go._reconcile_display_missing_in_raw(days) == 0
    assert len(days[0]["meals"][0]["ingredients_raw"]) == 2


def test_truly_missing_still_appended():
    # El caso original del fix (plan 55846e5e): miel SOLO en display → gana
    # visibilidad en compras/medidores.
    plan = _meal(
        ["8¾ cdta de miel", "4 tortillas integrales"],
        ["4 tortilla integral"],
    )
    days = plan["days"]
    added = go._reconcile_display_missing_in_raw(days)
    raw = days[0]["meals"][0]["ingredients_raw"]
    assert added == 1, "la línea genuinamente ausente SIGUE ganando visibilidad"
    assert any("miel" in str(r).lower() for r in raw)
    assert sum("tortilla" in str(r).lower() for r in raw) == 1, "sin duplicar la tortilla"


def test_bounded_stem_preserved():
    # agua ⊄ aguacate (P2-STEM-BOUNDED sigue vivo tras el fix).
    plan = _meal(["¼ taza de agua"], ["1 aguacate"])
    days = plan["days"]
    added = go._reconcile_display_missing_in_raw(days)
    raw = days[0]["meals"][0]["ingredients_raw"]
    assert added == 1 and any("agua" == str(r).split()[-1].lower() or "de agua" in str(r).lower() for r in raw), (
        f"'agua' NO está cubierta por 'aguacate' — debe añadirse: {raw}"
    )
