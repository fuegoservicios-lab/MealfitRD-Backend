"""[P1-PANTRY-PACKAGE-GRAMS · 2026-09-04] La Nevera en envases se descuenta con recetas en gramos.

Primer «Me lo comí» real del dueño (04-09, 14:39 UTC): de 14 ingredientes, 10 acabaron en
`[P2-DEDUCT-NOOP] unidad incompatible` — la Nevera tenía «2 paquete de Calamar», «1 pote de
Yogurt», «1 sobre de Pimienta», «1 mazo de Cilantro» (el restock copia la presentación comercial
de la lista) y las recetas piden 145 g, 1 taza, al gusto, 1 cda. El aviso decía «descontamos 3».

Tres cierres, todos en db_inventory:
  1. `convert_amount_container`: envase ↔ gramos con `market_packages` del maestro (unidad igual →
     única entrada → la más pequeña) o `density_g_per_unit` cuando el envase es la unidad por
     defecto. Usado en los DOS sitios de deducción (reservas y consumo).
  2. El restock guarda GRAMOS cuando la lista trae `package_grams` (el envase exacto que se compró).
  3. «al gusto» / «pizca» no se infieren (50 g de pimienta) ni se descuentan: `unquantified`.
"""
from __future__ import annotations

from pathlib import Path

import db_inventory as inv

_BACKEND = Path(__file__).resolve().parents[1]

CALAMAR = {"name": "Calamar", "default_unit": "lb", "density_g_per_cup": 145,
           "market_packages": [{"unit": "paquete", "grams": 907.18, "label": "2 Lb", "price": 460.0}]}
YOGURT = {"name": "Yogurt", "default_unit": "pote", "density_g_per_cup": 245,
          "market_packages": [{"grams": 150, "label": "150 g"}, {"grams": 1960, "label": "1.96 kg"}]}
CILANTRO = {"name": "Cilantro", "default_unit": "mazo", "density_g_per_unit": 50, "density_g_per_cup": 16, "market_packages": None}
SIN_DATO = {"name": "Misterio", "default_unit": "paquete", "market_packages": None}


def test_container_grams_resolution_order():
    assert inv._container_grams(CALAMAR, "paquete") == 907.18
    assert inv._container_grams(YOGURT, "pote") == 150          # dos tamaños sin unidad → el menor (conservador)
    assert inv._container_grams(CILANTRO, "mazo") == 50         # unidad por defecto → peso unitario
    assert inv._container_grams(CILANTRO, "paquete") is None
    assert inv._container_grams(SIN_DATO, "paquete") is None


def test_recipe_units_deduct_from_container_rows():
    # 145 g de calamar contra una fila en paquetes de 907 g
    assert round(inv.convert_amount_container(145, "g", "paquete", CALAMAR), 4) == round(145 / 907.18, 4)
    # 1 taza de yogurt (245 g) contra potes de 150 g
    assert round(inv.convert_amount_container(1, "taza", "pote", YOGURT), 3) == round(245 / 150, 3)
    # 1 cda de cilantro (16 g/taza → 1 g) contra mazos de 50 g
    assert round(inv.convert_amount_container(1, "cda", "mazo", CILANTRO), 4) == round((16 / 16) / 50, 4)
    # sin dato de envase sigue siendo None (no inventar)
    assert inv.convert_amount_container(100, "g", "paquete", SIN_DATO) is None
    # y lo que convert_amount ya sabía no cambia
    assert inv.convert_amount_container(1, "lb", "g", CALAMAR) == 453.592


HUEVO = {"name": "Huevo", "default_unit": "cartón", "density_g_per_unit": 50, "density_g_per_cup": 243,
         "market_packages": [{"label": "cartón 20 uds", "units": 20}, {"label": "cartón 30 uds", "units": 30}]}
PECHUGA = {"name": "Pechuga de pollo", "default_unit": "lb", "density_g_per_unit": 170, "market_packages": None}
LECHE = {"name": "Leche descremada", "default_unit": "litro", "density_g_per_cup": 244, "market_packages": None}
ARROZ = {"name": "Arroz blanco", "default_unit": "lb", "market_packages": [{"label": "Selecto 1 Lb", "price": 45}]}


def test_second_pass_counts_labels_and_default_unit_fallbacks():
    # «cartón (30 uds.)» guardado por el restock + receta «2 huevos» → 2/30 cartones
    assert round(inv.convert_amount_container(2, "unidad", "cartón (30 uds.)", HUEVO), 4) == round(2 / 30, 4)
    # sin conteo en la unidad: units de market_packages (el menor, 20)
    assert round(inv.convert_amount_container(2, "unidad", "paquete", HUEVO), 4) == round(2 / 20, 4)
    # etiqueta con tamaño y sin grams: «Selecto 1 Lb» → 453.6 g
    assert round(inv.convert_amount_container(60, "g", "paquete", ARROZ), 4) == round(60 / 453.592, 4)
    # sin datos de envase pero el alimento se compra por defecto en lb / litro: 1 envase = 1 lb / 1 L
    assert round(inv.convert_amount_container(150, "g", "paquete", PECHUGA), 4) == round(150 / 453.592, 4)
    assert round(inv.convert_amount_container(1, "taza", "paquete", LECHE), 4) == round(244 / 1000, 4)  # taza → g por densidad (244 g), 1 paquete = 1 L
    # un diente es una sub-pieza: NO cuenta contra «(4 uds.)» (sería 1/4 del paquete por diente) y sin
    # gramos por envase no se inventa → None
    AJO = {"name": "Ajo", "default_unit": "cabeza", "density_g_per_unit": 5, "market_packages": [{"label": "4 cabezas", "units": 4}]}
    assert inv.convert_amount_container(1, "diente", "paquete (4 uds.)", AJO) is None
    assert inv.convert_amount_container(1, "cabeza", "paquete (4 uds.)", AJO) == 0.25
    assert inv._grams_from_label("1.96 kg") == 1960.0 and inv._grams_from_label("250 ml") == 250.0 and inv._grams_from_label("Wala") is None
    assert inv._split_container_unit("paquete (4 uds.)") == ("paquete", 4)
    assert inv._split_container_unit("cartón (30 uds.)") == ("paquete", 30)
    assert inv._split_container_unit("pote") == ("pote", None)


LECHUGA = {"name": "Lechuga", "default_unit": "cabeza", "density_g_per_unit": 400, "density_g_per_cup": 36, "market_packages": None}


def test_volume_to_piece_goes_through_grams_and_garnish_is_unquantified(monkeypatch):
    # cena del dueño: «1½ tazas de lechuga» contra «1 cabeza» → 1.5 × 36 / 400
    assert round(inv.convert_amount_container(1.5, "taza", "cabeza", LECHUGA), 4) == round(1.5 * 36 / 400, 4)
    # «1 ramita de cilantro» es adorno: no infiere, no falla, no mueve el mazo
    monkeypatch.setattr(inv, "_db_available", lambda: True)
    monkeypatch.setattr(inv, "execute_sql_query", lambda *a, **k: [])
    monkeypatch.setattr(inv, "find_pantry_rows_for_name", lambda user_id, name, prefetched_rows=None: ([], None))
    monkeypatch.setattr(inv, "_persist_failed_inventory_deductions", lambda *a, **k: None)
    monkeypatch.setattr(inv, "_persist_consumption_events", lambda *a, **k: None)
    out = inv.deduct_consumed_meal_from_inventory("u", ["1 ramita de cilantro 1 cucharada"], source="plan_meal")
    assert out["unquantified"] == ["1 ramita de cilantro 1 cucharada"] and out["failed_to_deduct"] == []


def test_both_deduction_sites_use_the_container_aware_converter():
    src = (_BACKEND / "db_inventory.py").read_text(encoding="utf-8")
    # reserva, liberación de reserva y consumo: los tres sitios donde una receta se resta de una fila
    assert src.count("convert_amount_container(quantity, unit, current_unit, master_item") == 3


def test_restock_stores_exact_grams_when_the_list_knows_the_package(monkeypatch):
    calls = []
    monkeypatch.setattr(inv, "_db_available", lambda: True)
    monkeypatch.setattr(inv, "add_or_update_inventory_item", lambda user_id, name, qty, unit, **kw: calls.append((name, qty, unit)) or True)
    ok, names = inv.restock_inventory("u", [
        {"name": "Calamar", "quantity": 2, "unit": "paquete", "package_grams": 907.18},
        {"name": "Tomate", "quantity": 1.5, "unit": "lb"},                       # masa: intacto
        {"name": "Cilantro", "quantity": 1, "unit": "mazo"},                     # sin package_grams: intacto (lo cubre el conversor)
    ])
    assert ok and names == ["Calamar", "Tomate", "Cilantro"]
    assert calls[0] == ("Calamar", round(2 * 907.18, 2), "g")
    assert calls[1] == ("Tomate", 1.5, "lb")
    assert calls[2] == ("Cilantro", 1, "mazo")


def test_al_gusto_lines_are_unquantified_not_inferred(monkeypatch):
    monkeypatch.setattr(inv, "_db_available", lambda: True)
    monkeypatch.setattr(inv, "execute_sql_query", lambda *a, **k: [])
    monkeypatch.setattr(inv, "find_pantry_rows_for_name", lambda user_id, name, prefetched_rows=None: ([], None))
    writes = []
    monkeypatch.setattr(inv, "_persist_failed_inventory_deductions", lambda *a, **k: writes.append(a))
    monkeypatch.setattr(inv, "_persist_consumption_events", lambda *a, **k: writes.append(a))
    out = inv.deduct_consumed_meal_from_inventory("u", ["Pimienta negra al gusto", "Sal al gusto", "150 g de Calamar"], source="plan_meal")
    assert out["unquantified"] == ["Pimienta negra al gusto", "Sal al gusto"]
    assert out["not_in_pantry"] == ["150 g de Calamar"]
    assert out["failed_to_deduct"] == []


def test_marker_present():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P1-PANTRY-PACKAGE-GRAMS · 2026-09-04" in app
