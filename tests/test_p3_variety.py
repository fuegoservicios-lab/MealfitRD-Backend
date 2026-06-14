"""[P3-VARIETY · 2026-06-13] Variedad + pertinencia cultural es-DO + técnica (FS5).

Hallazgo HIGH de la auditoría: huevo en 6/12 comidas, 'cremoso' en 4/12, ricotta×3, mismo
plato-base (revoltillo) dos veces el mismo día, Tajín (no dominicano). La matriz de alimentos
YA está anclada en DD (DOMINICAN_PROTEINS/CARBS/VEGGIES/FRUITS) → FS5 es prompt (regla de
variedad+fidelidad cultural+técnica) + un reporte ADVISORY (observabilidad, NO gate → cero regen).

Cubre: (1) cap de huevo, (2) cap de 'cremoso', (3) cap premium, (4) plato-base repetido
intra-día, (5) plan limpio → ok, (6) las reglas de prompt presentes en preferences.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_orchestrator import build_variety_report


def _meal(name, ings=None):
    return {"name": name, "ingredients": ings or []}


def test_cap_de_huevo():
    # 5 comidas con huevo en un plan de 8 → supera el cap (~3).
    days = [{"day": 1, "meals": [
        _meal("Revoltillo", ["2 huevos"]), _meal("Tortilla", ["2 huevos"]),
        _meal("Ensalada con huevo", ["1 huevo"]), _meal("Pollo", ["pollo"])]},
        {"day": 2, "meals": [
        _meal("Huevos fritos", ["2 huevos"]), _meal("Avena", ["avena"]),
        _meal("Wrap de huevo", ["1 huevo"]), _meal("Pescado", ["pescado"])]}]
    rep = build_variety_report({"days": days})
    assert rep["egg_meals"] == 5
    assert any("Huevo en" in i for i in rep["issues"])


def test_cap_cremoso():
    days = [{"day": 1, "meals": [
        _meal("Avena Cremosa"), _meal("Batido Cremoso"), _meal("Puré Cremoso"), _meal("Pollo")]}]
    rep = build_variety_report({"days": days})
    assert rep["cremoso"] == 3
    assert any("cremoso" in i.lower() for i in rep["issues"])


def test_cap_premium():
    days = [{"day": 1, "meals": [
        _meal("Wrap de Ricotta", ["queso ricotta"]),
        _meal("Ensalada con ricotta", ["ricotta"]),
        _meal("Pasta", ["queso ricotta"]),
        _meal("Pollo", ["pollo"])]}]
    rep = build_variety_report({"days": days})
    assert rep["premium"] == 3
    assert any("premium" in i.lower() for i in rep["issues"])


def test_plato_base_repetido_mismo_dia():
    days = [{"day": 3, "meals": [
        _meal("Revoltillo de Huevos", ["huevos"]),  # desayuno
        _meal("Pollo Guisado", ["pollo"]),
        _meal("Fruta", ["mango"]),
        _meal("Revoltillo de Vegetales", ["vegetales"])]}]  # cena — repite revoltillo
    rep = build_variety_report({"days": days})
    assert rep["same_day_repeats"] >= 1
    assert any("revoltillo" in i.lower() for i in rep["issues"])


def test_plan_variado_es_ok():
    days = [{"day": 1, "meals": [
        _meal("Avena con Frutas", ["avena", "mango"]),
        _meal("Pollo a la Plancha con Arroz", ["pollo", "arroz"]),
        _meal("Yogur con Nueces", ["yogur"]),
        _meal("Pescado al Horno con Víveres", ["pescado", "yuca"])]}]
    rep = build_variety_report({"days": days})
    assert rep["ok"] is True
    assert rep["issues"] == []


def test_reglas_de_prompt_presentes_en_preferences():
    # Ancla las reglas de FS1/FS4/FS5 en el prompt: un renombre/borrado las haría fallar aquí.
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "prompts", "preferences.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "REGLA DE VARIEDAD Y FIDELIDAD CULTURAL" in src
    assert "REGLA DE SODIO" in src
    assert "REGLA DE SEGURIDAD ALIMENTARIA" in src
    assert "Tajín" in src  # prohibición explícita del ingrediente no-dominicano
    assert "cremoso" in src.lower()  # cap de técnica
