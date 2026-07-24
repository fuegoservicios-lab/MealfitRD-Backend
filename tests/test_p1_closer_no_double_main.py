"""[P1-CLOSER-NO-DOUBLE-MAIN · 2026-07-24] El cerrador no puede pegar una SEGUNDA proteína
animal principal a un plato que ya tiene la suya.

Defecto en vivo (plan a060108b, revisión de recetas del owner 2026-07-24):

    "Costilla de Cerdo Guisada al Estilo Bowl Poke"
      45 g de Costilla de cerdo … + 175 g de atún en agua
      paso: "Escurre e incorpora atún en agua (ya viene cocido) a la preparación antes de servir"

    Un guiso de costilla de 30 minutos con una lata de atún volcada encima. El nombre
    ("Bowl Poke") parece inventado aguas abajo para justificar el atún. Mismo patrón, más
    suave, en el locrio de pescado que recibía 70 g de camarones "como proteína del plato"
    (el plato YA tenía pescado) y en la merienda de chinola con 10 g de mozzarella.

Por qué pasaba los gates existentes:
    `_close_protein_gap_for_meal` ya tiene coherencia para plato DULCE (P1-CLOSER-SWEET-GUARD),
    licuado/frío (`_NO_COOK_SAFE_PROTEIN_HINT`) y merienda (P2-CLOSER-SNACK-CAP), más un techo
    en GRAMOS (`CLOSER_BOLT_MAX_ADD_G`=180). El atún entró con 175 g — justo bajo el techo. El
    cap es de gramos, no de plato: nada preguntaba "¿este plato YA tiene proteína principal?".

Fix:
    Si el plato es salado y ya contiene una proteína animal principal (`_MEAT_PROTEIN_HINT` en
    sus ingredientes), el pool del cerrador excluye las demás proteínas animales principales.
    Quedan los extensores que un cocinero SÍ añadiría (legumbres, huevo, queso). Si no queda
    nada compatible → return 0 y el piso de proteína se cubre en otra comida (misma degradación
    honesta que el guard dulce, que es su hermano directo).

    Escalar la proteína que ya está sigue teniendo prioridad (P3-PROTEIN-CLOSER-SCALE-FIRST
    corre antes); este guard solo acota el *fallback* de pegar una nueva.
"""
from __future__ import annotations

import graph_orchestrator as g
from constants import strip_accents as _sa


class _Info:
    def __init__(self, name, protein, carbs=2.0, fats=1.0, kcal=95.0):
        self.name, self.protein, self.carbs, self.fats, self.kcal = name, protein, carbs, fats, kcal


# Ordenados por "leanness" como el pool real: atún primero (lo que eligió en vivo).
_CANDS_ANIMAL = [(0.10, "Atún en agua", _Info("Atún en agua", 26)),
                 (0.20, "Camarones", _Info("Camarones", 24)),
                 (0.25, "Pechuga de pollo", _Info("Pechuga de pollo", 31))]

_CANDS_MIXTO = _CANDS_ANIMAL + [(0.40, "Habichuelas rojas", _Info("Habichuelas rojas", 9, carbs=20, kcal=130))]


def _meal_with(ings, name="Plato Salado"):
    return {"name": name, "protein": 6, "carbs": 25, "fats": 5, "cals": 220,
            "ingredients": list(ings)}


# ---------------------------------------------------------------------------
# 1. El detector
# ---------------------------------------------------------------------------
def test_detecta_proteina_principal_ya_presente():
    casos = [
        ["45 g de Costilla de cerdo", "¼ taza de arroz blanco", "½ Plátano maduro"],
        ["½ filete de pescado blanco (101g)", "¼ taza de arroz blanco"],
        ["½ pechuga de pollo", "2 tomates"],
    ]
    for ings in casos:
        assert g._meal_has_main_animal_protein(_meal_with(ings), _sa) is True, ings


def test_no_confunde_guarniciones_ni_lacteos():
    casos = [
        ["2 tazas de lechosa (cubos)", "1 taza de Queso cottage"],   # lácteo, no es main animal
        ["¾ taza de avena", "¼ taza de leche"],
        ["½ taza de Guisantes secos", "1 casabe", "1 pepino"],       # legumbre
        ["2 papas grandes", "1 rebanada de pan integral"],
    ]
    for ings in casos:
        assert g._meal_has_main_animal_protein(_meal_with(ings), _sa) is False, ings


def test_no_matchea_dentro_de_otra_palabra():
    """`"pollo"` vive dentro de `"re-POLLO"` y `"res"` dentro de `"resto"`. Sin frontera de
    palabra, media ensalada pasa por "plato con proteína principal" y el guard bloquea el
    cierre de proteína de comidas que sí lo necesitan (lo cazó el caso de control de abajo)."""
    for ings in (
        ["1 taza de Repollo morado (rallado)", "½ zanahoria"],
        ["½ taza de Repollo morado rallado"],
        ["1 cda del resto del aceite de oliva"],
    ):
        assert g._meal_has_main_animal_protein(_meal_with(ings), _sa) is False, ings
    # …y los plurales legítimos SÍ cuentan.
    for ings in (["2 filetes de pescado"], ["pechugas de pollo"], ["camarones cocidos"]):
        assert g._meal_has_main_animal_protein(_meal_with(ings), _sa) is True, ings


# ---------------------------------------------------------------------------
# 2. El guard (el caso reportado)
# ---------------------------------------------------------------------------
def test_no_pega_atun_sobre_costilla_de_cerdo():
    """El caso exacto del plan a060108b."""
    meal = _meal_with(["45 g de Costilla de cerdo", "¼ taza de arroz blanco", "½ Plátano maduro"],
                      name="Costilla de Cerdo Guisada")
    added = g._close_protein_gap_for_meal(meal, 40, None, _CANDS_ANIMAL)
    assert added == 0, "no se pega una segunda proteína animal a un plato que ya tiene la suya"
    joined = " ".join(str(i).lower() for i in meal["ingredients"])
    assert "atun" not in _sa(joined) and "camaron" not in _sa(joined) and "pollo" not in joined


def test_no_pega_camarones_sobre_locrio_de_pescado():
    meal = _meal_with(["½ filete de pescado blanco (101g)", "¼ taza de arroz blanco", "½ cebolla"],
                      name="Locrio de Pescado Blanco")
    added = g._close_protein_gap_for_meal(meal, 40, None, _CANDS_ANIMAL)
    assert added == 0
    assert "camaron" not in _sa(" ".join(str(i).lower() for i in meal["ingredients"]))


def test_si_hay_extensor_compatible_lo_usa_en_vez_de_abortar():
    """Degradar no es el objetivo: si hay una legumbre en el pool, cerrar el piso con ELLA
    (habichuelas con arroz y cerdo es un plato dominicano real)."""
    meal = _meal_with(["45 g de Costilla de cerdo", "¼ taza de arroz blanco"],
                      name="Costilla Guisada con Arroz")
    added = g._close_protein_gap_for_meal(meal, 40, None, _CANDS_MIXTO)
    assert added > 0, "con una legumbre disponible SÍ debe cerrar el piso"
    joined = _sa(" ".join(str(i).lower() for i in meal["ingredients"]))
    assert "habichuela" in joined
    assert "atun" not in joined, "sigue sin pegar la segunda proteína animal"


# ---------------------------------------------------------------------------
# 3. No romper lo que ya funcionaba
# ---------------------------------------------------------------------------
def test_plato_sin_proteina_principal_sigue_cerrando():
    """Regresión del comportamiento previo: una base vegetal/almidón SÍ recibe proteína animal."""
    meal = _meal_with(["2 papas grandes", "1 taza de Repollo morado", "½ zanahoria"],
                      name="Papas Asadas con Ensalada")
    added = g._close_protein_gap_for_meal(meal, 30, None, _CANDS_ANIMAL)
    assert added > 0, "sin proteína principal el cerrador debe seguir cerrando el piso"


def test_knob_permite_rollback(monkeypatch):
    monkeypatch.setattr(g, "CLOSER_NO_DOUBLE_MAIN_ENABLED", False)
    meal = _meal_with(["45 g de Costilla de cerdo", "¼ taza de arroz blanco"])
    added = g._close_protein_gap_for_meal(meal, 40, None, _CANDS_ANIMAL)
    assert added > 0, "con el knob OFF vuelve al comportamiento anterior"


def test_marker_y_knob_registrados():
    import pathlib
    src = pathlib.Path(g.__file__).with_suffix(".py").read_text(encoding="utf-8", errors="replace")
    assert "[P1-CLOSER-NO-DOUBLE-MAIN · 2026-07-24]" in src
    assert 'MEALFIT_CLOSER_NO_DOUBLE_MAIN' in src
