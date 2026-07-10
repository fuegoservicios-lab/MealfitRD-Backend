"""[P2-CLOSER-SNACK-CAP + P2-DISPLAY-NAME-SPECIFICITY · 2026-07-05] Las "meriendas ensambladas"
(plan vivo 7e4e5570): el closer de proteína añadió 145-155g de cottage a platos de fruta
(macro-perfecto, culinariamente plano) y el display perdía el calificador del alimento
("145 g de queso" mientras raw decía "145g de queso cottage cocido" — compras BIEN, usuario
veía el genérico y la nota 💪 nombraba 'cottage').

Fix: (a) cap físico del añadido del closer en meriendas/platos ligeros
(MEALFIT_CLOSER_SNACK_MAX_ADD_G=100; el déficit se cubre en comidas fuertes — mismo racional
del calorie-aware); (b) `_restore_gram_name_specificity` en humanize: el display adopta el
nombre específico del raw cuando lo extiende (limpiando el sufijo ' cocido/a').
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_knob_default_100():
    m = re.search(r'CLOSER_SNACK_MAX_ADD_G\s*=\s*_env_int\(\s*"MEALFIT_CLOSER_SNACK_MAX_ADD_G"\s*,\s*(\d+)', _GO)
    assert m and m.group(1) == "100"


def test_cap_applied_on_light_meals_inside_closer():
    i = _GO.index("def _close_protein_gap_for_meal")
    # [P1-CLOSER-DAY-AWARE-PROTEIN · 2026-07-10] ventana 12000→16000: el chooser ganó el filtro
    # day-aware (~2k chars) y el cap de snack quedaba fuera de la ventana fija.
    body = _GO[i:i + 16000]
    i_cap = body.index("int(CLOSER_SNACK_MAX_ADD_G)")  # el CÓDIGO (el comment nombra el knob antes)
    win = body[max(0, i_cap - 600):i_cap]
    assert "if light:" in win, "el cap aplica SOLO a meriendas/platos ligeros (light ya computado)"
    assert "grams = min(grams, max_add_g)" in body[:i_cap], "el cap corre DESPUÉS del cap general"


# ───────────────────── name specificity (humanize) ─────────────────────

def _restore(d, r):
    from humanize_ingredients import _restore_gram_name_specificity
    return _restore_gram_name_specificity(d, r)


def test_generic_cheese_recovers_cottage():
    out = _restore(["145 g de queso"], ["145g de queso cottage cocido"])
    assert out == ["145 g de queso cottage"], "adopta el específico y limpia ' cocido'"


def test_non_matching_lines_untouched():
    d = ["2 taza de Lechosa", "150 g de pollo", "80 g de arroz"]
    r = ["2 taza de Lechosa", "150g de pechuga de pollo", "90g de arroz"]
    out = _restore(d, r)
    assert out[0] == d[0], "líneas sin lead-gramos intactas"
    assert out[1] == d[1], \
        "'pechuga de pollo' NO empieza con 'pollo' → sin confianza de prefijo → intacta"
    assert out[2] == d[2], "cantidades distintas → jamás tocar"


def test_prefix_rule_is_strict():
    # raw no empieza con el food del display → no hay confianza para reescribir.
    out = _restore(["100 g de queso"], ["100g de mozzarella fresca"])
    assert out == ["100 g de queso"]


def test_misaligned_lists_untouched():
    out = _restore(["100 g de queso"], ["100g de queso cottage", "extra"])
    assert out == ["100 g de queso"]
