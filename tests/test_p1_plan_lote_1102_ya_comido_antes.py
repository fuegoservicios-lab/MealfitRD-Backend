"""[P1-PLAN-LOTE-1102 · 2026-11-18] Lo comido antes de la foto suma también sus kcal y macros (no solo el nombre).

Batería J7 en real (gpt-6-luna): «5 panecillos…» con 560 kcal (la estimación de la foto para los 4 que se ven), y luego
correcciones que deshacían el extra. Guard determinista antes de ejecutar la herramienta.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import ya_comido_antes as y  # noqa: E402

SYS = ("…\n1. Plato: Plato con 4 panecillos (pan de agua) abiertos untados con mantequilla. Porción: 4 unidades "
       "medianas. (Estimación: Calorías: 560, Proteína: 16g, Carbohidratos: 84g, Grasas Saludables: 18g)\n…")


class Humano:
    type = "human"

    def __init__(self, content):
        self.content = content


MSGS = [Humano("desayuno: estos panecillos con mantequilla, y ya me había comido uno antes de la foto")]
ARGS = {"meal_name": "5 panecillos de pan de agua con mantequilla", "calories": 560, "protein": 16, "carbs": 84,
        "healthy_fats": 18, "ingredients": ["5 panecillos medianos de pan de agua (≈250 g)", "mantequilla (≈25 g)"]}


def test_cuantos_ya_se_habia_comido():
    assert y.extra_ya_comido("y ya me había comido uno antes de la foto") == 1
    assert y.extra_ya_comido("ya me comí dos") == 2
    assert y.extra_ya_comido("me comí 2 de ellos antes") == 2
    assert y.extra_ya_comido("eran 4 y ya me comí uno") is None, "lo comido está dentro de lo que se ve"
    assert y.extra_ya_comido("me comí 3 tacos rellenos con carne") is None, "J5: no es «antes de la foto»"


def test_escala_kcal_y_macros_por_total_entre_lo_visto():
    n = y.ajustar_args("log_consumed_meal", ARGS, MSGS, SYS)
    assert (n["calories"], n["protein"], n["carbs"], n["healthy_fats"]) == (700, 20, 105, 22)
    assert n["meal_name"] == ARGS["meal_name"] and ARGS["calories"] == 560, "no muta el original"


def test_no_toca_si_el_modelo_ya_movio_las_cifras_o_si_se_olvido_del_extra():
    assert y.ajustar_args("log_consumed_meal", {**ARGS, "calories": 700}, MSGS, SYS)["calories"] == 700
    olvido = {**ARGS, "meal_name": "4 panecillos de pan de agua"}
    assert y.ajustar_args("log_consumed_meal", olvido, MSGS, SYS)["calories"] == 560, "no escala a medias"


def test_sin_frase_sin_foto_o_con_varias_fotos_no_hace_nada():
    assert y.ajustar_args("log_consumed_meal", ARGS, [Humano("me comí esto")], SYS) is ARGS
    assert y.ajustar_args("log_consumed_meal", ARGS, MSGS, "sin estimación") is ARGS
    dos = SYS + "\n2. Plato: algo (Estimación: Calorías: 100, Proteína: 1g, Carbohidratos: 1g, Grasas Saludables: 1g)"
    assert y.ajustar_args("log_consumed_meal", ARGS, MSGS, dos) is ARGS
    assert y.ajustar_args("correct_consumed_meal", ARGS, MSGS, SYS) is ARGS


def test_la_nota_avisa_al_coach_para_que_no_lo_corrija():
    n = y.ajustar_args("log_consumed_meal", ARGS, MSGS, SYS)
    nota = y.nota_para_el_coach(ARGS, n)
    assert "560 → 700 kcal" in nota and "no lo corrijas" in nota
    assert y.nota_para_el_coach(ARGS, ARGS) == ""


def test_una_correccion_que_deshace_el_extra_no_se_ejecuta():
    vuelve = {"meal_id": "x", "meal_name": "4 panecillos", "calories": 700, "ingredients": ["5 panecillos (≈250 g)"]}
    assert "Sin cambios" in y.correccion_que_deshace("correct_consumed_meal", vuelve, MSGS, SYS)
    a_560 = {"meal_id": "x", "meal_name": "5 panecillos", "calories": 560}
    assert "Sin cambios" in y.correccion_que_deshace("correct_consumed_meal", a_560, MSGS, SYS)
    otra = {"meal_id": "x", "meal_type": "merienda"}
    assert y.correccion_que_deshace("correct_consumed_meal", otra, MSGS, SYS) == ""
    assert y.correccion_que_deshace("correct_consumed_meal", {"delete": True, "calories": 560}, MSGS, SYS) == ""


def test_esta_conectado_antes_de_ejecutar_la_herramienta():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert 'tool_args = _ya_comido.ajustar_args(tool_name, tool_args, messages, sys_prompt)' in src
    assert '__import__("coach_checkpoint_ligero").prompt_del_turno(state))' in src
