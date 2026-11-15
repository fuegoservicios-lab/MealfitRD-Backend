"""[P1-PLAN-LOTE-1024 · J5] «Este fue el desayuno y me comí 3 tacos»: la segunda comida no se pierde en el desayuno."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guarda_dos_comidas_un_mensaje import franja_de_la_foto, nueva_franja  # noqa: E402

J5 = ("Este fue el desayuno y me comí 3 tacos rellenos con carne molido y pollo desmenuzado, los tacos eran grande, "
      "no tengo foto pero hasta un aprox ahí")


def test_el_caso_vivo_manda_los_tacos_al_almuerzo():
    assert franja_de_la_foto(J5) == "desayuno"
    assert nueva_franja(J5, "desayuno", ["desayuno"], ["desayuno"]) == "almuerzo"


def test_salta_a_la_siguiente_franja_libre_y_si_no_hay_queda_merienda():
    assert nueva_franja(J5, "desayuno", ["desayuno"], ["desayuno", "almuerzo"]) == "cena"
    assert nueva_franja(J5, "desayuno", ["desayuno"], ["desayuno", "almuerzo", "cena"]) == "merienda"
    t = "esto es mi almuerzo y también me tomé un batido"
    assert nueva_franja(t, "almuerzo", ["almuerzo"], ["almuerzo"]) == "cena"


@pytest.mark.parametrize("texto,pedida,turno,hoy", [
    (J5, "desayuno", [], ["desayuno"]),                   # la de la foto aún no se registró: esta ES la foto
    (J5, "almuerzo", ["desayuno"], ["desayuno"]),          # el modelo ya puso otra franja
    ("Este fue el desayuno", "desayuno", ["desayuno"], ["desayuno"]),   # no cuenta otra comida
    ("me comí 3 tacos y un jugo", "desayuno", ["desayuno"], ["desayuno"]),   # sin deíctico sobre la foto
    ("esto también fue parte de la cena", "cena", ["cena"], ["cena"]),    # es una fusión pedida (lote 982), no 2 comidas
    ("", "desayuno", ["desayuno"], ["desayuno"]),
])
def test_no_toca_nada_fuera_del_caso(texto, pedida, turno, hoy):
    assert nueva_franja(texto, pedida, turno, hoy) is None


def test_esta_enganchada_antes_de_la_fusion_por_franja():
    s = (Path(__file__).resolve().parent.parent / "agent.py").read_text(encoding="utf-8")
    i = s.index("[P1-PLAN-LOTE-1024 · J5]")
    assert i < s.index("_fusion_misma_franja = False   # [P1-PLAN-LOTE-982]")
    bloque = s[i:i + 2200]
    assert "nueva_franja(_txt_dc, tool_args.get(\"meal_type\"), _franjas_turno, _franjas_hoy)" in bloque
    assert "_meals_created_this_turn(list(messages) + list(tool_messages))" in bloque
