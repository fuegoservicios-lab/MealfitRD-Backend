# -*- coding: utf-8 -*-
"""[ARQ27-P1-09 · 2026-09-06] Conservación según estado, envase y equipo.

`pantry_durability` decidía por el ALIMENTO y su categoría, y le faltaban dos dimensiones que sí
cambian la logística:

**1. Un congelado de fábrica exige congelador.** Cuatro filas —Edamame, Papas ralladas, Wafles,
Bolitas de papa— estaban clasificadas `pantry` 90 con un comentario al lado que decía «congelados de
fábrica». O sea: la tabla afirmaba que un paquete de edamame aguanta tres meses en la alacena. En un
ciclo de una sola compra **sin congelador** pasaban el guard el día 30 y nadie avisaba. Ahora son
`frozen`: 1 día fuera, 365 dentro. El criterio de cierre del gap lo dice literal — «un SKU congelado
exige congelador incluso si se compró ya congelado».

**2. Una bebida estable cerrada no conserva el mismo horizonte después de abrirse.** La leche vegetal
era 365 días, abierta o cerrada. El mecanismo es el mismo que el de `fresh_state`
(P1-DURABILITY-FRESH-STATE): el calificativo del nombre manda sobre la tabla.

**Lo que este módulo NO hace, dicho en vez de fingido.** No deduce solo que un cartón abierto el día 1
ya no sirve el día 20 de la misma compra: eso exige saber en qué días se usa cada ingrediente, y
pertenece al modo «cocino por tandas». Y `Lentejas cocidas` sigue siendo despensa 180 **a propósito**
— el módulo responde «¿cuánto aguanta lo que el usuario COMPRA?», y lo que compra es lenteja seca; el
plato se cocina ese día. Tratar cada nombre cocinado como sobras de nevera bloquearía platos
correctos. Esa frontera ya estaba declarada antes de este gap y se respeta.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import pantry_durability as PD  # noqa: E402

CONGELADOS_DE_FABRICA = ["Edamame", "Papas ralladas", "Wafles", "Bolitas de papa"]


# ── 1. congelado de fábrica ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("nombre", CONGELADOS_DE_FABRICA)
def test_un_congelado_de_fabrica_no_es_despensa(nombre):
    d = PD.classify(nombre)
    assert d["cls"] == "frozen", f"{nombre} sigue siendo {d['cls']}"
    assert d["days_fresh"] <= 2, "fuera del congelador no aguanta un ciclo"
    assert d["days_frozen"] >= 180, "dentro del congelador sí aguanta el ciclo entero"


@pytest.mark.parametrize("nombre", CONGELADOS_DE_FABRICA)
def test_sin_congelador_el_dia_25_se_acusa(nombre):
    """El caso vivo: ciclo de 30 días, una sola compra, sin congelador."""
    assert PD.ingredient_issue_beyond_horizon(nombre, 24, allow_frozen=False) == "frozen_needs_freezer"


@pytest.mark.parametrize("nombre", CONGELADOS_DE_FABRICA)
def test_con_congelador_el_mismo_dia_pasa(nombre):
    """La otra mitad: con congelador el plato es perfectamente válido. Un guard que bloquea siempre no
    informa — bloquearía justo a quien puede permitírselo."""
    assert PD.ingredient_issue_beyond_horizon(nombre, 24, allow_frozen=True) is None


def test_el_codigo_es_propio_y_no_el_de_la_proteina_fresca():
    """El consejo NO es el mismo: a la proteína fresca se le ofrece una alternativa de despensa; a un
    congelado de fábrica hay que decirle que sin congelador ese plato no cabe en su compra única."""
    assert PD.ingredient_issue_beyond_horizon("Edamame", 24, False) == "frozen_needs_freezer"
    assert PD.ingredient_issue_beyond_horizon("Pechuga de pollo", 24, False) == "protein_beyond_freeze_window"
    assert PD.ingredient_issue_beyond_horizon("Lechuga", 24, False) == "fresh_beyond_horizon"


def test_el_plato_con_congelado_no_cabe_en_una_compra_unica_sin_congelador():
    """Lo que el selector consume: `durability_of` → `template_fits`."""
    d = PD.durability_of(["Edamame", "Arroz integral", "Ajo"])
    assert d["days_fresh_min"] == 1 and "frozen" in d["classes"]
    assert PD.template_fits(d["days_fresh_min"], d["days_with_freezer_min"], 25, allow_frozen=False) is False
    assert PD.template_fits(d["days_fresh_min"], d["days_with_freezer_min"], 25, allow_frozen=True) is True


def test_el_mensaje_del_issue_nombra_el_congelador():
    """Sin esto, un `frozen_needs_freezer` llegaría al prompt con el texto del fresco sin reposición y
    el modelo intentaría el arreglo equivocado."""
    fuente = (_BACKEND / "horizon.py").read_text(encoding="utf-8")
    assert "frozen_needs_freezer" in fuente
    assert "CONGELADO SIN CONGELADOR" in fuente


# ── 2. envase abierto ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cerrado,abierto", [
    ("Leche de soya", "Leche de soya abierta"),
    ("Leche de coco", "Leche de coco abierta"),
    ("Aceitunas", "Aceitunas abiertas"),
    ("Frijoles refritos", "Frijoles refritos abiertos"),
])
def test_el_envase_abierto_pierde_el_horizonte_del_cerrado(cerrado, abierto):
    c, a = PD.classify(cerrado), PD.classify(abierto)
    assert c["days_fresh"] >= 90, f"{cerrado} debería ser estable cerrado"
    assert a["rule"] == "opened_package" and a["cls"] == "cold"
    assert a["days_fresh"] < 30, f"{abierto} conservó {a['days_fresh']} días"


def test_sin_calificativo_nada_cambia():
    """El calificativo es la única señal. Sin él, la conducta anterior intacta."""
    assert PD.classify("Leche de soya")["rule"] == "leche de soya"
    assert PD.classify("Aceitunas")["cls"] == "pantry"


def test_un_calificativo_sobre_algo_que_no_es_conserva_no_hace_nada():
    """«Lechuga abierta» no significa nada: la lista es de alimentos estables SOLO por el envase."""
    d = PD.classify("Lechuga abierta")
    assert d["rule"] != "opened_package"


# ── frontera declarada: lo que NO se toca ─────────────────────────────────────────────────────
def test_lo_cocinado_sigue_resolviendo_por_lo_que_se_compra():
    """Decisión heredada de P1-DURABILITY-FRESH-STATE y respetada aquí: el módulo responde «¿cuánto
    aguanta lo que el usuario COMPRA?». Convertir cada nombre cocinado en sobras de nevera bloquearía
    platos correctos, y la conservación del plato ya hecho es otra pregunta."""
    assert PD.classify("Lentejas cocidas")["cls"] == "pantry"
    assert PD.classify("Arroz cocido")["cls"] == "pantry"


def test_las_cuatro_clases_anteriores_siguen_intactas():
    """Un cambio de clasificación que arrastrara a otros alimentos sería peor que el gap."""
    for nombre, cls in [("Lentejas", "pantry"), ("Huevo", "cold"), ("Pechuga de pollo", "freezable"),
                        ("Lechuga", "fresh"), ("Arroz blanco", "pantry"), ("Atun en agua", "pantry"),
                        ("Atun fresco", "freezable"), ("Tofu firme", "cold")]:
        assert PD.classify(nombre)["cls"] == cls, f"{nombre} cambió de clase"


def test_la_ventana_de_congelacion_no_cambio():
    assert PD.freeze_window_days("none", 30) == 0
    assert PD.freeze_window_days("limited", 30) == 14
    assert PD.freeze_window_days("full", 30) == 30
