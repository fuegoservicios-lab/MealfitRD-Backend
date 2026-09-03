"""[P2-DISPLAY-POP-VECINO · 2026-08-21] El pop de `_display` colgaba de siete funciones
con nombre, no del acto de reescribir.

La invariante es simple: si los gramos de un meal cambian, su `_display[locale]` —que
espeja `ingredients` y `recipe` POR ÍNDICE— pasa a mentir, y hay que tirarlo. Once
`pop("_display")` repartidos por `graph_orchestrator.py`, `routers/plans.py` y `tools.py`
la implementaban a mano, cada uno en su función.

MEDIDO: **seis** re-escritores plan-wide reescriben en sitio SIN pop. Y el séptimo que
alguien escriba mañana nacerá mintiendo por omisión, porque la invariante no vive en
ningún sitio que él vaya a tocar — vive en los nombres de sus vecinos.

EL ARREGLO: atarla al ACTO. `update_plan_data_atomic` es el cuello de botella por el que
pasa toda escritura de `plan_data` bajo lock (invariante I7), así que compara la huella de
cada meal antes y después del mutator y popea lo que cambió. Un re-escritor nuevo queda
cubierto sin wiring, que es exactamente la lección de `P1-COUNTRY-SYSTEM-F1`: gatear call
sites uno a uno es el agujero, no el cierre.

POR QUÉ NO SUSTITUYE A LOS ONCE POPS: siguen siendo correctos y más baratos —popean sin
recorrer el plan— y algunos corren fuera de este persist. Esto es la red de debajo, no su
reemplazo. Un pop de más es gratis: el peor caso es re-traducir un meal.

LO QUE NO SE TOCA: el contrato de pureza del mutator (`P2-MUTATOR-PURITY`). El barrido es
CPU puro sobre el dict, sin IO ni re-entrada al pool — corre dentro del `FOR UPDATE`.
"""

from __future__ import annotations

import importlib
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MARKER = "P2-DISPLAY-POP-VECINO"


@pytest.fixture()
def db():
    import db_plans

    return importlib.reload(db_plans)


def _plan(**kw) -> dict:
    meal = {
        "name": "Pollo guisado",
        "ingredients": ["180 g de Pechuga de pollo", "100 g de Arroz blanco"],
        "recipe": ["Mise en place: pica la cebolla.", "Cocina 20 min."],
        "_display": {"en-US": {
            "name": "Stewed chicken",
            "ingredients": ["180 g chicken breast (Pechuga de pollo)",
                            "100 g white rice (Arroz blanco)"],
            "recipe": ["Mise en place: chop the onion.", "Cook 20 min."],
        }},
    }
    meal.update(kw)
    # `deepcopy` y no `dict(meal, …)`: la copia superficial comparte la MISMA lista de
    # `ingredients` entre los dos platos, asi que tocar el primero cambia la huella del
    # segundo y el test del vecino falla acusando al barrido de algo que hizo el fixture.
    # Es la misma leccion que ya esta escrita en `_fingerprint_lines`: la copia es
    # load-bearing, no cosmetica.
    otro = deepcopy(meal)
    otro["name"] = "Otro plato"
    return {"days": [{"meals": [meal, otro]}]}


# ============================================================
# 1 · El barrido existe y es puro
# ============================================================

def test_existe_el_barrido(db) -> None:
    assert hasattr(db, "_popear_display_de_lo_que_cambio"), (
        f"No existe el barrido. La invariante sigue colgando de siete funciones con "
        f"nombre, y el octavo re-escritor nacerá mintiendo por omisión. [{_MARKER}]"
    )


def test_el_barrido_popea_el_meal_que_cambio(db) -> None:
    antes = db._huellas_de_meals(_plan())
    plan = _plan()
    plan["days"][0]["meals"][0]["ingredients"][0] = "90 g de Pechuga de pollo"

    db._popear_display_de_lo_que_cambio(antes, plan)

    assert "_display" not in plan["days"][0]["meals"][0], (
        f"el meal re-cuantizado conserva su `_display`, que ahora miente los gramos: "
        f"la pantalla dice 180 g y el motor calcula sobre 90. [{_MARKER}]"
    )


def test_el_barrido_no_toca_al_vecino(db) -> None:
    """La otra mitad, y la que hace que esto no sea «popear todo por si acaso»: un pop de
    más cuesta una re-traducción, y con el plan entero cada persist la pagaría."""
    antes = db._huellas_de_meals(_plan())
    plan = _plan()
    plan["days"][0]["meals"][0]["ingredients"][0] = "90 g de Pechuga de pollo"

    db._popear_display_de_lo_que_cambio(antes, plan)

    assert "_display" in plan["days"][0]["meals"][1], (
        f"se popeó el `_display` del meal que NADIE tocó. [{_MARKER}]"
    )


@pytest.mark.parametrize(
    "cambio",
    [
        ("name", "Pollo al horno"),
        ("ingredients", ["180 g de Pechuga de pollo"]),          # una línea menos
        ("recipe", ["Mise en place: pica.", "Cocina.", "Sirve."]),  # una línea más
    ],
)
def test_los_tres_campos_espejados_disparan_el_pop(db, cambio) -> None:
    """`name`, `ingredients` y `recipe` son los tres que `_display` espeja. Un array que
    cambia de LONGITUD es el caso peor: el espejo es por índice, así que pinta el gramaje
    de un ingrediente junto al nombre de otro."""
    campo, valor = cambio
    antes = db._huellas_de_meals(_plan())
    plan = _plan()
    plan["days"][0]["meals"][0][campo] = valor

    db._popear_display_de_lo_que_cambio(antes, plan)
    assert "_display" not in plan["days"][0]["meals"][0], (
        f"cambiar `{campo}` no invalidó el `_display`. [{_MARKER}]"
    )


def test_un_persist_que_no_cambia_nada_no_popea(db) -> None:
    """MUTACIÓN DE CONTROL. Un barrido que popee siempre haría pasar todo lo de arriba y
    convertiría cada escritura de `plan_data` —hay muchas que no tocan los meals— en una
    re-traducción del plan entero."""
    antes = db._huellas_de_meals(_plan())
    plan = _plan()
    plan["_alguna_clave_de_control"] = 1     # una escritura que no toca meals

    db._popear_display_de_lo_que_cambio(antes, plan)
    for i in (0, 1):
        assert "_display" in plan["days"][0]["meals"][i], (
            f"se popeó sin que cambiara nada del meal {i}. [{_MARKER}]"
        )


def test_el_barrido_sobrevive_a_formas_que_no_son_las_esperadas(db) -> None:
    """Nunca puede tumbar un persist: corre dentro del `FOR UPDATE`."""
    for basura in (None, {}, {"days": None}, {"days": [None]}, {"days": [{"meals": "x"}]}):
        db._popear_display_de_lo_que_cambio(db._huellas_de_meals(basura), basura)
    assert True


# ============================================================
# 2 · Está cableado dentro del persist, y sigue siendo puro
# ============================================================

def test_el_persist_lo_invoca(db) -> None:
    import inspect

    fuente = inspect.getsource(db.update_plan_data_atomic)
    assert "_huellas_de_meals" in fuente and "_popear_display_de_lo_que_cambio" in fuente, (
        f"`update_plan_data_atomic` no toma la huella antes del mutator ni barre después. "
        f"Es el cuello de botella por el que pasa toda escritura de `plan_data` bajo lock "
        f"(invariante I7): atarlo aquí cubre a los re-escritores futuros sin wiring. "
        f"[{_MARKER}]"
    )
    i_huella = fuente.index("_huellas_de_meals")
    i_mut = fuente.index("result = mutator(current)")
    i_pop = fuente.index("_popear_display_de_lo_que_cambio")
    assert i_huella < i_mut < i_pop, (
        f"el orden está mal: la huella tiene que tomarse ANTES del mutator y el barrido "
        f"DESPUÉS. [{_MARKER}]"
    )


def test_el_barrido_no_hace_io(db) -> None:
    """`P2-MUTATOR-PURITY`: esto corre dentro del `SELECT … FOR UPDATE`, reteniendo el
    row-lock y una conexión del pool. Una llamada a DB aquí es starvation."""
    import inspect

    fuente = inspect.getsource(db._popear_display_de_lo_que_cambio)
    for prohibido in ("connection_pool", "execute_sql", "cursor", "requests", "sleep"):
        assert prohibido not in fuente, (
            f"el barrido menciona `{prohibido}`: corre dentro del FOR UPDATE y tiene que "
            f"ser CPU puro. [{_MARKER}]"
        )
