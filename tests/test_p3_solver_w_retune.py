"""[P3-SOLVER-W-RETUNE · 2026-08-04] (audit solver+seeder v7 · Task 17) Re-tune conjunto de los
CUATRO pesos `SOLVER_W_*` del LSQ + el helper que vuelve MEDIBLE lo que hasta ahora era un número
copiado a mano en un comentario.

El residuo lo declaraba `P1-SOLVER-KCAL-ROW-REDUNDANT` en el propio archivo: «con 0.1 la fila kcal
SIGUE dominando (share 81.7-84.0%) — los pesos declarados siguen sin ser los efectivos. Devolverles
su significado exige re-tunear los CUATRO simultáneamente contra un harness de comidas vivas».

Dos entregas:
  1. `effective_row_shares(A_rows, b, w)` — función PURA que computa `w·b²` normalizado, o sea el
     peso que cada fila EJERCE (no el que el operador escribe). El próximo que quiera tocar los
     pesos parte del dato, no del comentario.
  2. El re-tune medido (`scripts/retune_solver_weights.py`, harness offline sobre comidas vivas).

100% offline: catálogo inyectado a `IngredientNutritionDB`, cero DB, cero red.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import portion_solver as ps
from nutrition_db import IngredientNutritionDB


_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel: str) -> str:
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_PS = _read("portion_solver.py")

# Comida sintética con la forma que arma producción: 4 filas (kcal + 3 macros) y coeficientes
# por línea en unidades absolutas. Reusa el caso de `test_p1_solver_seeder_v4_batch` a propósito:
# los dos tests hablan del MISMO fenómeno y comparar sus números debe ser directo.
A_ROWS = [[247.0, 216.0, 80.0, 45.0],   # kcal por línea
          [46.5, 4.2, 1.0, 0.0],        # proteína
          [0.0, 47.0, 4.3, 0.0],        # carbos
          [5.4, 0.5, 7.3, 5.0]]         # grasa
B_ROW = [700.0, 45.0, 70.0, 25.0]
_MACROS = ("kcal", "protein", "carbs", "fats")


def _w_now() -> list:
    return [ps.SOLVER_W_KCAL, ps.SOLVER_W_PROTEIN, ps.SOLVER_W_CARBS, ps.SOLVER_W_FATS]


# ═══════════════════ 1 · el helper puro `effective_row_shares` ═══════════════════

def test_shares_son_w_por_b_al_cuadrado_normalizado():
    """El contrato exacto: `w_r·b_r²` normalizado. Es la fórmula que el comentario de
    P1-SOLVER-KCAL-ROW-REDUNDANT citaba en prosa; aquí queda ejecutable."""
    w = [0.1, 1.5, 1.1, 1.4]
    got = ps.effective_row_shares(A_ROWS, B_ROW, w)
    manual = [w[r] * B_ROW[r] ** 2 for r in range(4)]
    total = sum(manual)
    assert got == pytest.approx([m / total for m in manual])
    assert sum(got) == pytest.approx(1.0)


def test_una_fila_sin_ningun_portador_no_pesa():
    """`A_rows` NO es decorativo. Una fila con todos los coeficientes en cero es una CONSTANTE del
    objetivo (ninguna `x` la reduce) ⇒ influencia real 0, por grande que sea `w·b²`. Es el caso
    `no_carrier` de `_feasibility_report`: target de grasa contra un plato sin grasa. Contarla
    escondería quién manda de verdad."""
    A = [list(A_ROWS[0]), list(A_ROWS[1]), list(A_ROWS[2]), [0.0, 0.0, 0.0, 0.0]]
    shares = ps.effective_row_shares(A, B_ROW, [0.1, 1.5, 1.1, 1.4])
    assert shares[3] == 0.0
    assert sum(shares) == pytest.approx(1.0)
    # y el resto se re-normaliza entre las filas accionables
    assert shares[0] > 0 and shares[1] > 0 and shares[2] > 0


def test_largos_incoherentes_devuelven_vacio_sin_adivinar():
    assert ps.effective_row_shares(A_ROWS, B_ROW[:3], [0.1, 1.5, 1.1, 1.4]) == []
    assert ps.effective_row_shares(A_ROWS, B_ROW, [0.1, 1.5]) == []
    assert ps.effective_row_shares([], [], []) == []


def test_sin_ninguna_fila_accionable_devuelve_ceros_no_nan():
    ceros = [[0.0, 0.0], [0.0, 0.0]]
    assert ps.effective_row_shares(ceros, [500.0, 40.0], [1.0, 1.0]) == [0.0, 0.0]


def test_el_helper_es_puro():
    """No lee knobs, no muta estado: mismo input ⇒ mismo output, y los `SOLVER_W_*` del módulo
    quedan intactos (el helper es para MEDIR el objetivo, no para configurarlo)."""
    antes = _w_now()
    a = ps.effective_row_shares(A_ROWS, B_ROW, [0.1, 1.5, 1.1, 1.4])
    b = ps.effective_row_shares(A_ROWS, B_ROW, [0.1, 1.5, 1.1, 1.4])
    assert a == b
    assert _w_now() == antes


# ═══════════════════ 2 · caracterización del share con los defaults vigentes ═══════════════════

def test_caracterizacion_share_kcal_tras_el_retune():
    """ANCLA DE CARACTERIZACIÓN (medida, no aspiracional). En esta comida sintética la fila kcal se
    llevaba **98.4%** del objetivo con el default original `w=1.2`, **84.0%** con `w=0.1`, y tras el
    re-tune medido baja a **41.7%**. La grasa iba del 0.1% → 1.5% → **13.3%**: `SOLVER_W_FATS` era
    literalmente decorativo y ahora ejerce.

    Las cotas son tolerantes a propósito (no clavan el decimal): lo que anclan es el RÉGIMEN — la
    fila redundante dejó de mandar y ninguna fila quedó inerte. Si alguien revierte los pesos a la
    vecindad previa, esto falla antes de que el cambio llegue a producción.

    ⚠️ NO dice que los pesos declarados sean YA los efectivos: `w·b²` sigue sin ser `w` (igualar
    ambos exige normalizar las filas, REFUTADO por medición, −27.1 pp). Dice que los cuatro pesan el
    mismo ORDEN DE MAGNITUD, que es lo que vuelve accionable tocarlos."""
    shares = dict(zip(_MACROS, ps.effective_row_shares(A_ROWS, B_ROW, _w_now())))
    _pretty = {k: round(v, 3) for k, v in shares.items()}
    assert shares["kcal"] <= 0.50, (
        f"la fila kcal volvió a dominar el objetivo ({shares['kcal']:.1%}) — "
        f"régimen previo al re-tune. Shares: {_pretty}")
    assert shares["protein"] >= 0.25, f"la fila de proteína se diluyó: {_pretty}"
    # ninguna fila queda INERTE (la grasa estaba en 1.5% — un knob que no movía nada)
    assert min(shares.values()) >= 0.08, f"hay una fila prácticamente inerte: {_pretty}"
    # y la fila redundante ya no pesa un ORDEN DE MAGNITUD más que la clínicamente crítica
    assert shares["kcal"] / shares["protein"] <= 1.5, (
        f"kcal vuelve a pesar {shares['kcal'] / shares['protein']:.1f}× la proteína: {_pretty}")


def test_los_cuatro_defaults_son_el_punto_medido():
    """Los CUATRO defaults se movieron a la vez — que es el punto de la tarea: tunear uno solo deja
    a los otros tres siendo décimas. Si alguien los cambia, tiene que cambiar también la medición
    documentada en el comentario (y volver a correr el harness)."""
    assert 'SOLVER_W_KCAL = _envf("MEALFIT_SOLVER_W_KCAL", 0.02' in _PS
    assert 'SOLVER_W_PROTEIN = _envf("MEALFIT_SOLVER_W_PROTEIN", 4.0' in _PS
    assert 'SOLVER_W_CARBS = _envf("MEALFIT_SOLVER_W_CARBS", 0.5' in _PS
    assert 'SOLVER_W_FATS = _envf("MEALFIT_SOLVER_W_FATS", 5.0' in _PS
    assert (ps.SOLVER_W_KCAL, ps.SOLVER_W_PROTEIN, ps.SOLVER_W_CARBS,
            ps.SOLVER_W_FATS) == (0.02, 4.0, 0.5, 5.0)
    # los 4 siguen siendo knobs con validador (rollback sin redeploy al punto previo)
    for k in ("MEALFIT_SOLVER_W_KCAL", "MEALFIT_SOLVER_W_PROTEIN",
              "MEALFIT_SOLVER_W_CARBS", "MEALFIT_SOLVER_W_FATS"):
        assert f'_envf("{k}"' in _PS and "0.0 < v <= 10.0" in _PS


def test_los_cuatro_pesos_llevan_la_medicion_en_el_comentario():
    """Doc-first: el comentario de los knobs debe declarar CONTRA QUÉ se midió. Sin esto, el
    próximo audit vuelve a leer el ⚠️ viejo ('los pesos declarados siguen sin ser los efectivos')
    que este fix ya cerró, y re-abre trabajo hecho."""
    assert "P3-SOLVER-W-RETUNE" in _PS
    assert "349 comidas" in _PS, "el comentario no dice contra qué dataset se midió"
    assert "retune_solver_weights.py" in _PS, "el comentario no apunta al harness reproducible"
    # el aviso anti-normalización de filas (REFUTADO por medición) sigue vivo: no lo pisamos
    assert "NO \"arreglar\" esto normalizando las filas" in _PS


# ═══════════════════ 3 · los pesos MUEVEN el resultado (no son decorativos) ═══════════════════

_ROWS = [
    {"name": "Pechuga de pollo", "aliases": ["pollo", "pechuga"], "kcal_per_100g": 107.0,
     "protein_g_per_100g": 22.5, "carbs_g_per_100g": 0.0, "fats_g_per_100g": 1.93},
    {"name": "Arroz blanco", "aliases": ["arroz"], "kcal_per_100g": 360.0,
     "protein_g_per_100g": 6.6, "carbs_g_per_100g": 79.0, "fats_g_per_100g": 0.6},
    {"name": "Aceite de oliva", "aliases": ["aceite"], "kcal_per_100g": 900.0,
     "protein_g_per_100g": 0.0, "carbs_g_per_100g": 0.0, "fats_g_per_100g": 100.0},
]


@pytest.fixture
def db():
    return IngredientNutritionDB(rows=_ROWS)


# Comida donde el target NO sale gratis: llegar a 70 g de proteína con estas 3 líneas obliga a
# estirar el pollo contra el clamp y arrastra kcal/carbos ⇒ hay una decisión que los pesos gobiernan.
# (Con un target ya alcanzable a factor≈1 CUALQUIER vector de pesos da lo mismo y el test no mediría
# nada — se probó, y por eso el caso es éste y no el fácil.)
_LINEAS = ["100 g de pollo", "80 g de arroz", "10 g de aceite"]
_TGT_DURO = {"kcal": 600, "protein": 70, "carbs": 70, "fats": 20}


def _entregado(db, **pesos) -> float:
    prev = (ps.SOLVER_W_KCAL, ps.SOLVER_W_PROTEIN, ps.SOLVER_W_CARBS, ps.SOLVER_W_FATS)
    for k, v in pesos.items():
        setattr(ps, k, v)
    try:
        return ps.solve_meal_macros(_LINEAS, _TGT_DURO, db=db)["achieved"]["protein"]
    finally:
        (ps.SOLVER_W_KCAL, ps.SOLVER_W_PROTEIN, ps.SOLVER_W_CARBS, ps.SOLVER_W_FATS) = prev


def test_duplicar_w_protein_reduce_el_error_de_proteina(db):
    """El knob TIENE autoridad: duplicarlo acerca la proteína entregada al target de forma medible
    (MEDIDO con los defaults vigentes: error 1.40 g → 0.70 g, la mitad).

    ⚠️ HONESTIDAD: esta propiedad NO la compró el re-tune — con los pesos previos duplicar también
    reducía el error (4.20 → 2.20 g). Lo que el re-tune compró es el test de abajo: el error BASE.
    Este test existe como guard de inercia: si un cambio futuro re-infla la fila kcal, los tres
    knobs macro vuelven a ser decorativos y el operador que los toque medirá ruido."""
    base = _entregado(db)
    doble = _entregado(db, SOLVER_W_PROTEIN=ps.SOLVER_W_PROTEIN * 2.0)
    err_base, err_doble = abs(base - 70.0), abs(doble - 70.0)
    assert err_doble < err_base, (
        f"duplicar W_PROTEIN no acercó la proteína al target ({base} → {doble}, target 70) — "
        f"el knob volvió a ser decorativo")


def test_el_retune_baja_el_error_base_de_proteina(db):
    """ESTO es lo que el re-tune compró, y lo que discrimina el régimen nuevo del viejo: con el
    MISMO plato y el MISMO target, el error de proteína entregado por los defaults cae de **4.20 g**
    (punto previo 0.1/1.5/1.1/1.4) y **4.70 g** (original 1.2/1.5/1.1/1.4) a **1.40 g**.

    Se compara contra los pesos históricos EJECUTÁNDOLOS, no contra una constante copiada: si el
    solver cambia por otra razón, los dos lados se mueven juntos y el test sigue midiendo el efecto
    de los PESOS, que es lo que ancla."""
    err_ahora = abs(_entregado(db) - 70.0)
    err_previo = abs(_entregado(db, SOLVER_W_KCAL=0.1, SOLVER_W_PROTEIN=1.5,
                                SOLVER_W_CARBS=1.1, SOLVER_W_FATS=1.4) - 70.0)
    err_original = abs(_entregado(db, SOLVER_W_KCAL=1.2, SOLVER_W_PROTEIN=1.5,
                                  SOLVER_W_CARBS=1.1, SOLVER_W_FATS=1.4) - 70.0)
    assert err_ahora < err_previo, (
        f"los defaults vigentes entregan MÁS error de proteína ({err_ahora:.2f} g) que el punto "
        f"previo al re-tune ({err_previo:.2f} g)")
    assert err_ahora < err_original
    assert err_ahora <= 2.0, (
        f"error de proteína {err_ahora:.2f} g — muy por encima de los 1.40 g medidos en el re-tune")


# ═══════════════════ 4 · el harness queda reproducible en el repo ═══════════════════

def test_el_harness_del_tune_esta_versionado_y_es_determinista():
    """El tune previo se hizo con un script efímero que nunca se commiteó: por eso el único
    registro del share era una frase en un comentario. Este harness vive en el repo."""
    src = _read(os.path.join("scripts", "retune_solver_weights.py"))
    assert "P3-SOLVER-W-RETUNE" in src
    assert "--meals" in src, "el JSON de comidas vivas debe entrar por CLI (no se versiona)"
    assert "import random" not in src and "random." not in src, (
        "búsqueda no determinista: dos corridas darían tunes distintos")
    assert "effective_row_shares" in src, "el harness debe reportar el share efectivo medido"


def test_el_stub_catalogo_del_harness_resuelve_offline():
    """El harness no puede depender de Neon: en un worktree sin `.env` el catálogo sale VACÍO y el
    tune mediría el vacío, no el sistema. El stub embebido tiene que resolver líneas reales."""
    sys.path.insert(0, os.path.join(_BACKEND, "scripts"))
    import retune_solver_weights as rt

    rows = rt.build_stub_rows()
    assert len(rows) >= 100, "stub-catálogo demasiado pobre para un harness creíble"
    d = IngredientNutritionDB(rows=rows)
    m = d.macros_from_ingredient_string("150 g de arroz blanco")
    assert m and m["carbs"] > 0
    assert d.macros_from_ingredient_string("2 huevos")["protein"] > 0
    # y las 4 claves de macro que el solver consume están pobladas en TODAS las filas
    for r in rows:
        for k in ("kcal_per_100g", "protein_g_per_100g", "carbs_g_per_100g", "fats_g_per_100g"):
            assert isinstance(r.get(k), float), f"{r['name']} sin {k}"
