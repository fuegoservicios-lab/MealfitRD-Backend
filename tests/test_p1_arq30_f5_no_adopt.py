# -*- coding: utf-8 -*-
"""[P1-ARQ30-F5-NO-ADOPT · 2026-09-06] Factibilidad conjunta: decisión MEDIDA de no adoptar.

`ARQ30-P1-02` propone sustituir el cribado actual por factibilidad conjunta (LP/QP/MILP o CP-SAT).
La limitación que denuncia es **real y está reproducida abajo**: `_feasibility_report` calcula una
cota por COORDENADA, así que un objetivo alcanzable en cada macro por separado pero imposible
conjuntamente sale como `{}` — «todo alcanzable».

El propio gap pone las dos condiciones: **«primero cerrar ARQ27-P1-08 y medir el residual»** y
**«adoptar factibilidad conjunta solo con mejora demostrada»**. `ARQ27-P1-08` está cerrado
(`test_p1_arq27_f2_solver_signal.py`), así que se midió el residual sobre 30 días:

    comidas dimensionadas ....... 3.217
      no convergieron ........... 1.224   38,0 %
      declaradas infactibles ...... 326   10,1 %
    corridas con >=1 infactible . 156 de 264   59,1 %

Cifras altas. Y entonces la pregunta que decide, que es el criterio de cierre —«no gastar LLM
regenerando con el mismo conjunto inviable»—, emparejando por `session_id`:

                        CON infactibles (6.706)   sin (4.018)
    attempts medio ............ 1,552             1,544
    review_passed ............. 92,4 %            93,0 %
    desviacion p50 / p90 ...... 0,020 / 0,036     0,019 / 0,035
    fallback .................. 2,5 %             3,7 %

**La infactibilidad no cuesta ni un reintento ni un punto de calidad**, y las corridas que la sufren
caen MENOS al fallback. El daño que el criterio persigue no está ocurriendo, así que no hay mejora
que obtener: **se conserva el solver actual.** El encargo lo prevé literalmente — «si el prototipo no
mejora, conserva el actual y registra el resultado experimental; no cambies de librería solo para
usar el nombre 3.0».

Este fichero ES el registro de ese experimento. La decisión es de los números, y
`scripts/solver_residual_probe.py` los vuelve a sacar cuando alguien quiera reabrirla.

## Lo que sí quedaba sin cubrir, y aquí se cubre

El criterio «distinguir factibilidad de una plantilla, de una comida, de un día y del horizonte; un
testigo local no certifica todos los niveles». Medido: `_feasibility_report` se invoca **solo desde
`portion_solver`, dos veces, siempre a nivel de COMIDA**. En `horizon.py` y `plan_policy.py` no hay
nada de factibilidad. Un `_solver_infeasible` vacío dice que ESA comida es dimensionable — no que el
día ni el horizonte lo sean. Sin esto escrito, alguien lo leerá como una garantía que nadie dio.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

_SOLVER_SRC = (_BACKEND / "portion_solver.py").read_text(encoding="utf-8")


@pytest.fixture
def ps(monkeypatch):
    monkeypatch.setenv("MEALFIT_SOLVER_FEASIBILITY_SIGNAL", "1")
    import portion_solver
    return importlib.reload(portion_solver)


# ── el contraejemplo de la auditoría, literal ─────────────────────────────────────────────────
def test_el_contraejemplo_sintetico_de_la_auditoria(ps):
    """El gap exige incluirlo. UNA línea con proteína=20x y grasa=10x, x ∈ [0,5 · 2]:

        · proteína 40 sola  → alcanzable (cota alta 20·2 = 40)
        · grasa 5 sola      → alcanzable (cota baja 10·0,5 = 5)
        · CONJUNTAMENTE     → imposible: proteína 40 exige x=2, y con x=2 la grasa es 20

    El reporte devuelve `{}`, que en su contrato significa «todos alcanzables». La cota es NECESARIA,
    no suficiente, y esto lo demuestra en cuatro líneas."""
    entries = [{"macros": {"protein": 20.0, "fats": 10.0}, "movable": True}]
    r = ps._feasibility_report(entries, {"protein": 40.0, "fats": 5.0},
                               min_scale=0.5, hi_by_entry=[2.0])
    assert r == {}, ("si esto deja de ser {} es que alguien implementó factibilidad conjunta — "
                     "actualiza la decisión de no adoptar con la medición que lo justifique")


def test_las_coordenadas_por_separado_si_caben(ps):
    """Cada mitad del contraejemplo, sola, es alcanzable. Es lo que hace que el conjunto engañe."""
    entries = [{"macros": {"protein": 20.0, "fats": 10.0}, "movable": True}]
    assert ps._feasibility_report(entries, {"protein": 40.0}, 0.5, [2.0]) == {}
    assert ps._feasibility_report(entries, {"fats": 5.0}, 0.5, [2.0]) == {}


def test_la_cota_si_detecta_lo_que_le_toca(ps):
    """No se está diciendo que el cribado no sirva: detecta la falta de PORTADOR, que es su trabajo.
    Proteína 100 con una sola línea que da 40 al techo es infactible por coordenada."""
    entries = [{"macros": {"protein": 20.0}, "movable": True}]
    r = ps._feasibility_report(entries, {"protein": 100.0}, 0.5, [2.0])
    assert r.get("protein") == "high"


def test_el_reporte_es_una_cota_necesaria_y_lo_dice(ps):
    i = _SOLVER_SRC.index("def _feasibility_report(")
    doc = _SOLVER_SRC[i:i + 900]
    assert "NECESARIA" in doc, "el docstring tiene que declarar que la cota no es suficiente"


# ── el nivel del testigo ──────────────────────────────────────────────────────────────────────
def test_el_testigo_es_por_COMIDA_y_no_certifica_el_dia():
    """«Un testigo local no certifica todos los niveles» — el criterio, literal. Si alguien añade una
    llamada fuera de `portion_solver`, este test cae y tendrá que declarar a qué nivel opera."""
    import re

    # La frontera de palabra es load-bearing: sin ella el patron casa por SUBCADENA con
    # `_pantry_feasibility_report`, que es OTRA funcion y otro concepto — la factibilidad de la
    # NEVERA («cuantos dias aguanta la despensa»), que si opera a nivel de dia. Confundirlas
    # seria leer una garantia de la despensa como una garantia del solver.
    #
    # Y se escanea con `pathlib`, no con un subproceso: la primera version llamaba a `git grep`
    # y, cuando este no devolvia nada, el conjunto salia vacio y el test PASABA sin haber mirado.
    # Un centinela que no distingue «no hay violaciones» de «no pude buscar» es una coartada.
    rx = re.compile(r"\b_feasibility_report\(")
    ficheros = set()
    for _p in _BACKEND.rglob("*.py"):
        rel = _p.relative_to(_BACKEND).as_posix()
        if rel.startswith(("tests/", "venv", "scripts/")) or "__pycache__" in rel:
            continue
        try:
            if rx.search(_p.read_text(encoding="utf-8")):
                ficheros.add(rel)
        except Exception:
            continue

    assert "portion_solver.py" in ficheros, (
        "el escaneo no encontro ni la llamada que SI existe: no esta midiendo nada")
    ajenos = ficheros - {"portion_solver.py"}
    assert not ajenos, f"la factibilidad del SOLVER se evalua ahora fuera de el: {ajenos}"


def test_la_factibilidad_de_la_NEVERA_es_otra_cosa():
    """`_pantry_feasibility_report` existe, opera a nivel de DÍA («soporta≈Nd») y no tiene nada que
    ver con el dimensionado de porciones. Se ancla para que nadie las cruce: una despensa que aguanta
    siete días no dice que las comidas de esos días sean dimensionables, ni al revés."""
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert "def _pantry_feasibility_report(" in src
    assert "days_supported" in src


def test_no_hay_factibilidad_de_dia_ni_de_horizonte():
    """Hoy no existe, y esa ausencia es el hueco que el criterio nombra. Que el test lo diga evita
    que un `_solver_infeasible` vacío se lea como una garantía del día."""
    for modulo in ("horizon.py", "plan_policy.py"):
        src = (_BACKEND / modulo).read_text(encoding="utf-8")
        assert "_feasibility_report" not in src, (
            f"{modulo} evalúa factibilidad: documenta a qué nivel y actualiza esta decisión")


# ── la precondición del gap ───────────────────────────────────────────────────────────────────
def test_arq27_p1_08_esta_cerrado():
    """El gap dice «primero cerrar ARQ27-P1-08 y medir el residual». Sin eso, cualquier medición del
    residual estaría contando fallos que aquel arreglo ya absorbe."""
    assert (_BACKEND / "tests" / "test_p1_arq27_f2_solver_signal.py").exists()
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '(meal.get("_solver_infeasible") or {}).get("protein")' in go, (
        "el cerrador ya no consume la señal de infactibilidad")


# ── la decisión, con su registro ──────────────────────────────────────────────────────────────
def test_la_sonda_del_residual_esta_commiteada():
    """«Recontar mañana» no es un entregable: lo que cierra el gap es la sonda, para que la decisión
    se pueda revisar con números en vez de con memoria."""
    p = _BACKEND / "scripts" / "solver_residual_probe.py"
    assert p.exists()
    src = p.read_text(encoding="utf-8")
    for cifra in ("1,55", "1,54", "38,0 %", "10,1 %"):
        assert cifra in src, f"la medición que justificó la decisión no está registrada: {cifra}"


def test_no_se_anadio_ninguna_libreria_de_optimizacion():
    """«No cambies de librería solo para usar el nombre 3.0». La decisión fue no adoptar, así que el
    árbol no debe tener un solver LP/MILP nuevo."""
    req = ""
    for nombre in ("requirements.txt", "requirements-dev.txt", "pyproject.toml"):
        p = _BACKEND / nombre
        if p.exists():
            req += p.read_text(encoding="utf-8").lower()
    for lib in ("ortools", "pulp", "cvxpy", "mip", "gurobi", "cplex"):
        assert lib not in req, f"se añadió {lib} sin una medición que demuestre la mejora"
