# -*- coding: utf-8 -*-
"""[P3-SOLVER-TERMINATION · 2026-09-06] Los estados de terminación del solver, anclados.

El roadmap 2.6 pedía «distinguir *no converge* de *no hay solución* y de *se acabó el tiempo*» y
lo estimaba en ≈1 semana. **Dos de los tres ya existían** —`_solver_not_converged` con el macro
culpable y `_solver_infeasible` con la dirección— y del tercero no hay ni un caso: el solver es un
mínimo cuadrados sobre pocas variables, no una búsqueda.

Medido sobre 1.174 platos de 200 planes vivos: no converge el 41 %, infactible el 7 %, greedy el
0 %. Ese 41 % es por diseño —el clamp acota cuánto se mueve cada línea y los closers terminan— y
leerlo como «el solver falla mucho» sería confundir *no clavó el target él solo* con *el plato
salió mal*.

Este test ancla que las tres marcas sigan escribiéndose y que sigan siendo **distinguibles**: si
alguien las colapsara en un único `_solver_failed`, la reparación aguas abajo perdería la única
señal que le dice si añadir un portador o re-escalar.

Doc: `backend/docs/solver_estado_terminacion.md`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_DOC = _BACKEND / "docs" / "solver_estado_terminacion.md"


def _bloque() -> str:
    """El tramo donde el solver estampa sus marcas de terminación."""
    i = _SRC.find('meal["_solver_greedy_fallback"] = True')
    assert i > 0, "desapareció el estampado del greedy fallback"
    return _SRC[max(0, i - 600):i + 2200]


@pytest.mark.parametrize("marca", [
    "_solver_greedy_fallback",
    "_solver_not_converged",
    "_solver_failed_macros",
    "_solver_infeasible",
    "_solver_residuals",
])
def test_las_marcas_siguen_escribiendose(marca):
    assert f'meal["{marca}"]' in _bloque(), f"el solver dejó de estampar {marca}"


def test_no_converger_y_ser_infactible_son_ramas_distintas():
    """La distinción entera: `not converged` dice «no clavé el target»; `infeasible` dice «con
    estos alimentos NO existe solución dentro del clamp». La segunda trae dirección por macro y es
    la única sobre la que se puede actuar añadiendo un portador."""
    b = _bloque()
    i_nc = b.find('meal["_solver_not_converged"]')
    i_in = b.find('meal["_solver_infeasible"]')
    assert i_nc > 0 and i_in > 0 and i_nc != i_in, "las dos marcas se colapsaron"
    entre = b[min(i_nc, i_in):max(i_nc, i_in)]
    assert "if" in entre, "dejaron de ser ramas separadas"


def test_el_macro_culpable_acompana_a_la_no_convergencia():
    """«no convergió» a secas no es accionable; «no convergió en grasas» sí. Medido: grasas 431,
    proteína 353, carbos 92 — el reparto importa."""
    b = _bloque()
    assert "converged_per_macro" in b
    assert "_solver_failed_macros" in b


def test_la_infactibilidad_lleva_direccion():
    """`{'protein': 'high'}` = falta portador de proteína. Sin la dirección, la reparación no
    puede saber si añadir o quitar."""
    b = _bloque()
    i = b.find('meal["_solver_infeasible"]')
    assert "res.get(\"infeasible\")" in b[:i], "la dirección dejó de venir del resultado del solver"


# ── el doc y sus números ─────────────────────────────────────────────────────────────────────
def test_el_doc_existe_y_trae_la_medicion():
    assert _DOC.exists(), "se borró el doc del estado de terminación"
    txt = _DOC.read_text(encoding="utf-8")
    for cifra in ("1.174", "41 %", "7 %", "0 %"):
        assert cifra in txt, f"el doc perdió la cifra {cifra}"


def test_el_doc_no_promete_un_estado_de_timeout():
    """Modelar «se acabó el tiempo» sería inventar una situación de la que no hay ni un caso. El
    doc lo dice explícitamente para que nadie lo reabra sin evidencia nueva."""
    txt = _DOC.read_text(encoding="utf-8")
    assert "no aparece" in txt and "timeout" in txt.lower()
    assert not re.search(r'meal\["_solver_timeout"\]', _SRC), (
        "apareció un estado de timeout sin evidencia que lo justifique")
