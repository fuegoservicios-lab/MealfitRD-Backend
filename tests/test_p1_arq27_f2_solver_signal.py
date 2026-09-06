# -*- coding: utf-8 -*-
"""[ARQ27-P1-08 · 2026-09-06] El cerrador de proteína re-deducía lo que el solver ya había escrito.

`_solver_infeasible` existe desde `P3-SOLVER-FEASIBILITY` y dice, **por macro y con dirección**, que
no hay solución con estos alimentos dentro del clamp: `{'protein': 'high'}` = falta un portador de
proteína y **ninguna re-escala lo arregla**.

Y aun así `_close_protein_gap_for_meal` empezaba SIEMPRE por `_try_scale_existing_protein`, o sea por
la operación que el solver acababa de demostrar imposible. La medición del 06-sep sobre 1.174 platos
de 200 planes vivos lo cuantificó: 86 platos infactibles, de ellos **34 con `protein:high`** — dos
capas trabajando sobre la misma señal sin compartirla. Era la recomendación explícita del propio
análisis del solver, por encima de invertir una semana en factibilidad conjunta.

**Saltarse el escalado no abre ningún hueco.** `_try_scale_existing_protein` es «cierre completo o
nada» desde `P3-PROTEIN-CLOSER-SCALE-FIRST` (antes un crecimiento PARCIAL marcaba `_protein_closed` y
dejaba el déficit abierto en perfiles bariátricos). Lo que se ahorra es la pasada inútil; lo que se
gana es que el motivo quede escrito en el log y en el plato.

Lo que este cierre **no** hace, dicho en vez de fingido: las otras direcciones de infactibilidad
—`fats:low` 20, `carbs:high` 16, `fats:high` 16, `carbs:low` 7— siguen sin consumidor. No hay un
cerrador de grasa ni de carbohidrato al que dárselas, y fabricar uno sin medir antes si mejora el
plato sería justo lo que el análisis del solver desaconseja.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def _cuerpo_del_cerrador() -> str:
    i = _SRC.find("def _close_protein_gap_for_meal(")
    assert i > 0, "desapareció el cerrador de proteína"
    return _SRC[i:i + 4000]


def test_el_cerrador_lee_la_senal_del_solver():
    b = _cuerpo_del_cerrador()
    assert '_solver_infeasible' in b, "el cerrador volvió a ignorar la señal del solver"
    assert '.get("protein")' in b and '"high"' in b


def test_la_senal_se_lee_antes_del_escalado():
    """Leerla después no sirve de nada: el escalado ya habría corrido. El orden es el arreglo."""
    b = _cuerpo_del_cerrador()
    i_signal = b.find("_solver_infeasible")
    i_scale = b.find("_try_scale_existing_protein")
    assert 0 < i_signal < i_scale, "la señal se lee DESPUÉS del escalado que debía evitar"


def test_el_escalado_sigue_corriendo_cuando_no_hay_senal():
    """La conducta previa intacta para el 93 % de los platos que no traen infactibilidad. Un cambio
    que apagara el escalado en general sustituiría un cierre «de chef» por un bolt-on en todas
    partes."""
    b = _cuerpo_del_cerrador()
    assert "if PROTEIN_CLOSER_SCALE_FIRST and not _skip_scale:" in b


def test_queda_marca_en_el_plato_para_poder_medirlo():
    """Sin marca no se puede comparar el antes y el después en producción, y este arreglo se justificó
    con una medición: dejarlo sin instrumentar sería no poder repetirla."""
    assert "_closer_used_solver_signal" in _cuerpo_del_cerrador()


# ── el contrato del solver que este cierre consume ────────────────────────────────────────────
def test_la_senal_sigue_trayendo_direccion_por_macro():
    """Si `_solver_infeasible` dejara de ser un dict {macro: dirección}, el consumo de arriba se
    volvería inerte en silencio — leería `.get("protein")` sobre algo que ya no lo tiene."""
    i = _SRC.find('meal["_solver_infeasible"] = _infe')
    assert i > 0, "cambió el sitio donde se emite la señal"
    bloque = _SRC[max(0, i - 900):i + 300]
    assert "res.get(\"infeasible\")" in bloque
    assert "_solver_residuals" in bloque


def test_no_converge_y_no_hay_solucion_siguen_siendo_distintos():
    """La distinción es la que hace accionable la señal: `_solver_not_converged` (41 % de los platos)
    es «el clamp no clavó el target y los closers terminan», y `_solver_infeasible` (7 %) es «falta un
    portador». Colapsarlas haría que el cerrador se saltara el escalado en cuatro de cada diez platos."""
    assert '_solver_not_converged' in _SRC and '_solver_infeasible' in _SRC
    i_nc = _SRC.find('meal["_solver_not_converged"] = True')
    i_in = _SRC.find('meal["_solver_infeasible"] = _infe')
    assert i_nc > 0 and i_in > 0 and i_nc != i_in


def test_el_cerrador_no_reacciona_a_no_converger():
    """Verificación contra el error contrario: si el `_skip_scale` mirara `_solver_not_converged`, se
    saltaría el escalado en el 41 % de los platos y casi todos podían crecer perfectamente."""
    b = _cuerpo_del_cerrador()
    i_skip = b.find("_skip_scale =")
    linea = b[i_skip:b.find("\n", i_skip)]
    assert "_solver_not_converged" not in linea, linea


def test_el_doc_del_solver_sigue_documentando_la_recomendacion():
    """Este cierre ES la recomendación de ese documento; si alguien la borra, el consumo de arriba se
    queda sin la medición que lo justifica."""
    doc = (_BACKEND / "docs" / "solver_estado_terminacion.md").read_text(encoding="utf-8")
    assert "_solver_infeasible" in doc
    assert "protein:high" in doc or "protein': 'high'" in doc


@pytest.mark.parametrize("direccion", ["fats", "carbs"])
def test_las_direcciones_sin_consumidor_quedan_declaradas(direccion):
    """No se fingen cerradas. `fats:low` (20 platos), `carbs:high` (16), `fats:high` (16) y
    `carbs:low` (7) siguen sin consumidor, y eso está escrito donde se pueda leer."""
    doc = (_BACKEND / "docs" / "solver_estado_terminacion.md").read_text(encoding="utf-8")
    assert direccion in doc
