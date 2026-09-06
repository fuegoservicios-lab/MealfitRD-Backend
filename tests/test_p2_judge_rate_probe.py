# -*- coding: utf-8 -*-
"""[P2-JUDGE-RATE-PROBE · 2026-09-06] La sonda de la tasa del juez, y su línea base.

El gap decía «recontar mañana». Recontar a mano no es un entregable: la próxima vez que haga
falta, alguien vuelve a escribir la consulta y elige otra ventana, y las dos cifras dejan de ser
comparables. Lo que cierra el gap es **la sonda commiteada más la línea base tomada antes de
desplegar** — sin ese antes, el después no mide nada.

Ese error ya se pagó hoy: corrí una comprobación en seco *después* de que el cron hubiera actuado
y reporté «cierra 0» cuando había cerrado 10.

El test no ejecuta la sonda (necesita la base): ancla que existe, que su contrato no se pierde y
que la línea base sigue documentada con sus cifras.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

_SONDA = _BACKEND / "scripts" / "judge_violation_rate.py"
_BASE = _BACKEND / "docs" / "judge_violation_baseline.md"


def test_la_sonda_existe_y_compila():
    import ast
    assert _SONDA.exists(), "se borró la sonda de la tasa del juez"
    ast.parse(_SONDA.read_text(encoding="utf-8"))


def test_agrupa_por_fecha_de_creacion_no_por_ventana():
    """La unidad es la fecha de creación del plan. Una tasa agregada sobre 7 días mezcla planes
    anteriores y posteriores a cada arreglo — que es exactamente lo que me hizo recomendar atacar
    algo ya cerrado."""
    txt = _SONDA.read_text(encoding="utf-8")
    assert "created_at" in txt and "str(creado)[:10]" in txt


def test_no_gasta_llm():
    """Lee el historial que el juez ya dejó en el plan. Si esta sonda invocara al juez, correrla a
    diario costaría dinero y nadie la correría."""
    txt = _SONDA.read_text(encoding="utf-8")
    assert "_culinary_judge_history" in txt
    for caro in ("run_culinary_judge", "ainvoke", "llm"):
        assert caro not in txt, f"la sonda empezó a gastar LLM ({caro})"


def test_abre_el_pool_antes_de_leer_el_catalogo():
    """Fuera de FastAPI el pool no está abierto y `master_ingredients` sale vacío: la columna de
    hierbas mediría cero siempre y parecería una buena noticia."""
    txt = _SONDA.read_text(encoding="utf-8")
    assert "connection_pool.open()" in txt


@pytest.mark.parametrize("prediccion", ["hierbas", "hints"])
def test_mide_las_predicciones_de_los_arreglos(prediccion):
    """Cada arreglo del 06-sep predice que una columna tiende a cero. Una sonda que solo cuenta
    violaciones no puede desmentirlos."""
    assert prediccion in _SONDA.read_text(encoding="utf-8")


# ── la línea base ────────────────────────────────────────────────────────────────────────────
def test_la_linea_base_existe_con_sus_cifras():
    assert _BASE.exists(), "se borró la línea base: sin ella el recuento de mañana no compara nada"
    txt = _BASE.read_text(encoding="utf-8")
    for cifra in ("1,80", "2,47", "4,00", "2.043"):
        assert cifra in txt, f"la línea base perdió {cifra}"


def test_la_linea_base_avisa_del_tamano_de_muestra():
    """El 06-sep son 2 planes. Sin la advertencia, alguien leerá «4,00 por plan» como una
    tendencia y actuará sobre ruido."""
    txt = _BASE.read_text(encoding="utf-8")
    assert "2 planes" in txt and "demasiado pequeña" in txt


def test_la_linea_base_dice_que_todo_es_anterior_al_despliegue():
    """Es la mitad que hace comparable la otra: si alguien cree que estas filas ya incluyen los
    arreglos, leerá el fracaso donde no lo hay."""
    txt = _BASE.read_text(encoding="utf-8")
    assert "anteriores" in txt and "07-sep" in txt
