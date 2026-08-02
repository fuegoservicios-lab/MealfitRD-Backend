"""[P1-CYCLE-QTY-FRACTIONAL · 2026-08-02] La CANTIDAD del periodo cubre 15/30 dias, no 14/28.

Contexto: los callsites que construyen las listas biweekly/monthly llaman a
`get_shopping_list_delta(..., multiplier=household_multiplier * N, ...)` con `N` HARDCODEADO
en 2.0/4.0. El delta YA proyecta a 7 dias (`base_duration_scale = 7.0/num_days` dentro de
`get_shopping_list_delta`), asi que `N=2.0/4.0` compra 2/4 SEMANAS ENTERAS = 14/28 dias para
ciclos declarados de 15/30 dias -> deficit sistematico ~6.7% en TODO estable (arroz, aceite,
avena), invisible al guard de coherencia (compara contra la base SEMANAL con tolerancia 10%,
nunca ve el ciclo completo). El propio repo ya diagnostico y arreglo este gap del lado del
COSTO (`_cycle_cost_multiplier`, marker P1-CYCLE-COVERAGE-FRACTIONAL) pero dejo la CANTIDAD
corta. Este test cierra el mismo gap del lado de la CANTIDAD comprada.

El barrido real (no la estimacion de ~12 callsites del brief original) encontro 26 lineas en
2 ORDENES distintos: `2.0 * household` (numero primero) Y `household_multiplier * 2.0` /
`mult * 2.0` (variable primero) — ambas formas conviven en el repo. El regex de este test
cubre ambos ordenes para que un callsite olvidado en cualquier orden falle el test.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import shopping_calculator as sc

_BACKEND = Path(__file__).resolve().parents[1]

_CALLSITE_FILES = (
    "graph_orchestrator.py",
    "cron_tasks.py",
    "tools.py",
    "routers/plans.py",
)

# Bidireccional: "2.0 * household" (numero->variable) Y "household_multiplier * 2.0"
# (variable->numero, encontrado en cron_tasks.py/routers/plans.py). Ancla ambos nombres
# de variable vistos en el barrido: `household`/`household_multiplier` y `mult`.
_HARDCODED_QTY_MULT = re.compile(
    r"[24]\.0\s*\*\s*\w*(?:household|mult)\w*"
    r"|\w*(?:household|mult)\w*\s*\*\s*[24]\.0"
)


def test_helper_ssot_fraccional():
    assert sc.cycle_qty_multiplier("weekly") == pytest.approx(1.0)
    assert sc.cycle_qty_multiplier("biweekly") == pytest.approx(15 / 7)
    assert sc.cycle_qty_multiplier("monthly") == pytest.approx(30 / 7)
    # Debe ser estrictamente > que los literales viejos (14/28 dias subian de menos).
    assert sc.cycle_qty_multiplier("biweekly") > 2.0
    assert sc.cycle_qty_multiplier("monthly") > 4.0


def test_duracion_desconocida_es_failsafe_1x():
    """Nunca inflar una compra por un valor de duracion que no se entiende."""
    assert sc.cycle_qty_multiplier("bogus") == 1.0
    assert sc.cycle_qty_multiplier("") == 1.0
    assert sc.cycle_qty_multiplier(None) == 1.0


def test_knob_rollback_devuelve_literales_viejos(monkeypatch):
    """MEALFIT_CYCLE_QTY_FRACTIONAL=false -> rollback exacto sin redeploy."""
    monkeypatch.setenv("MEALFIT_CYCLE_QTY_FRACTIONAL", "false")
    assert sc.cycle_qty_multiplier("biweekly") == 2.0
    assert sc.cycle_qty_multiplier("monthly") == 4.0
    assert sc.cycle_qty_multiplier("weekly") == 1.0


def test_knob_default_true_es_fraccional(monkeypatch):
    monkeypatch.delenv("MEALFIT_CYCLE_QTY_FRACTIONAL", raising=False)
    assert sc.cycle_qty_multiplier("monthly") == pytest.approx(30 / 7)


def test_no_quedan_literales_2x_4x_en_callsites_de_cantidad():
    """Barrido exhaustivo: ningun callsite que construya la lista de PERIODO
    (biweekly/monthly) via get_shopping_list_delta puede hardcodear 2.0/4.0
    sobre el household_multiplier/mult. tooltip-anchor: P1-CYCLE-QTY-FRACTIONAL"""
    offenders = []
    for f in _CALLSITE_FILES:
        src = (_BACKEND / f).read_text(encoding="utf-8")
        for m in _HARDCODED_QTY_MULT.finditer(src):
            offenders.append(f"{f}: {m.group()!r}")
    assert not offenders, (
        "multiplier de CANTIDAD hardcodeado 14/28 dias (debe ser "
        f"cycle_qty_multiplier(...)): {offenders}"
    )


def test_callsites_importan_el_helper_ssot():
    """Los 4 archivos que construyen listas de periodo deben importar
    `cycle_qty_multiplier` de shopping_calculator (no reinventar el calculo)."""
    for f in _CALLSITE_FILES:
        src = (_BACKEND / f).read_text(encoding="utf-8")
        assert "cycle_qty_multiplier" in src, f"{f}: no usa el helper SSOT cycle_qty_multiplier"
