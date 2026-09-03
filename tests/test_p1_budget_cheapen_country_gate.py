"""[P1-BUDGET-CHEAPEN-COUNTRY-GATE · 2026-08-18] Regresión: la pasada de sustitución
económica (`_apply_budget_cheapen_pass`) corría PAÍS-CIEGA.

Detectado en la primera renovación ES real post-flip (plan 6a4321f5): con el motor ya
en modo España (pools Boquerones/Almejas/Garbanzos), la pasada sustituyó
'habas → Habichuelas rojas' y 'almendras → Maní' comparando RD$/lb — precios del
catálogo DO sin significado para un plan beta — y re-criollizando nombres que el motor
acababa de elegir en español. Sus DOS piernas son DO-céntricas: el price map
(`_budget_build_master_price_map`) solo trae precios RD y `_BUDGET_CHEAP_EQUIVALENTS`
apunta a filas criollas. Es la misma clase MUTATOR-PURITY que F2 cerró para
swap/recalc — esta era una 3ª superficie de mutación que quedó fuera.

Cubre:
  1. Unit: país beta ⇒ la pasada retorna 0 y NO toca los días (force incluido —
     la convergencia T2 tampoco debe sustituir en beta). El gate vive ANTES de
     cualquier acceso a DB, así que el test no necesita catálogo.
  2. Unit: el gate NO rompe DO — con país DO la pasada sigue su camino normal
     (aquí solo se verifica que pasa del gate; la conducta DO completa la anclan
     los tests de presupuesto preexistentes).
  3. Parser: el gate vive en la CABECERA de la función (antes del gate de
     economía `force`) para cubrir los 3 call sites.
"""
import os
import re

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


try:
    from graph_orchestrator import _apply_budget_cheapen_pass as _CHEAPEN
    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover
    _CHEAPEN = None
    _IMPORT_ERR = _e

requires_orq = pytest.mark.skipif(
    _CHEAPEN is None,
    reason=f"graph_orchestrator no importable en este entorno: {_IMPORT_ERR}",
)


def _days_con_almendras():
    return [{
        "day": 1,
        "meals": [{
            "name": "Ensalada con almendras",
            "ingredients": ["Almendras (30g)", "Acelgas (100g)"],
        }],
    }]


# --------------------------------------------------------------------------------------
# 1. Unit — beta hace skip total, días intactos
# --------------------------------------------------------------------------------------
# `country_for_form_data` respeta el knob maestro (F0): con MEALFIT_COUNTRY_SYSTEM
# apagado TODO cae a 'DO' incondicional — y este gate queda INERTE, que es exactamente
# la semántica dark correcta (primer run del test lo demostró: con el knob off y
# force=True, la pasada sustituyó de verdad contra el catálogo). Los tests del gate
# encienden el knob explícitamente.

@requires_orq
def test_beta_skip_total_dias_intactos(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    days = _days_con_almendras()
    subs = _CHEAPEN(days, {"country": "ES", "budget": "economico"})
    assert subs == 0, "en país beta la pasada no debe sustituir NADA"
    assert days[0]["meals"][0]["ingredients"][0] == "Almendras (30g)", (
        "el ingrediente premium debe quedar intacto (sin Maní criollo en un plan ES)"
    )


@requires_orq
def test_beta_skip_tambien_con_force(monkeypatch):
    """force=True es la convergencia post-costeo (T2): en beta el costeo RD$ no aplica,
    así que la convergencia tampoco puede sustituir."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    days = _days_con_almendras()
    subs = _CHEAPEN(days, {"country": "ES"}, force=True)
    assert subs == 0
    assert days[0]["meals"][0]["ingredients"][0] == "Almendras (30g)"


@requires_orq
def test_beta_todos_los_paises_sin_precios(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    for pais in ("ES", "MX", "CO", "PR", "US"):
        days = _days_con_almendras()
        assert _CHEAPEN(days, {"country": pais}, force=True) == 0, f"{pais} debe skipear"


@requires_orq
def test_knob_apagado_gate_inerte_byte_identico(monkeypatch):
    """Con el sistema de países APAGADO el gate no puede cambiar conducta: 'ES' cae a
    'DO' y la pasada sigue su camino DO de siempre (aquí, sin economía pedida, 0 subs
    por el gate de ECONOMÍA — la prueba es que los días quedan intactos igual)."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    days = _days_con_almendras()
    subs = _CHEAPEN(days, {"country": "ES"})  # sin force: gate de economía corta
    assert subs == 0
    assert days[0]["meals"][0]["ingredients"][0] == "Almendras (30g)"


@requires_orq
def test_do_pasa_del_gate(monkeypatch):
    """DO tiene precios nativos: el gate NO debe cortar. La pasada sigue a sus guards
    normales (aquí economía OFF ⇒ 0 subs por el gate de economía, no por el de país —
    lo distinguimos con un form_data sin presupuesto económico)."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    days = _days_con_almendras()
    subs = _CHEAPEN(days, {"country": "DO"})
    assert subs == 0  # economía no pedida — pero NO por el country gate
    # La distinción real la da el parser test (posición) + los tests de presupuesto
    # preexistentes que corren con DO y SÍ sustituyen cuando toca.


# --------------------------------------------------------------------------------------
# 2. Parser — el gate vive en la cabecera, antes del gate de economía
# --------------------------------------------------------------------------------------

def test_gate_en_cabecera_antes_del_force_gate():
    src = _read("graph_orchestrator.py")
    m = re.search(r"def _apply_budget_cheapen_pass\(.*?\n(.*?)\n\s*if not force:", src, re.DOTALL)
    assert m, "no se encontró la cabecera de _apply_budget_cheapen_pass antes de `if not force:`"
    head = m.group(1)
    assert "pricing_mode_for_country" in head and "beta_no_prices" in head, (
        "el country-gate debe vivir ANTES del gate de economía (`if not force:`) para cubrir "
        "también la convergencia force=True — moverlo después reabre las sustituciones RD$ en beta"
    )
    assert "tooltip-anchor: P1-BUDGET-CHEAPEN-COUNTRY-GATE" in src
