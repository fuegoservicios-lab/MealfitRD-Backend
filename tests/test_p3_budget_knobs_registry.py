"""[P3-BUDGET-KNOBS-REGISTRY · 2026-08-04] `nutrition_calculator.py` leía ~10 knobs
`MEALFIT_BUDGET_*` con `os.environ.get(...)` crudo — invisibles en `_KNOBS_REGISTRY` /
`/health/version`. Un override de cualquiera de estos (piso de presupuesto, bandas por tier,
reconciliación de costo) era indetectable para el operador sin abrir el `.env` a mano.

Precedente en el MISMO archivo: `P2-SOLVER-KNOBS-REGISTRY` (línea ~285) migró
`MEALFIT_PROTEIN_CEILING_G_PER_KG` a `_env_float`. Este P3 hace lo mismo para el bloque de
presupuesto — con una diferencia deliberada: la lectura sigue siendo POR-LLAMADA (no cacheada a
constante de módulo), para que un rollback de un knob tome efecto sin reiniciar el proceso.

100% OFFLINE: no golpea red ni DB (las funciones de presupuesto son cálculo puro).
"""
import inspect
import logging
import re

import pytest

import nutrition_calculator as nc
from knobs import get_knobs_registry_snapshot


# Los 10 knobs `MEALFIT_BUDGET_*` documentados en el brief de la tarea (grepeados contra el
# archivo). Tres son NOMBRES DINÁMICOS (interpolan `days` o `tier`) — se listan ya expandidos a
# sus valores concretos de producción (los mismos que usan `_BUDGET_CYCLE_FLOOR_DEFAULTS_DOP` /
# `_BUDGET_TIER_BAND_DEFAULTS`).
EXPECTED_KNOB_NAMES = {
    "MEALFIT_BUDGET_FLOOR_ENABLED",
    "MEALFIT_BUDGET_FLOOR_TOTAL_7D_DOP",
    "MEALFIT_BUDGET_FLOOR_TOTAL_15D_DOP",
    "MEALFIT_BUDGET_FLOOR_TOTAL_30D_DOP",
    "MEALFIT_BUDGET_FLOOR_KCAL_REF",
    "MEALFIT_BUDGET_USD_TO_DOP",
    "MEALFIT_BUDGET_FLOOR_TOLERANCE_PCT",
    "MEALFIT_BUDGET_RECONCILE",
    "MEALFIT_BUDGET_RECONCILE_TOL_PCT",
    "MEALFIT_BUDGET_RECONCILE_MIN_COVERAGE",
    "MEALFIT_BUDGET_BAND_LOW",
    "MEALFIT_BUDGET_BAND_MEDIUM",
    "MEALFIT_BUDGET_BAND_HIGH",
    "MEALFIT_BUDGET_TIGHT_CUSTOM_FACTOR",
}


def _invoke_every_budget_knob_reader():
    """Fuerza la lectura de los 14 nombres concretos — incluidos los 3 ciclos (7/15/30) y los 3
    tiers (low/medium/high) que nacen de nombres DINÁMICOS (`f"...{int(days)}..."` /
    `f"...{tier.upper()}"`), que un `grep` de string literal por sí solo no expandiría."""
    nc._budget_floor_enabled()
    for days in (7, 15, 30):
        nc._budget_cycle_floor_dop(days)
    nc._budget_floor_kcal_ref()
    nc._budget_usd_to_dop()
    nc._budget_floor_tolerance_pct()
    nc._budget_reconcile_enabled()
    nc._budget_reconcile_tolerance_pct()
    nc._budget_reconcile_min_coverage()
    for tier in ("low", "medium", "high"):
        nc._budget_tier_band_factor(tier)
    nc._budget_tight_custom_factor()


def test_grep_del_archivo_no_encuentra_mas_knobs_budget_que_los_documentados():
    """Ancla el inventario: si alguien añade un `MEALFIT_BUDGET_*` nuevo sin pasar por este test,
    lo detecta como nombre no documentado (grep sobre los literales `"MEALFIT_BUDGET_..."` y sobre
    los dos templates f-string dinámicos conocidos)."""
    src = inspect.getsource(nc)
    literal_names = set(re.findall(r'"(MEALFIT_BUDGET_[A-Z0-9_]+)"', src))
    # Los dos nombres dinámicos no aparecen como literal completo — se verifican por su prefijo
    # f-string (que SÍ debe seguir presente en el source tras la migración).
    assert 'f"MEALFIT_BUDGET_FLOOR_TOTAL_{int(days)}D_DOP"' in src or \
        "f\"MEALFIT_BUDGET_FLOOR_TOTAL_{int(days)}D_DOP\"" in src, (
            "el template dinámico del piso por ciclo cambió de forma; actualizar el test")
    assert 'f"MEALFIT_BUDGET_BAND_{tier.upper()}"' in src, (
        "el template dinámico de banda por tier cambió de forma; actualizar el test")
    # Todo literal ESTÁTICO encontrado debe estar en el inventario esperado (drift detector).
    unexpected = literal_names - EXPECTED_KNOB_NAMES
    assert not unexpected, f"knob(s) MEALFIT_BUDGET_* no documentados en el test: {unexpected}"


def test_cero_lectura_cruda_de_os_environ_para_budget():
    """El fix elimina TODA lectura `os.environ.get("MEALFIT_BUDGET...")` cruda — todo pasa por los
    helpers `_env_float`/`_env_bool` de `knobs.py` (auto-registro)."""
    src = inspect.getsource(nc)
    assert 'os.environ.get("MEALFIT_BUDGET' not in src, (
        "sigue habiendo una lectura os.environ cruda de un knob MEALFIT_BUDGET_* — "
        "migrar a knobs._env_float/_env_bool")
    assert "os.environ.get(f\"MEALFIT_BUDGET" not in src, (
        "el template dinámico también debe pasar por el helper, no por os.environ.get crudo")


def test_los_14_nombres_concretos_aparecen_en_el_registry_tras_invocar():
    """Paridad derivada: cada nombre CONCRETO que el archivo puede producir (estático o
    interpolado) debe quedar visible en `get_knobs_registry_snapshot()` — la ausencia significa
    que ese knob sigue leyendo `os.environ` crudo por fuera del registry."""
    _invoke_every_budget_knob_reader()
    snapshot = get_knobs_registry_snapshot()
    missing = EXPECTED_KNOB_NAMES - set(snapshot.keys())
    assert not missing, f"knob(s) ausentes del registry tras invocar sus lectores: {missing}"


def test_los_defaults_registrados_coinciden_con_los_defaults_historicos():
    """Migrar al helper NO debe cambiar ni un default — solo darle visibilidad."""
    _invoke_every_budget_knob_reader()
    snap = get_knobs_registry_snapshot()
    expected_defaults = {
        "MEALFIT_BUDGET_FLOOR_ENABLED": True,
        "MEALFIT_BUDGET_FLOOR_TOTAL_7D_DOP": 4000.0,
        "MEALFIT_BUDGET_FLOOR_TOTAL_15D_DOP": 7000.0,
        "MEALFIT_BUDGET_FLOOR_TOTAL_30D_DOP": 13000.0,
        "MEALFIT_BUDGET_FLOOR_KCAL_REF": 2000.0,
        "MEALFIT_BUDGET_USD_TO_DOP": 60.0,
        "MEALFIT_BUDGET_FLOOR_TOLERANCE_PCT": 0.05,
        "MEALFIT_BUDGET_RECONCILE": True,
        "MEALFIT_BUDGET_RECONCILE_TOL_PCT": 0.10,
        "MEALFIT_BUDGET_RECONCILE_MIN_COVERAGE": 0.7,
        "MEALFIT_BUDGET_BAND_LOW": 1.15,
        "MEALFIT_BUDGET_BAND_MEDIUM": 1.6,
        "MEALFIT_BUDGET_BAND_HIGH": 2.5,
        "MEALFIT_BUDGET_TIGHT_CUSTOM_FACTOR": 1.3,
    }
    for name, expected_default in expected_defaults.items():
        assert snap[name]["default"] == expected_default, (
            f"{name}: default registrado {snap[name]['default']!r} != histórico {expected_default!r}")


def test_override_sigue_funcionando_sin_reiniciar_proceso(monkeypatch):
    """La lectura POR-LLAMADA se conserva: un override tomado a mitad de proceso (sin restart)
    debe reflejarse en la SIGUIENTE invocación — es la propiedad que motivó no migrar a constante
    de módulo (a diferencia de P2-SOLVER-KNOBS-REGISTRY)."""
    monkeypatch.setenv("MEALFIT_BUDGET_USD_TO_DOP", "75")
    assert nc._budget_usd_to_dop() == 75.0
    monkeypatch.delenv("MEALFIT_BUDGET_USD_TO_DOP", raising=False)
    assert nc._budget_usd_to_dop() == 60.0


# ---- [I-2 · review final] fuera-de-rango cae al DEFAULT completo, NO al borde clampeado -------

@pytest.mark.parametrize("env_name, fuera_de_rango, reader, default_esperado", [
    # KCAL_REF: validator `v >= 800.0`. 500 está fuera; el clamp viejo daba 800, el helper da 2000.
    ("MEALFIT_BUDGET_FLOOR_KCAL_REF", "500", nc._budget_floor_kcal_ref, 2000.0),
    # USD_TO_DOP: validator `v >= 1.0`. 0.5 está fuera.
    ("MEALFIT_BUDGET_USD_TO_DOP", "0.5", nc._budget_usd_to_dop, 60.0),
    # TOLERANCE_PCT: validator `0.0 <= v <= 0.5`. 0.9 está fuera.
    ("MEALFIT_BUDGET_FLOOR_TOLERANCE_PCT", "0.9", nc._budget_floor_tolerance_pct, 0.05),
])
def test_override_fuera_de_rango_cae_al_default_no_al_borde(
        monkeypatch, caplog, env_name, fuera_de_rango, reader, default_esperado):
    """[I-2 · review final] `_env_float(..., validator=...)` (knobs.py) NO clampa al borde del
    rango permitido cuando el validator rechaza el valor: loguea WARNING y cae al DEFAULT
    COMPLETO. Antes de la migración a knobs.py, el código recortaba (`max`/`min`) — un comentario
    decía "el rango pasa a validator=" como si fuera equivalente, y no lo es: KCAL_REF=500 (bajo
    el piso 800 del validator) daba 800 con el clamp viejo, da 2000 (el default) con el helper
    nuevo — un piso 2,5× MÁS ALTO que el que el operador buscaba BAJAR."""
    monkeypatch.setenv(env_name, fuera_de_rango)
    with caplog.at_level(logging.WARNING):
        resultado = reader()
    assert resultado == default_esperado, (
        f"{env_name}={fuera_de_rango!r} fuera de rango debe caer al DEFAULT completo "
        f"({default_esperado}), no clamparse al borde del validator"
    )
    assert any("fuera de rango" in rec.message for rec in caplog.records), (
        f"{env_name} fuera de rango debe loguear WARNING (patrón `knobs.py:_env_float`)"
    )


def test_override_bool_si_en_espanol_no_es_verdadero(monkeypatch, caplog):
    """[I-2 · review final] Colateral documentado en el mismo comentario del bloque: el parser
    laxo anterior aceptaba `si` (español) como verdadero; `_env_bool` (knobs.py) acepta
    LITERALMENTE solo `1/true/yes/on` — `si`/`sí` no están en esa lista y caen a `False`. No es
    un bug de `_env_bool` (mismo helper que usa todo el repo) sino una operación .env que debe
    escribirse en el vocabulario que el helper entiende."""
    monkeypatch.setenv("MEALFIT_BUDGET_FLOOR_ENABLED", "si")
    assert nc._budget_floor_enabled() is False, (
        "'si' (español) no está en la lista aceptada por _env_bool (1/true/yes/on) y debe caer "
        "a False, no a True"
    )
