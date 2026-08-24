"""[P1-COUNTRY-BUDGET-FLOOR-FX · 2026-08-23] G13: el piso de presupuesto bloqueaba con un
número que no salía de ninguna cesta del país.

MEDIDO contra el endpoint público de producción antes de tocar nada:

    ES/EUR → 94 EUR    MX/MXN → 1750 MXN    CO/COP → 437.500 COP    DO/DOP → 5.100 DOP

Un colombiano que declara 200.000 COP/semana —cifra realista— recibía 422
`budget_below_goal_floor` y no podía generar plan. Para 2500 kcal el piso sube a 437.500
COP/semana ≈ 1,88 M COP/mes para UNA persona: por encima del salario mínimo mensual
colombiano. Y el número no venía de datos colombianos; el propio comentario de la
derivación lo dice: EUR=USD×0,95 · MXN=USD×18 · COP=USD×4200.

Peor: pasado el gate ese número era estructuralmente inútil. Al ser país beta la lista sale
sin precios, `compute_shopping_cost_summary` devuelve None y el pase de abaratamiento queda
inalcanzable. Bloqueaba con una cifra que después nadie usaba.

EL GAP OFRECÍA DOS SALIDAS y este cierre toma la (b) a sabiendas: mientras la moneda no
tenga precios propios, el piso pasa de BLOQUEO a AVISO. La (a) —curar cestas reales por
país con fuente citada— sigue abierta y es la correcta a largo plazo; no se puede
improvisar sin datos. *Un número sin procedencia puede orientar; no puede impedir una
compra.*

DOS LADOS, o el arreglo no existe: el wizard tiene su propio piso en
`formValidation.BUDGET_MIN_TOTAL` y bloquea el botón «Siguiente» ANTES de enviar nada. Si
sólo se degradara el backend, el usuario ni llegaría a que le respondieran.
"""
from __future__ import annotations

import io
import os
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FRONT = _BACKEND.parent / "frontend"


def _leer(p: Path) -> str:
    return io.open(p, encoding="utf-8").read()


@pytest.fixture()
def _con_paises(monkeypatch):
    """El sistema de países encendido: es el entorno de producción desde el flip."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    yield


def _form(currency: str, amount) -> dict:
    return {
        "budget": "custom", "budgetCurrency": currency, "budgetAmount": str(amount),
        "goal": "maintenance", "gender": "male", "age": 30, "weight": 75, "height": 175,
        "weightUnit": "kg", "activityLevel": "moderate", "planDuration": "7",
    }


# ── el defecto, en el caso que lo destapó ──────────────────────────────────────

def test_el_colombiano_con_una_cifra_realista_ya_no_queda_bloqueado(_con_paises):
    from nutrition_calculator import validate_budget_sufficient
    ok, detalle = validate_budget_sufficient(_form("COP", 200000))
    assert ok is True, (
        "200.000 COP/semana es una cifra realista en Colombia y seguía bloqueando contra un "
        "piso de 437.500 que no sale de ninguna cesta colombiana"
    )
    # El aviso viaja: el usuario merece ver la orientación, sólo que no le cierra la puerta.
    assert detalle is not None and detalle.get("warning_code") == "budget_below_goal_floor_advisory"
    assert "error_code" not in detalle, "un aviso no puede llevar error_code: el caller lanzaría 422"


@pytest.mark.parametrize("moneda,monto", [("EUR", 60), ("MXN", 1200), ("COP", 200000)])
def test_las_tres_monedas_sin_cesta_propia_avisan_en_vez_de_bloquear(_con_paises, moneda, monto):
    from nutrition_calculator import validate_budget_sufficient
    ok, _ = validate_budget_sufficient(_form(moneda, monto))
    assert ok is True, f"{moneda} sigue bloqueando con un piso sin procedencia"


# ── y lo que NO puede haber cambiado ───────────────────────────────────────────

@pytest.mark.parametrize("moneda,monto", [("DOP", 1000), ("USD", 40)])
def test_el_gate_duro_sigue_intacto_en_dop_y_usd(_con_paises, moneda, monto):
    """El contrato del gap es explícito: DOP y USD conservan el bloqueo. USD arrastra el
    mismo defecto (su piso es la cesta dominicana entre 50) pero es el camino histórico, y
    ensancharle la puerta es una decisión de producto, no un efecto lateral de este fix."""
    from nutrition_calculator import validate_budget_sufficient
    ok, detalle = validate_budget_sufficient(_form(moneda, monto))
    assert ok is False, f"{moneda} debe seguir bloqueando: es el camino probado"
    assert detalle["error_code"] == "budget_below_goal_floor"


def test_un_presupuesto_holgado_pasa_sin_aviso_en_moneda_beta(_con_paises):
    """El aviso no puede ser universal: si la cifra supera el piso, no hay nada que avisar."""
    from nutrition_calculator import validate_budget_sufficient
    ok, detalle = validate_budget_sufficient(_form("COP", 900000))
    assert ok is True and detalle is None


# ── la regla se lee del SSOT, no de una lista a mano ───────────────────────────

def test_la_condicion_es_has_native_prices_y_no_una_lista_de_monedas():
    """Si mañana España tiene precios curados, su gate debe volver a ser duro SOLO. Una
    lista a mano de monedas no se enteraría."""
    from nutrition_calculator import _piso_sin_procedencia
    assert _piso_sin_procedencia("COP") is True
    assert _piso_sin_procedencia("DOP") is False
    assert _piso_sin_procedencia("USD") is False
    assert _piso_sin_procedencia("XYZ") is False, "moneda desconocida: no se degrada nada"

    fuente = _leer(_BACKEND / "nutrition_calculator.py")
    i = fuente.index("def _piso_sin_procedencia")
    cuerpo = fuente[i:fuente.index("\ndef ", i + 10)]
    assert "has_native_prices" in cuerpo, "la condición dejó de leerse del SSOT de países"


def test_espana_con_precios_propios_volveria_a_bloquear(monkeypatch, _con_paises):
    """La prueba de que la regla es la propiedad y no la moneda: se enciende
    `has_native_prices` para ES y su piso vuelve a ser un bloqueo, sin tocar código."""
    import constants
    from nutrition_calculator import _piso_sin_procedencia
    assert _piso_sin_procedencia("EUR") is True
    perfil = dict(constants.COUNTRY_PROFILES["ES"])
    perfil["has_native_prices"] = True
    monkeypatch.setitem(constants.COUNTRY_PROFILES, "ES", perfil)
    assert _piso_sin_procedencia("EUR") is False


# ── el otro lado: sin él, el de arriba es inalcanzable ─────────────────────────

def test_el_wizard_tambien_deja_pasar_o_el_arreglo_del_backend_no_se_alcanza():
    src = _leer(_FRONT / "src" / "components" / "assessment" / "InteractiveAssessmentFlow.jsx")
    codigo = "\n".join(l for l in src.split("\n") if not l.strip().startswith("//"))
    assert "pisoSinProcedencia(moneda)" in codigo, (
        "el gate del wizard sigue duro: el usuario no llegaría ni a enviar el formulario, "
        "así que la degradación del backend no se alcanzaría nunca"
    )


def test_el_espejo_del_frontend_lee_la_misma_propiedad():
    src = _leer(_FRONT / "src" / "config" / "countries.js")
    assert "export function pisoSinProcedencia" in src
    assert "hasNativePrices" in src, "la tabla del frontend no espeja has_native_prices"
    # paridad fila a fila con el SSOT del backend
    import constants
    filas = dict(re.findall(r"code: '([A-Z]{2})'[^}]*hasNativePrices: (true|false)", src))
    assert filas, "no pude parsear las filas del frontend"
    for cc, perfil in constants.COUNTRY_PROFILES.items():
        esperado = "true" if perfil["has_native_prices"] else "false"
        assert filas.get(cc) == esperado, (
            f"{cc}: el frontend dice hasNativePrices={filas.get(cc)} y el backend "
            f"has_native_prices={perfil['has_native_prices']}"
        )


def test_el_knob_apagado_no_cambia_nada(monkeypatch):
    """Byte-identidad con la conducta pre-países: sin el sistema encendido, EUR/MXN/COP
    caen en el camino DOP de siempre y ese gate es duro."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    from nutrition_calculator import validate_budget_sufficient
    # Cifra BAJO el piso dominicano a propósito: con 200.000 el test pasaba por holgura y
    # no medía nada. Con el knob apagado, COP se trata como DOP y 1.000 debe RECHAZARSE
    # igual que antes de que existieran los países — si la degradación se colara aquí,
    # estaríamos cambiando la conducta de todos los usuarios históricos.
    ok, detalle = validate_budget_sufficient(_form("COP", 1000))
    assert ok is False, "con el sistema de países apagado el gate duro debe seguir intacto"
    assert detalle["error_code"] == "budget_below_goal_floor"
    assert "warning_code" not in detalle
