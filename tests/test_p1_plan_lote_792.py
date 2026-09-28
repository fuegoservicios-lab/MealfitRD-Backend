# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-792 · 2026-09-28] (G13, parte (a)) Pisos de presupuesto por moneda con una fuente citada, y el
aviso-en-vez-de-bloqueo decidido por el PAÍS DE MERCADO, no por la moneda.

DECISIÓN DEL DUEÑO (tomada antes de este lote, no de aquí):

* Método del Banco Mundial — *Food Prices for Nutrition 5.0*, indicador ``CoHD_LCU`` (coste de una dieta sana
  por persona y día en moneda local), licencia CC BY 4.0, datos del 2026-07-21.
  ``piso semanal = dieta sana × 7 × 4,286`` — 4,286 es la proporción que ya guarda el piso dominicano (RD$4.000)
  respecto a su propia dieta sana. Escalones como en DO: ×1,75 a 15 días y ×3,25 a 30.
* EUR 75 (igual que antes), MXN 1.500, COP 240.000/420.000/780.000, USD 80 (se mantiene: lo respalda el
  USDA Thrifty Food Plan). Puerto Rico = lo declarado para US.
* En un país de MERCADO beta (``pricing_mode_for_country == 'beta_no_prices'``) el piso AVISA en lugar de
  bloquear. Antes la decisión iba por MONEDA y USD estaba excluida a mano: un usuario de Estados Unidos o de
  Puerto Rico con US$70/semana recibía 422, mientras la lista que se le iba a generar sale SIN precios — el piso
  bloqueaba con una cifra que después nadie usa. El visitante de EE. UU. en RD (mercado DO, paga en USD) sigue
  con el gate duro: su mercado sí tiene precios.

MEDIDO (replay de presupuesto, sin IA, mujer 30 a · 65 kg · 2100 kcal · 7 d):
    antes: US/USD 70 → 422 · PR/USD 70 → 422 · CO/COP 200.000 → aviso (piso 367.500)
    después: US/USD 70 → aviso · PR/USD 70 → aviso · DO/USD 70 → 422 · CO/COP 200.000 → aviso (piso 252.000)

tooltip-anchor: P1-PLAN-LOTE-792
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"
_CICLO = {7: "weekly", 15: "biweekly", 30: "monthly"}

# Los números de la decisión del dueño. Escritos aquí a mano A PROPÓSITO: son el contrato, no una copia del
# código (un test que los leyera del propio módulo aprobaría cualquier cambio).
_PISOS = {
    "EUR": {7: 75, 15: 131, 30: 244},       # 75 × 1,75 = 131,25 → 131 · 75 × 3,25 = 243,75 → 244
    "MXN": {7: 1500, 15: 2625, 30: 4875},
    "COP": {7: 240000, 15: 420000, 30: 780000},
    "USD": {7: 80, 15: 140, 30: 260},
}


@pytest.fixture(scope="module")
def nc():
    import nutrition_calculator as _nc
    return _nc


@pytest.fixture
def paises(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


def _form(moneda, monto, pais=None):
    f = {"budget": "custom", "budgetCurrency": moneda, "budgetAmount": str(monto), "groceryDuration": "weekly",
         "householdSize": 1, "age": 30, "gender": "female", "weight": 65, "weightUnit": "kg", "height": 165,
         "heightUnit": "cm", "activityLevel": "moderate", "mainGoal": "maintenance"}
    if pais is not None:
        f["country"] = pais
    return f


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _front(rel: str) -> str:
    return (_FRONT / rel).read_text(encoding="utf-8")


def _sin_comentarios_js(src: str) -> str:
    return "\n".join(l for l in src.split("\n") if not l.strip().startswith("//"))


# ── A. Los pisos son los del método, con su escalón ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("moneda", sorted(_PISOS))
@pytest.mark.parametrize("dias", [7, 15, 30])
def test_los_pisos_son_los_de_la_decision(nc, moneda, dias):
    assert nc._budget_cycle_floor_for_currency(dias, moneda) == _PISOS[moneda][dias]


@pytest.mark.parametrize("moneda", ["DOP", "EUR", "MXN", "COP", "USD"])
def test_los_escalones_son_los_dominicanos_en_todas_las_monedas(nc, moneda):
    """×1,75 a 15 días y ×3,25 a 30, «como en DO» (4.000 → 7.000 → 13.000). Redondeo a la unidad."""
    semana = nc._budget_cycle_floor_for_currency(7, moneda)
    assert nc._budget_cycle_floor_for_currency(15, moneda) == round(semana * 1.75)
    assert nc._budget_cycle_floor_for_currency(30, moneda) == round(semana * 3.25)


def test_puerto_rico_usa_lo_declarado_para_us():
    """PR = US declarado: misma moneda en el SSOT de países, así que el mismo piso — sin una fila aparte."""
    import constants
    assert constants.COUNTRY_PROFILES["PR"]["currency"] == constants.COUNTRY_PROFILES["US"]["currency"] == "USD"


def test_el_espejo_del_frontend_tiene_los_mismos_numeros():
    src = _front("src/config/formValidation.js")
    i = src.index("export const BUDGET_MIN_TOTAL")
    cuerpo = src[i:src.index("};", i)]
    for moneda, pisos in _PISOS.items():
        fila = re.search(moneda + r":\s*\{([^}]*)\}", cuerpo)
        assert fila, f"{moneda} desapareció de BUDGET_MIN_TOTAL"
        vals = {k: int(v) for k, v in re.findall(r"(\w+):\s*(\d+)", fila.group(1))}
        assert vals == {_CICLO[d]: pisos[d] for d in (7, 15, 30)}, f"{moneda}: frontend {vals}"


@pytest.mark.parametrize("donde", ["nutrition_calculator.py", "docs/country_system_f1.md", "FRONT"])
def test_la_fuente_esta_citada_con_su_fecha(donde):
    """Un piso sin procedencia puede orientar pero no puede impedir una compra (G13). Ahora hay procedencia, y
    tiene que estar ESCRITA donde vive el número: si mañana alguien lo cambia, que sepa de dónde salió."""
    txt = _front("src/config/formValidation.js") if donde == "FRONT" else _src(donde)
    for trozo in ("Food Prices for Nutrition 5.0", "CoHD_LCU", "CC BY 4.0", "2026-07-21", "4,286", "Thrifty"):
        assert trozo in txt, f"{donde}: falta «{trozo}»"


# ── B. Avisa en mercado beta; bloquea en mercado con precios ─────────────────────────────────────────────────────

@pytest.mark.parametrize("pais", ["US", "PR"])
def test_us_y_pr_con_70_dolares_ya_no_reciben_422(nc, paises, pais):
    ok, det = nc.validate_budget_sufficient(_form("USD", 70, pais))
    assert ok is True, f"{pais}: US$70 sigue bloqueando en un mercado cuya lista sale sin precios"
    assert det and det.get("warning_code") == "budget_below_goal_floor_advisory"
    assert "error_code" not in det, "un aviso con error_code lo convertiría el caller en 422"
    assert det["currency"] == "USD" and det["min_budget"] >= 80


@pytest.mark.parametrize("pais", [None, "DO"])
def test_el_visitante_de_eeuu_en_rd_sigue_con_el_gate_duro(nc, paises, pais):
    """Mercado DO (sin país ⇒ DO): los precios son reales y el piso bloquea, pague en la moneda que pague."""
    ok, det = nc.validate_budget_sufficient(_form("USD", 70, pais))
    assert ok is False and det["error_code"] == "budget_below_goal_floor"


def test_la_decision_es_el_mercado_y_no_la_moneda(nc, paises):
    """Pesos dominicanos declarados desde un mercado beta: avisa. Pesos colombianos sin país (mercado DO): bloquea.
    Si la regla volviera a mirar la moneda, las dos mitades cambiarían de lado."""
    ok_us, det_us = nc.validate_budget_sufficient(_form("DOP", 3000, "US"))
    assert ok_us is True and det_us["warning_code"] == "budget_below_goal_floor_advisory"
    ok_do, det_do = nc.validate_budget_sufficient(_form("COP", 100000, None))
    assert ok_do is False and det_do["error_code"] == "budget_below_goal_floor"


def test_colombia_con_200_mil_avisa_contra_el_piso_nuevo(nc, paises):
    ok, det = nc.validate_budget_sufficient(_form("COP", 200000, "CO"))
    assert ok is True and det["warning_code"] == "budget_below_goal_floor_advisory"
    # 240.000 × (2100/2000) = 252.000: el mínimo ya no es el 367.500 de la conversión FX.
    assert det["min_budget"] == 252000


def test_un_mercado_que_gane_precios_vuelve_a_bloquear_solo(nc, paises, monkeypatch):
    """La regla es la PROPIEDAD del país (la puerta `pricing_mode_for_country`), no una lista a mano."""
    import constants
    perfil = dict(constants.COUNTRY_PROFILES["US"], has_native_prices=True)
    monkeypatch.setitem(constants.COUNTRY_PROFILES, "US", perfil)
    ok, det = nc.validate_budget_sufficient(_form("USD", 70, "US"))
    assert ok is False and det["error_code"] == "budget_below_goal_floor"


def test_con_el_sistema_de_paises_apagado_nada_cambia(nc, monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
    ok, det = nc.validate_budget_sufficient(_form("USD", 40, "US"))  # 40 × 60 = 2.400 DOP < 4.200
    assert ok is False and det["error_code"] == "budget_below_goal_floor"


def test_la_puerta_delega_en_el_ssot_de_mercado_y_no_lee_la_moneda():
    src = _src("nutrition_calculator.py")
    arbol = ast.parse(src)
    fn = next(n for n in ast.walk(arbol) if isinstance(n, ast.FunctionDef) and n.name == "_piso_solo_orienta")
    cuerpo = fn.body[1:] if isinstance(fn.body[0], ast.Expr) else fn.body  # sin el docstring: la prosa puede nombrarla
    codigo = " ; ".join(ast.unparse(n) for n in cuerpo)
    assert "pricing_mode_for_form_data" in codigo, "la decisión dejó de pasar por la puerta del mercado"
    assert "currency" not in codigo.lower() and "USD" not in codigo, "la puerta vuelve a mirar la moneda"
    vivos = {n.name for n in ast.walk(arbol) if isinstance(n, ast.FunctionDef)} | {
        n.id for n in ast.walk(arbol) if isinstance(n, ast.Name)}
    assert "_piso_sin_procedencia" not in vivos, "la regla por moneda sigue viva al lado de la nueva"
    gate = next(n for n in ast.walk(arbol) if isinstance(n, ast.FunctionDef) and n.name == "validate_budget_sufficient")
    assert "_piso_solo_orienta(form_data)" in ast.unparse(gate)


# ── C. El wizard decide igual: si no, el aviso del backend sería inalcanzable ──────────────────────────────────

def test_el_wizard_decide_por_el_pais_y_no_por_la_moneda():
    flow = _sin_comentarios_js(_front("src/components/assessment/InteractiveAssessmentFlow.jsx"))
    i = flow.index("const isCustomBudgetValid")
    gate = flow[i:flow.index("\n};", i)]
    assert "pisoSoloOrienta(fd?.country, COUNTRY_SYSTEM_UI)" in gate
    assert "pisoSinProcedencia" not in flow
    paises = _sin_comentarios_js(_front("src/config/countries.js"))
    j = paises.index("export function pisoSoloOrienta")
    cuerpo = paises[j:paises.index("\n}", j)]
    assert "hasNativePrices" in cuerpo and "coerceCountry" in cuerpo
    assert "currency" not in cuerpo and "USD" not in cuerpo, "el espejo vuelve a mirar la moneda"
    assert "pisoSinProcedencia" not in paises
