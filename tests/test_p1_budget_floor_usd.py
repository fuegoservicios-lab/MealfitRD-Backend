"""[P1-BUDGET-FLOOR-USD · 2026-08-21] El presupuesto de US y Puerto Rico se juzgaba con la cesta
dominicana al tipo de cambio.

Fase 1 le dio piso PROPIO a EUR, MXN y COP, y los derivó todos del piso USD que el producto ya
declaraba (80/140/260, `frontend/src/config/formValidation.js::BUDGET_MIN_TOTAL`). A USD, que es la
moneda de **2 de los 5 países beta**, no se lo dio: cae al `else` histórico, que multiplica lo
declarado por `_budget_usd_to_dop()` y lo compara contra el piso DOP.

Medido:

| ciclo | piso backend (DOP ÷ 60) | piso que el producto DECLARA |
|---|---|---|
| 7 d  | US$ 66,67  | US$ 80 |
| 15 d | US$ 116,67 | US$ 140 |
| 30 d | US$ 216,67 | US$ 260 |

Un 17% de desacuerdo entre las dos capas. El comentario del piso DOP dice literalmente «DEBE quedar
consistente con BUDGET_MIN_TOTAL del frontend», y existe un test de paridad cross-file… que sólo
cubre EUR/MXN/COP, porque el backend no tenía entrada USD que comparar. La regla estaba escrita y
la moneda que la incumplía era justamente la que no se miraba.

QUÉ SE ARREGLA Y QUÉ NO. **No se inventa ningún número**: se usa el que ya está declarado y del que
salieron los otros tres. Lo que NO cierra esto es la otra mitad de P1-19 — que los pisos de
EUR/MXN/COP siguen siendo conversiones de tipo de cambio y no precios de cesta medidos en cada país.
Eso es curación de datos y una decisión del dueño; fabricar cifras de la compra semanal en España
sería exactamente la clase de afirmación sin respaldo que este repo ya pagó en la procedencia del
catálogo.

DE PASO, UN ACOPLAMIENTO INVISIBLE QUE DESAPARECE: mientras el piso de USD salía de dividir pesos
dominicanos, **una devaluación movía el mínimo de un usuario de Florida** sin que nadie tocara nada.

DIRECCIÓN DEL CAMBIO: el umbral SUBE (66,67 → 80), o sea que se vuelve más estricto. Nadie que hoy
pase el formulario queda fuera: el frontend ya bloquea en 80. Lo que se cierra es la ventana de
quien entra por la API sin pasar por él.

Cubre:
  A. USD tiene piso propio y es el declarado.
  B. Paridad cross-file de las CINCO monedas (la que faltaba incluida).
  C. La comparación deja de depender del tipo de cambio.
  D. Byte-identidad de DOP y del knob apagado.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_FORM_VALIDATION = _REPO / "frontend" / "src" / "config" / "formValidation.js"

_CICLO_A_CLAVE = {7: "weekly", 15: "biweekly", 30: "monthly"}


@pytest.fixture(scope="module")
def nc():
    import nutrition_calculator as _nc
    return _nc


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


def _budget_min_total_del_frontend() -> dict:
    """Lee `BUDGET_MIN_TOTAL` del fuente del frontend. Es el SSOT del que salieron los otros tres
    pisos, así que el test lo consulta en vez de re-teclear sus números."""
    src = _FORM_VALIDATION.read_text(encoding="utf-8", errors="replace")
    i = src.index("export const BUDGET_MIN_TOTAL")
    j = src.index("};", i)
    cuerpo = src[i:j]
    out = {}
    for m in re.finditer(r"(\w+):\s*\{([^}]*)\}", cuerpo):
        moneda, campos = m.group(1), m.group(2)
        vals = {k: float(v) for k, v in re.findall(r"(\w+):\s*([\d.]+)", campos)}
        if vals:
            out[moneda] = vals
    return out


# ── A. USD tiene piso propio ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dias,esperado", [(7, 80.0), (15, 140.0), (30, 260.0)])
def test_usd_tiene_piso_propio(nc, dias, esperado):
    """El número no es nuevo: es el que el producto ya declara y del que se derivaron EUR/MXN/COP
    (EUR×0,95, MXN×18, COP×4200 sobre estos mismos 80/140/260)."""
    assert nc._budget_cycle_floor_for_currency(dias, "USD") == esperado


def test_usd_ya_no_hereda_el_piso_dominicano(nc):
    """El síntoma: antes USD devolvía 4000/7000/13000 — las cifras en pesos dominicanos."""
    for dias in (7, 15, 30):
        assert nc._budget_cycle_floor_for_currency(dias, "USD") != nc._budget_cycle_floor_dop(dias)


# ── B. Paridad cross-file de las cinco monedas ──────────────────────────────────────────────────

@pytest.mark.parametrize("moneda", ["USD", "EUR", "MXN", "COP"])
@pytest.mark.parametrize("dias", [7, 15, 30])
def test_paridad_con_el_frontend(nc, moneda, dias):
    """La paridad ya se exigía por escrito («DEBE quedar consistente con BUDGET_MIN_TOTAL del
    frontend») y ya tenía test… para EUR/MXN/COP. USD quedaba fuera porque el backend no tenía
    entrada que comparar — o sea que la única moneda que incumplía la regla era justo la que nadie
    miraba. Ahora las cuatro se comprueban por la misma puerta."""
    front = _budget_min_total_del_frontend()
    assert moneda in front, f"{moneda} desapareció de BUDGET_MIN_TOTAL"
    esperado = front[moneda][_CICLO_A_CLAVE[dias]]
    assert nc._budget_cycle_floor_for_currency(dias, moneda) == esperado, (
        f"{moneda} {dias}d: backend y frontend discrepan"
    )


def test_el_piso_dop_tambien_coincide(nc):
    """Control: el piso dominicano ya coincidía y debe seguir coincidiendo."""
    front = _budget_min_total_del_frontend()
    for dias in (7, 15, 30):
        assert nc._budget_cycle_floor_dop(dias) == front["DOP"][_CICLO_A_CLAVE[dias]]


# ── C. Fuera el tipo de cambio ──────────────────────────────────────────────────────────────────

def test_el_gate_de_usd_deja_de_depender_del_tipo_de_cambio(nc, knob_on, monkeypatch):
    """Mientras el piso de USD salía de dividir pesos, una devaluación movía el mínimo de un
    usuario de Florida sin que nadie tocara nada. Se mueve el tipo de cambio a un valor absurdo y
    el veredicto no puede cambiar."""
    fd = {"weight": "75", "height": "175", "age": "35", "gender": "male",
          "activityLevel": "moderate", "goal": "maintain", "budget": "custom",
          "budgetAmount": "100", "budgetCurrency": "USD", "groceryFrequency": "weekly",
          "householdSize": "1", "country": "US"}
    monkeypatch.setattr(nc, "_budget_usd_to_dop", lambda *a, **k: 60.0)
    a = nc.validate_budget_sufficient(dict(fd))
    monkeypatch.setattr(nc, "_budget_usd_to_dop", lambda *a, **k: 5.0)
    b = nc.validate_budget_sufficient(dict(fd))
    assert a[0] == b[0] is True, (
        f"el veredicto de un presupuesto en USD cambió al mover el tipo de cambio: {a} vs {b}"
    )


def test_un_presupuesto_por_debajo_del_piso_declarado_se_bloquea(nc, knob_on):
    """La dirección del cambio: el umbral SUBE de 66,67 a 80. Nadie que pase el formulario queda
    fuera (el frontend ya bloquea en 80); se cierra la ventana de quien entra por la API."""
    fd = {"weight": "75", "height": "175", "age": "35", "gender": "male",
          "activityLevel": "moderate", "goal": "maintain", "budget": "custom",
          "budgetAmount": "70", "budgetCurrency": "USD", "groceryFrequency": "weekly",
          "householdSize": "1", "country": "US"}
    ok, detail = nc.validate_budget_sufficient(fd)
    assert ok is False, "US$70/semana pasa el gate pese a estar bajo el piso declarado (US$80)"
    assert detail, "bloquea sin explicar por qué"


def test_el_mensaje_no_habla_en_pesos_dominicanos(nc, knob_on):
    """El detalle se le enseña al usuario: un estadounidense no puede leer «RD$» en su bloqueo."""
    fd = {"weight": "75", "height": "175", "age": "35", "gender": "male",
          "activityLevel": "moderate", "goal": "maintain", "budget": "custom",
          "budgetAmount": "70", "budgetCurrency": "USD", "groceryFrequency": "weekly",
          "householdSize": "1", "country": "US"}
    _, detail = nc.validate_budget_sufficient(fd)
    assert "RD$" not in json.dumps(detail, ensure_ascii=False, default=str)


# ── D. Byte-identidad ───────────────────────────────────────────────────────────────────────────

def test_el_camino_dominicano_no_cambia(nc, knob_on):
    fd = {"weight": "75", "height": "175", "age": "35", "gender": "male",
          "activityLevel": "moderate", "goal": "maintain", "budget": "custom",
          "budgetAmount": "5000", "budgetCurrency": "DOP", "groceryFrequency": "weekly",
          "householdSize": "1", "country": "DO"}
    assert nc.validate_budget_sufficient(fd) == (True, None)


def test_con_el_knob_apagado_usd_vuelve_al_camino_historico(nc, monkeypatch):
    """El contrato de rollback del sistema de países: knob apagado ⇒ conducta pre-Fase-1 EXACTA,
    o sea USD por el `else` con conversión FX contra el piso dominicano."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
    fd = {"weight": "75", "height": "175", "age": "35", "gender": "male",
          "activityLevel": "moderate", "goal": "maintain", "budget": "custom",
          "budgetAmount": "70", "budgetCurrency": "USD", "groceryFrequency": "weekly",
          "householdSize": "1", "country": "US"}
    ok, _ = nc.validate_budget_sufficient(fd)
    assert ok is True, "con el knob apagado, US$70×60 = 4200 DOP supera el piso de 4000 y pasaba"
