"""[P1-COUNTRY-BUDGET-CURRENCY-DEFAULT · 2026-08-23]

Guard cross-repo del SSOT de moneda. El comportamiento visible se prueba montando
los pasos reales en Vitest; aquí se protege que las filas frontend espejen todas
las monedas de COUNTRY_PROFILES y que el estado inicial no vuelva a sembrar DOP.
"""
from __future__ import annotations

import re
from pathlib import Path

import constants


_BACKEND = Path(__file__).resolve().parents[1]
_FRONTEND = _BACKEND.parent / "frontend"


def _countries_source() -> str:
    return (_FRONTEND / "src" / "config" / "countries.js").read_text(encoding="utf-8")


def _country_rows() -> dict[str, str]:
    source = re.sub(r"//.*", "", _countries_source())
    return dict(re.findall(
        r"code:\s*'([A-Z]{2})'[^}]*currency:\s*'([A-Z]{3})'",
        source,
    ))


def test_cada_fila_frontend_espeja_la_moneda_del_backend():
    rows = _country_rows()
    assert rows, "countries.js no expone currency en sus filas"
    assert rows == {
        code: profile["currency"]
        for code, profile in constants.COUNTRY_PROFILES.items()
    }


def test_assessment_no_siembra_dop_como_si_fuera_eleccion():
    source = (_FRONTEND / "src" / "context" / "AssessmentContext.jsx").read_text(
        encoding="utf-8"
    )
    match = re.search(r"const\s+initialFormData\s*=\s*\{([\s\S]*?)\n\s*\};", source)
    assert match, "no se pudo aislar initialFormData"
    budget = re.search(r"budgetCurrency:\s*([^,}\n]+)", match.group(1))
    assert budget, "initialFormData perdió la clave budgetCurrency"
    assert budget.group(1).strip() in ("''", '""', "null"), (
        "budgetCurrency vuelve a nacer como una elección DOP; el fallback por país no correrá"
    )


def test_marker_movil_y_guard_frontend_existen():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert '_LAST_KNOWN_PFIX = "P' in app and " · 2026-" in app
    guard = _FRONTEND / "src" / "__tests__" / "countryBudgetCurrencyDefault.p1.test.jsx"
    assert guard.exists()
