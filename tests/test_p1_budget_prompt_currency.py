"""[P1-BUDGET-PROMPT-CURRENCY · 2026-08-21] Al LLM se le decía «RD$15,000 (pesos dominicanos)»
para un plan español, y la guía cualitativa que lo acompaña era criolla.

F1 FINAL-FIX F3 ya cerró la mitad visible: si `budgetCurrency` es EUR/MXN/COP, el prompt dice
«15,000 EUR» y no miente. Lo que quedó abierto es el caso que **de hecho ocurre**: medido en
Neon, las 8 filas de `user_profiles` tienen `budgetCurrency` en 'DOP' o NULL — **cero usuarios
con moneda beta**, incluida la cuenta que generó los dos planes beta. Y con DOP el gate de F3 no
entra:

    ES  budgetCurrency=DOP  ->  «presupuesto TOTAL de RD$15,000 (pesos dominicanos)»
    US  budgetCurrency=DOP  ->  «presupuesto TOTAL de RD$15,000 (pesos dominicanos)»
    MX  budgetCurrency=DOP  ->  «presupuesto TOTAL de RD$15,000 (pesos dominicanos)»

POR QUÉ NO BASTA CON CAMBIAR EL SÍMBOLO. El número lo tecleó el usuario en un campo rotulado
«RD$» —así estaba el wizard antes de P1-QCOUNTRY-BEFORE-BUDGET— viviendo en España. No sabemos
si quiso decir pesos dominicanos o euros, y **ninguna de las dos lecturas es defendible**:
reetiquetarlo como euros inventa un dato, y dejarlo como pesos le pide al modelo que planifique
una compra española con un presupuesto dominicano. Lo honesto es no afirmar ninguna moneda:
se conserva la guía CUALITATIVA (ajustado / moderado / holgado), que es señal real y no depende
de la unidad, y se omite la cifra.

Y LA GUÍA CUALITATIVA TAMBIÉN ERA CRIOLLA. El tramo «ajustado» recomienda «guineo, batata» y el
resto nombra productos locales dominicanos — `prompts/*.py` quedó fuera del barrido de país de
Fase 1. Para un país beta esos nombres se sustituyen por la categoría genérica, que es lo que la
guía quiere decir de verdad.

Cubre:
  A. Byte-identidad dominicana (con el knob encendido y apagado).
  B. El país beta con moneda propia conserva lo que F3 arregló.
  C. El país beta con DOP deja de recibir una cifra en una moneda que no es la suya.
  D. La guía cualitativa sobrevive: se pierde la cifra, no la señal.
  E. La guía cualitativa deja de nombrar productos criollos en beta.
  F. Parser-based.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PG_PATH = _BACKEND_ROOT / "prompts" / "plan_generator.py"


@pytest.fixture(scope="module")
def build():
    from prompts.plan_generator import build_budget_context
    return build_budget_context


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


def _fd(country, currency="DOP", budget="custom", amount=15000):
    return {"budget": budget, "budgetAmount": amount, "budgetCurrency": currency,
            "groceryDuration": "monthly", "country": country}


# ── A. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

def test_el_dominicano_sigue_viendo_su_monto_en_rd(build, knob_on):
    out = build(_fd("DO"))
    assert "RD$15,000" in out and "pesos dominicanos" in out


def test_con_el_knob_apagado_el_beta_cae_a_dominicano(build, monkeypatch):
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    assert build(_fd("ES")) == build(_fd("DO"))


# ── B. Lo que F3 arregló no se toca ─────────────────────────────────────────────────────────────

def test_el_pais_beta_con_su_moneda_conserva_la_cifra(build, knob_on):
    """Cuando SÍ sabemos la moneda, la cifra es información buena y se queda. Este control impide
    que el fix se lleve por delante lo que F1 FINAL-FIX F3 ya había resuelto."""
    out = build(_fd("ES", currency="EUR"))
    assert "15,000 EUR" in out
    assert "RD$" not in out


# ── C/D. El caso que de hecho ocurre ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["ES", "MX", "CO", "PR", "US"])
def test_el_pais_beta_con_dop_no_recibe_una_cifra_en_pesos_dominicanos(build, knob_on, cc):
    """RED pre-fix: los 5 recibían «RD$15,000 (pesos dominicanos)». Es el estado real de los 8
    perfiles vivos, no un caso hipotético."""
    out = build(_fd(cc))
    assert "RD$" not in out, f"{cc}: el prompt sigue afirmando pesos dominicanos"
    assert "pesos dominicanos" not in out


@pytest.mark.parametrize("cc", ["ES", "MX", "US"])
def test_la_senal_cualitativa_sobrevive(build, knob_on, cc):
    """Se pierde la CIFRA, no la SEÑAL. Vaciar el bloque entero habría sido el otro error: el
    nivel de presupuesto orienta la selección de ingredientes y no depende de la unidad."""
    out = build(_fd(cc, budget="low", amount=None))
    assert out.strip(), f"{cc}: el bloque de presupuesto se quedó vacío"
    assert "AJUSTADO" in out


def test_el_monto_custom_sin_moneda_fiable_conserva_el_nivel(build, knob_on):
    """El caso exacto del usuario vivo: budget='custom' + cifra en DOP + país beta. Se omite la
    cifra y se conserva la orientación, en vez de dejar al modelo sin ninguna guía de coste."""
    out = build(_fd("ES"))
    assert out.strip()
    assert "presupuesto" in out.lower()


# ── E. La guía cualitativa deja de ser criolla ──────────────────────────────────────────────────

@pytest.mark.parametrize("criollo", ["guineo", "batata"])
def test_la_guia_cualitativa_no_nombra_productos_criollos_en_beta(build, knob_on, criollo):
    """`prompts/*.py` quedó fuera del barrido de país de Fase 1. El tramo «ajustado» recomienda
    guineo y batata por nombre — para un español son dos palabras que no usa."""
    out = build(_fd("ES", budget="low", amount=None)).lower()
    assert criollo not in out, f"la guía de presupuesto beta sigue nombrando «{criollo}»"


def test_la_guia_dominicana_conserva_sus_nombres(build, knob_on):
    """Control: en RD esos nombres son los correctos y se quedan."""
    out = build(_fd("DO", budget="low", amount=None)).lower()
    assert "guineo" in out and "batata" in out


# ── F. Parser-based ─────────────────────────────────────────────────────────────────────────────

def test_el_fuente_declara_el_marker_y_la_puerta_unica():
    src = _PG_PATH.read_text(encoding="utf-8", errors="replace")
    assert "P1-BUDGET-PROMPT-CURRENCY" in src
    i = src.find("def build_budget_context")
    assert i > 0
    _fin = src.find("\ndef ", i + 1)
    cuerpo = src[i:_fin if _fin > 0 else len(src)]
    assert "country_for_form_data" in cuerpo, (
        "el bloque de presupuesto no deriva el país por la única puerta del motor"
    )
