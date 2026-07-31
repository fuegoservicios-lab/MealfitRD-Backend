"""[P1-DAYGEN-TIER-MODEL · 2026-07-31] Generador de días enrutado por TIER.

Decisión del owner tras el A/B medido con el índice de calidad (2026-07-31):
"deepseek medium dura mucho, el ganador es gpt 5.6 luna medium; agrega el 5.6
medium en las cuentas plus nadamás por ahora y agrega el low o sin pensamiento
en las cuentas gratis; deja a deepseek en lugares donde no sea necesario luna".

Números que sostienen la decisión (índice 0-100, cuenta real):
  · luna-medium 95,4 (coherencia 90)  vs  flash base 82-92 (coherencia 57-83)
  · luna-high 90,8 — PEOR que medium en todo con 3× latencia
  · flash+thinking-medium 76,9 — DESCALIFICADO: 36k tokens de razonamiento,
    266 s/día, día muerto contra el techo → plan degradado

Contratos que ancla:
  A. Tier map: plus/ultra → (gpt-5.6-luna, medium); gratis/basic/desconocido →
     (gpt-5.6-luna, low). Fail-cheap simétrico al reviewer: sin contexto → free.
  B. Knobs per-tier MEALFIT_DAYGEN_{MODEL,EFFORT}_{PLUS,FREE} ganan sobre el
     default; effort inválido cae al default del tier (nunca a uno más caro).
  C. Fail-safe: modelo OpenAI sin OPENAI_API_KEY → (None, "") ⇒ cadena base
     [flash, red] de P1-FLASH-PRIMARY intacta.
  D. La CADENA pone el primario del tier delante y conserva flash de red:
     [luna, flash, ...] — un fallo de Luna cae a flash sin razonamiento.
  E. El effort del tier aplica SOLO al modelo primario del tier: la red (flash)
     jamás hereda thinking (la red rescata, no profundiza — flash-medium fue
     descalificado justo por eso). El knob GLOBAL MEALFIT_DAYGEN_EFFORT
     (experimentos) sigue ganando sobre el del tier.
  F. Marker bumpeado.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parent.parent
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_APP_SRC = (_BACKEND / "app.py").read_text(encoding="utf-8")

_NON = {"user_id": "u-test", "medicalConditions": []}


@pytest.fixture()
def _clean(monkeypatch):
    for knob in (
        "MEALFIT_DAYGEN_MODEL_PLUS", "MEALFIT_DAYGEN_EFFORT_PLUS",
        "MEALFIT_DAYGEN_MODEL_FREE", "MEALFIT_DAYGEN_EFFORT_FREE",
    ):
        monkeypatch.delenv(knob, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setattr(go, "DAYGEN_EFFORT", "")


def _set_tier(monkeypatch, tier):
    monkeypatch.setattr(go, "get_user_tier", lambda _uid: tier)


# ---------------------------------------------------------------- A. tier map
@pytest.mark.parametrize("tier,esperado", [
    ("plus", ("gpt-5.6-luna", "medium")),
    ("ultra", ("gpt-5.6-luna", "medium")),
    ("gratis", ("gpt-5.6-luna", "low")),
    ("basic", ("gpt-5.6-luna", "low")),
    ("free", ("gpt-5.6-luna", "low")),   # alias legacy en DB → rama free
])
def test_tier_map(_clean, monkeypatch, tier, esperado):
    _set_tier(monkeypatch, tier)
    assert go._daygen_tier_profile() == esperado


def test_sin_contexto_cae_a_free(_clean, monkeypatch):
    """get_user_tier explotando NO puede regalar el modelo caro."""
    def _boom(_uid):
        raise RuntimeError("db caída")
    monkeypatch.setattr(go, "get_user_tier", _boom)
    assert go._daygen_tier_profile() == ("gpt-5.6-luna", "low")


# ---------------------------------------------------------------- B. knobs
def test_knobs_per_tier_ganan(_clean, monkeypatch):
    _set_tier(monkeypatch, "plus")
    monkeypatch.setenv("MEALFIT_DAYGEN_MODEL_PLUS", "deepseek-v4-flash")
    monkeypatch.setenv("MEALFIT_DAYGEN_EFFORT_PLUS", "none")
    assert go._daygen_tier_profile() == ("deepseek-v4-flash", "none")


def test_effort_invalido_cae_al_default_del_tier(_clean, monkeypatch):
    """Fail-safe hacia el default, NUNCA hacia un esfuerzo más caro (el
    razonamiento se factura como output)."""
    _set_tier(monkeypatch, "gratis")
    monkeypatch.setenv("MEALFIT_DAYGEN_EFFORT_FREE", "turbo-maximo")
    assert go._daygen_tier_profile()[1] == "low"


# ---------------------------------------------------------------- C. fail-safe
def test_sin_openai_key_devuelve_none(_clean, monkeypatch):
    _set_tier(monkeypatch, "plus")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert go._daygen_tier_profile() == (None, "")


# ---------------------------------------------------------------- D. cadena
def test_cadena_pone_tier_delante_y_flash_de_red(_clean, monkeypatch):
    _set_tier(monkeypatch, "plus")
    monkeypatch.setattr(go, "DAYGEN_CANARY_MODEL", "")
    monkeypatch.setattr(go, "DAYGEN_CANARY_PCT", 0)
    ch = go._day_model_chain(_NON, 1)
    assert ch[0] == "gpt-5.6-luna", "el primario del tier va DELANTE"
    assert "deepseek-v4-flash" in ch[1:], "flash DEBE quedar de red"


def test_cadena_sin_key_es_la_base(_clean, monkeypatch):
    _set_tier(monkeypatch, "plus")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(go, "DAYGEN_CANARY_MODEL", "")
    monkeypatch.setattr(go, "DAYGEN_CANARY_PCT", 0)
    ch = go._day_model_chain(_NON, 1)
    assert ch[0] == go._FLASH_MODEL_NAME, (
        "sin OPENAI_API_KEY la cadena vuelve a la base flash-primaria "
        "(P1-FLASH-PRIMARY) — jamás un modelo incobrable delante"
    )


# ---------------------------------------------------------------- E. effort
def test_effort_del_tier_solo_al_primario():
    """Anclaje estructural: el effort del tier se aplica con el guard
    `_t_model == _model` — la red no hereda thinking. Un fallback que razona
    4 minutos no rescata nada (flash-medium: 266 s/día, plan degradado)."""
    i = _GO_SRC.index("_es_openai = is_openai_model(_model)")
    win = _GO_SRC[i:i + 1400]
    assert "_t_model == _model" in win, (
        "el effort del tier DEBE gatearse al modelo primario del tier; sin el "
        "guard, flash-de-red heredaría el thinking que lo descalificó"
    )
    assert "_eff = DAYGEN_EFFORT" in win, (
        "el knob global (experimentos A/B) debe seguir ganando sobre el tier"
    )


# ---------------------------------------------------------------- F. marker
def test_marker_bumpeado():
    assert re.search(r'_LAST_KNOWN_PFIX = "P1-DAYGEN-TIER-MODEL', _APP_SRC)
