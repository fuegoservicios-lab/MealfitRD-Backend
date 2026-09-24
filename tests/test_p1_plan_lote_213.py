# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-213 · 2026-09-24] Regla (b): cuando el revisor va a EXIGIR la Nevera, el esqueleto se queda con ella.

Dos reglas se contradecían: P2-PANTRY-ROTATION-FLOOR («la Nevera es un piso») completaba el esqueleto con el sorteo del
catálogo cuando la Nevera traía pocas proteínas, y el revisor —con la Nevera exigida— rechaza todo ingrediente que no
esté en ella. Si el modelo usaba la proteína asignada: rechazo por Nevera; si no: rechazo por «omitió proteínas
asignadas». El dueño: «haz la que consideres mejor». Gana el revisor (y la compra única: cocinar con lo comprado). Donde
el revisor no la exige —renovación con variedad, Nevera de referencia, Nevera bajo el piso de viabilidad— el piso sigue.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import ai_helpers as ah  # noqa: E402
import nevera_exigida as ne  # noqa: E402

# 12+ alimentos (el piso de viabilidad), UNA sola proteína reconocible
NEVERA_VIABLE = ["20 Huevo", "Arroz blanco", "Yuca", "Plátano verde", "Batata", "Cebolla", "Ajo",
                 "Aceite de oliva", "Avena", "Zanahoria", "Repollo", "Limón", "Orégano", "Sal"]


def test_manda_solo_si_el_revisor_la_exige_y_es_viable():
    assert ne.nevera_manda({"current_pantry_ingredients": NEVERA_VIABLE}) is True
    assert ne.nevera_manda({"current_pantry_ingredients": NEVERA_VIABLE, "_pantry_advisory_only": True}) is False
    assert ne.nevera_manda({"current_pantry_ingredients": ["Huevo", "Arroz"]}) is False, "bajo el piso: insatisfacible"
    assert ne.nevera_manda({}) is False and ne.nevera_manda(None) is False
    # espejo: nunca manda donde el revisor no valida
    for fd in ({"current_pantry_ingredients": NEVERA_VIABLE, "update_reason": "renewal"},
               {"current_pantry_ingredients": [], "_is_rotation_reroll": False}):
        if ne.lista(fd) is None:
            assert ne.nevera_manda(fd) is False


def test_el_knob_lo_apaga(monkeypatch):
    monkeypatch.setenv("MEALFIT_CLOSERS_RESPECT_PANTRY", "false")
    assert ne.nevera_manda({"current_pantry_ingredients": NEVERA_VIABLE}) is False


def _proteinas_y_lock(caplog, pantry, **extra):
    caplog.clear()
    caplog.set_level(logging.INFO, logger=ah.logger.name)
    fd = {"current_pantry_ingredients": pantry}
    fd.update(extra)
    out = ah.get_deterministic_variety_prompt("", fd, user_id=None)
    msgs = [r.getMessage() for r in caplog.records]
    estricta = any("[P1-PLAN-LOTE-213]" in m for m in msgs)
    piso = any("[P2-PANTRY-ROTATION-FLOOR]" in m for m in msgs)
    return out, estricta, piso


def test_con_la_nevera_exigida_el_esqueleto_no_sale_de_ella(caplog):
    out, estricta, piso = _proteinas_y_lock(caplog, NEVERA_VIABLE)
    assert estricta and not piso, "con la Nevera exigida no se completa con el catálogo"
    assert "REGLA DE AHORRO EXTREMA" in out.upper(), "el esqueleto queda fijado a lo que la Nevera tiene"
    m = [r.getMessage() for r in caplog.records if "[P1-PLAN-LOTE-213]" in r.getMessage()]
    assert re.search(r"la nevera aportó 1 proteína", m[0]), m


def test_donde_el_revisor_no_la_exige_el_piso_sigue(caplog):
    out, estricta, piso = _proteinas_y_lock(caplog, NEVERA_VIABLE, _pantry_advisory_only=True)
    assert piso and not estricta, "Nevera de referencia: la Nevera es un piso"
    assert "REGLA DE AHORRO EXTREMA" not in out.upper()


def test_cableado():
    src = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    assert '_nevera_estricta = __import__("nevera_exigida").nevera_manda(form_data)' in src
    assert "if len(extracted_p) >= _min_p or _nevera_estricta:" in src
    assert "if len(_extracted) >= _min or _nevera_estricta:" in src
    # el ancla de P2-PANTRY-ROTATION-FLOOR sigue en pie
    assert "_pantry_sustains_rotation = bool(extracted_p) and len(extracted_p) >= _min_p" in src
    assert "tooltip-anchor: P1-PLAN-LOTE-213-NEVERA-MANDA" in (_BACKEND / "nevera_exigida.py").read_text(encoding="utf-8")
