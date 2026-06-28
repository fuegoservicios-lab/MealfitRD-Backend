"""[P1-BARIATRIC-DAYGEN-PRO · 2026-06-28] Dos mejoras a la GENERACIÓN bariátrica (no al fallback), del workflow de diseño
"motor LLM-suficiente":

FASE 1 — _route_model_for_day_generator: el perfil bariátrico (clínicamente el más denso) usa PRO desde attempt 1 en
TODOS los tiers, ganando sobre el routing por tier (FLASH en gratis) Y sobre el override DAYGEN_LITE_FOR_EASY. Cierra la
asimetría generador-débil/revisor-fuerte. NOTA: en la config de prod actual el owner tiene MEALFIT_FLASH_MODEL=pro, así
que es un no-op defensivo hoy; pasa a tener efecto si flash se abarata. Fail-safe: detección falla → routing normal.

FASE 2 — few-shot en el prompt bariátrico (condition_rules): un día modelo completo (show-don't-tell) que ancla la FORMA
(proteína primero, porciones en gramos enteros, nombre↔ingredientes, cocido, sin azúcar).

Tests PUROS: monkeypatch de los model names + get_user_tier para no depender de Neon.
"""
from __future__ import annotations

import graph_orchestrator as g
import condition_rules as cr

_BAR = {"medicalConditions": ["Cirugía Bariátrica (manga gástrica)"]}
_NON = {"medicalConditions": ["Diabetes tipo 2"]}


def _distinct_models(monkeypatch):
    monkeypatch.setattr(g, "_FLASH_MODEL_NAME", "flash-test")
    monkeypatch.setattr(g, "_PRO_MODEL_NAME", "pro-test")
    monkeypatch.setattr(g, "get_user_tier", lambda uid: "gratis")  # sin Neon


def test_bariatric_routes_to_pro_attempt1(monkeypatch):
    _distinct_models(monkeypatch)
    assert g._route_model_for_day_generator(_BAR, 1) == "pro-test"


def test_non_bariatric_stays_flash_free_tier(monkeypatch):
    _distinct_models(monkeypatch)
    assert g._route_model_for_day_generator(_NON, 1) == "flash-test"


def test_knob_off_reverts_to_tier(monkeypatch):
    _distinct_models(monkeypatch)
    monkeypatch.setattr(g, "BARIATRIC_DAYGEN_PRO", False)
    assert g._route_model_for_day_generator(_BAR, 1) == "flash-test"  # vuelve a tier (gratis→flash)


def test_bariatric_pro_beats_daygen_lite(monkeypatch):
    # DAYGEN_LITE_FOR_EASY degradaría a flash-lite; el override bariátrico debe ganar
    _distinct_models(monkeypatch)
    monkeypatch.setattr(g, "DAYGEN_LITE_FOR_EASY", True)
    monkeypatch.setattr(g, "DAYGEN_EASY_MODEL", "flash-lite-test")
    assert g._route_model_for_day_generator(_BAR, 1) == "pro-test"


def test_detection_failure_is_failsafe(monkeypatch):
    # si detect_active_rules lanza, NO rompe: cae al routing normal
    _distinct_models(monkeypatch)
    import condition_rules
    monkeypatch.setattr(condition_rules, "detect_active_rules",
                        lambda fd: (_ for _ in ()).throw(RuntimeError("boom")))
    assert g._route_model_for_day_generator(_BAR, 1) == "flash-test"  # degradó, no crasheó


def test_fewshot_in_bariatric_prompt():
    b = cr.build_condition_prompt(_BAR)
    assert "EJEMPLO DE UN DÍA BARIÁTRICO CORRECTO" in b
    assert b.count("«") >= 6  # los 6 slots del día modelo
    assert "proteína blanda primero" in b


def test_fewshot_absent_in_non_bariatric_prompt():
    nb = cr.build_condition_prompt(_NON)
    assert "EJEMPLO DE UN DÍA BARIÁTRICO" not in nb


def test_knob_and_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-BARIATRIC-DAYGEN-PRO" in src
    assert "BARIATRIC_DAYGEN_PRO" in src
    assert g.BARIATRIC_DAYGEN_PRO is True
