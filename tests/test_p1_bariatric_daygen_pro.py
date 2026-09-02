"""[P1-BARIATRIC-DAYGEN-PRO · 2026-06-28] Dos mejoras a la GENERACIÓN bariátrica (no al fallback), del workflow de diseño
"motor LLM-suficiente":

FASE 1 — _route_model_for_day_generator: el perfil bariátrico (clínicamente el más denso) gana sobre el routing por
tier Y sobre el override DAYGEN_LITE_FOR_EASY. Fail-safe: detección falla → routing normal.

[P1-FLASH-PRIMARY · 2026-07-31] El branch bariátrico ya NO devuelve PRO: devuelve
`MEALFIT_BARIATRIC_DAYGEN_MODEL` (default `_FLASH_MODEL_NAME`) — el owner midió que flash es actualmente MEJOR que
pro, lo que invalida la premisa original ("PRO más capaz para el perfil denso"). La garantía VIGENTE del branch:
bariátrico NUNCA degrada a flash-lite y NUNCA entra al canario. Rollback per-feature:
`MEALFIT_BARIATRIC_DAYGEN_MODEL=glm-5.3`.

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


def test_bariatric_routes_to_flash_primary_attempt1(monkeypatch):
    # [P1-FLASH-PRIMARY] era == "pro-test"; hoy el default del branch es _FLASH_MODEL_NAME.
    _distinct_models(monkeypatch)
    monkeypatch.delenv("MEALFIT_BARIATRIC_DAYGEN_MODEL", raising=False)
    assert g._route_model_for_day_generator(_BAR, 1) == "flash-test"


def test_bariatric_model_knob_rollback(monkeypatch):
    # Rollback per-feature sin redeploy: el knob puede devolver el branch a pro.
    _distinct_models(monkeypatch)
    monkeypatch.setenv("MEALFIT_BARIATRIC_DAYGEN_MODEL", "glm-5.3")
    assert g._route_model_for_day_generator(_BAR, 1) == "glm-5.3"


def test_non_bariatric_stays_flash_free_tier(monkeypatch):
    _distinct_models(monkeypatch)
    assert g._route_model_for_day_generator(_NON, 1) == "flash-test"


def test_knob_off_reverts_to_tier(monkeypatch):
    _distinct_models(monkeypatch)
    monkeypatch.setattr(g, "BARIATRIC_DAYGEN_PRO", False)
    assert g._route_model_for_day_generator(_BAR, 1) == "flash-test"  # vuelve a tier (gratis→flash)


def test_bariatric_beats_daygen_lite(monkeypatch):
    # DAYGEN_LITE_FOR_EASY degradaría a flash-lite; el override bariátrico debe ganar.
    # [P1-FLASH-PRIMARY] La garantía VIGENTE es "nunca lite" (flash full), ya no "pro".
    _distinct_models(monkeypatch)
    monkeypatch.delenv("MEALFIT_BARIATRIC_DAYGEN_MODEL", raising=False)
    monkeypatch.setattr(g, "DAYGEN_LITE_FOR_EASY", True)
    monkeypatch.setattr(g, "DAYGEN_EASY_MODEL", "flash-lite-test")
    _resolved = g._route_model_for_day_generator(_BAR, 1)
    assert _resolved == "flash-test"
    assert _resolved != "flash-lite-test"


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
