# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-290 · 2026-09-25] Suplementos en la Alacena: datos, aislamiento del generador y Nevera que se
enciende sola. Spec: docs/superpowers/specs/2026-09-25-suplementos-alacena-design.md. Tooltip-anchor: P1-PLAN-LOTE-290"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel):
    return (_BACKEND / rel).read_text(encoding="utf-8")


WHEY = {"serving_g": 31, "kcal": 120, "protein_g": 24, "carbs_g": 3, "fats_g": 1.5}


# ── Task 1: datos y SSOT ────────────────────────────────────────────────────────────────────────────────────────────

def test_macros_de_porciones_multiplica_la_etiqueta():
    import suplementos as s
    assert s.macros_de_porciones(WHEY, 2) == {"kcal": 240.0, "protein_g": 48.0, "carbs_g": 6.0, "fats_g": 3.0}
    assert s.macros_de_porciones(None, 2) is None
    assert s.macros_de_porciones(WHEY, 0) is None


def test_etiqueta_absurda_se_rechaza():
    import suplementos as s
    assert s.etiqueta_valida(WHEY) == WHEY
    assert s.etiqueta_valida({**WHEY, "kcal": 1200}) is None          # 1200 kcal en 31 g: foto mal leída
    assert s.etiqueta_valida({**WHEY, "protein_g": 40}) is None       # más macros que gramos de porción
    assert s.etiqueta_valida({"kcal": 0, "serving_g": 5}) == {"serving_g": 5, "kcal": 0, "protein_g": 0,
                                                              "carbs_g": 0, "fats_g": 0}   # creatina
    assert s.etiqueta_valida("no") is None


def test_estimados_cubren_las_claves_del_formulario():
    import suplementos as s
    from constants import SUPPLEMENT_NAMES
    assert set(s.ESTIMADOS) == set(SUPPLEMENT_NAMES)
    assert s.ESTIMADOS["creatine"]["kcal"] == 0
    assert s.ESTIMADOS["whey_protein"]["protein_g"] == 24


def test_migracion_idempotente_y_en_los_dos_directorios():
    rel = "p1_plan_lote_290_suplementos_alacena.sql"
    a = (_BACKEND / "migrations" / rel).read_text(encoding="utf-8")
    raiz = _BACKEND.parent / "migrations" / rel
    if raiz.parent.exists():
        assert raiz.exists() and raiz.read_text(encoding="utf-8") == a
    for frag in ("ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'food'",
                 "ADD COLUMN IF NOT EXISTS serving_label JSONB",
                 "ADD COLUMN IF NOT EXISTS serving_unit TEXT",
                 "ADD COLUMN IF NOT EXISTS label_source TEXT",
                 "DROP CONSTRAINT IF EXISTS user_inventory_kind_check",
                 "RAISE EXCEPTION"):
        assert frag in a, frag
