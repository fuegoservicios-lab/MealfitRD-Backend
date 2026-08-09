# [P1-LEGUME-DRY-WORDING · 2026-08-08] La última costura de vegana_dm2 (issue #14): el rewrite
# cocido→seco (P1-COOKED-GRAIN-DRY, para que la lista compre paquetes secos) reescribía
# «40 g de habichuelas cocidas» como «15 g de habichuelas CRUDAS» — y el reviewer rechazaba el
# plan entero por fitohemaglutinina (food-safety REAL: "deben cocinarse completamente"). Para
# granos «crudo» es inocuo; para LEGUMINOSAS la palabra culinaria correcta es «secas» — misma
# conversión, cero implicación de consumo crudo. El productor se encontró por las cantidades
# (15-17 g = 40 g cocidos × 127/340) tras descartar closer (formato "15g") y micro-seed
# (solo omega3/vitE/vitA).
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import graph_orchestrator as go


def _days(line):
    return [{"day": 1, "meals": [{"name": "Cena", "ingredients": [line]}]}]


def _rewrite(monkeypatch, line, kcal_idx):
    monkeypatch.setattr(go, "_catalog_kcal_by_name", lambda: kcal_idx)
    days = _days(line)
    telem = go._normalize_cooked_grain_lines(days)
    return days[0]["meals"][0]["ingredients"][0], telem


def test_leguminosa_se_reescribe_como_secas(monkeypatch):
    new, telem = _rewrite(monkeypatch, "40 g de habichuelas negras cocidas",
                          {"habichuelas negras": 340.0})
    assert telem, "la conversión cocido→seco debe seguir ocurriendo (la lista compra seco)"
    assert "secas" in new, f"leguminosa seca se dice 'secas', no 'crudas': {new!r}"
    assert "crud" not in new, f"'crudas' dispara el food-safety del reviewer: {new!r}"
    assert "15 g" in new, f"la conversión 40×127/340≈15 debe preservarse: {new!r}"


def test_grano_conserva_crudo(monkeypatch):
    new, telem = _rewrite(monkeypatch, "130 g de arroz blanco cocido",
                          {"arroz blanco": 360.0})
    assert telem and "crudo" in new, f"los granos conservan el wording existente: {new!r}"


def test_lenteja_masculino_plural(monkeypatch):
    new, _ = _rewrite(monkeypatch, "40 g de gandules cocidos", {"gandules": 340.0})
    assert "secos" in new and "crud" not in new, new
