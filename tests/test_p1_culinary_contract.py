"""[P1-CULINARY-CONTRACT · 2026-07-31] Capa determinista de coherencia culinaria
(F1 del spec): migración de metadata + scan V1/V2/V3 + 3 superficies en warn."""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_MIG = _BACKEND / "migrations" / "p1_culinary_metadata_master_ingredients_2026_07_31.sql"
_MIG_ROOT = _BACKEND.parent / "migrations" / _MIG.name


def test_migracion_existe_en_ambos_dirs_identica():
    assert _MIG.exists(), "falta la migración en backend/migrations/"
    assert _MIG_ROOT.exists(), "falta la copia en migrations/ (P3-MIGRATIONS-SSOT)"
    assert _MIG.read_bytes() == _MIG_ROOT.read_bytes(), "las dos copias divergen"


def test_migracion_idempotente_y_con_sanity():
    sql = _MIG.read_text(encoding="utf-8")
    assert sql.count("ADD COLUMN IF NOT EXISTS") >= 2
    assert "prep_methods text[]" in sql and "ready_to_eat boolean" in sql
    assert "DO $$" in sql, "falta el bloque sanity DO $$"
    # Regex (no substring literal): las filas de Backfill 1 combinan la categoría con
    # el filtro IS NULL en la misma cláusula WHERE ("WHERE category = 'X' AND
    # prep_methods IS NULL"), así que la subcadena "WHERE prep_methods IS NULL" sola
    # nunca aparece verbatim aunque el filtro SÍ esté aplicado.
    assert re.search(r"WHERE\b.*prep_methods IS NULL", sql), (
        "el backfill debe filtrar IS NULL para ser re-ejecutable sin pisar "
        "overrides manuales posteriores")


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT · Task 4] Matching + V1 (verbo↔alimento). Módulo puro
# `culinary_coherence.py` — sin env vars, sin LLM, sin DB.
# ---------------------------------------------------------------------------
import culinary_coherence as cc

_CAT = [
    {"name": "Casabe", "prep_methods": ["tostar", "ninguno"], "ready_to_eat": True},
    {"name": "Pechuga de pollo", "prep_methods": ["hervir", "plancha", "freir", "hornear", "guisar", "saltear"], "ready_to_eat": False},
    {"name": "Repollo", "prep_methods": ["hervir", "saltear", "crudo"], "ready_to_eat": None},
    {"name": "Pollo guisado", "prep_methods": ["guisar"], "ready_to_eat": False},
    {"name": "Tomate", "prep_methods": ["crudo", "saltear"], "ready_to_eat": True},
    {"name": "Bistec de res", "prep_methods": ["plancha", "guisar"], "ready_to_eat": False},
    {"name": "Misterio sin metadata", "prep_methods": None, "ready_to_eat": None},
]


def _plan(pasos, ingredientes=None, nombre="Plato de prueba"):
    return {"days": [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": nombre,
        "ingredients": ingredientes or ["100 g Pechuga de pollo"],
        "recipe": pasos}]}]}


def test_v1_verbo_imposible_sobre_ready_to_eat():
    v = cc.culinary_contract_scan(_plan(["Cuece el Casabe según el paquete."],
                                        ["30 g Casabe"]), _CAT)
    assert any(x["check"] == "V1" and x["food"] == "Casabe" for x in v), v


def test_v1_metodo_fuera_de_prep_methods():
    v = cc.culinary_contract_scan(_plan(["Licúa el Bistec de res hasta obtener crema."],
                                        ["120 g Bistec de res"]), _CAT)
    assert any(x["check"] == "V1" and x["food"] == "Bistec de res" for x in v), v


def test_v1_no_dispara_sobre_metodo_valido():
    v = cc.culinary_contract_scan(_plan(["Cocina la Pechuga de pollo a la plancha."]), _CAT)
    assert not [x for x in v if x["check"] == "V1"], v


def test_v1_word_boundary_pollo_no_es_repollo():
    """pollo⊂repollo: 'Saltea el Repollo' NO debe leerse como pollo."""
    v = cc.culinary_contract_scan(_plan(["Saltea el Repollo con ajo."], ["80 g Repollo"]), _CAT)
    assert not [x for x in v if x["check"] == "V1"], v


def test_v1_alias_mas_largo_gana():
    """'Pollo guisado' (plato del catálogo) menciona 'guisa' — el match debe ser
    el alias LARGO, no 'pollo' suelto con verbo guisar."""
    v = cc.culinary_contract_scan(_plan(["Guisa el Pollo guisado a fuego lento."],
                                        ["150 g Pollo guisado"]), _CAT)
    assert not [x for x in v if x["check"] == "V1"], v


def test_v1_fail_open_sin_metadata():
    v = cc.culinary_contract_scan(_plan(["Hornea el Misterio sin metadata 20 min."],
                                        ["50 g Misterio sin metadata"]), _CAT)
    assert not v, "sin metadata el scan debe callar (fail-open), no inventar"


def test_plural_bidireccional():
    """FP real del dry-run: '2 tomates' (ingrediente) vs 'el tomate' (paso)."""
    assert cc.find_catalog_foods("Ralla el tomate encima.", cc.build_culinary_index(_CAT)) == ["Tomate"]
    assert cc.find_catalog_foods("2½ tomates maduros", cc.build_culinary_index(_CAT)) == ["Tomate"]


# --- Resoluciones de ambigüedad del controller (task-4-brief.md), ganan sobre
# la letra del brief. Ver comentarios inline en culinary_coherence.py. ---

def test_v1_sofreir_no_es_freir_permite_saltear():
    """RESOLUCIÓN 1: 'sofr[ií]\\w*' vive bajo 'saltear', NO 'freir'. Tomate
    acepta 'saltear' pero no 'freir' — si sofreír colgara de freir, este paso
    dispararía un V1 falso-positivo sobre una técnica legítima RD."""
    v = cc.culinary_contract_scan(_plan(["Sofríe el Tomate picado."],
                                        ["50 g Tomate"]), _CAT)
    assert not [x for x in v if x["check"] == "V1"], v


def test_v1_multi_alimento_no_acusa_acompanantes():
    """RESOLUCIÓN 2: 'Hierve el Repollo y sirve con Casabe' — Repollo SÍ acepta
    hervir; Casabe (ready-to-eat) NO. Con ≥2 alimentos y ≥1 destinatario válido,
    el check no acusa al acompañante Casabe."""
    v = cc.culinary_contract_scan(_plan(["Hierve el Repollo y sirve con Casabe."],
                                        ["80 g Repollo", "30 g Casabe"]), _CAT)
    assert not [x for x in v if x["check"] == "V1"], v


def test_v1_multi_alimento_sin_destinatario_valido_acusa_a_todos():
    """RESOLUCIÓN 2, contraste: si NINGÚN alimento del paso acepta el método,
    la salvaguarda no aplica — ambos deben acusarse (no hay 'destinatario
    válido' que la lea como un paso legítimo con acompañante inocente)."""
    v = cc.culinary_contract_scan(_plan(["Licúa el Casabe y el Bistec de res."],
                                        ["30 g Casabe", "120 g Bistec de res"]), _CAT)
    v1_foods = {x["food"] for x in v if x["check"] == "V1"}
    assert v1_foods == {"Casabe", "Bistec de res"}, v
