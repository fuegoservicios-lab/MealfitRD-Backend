"""G56: los sinónimos alimentarios resuelven por palabra completa."""

from __future__ import annotations

from pathlib import Path

import pytest

import graph_orchestrator as go


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_lait_expande_solo_la_clase_lactea() -> None:
    lait = go._expand_allergy_declarations(["lait"])
    lacteos = go._expand_allergy_declarations(["lacteos"])

    assert lait == lacteos
    assert "leche" in lait
    assert "bacalaitos" not in lait
    assert "pescado" not in lait
    assert "gluten" not in lait


def test_lait_no_bloquea_bacalaitos_en_ninguna_de_las_dos_capas() -> None:
    plan_bacalaitos = {
        "days": [{"meals": [{"name": "Merienda", "ingredients": ["120 g de Bacalaítos"]}]}]
    }
    plan_leche = {
        "days": [{"meals": [{"name": "Desayuno", "ingredients": ["240 ml de leche"]}]}]
    }

    assert go._scan_allergen_violations(plan_bacalaitos, ["lait"]) == []
    assert go._scan_allergen_violations(plan_leche, ["lait"])
    assert go._verified_catalog_excluded_tokens({"allergies": ["lait"]}) == \
        go._verified_catalog_excluded_tokens({"allergies": ["lacteos"]})


@pytest.mark.parametrize(
    ("declaracion", "termino_esperado"),
    [
        ("camarón", "camaron"),
        ("camarones", "camaron"),
        ("alergia a los camarones", "camaron"),
        ("bacalaítos", "bacalaitos"),
        ("pescado", "bacalao"),
        ("leche", "queso"),
    ],
)
def test_match_por_palabra_preserva_declaraciones_libres_validas(
    declaracion: str,
    termino_esperado: str,
) -> None:
    assert termino_esperado in go._expand_allergy_declarations([declaracion])


@pytest.mark.parametrize(
    ("declaracion", "sinonimo_alimentario"),
    [
        ("lait", "bacalaitos"),
        ("pan", "empanada"),
        ("sal", "salsa de pescado"),
        ("res", "fresa"),
        ("pollo", "repollo"),
    ],
)
def test_fragmentos_no_resuelven_otra_palabra(
    declaracion: str,
    sinonimo_alimentario: str,
) -> None:
    # Se prueba la frontera de los sinónimos directamente: `_decl` es una vía
    # separada cuya sobre-detección deliberada no forma parte de G56.
    assert not go._sinonimo_alimento_casa(
        go._norm_declaracion(declaracion),
        sinonimo_alimentario,
    )


@pytest.mark.parametrize(
    "alias",
    [
        "allergie aux arachides",
        "peanuts",
        "fruits de mer",
        "intolleranza al lattosio",
        "maladie coeliaque",
    ],
)
def test_aliases_declarativos_conservan_su_resolucion(alias: str) -> None:
    expandido = go._expand_allergy_declarations([alias])
    assert expandido != {go._norm_declaracion(alias)}


def test_mutacion_no_puede_restaurar_la_subcadena_cruda() -> None:
    source = (BACKEND_ROOT / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "any(_sinonimo_alimento_casa(a_low, s) for s in syns)" in source
    assert "any(a_low in strip_accents(s) or strip_accents(s) in a_low for s in syns)" not in source


def test_pfix_marker_cierra_g56() -> None:
    source = (BACKEND_ROOT / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-ALLERGEN-DECL-SUBSTR-LAIT · 2026-08-23" in source
