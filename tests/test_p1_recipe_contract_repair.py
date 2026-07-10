"""[P1-RECIPE-CONTRACT-REPAIR · 2026-07-10] Reparación determinista del contrato de pasos ANTES
de flagear el badge "Receta con pasos incompletos — regenera para detalle".

Caso vivo (plan 3d11d96e, desayuno 'Tostadas con Mantequilla de Maní y Guineo y Queso Cottage'):
el lint `_recipe_step_contract_issues` era read-only en los persist boundaries → cualquier issue
→ advisory → badge amarillo al usuario, aunque el fix fuera mecánicamente derivable del propio
meal. Este pase compone la maquinaria que YA existe (split de cocción, backstop de tiempos,
templates de 3 pilares) para reparar las clases reparables; el badge queda SOLO para lo
genuinamente irreparable (inglés residual, receta ausente, excepciones fail-open).

Función pura (sin DB). tooltip-anchor: P1-RECIPE-CONTRACT-REPAIR
"""
from pathlib import Path

import graph_orchestrator as go

_GO_SRC = Path(go.__file__).read_text(encoding="utf-8")


def _lint(meal):
    return go._recipe_step_contract_issues(meal)


def test_knob_default_on():
    # familia de correctores deterministas (MISE_COOK_SPLIT, REVERSE-COHERENCE... default ON):
    # no puede rechazar nada; solo toca meals YA rotos; fail-open.
    assert go.RECIPE_CONTRACT_REPAIR_ENABLED is True


def test_repairs_missing_montaje():
    meal = {"name": "Pollo Guisado con Arroz",
            "ingredients": ["200g de pollo", "100g de arroz"],
            "recipe": ["Mise en place: Corta el pollo y mide el arroz. Ten todo listo.",
                       "El Toque de Fuego: Guisa el pollo 20-25 min a fuego medio."]}
    issues = _lint(meal)
    assert "falta 'Montaje'" in issues
    residual = go._repair_recipe_contract(meal, issues)
    assert residual == []                      # reparado por completo
    assert meal["_recipe_contract_repaired"] is True
    assert any(str(s).lower().startswith("montaje") for s in meal["recipe"])


def test_repairs_missing_mise():
    meal = {"name": "Tilapia a la Plancha",
            "ingredients": ["200g de tilapia", "1 limón"],
            "recipe": ["El Toque de Fuego: Cocina la tilapia a la plancha 3-4 min por lado.",
                       "Montaje: Emplata con el limón y sirve."]}
    issues = _lint(meal)
    assert "falta 'Mise en place'" in issues
    residual = go._repair_recipe_contract(meal, issues)
    assert residual == []
    assert str(meal["recipe"][0]).lower().startswith("mise en place")


def test_extracts_toasting_trapped_in_single_sentence_mise():
    # El caso de las Tostadas: cocción atrapada en un Mise de UNA oración — el split v2
    # (guard len<2) no podía extraerla. La v3 del repair la mueve y rellena el Mise con template.
    meal = {"name": "Tostadas con Mantequilla de Maní",
            "ingredients": ["2 rebanadas de pan integral", "30g de mantequilla de maní"],
            "recipe": ["Mise en place: Tuesta el pan integral hasta dorar.",
                       "Montaje: Unta la mantequilla de maní y sirve."]}
    issues = _lint(meal)
    assert "falta 'El Toque de Fuego'" in issues
    residual = go._repair_recipe_contract(meal, issues)
    assert residual == []
    joined = " | ".join(meal["recipe"]).lower()
    assert "el toque de fuego" in joined
    assert "tuesta" in joined                              # verbatim move, no inventado
    assert str(meal["recipe"][0]).lower().startswith("mise en place")  # Mise backfilled


def test_never_invents_fire_for_no_cook_meal():
    # Plato frío sin TdF: el lint (2 pilares) NO exige fuego; el repair JAMÁS lo inventa.
    meal = {"name": "Fruta Fresca con Yogurt",
            "ingredients": ["1 taza de lechosa", "1 yogurt griego"],
            "recipe": ["Mise en place: Pica la lechosa en cubos."]}
    issues = _lint(meal)
    assert "falta 'El Toque de Fuego'" not in issues       # no-cook → 2 pilares
    assert "falta 'Montaje'" in issues
    residual = go._repair_recipe_contract(meal, issues)
    assert residual == []
    joined = " ".join(meal["recipe"]).lower()
    assert "toque de fuego" not in joined                  # cero fuego inventado


def test_reorders_out_of_order_pillars():
    meal = {"name": "Res Guisada",
            "ingredients": ["200g de res"],
            "recipe": ["Montaje: Emplata la res y sirve.",
                       "Mise en place: Corta la res en cubos.",
                       "El Toque de Fuego: Guisa 25-30 min a fuego medio."]}
    issues = _lint(meal)
    assert "prefijos fuera de orden" in issues
    residual = go._repair_recipe_contract(meal, issues)
    assert residual == []
    lows = [str(s).lower() for s in meal["recipe"]]
    i_m = next(i for i, s in enumerate(lows) if s.startswith("mise en place"))
    i_f = next(i for i, s in enumerate(lows) if s.startswith("el toque de fuego"))
    i_mo = next(i for i, s in enumerate(lows) if s.startswith("montaje"))
    assert i_m < i_f < i_mo


def test_injects_missing_time_in_tdf():
    meal = {"name": "Pollo al Horno",
            "ingredients": ["200g de pollo"],
            "recipe": ["Mise en place: Sazona el pollo.",
                       "El Toque de Fuego: Hornea el pollo hasta dorar.",
                       "Montaje: Sirve caliente."]}
    issues = _lint(meal)
    assert "Toque de Fuego sin tiempo/temperatura concreta" in issues
    residual = go._repair_recipe_contract(meal, issues)
    assert residual == []                                   # backstop de tiempos lo cierra


def test_residual_english_keeps_badge():
    meal = {"name": "Pollo Salteado",
            "ingredients": ["200g de pollo"],
            "recipe": ["Mise en place: Corta el pollo.",
                       "El Toque de Fuego: Saltea 5-6 min. Add the chicken and cook until golden.",
                       "Montaje: Sirve."]}
    issues = _lint(meal)
    assert "paso con inglés residual" in issues
    residual = go._repair_recipe_contract(meal, issues)
    assert "paso con inglés residual" in residual           # irreparable → badge legítimo


def test_knob_off_returns_issues_unchanged(monkeypatch):
    monkeypatch.setattr(go, "RECIPE_CONTRACT_REPAIR_ENABLED", False)
    meal = {"name": "X", "ingredients": [], "recipe": ["El Toque de Fuego: Cocina 10 min."]}
    issues = _lint(meal)
    assert go._repair_recipe_contract(meal, issues) == issues


def test_wired_at_both_persist_callsites():
    # parser: el repair debe envolver el lint en AMBOS persist boundaries (finalize plan + update dish)
    occurrences = _GO_SRC.count("_repair_recipe_contract(")
    assert occurrences >= 3, (  # 1 def + >=2 callsites
        f"esperaba def + 2 callsites de _repair_recipe_contract, encontré {occurrences}"
    )
