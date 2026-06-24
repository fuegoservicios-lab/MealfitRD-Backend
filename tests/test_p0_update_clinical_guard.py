"""[P0-UPDATE-CLINICAL-GUARD · 2026-06-23] Regresión del backstop clínico determinista en las
superficies de UPDATE de platos (swap individual S3, regenerate-day S2, chat modify del coach).

Contexto: el grafo de generación (S1) corre `_scan_allergen_violations` + `_scan_diet_violations` +
reviewer médico en `review_plan_node`. Las superficies de edición re-arman el plato 100% con el LLM
y NO pasan por el grafo → sin este backstop un alérgico que pedía "cámbiale el pollo por camarones"
obtenía camarones persistidos, o un swap reintroducía cerdo/res en un plan vegano.

Cubre:
  1. El helper `clinical_backstop_for_meal` detecta alérgenos y violaciones de dieta, y es no-op con
     entradas limpias (clave para no romper los swaps normales).
  2. Las 3 superficies invocan el backstop (parser-based sobre el source de prod, con tooltip-anchor).
  3. El router enriquece allergies/diet SERVER-SIDE desde health_profile antes del swap.
  4. El knob de rollback existe.
"""
import ast
import os
import re

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


def _func_src(source, name):
    """Devuelve el source del primer def/async def `name` (cuerpo incluido)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"función {name!r} no encontrada")


# --------------------------------------------------------------------------------------
# 1. Helper determinista
# --------------------------------------------------------------------------------------

def test_helper_detects_declared_allergen():
    from graph_orchestrator import clinical_backstop_for_meal
    meal = {"name": "Arroz con camarones", "ingredients": ["Camarones", "Arroz blanco", "Aceite"]}
    viol = clinical_backstop_for_meal(meal, allergies=["camarones"], diet_type="balanced")
    assert viol, "un meal con el alérgeno declarado debe producir violación"
    assert any("camaron" in v.lower() for v in viol)


def test_helper_detects_diet_violation():
    import graph_orchestrator as go
    go.DIET_HARD_GUARD = True  # determinismo (default ya es True)
    meal = {"name": "Pollo guisado", "ingredients": ["Pechuga de pollo", "Cebolla", "Arroz"]}
    viol = go.clinical_backstop_for_meal(meal, allergies=[], diet_type="vegano")
    assert viol, "carne en un plan vegano debe producir violación"
    assert any("pollo" in v.lower() for v in viol)


def test_helper_clean_meal_no_violation():
    from graph_orchestrator import clinical_backstop_for_meal
    meal = {"name": "Ensalada", "ingredients": ["Lechuga", "Tomate", "Aguacate"]}
    assert clinical_backstop_for_meal(meal, allergies=["camarones"], diet_type="vegano") == []


def test_helper_noop_without_allergies_and_balanced():
    """No-op con allergies=[] y dieta balanced AUNQUE el meal tenga carne — esto garantiza que
    el guard default-ON no rompe los swaps normales (la mayoría de usuarios)."""
    from graph_orchestrator import clinical_backstop_for_meal
    meal = {"name": "Pollo a la plancha", "ingredients": ["Pechuga de pollo", "Arroz", "Ensalada"]}
    assert clinical_backstop_for_meal(meal, allergies=[], diet_type="balanced") == []


def test_helper_failsecure_on_scanner_error(monkeypatch):
    """Si el escáner revienta, el helper devuelve una violación (conservador → no persistir)."""
    import graph_orchestrator as go

    def _boom(*a, **k):
        raise RuntimeError("scanner roto")

    monkeypatch.setattr(go, "_scan_allergen_violations", _boom)
    viol = go.clinical_backstop_for_meal(
        {"ingredients": ["x"]}, allergies=["maní"], diet_type="balanced"
    )
    assert viol and "conservador" in viol[0].lower()


def test_helper_handles_non_dict():
    from graph_orchestrator import clinical_backstop_for_meal
    assert clinical_backstop_for_meal(None, allergies=["maní"]) == []


# --------------------------------------------------------------------------------------
# 2. Las 3 superficies invocan el backstop
# --------------------------------------------------------------------------------------

def test_swap_meal_invokes_backstop_in_retry_loop():
    src = _func_src(_read("agent.py"), "swap_meal")
    assert "clinical_backstop_for_meal" in src, "swap_meal debe correr el backstop clínico"
    assert "CLINICAL_VIOLATION" in src, "swap_meal debe levantar CLINICAL_VIOLATION en violación"
    # corre dentro del invoke_with_retry (antes del return res del retry)
    assert "UPDATE_CLINICAL_GUARD" in src


def test_modify_single_meal_invokes_backstop_before_persist():
    src = _func_src(_read("tools.py"), "execute_modify_single_meal")
    assert "clinical_backstop_for_meal" in src
    # el backstop corre ANTES de la asignación que persiste el meal
    i_guard = src.index("clinical_backstop_for_meal")
    i_persist = src.index('target_day["meals"][target_meal_index] = new_meal_data')
    assert i_guard < i_persist, "el backstop debe correr ANTES de persistir el meal"
    assert "FALLO POR SEGURIDAD CLÍNICA" in src


def test_modify_single_meal_loads_allergies_server_side():
    """El modify debe leer alergias del health_profile (server-side), NO del prompt del cliente."""
    src = _func_src(_read("tools.py"), "execute_modify_single_meal")
    assert "get_user_profile" in src and "health_profile" in src


# --------------------------------------------------------------------------------------
# 3. Enriquecimiento server-side en el router
# --------------------------------------------------------------------------------------

def test_router_enriches_clinical_in_swap_and_regenerate():
    plans_src = _read("routers/plans.py")
    assert "def _enrich_clinical_from_profile" in plans_src
    swap_src = _func_src(plans_src, "api_swap_meal")
    regen_src = _func_src(plans_src, "api_regenerate_day")
    assert "_enrich_clinical_from_profile(data, user_id)" in swap_src
    assert "_enrich_clinical_from_profile(data, user_id)" in regen_src


def test_router_soft_fails_on_clinical_violation():
    swap_src = _func_src(_read("routers/plans.py"), "api_swap_meal")
    assert "CLINICAL_VIOLATION" in swap_src
    assert "swap_clinical_violation" in swap_src


# `routers.plans` arrastra el grafo + clientes LLM; en entornos sin `langchain_openai` real
# (algunos locales) su import revienta al construir un ChatDeepSeek con el stub. En CI la dep
# está instalada y estos tests corren. Skip elegante en vez de falso-rojo.
try:
    from routers.plans import _enrich_clinical_from_profile as _ENRICH
    _ENRICH_IMPORT_ERR = None
except Exception as _e:  # pragma: no cover - depende del entorno
    _ENRICH = None
    _ENRICH_IMPORT_ERR = _e

requires_router = pytest.mark.skipif(
    _ENRICH is None,
    reason=f"routers.plans no importable en este entorno (¿falta langchain_openai?): {_ENRICH_IMPORT_ERR}",
)


@requires_router
def test_enrich_unions_allergies_from_profile(monkeypatch):
    import db

    monkeypatch.setattr(
        db, "get_user_profile",
        lambda uid: {"health_profile": {"allergies": ["maní", "camarones"], "dietType": "vegano"}},
    )
    data = {"allergies": ["lactosa"], "user_id": "u1"}
    _ENRICH(data, "u1")
    assert set(data["allergies"]) == {"lactosa", "maní", "camarones"}, "debe UNIR body+perfil"
    assert data["diet_type"] == "vegano"


@requires_router
def test_enrich_is_noop_for_guest(monkeypatch):
    import db

    def _should_not_be_called(uid):
        raise AssertionError("no debe consultar perfil para guests")

    monkeypatch.setattr(db, "get_user_profile", _should_not_be_called)
    data = {"allergies": ["x"], "user_id": "guest"}
    _ENRICH(data, "guest")
    assert data["allergies"] == ["x"]  # intacto


# --------------------------------------------------------------------------------------
# 4. Knob de rollback
# --------------------------------------------------------------------------------------

def test_rollback_knob_exists():
    src = _read("graph_orchestrator.py")
    assert 'MEALFIT_UPDATE_CLINICAL_GUARD' in src
    assert re.search(r"UPDATE_CLINICAL_GUARD\s*=\s*_env_bool", src)
