"""[P1-NEW-11 · 2026-05-11] Pre-LLM dedup check en `/api/plans/recipe/expand`.

Bug original (re-audit 2026-05-11 post primer audit cerrado):
    Usuario click duplicado en "Expandir receta" (100-300ms entre clicks)
    causaba 2× quema de cuota Gemini:
      Request #1: verify_api_quota → log_api_usage → expand_recipe_agent (LLM)
      Request #2: verify_api_quota → log_api_usage → expand_recipe_agent (LLM)
    Ambos eventualmente sobreescribían `plan_data.days[i].meals[j].recipe`
    con sus respectivas expansiones (válidas pero gastando 2× tokens).

Fix:
    Pre-check del meal target ANTES de `log_api_usage` y `expand_recipe_agent`:
      - SELECT del meal en `plan_data->days->day_index->meals->meal_index`.
      - Si tiene `isExpanded=True` AND `name == req_name` AND
        `recipe != req_recipe_original` (la recipe ya fue mutada por
        expansión previa) → return early con cached recipe.
    Solo activo si cliente envía plan_id+day_index+meal_index (clientes
    modernos post-P1-HIST-RECIPE-1). Best-effort: si SELECT falla, cae
    al path normal (mejor 1 quema extra que abortar endpoint).

NO cierra el 5% residual de requests CONCURRENTES exactos (ambos leen
pre-fix antes del primer commit). Ese caso requeriría advisory lock
(out-of-scope; YAGNI hasta que telemetría lo justifique).

Estrategia del test (parser-based):
    1. El pre-check existe antes de `log_api_usage` + `expand_recipe_agent`.
    2. Los 3 campos req_plan_id/req_day_index/req_meal_index gateaan
       la activación (cliente sin ellos cae al path legacy).
    3. La condición de match incluye `isExpanded=True` AND name match
       AND recipe ≠ original.
    4. Early-return shape incluye `skipped_llm=True` + `skip_reason=
       "already_expanded"` para telemetría.
    5. Fallback best-effort: try/except wrap del pre-check.
    6. Marker `P1-NEW-11` y log identificable.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLANS_FP = _REPO_ROOT / "backend" / "routers" / "plans.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _PLANS_FP.read_text(encoding="utf-8")


def _extract_endpoint_body(src: str) -> str:
    """Devuelve el body de `api_expand_recipe` (hasta el siguiente @router)."""
    start = src.find("def api_expand_recipe(")
    assert start > 0
    # Boundary: siguiente @router.post o def @ col 0.
    after = src[start + 30:]
    next_match = re.search(r"\n@router\.", after)
    end = (start + 30 + next_match.start()) if next_match else (start + 30 + len(after))
    return src[start:end]


def test_dedup_precheck_before_llm_call(src: str):
    """El pre-check debe estar ANTES de `log_api_usage(... "gemini_recipe_expand")`
    Y de `expand_recipe_agent(data)`."""
    body = _extract_endpoint_body(src)

    dedup_idx = body.find("[P1-NEW-11]")
    log_idx = body.find('log_api_usage(user_id, "gemini_recipe_expand")')
    llm_idx = body.find("expand_recipe_agent(data)")

    assert dedup_idx > 0, "P1-NEW-11 marker no encontrado en api_expand_recipe"
    assert log_idx > 0
    assert llm_idx > 0
    # El log_api_usage que cuenta es el del path LLM (último), no menciones en
    # comentarios. Buscamos el último.
    last_log_idx = body.rfind('log_api_usage(user_id, "gemini_recipe_expand")')
    last_llm_idx = body.rfind("expand_recipe_agent(data)")
    assert dedup_idx < last_log_idx, (
        "P1-NEW-11 regresión: el pre-check dedup ya no está ANTES de "
        "`log_api_usage`. Si se mueve después, la cuota se quema antes "
        "de verificar si ya estaba expandida."
    )
    assert dedup_idx < last_llm_idx, (
        "P1-NEW-11 regresión: el pre-check ya no está ANTES de "
        "`expand_recipe_agent`. Sin precedencia, el LLM se llama "
        "siempre — el dedup no tiene efecto."
    )


def test_dedup_gates_on_three_request_fields(src: str):
    """El pre-check solo se activa con plan_id+day_index+meal_index."""
    body = _extract_endpoint_body(src)
    dedup_start = body.find("[P1-NEW-11 · 2026-05-11]")
    dedup_block = body[dedup_start:dedup_start + 5500]

    required_gates = [
        "req_plan_id",
        "req_day_index is not None",
        "req_meal_index is not None",
        'user_id and user_id != "guest"',
    ]
    for gate in required_gates:
        assert gate in dedup_block, (
            f"P1-NEW-11 regresión: el pre-check ya no gate-a en `{gate}`. "
            f"Sin este gate, clientes legacy o requests guest pueden "
            f"disparar el dedup sin contexto válido."
        )


def test_match_condition_isExpanded_and_name_and_recipe(src: str):
    """La condición de match para early-return debe verificar 3 cosas."""
    body = _extract_endpoint_body(src)
    dedup_start = body.find("[P1-NEW-11 · 2026-05-11]")
    block = body[dedup_start:dedup_start + 5500]

    required_conditions = [
        'existing_meal.get("isExpanded") is True',
        'existing_meal.get("name") == req_name',
        'existing_meal["recipe"] != req_recipe_original',
    ]
    for cond in required_conditions:
        assert cond in block, (
            f"P1-NEW-11 regresión: la condición de match `{cond}` ya no "
            f"está. Sin las 3 condiciones combinadas el dedup podría "
            f"retornar cached recipe cuando el meal cambió (nombre o "
            f"receta base distintos)."
        )


def test_early_return_shape_includes_skipped_llm_telemetry(src: str):
    """El return early debe incluir `skipped_llm=True` y `skip_reason` para
    telemetría/observabilidad."""
    body = _extract_endpoint_body(src)
    dedup_start = body.find("[P1-NEW-11 · 2026-05-11]")
    block = body[dedup_start:dedup_start + 5500]

    for token in (
        '"skipped_llm": True',
        '"skip_reason": "already_expanded"',
        '"expanded_recipe": existing_meal["recipe"]',
    ):
        assert token in block, (
            f"P1-NEW-11 regresión: el early-return ya no incluye `{token}`. "
            f"Sin esto, frontend/telemetría no pueden distinguir "
            f"`expand cached` de `expand fresh`."
        )


def test_select_meal_via_jsonb_path(src: str):
    """El SELECT debe usar jsonb path para evitar cargar el plan_data entero."""
    body = _extract_endpoint_body(src)
    dedup_start = body.find("[P1-NEW-11 · 2026-05-11]")
    block = body[dedup_start:dedup_start + 5500]

    # Patrón canónico: SELECT plan_data->'days'->%s->'meals'->%s
    pattern = re.compile(
        r"SELECT\s+plan_data->'days'->%s->'meals'->%s\s+AS\s+meal\s+"
        r"FROM\s+meal_plans\s+WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s",
    )
    assert pattern.search(block), (
        "P1-NEW-11 regresión: el SELECT del meal target ya no usa el path "
        "jsonb estrechamente (`plan_data->'days'->%s->'meals'->%s`). Si "
        "se cambia a SELECT plan_data completo, el endpoint paga el costo "
        "de transferir un JSON grande para cada request."
    )


def test_fallback_best_effort_on_exception(src: str):
    """Si el pre-check explota, DEBE caer al path normal — NO abortar
    el endpoint completo."""
    body = _extract_endpoint_body(src)
    dedup_start = body.find("[P1-NEW-11 · 2026-05-11]")
    block = body[dedup_start:dedup_start + 5500]

    assert "try:" in block, (
        "P1-NEW-11 regresión: el pre-check ya no está envuelto en try/. "
        "Una excepción del SELECT abortaría el endpoint entero."
    )
    assert re.search(r"except\s+Exception\s+as\s+_dedup_err", block), (
        "P1-NEW-11 regresión: el except del pre-check ya no captura "
        "`Exception as _dedup_err`. Sin captura amplia, errores no "
        "anticipados crashearían el endpoint."
    )
    # NO debe haber `raise` dentro del except (ese sería el bug).
    except_block = block[block.find("except Exception"):block.find("except Exception") + 500]
    assert "raise" not in except_block, (
        "P1-NEW-11 regresión: el except del pre-check ahora hace `raise`. "
        "Eso aborta el endpoint en lugar de caer al path normal — "
        "rompiendo el fallback best-effort."
    )


def test_p1_new_11_marker_in_block(src: str):
    body = _extract_endpoint_body(src)
    assert "[P1-NEW-11]" in body, (
        "P1-NEW-11 regresión: el marker desapareció del endpoint. Sin él, "
        "un grep desde otro test/cron no encuentra esta defensa."
    )
