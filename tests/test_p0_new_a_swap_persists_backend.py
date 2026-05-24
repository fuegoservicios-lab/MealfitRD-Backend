"""[P0-NEW-A · 2026-05-11] Test parser-based: la persistencia post-swap
DEBE pasar por el endpoint backend `/api/plans/{plan_id}/swap-meal/persist`,
NO por un direct write desde el frontend a `meal_plans`.

Bug original (gap P0 detectado audit 2026-05-11):
    `frontend/src/context/AssessmentContext.jsx` (función swap, ~línea 1240)
    hacía ``supabase.from('meal_plans').update({plan_data: updatedPlan}).eq('id', planId)``
    desde el cliente. Ese patrón producía LOST-UPDATE:

      - El frontend leía `planData` del state local (potencialmente stale
        respecto al snapshot persistido en DB).
      - `_chunk_worker` podía finalizar un chunk entre el read y el write
        del cliente, persistiendo `days[7-14]`, `_chunk_lessons`,
        `aggregated_shopping_list*`, etc. vía `update_meal_plan_data`.
      - El frontend hacía UPDATE plan_data = <snapshot stale> pisando TODO
        lo recién persistido por el worker.

    RLS protegía contra IDOR (el filter `.eq('id', planId)` y la policy
    forced de meal_plans → solo el dueño escribe), pero RLS no previene
    el lost-update: ambos escritores son legítimos del mismo `user_id`.

Cierre P0-NEW-A:
    1. Backend gana endpoint `/{plan_id}/swap-meal/persist` que hace
       `jsonb_set(plan_data, '{days,<i>,meals,<j>}', new_meal, true)` —
       jsonb_set quirúrgico, NO full overwrite. Solo el meal cambiado
       se reescribe; resto del JSONB queda intacto, conservando lo que
       el worker haya persistido en paralelo.
    2. El UPDATE incluye `AND user_id = %s` defense-in-depth (espejo de
       `/retry-chunk` P0-HIST-IDOR-1 y `/recipe/expand` P1-HIST-RECIPE-1).
    3. El UPDATE strippea las 4 listas pre-calculadas
       (`aggregated_shopping_list*`) y bumpea `_plan_modified_at` en la
       misma transacción.
    4. Frontend reemplaza el `supabase.from('meal_plans').update(...)`
       por `fetchWithAuth(POST /swap-meal/persist)`.

Tests (parser-based, sin DB ni red):
    1. `test_handler_exists_with_jsonb_set_and_user_id_filter` —
       `api_swap_meal_persist` existe en plans.py, su SQL usa `jsonb_set`
       sobre el path `'{days,...,meals,...}'` y filtra `AND user_id = %s`.
    2. `test_frontend_swap_uses_backend_endpoint` — AssessmentContext.jsx
       referencia `/api/plans/${planId}/swap-meal/persist` dentro del
       flow de swap.
    3. `test_frontend_swap_no_longer_direct_writes_meal_plans` — el
       patrón legacy `supabase.from('meal_plans').update({plan_data` ya
       NO aparece dentro de la función de swap (escaneamos la función
       completa, no todo el archivo, porque el INSERT inicial en Plan.jsx
       y el grocery-date sync legacy viven fuera de este scope).
    4. `test_marker_anchor_present` — slug `p0_new_a` aparece en el
       filename (cross-link con `test_p2_hist_audit_14_marker_test_link`).

Tooltip-anchor: P0-NEW-A-START | gap P0 audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"
_ASSESSMENT_JSX = _REPO_ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx"


def _extract_function_body(src: str, fn_name: str) -> str:
    """Extrae el cuerpo de la función `fn_name` (Python) desde
    `def fn_name(...)` hasta justo antes del siguiente `@router.`/
    `@app.`/`def `. Mismo helper que `test_p1_new_7_recipe_expand_persists`
    para evitar drift cross-test.
    """
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    if not m:
        raise AssertionError(
            f"No se encontró `def {fn_name}(` en plans.py — "
            f"endpoint renombrado/eliminado."
        )
    start = m.start()
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


def _extract_js_function_block(src: str, marker: str, *, max_chars: int = 8000) -> str:
    """Extrae un bloque de hasta `max_chars` caracteres alrededor de un
    `marker` literal (e.g., comentario inline). Suficiente para escanear
    el cuerpo del handler de swap en AssessmentContext.jsx sin depender
    del balanceo exacto de llaves (la función `regenerateSingleMeal` no
    tiene una marca canónica `// END` y braces nesteadas dificultan el
    matching).
    """
    idx = src.find(marker)
    if idx < 0:
        raise AssertionError(
            f"No se encontró el marcador `{marker}` en AssessmentContext.jsx — "
            f"refactor del flow de swap. Si el bloque se movió, actualizar "
            f"el marker en este test."
        )
    start = max(0, idx - 500)
    end = min(len(src), idx + max_chars)
    return src[start:end]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def assessment_src() -> str:
    return _ASSESSMENT_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def swap_persist_body(plans_src: str) -> str:
    return _extract_function_body(plans_src, "api_swap_meal_persist")


# ---------------------------------------------------------------------------
# 1. Backend handler existe y aplica jsonb_set quirúrgico + filtro user_id
# ---------------------------------------------------------------------------
def test_handler_exists_with_jsonb_set_and_user_id_filter(swap_persist_body: str):
    """El handler `api_swap_meal_persist` debe escribir el meal SIN
    full-overwrite del plan_data. Dos patrones aceptados:

      1. **Legacy (P0-NEW-A · 2026-05-11):** `jsonb_set(...)` sobre el
         path `{days,...,meals,...}` directo en SQL — quirúrgico, NO
         requiere row lock porque jsonb_set serializa server-side a
         nivel Postgres.
      2. **Atómico (P1-SWAP-PERSIST-ATOMIC · 2026-05-22):**
         `update_plan_data_atomic(plan_id, _mutator, user_id=...)` —
         `SELECT … FOR UPDATE` + mutator callback que muta solo
         `plan_data['days'][i]['meals'][j]`. Cierra estructuralmente
         la ventana lost-update incluso si un futuro refactor del
         mutator degrada a operaciones overlap con el worker.

    Ambos preservan la garantía contra lost-update con `_chunk_worker`
    en paralelo. Sin alguno de los dos, el handler reabriría el bug
    original P0-NEW-A (UPDATE plan_data = <snapshot stale>).

    Adicional: SELECT inicial de ownership DEBE incluir `AND user_id = %s`
    para devolver 404 al user ajeno (defensa-en-depth).
    """
    # Distinguir uso ACTIVO (jsonb_set como call) vs menciones en docstring/
    # comentarios que cuentan la historia del refactor P1-SWAP-PERSIST-ATOMIC.
    # Solo `jsonb_set(` (open paren) en código ejecutable cuenta como "use".
    # Strip de comentarios `# ...` y triple-quoted strings antes del match.
    _body_no_comments = re.sub(r"#[^\n]*", "", swap_persist_body)
    _body_no_docstrings = re.sub(r'"""[\s\S]*?"""', "", _body_no_comments)
    has_jsonb_set = bool(re.search(r"jsonb_set\s*\(", _body_no_docstrings))
    has_atomic_helper = "update_plan_data_atomic" in swap_persist_body

    assert has_jsonb_set or has_atomic_helper, (
        "P0-NEW-A / P1-SWAP-PERSIST-ATOMIC regresión: "
        "`api_swap_meal_persist` no usa `jsonb_set` ni "
        "`update_plan_data_atomic`. Si degradó a full overwrite "
        "(`UPDATE meal_plans SET plan_data = %s::jsonb` directo sin "
        "FOR UPDATE), reabriría el lost-update que P0-NEW-A cerró. "
        "Restaurar alguno de los dos patrones."
    )

    if has_atomic_helper:
        # P1-SWAP-PERSIST-ATOMIC · 2026-05-22: helper recibe el mutator
        # como callback. El mutator debe operar sobre `days[day_index]`
        # y `meals[meal_index]` directamente (NO sobre keys arbitrarios).
        atomic_pattern = re.compile(
            r"update_plan_data_atomic\s*\(\s*plan_id\s*,",
            re.DOTALL,
        )
        assert atomic_pattern.search(swap_persist_body), (
            "P1-SWAP-PERSIST-ATOMIC regresión: el handler invoca "
            "`update_plan_data_atomic` pero NO pasa `plan_id` como "
            "primer arg posicional. Verificar signature "
            "(db_plans.py:381)."
        )
        # Defense-in-depth: `user_id=verified_user_id` debe propagarse
        # al helper para que el SELECT FOR UPDATE incluya `AND user_id`
        # (cierra I2 + previene logger.warning("[I2-MISS] ...")).
        user_id_kwarg_pattern = re.compile(
            r"update_plan_data_atomic[^)]*user_id\s*=\s*verified_user_id",
            re.DOTALL,
        )
        assert user_id_kwarg_pattern.search(swap_persist_body), (
            "P1-SWAP-PERSIST-ATOMIC regresión: el handler invoca "
            "`update_plan_data_atomic` SIN `user_id=verified_user_id`. "
            "Esto degrada a `[I2-MISS]` warning en logs y rompe la "
            "defensa-en-depth I2 (el SELECT FOR UPDATE no filtra owner). "
            "Restaurar el kwarg."
        )
        # El mutator debe tocar days[day_index] y meals[meal_index].
        assert "days[day_index]" in swap_persist_body or \
               "days[" in swap_persist_body, (
            "P1-SWAP-PERSIST-ATOMIC regresión: el mutator no muta "
            "`plan_data['days']` — probablemente cambió a una key "
            "incorrecta. Verificar shape del plan."
        )

    if has_jsonb_set:
        # Legacy: validamos el shape `{days,<int>,meals,<int>}`
        meal_path_pattern = re.compile(r"\{days,\s*\"?\s*\+\s*str\(day_index\)", re.DOTALL)
        has_dynamic_path = bool(meal_path_pattern.search(swap_persist_body))
        has_meal_path_literal = (
            '"{days," + str(day_index)' in swap_persist_body
            or "'{days,' + str(day_index)" in swap_persist_body
        )
        assert has_dynamic_path or has_meal_path_literal, (
            "P0-NEW-A regresión: el handler usa `jsonb_set` pero no "
            "construye el path `{days,<day_index>,meals,<meal_index>}`. "
            "Sin path dinámico, el handler escribiría sobre un meal "
            "hardcoded o sobre la raíz del plan_data (no-op write)."
        )

    # SELECT inicial de ownership: `AND user_id = %s` siempre presente
    # (independiente del patrón de write — es la primera capa que cierra
    # IDOR antes de invocar al write helper).
    where_pattern = re.compile(
        r"WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s",
        re.IGNORECASE | re.DOTALL,
    )
    assert where_pattern.search(swap_persist_body), (
        "P0-NEW-A regresión: el SELECT inicial de ownership no filtra "
        "`AND user_id = %s`. Sin esto, un user ajeno recibiría 200 al "
        "intentar swap-ear un plan que no le pertenece (el FOR UPDATE "
        "del helper atómico devolvería `{}` y mapearíamos a 404 igual, "
        "pero la primera capa debe rechazar antes del network round-trip "
        "extra). Restaurar el doble candado (espejo de retry-chunk "
        "P0-HIST-IDOR-1)."
    )


# ---------------------------------------------------------------------------
# 2. Frontend invoca el nuevo endpoint dentro del flow de swap
# ---------------------------------------------------------------------------
def test_frontend_swap_uses_backend_endpoint(assessment_src: str):
    """AssessmentContext.jsx debe invocar `/api/plans/${planId}/swap-meal/persist`
    como parte del flow de swap. Sin esta llamada, las mutaciones quedan
    en state local y se pierden al reload o se pisan con el snapshot
    de la nube en el próximo `fetchProfile`.
    """
    persist_call_pattern = re.compile(
        r"/api/plans/\$\{[^}]*\}/swap-meal/persist",
    )
    assert persist_call_pattern.search(assessment_src), (
        "P0-NEW-A regresión: AssessmentContext.jsx NO referencia "
        "`/api/plans/${planId}/swap-meal/persist`. La función de swap "
        "debe POST-ear al endpoint backend en lugar de hacer "
        "`supabase.from('meal_plans').update({plan_data})` directo "
        "(que produce lost-update con _chunk_worker). Restaurar el "
        "fetchWithAuth al endpoint nuevo."
    )


# ---------------------------------------------------------------------------
# 3. Frontend ya no hace direct write a meal_plans dentro del flow de swap
# ---------------------------------------------------------------------------
def test_frontend_swap_no_longer_direct_writes_meal_plans(assessment_src: str):
    """El bloque del flow de swap (delimitado por la llamada al endpoint
    `/api/plans/swap-meal` para la generación + el invoke del nuevo
    `/swap-meal/persist`) NO debe contener el patrón legacy
    `supabase.from('meal_plans').update({plan_data` que producía
    lost-update.

    Escaneamos un bloque acotado (no todo el archivo) porque:
      - `Plan.jsx:398` hace INSERT inicial — legítimo, fuera de scope.
      - `AssessmentContext.jsx:560` (grocery_start_date sync) cae bajo
        P0-NEW-B (siguiente fix), no P0-NEW-A.
    """
    # El marker es la llamada al endpoint legacy de generación
    # `/api/plans/swap-meal` (NO confundir con `/swap-meal/persist`).
    swap_block = _extract_js_function_block(
        assessment_src, "/api/plans/swap-meal'",
    )
    # Strip líneas que son puro comentario JS (`//` al inicio, ignorando
    # whitespace). Esto evita falsos positivos en la documentación inline
    # del propio fix P0-NEW-A, que cita el patrón legacy literal en un
    # comentario explicativo. Solo nos interesa el código ejecutable.
    no_comments = re.sub(
        r"^\s*//.*$",
        "",
        swap_block,
        flags=re.MULTILINE,
    )
    legacy_pattern = re.compile(
        r"supabase\s*\.\s*from\s*\(\s*['\"]meal_plans['\"]\s*\)"
        r"\s*\.\s*update\s*\(\s*\{\s*plan_data",
        re.DOTALL,
    )
    matches = legacy_pattern.findall(no_comments)
    assert not matches, (
        "P0-NEW-A regresión: el bloque del swap en AssessmentContext.jsx "
        "todavía contiene `supabase.from('meal_plans').update({plan_data:...})` "
        "directo (línea ejecutable, no comentario). Este patrón produce "
        "lost-update con `_chunk_worker` y fue reemplazado por POST "
        "`/api/plans/${planId}/swap-meal/persist`. Restaurar el reemplazo "
        "del flow."
    )


# ---------------------------------------------------------------------------
# 4. Cross-link slug del marker (test_p2_hist_audit_14_marker_test_link)
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """Slug del filename matchea el marker `P0-NEW-A` para que el
    enforcer `test_p2_hist_audit_14_marker_test_link` lo detecte.
    """
    expected_slug = "p0_new_a"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p0_new_a`) para que el cross-link `test_p2_hist_audit_14_"
        "marker_test_link` lo matchee con `_LAST_KNOWN_PFIX` cuando "
        "se bumpee a `P0-NEW-A · 2026-05-11`."
    )
