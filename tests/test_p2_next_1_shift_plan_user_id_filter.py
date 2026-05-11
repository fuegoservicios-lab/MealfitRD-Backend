"""[P2-NEXT-1 · 2026-05-11] `/shift-plan` (HTTP) y `_background_shift_plan_for_user`
(cron) DEBEN filtrar `AND user_id = %s` en TODA mutación `UPDATE meal_plans`.

Bug original (audit conversacional 2026-05-11):
    Tres UPDATE de `meal_plans` ejecutaban `WHERE id = %s` sin el filtro de
    user_id, violando la invariante I2 documentada en CLAUDE.md
    (`Toda mutación de meal_plans filtra AND user_id = %s`):
      1. `routers/plans.py:api_shift_plan` — UPDATE final tras shift+enqueue.
      2. `cron_tasks.py:_background_shift_plan_for_user` — path pantry-pause.
      3. Mismo helper — path success (renovación encolada).

    Funcionalmente seguros HOY porque `plan_id` se resuelve previamente con
    `SELECT id FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC`,
    pero un refactor que reordene/elimine ese SELECT abriría IDOR silente:
    un atacante con `verified_user_id == self` que envíe un `plan_id` ajeno
    por otro canal (e.g. body futuro de /shift-plan) podría pisar plan_data
    de la víctima.

Estrategia (parser estático, mismo patrón que `test_p0_new_1_restock_*`,
`test_p1_new_4_regenerate_*` y `test_p3_audit_7_plan_result_keys_contract`):
    1. Extraer body de `api_shift_plan` y `_background_shift_plan_for_user`.
    2. Para cada función, escanear todas las apariciones de
       `UPDATE meal_plans` (en raw SQL strings).
    3. Para cada UPDATE, verificar que dentro del mismo statement aparece
       `user_id` en la cláusula WHERE (literal `user_id = %s` o
       `AND user_id = %s`).
    4. Fallar con mensaje accionable indicando línea y snippet.

Drift detection:
    - Si alguien revierte cualquiera de los 3 sitios → falla el test que
      enumera ese statement con copy explicando la familia P2-NEXT-1.
    - Si alguien añade un 4º UPDATE `meal_plans` sin user_id en estas
      funciones → el test lo captura automáticamente.
    - Si las funciones se renombran → AssertionError explícita.

Whitelist intencional:
    - El SELECT `... FROM meal_plans WHERE id = %s FOR UPDATE` NO se enforza
      (row lock post-resolución upstream user-scoped + advisory lock).
    - UPDATEs de OTRAS tablas (e.g. `plan_chunk_queue`) NO se enforzan acá;
      su ownership se valida vía join al meal_plan_id resuelto.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"
_CRON_PY = _REPO_ROOT / "backend" / "cron_tasks.py"


def _extract_function_body(src: str, fn_name: str, src_label: str) -> str:
    """Devuelve `def <fn>(` hasta el siguiente top-level `def`/`@router`/
    `@app`. Mismo helper que `test_p0_new_1_restock_*`.
    """
    pattern = re.compile(rf"\ndef\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search("\n" + src)
    if not m:
        # Búsqueda fallback sin newline (function al top sin blank line).
        m = re.search(rf"def\s+{re.escape(fn_name)}\s*\(", src)
        if not m:
            raise AssertionError(
                f"No se encontró `def {fn_name}(` en {src_label}. "
                f"Si el rename es intencional, actualizar P2-NEXT-1 test "
                f"con el nuevo nombre. Si fue eliminado, verificar que la "
                f"familia de UPDATE meal_plans no quedó sin ownership filter."
            )
        start = m.start()
    else:
        start = m.start() - 1  # compensar el "\n" prepended

    rest = src[start + 1:]
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s|async\s+def\s)", rest)
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


def _find_meal_plans_updates(body: str) -> list[tuple[int, str]]:
    """Devuelve [(line_no_relativo, statement_completo)] para cada
    `UPDATE meal_plans` encontrado en el body. Reconstruye la sentencia
    SQL concatenando líneas hasta el cierre del string (heurística:
    paréntesis balanceados o cierre de execute()).

    Robusto a tres formas de pasar SQL a `cursor.execute`/`execute_sql_write`:
      a) Single-line string: `cursor.execute("UPDATE meal_plans ... WHERE id = %s AND user_id = %s", ...)`
      b) Triple-quoted:  `cursor.execute(\"\"\"UPDATE meal_plans\\n SET ... WHERE id = %s AND user_id = %s\"\"\", ...)`
      c) Concatenated:   `cursor.execute("UPDATE meal_plans " "SET ... " "WHERE id = %s AND user_id = %s", ...)`
    """
    results: list[tuple[int, str]] = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        if "UPDATE meal_plans" in lines[i]:
            # Reconstruir el statement: capturar hasta encontrar el cierre del
            # execute() o un statement Python que claramente NO sea continuación.
            # Heurística pragmática: tomar bloque desde esta línea hasta encontrar
            # ");" o `)` solo, o el cierre con args.
            chunk_lines = [lines[i]]
            j = i + 1
            depth = lines[i].count("(") - lines[i].count(")")
            while j < len(lines) and (depth > 0 or "WHERE" not in " ".join(chunk_lines).upper()):
                chunk_lines.append(lines[j])
                depth += lines[j].count("(") - lines[j].count(")")
                j += 1
                # Safety: no más de 40 líneas (todos los UPDATE meal_plans en
                # las dos funciones bajo test caben en <20).
                if j - i > 40:
                    break
            statement = " ".join(chunk_lines)
            results.append((i, statement))
            i = j
        else:
            i += 1
    return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shift_plan_body() -> str:
    src = _PLANS_PY.read_text(encoding="utf-8")
    return _extract_function_body(src, "api_shift_plan", "routers/plans.py")


@pytest.fixture(scope="module")
def bg_refill_body() -> str:
    src = _CRON_PY.read_text(encoding="utf-8")
    return _extract_function_body(
        src, "_background_shift_plan_for_user", "cron_tasks.py"
    )


# ---------------------------------------------------------------------------
# Core invariant tests
# ---------------------------------------------------------------------------


def test_api_shift_plan_all_meal_plans_updates_filter_user_id(shift_plan_body: str):
    """Todo `UPDATE meal_plans` dentro de `api_shift_plan` DEBE incluir
    `user_id` en su cláusula WHERE."""
    updates = _find_meal_plans_updates(shift_plan_body)
    assert updates, (
        "P2-NEXT-1 anchor: no se encontró ningún `UPDATE meal_plans` en "
        "`api_shift_plan`. Si la función ya no muta meal_plans, simplificar "
        "este test. Si se movió a un helper, el helper también necesita "
        "anchor de I2."
    )
    offenders = []
    for rel_line, stmt in updates:
        # Buscar `user_id` en la cláusula WHERE de ese statement.
        # Patrones aceptados: `user_id = %s`, `AND user_id = %s`,
        # `WHERE user_id = %s` (case-insensitive en la palabra WHERE/AND).
        if not re.search(r"\buser_id\s*=\s*%s\b", stmt, flags=re.IGNORECASE):
            offenders.append((rel_line, stmt[:200]))
    assert not offenders, (
        "P2-NEXT-1 violation: `api_shift_plan` tiene UPDATE meal_plans "
        "sin filtro `AND user_id = %s`. Esto rompe la invariante I2 de "
        "CLAUDE.md y reabre el riesgo IDOR si un refactor cambia la "
        "resolución upstream de plan_id. Offenders:\n"
        + "\n".join(f"  line +{ln}: {snip}" for ln, snip in offenders)
    )


def test_bg_refill_all_meal_plans_updates_filter_user_id(bg_refill_body: str):
    """Todo `UPDATE meal_plans` dentro de `_background_shift_plan_for_user`
    DEBE incluir `user_id` en su cláusula WHERE. Mirror background de
    `api_shift_plan` — mismo invariante I2."""
    updates = _find_meal_plans_updates(bg_refill_body)
    assert updates, (
        "P2-NEXT-1 anchor: no se encontró ningún `UPDATE meal_plans` en "
        "`_background_shift_plan_for_user`. Si la función ya no muta "
        "meal_plans, simplificar este test."
    )
    offenders = []
    for rel_line, stmt in updates:
        if not re.search(r"\buser_id\s*=\s*%s\b", stmt, flags=re.IGNORECASE):
            offenders.append((rel_line, stmt[:200]))
    assert not offenders, (
        "P2-NEXT-1 violation: `_background_shift_plan_for_user` tiene "
        "UPDATE meal_plans sin filtro `AND user_id = %s`. Mirror del path "
        "HTTP /shift-plan; mismo invariante I2 (CLAUDE.md). Offenders:\n"
        + "\n".join(f"  line +{ln}: {snip}" for ln, snip in offenders)
    )


# ---------------------------------------------------------------------------
# Sanity / floor anchors
# ---------------------------------------------------------------------------


def _normalize_python_string_concat(src: str) -> str:
    """Colapsa Python adjacent-string concatenation (`"abc" "def"`) en
    `"abcdef"` para que regex SQL no se rompa con splits multilinea.
    Pragmático — no es un parser, solo borra `"\\s*"` entre literales."""
    return re.sub(r'"\s*"', "", src)


def test_api_shift_plan_resolves_plan_id_user_scoped(shift_plan_body: str):
    """La resolución upstream de `plan_id` en `api_shift_plan` DEBE seguir
    filtrando por user_id. Sin esto, añadir `AND user_id = %s` en el UPDATE
    es seguridad parcial: el SELECT también necesita el filtro para que
    `plan_id` esté garantizado user-scoped."""
    normalized = _normalize_python_string_concat(shift_plan_body)
    canonical = re.search(
        r"SELECT\s+id\s+FROM\s+meal_plans\s+WHERE\s+user_id\s*=\s*%s",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert canonical, (
        "P2-NEXT-1 floor anchor violation: `api_shift_plan` ya NO resuelve "
        "`plan_id` con `SELECT id FROM meal_plans WHERE user_id = %s`. "
        "Esto debilita la defensa-en-profundidad de P2-NEXT-1 — el filtro "
        "en el UPDATE depende de que `plan_id` esté pre-validado. Restaurar "
        "el SELECT user-scoped o documentar el cambio con un test sustituto."
    )


def test_bg_refill_resolves_plan_id_user_scoped(bg_refill_body: str):
    """Mirror del anchor anterior para el path background."""
    normalized = _normalize_python_string_concat(bg_refill_body)
    canonical = re.search(
        r"SELECT\s+id\s+FROM\s+meal_plans\s+WHERE\s+user_id\s*=\s*%s",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert canonical, (
        "P2-NEXT-1 floor anchor violation: `_background_shift_plan_for_user` "
        "ya NO resuelve `plan_id` con `SELECT id FROM meal_plans WHERE "
        "user_id = %s`. Mismo argumento que el path HTTP."
    )


def test_p2_next_1_marker_present_in_at_least_one_site():
    """Sanity: el comentario `[P2-NEXT-1` debe aparecer al menos en uno de
    los dos archivos modificados. Detecta una eliminación accidental de
    todos los anchors textuales (que rompería traceability)."""
    plans_src = _PLANS_PY.read_text(encoding="utf-8")
    cron_src = _CRON_PY.read_text(encoding="utf-8")
    plans_hit = "P2-NEXT-1" in plans_src
    cron_hit = "P2-NEXT-1" in cron_src
    assert plans_hit or cron_hit, (
        "P2-NEXT-1 marker desapareció de ambos archivos. Si fue intencional "
        "(rename del fix), actualizar el slug en este test."
    )
