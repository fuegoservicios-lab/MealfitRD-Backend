"""[P1-CONSUMED-MEALS-JSONB · 2026-05-20] Test anti-regresión del wrapping
`Jsonb(...)` sobre el parámetro `ingredients` del INSERT a `consumed_meals`.

Pre-fix:
    execute_sql_write(
        "INSERT INTO consumed_meals (...) VALUES (..., %s, ...)",
        (..., ingredients if ingredients is not None else [], ...),
    )

`consumed_meals.ingredients` es jsonb. psycopg3 sin Jsonb() wrap convierte
`list[str]` a literal array Postgres `{a,b,c}` → INSERT falla con
"invalid input syntax for type json (Expected ':', but found ',')".

El bug pasó desapercibido porque:
  - El except del helper captura el error y retorna None (graceful degradation).
  - El LLM verbaliza "tuvimos un pequeño fallo técnico" al user — UX OK.
  - La tabla `consumed_meals` quedó vacía en producción (verified via Supabase MCP
    audit 2026-05-20: COUNT(*)=0 + MAX(created_at)=NULL).

El test es parser-based: extrae el cuerpo de `log_consumed_meal` y asserta:
  1. El import `from psycopg.types.json import Jsonb` está presente.
  2. El SQL `INSERT INTO consumed_meals` existe en el cuerpo.
  3. El parámetro `ingredients` que va al INSERT está envuelto en `Jsonb(...)`.

Si en el futuro `consumed_meals.ingredients` migra a `text[]`, este test
debe actualizarse junto con la migration.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_FACTS_PY = _BACKEND_ROOT / "db_facts.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_fn_body(src: str, fn_name: str) -> str:
    """Extrae el cuerpo de una función entre su `def` y el siguiente `def`
    al mismo nivel de indentación (o EOF). Funcional para módulos top-level
    como `db_facts.py`."""
    match = re.search(
        rf"def {re.escape(fn_name)}\(.*?\n(.*?)(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert match, f"función {fn_name} no encontrada"
    return match.group(1)


def test_log_consumed_meal_wraps_ingredients_in_jsonb():
    """[P1-CONSUMED-MEALS-JSONB] El parámetro `ingredients` del INSERT a
    `consumed_meals` debe ir envuelto en `Jsonb(...)` para que psycopg3
    serialize como JSON y no como Postgres array literal.

    Anchor anti-regresión: si alguien hace 'cleanup' del `Jsonb(...)` wrap
    pensando que es redundante (no lo es — el import suelto sí lo era
    pre-fix), este test bloquea el commit.
    """
    src = _read(_DB_FACTS_PY)
    body = _extract_fn_body(src, "log_consumed_meal")

    # Sanity: el SQL INSERT debe existir.
    assert "INSERT INTO consumed_meals" in body, (
        "INSERT INTO consumed_meals no encontrado en log_consumed_meal — "
        "refactor inesperado."
    )

    # Sanity: el import de Jsonb debe estar dentro o por encima del body
    # (puede estar inline `from psycopg.types.json import Jsonb` para no
    # acoplar el module-init a psycopg).
    has_import = (
        "from psycopg.types.json import Jsonb" in body
        or "from psycopg.types.json import Jsonb" in src
    )
    assert has_import, (
        "Import `from psycopg.types.json import Jsonb` ausente — sin él, "
        "no se puede envolver el parámetro."
    )

    # Core: el parámetro `ingredients` que va al execute_sql_write debe
    # ir envuelto en `Jsonb(...)`. Buscamos el patrón `Jsonb(ingredients`
    # con o sin condicional ternario después.
    assert re.search(
        r"Jsonb\s*\(\s*ingredients\b",
        body,
    ), (
        "Parámetro `ingredients` NO envuelto en `Jsonb(...)` — psycopg3 "
        "convertirá list[str] a Postgres ARRAY literal `{a,b,c}` y el "
        "INSERT fallará con 'invalid input syntax for type json'. Ver "
        "P1-CONSUMED-MEALS-JSONB · 2026-05-20."
    )


def test_log_consumed_meal_jsonb_wrap_near_insert():
    """[P1-CONSUMED-MEALS-JSONB] Defensa adicional: el `Jsonb(ingredients`
    wrap debe estar en proximidad textual al `INSERT INTO consumed_meals`
    (mismo bloque de código). NO en otro callsite no relacionado.

    Approach: regex sobre paren-balanced SQL es frágil; usamos proximidad
    de líneas. Si el `INSERT INTO consumed_meals` y el `Jsonb(ingredients`
    están separados por más de 10 líneas, alguien movió el wrap a otro
    callsite y este test debería bloquear.
    """
    src = _read(_DB_FACTS_PY)
    body = _extract_fn_body(src, "log_consumed_meal")

    lines = body.splitlines()
    insert_lineno = next(
        (i for i, ln in enumerate(lines) if "INSERT INTO consumed_meals" in ln),
        None,
    )
    jsonb_lineno = next(
        (i for i, ln in enumerate(lines) if re.search(r"Jsonb\s*\(\s*ingredients\b", ln)),
        None,
    )
    assert insert_lineno is not None, "INSERT INTO consumed_meals no encontrado"
    assert jsonb_lineno is not None, "Jsonb(ingredients no encontrado en log_consumed_meal"
    assert abs(jsonb_lineno - insert_lineno) <= 10, (
        f"Jsonb(ingredients está a {abs(jsonb_lineno - insert_lineno)} líneas del "
        f"INSERT INTO consumed_meals — refactor rompió la proximidad. Ver "
        f"P1-CONSUMED-MEALS-JSONB · 2026-05-20."
    )
