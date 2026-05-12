"""[P0-LIVE-1 · 2026-05-12] Tests parser-based para alias consistency en
`SELECT DISTINCT ON (...)` vs `ORDER BY ...`.

Contexto del bug:
    En `_proactive_refresh_pending_pantry_snapshots` (cron_tasks.py) la query
    declaraba `DISTINCT ON (user_id, meal_plan_id)` SIN prefix de alias,
    mientras el SELECT proyectaba aliases idénticos (`q.user_id::text AS
    user_id`) que SOMBREABAN las columnas originales, y el ORDER BY usaba
    `q.user_id, q.meal_plan_id` con prefix. Postgres no considera estas
    expresiones equivalentes y lanzaba en runtime:

        ERROR: SELECT DISTINCT ON expressions must match initial ORDER BY
               expressions

    Consecuencia: cada tick del cron P0-C/PROACTIVE crasheaba → snapshots
    de despensa de chunks vivos NUNCA se refrescaban proactivamente → planes
    multi-week generaban con pantry stale → divergencia receta↔lista (la
    clase exacta de bug que `run_shopping_coherence_guard` debía cerrar).

Estrategia:
    Escanear `backend/*.py` (excluido tests/) y, para cada bloque que
    contenga `SELECT DISTINCT ON (...)`, parsear:
      1) las expresiones del DISTINCT ON
      2) las expresiones iniciales del primer ORDER BY que aparece dentro
         del mismo string literal SQL
    y validar que las N expresiones del DISTINCT ON coincidan exactamente
    con los N primeros tokens del ORDER BY.

    El test es heurístico (parsea strings literales Python que contengan
    SQL inline) — falla en caso ambiguo dando mensaje accionable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Anchor de cierre (no eliminar — referenciado por marker-test link test
# P2-HIST-AUDIT-14 si el _LAST_KNOWN_PFIX bumpea a P0-LIVE-1).
P0_LIVE_1_ANCHOR = "P0-LIVE-1"

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Archivos de prod a escanear (excluye tests/, scripts/, scratch/, venv).
_PROD_GLOBS = [
    "*.py",
    "routers/*.py",
]

_DISTINCT_ON_RE = re.compile(
    r"SELECT\s+DISTINCT\s+ON\s*\(\s*(?P<keys>[^)]+?)\s*\)",
    re.IGNORECASE | re.DOTALL,
)
_ORDER_BY_RE = re.compile(
    r"\bORDER\s+BY\s+(?P<cols>.+?)(?:\s+LIMIT\b|\s*\"\"\"|\s*\)\s*,?\s*$|\s+OFFSET\b)",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_expr(expr: str) -> str:
    """Quita whitespace redundante + lowercase para comparación flexible."""
    return re.sub(r"\s+", " ", expr.strip().lower())


def _split_csv(expr: str) -> list[str]:
    """Divide por coma top-level (ignora comas dentro de paréntesis)."""
    out: list[str] = []
    depth = 0
    buf = []
    for ch in expr:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [t for t in out if t]


def _strip_asc_desc_collate(token: str) -> str:
    """Quita sufijos ASC/DESC/NULLS FIRST/LAST/COLLATE de un token ORDER BY."""
    t = token.strip()
    t = re.sub(r"\s+ASC\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+DESC\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+NULLS\s+(FIRST|LAST)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+COLLATE\s+\S+", "", t, flags=re.IGNORECASE)
    return t.strip()


_ORDER_BY_ANY_RE = re.compile(
    r"\bORDER\s+BY\s+([^;\"\n]+(?:\n\s*[^;\"\n]+)*?)"
    r"(?=\s+LIMIT\b|\s+OFFSET\b|\s+FETCH\b|\s*\"\"\"|\s*\"|\s*\)|$)",
    re.IGNORECASE,
)


def _gather_distinct_on_queries(text: str) -> list[tuple[list[str], list[list[str]], int]]:
    """Devuelve lista de (distinct_on_keys, list_of_order_by_token_lists, line_no).

    Para queries que tienen subqueries con su propio ORDER BY (e.g.
    `_check_proactive_outreach` en cron_tasks.py donde la subquery interna
    ordena por `mp2.created_at DESC` y el outer SELECT ordena por
    `mp.user_id, mp.created_at DESC`), el chequeo de matching debe aceptar
    que CUALQUIERA de los ORDER BY del chunk matchee — el ORDER BY del
    outer SELECT es el load-bearing, y el parser no puede distinguir scope
    SQL sin parser completo. Aceptar match a cualquiera evita falsos
    positivos en casos válidos.

    Para queries en string-concat (e.g. `proactive_agent.py:21` donde el
    SELECT está construido como ("SELECT..." "FROM..." "ORDER BY...")),
    extendemos la ventana de búsqueda 3000 chars en lugar de cortar en
    triple-quote, así capturamos el ORDER BY aunque no esté en
    triple-quote literal.
    """
    out: list[tuple[list[str], list[list[str]], int]] = []
    for m in _DISTINCT_ON_RE.finditer(text):
        keys_raw = m.group("keys")
        keys = [_normalize_expr(k) for k in _split_csv(keys_raw)]
        # Ventana fija tras el DISTINCT ON: cubre triple-quote, string-concat,
        # multi-line SQL embebido en raw strings, etc.
        after = text[m.end():m.end() + 3000]
        # Cortamos en el próximo `def `/`async def ` (siguiente función)
        # o en `\n\n\n` (separador de bloque) para no leer queries vecinas.
        for boundary in ("\ndef ", "\nasync def ", "\nclass "):
            bidx = after.find(boundary)
            if bidx != -1:
                after = after[:bidx]
                break
        # Capturamos TODOS los ORDER BY del chunk.
        order_bys: list[list[str]] = []
        for om in _ORDER_BY_ANY_RE.finditer(after):
            cols_raw = om.group(1)
            # Cortar en cierre de string literal o paren.
            cols_raw = re.split(r'"""|\)\s*,', cols_raw, maxsplit=1)[0]
            cols = [
                _normalize_expr(_strip_asc_desc_collate(c))
                for c in _split_csv(cols_raw)
                if _strip_asc_desc_collate(c).strip()
            ]
            if cols:
                order_bys.append(cols)
        line_no = text.count("\n", 0, m.start()) + 1
        out.append((keys, order_bys, line_no))
    return out


def _iter_prod_py_files() -> list[Path]:
    files: list[Path] = []
    for glob in _PROD_GLOBS:
        for f in BACKEND_DIR.glob(glob):
            if f.is_file():
                files.append(f)
    return files


@pytest.fixture(scope="module")
def all_distinct_on_blocks() -> list[tuple[Path, list[str], list[str], int]]:
    blocks: list[tuple[Path, list[str], list[str], int]] = []
    for f in _iter_prod_py_files():
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        for keys, cols, line in _gather_distinct_on_queries(text):
            blocks.append((f, keys, cols, line))
    return blocks


def test_a_at_least_one_distinct_on_in_prod(all_distinct_on_blocks):
    """Sanity: confirma que el parser encontró al menos un DISTINCT ON
    (sin esto, los tests siguientes pasarían triviales y la regresión no
    sería detectada si el código cambia)."""
    assert len(all_distinct_on_blocks) >= 1, (
        "[P0-LIVE-1] El parser no encontró ningún 'SELECT DISTINCT ON' en "
        "backend/*.py — posiblemente el patrón fue refactorizado o el regex "
        "necesita ajuste. Sin DISTINCT ON detectado, este test no protege "
        "contra la regresión del cron P0-C."
    )


def test_b_distinct_on_keys_match_order_by_prefix(all_distinct_on_blocks):
    """Para cada bloque DISTINCT ON, AL MENOS UN ORDER BY del chunk DEBE
    tener los N primeros tokens iguales a los N keys del DISTINCT ON.

    Aceptar "al menos uno" cubre los casos legítimos donde una subquery
    interna ordena por otro criterio (e.g. en `trigger_background_rolling_refill`
    la subquery interior ordena `mp2.created_at DESC` pero el outer SELECT
    ordena `mp.user_id, mp.created_at DESC` y ESE matchea el DISTINCT ON).
    El parser no hace análisis de scope SQL completo, pero el OR-match
    captura la regresión del cron P0-C (1 sólo ORDER BY que NO matcheaba).
    """
    errors: list[str] = []
    for f, keys, order_bys, line in all_distinct_on_blocks:
        if not keys:
            continue
        if not order_bys:
            errors.append(
                f"  {f.name}:{line} — DISTINCT ON sin ORDER BY explícito "
                f"(keys={keys}). Postgres NO garantiza qué fila se conserva "
                f"sin ORDER BY; añadir ORDER BY que empiece con {keys}."
            )
            continue
        # Buscamos si AL MENOS UN ORDER BY matchea el prefix.
        matched = False
        for cols in order_bys:
            if len(cols) < len(keys):
                continue
            if all(k == c for k, c in zip(keys, cols[:len(keys)])):
                matched = True
                break
        if not matched:
            errors.append(
                f"  {f.name}:{line} — DISTINCT ON keys={keys} no matchean "
                f"el prefix inicial de NINGÚN ORDER BY del chunk. "
                f"ORDER BYs encontrados: {order_bys}. "
                f"Postgres exige que el ORDER BY del SELECT con DISTINCT ON "
                f"empiece textualmente con las mismas expresiones del "
                f"DISTINCT ON (con o sin alias prefix, pero consistente)."
            )
    if errors:
        msg = (
            "[P0-LIVE-1] Encontradas violaciones de alias consistency entre "
            "DISTINCT ON y ORDER BY:\n" + "\n".join(errors) + "\n\n"
            "Fix canónico: si el SELECT usa `q.col::text AS col` (alias "
            "shadowing), entonces DISTINCT ON y ORDER BY DEBEN ambos usar "
            "`q.col` (no `col` solo)."
        )
        pytest.fail(msg)


def test_c_pantry_refresh_uses_qualified_alias():
    """Regresión específica del bug original: la query del cron P0-C
    `_proactive_refresh_pending_pantry_snapshots` debe tener
    `DISTINCT ON (q.user_id, q.meal_plan_id)` con prefix `q.`."""
    f = BACKEND_DIR / "cron_tasks.py"
    text = f.read_text(encoding="utf-8")
    fn_marker = "def _proactive_refresh_pending_pantry_snapshots"
    idx = text.find(fn_marker)
    assert idx != -1, (
        "[P0-LIVE-1] Función `_proactive_refresh_pending_pantry_snapshots` "
        "no encontrada en cron_tasks.py — posible refactor masivo. Revisar "
        "este test y el cron P0-C antes de eliminar."
    )
    # Tomamos los próximos ~150 lines tras la definición.
    body = text[idx : idx + 8000]
    assert "DISTINCT ON (q.user_id, q.meal_plan_id)" in body, (
        "[P0-LIVE-1] Esperado `DISTINCT ON (q.user_id, q.meal_plan_id)` con "
        "prefix `q.` en `_proactive_refresh_pending_pantry_snapshots`. Sin "
        "el prefix, Postgres lanza 'must match initial ORDER BY expressions' "
        "en cada tick (cron P0-C roto)."
    )
    assert "ORDER BY q.user_id, q.meal_plan_id, q.execute_after" in body, (
        "[P0-LIVE-1] Esperado `ORDER BY q.user_id, q.meal_plan_id, "
        "q.execute_after` matcheando los keys del DISTINCT ON."
    )


def test_d_anchor_marker_present():
    """El comentario anchor `P0-LIVE-1` debe seguir presente cerca del fix
    para que un futuro audit pueda rastrear el contexto."""
    f = BACKEND_DIR / "cron_tasks.py"
    text = f.read_text(encoding="utf-8")
    assert "[P0-LIVE-1" in text, (
        "[P0-LIVE-1] Anchor `[P0-LIVE-1 · ...]` removido de cron_tasks.py. "
        "El comentario explica por qué el prefix `q.` es load-bearing; "
        "removerlo arriesga regresión silenciosa en un futuro refactor."
    )
