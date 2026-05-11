"""[P3-NEXT-1 · 2026-05-11] Lock-the-contract sistémico de la invariante I2
(CLAUDE.md): TODO `UPDATE meal_plans` dentro de un handler `@router.<verb>` en
`routers/plans.py` DEBE filtrar `AND user_id = %s` en su cláusula WHERE.

Contexto:
    P2-NEXT-1 (2026-05-11) cerró los 3 sitios conocidos (api_shift_plan +
    _background_shift_plan_for_user) que omitían el filtro user_id en
    `UPDATE meal_plans`. P3-NEXT-1 generaliza la defensa: en lugar de
    enumerar funciones específicas, escanea TODOS los handlers
    `@router.<verb>` de `routers/plans.py` y exige el filtro en cada
    UPDATE encontrado.

    El patrón es paralelo a otros lock-the-contract tests del repo:
      - `test_p3_audit_7_plan_result_keys_contract.py` (plan_result keys)
      - `test_p2_audit_4_alert_keys_documented.py` (system_alerts)
      - `test_p0_audit_1_history_quota_exemption.py` (GET quota gates)

Estrategia:
    1. Parsear `routers/plans.py` y extraer el body de CADA función con
       decorador `@router.<verb>` (get|post|patch|delete|put).
    2. En cada body, buscar todas las apariciones de `UPDATE meal_plans`.
    3. Para cada UPDATE, reconstruir el statement (tolerante a Python
       adjacent-string concatenation multi-línea y triple-quoted strings).
    4. Verificar que el statement contiene `user_id = %s` (case-insensitive).
    5. Si no, registrar como offender con función + línea + snippet.
    6. Whitelist explícita vía marker inline:
         `# [P3-NEXT-1 WHITELIST: <razón>]`
       dentro de la función. Si futuro un endpoint necesita excepción
       documentada, el marker la habilita sin disolver el contrato.

Drift detection bidireccional:
    - Nuevo endpoint con `UPDATE meal_plans` sin user_id → el test falla
      con mensaje accionable nombrando función + snippet.
    - Reverter cualquier filtro user_id existente → mismo failure.
    - Whitelist sin justificación → el marker regex exige texto después
      del colon (impide "WHITELIST:" vacío).

Pareja con P2-NEXT-1:
    P2-NEXT-1 ancla 2 funciones específicas (api_shift_plan + cron mirror).
    P3-NEXT-1 cubre el universo de routers user-facing. Los dos se solapan
    intencionalmente en api_shift_plan — redundancia barata; si alguien
    elimina P3-NEXT-1 por accidente, P2-NEXT-1 sigue protegiendo el sitio
    crítico.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"

# Verbos HTTP que registra el router. Si FastAPI gana otro decorador
# (e.g. router.head, router.options), añadirlo acá.
_ROUTER_VERBS = ("get", "post", "patch", "delete", "put")

# Marker para whitelist explícita dentro de una función. Texto después del
# colon es OBLIGATORIO (no se acepta `WHITELIST:` solo). El test extrae
# este marker y lo cuenta como exención solo si tiene justificación.
_WHITELIST_MARKER = re.compile(
    r"#\s*\[P3-NEXT-1\s+WHITELIST:\s*(?P<reason>[^\]]+)\]"
)


def _extract_router_handlers(src: str) -> dict[str, tuple[int, str]]:
    """Devuelve {fn_name: (start_line, body)} para cada función decorada
    con `@router.<verb>`. El body va desde la línea del `def`/`async def`
    hasta el siguiente top-level `@router.*`/`@app.*`/`def`.
    """
    lines = src.splitlines()
    handlers: dict[str, tuple[int, str]] = {}

    # Pattern: `@router.<verb>(...)` posiblemente multi-línea, seguido por
    # `def <name>(` o `async def <name>(`. Encontramos el @router primero.
    router_decorator_re = re.compile(
        rf"^@router\.({'|'.join(_ROUTER_VERBS)})\s*\("
    )
    fn_re = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(")
    boundary_re = re.compile(r"^(?:@router\.|@app\.|def\s|async\s+def\s)")

    i = 0
    while i < len(lines):
        if router_decorator_re.match(lines[i]):
            # Avanzar hasta el `def` correspondiente (decorador puede ser
            # multi-línea o tener otros decoradores intermedios).
            j = i + 1
            fn_name = None
            while j < len(lines):
                m = fn_re.match(lines[j])
                if m:
                    fn_name = m.group(1)
                    fn_start = j
                    break
                # Skip otros decoradores (e.g. `@deprecated`).
                if lines[j].lstrip().startswith("@"):
                    j += 1
                    continue
                # Línea de continuación del decorador (paréntesis abierto).
                j += 1
            if fn_name is None:
                i += 1
                continue

            # Body: de fn_start hasta siguiente boundary.
            k = fn_start + 1
            while k < len(lines):
                if boundary_re.match(lines[k]):
                    break
                k += 1
            body = "\n".join(lines[fn_start:k])
            handlers[fn_name] = (fn_start + 1, body)
            i = k
        else:
            i += 1

    return handlers


def _strip_function_docstring(body: str) -> str:
    """Elimina el docstring inicial de la función (si existe) para que
    referencias SQL-en-prosa dentro del docstring (e.g. `api_restore_plan`
    citando el bug original con `UPDATE meal_plans SET plan_data = ? WHERE
    id = <latest>`) no sean confundidas con SQL real ejecutado.

    Heurística: el docstring es el primer literal triple-quoted que aparece
    DESPUÉS de la línea `def`/`async def` y antes de cualquier statement
    Python ejecutable (en línea con la convención PEP 257). Si no hay
    docstring, el body se devuelve intacto.
    """
    # Encontrar la primera línea no-blanca después de `def ...:`
    lines = body.splitlines()
    # Saltar la signature (posiblemente multi-línea: encontrar el `:` final).
    paren_depth = 0
    sig_end = 0
    for idx, line in enumerate(lines):
        for ch in line:
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
        if paren_depth == 0 and line.rstrip().endswith(":") and "def" in lines[0]:
            sig_end = idx
            break
    # Buscar primer non-blank después de la signature.
    docstring_start = None
    for idx in range(sig_end + 1, len(lines)):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        # Si arranca con triple-quote, es docstring.
        if stripped.startswith('"""') or stripped.startswith("'''"):
            docstring_start = idx
        break
    if docstring_start is None:
        return body
    # Encontrar el cierre del docstring.
    quote = '"""' if lines[docstring_start].strip().startswith('"""') else "'''"
    # Caso 1: docstring de una sola línea (`"""foo"""`).
    first_line_stripped = lines[docstring_start].strip()
    if first_line_stripped.endswith(quote) and len(first_line_stripped) >= 6:
        # `"""x"""` o `"""..."""` en la misma línea.
        return "\n".join(lines[: docstring_start] + lines[docstring_start + 1 :])
    # Caso 2: docstring multi-línea — buscar cierre.
    for idx in range(docstring_start + 1, len(lines)):
        if quote in lines[idx]:
            return "\n".join(lines[: docstring_start] + lines[idx + 1 :])
    # Sin cierre encontrado — devolver original (defensa contra parser roto).
    return body


def _find_meal_plans_updates(body: str) -> list[tuple[int, str]]:
    """[(line_no_relativo_1based, statement_reconstruido)] por cada
    `UPDATE meal_plans` en el body. Reconstruye uniendo líneas hasta
    encontrar la cláusula WHERE (heurística: la mayoría de UPDATE caben
    en ≤25 líneas; cap defensivo a 50).

    Strippea el docstring de la función primero para evitar matches en
    prosa documental que cita SQL del bug histórico.
    """
    body = _strip_function_docstring(body)
    results: list[tuple[int, str]] = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        if "UPDATE meal_plans" in lines[i]:
            chunk_lines = [lines[i]]
            j = i + 1
            # Acumular hasta encontrar WHERE + cierre del statement.
            # Cierre = línea con `"""` o `"`+`,` o `)` solo o ); .
            while j < len(lines):
                chunk_lines.append(lines[j])
                joined_so_far = " ".join(chunk_lines).upper()
                if " WHERE " in joined_so_far:
                    # Tras WHERE, capturar hasta el cierre del statement
                    # (otra ~8 líneas máximo para incluir AND user_id que
                    # podría venir post-WHERE en multi-línea).
                    after_where_count = sum(
                        1 for l in chunk_lines[-1:] if " WHERE " in l.upper()
                    )
                    if after_where_count or j - i > 6:
                        # Capturar 6 líneas más para asegurar AND user_id.
                        end_extra = min(j + 6, len(lines) - 1)
                        chunk_lines.extend(lines[j + 1 : end_extra + 1])
                        j = end_extra
                        break
                j += 1
                if j - i > 50:
                    break
            statement = "\n".join(chunk_lines)
            results.append((i + 1, statement))
            i = j + 1
        else:
            i += 1
    return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def handlers(plans_src: str) -> dict[str, tuple[int, str]]:
    return _extract_router_handlers(plans_src)


# ---------------------------------------------------------------------------
# Core contract
# ---------------------------------------------------------------------------


def test_all_router_handlers_filter_user_id_on_meal_plans_update(handlers):
    """TODO `UPDATE meal_plans` dentro de un handler `@router.<verb>` DEBE
    incluir `user_id = %s` en su WHERE. Whitelist explícita vía marker
    `# [P3-NEXT-1 WHITELIST: <razón>]` dentro de la función."""
    assert handlers, (
        "P3-NEXT-1 sanity: no se encontró ningún handler `@router.<verb>` "
        "en `routers/plans.py`. El extractor regex puede estar roto, "
        "o el archivo se renombró. Verificar `_extract_router_handlers`."
    )

    offenders: list[str] = []
    for fn_name, (start_line, body) in handlers.items():
        updates = _find_meal_plans_updates(body)
        if not updates:
            continue

        whitelist_match = _WHITELIST_MARKER.search(body)
        whitelist_reason = (
            whitelist_match.group("reason").strip()
            if whitelist_match
            else None
        )

        for rel_line, stmt in updates:
            has_user_id_filter = bool(
                re.search(r"\buser_id\s*=\s*%s\b", stmt, flags=re.IGNORECASE)
            )
            if has_user_id_filter:
                continue
            if whitelist_reason:
                # Excepción documentada explícitamente. El marker se
                # registra una sola vez; aplica a todos los UPDATE del
                # handler (decisión simple — si en el futuro un handler
                # necesita whitelist por-statement, refactorizar el marker).
                continue
            abs_line = start_line + rel_line - 1
            snippet = stmt.replace("\n", " ").strip()[:240]
            offenders.append(
                f"  {fn_name} (plans.py:~{abs_line}): {snippet}"
            )

    assert not offenders, (
        "P3-NEXT-1 violation: uno o más handlers `@router.<verb>` en "
        "`routers/plans.py` tienen `UPDATE meal_plans` sin `AND user_id = %s` "
        "en el WHERE. Esto rompe la invariante I2 documentada en CLAUDE.md "
        "(`Toda mutación de meal_plans filtra AND user_id = %s`) y reabre el "
        "riesgo IDOR si la resolución upstream del plan_id se refactoriza. "
        "Para cada offender:\n"
        + "\n".join(offenders)
        + "\n\nSi la excepción es legítima (e.g. admin endpoint que opera "
        "sobre planes arbitrarios), añadir dentro de la función un comentario:\n"
        "  # [P3-NEXT-1 WHITELIST: <razón clara y específica>]"
    )


# ---------------------------------------------------------------------------
# Whitelist hygiene
# ---------------------------------------------------------------------------


def test_whitelist_markers_have_justification(plans_src: str):
    """Si alguien añade `# [P3-NEXT-1 WHITELIST:` con justificación vacía
    o solo whitespace, el regex `[^\\]]+` ya lo rechaza al parsear. Este
    test adicional valida que CADA marker presente tenga texto significativo
    (>= 12 chars) — bloquea `WHITELIST: ok` o `WHITELIST: -`."""
    for match in _WHITELIST_MARKER.finditer(plans_src):
        reason = match.group("reason").strip()
        assert len(reason) >= 12, (
            f"P3-NEXT-1 whitelist marker con justificación insuficiente: "
            f"'{reason}'. Mínimo 12 caracteres de razón documentada. "
            f"Whitelist es una excepción al contrato I2 — debe explicar "
            f"POR QUÉ este endpoint no necesita el filtro user_id."
        )


# ---------------------------------------------------------------------------
# Coverage floor
# ---------------------------------------------------------------------------


def test_at_least_one_handler_mutates_meal_plans(handlers, plans_src):
    """Sanity: el test pasaría trivialmente si NINGÚN handler tocara
    `meal_plans`. Verificamos que hay al menos 3 handlers conocidos que
    mutan meal_plans, asegurando que el contract test realmente está
    ejerciendo el invariante.

    Si en el futuro `routers/plans.py` se splittea y los UPDATE migran a
    otros routers, este test debe actualizarse (o moverse) — pero falla
    explícitamente en vez de pasar trivialmente.
    """
    handlers_with_updates = [
        fn_name
        for fn_name, (_, body) in handlers.items()
        if _find_meal_plans_updates(body)
    ]
    assert len(handlers_with_updates) >= 3, (
        f"P3-NEXT-1 coverage floor: solo {len(handlers_with_updates)} "
        f"handler(s) mutan meal_plans en plans.py "
        f"({handlers_with_updates}). Esperábamos >=3 (api_shift_plan, "
        f"api_retry_chunk, api_restore_plan, api_rename_plan, "
        f"api_regenerate_dead_lettered_simplified, api_regen_degraded_chunks). "
        f"Si los UPDATE migraron a otro router/módulo, mover este test "
        f"o expandir el scope del extractor."
    )


def test_router_extractor_finds_known_endpoints(handlers):
    """Floor anchor: el extractor encuentra los endpoints user-facing
    críticos. Si el regex de decoradores se rompe, este test falla antes
    que el contract test pase vacío."""
    expected_present = {
        "api_shift_plan",
        "api_retry_chunk",
        "api_restore_plan",
        "api_rename_plan",
        "api_recalculate_shopping_list",
        "api_chunk_status",
        "api_restock",
    }
    missing = expected_present - set(handlers.keys())
    assert not missing, (
        f"P3-NEXT-1 extractor floor: faltan endpoints conocidos en el "
        f"resultado del extractor: {missing}. "
        f"`_extract_router_handlers` puede estar fallando con cierto patrón "
        f"de decorador. Si los endpoints se renombraron, actualizar la "
        f"lista esperada."
    )
