"""[P2-PROD-AUDIT-1 · 2026-05-23] f-strings SQL en `db_*.py` deben
interpolar SOLO constantes locales (column names), NUNCA user input.

Gap original (audit production-readiness 2026-05-23, B-P2-1):
    7 f-strings SQL distribuidos en db_chat.py / db_facts.py:
      - db_chat.py:70,79,93,99,110 → `_AGENT_SESSION_COLS`
      - db_facts.py:505,540 → `_COLUMNS`

    Cada uno interpola una constante function-local (string fija con
    column names del SELECT). Esto es SEGURO (no injection vector), pero:
      (a) SAST tools (bandit S608, ruff S608) flagean f-strings SQL como
          posible injection — escaneo genera ruido + false alarm.
      (b) Sin enforcement, alguien podría introducir un nuevo f-string SQL
          que SÍ interpola variable user-controlled → injection real.

Fix:
    (1) Markers `# noqa: S608` en cada callsite + comment apuntando a
        este test.
    (2) Este test parsea TODO db_*.py via AST y valida que cada f-string
        usado como SQL (heurística: contains keyword SELECT/UPDATE/INSERT/
        DELETE/RETURNING) interpola SOLO `Name` nodes referenciando
        variables que se asignan a string literals constantes en el mismo
        scope.

    Si un nuevo f-string SQL interpola algo dinámico (function call,
    Attribute access, expresión compleja), el test FALLA loud con copy
    explicativo + opciones.

Tooltip-anchor: P2-PROD-AUDIT-1-SQL-FSTRING-CONSTANTS | audit 2026-05-23.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_FILES = [
    _BACKEND_ROOT / "db_chat.py",
    _BACKEND_ROOT / "db_facts.py",
    _BACKEND_ROOT / "db_inventory.py",
    _BACKEND_ROOT / "db_plans.py",
    _BACKEND_ROOT / "db_profiles.py",
    _BACKEND_ROOT / "db_core.py",
    _BACKEND_ROOT / "db_meal_plans_audit.py",
]

_SQL_OPENERS = ("SELECT ", "UPDATE ", "INSERT ", "DELETE ", "WITH ")


def _is_sql_string(node: ast.AST) -> bool:
    """Heurística: el f-string EMPIEZA (primer string part) con un SQL
    opener canónico (SELECT/UPDATE/INSERT/DELETE/WITH). Esto distingue
    queries SQL reales de logger.warning(f"SELECT inicial falló...") que
    contiene la palabra pero no es una query.
    """
    if isinstance(node, ast.JoinedStr):
        # Primer string part del JoinedStr.
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                first = v.value.lstrip().upper()
                if not first:
                    continue
                return any(first.startswith(kw) for kw in _SQL_OPENERS)
            elif isinstance(v, ast.FormattedValue):
                # f-string que empieza con {expr} antes de cualquier static
                # text — improbable que sea SQL puro. Conservative: no SQL.
                return False
    return False


def _collect_local_string_constants(fn_node: ast.AST) -> dict[str, str]:
    """Recorre el body de fn_node y captura assignments `name = "string"`.
    Útil para validar que las variables interpolad as son constantes.
    Recursive a través de control flow (if/try/for) para captar nested.
    """
    out: dict[str, str] = {}
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Assign):
            # Casos comunes:
            #   _COLS = "a, b, c"
            #   _COLS = ("a, b, c "
            #            "d, e")  ← tuple? no — paréntesis de string concat.
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    val = node.value
                    if isinstance(val, ast.Constant) and isinstance(val.value, str):
                        out[name] = val.value
                    # `(STRING "..." STRING "..." )` produce un Constant
                    # implícito (Python lo collapsa al parse). Pero a veces
                    # se asignan tuples con multiline strings — verificar
                    # el constant fold.
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                val = node.value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    out[node.target.id] = val.value
    # También nivel module-level: si fn_node es Module, ya está capturado.
    return out


def _validate_fstring_interpolation_is_constant(
    fstring_node: ast.JoinedStr,
    local_constants: dict[str, str],
) -> tuple[bool, str]:
    """Returns (is_safe, reason).

    Itera los `FormattedValue` (las `{}` partes del f-string) y valida que
    el `value` interpolated sea un `Name` que referencia una constante en
    local_constants.

    Cualquier otro patrón (Call, Attribute, BinOp, Subscript) = posible
    injection → unsafe.
    """
    for part in fstring_node.values:
        if isinstance(part, ast.Constant):
            continue  # static part del f-string
        if not isinstance(part, ast.FormattedValue):
            continue
        expr = part.value
        if isinstance(expr, ast.Name):
            if expr.id not in local_constants:
                return False, (
                    f"interpolated `{expr.id}` no es constante local conocida "
                    f"(local_constants tiene: {list(local_constants.keys())})"
                )
        else:
            # Cualquier expresión que no sea Name simple = unsafe.
            return False, (
                f"interpolación dinámica `{ast.unparse(expr)}` (tipo "
                f"{type(expr).__name__}) — NO es referencia a constante local"
            )
    return True, "OK"


def _find_sql_fstrings(file_path: Path):
    """Yield (lineno, fstring_node, enclosing_fn_constants) para cada
    f-string SQL detectado en el file.
    """
    if not file_path.exists():
        return
    src = file_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(file_path))

    # Map de function → set of local string constants.
    # Top-level constants también cuentan.
    module_constants = _collect_local_string_constants(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_constants = {**module_constants, **_collect_local_string_constants(node)}
            for inner in ast.walk(node):
                if isinstance(inner, ast.JoinedStr) and _is_sql_string(inner):
                    yield (file_path.name, inner.lineno, inner, fn_constants)


def test_at_least_one_sql_fstring_detected():
    """Sanity: el scan debe encontrar al menos 1 SQL f-string conocido
    (`_AGENT_SESSION_COLS` en db_chat.py). Si retorna 0, la heurística
    rompió."""
    total = 0
    for db_file in _DB_FILES:
        for _ in _find_sql_fstrings(db_file):
            total += 1
    assert total >= 5, (
        f"Solo {total} SQL f-strings detectados — esperábamos >=5 "
        f"(db_chat.py tiene ~5 con _AGENT_SESSION_COLS). Heurística rota."
    )


def _has_noqa_s608_near(file_path: Path, lineno: int, window: int = 5) -> bool:
    """True si `# noqa: S608` aparece en las `window` líneas alrededor del
    lineno. Exemption manual + comment justificando son la vía para
    interpolaciones dinámicas safe (e.g. parameterized WHERE clause).
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return False
    start = max(0, lineno - 1 - window)
    end = min(len(lines), lineno + window)
    block = "".join(lines[start:end])
    return "# noqa: S608" in block or "# noqa:S608" in block


def test_all_sql_fstrings_interpolate_only_constants():
    """**Core test**: cada f-string SQL DEBE interpolar SOLO constantes
    locales, O tener `# noqa: S608` marker con comment justificando.

    Si falla: opciones para arreglar el callsite nuevo:
      (a) Convertir la variable interpolada a constante local (string
          literal) si es column name fija conocida.
      (b) Usar placeholders %s + tuple de params (parametrized query).
      (c) Si MUST interpolar runtime value, validar contra whitelist ANTES
          + añadir `# noqa: S608` + comment justificando inline.
    """
    violations = []
    for db_file in _DB_FILES:
        for filename, lineno, fstring_node, local_constants in _find_sql_fstrings(db_file):
            is_safe, reason = _validate_fstring_interpolation_is_constant(
                fstring_node, local_constants
            )
            if is_safe:
                continue
            # Acepta exemption inline.
            if _has_noqa_s608_near(db_file, lineno, window=8):
                continue
            violations.append({
                "file": filename,
                "lineno": lineno,
                "reason": reason,
                "snippet": ast.unparse(fstring_node)[:120],
            })

    if violations:
        detail = "\n".join(
            f"  - {v['file']}:{v['lineno']}\n"
            f"      reason: {v['reason']}\n"
            f"      snippet: {v['snippet']}"
            for v in violations
        )
        msg = (
            f"\n[P2-PROD-AUDIT-1] {len(violations)} SQL f-string(s) interpolan "
            f"valores NO-constantes — posible SQL injection:\n\n{detail}\n\n"
            f"Opciones para arreglar:\n"
            f"  (a) Cambiar a constante local (string literal) si es column\n"
            f"      name fijo conocido.\n"
            f"  (b) Usar placeholders %s + tuple de params (parametrized query).\n"
            f"  (c) Si MUST interpolar runtime value (e.g. table name dynamic):\n"
            f"      validar contra whitelist explícita + comment justificando\n"
            f"      el patrón.\n"
        )
        pytest.fail(msg)


def test_noqa_markers_present_on_known_fstrings():
    """Cada f-string SQL conocido debe tener `# noqa: S608` para silenciar
    SAST sin perder el escaneo legítimo de nuevos casos.
    """
    expected_files = [
        ("db_chat.py", 5),    # 5 f-strings con _AGENT_SESSION_COLS
        ("db_facts.py", 2),   # 2 f-strings con _COLUMNS
    ]
    for filename, expected_count in expected_files:
        path = _BACKEND_ROOT / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        # Contar noqa S608 markers.
        noqa_count = text.count("# noqa: S608")
        assert noqa_count >= expected_count, (
            f"{filename}: solo {noqa_count} markers `# noqa: S608` encontrados, "
            f"esperaba >={expected_count}. Si añadiste un nuevo f-string SQL, "
            f"añadir el marker. Si quitaste uno, validar que el f-string no "
            f"existe o pasó a usar placeholders parametrizados."
        )


def test_anchor_present_in_db_chat():
    """Anchor `P2-PROD-AUDIT-1` presente en db_chat.py — futuro lector que
    vea `# noqa: S608` puede grep el marker y encontrar este test.
    """
    text = (_BACKEND_ROOT / "db_chat.py").read_text(encoding="utf-8")
    assert "P2-PROD-AUDIT-1" in text, (
        "db_chat.py perdió anchor `P2-PROD-AUDIT-1` que explica los "
        "`# noqa: S608` markers. Restaurar."
    )
