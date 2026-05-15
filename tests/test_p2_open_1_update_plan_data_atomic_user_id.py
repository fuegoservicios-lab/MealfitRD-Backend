"""[P2-OPEN-1 · 2026-05-11] Lock-the-contract: TODO caller de
`update_plan_data_atomic` en producción DEBE pasar `user_id=<owner>`
como kwarg.

Motivación:
    `update_plan_data_atomic` (db_plans.py:215) muta `meal_plans.plan_data`
    bajo `SELECT … FOR UPDATE`. Pre-P2-OPEN-1 NO aceptaba `user_id` y
    NO incluía `AND user_id = %s` en SELECT/UPDATE — confiaba en que el
    caller resolviera ownership upstream (todos los callers de prod
    obtienen `meal_plan_id` desde `plan_chunk_queue` claimeado, que ya
    filtró user_id).

    Eso funciona HOY, pero un refactor futuro que reordene la resolución
    (e.g., aceptar `plan_id` desde un body sin re-validar ownership) abre
    IDOR silente. P2-OPEN-1 añadió el parámetro `user_id` opcional + warning
    `[I2-MISS]` cuando el caller lo omite. Este test es el guard:

      1. Enumera los callers conocidos en `backend/**/*.py` (excluyendo
         tests, db_plans.py mismo, y comments).
      2. Cada call site DEBE incluir `user_id=` en sus kwargs — sin
         excepción.

Drift detection:
    Si alguien añade un caller nuevo sin `user_id`, el test falla con la
    línea exacta. Para añadir excepción explícita (e.g., contexto donde
    el helper se invoca desde un script SRE-only sin user scope), añadir
    a `_EXEMPT_CALLERS` con razón inline.

Tooltip-anchor: P2-OPEN-1-START | gap P2 audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND = _REPO_ROOT / "backend"

# Archivos excluidos del scan (no son callers de prod):
#   - db_plans.py: donde la función vive (la definición no es un caller).
#   - tests/: tests legacy en `test_plan_data_atomic_update.py` y demás
#     verifican el path back-compat sin user_id (cubren el comportamiento
#     legacy + el warning I2-MISS); no requerimos migración ahí.
#   - constants.py: solo menciones en comentarios, no llamadas.
_EXCLUDED_NAMES = {"db_plans.py", "constants.py"}

# Excepciones explícitas (si en el futuro un caller no puede pasar user_id
# por contexto operacional — e.g., script SRE migration). Añadir como
# tuple (filename:lineno_approx, razón). Hoy: cero excepciones.
_EXEMPT_CALLERS: dict[str, str] = {}

_CALL_PATTERN = re.compile(
    r"\bupdate_plan_data_atomic\s*\(", re.MULTILINE,
)


def _iter_python_files():
    for f in _BACKEND.rglob("*.py"):
        if not f.is_file():
            continue
        if f.name in _EXCLUDED_NAMES:
            continue
        # Skip tests directory (back-compat tests sí invocan sin user_id).
        parts = {p.lower() for p in f.parts}
        if "tests" in parts:
            continue
        yield f


def _strip_python_comments(src: str) -> str:
    """Elimina comentarios `# ...` desde el primer `#` no-string hasta EOL.

    Heurístico: NO maneja `#` dentro de strings literales (raro en este
    repo — los comments con `update_plan_data_atomic` se escribieron como
    prosa con `(...)` que no matchea el regex de call). Si causa falsos
    negativos, refinar con `ast` parsing.
    """
    out_lines = []
    for line in src.splitlines():
        # Si la línea no tiene `#`, pásala.
        idx = line.find("#")
        if idx < 0:
            out_lines.append(line)
            continue
        # Heurística simple: si el `#` está dentro de un string literal
        # iniciado en la misma línea, no es comentario. Conteo de comillas
        # impar antes del `#` = dentro de string.
        prefix = line[:idx]
        if (prefix.count("'") - prefix.count("\\'")) % 2 == 1:
            out_lines.append(line)
            continue
        if (prefix.count('"') - prefix.count('\\"')) % 2 == 1:
            out_lines.append(line)
            continue
        # Es comentario. Cortar.
        out_lines.append(prefix)
    return "\n".join(out_lines)


def _extract_call_block(src: str, call_start: int) -> str:
    """Extrae el bloque de la llamada `update_plan_data_atomic(...)` desde
    `call_start` (índice del `(` post-nombre) balanceando paréntesis. Útil
    cuando la llamada se extiende multi-línea con kwargs.
    """
    depth = 0
    i = call_start
    while i < len(src):
        ch = src[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return src[call_start:i + 1]
        i += 1
    return src[call_start:]


# ---------------------------------------------------------------------------
# 1. Todos los callers de prod incluyen user_id=
# ---------------------------------------------------------------------------
def test_all_prod_callers_pass_user_id() -> None:
    """Escanea backend/**/*.py (excluyendo tests, db_plans.py mismo y
    constants.py) y exige `user_id=` en cada call site de
    `update_plan_data_atomic`.

    Sin esto, un refactor futuro que reordene la resolución del plan_id
    upstream abre IDOR silente — el SELECT/UPDATE local no tendría el
    filtro defense-in-depth.
    """
    offenders: list[str] = []
    for f in _iter_python_files():
        try:
            src = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        no_comments = _strip_python_comments(src)
        for m in _CALL_PATTERN.finditer(no_comments):
            call_start = m.end() - 1  # posición del `(` de la llamada
            call_block = _extract_call_block(no_comments, call_start)
            # Match estricto: kwarg `user_id=` debe estar en el call block.
            # Aceptamos `user_id=user_id`, `user_id=str(...)`, `user_id=fd.get(...)`.
            if not re.search(r"\buser_id\s*=", call_block):
                # Linea aproximada del call (basada en el src original).
                line_no = no_comments.count("\n", 0, m.start()) + 1
                rel_path = f.relative_to(_REPO_ROOT)
                key = f"{rel_path}:{line_no}"
                if key in _EXEMPT_CALLERS:
                    continue
                snippet = call_block[:200].replace("\n", " ")
                offenders.append(f"  {key} → call sin user_id: {snippet!r}")

    assert not offenders, (
        "P2-OPEN-1 violation: uno o más callers de "
        "`update_plan_data_atomic` NO pasan `user_id=` como kwarg. "
        "Cada call site DEBE pasar el `user_id` del owner para que el "
        "SELECT/UPDATE incluya `AND user_id = %s` (defense-in-depth I2 "
        "según CLAUDE.md).\n\n"
        "Offenders:\n"
        + "\n".join(offenders)
        + "\n\nOpciones de cierre:\n"
        "  1. Resolver el `user_id` upstream (típicamente desde el `task` "
        "claimeado de `plan_chunk_queue`, o desde el `verified_user_id` del "
        "handler) y pasarlo como kwarg `user_id=<owner>`.\n"
        "  2. Si por contexto operacional NO es factible pasar user_id, "
        "añadir el callsite a `_EXEMPT_CALLERS` arriba con razón documentada."
    )


# ---------------------------------------------------------------------------
# 2. Helper soporta el parámetro y emite warning I2-MISS si se omite
# ---------------------------------------------------------------------------
def test_helper_accepts_user_id_kwarg() -> None:
    """`update_plan_data_atomic` debe aceptar `user_id: Optional[str] = None`
    como keyword-only. Sin esto, los callers no pueden migrar."""
    db_plans = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    # Buscar la firma multi-línea.
    sig_pat = re.compile(
        r"def\s+update_plan_data_atomic\s*\("
        r"[\s\S]+?"
        r"user_id\s*:\s*Optional\[\s*str\s*\]\s*=\s*None",
        re.MULTILINE,
    )
    assert sig_pat.search(db_plans), (
        "P2-OPEN-1: `update_plan_data_atomic` debe aceptar "
        "`user_id: Optional[str] = None` como keyword-only argument. "
        "La firma actual no lo declara — los callers no podrán migrar."
    )


def test_helper_emits_i2_miss_warning_when_user_id_none() -> None:
    """El helper debe loggear `[I2-MISS]` cuando un caller omite `user_id`.
    Sin esto, SRE no detecta callers nuevos no-migrados en producción."""
    db_plans = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    assert "[I2-MISS]" in db_plans, (
        "P2-OPEN-1: el helper debe emitir `logger.warning('[I2-MISS] ...')` "
        "cuando un caller invoca sin `user_id`. Este log permite a SRE "
        "detectar callers no-migrados en grep de logs de prod."
    )


def test_helper_filters_select_and_update_with_user_id() -> None:
    """Cuando `user_id` está presente, tanto el SELECT como el UPDATE
    deben incluir `AND user_id = %s`."""
    db_plans = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    # Extraer función completa via brace-style heurística: del `def
    # update_plan_data_atomic` hasta la siguiente declaración top-level.
    fn_pat = re.compile(
        r"def\s+update_plan_data_atomic\s*\([\s\S]+?(?=\ndef\s|\Z)",
        re.MULTILINE,
    )
    m = fn_pat.search(db_plans)
    assert m, "Función `update_plan_data_atomic` no encontrada."
    fn_body = m.group(0)

    # SELECT con AND user_id = %s
    has_select_filter = bool(re.search(
        r"SELECT\s+plan_data\s+FROM\s+meal_plans\s+"
        r"WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s\s+FOR\s+UPDATE",
        fn_body, re.IGNORECASE,
    ))
    assert has_select_filter, (
        "P2-OPEN-1: cuando `user_id` está presente, el SELECT FOR UPDATE "
        "debe incluir `AND user_id = %s` además de `id = %s`."
    )

    # UPDATE con AND user_id = %s
    has_update_filter = bool(re.search(
        r"UPDATE\s+meal_plans\s+SET\s+plan_data\s*=\s*%s::jsonb\s+"
        r"WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s",
        fn_body, re.IGNORECASE,
    ))
    assert has_update_filter, (
        "P2-OPEN-1: cuando `user_id` está presente, el UPDATE debe incluir "
        "`AND user_id = %s` además de `id = %s`."
    )


# ---------------------------------------------------------------------------
# 3. P3-OPEN-1 anchor: enumerar callers conocidos por (filename, función helper)
# ---------------------------------------------------------------------------
#
# Complementa el blanket arriba: P2-OPEN-1 exige `user_id=` en CADA call site
# (regex). P3-OPEN-1 enumera los callers conocidos por NOMBRE — si un call
# site nuevo aparece en un archivo no listado, el test falla pidiendo
# decisión explícita (añadir al anchor o reusar uno existente). Drift
# detection más estricta paralela a P3-NEXT-1 sobre el filter user_id.
#
# Si añades un caller nuevo y es legítimo, simplemente añádelo a la set
# `_KNOWN_CALLERS_BY_FILE`. Si un caller existente desaparece (e.g., refactor
# que delegue a otro helper), elimínalo. El test convierte el conjunto de
# archivos con calls en estado-verificado por humano.
_KNOWN_CALLERS_BY_FILE: set[str] = {
    # cron_tasks.py 5 sitios documentados:
    #   - _activate_flexible_mode → _append_mode_event       (P1-3/MODE-HISTORY)
    #   - _record_learning_loss → _mutator                    (P1-6/LEARNING-LOSS)
    #   - _chunk_worker → _p13_proxy_mutator                  (P1-3 proxy branch)
    #   - _chunk_worker → _p13_strong_mutator                 (P1-3 strong branch)
    #   - _chunk_worker → _bump_zero_log                      (P0-2/ZERO-LOG)
    "backend/cron_tasks.py",
    # routers/plans.py 2 sitios documentados:
    #   - api_recalculate_shopping_list → _apply_recalc           (P1-RECALC-LOSTUPDATE · 2026-05-14)
    #   - api_expand_recipe → _apply_recipe_expansion             (P1-AUDIT-1 · 2026-05-15)
    # Resuelve ownership upstream con `verified_user_id == user_id` check
    # al inicio del handler + SELECT explícito con `AND user_id = %s` en
    # el branch req_plan_id (P2-NEW-B) o `get_latest_meal_plan_with_id(user_id)`
    # en el fallback. Pasa `user_id=user_id` al helper.
    "backend/routers/plans.py",
    # proactive_agent.py 1 sitio documentado:
    #   - _trigger_week2_background_generation → _apply_week2_append (P1-AUDIT-1 · 2026-05-15)
    # Resuelve ownership upstream: `plan_id` viene del cron
    # `check_and_trigger_jit_rolling_windows` que hace
    # `SELECT id, user_id, plan_data FROM meal_plans WHERE created_at >= NOW() - 6 days`
    # — `user_id` recuperado del mismo row, así la pareja (plan_id, user_id) es
    # consistent-by-construction. Pasa `user_id=user_id` al helper.
    "backend/proactive_agent.py",
    # tools.py 1 sitio documentado:
    #   - execute_modify_single_meal → _apply_meal_modification   (P1-AUDIT-1 · 2026-05-15)
    # Resuelve ownership upstream: el `@tool modify_single_meal` recibe
    # `user_id` force-overrideado por `execute_tools` con `_trusted_uid`
    # (P0-AGENT-1, defense contra prompt injection que emite user_id ajeno);
    # plan_id viene de `get_latest_meal_plan_with_id(user_id)` que filtra
    # por user_id. Pasa `user_id=user_id` al helper.
    "backend/tools.py",
}


def test_known_callers_anchor() -> None:
    """[P3-OPEN-1 · 2026-05-11] Enumera explícitamente los archivos con
    callers de `update_plan_data_atomic`. Si un archivo nuevo aparece sin
    estar en `_KNOWN_CALLERS_BY_FILE`, el test falla pidiendo decisión
    humana: ¿el caller nuevo es legítimo y resuelve ownership upstream
    correctamente?

    Esto es ortogonal al blanket `test_all_prod_callers_pass_user_id`:
    aquel solo checa que cada call site tenga `user_id=`; éste pide
    aprobación humana del archivo entero al introducir uno nuevo.
    Útil porque un caller nuevo puede pasar `user_id` correctamente pero
    aún así romper la cadena de ownership upstream (e.g., resolver
    plan_id desde un body sin verificar que pertenezca al user).
    """
    discovered_files: set[str] = set()
    for f in _iter_python_files():
        try:
            src = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        no_comments = _strip_python_comments(src)
        if _CALL_PATTERN.search(no_comments):
            # Normalizar a posix-style relative path.
            rel = f.relative_to(_REPO_ROOT).as_posix()
            discovered_files.add(rel)

    unexpected = discovered_files - _KNOWN_CALLERS_BY_FILE
    missing = _KNOWN_CALLERS_BY_FILE - discovered_files

    msgs: list[str] = []
    if unexpected:
        msgs.append(
            "Archivos NUEVOS con calls a `update_plan_data_atomic` no listados "
            "en `_KNOWN_CALLERS_BY_FILE`:\n"
            + "\n".join(f"  + {p}" for p in sorted(unexpected))
            + "\n\nDecide manualmente: ¿el caller nuevo resuelve ownership "
              "upstream correctamente (plan_id viene de plan_chunk_queue "
              "claimeado / verified_user_id check)? Si sí, añade el archivo "
              "a `_KNOWN_CALLERS_BY_FILE` con un comentario que enumere los "
              "helpers/mutators concretos. Si no, esa es la raíz del bug — "
              "resuelve ownership antes de llamar al helper."
        )
    if missing:
        msgs.append(
            "Archivos listados en `_KNOWN_CALLERS_BY_FILE` ya NO contienen "
            "calls a `update_plan_data_atomic`:\n"
            + "\n".join(f"  - {p}" for p in sorted(missing))
            + "\n\nProbablemente fueron refactorizados a otro helper. "
              "Elimínalos de la lista para que el anchor refleje el estado real."
        )

    assert not msgs, "[P3-OPEN-1] Drift detection:\n\n" + "\n\n".join(msgs)


# ---------------------------------------------------------------------------
# 4. Slug del marker en filename (cross-link audit)
# ---------------------------------------------------------------------------
def test_marker_anchor_present() -> None:
    """Filename contiene `p2_open_1` para cross-link con
    `test_p2_hist_audit_14_marker_test_link` cuando el marker se bumpee."""
    assert "p2_open_1" in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p2_open_1`) para el cross-link."
    )
