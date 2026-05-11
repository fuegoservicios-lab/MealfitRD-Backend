"""[P1-NEW-A · 2026-05-11] Test blanket: el frontend NO debe escribir
directo a `meal_plans` vía `supabase.from('meal_plans').(update|delete|upsert)`.

Motivación:
    P1-HIST-5 (rename atómico, 2026-05-09) cerró el primer caso conocido
    de direct write client-side: `History.jsx::handleEditSave` que hacía
    ``supabase.from('meal_plans').update({ name }).eq('id', planId)`` —
    solo actualizaba la columna top-level, dejando ``plan_data.name`` con
    el valor viejo. Cerró con endpoint `/api/plans/{plan_id}/name` (jsonb_set
    sobre ambas representaciones).

    P0-NEW-A y P0-NEW-B (2026-05-11) cerraron dos casos más graves:
      - Swap de meal: `supabase.from('meal_plans').update({plan_data}).eq('id', planId)`
        pisando JSONB completo desde el cliente → lost-update con
        `_chunk_worker`.
      - Inyección de `grocery_start_date`/`cycle_start_date`: mismo
        patrón, menor frecuencia pero mismo modo de fallo.

    La clase de bug se ramificó silenciosamente porque NINGÚN test
    bloqueaba un futuro callsite de `supabase.from('meal_plans').update(...)`.
    Cada uno requirió un test específico (`History.rename_atomic.test.js`,
    `test_p0_new_a_swap_persists_backend.py`, etc.). P1-NEW-A generaliza
    la defensa: en lugar de añadir un test por callsite, escanea TODO
    `frontend/src/**/*.{js,jsx,ts,tsx}` y prohíbe el patrón salvo
    whitelist explícita.

Whitelist (marker inline):
    Comentario JS dentro del archivo:
        // [P1-NEW-A WHITELIST: <razón clara y específica>]
    El test extrae el marker más cercano (≤ 30 líneas arriba del match)
    y cuenta como exención. Sin marker, el matche es violation.

    Migración esperada:
      - `restorePlan` (AssessmentContext.jsx ~L1501): patrón legacy para
        revertir regen rechazado desde state local. Candidato a migrar a
        endpoint `/api/plans/{plan_id}/restore-local` en P2. Whitelist
        documentada hasta entonces.
      - `Plan.jsx:398`: INSERT inicial del plan recién generado. Este
        test NO bloquea INSERT (solo update/delete/upsert) — el INSERT
        inicial es legítimo porque el plan no existe aún en DB.

Drift detection:
    - Nuevo callsite `.update|delete|upsert` sin marker → test falla
      con archivo + línea + snippet.
    - Whitelist sin texto (e.g. `// [P1-NEW-A WHITELIST: ]`) → marker
      regex no matchea, el match queda como violation.

Tooltip-anchor: P1-NEW-A-START | gap P1 audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"

# Verbos prohibidos (insert queda permitido — INSERT inicial es legítimo;
# el cliente no tiene plan_id hasta que el INSERT lo crea).
_FORBIDDEN_VERBS = ("update", "delete", "upsert")

_PATTERN = re.compile(
    r"supabase\s*\.\s*from\s*\(\s*['\"]meal_plans['\"]\s*\)"
    r"\s*\.\s*(?P<verb>" + "|".join(_FORBIDDEN_VERBS) + r")\b",
    re.IGNORECASE | re.DOTALL,
)

_WHITELIST_MARKER = re.compile(
    r"//\s*\[P1-NEW-A\s+WHITELIST:\s*(?P<reason>[^\]]+?)\s*\]"
)


def _strip_js_comments(src: str) -> str:
    """Elimina:
      - Bloques `/* ... */` (anclados en cualquier columna).
      - Líneas de comentario `// ...` (incluyendo cuando aparecen al
        final de una línea con código antes).

    El segundo caso es importante: una línea como
    ``foo();  // supabase.from('meal_plans').update(...)`` no es código
    ejecutable real. Eliminamos del primer `//` hasta el EOL para evitar
    falsos positivos en comentarios de fin de línea que citen el patrón
    en prosa.

    Nota: este stripper es heurístico — no es un parser JS completo. NO
    cubre el caso patológico de `//` dentro de un string literal
    (e.g. ``const url = 'http://...';``). Si causa falsos negativos,
    refinar con un parser real (acorn-loose) en P2.
    """
    # Bloques /* ... */ primero (incluye multi-línea con DOTALL).
    no_block = re.sub(r"/\*[\s\S]*?\*/", "", src)
    # Líneas con `//`: cortar desde `//` hasta el EOL. NO eliminamos la
    # línea entera (preserva números de línea consistentes con el archivo).
    no_line = re.sub(r"//[^\n]*", "", no_block)
    return no_line


def _iter_frontend_files():
    """Yields paths de todos los .js/.jsx/.ts/.tsx bajo frontend/src
    excluyendo:
      - Carpetas `__tests__` (los tests pueden citar el patrón).
      - Archivos `*.test.js`/`*.test.jsx`/`*.test.ts`/`*.test.tsx`.
      - Archivos de declaración `*.d.ts` (no contienen código ejecutable).
    """
    for f in _FRONTEND_SRC.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix not in {".js", ".jsx", ".ts", ".tsx"}:
            continue
        # Excluir carpetas y archivos de test.
        parts = {p.lower() for p in f.parts}
        if "__tests__" in parts:
            continue
        name_low = f.name.lower()
        if (
            name_low.endswith(".test.js")
            or name_low.endswith(".test.jsx")
            or name_low.endswith(".test.ts")
            or name_low.endswith(".test.tsx")
            or name_low.endswith(".d.ts")
        ):
            continue
        yield f


# ---------------------------------------------------------------------------
# 1. No direct writes (excepto whitelist documentada)
# ---------------------------------------------------------------------------
def test_frontend_has_no_direct_meal_plans_writes():
    """Escanea frontend/src/** y falla si encuentra
    `supabase.from('meal_plans').(update|delete|upsert)(...)` sin un
    marker `// [P1-NEW-A WHITELIST: <razón>]` dentro de las 30 líneas
    previas.
    """
    offenders: list[str] = []
    for f in _iter_frontend_files():
        try:
            src = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Archivos binarios bajo src son síntoma de bug, no del
            # patrón que estamos enforzando. Skip silente.
            continue
        no_comments = _strip_js_comments(src)
        for m in _PATTERN.finditer(no_comments):
            line_no = no_comments.count("\n", 0, m.start()) + 1
            verb = m.group("verb").lower()
            # Buscar whitelist marker en las 30 líneas anteriores del
            # archivo ORIGINAL (no del stripped — los comentarios viven
            # ahí).
            src_lines = src.splitlines()
            start_window = max(0, line_no - 30)
            window_text = "\n".join(src_lines[start_window:line_no])
            wl_match = _WHITELIST_MARKER.search(window_text)
            if wl_match and wl_match.group("reason").strip():
                continue
            snippet = m.group(0)[:120]
            rel_path = f.relative_to(_REPO_ROOT)
            offenders.append(
                f"  {rel_path}:{line_no} → .{verb}(...) ({snippet!r})"
            )

    assert not offenders, (
        "P1-NEW-A violation: el frontend hace direct write a `meal_plans` "
        "sin pasar por endpoint backend (update/delete/upsert). Cada "
        "callsite reabre el modo de fallo lost-update que P0-NEW-A "
        "(swap) y P0-NEW-B (grocery-date) cerraron — un write desde el "
        "cliente pisa el JSONB completo, perdiendo lo que "
        "`_chunk_worker` o crons mutaron en paralelo.\n\n"
        "Offenders:\n"
        + "\n".join(offenders)
        + "\n\nOpciones de cierre:\n"
        "  1. Migrar el callsite a un endpoint backend que use "
        "`jsonb_set` quirúrgico + `AND user_id = %s` (espejo P0-NEW-A/B).\n"
        "  2. Si la migración no es prioridad inmediata, marcar la "
        "exception inline con `// [P1-NEW-A WHITELIST: <razón clara>]` "
        "dentro de las 30 líneas previas al callsite. El marker DEBE "
        "tener texto después del `:`."
    )


# ---------------------------------------------------------------------------
# 2. Whitelist hygiene: cada whitelist exception tiene razón documentada
# ---------------------------------------------------------------------------
def test_whitelist_markers_have_non_empty_reason():
    """Cada `// [P1-NEW-A WHITELIST: <reason>]` debe tener texto no-vacío
    después del colon. Sin razón documentada, el whitelist se convierte
    en `noqa` ciego.
    """
    bad: list[str] = []
    # Permitimos marker vacío como sintaxis válida (e.g. `// [P1-NEW-A
    # WHITELIST: ]`), pero el _WHITELIST_MARKER ya exige `[^\]]+?` (al
    # menos 1 char). Escaneamos manualmente para detectar el shape
    # vacío que el regex rechaza pero un humano podría intentar.
    bare_pat = re.compile(r"//\s*\[P1-NEW-A\s+WHITELIST:\s*\]")
    for f in _iter_frontend_files():
        try:
            src = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for m in bare_pat.finditer(src):
            line_no = src.count("\n", 0, m.start()) + 1
            rel_path = f.relative_to(_REPO_ROOT)
            bad.append(f"  {rel_path}:{line_no} → whitelist marker vacío")

    assert not bad, (
        "P1-NEW-A whitelist hygiene: uno o más `// [P1-NEW-A WHITELIST: ]` "
        "tienen razón vacía. Documentar la razón específica del bypass "
        "(e.g. 'legacy restorePlan path, candidato P2 fix'):\n"
        + "\n".join(bad)
    )


# ---------------------------------------------------------------------------
# 3. Cross-link slug del marker
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """Slug del filename matchea el marker `P1-NEW-A`."""
    expected_slug = "p1_new_a"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p1_new_a`) para que el cross-link "
        "`test_p2_hist_audit_14_marker_test_link` lo matchee cuando "
        "el marker se bumpee a `P1-NEW-A · 2026-05-11`."
    )
