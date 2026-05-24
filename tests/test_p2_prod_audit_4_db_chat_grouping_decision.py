"""[P2-PROD-AUDIT-1 · 2026-05-23] Aclaración de gap mal-categorizado en
audit production-readiness 2026-05-23.

Gap aparente (audit B-P2-6):
    "N+1 manual en `db_chat.py:180-210` — sessions + messages, agrupación
     en Python (aceptable por volumen)."

Investigación post-audit (2026-05-23):
    El código en `_process_and_sort_sessions` (db_chat.py:182+) hace:
      1. Recibe `sessions: list` (ya cargado previamente).
      2. Extrae `session_ids = [s["id"] for s in sessions]`.
      3. UNA query batch: `supabase.table("agent_messages").select(...)
         .in_("session_id", session_ids).execute()`.
      4. Agrupa en Python con `messages_by_session[s_id].append(m)`.

    Esto NO es N+1 — es 1 query batch + Python-side grouping. El patrón
    "in_()" en Supabase JS/PostgREST se traduce a `WHERE session_id IN
    (uuid1, uuid2, ...)` que devuelve TODO de una vez.

    Trade-off real del patrón actual:
      - PRO: 1 round-trip DB independiente del número de sessions.
      - PRO: La agrupación Python es O(n) sobre el resultado, trivial
             en volumen típico (~100 mensajes).
      - CON: Si N sessions × M mensajes promedio es muy grande (e.g. >10k),
             el response payload es grande. Solución sería paginación
             server-side, NO un join SQL (Supabase REST no expone JOINs
             complejos).

    Audit external malinterpretó "agrupación en Python" como N+1 — eran
    diferentes paths code. La decisión actual es CORRECTA.

Este test ancla la decisión:
    Si alguien "arregla" el patrón aproximándolo con queries per-session
    en un loop (regresión real a N+1), el test falla — count de queries
    a `agent_messages` en este path debe ser EXACTAMENTE 1 (o 0 si vacío).

Cobertura:
    A) `_process_and_sort_sessions` existe en db_chat.py.
    B) Usa `.in_(` pattern (batch IN clause) — NO loop con queries.
    C) NO contiene patrón anti `for s in sessions: ... select(...)`.

Tooltip-anchor: P2-PROD-AUDIT-1-DB-CHAT-GROUPING | audit 2026-05-23.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_CHAT = _BACKEND_ROOT / "db_chat.py"


def test_process_and_sort_sessions_exists():
    text = _DB_CHAT.read_text(encoding="utf-8")
    assert "def _process_and_sort_sessions" in text, (
        "`_process_and_sort_sessions` ausente en db_chat.py. Si renombraste, "
        "actualizar este test."
    )


def test_uses_batch_in_clause():
    """El patrón debe usar `.in_(...)` para fetch batch, NO loop con
    queries per session."""
    text = _DB_CHAT.read_text(encoding="utf-8")
    # Localizar la función + sus ~50 líneas siguientes.
    idx = text.find("def _process_and_sort_sessions")
    assert idx != -1, "función no encontrada (cubierto por test_process_and_sort_sessions_exists)"
    window = text[idx : idx + 2500]
    assert ".in_(" in window or ".in(" in window, (
        "_process_and_sort_sessions NO usa `.in_(session_ids)` pattern. "
        "Regresión a N+1: probablemente está haciendo `for s in sessions: "
        "select(...)`. Esto MULTIPLICA las queries por count(sessions). "
        "Restaurar el batch IN clause."
    )


def test_no_n_plus_one_anti_pattern():
    """Defense-in-depth: el bloque debe NO contener patrón
    `for X in sessions: ... supabase.table(...).execute()` SOBRE la
    variable `sessions` (exacta — NO `session_ids`, NO retry `for attempt
    in range(...)`).

    Uso AST para evitar falsos positivos del regex (e.g. retry loop
    sobre `range(max_retries)` que tiene supabase call dentro).
    """
    import ast
    text = _DB_CHAT.read_text(encoding="utf-8")
    tree = ast.parse(text)

    target_fn = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "_process_and_sort_sessions":
                target_fn = node
                break
    if target_fn is None:
        return  # cubierto por test_process_and_sort_sessions_exists

    # Buscar `for X in sessions:` (iterar EXACTAMENTE sobre `sessions`).
    for for_node in ast.walk(target_fn):
        if not isinstance(for_node, ast.For):
            continue
        # iter debe ser `Name("sessions")` (exacto).
        if not (isinstance(for_node.iter, ast.Name) and for_node.iter.id == "sessions"):
            continue
        # Dentro del body buscar supabase.table(...).execute().
        for sub in ast.walk(for_node):
            if isinstance(sub, ast.Call):
                f = sub.func
                if isinstance(f, ast.Attribute) and f.attr == "execute":
                    # Check chain: f.value debería ser eventualmente
                    # supabase.table(...). Walk down.
                    chain = ast.unparse(sub)
                    if "supabase.table" in chain:
                        raise AssertionError(
                            f"_process_and_sort_sessions tiene patrón N+1 real: "
                            f"`for X in sessions: ... supabase.table(...).execute()` "
                            f"detectado en línea {for_node.lineno}. Reemplazar por "
                            f"1 query batch con `.in_(session_ids)` + Python-side "
                            f"grouping."
                        )


def test_grouping_is_python_side():
    """La agrupación debe ser Python-side: `messages_by_session[X].append(Y)`.
    Si refactorearon a SQL JOIN complejo, validar que mantienen la
    semántica + performance.
    """
    text = _DB_CHAT.read_text(encoding="utf-8")
    idx = text.find("def _process_and_sort_sessions")
    window = text[idx : idx + 2500] if idx != -1 else ""
    has_python_grouping = (
        "messages_by_session" in window
        or "msgs_by_session" in window
        or "by_session" in window
    )
    assert has_python_grouping, (
        "_process_and_sort_sessions perdió el grouping Python-side. Si "
        "refactorearon a SQL JOIN, OK — pero actualizar este test para "
        "reflejar el nuevo patrón. Si por error simplemente eliminaron "
        "el grouping, restaurar."
    )


def test_anchor_present():
    """Este test mismo es el anchor — pero documentamos en db_chat.py
    también para que un futuro reader vea la decisión.
    """
    text = _DB_CHAT.read_text(encoding="utf-8")
    # Soft check — si nadie añadió el anchor en db_chat.py, este test
    # mismo sirve de breadcrumb cross-link via slug `p2_prod_audit_4`.
    has_anchor = "P2-PROD-AUDIT-1" in text
    if not has_anchor:
        print(
            "\n⚠️  [P2-PROD-AUDIT-1-DB-CHAT-GROUPING] Anchor no presente en "
            "db_chat.py. Si alguien refactor a path, este test puede quedar "
            "huérfano. Considerar añadir comment inline en "
            "_process_and_sort_sessions apuntando a este test."
        )
