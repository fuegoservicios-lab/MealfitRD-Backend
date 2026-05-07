"""[P1-22] Tests para que `_lifetime_lessons_summary` excluya las lecciones
originadas en chunks dead-lettered al recomputarse en `summarize_and_prune` /
worker chunked.

Bug original (audit P1-22):
  `_filter_lessons_excluding_dead_lettered` ya filtra `_last_chunk_learning`
  y `_recent_chunk_lessons` en read-path: las lecciones de chunks marcados
  `recovery_exhausted` por `_escalate_unrecoverable_chunk` no llegan al LLM
  del chunk siguiente.

  Pero `_lifetime_lessons_summary` se recomputaba desde
  `_lifetime_lessons_history` SIN ese filtrado. Resultado: contadores
  acumulados, sets de top_rejection_hits / top_repeated_bases /
  top_repeated_meal_names y `permanent_meal_blocklist` incorporaban
  contribuciones de chunks que nunca llegaron a shippearse.

  Consecuencias en producción:
    - permanent_meal_blocklist crecía con platos que el usuario nunca
      vio (chunk se dead-letteró antes de llegar al PDF / app).
    - `_lifetime_proxy_ratio` se sesgaba por proxy_count de chunks
      muertos, disparando alertas de telemetría falsamente.
    - El ranking por decay temporal en _meal_weights favorecía lecciones
      recientes pero fantasma sobre lecciones más viejas pero reales.

Fix:
  1. Helper `_filter_lifetime_history_excluding_dead_lettered(history,
     prior_plan_data)` lee `_recovery_exhausted_chunks` y filtra entradas
     cuyo `chunk` esté en la lista. Devuelve `(filtered_history, dead_weeks)`.
  2. El path normal de `summarize_and_prune` (merge ~línea 18655 de
     cron_tasks.py) llama al helper antes de la agregación; los totals,
     sets, decay weights y provenance ratio se computan sobre el filtrado.
  3. El path backfill (chunk_already_merged ~línea 18250) también filtra
     antes de su agregación simétrica.
  4. `_lifetime_lessons_history` en BD se conserva sin filtrar como audit
     trail (permite reversión de dead-letter por intervención manual sin
     pérdida de datos).

Cobertura:
  - test_helper_exists_and_signature
  - test_filter_returns_input_when_no_dead_lettered
  - test_filter_excludes_lessons_from_dead_lettered_chunks
  - test_filter_preserves_lessons_with_no_chunk_field
  - test_filter_handles_non_dict_entries
  - test_filter_returns_dead_weeks_sorted
  - test_filter_handles_malformed_recovery_exhausted_list
  - test_filter_called_in_normal_merge_path_source
  - test_filter_called_in_backfill_path_source
  - test_lifetime_summary_uses_filtered_history_in_aggregation
  - test_documentation_p1_22_present
"""
import inspect
import re

import pytest

import cron_tasks


_SRC = inspect.getsource(cron_tasks)


def _strip_comments(src: str) -> str:
    return "\n".join(
        ln for ln in src.splitlines() if not ln.strip().startswith("#")
    )


# ---------------------------------------------------------------------------
# 1. Helper: signature y contrato.
# ---------------------------------------------------------------------------
def test_helper_exists_and_signature():
    """`_filter_lifetime_history_excluding_dead_lettered(history,
    prior_plan_data)` debe estar exportado a nivel módulo."""
    assert hasattr(cron_tasks, "_filter_lifetime_history_excluding_dead_lettered")
    sig = inspect.signature(
        cron_tasks._filter_lifetime_history_excluding_dead_lettered
    )
    params = list(sig.parameters.keys())
    assert params == ["history", "prior_plan_data"], (
        f"P1-22: signature inesperada: {params}"
    )


# ---------------------------------------------------------------------------
# 2. Comportamiento: sin dead-letter, retorna la entrada sin cambios.
# ---------------------------------------------------------------------------
def test_filter_returns_input_when_no_dead_lettered():
    """Si `_recovery_exhausted_chunks` está vacío o ausente, no filtra nada."""
    history = [
        {"chunk": 1, "rejection_violations": 2},
        {"chunk": 2, "rejection_violations": 3},
    ]
    out, dead = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        history, {}
    )
    assert out == history
    assert dead == []

    out2, dead2 = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        history, {"_recovery_exhausted_chunks": []}
    )
    assert out2 == history
    assert dead2 == []


def test_filter_returns_input_when_no_dict_plan_data():
    """`prior_plan_data` no-dict (None, list, str) → retorna input sin cambios."""
    history = [{"chunk": 1}]
    out, dead = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        history, None
    )
    assert out == history
    assert dead == []
    out2, dead2 = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        history, []
    )
    assert out2 == history
    assert dead2 == []


def test_filter_returns_input_when_history_not_list():
    """`history` no-lista → retorna input sin cambios."""
    out, dead = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        None, {"_recovery_exhausted_chunks": [{"week_number": 2}]}
    )
    assert out is None
    assert dead == []


# ---------------------------------------------------------------------------
# 3. Filtrado correcto.
# ---------------------------------------------------------------------------
def test_filter_excludes_lessons_from_dead_lettered_chunks():
    """Lecciones cuyo `chunk` figura en `_recovery_exhausted_chunks`
    [{"week_number": N}] deben quedar fuera del output."""
    history = [
        {"chunk": 1, "rejection_violations": 2},
        {"chunk": 2, "rejection_violations": 5},  # dead-lettered
        {"chunk": 3, "rejection_violations": 1},
        {"chunk": 4, "rejection_violations": 7},  # dead-lettered
    ]
    plan_data = {
        "_recovery_exhausted_chunks": [
            {"week_number": 2, "reason": "missing_anchor"},
            {"week_number": 4, "reason": "max_attempts"},
        ]
    }
    out, dead = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        history, plan_data
    )
    chunks_kept = sorted(l["chunk"] for l in out)
    assert chunks_kept == [1, 3], (
        f"P1-22: chunks dead-lettered (2,4) no fueron filtrados: kept={chunks_kept}"
    )
    assert dead == [2, 4]


def test_filter_preserves_lessons_with_no_chunk_field():
    """Una lección sin campo `chunk` (legacy / corrupted) NO debe
    descartarse — el filtro solo aplica a entradas con chunk identificable."""
    history = [
        {"rejection_violations": 1},  # sin chunk
        {"chunk": 2, "rejection_violations": 5},  # dead-lettered
        {"chunk": "garbage", "rejection_violations": 1},  # chunk no-numérico
    ]
    plan_data = {
        "_recovery_exhausted_chunks": [{"week_number": 2}]
    }
    out, dead = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        history, plan_data
    )
    # Solo la entrada chunk=2 se filtra. Las otras dos quedan.
    assert len(out) == 2
    chunks = [l.get("chunk") for l in out]
    assert "garbage" in chunks
    assert dead == [2]


def test_filter_handles_non_dict_entries():
    """Items no-dict (None, str, int) en history pasan tal cual al output
    sin crashear (defensa contra corrupción JSON)."""
    history = [
        None,
        "weird",
        42,
        {"chunk": 5, "rejection_violations": 3},  # dead-lettered
        {"chunk": 6, "rejection_violations": 1},
    ]
    plan_data = {"_recovery_exhausted_chunks": [{"week_number": 5}]}
    out, dead = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        history, plan_data
    )
    # Los items no-dict sobreviven; solo el dict con chunk=5 se filtra.
    assert None in out
    assert "weird" in out
    assert 42 in out
    assert {"chunk": 6, "rejection_violations": 1} in out
    assert {"chunk": 5, "rejection_violations": 3} not in out
    assert dead == [5]


def test_filter_returns_dead_weeks_sorted():
    """El segundo elemento del retorno es la lista de dead_weeks ordenada
    ascendente — útil para logs y telemetría reproducibles."""
    plan_data = {
        "_recovery_exhausted_chunks": [
            {"week_number": 7},
            {"week_number": 1},
            {"week_number": 4},
        ]
    }
    _, dead = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        [{"chunk": 99}], plan_data
    )
    assert dead == [1, 4, 7]


def test_filter_handles_malformed_recovery_exhausted_list():
    """`_recovery_exhausted_chunks` con tipos inesperados (no-lista, items
    no-dict, week_number no-numérico) → fail-open: no filtra nada."""
    history = [{"chunk": 1}, {"chunk": 2}]

    # _recovery_exhausted_chunks no es lista.
    out, dead = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        history, {"_recovery_exhausted_chunks": "not a list"}
    )
    assert out == history
    assert dead == []

    # week_number no-numérico → no se añade a dead_weeks.
    out2, dead2 = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        history, {"_recovery_exhausted_chunks": [{"week_number": "abc"}]}
    )
    assert out2 == history
    assert dead2 == []

    # Items no-dict en la lista → ignorados silenciosamente.
    out3, dead3 = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        history,
        {"_recovery_exhausted_chunks": ["bad", 42, {"week_number": 1}]},
    )
    chunks = sorted(l["chunk"] for l in out3)
    assert chunks == [2]
    assert dead3 == [1]


# ---------------------------------------------------------------------------
# 4. Defensa estructural: el filtro se invoca en ambos paths.
# ---------------------------------------------------------------------------
def test_filter_called_in_normal_merge_path_source():
    """El path normal de merge debe invocar el helper antes de la
    agregación. Defensa contra reintroducción del bug por refactor.

    Usamos `_history_for_summary` como landmark — es la variable que
    captura el output del filter en el path normal y es única (el path
    backfill usa `_bf_history_for_summary`)."""
    code = _strip_comments(_SRC)
    # `_history_for_summary` solo aparece en el path normal del worker.
    # Verificamos que (a) existe la asignación y (b) el filter se invocó
    # en una ventana cercana antes de su uso.
    assignments = [
        m.start() for m in re.finditer(
            r"_history_for_summary\s*,\s*_p122_dead_weeks\s*=", code
        )
    ]
    assert assignments, (
        "P1-22: no se encontró la asignación "
        "`_history_for_summary, _p122_dead_weeks = ...` en el path normal."
    )
    # Verificar que la asignación esté seguida (en pocos chars) de la
    # invocación al filter helper.
    for idx in assignments:
        window = code[idx : idx + 500]
        if "_filter_lifetime_history_excluding_dead_lettered(" in window:
            return
    pytest.fail(
        "P1-22: la asignación `_history_for_summary` no proviene de "
        "`_filter_lifetime_history_excluding_dead_lettered(...)`."
    )


def test_filter_called_in_backfill_path_source():
    """El path backfill (chunk_already_merged) también debe invocar el
    helper. Sin esto, un chunk re-mergeado tras crash inflaba lifetime."""
    code = _strip_comments(_SRC)
    # El backfill está dentro del bloque chunk_already_merged.
    block_idx = code.find("if chunk_already_merged:")
    assert block_idx > -1
    # Tomamos hasta el siguiente `else: # Merge normal` o 60K chars.
    rest = code[block_idx:]
    m = re.search(r"\n( {24}else:.*)", rest)
    block = rest[: m.start()] if m else rest[:60_000]
    assert "_filter_lifetime_history_excluding_dead_lettered(" in block, (
        "P1-22: el path backfill chunk_already_merged debe invocar el "
        "filter helper antes de recomputar _lifetime_lessons_summary."
    )


def test_filter_invocations_appear_at_least_twice():
    """Defensa cuantitativa: el helper se debe invocar al menos 2 veces en
    cron_tasks.py (path normal + path backfill). Si baja a 1, alguien
    eliminó accidentalmente una de las invocaciones."""
    code = _strip_comments(_SRC)
    invocations = code.count("_filter_lifetime_history_excluding_dead_lettered(")
    assert invocations >= 2, (
        f"P1-22: esperaba >= 2 invocaciones del filter helper, encontré "
        f"{invocations}. Path normal y backfill deben filtrar por separado."
    )


# ---------------------------------------------------------------------------
# 5. Comportamiento end-to-end del filtro (input → contadores correctos).
# ---------------------------------------------------------------------------
def test_lifetime_summary_uses_filtered_history_in_aggregation():
    """Verificación funcional: si pasamos un history mezclado (chunks vivos
    + dead-lettered) al helper y luego sumamos `rejection_violations` /
    `allergy_violations` sobre el output, el total debe reflejar SOLO los
    chunks vivos. Esto valida el contrato que ambos paths del worker
    (normal y backfill) usan para construir `total_rejection_violations`
    y `total_allergy_violations`."""
    history = [
        {"chunk": 1, "rejection_violations": 2, "allergy_violations": 1},
        {"chunk": 2, "rejection_violations": 100, "allergy_violations": 100},  # dead
        {"chunk": 3, "rejection_violations": 5, "allergy_violations": 0},
    ]
    plan_data = {"_recovery_exhausted_chunks": [{"week_number": 2}]}
    filtered, _ = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        history, plan_data
    )
    total_rej = sum(int(l.get("rejection_violations") or 0) for l in filtered)
    total_alg = sum(int(l.get("allergy_violations") or 0) for l in filtered)
    # Sin filtro → 107 / 101. Con filtro → 7 / 1.
    assert total_rej == 7, f"P1-22: filtro no aplicó en agregación: rej={total_rej}"
    assert total_alg == 1, f"P1-22: filtro no aplicó en agregación: alg={total_alg}"


def test_lifetime_summary_blocklist_excludes_dead_lettered_meal_names():
    """Si un plato `repeated_meal_names` aparece SOLO en chunks dead-lettered,
    no debe contribuir al `permanent_meal_blocklist`. Reproduce la lógica del
    path normal: un meal con count >= 2 chunks vivos va al blocklist; un meal
    con count en chunks muertos NO debe llegar."""
    history = [
        {
            "chunk": 1,
            "repeated_meal_names": ["pollo_curry"],
            "rejection_violations": 0,
            "allergy_violations": 0,
        },
        {
            "chunk": 2,  # dead
            "repeated_meal_names": ["pollo_curry", "tofu_thai"],
            "rejection_violations": 0,
            "allergy_violations": 0,
        },
        {
            "chunk": 3,  # dead
            "repeated_meal_names": ["tofu_thai"],
            "rejection_violations": 0,
            "allergy_violations": 0,
        },
    ]
    plan_data = {
        "_recovery_exhausted_chunks": [
            {"week_number": 2}, {"week_number": 3}
        ]
    }
    filtered, dead = cron_tasks._filter_lifetime_history_excluding_dead_lettered(
        history, plan_data
    )
    assert dead == [2, 3]
    # Replicamos heurística: meal_name → set(chunk).
    counts: dict = {}
    for l in filtered:
        ch = l.get("chunk")
        for m in (l.get("repeated_meal_names") or []):
            counts.setdefault(m, set()).add(ch)
    blocklist = [m for m, chunks in counts.items() if len(chunks) >= 2]
    # tofu_thai aparecía SOLO en chunks 2 y 3 (ambos dead) → debería NO estar.
    # pollo_curry aparecía en chunk 1 y 2: tras filtrar, solo queda chunk 1
    # (1 chunk vivo) → NO blocklist.
    assert "tofu_thai" not in blocklist, (
        "P1-22: tofu_thai estaba SOLO en chunks dead — no debería ir al "
        "permanent_meal_blocklist."
    )
    assert "pollo_curry" not in blocklist, (
        "P1-22: pollo_curry queda en 1 chunk vivo tras filtro → no >=2."
    )


# ---------------------------------------------------------------------------
# 6. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_22_present():
    """Comentario `[P1-22]` debe documentar el rationale en el módulo."""
    assert "[P1-22]" in _SRC, (
        "P1-22: falta marker de documentación que explique por qué se "
        "filtra el lifetime history al recomputar el summary."
    )


def test_documentation_mentions_audit_trail_or_recovery_exhausted():
    """El comentario debe mencionar la naturaleza del filtro: leer
    `_recovery_exhausted_chunks` y/o conservar el history como audit
    trail. Ayuda a futuros lectores a no confundir el filtro con un
    delete del history."""
    idx = _SRC.find("[P1-22]")
    assert idx > -1
    window = _SRC[idx : idx + 2500]
    needles = ["recovery_exhausted", "audit trail", "audit", "dead-letter"]
    assert any(n in window.lower() for n in needles), (
        "P1-22: el comentario debe explicar el rationale "
        "(recovery_exhausted source / audit trail / dead-letter)."
    )
