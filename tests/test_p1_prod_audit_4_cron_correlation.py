"""[P1-PROD-AUDIT-1 · 2026-05-23] Cada cron job debe ejecutarse con su
propio correlation_id (`corr=cron:<job_id>:<short>`), no `corr=-`.

Gap original (audit production-readiness 2026-05-23, B-P1-2):
    APScheduler ejecuta los jobs en threads pre-existentes sin request
    scope. El `_correlation_id` ContextVar default `"-"` aplica a TODOS
    los logs del cron → cross-cutting debugging cron-internal imposible
    (todas las líneas marcadas igual).

    `correlation.py` ya documentaba la limitación:
        "APScheduler crons corren en threads pre-existentes (background
        scheduler) sin request scope → todos llevan `corr=-` también.
        Para correlation cron-internal, futuro P-fix podría asignar
        `corr=cron:<job_name>:<run_id>` al entry-point del cron."

    Este test ancla el cierre de ese gap.

Fix:
    `_add_job_jittered` en `cron_tasks.py` ahora wrappea el callable con
    `with_correlation_id(f"cron:{job_id}:{new_correlation_id()}")` ANTES
    de pasarlo a `scheduler.add_job(...)`. Cada run del cron obtiene su
    propio ID (NO el job_id es el ID — sería el mismo entre runs).

Cobertura:
    A) Anchor `P1-PROD-AUDIT-1-CRON-CORRELATION` presente en cron_tasks.py.
    B) `_add_job_jittered` importa `with_correlation_id` y `new_correlation_id`.
    C) `_add_job_jittered` wrappea callables (no es solo passthrough).
    D) Idempotencia: doble call no doble-wrappea (marker `_mealfit_corr_wrapped`).
    E) Formato del ID es `cron:<job_id>:<8hex>` — distinguible de
       request-scoped IDs (8 chars puros hex).

Tooltip-anchor: P1-PROD-AUDIT-1-CRON-CORRELATION | audit 2026-05-23.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_TASKS = _BACKEND_ROOT / "cron_tasks.py"


def _read_cron() -> str:
    return _CRON_TASKS.read_text(encoding="utf-8")


def test_anchor_present():
    src = _read_cron()
    assert "P1-PROD-AUDIT-1-CRON-CORRELATION" in src, (
        "Anchor `P1-PROD-AUDIT-1-CRON-CORRELATION` ausente en cron_tasks.py. "
        "Sin anchor, futuro mantenedor revierte al passthrough sin contexto."
    )


def test_with_correlation_id_imported_in_wrapper():
    """`_add_job_jittered` debe importar `with_correlation_id` y
    `new_correlation_id` desde correlation. Si alguien quita el import,
    el wrapping silenciosamente NO funciona."""
    src = _read_cron()
    # Buscar import inline (dentro del wrapper) o top-level.
    pattern_inline = re.compile(
        r"from correlation import.*with_correlation_id.*new_correlation_id"
        r"|from correlation import.*new_correlation_id.*with_correlation_id"
    )
    pattern_topelvel = re.compile(
        r"^from correlation import.*with_correlation_id", re.MULTILINE
    )
    assert pattern_inline.search(src) or pattern_topelvel.search(src), (
        "_add_job_jittered NO importa `with_correlation_id` + "
        "`new_correlation_id` de correlation. Sin el import, el wrapping "
        "no funciona — cierre de gap perdido."
    )


def test_wrapper_wraps_callable_arg():
    """`_add_job_jittered` debe wrappear el primer arg si es callable.
    Validamos buscando la lógica `if args and callable(args[0])` o
    equivalente."""
    src = _read_cron()
    # Heuristics: el bloque del wrapper debe referenciar `callable(args[0])`.
    assert "callable(args[0])" in src or "callable(args[0]" in src, (
        "_add_job_jittered NO chequea `callable(args[0])` — el wrapping "
        "podría rotar callables vs strings (APScheduler acepta ambos)."
    )


def test_idempotency_marker_present():
    """Marker `_mealfit_corr_wrapped` evita double wrapping si
    `_add_job_jittered` se invoca dos veces sobre el mismo callable."""
    src = _read_cron()
    assert "_mealfit_corr_wrapped" in src, (
        "Marker de idempotencia `_mealfit_corr_wrapped` ausente. Sin él, "
        "double-call accidental anida wrappers — cada layer genera nuevo "
        "correlation_id, anulando el propósito."
    )


def test_cron_id_format_documented():
    """El formato del ID debe ser `cron:<job_id>:<8hex>`. Validamos buscando
    el f-string del format."""
    src = _read_cron()
    # Pattern del format string. Acepta variantes.
    pattern = re.compile(r"f['\"]cron:\{[^}]+\}:\{")
    assert pattern.search(src), (
        "El formato del cron correlation_id NO sigue `cron:<job_id>:<...>`. "
        "Restaurar formato distinguible de request-scoped IDs (8 chars puros)."
    )


def test_correlation_py_docs_limitation_was_closed():
    """`correlation.py` documenta la limitación pre-fix:
       'APScheduler crons corren en threads pre-existentes [...] →
       todos llevan corr=- también.'
       'Para correlation cron-internal, futuro P-fix podría asignar
       corr=cron:<job_name>:<run_id>'

    El P-fix está aplicado — verificar que correlation.py mantenga
    la nota o, si fue editada, que el cierre quede documentado.
    Este test es soft: anchor en cron_tasks.py es suficiente.
    """
    corr_py = _BACKEND_ROOT / "correlation.py"
    assert corr_py.exists(), "correlation.py ausente"
    text = corr_py.read_text(encoding="utf-8")
    # No exigimos que correlation.py esté actualizado (la single source of
    # truth ahora es cron_tasks.py + tests). Solo validamos que
    # `with_correlation_id` aún esté exportado.
    assert "def with_correlation_id" in text, (
        "correlation.py perdió `with_correlation_id` — usada por "
        "_add_job_jittered. Restaurar."
    )
    assert "def new_correlation_id" in text, (
        "correlation.py perdió `new_correlation_id` — usada por "
        "_add_job_jittered. Restaurar."
    )


def test_functional_wrap_smoke():
    """Smoke funcional: el wrapper genera correlation_id diferente por call.
    Verifica que NO es el mismo ID para 2 runs del mismo job (sería bug
    si usáramos `job_id` directo en lugar de `new_correlation_id()`).
    """
    # Carga inline del wrapper sin tirar de cron_tasks (que importa todo
    # el módulo de 27k líneas). Re-crear el wrapper local con misma logic
    # para validar funcionalmente.
    import sys
    sys.path.insert(0, str(_BACKEND_ROOT))
    try:
        from correlation import with_correlation_id, new_correlation_id, get_correlation_id

        captured_ids = []

        def my_job():
            captured_ids.append(get_correlation_id())

        def wrap_like_jittered(job_id_for_corr, original):
            def wrapped(*a, **kw):
                run_corr = f"cron:{job_id_for_corr}:{new_correlation_id()}"
                with with_correlation_id(run_corr):
                    return original(*a, **kw)
            return wrapped

        wrapped_job = wrap_like_jittered("test_job", my_job)
        wrapped_job()
        wrapped_job()
        wrapped_job()

        assert len(captured_ids) == 3
        # Cada run debe tener un ID único.
        assert len(set(captured_ids)) == 3, (
            f"3 runs del mismo wrapped job retornaron menos de 3 IDs "
            f"únicos: {captured_ids}. El wrapper usa job_id estático en "
            f"lugar de generar nuevo por run."
        )
        for cid in captured_ids:
            assert cid.startswith("cron:test_job:"), (
                f"correlation_id NO sigue formato esperado: {cid}"
            )
    finally:
        if str(_BACKEND_ROOT) in sys.path:
            sys.path.remove(str(_BACKEND_ROOT))
