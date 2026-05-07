"""[P1-1] Pantry validation post-LLM retry cap + escalación a pause.

Antes del fix:
    1. `_PANTRY_MAX_RETRIES = 2` estaba hardcoded como local en cron_tasks.py
       (línea ~12697). La docstring de `CHUNK_PANTRY_QUANTITY_MODE` en
       constants.py (~líneas 230-237) ya referenciaba `CHUNK_PANTRY_MAX_RETRIES`
       como si existiera, pero la constante NO estaba definida → inconsistencia.
    2. En el path de **existence violations** (LLM alucina ingredientes que no
       están en la nevera), tras agotar reintentos se marcaba el chunk como
       'failed' (`_cas_update_chunk_status(task_id, _pickup_attempts, "failed")`).
       Eso disparaba `_recover_failed_chunks_for_long_plans` que re-encolaba
       el chunk con backoff exponencial (2→4→8→16→32 min). Cada reintento
       corría el LLM con la MISMA nevera → mismas alucinaciones → tokens
       quemados sin convergencia hasta CHUNK_MAX_RECOVERY_ATTEMPTS.
    3. El path de **quantity violations** strict ya usaba la estrategia
       correcta (pausa en pending_user_action), pero existence no la replicaba
       — asimetría dañina porque existence es el caso MÁS severo.

Después del fix:
    1. `CHUNK_PANTRY_MAX_RETRIES` definida en constants.py (default 2, override
       por env var).
    2. cron_tasks.py importa y usa la constante en lugar del hardcoded local.
    3. Existence violations exhausted → pausa en pending_user_action con
       reason='pantry_violation_after_retries', simétrico al path de quantity
       violations strict (~líneas 12947-12958). El recovery cron
       `_recover_pantry_paused_chunks` sólo lo reanuda cuando la nevera del
       usuario cambia — i.e., cuando hay esperanza de que el LLM converja.
"""
import inspect
import os

import pytest


def test_constant_is_defined_in_constants_module():
    """`CHUNK_PANTRY_MAX_RETRIES` está exportado desde constants.py.
    Antes vivía como hardcoded local en cron_tasks._chunk_worker."""
    from constants import CHUNK_PANTRY_MAX_RETRIES
    assert isinstance(CHUNK_PANTRY_MAX_RETRIES, int)
    assert CHUNK_PANTRY_MAX_RETRIES >= 1, (
        "CHUNK_PANTRY_MAX_RETRIES debe ser >= 1; el clamp en constants.py "
        "lo enforce vía max(1, int(...))."
    )


def test_constant_default_value_matches_legacy_hardcoded():
    """Default = 2 → range(2+1) = 3 attempts (0, 1, 2 inclusive). Mantiene
    paridad con el comportamiento previo donde `_PANTRY_MAX_RETRIES = 2`
    estaba hardcoded — sin esto, planes en producción cambiarían su número
    de retries silentemente al deployar P1-1."""
    # Reset env override si estuviera seteado por otra suite.
    prev = os.environ.pop("CHUNK_PANTRY_MAX_RETRIES", None)
    try:
        # Re-import limpio para que el default aplique.
        import importlib
        import constants
        importlib.reload(constants)
        assert constants.CHUNK_PANTRY_MAX_RETRIES == 2
    finally:
        if prev is not None:
            os.environ["CHUNK_PANTRY_MAX_RETRIES"] = prev
            import importlib
            import constants
            importlib.reload(constants)


def test_constant_clamps_invalid_env_value():
    """Si alguien setea CHUNK_PANTRY_MAX_RETRIES=0 (o negativo) por accidente,
    el clamp eleva a 1 — sin esto, range(0+1)=range(1) sería 1 sólo intento
    sin retries reales. range(-1+1) = range(0) sería ZERO intentos → ningún
    chunk generaría nunca."""
    os.environ["CHUNK_PANTRY_MAX_RETRIES"] = "0"
    try:
        import importlib
        import constants
        importlib.reload(constants)
        assert constants.CHUNK_PANTRY_MAX_RETRIES == 1, (
            "Clamp a 1 debe rescatar valores <1 para que siempre haya al "
            "menos 1 retry real."
        )
    finally:
        os.environ.pop("CHUNK_PANTRY_MAX_RETRIES", None)
        import importlib
        import constants
        importlib.reload(constants)


def test_cron_tasks_imports_constant():
    """cron_tasks.py debe importar `CHUNK_PANTRY_MAX_RETRIES` desde constants.
    Si el import desaparece, el local `_PANTRY_MAX_RETRIES = CHUNK_PANTRY_MAX_RETRIES`
    rompería con NameError silenciado (en el path try/except del worker), y
    el worker caería al except path con 'failed' — exactamente el bug
    pre-fix."""
    import cron_tasks
    assert hasattr(cron_tasks, "CHUNK_PANTRY_MAX_RETRIES"), (
        "cron_tasks no tiene CHUNK_PANTRY_MAX_RETRIES como atributo de módulo "
        "(¿se removió del bloque `from constants import ...`?). Sin esto, el "
        "local _PANTRY_MAX_RETRIES = CHUNK_PANTRY_MAX_RETRIES rompería."
    )


def test_hardcoded_local_was_removed():
    """Regression guard: el hardcoded `_PANTRY_MAX_RETRIES = 2` debe haber
    sido reemplazado por `_PANTRY_MAX_RETRIES = CHUNK_PANTRY_MAX_RETRIES`
    en el worker. Si por refactor accidental volviera el hardcoded, el env
    var override dejaría de funcionar y la constante quedaría documentada
    pero unused."""
    import cron_tasks
    src = inspect.getsource(cron_tasks)
    assert "_PANTRY_MAX_RETRIES = 2" not in src, (
        "El hardcoded `_PANTRY_MAX_RETRIES = 2` sigue presente. Debe ser "
        "`_PANTRY_MAX_RETRIES = CHUNK_PANTRY_MAX_RETRIES` para honrar la "
        "constante / env var override."
    )
    assert "_PANTRY_MAX_RETRIES = CHUNK_PANTRY_MAX_RETRIES" in src, (
        "El binding `_PANTRY_MAX_RETRIES = CHUNK_PANTRY_MAX_RETRIES` no "
        "está presente. El worker no honraría el override por env var."
    )


def test_existence_violation_exhaustion_pauses_instead_of_failing():
    """Regression guard del comportamiento P1-1: cuando existence violations
    agotan los retries, el path debe llamar a `_pause_chunk_for_pantry_refresh`
    con `reason='pantry_violation_after_retries'`, NO a `_cas_update_chunk_status(
    task_id, _pickup_attempts, 'failed')`.

    Sin este guard, alguien podría 'simplificar' volviendo al fail-and-recover
    pattern, reabriendo el bucle de quemado de tokens sobre la misma nevera.
    """
    import cron_tasks
    src = inspect.getsource(cron_tasks)
    # El reason canónico debe aparecer en el código del worker.
    assert 'reason="pantry_violation_after_retries"' in src, (
        "El reason canónico 'pantry_violation_after_retries' no aparece en "
        "cron_tasks. La pausa P1-1 fue removida o renombrada — eso reabriría "
        "el bucle de retries fallidos quemando tokens contra la misma nevera."
    )


def test_pause_is_symmetric_with_quantity_violation_strict_path():
    """El path de existence violations exhausted debe usar el MISMO helper
    `_pause_chunk_for_pantry_refresh` que el path de quantity violations
    strict (~líneas 12947-12958), demostrando la simetría del fix.
    Si un dev cambia uno y olvida el otro, tendremos asimetría regresiva.
    """
    import cron_tasks
    src = inspect.getsource(cron_tasks)
    # Quantity strict ya usa este reason — confirma que el patrón es
    # establecido en el código.
    assert 'reason="quantity_unfeasible"' in src, (
        "El reason 'quantity_unfeasible' (path de quantity strict) se removió. "
        "Si cambia, también debe revisar el path de existence violations para "
        "mantener simetría."
    )
    # Y el path de existence usa una variante consistente (ambos pausan).
    assert '_pause_chunk_for_pantry_refresh' in src
    assert src.count("_pause_chunk_for_pantry_refresh(") >= 2, (
        "Esperaba al menos 2 invocaciones a _pause_chunk_for_pantry_refresh "
        "(quantity strict + existence retries exhausted, mínimo). Si hay <2, "
        "uno de los paths perdió la pausa."
    )


def test_constant_referenced_in_constants_docstring_now_resolves():
    """La docstring de CHUNK_PANTRY_QUANTITY_MODE en constants.py:230-237
    referenciaba `CHUNK_PANTRY_MAX_RETRIES` como si existiera. Antes del fix
    era una promesa documentada pero no implementada. Ahora la constante
    existe y la docstring queda consistente con la implementación.
    """
    import constants
    src = inspect.getsource(constants)
    assert "CHUNK_PANTRY_MAX_RETRIES" in src
    # La constante debe ser accesible (no solo mencionada en comentarios).
    assert hasattr(constants, "CHUNK_PANTRY_MAX_RETRIES")


def test_env_var_override_works():
    """`CHUNK_PANTRY_MAX_RETRIES` puede ser sobrescrita por env var, igual
    que las demás constantes de chunk en constants.py."""
    os.environ["CHUNK_PANTRY_MAX_RETRIES"] = "5"
    try:
        import importlib
        import constants
        importlib.reload(constants)
        assert constants.CHUNK_PANTRY_MAX_RETRIES == 5
    finally:
        os.environ.pop("CHUNK_PANTRY_MAX_RETRIES", None)
        import importlib
        import constants
        importlib.reload(constants)
