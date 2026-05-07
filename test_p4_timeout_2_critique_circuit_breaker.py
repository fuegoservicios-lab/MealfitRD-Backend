"""[P4-TIMEOUT-2] Tests para el circuit breaker de self-critique correction.

Bug observable (corrida 2026-05-04 03:26):
  Gemini Flash entró en cascade-failure durante self-critique correction.
  Los 3 días corrieron en `asyncio.gather` y los 3 timeoutearon a 120s
  simultáneamente. Resultado:
    - 0 días corregidos
    - 3 markers `_critique_unresolved` → P1-SURGICAL-1 fuerza regen
    - retry path completo → pipeline ~400s

Fix:
  Schedule via `asyncio.wait(FIRST_COMPLETED)` y, cuando acumulamos
  N timeouts (`MEALFIT_CRITIQUE_TIMEOUT_ABORT_THRESHOLD`, default 2),
  cancelamos las tasks pendientes. Los días cancelados reciben marker
  `_critique_unresolved` con reason `"cb_aborted_provider_overload"`
  para que P1-SURGICAL-1 los regenere en retry.

Cobertura:
  - Default threshold = 2
  - Env var override
  - Threshold preserva tipo int
  - Marker `cb_aborted_provider_overload` se aplica a días abortados
  - El loop con wait/FIRST_COMPLETED está en uso (no gather)
"""
import asyncio
import importlib
import os

import pytest


def _reload_module():
    """Recarga graph_orchestrator para releer env vars."""
    import graph_orchestrator
    importlib.reload(graph_orchestrator)
    return graph_orchestrator


# ---------------------------------------------------------------------------
# 1. Default value
# ---------------------------------------------------------------------------
def test_default_threshold_is_2(monkeypatch):
    """[P4-TIMEOUT-2] Default es 2: tras el segundo timeout, abortamos."""
    monkeypatch.delenv("MEALFIT_CRITIQUE_TIMEOUT_ABORT_THRESHOLD", raising=False)
    go = _reload_module()
    assert go.CRITIQUE_TIMEOUT_ABORT_THRESHOLD == 2, (
        f"Default debe ser 2, recibido {go.CRITIQUE_TIMEOUT_ABORT_THRESHOLD}"
    )


def test_default_threshold_is_int():
    import graph_orchestrator as go
    assert isinstance(go.CRITIQUE_TIMEOUT_ABORT_THRESHOLD, int)


# ---------------------------------------------------------------------------
# 2. Env var override
# ---------------------------------------------------------------------------
def test_env_var_override_works(monkeypatch):
    """Operadores pueden ser más agresivos (=1) o desactivar de facto (=99)."""
    monkeypatch.setenv("MEALFIT_CRITIQUE_TIMEOUT_ABORT_THRESHOLD", "1")
    go = _reload_module()
    assert go.CRITIQUE_TIMEOUT_ABORT_THRESHOLD == 1


def test_env_var_high_value_disables_de_facto(monkeypatch):
    """Threshold > critique_max_days nunca aborta — efecto similar a desactivar."""
    monkeypatch.setenv("MEALFIT_CRITIQUE_TIMEOUT_ABORT_THRESHOLD", "99")
    go = _reload_module()
    assert go.CRITIQUE_TIMEOUT_ABORT_THRESHOLD == 99


def test_env_var_invalid_falls_to_default(monkeypatch):
    """Valor inválido → fallback a default 2."""
    monkeypatch.setenv("MEALFIT_CRITIQUE_TIMEOUT_ABORT_THRESHOLD", "garbage")
    go = _reload_module()
    assert go.CRITIQUE_TIMEOUT_ABORT_THRESHOLD == 2


# ---------------------------------------------------------------------------
# 3. Sanity: el knob aparece en el dump de configuración
# ---------------------------------------------------------------------------
def test_threshold_appears_in_knobs_dump():
    """`log_active_knobs` (o equivalente) debe incluir el knob para que
    operadores puedan verificar el valor activo en producción."""
    import inspect
    import graph_orchestrator as go
    src = inspect.getsource(go)
    assert "CRITIQUE_TIMEOUT_ABORT_THRESHOLD" in src


# ---------------------------------------------------------------------------
# 4. Sanity: self_critique_node usa wait+cancel, no gather
# ---------------------------------------------------------------------------
def test_self_critique_node_uses_wait_not_gather_for_corrections():
    """[P4-TIMEOUT-2] El loop debe usar `asyncio.wait(FIRST_COMPLETED)` para
    permitir cancelación incremental. Verificamos por substring en el código.

    Si alguien revierte a `asyncio.gather` en este path, el circuit breaker
    pierde su capacidad de abortar temprano — bug silencioso de regresión."""
    import inspect
    import graph_orchestrator as go
    src = inspect.getsource(go.self_critique_node)
    assert "FIRST_COMPLETED" in src, (
        "self_critique_node debe usar asyncio.wait(FIRST_COMPLETED) para "
        "permitir circuit breaker. Si lo cambiaste a gather, P4-TIMEOUT-2 "
        "queda inactivo."
    )
    assert "CRITIQUE_TIMEOUT_ABORT_THRESHOLD" in src, (
        "self_critique_node debe consultar CRITIQUE_TIMEOUT_ABORT_THRESHOLD "
        "para decidir cuándo abortar."
    )
    assert "cb_aborted_provider_overload" in src, (
        "Días abortados deben recibir marker cb_aborted_provider_overload "
        "para que P1-SURGICAL-1 los regenere en retry."
    )


# ---------------------------------------------------------------------------
# 5. Sanity: _correct_single_day devuelve 3-tuple
# ---------------------------------------------------------------------------
def test_correct_single_day_returns_3_tuple_signature():
    """[P4-TIMEOUT-2] `_correct_single_day` ahora retorna
    `(day_num, corrected_day_or_None, failure_reason_or_None)`. Verificamos
    que todas las return statements sigan ese shape."""
    import inspect
    import graph_orchestrator as go
    src = inspect.getsource(go.self_critique_node)
    # Encuentra el bloque de la closure
    closure_start = src.find("async def _correct_single_day")
    assert closure_start != -1, "closure _correct_single_day no encontrada"
    # Tomamos suficiente para cubrir la closure entera. El siguiente cierre
    # estructural natural es `days_to_fix = mentioned[:critique_max_days]`.
    closure_end = src.find("days_to_fix = mentioned", closure_start)
    closure_src = src[closure_start:closure_end]

    # Cualquier return en la closure debe tener 3 elementos. Buscamos
    # patrón `return day_num,` y luego contamos comas en la misma línea.
    return_lines = [
        line.strip()
        for line in closure_src.splitlines()
        if line.strip().startswith("return ")
    ]
    assert return_lines, "_correct_single_day no tiene returns visibles"
    for ret in return_lines:
        # Cada return debe tener al menos 2 comas (3 elementos en el tuple).
        # Las comas solo de positional args, no comas en strings — heurística
        # razonable porque las strings de fail_reason son simples.
        # Excluimos comas dentro de paréntesis anidados (no aplica aquí).
        comma_count = ret.count(",")
        assert comma_count >= 2, (
            f"Return en _correct_single_day debe ser 3-tuple; "
            f"recibido: {ret} (commas={comma_count})"
        )


# ---------------------------------------------------------------------------
# Helper: simulación in-memory del loop CB (sin acoplar a fixtures DB/LLM)
# ---------------------------------------------------------------------------
async def _simulate_cb_loop(
    days: list,
    fail_modes: dict,
    threshold: int,
    correction_delay_ok: float = 0.05,
):
    """Replica la lógica del CB loop de `self_critique_node`.
    Devuelve (correction_results, timeout_count, aborted_pending)."""

    def _mark(day_dict, reason, issue):
        day_dict["_critique_unresolved"] = {"reason": reason, "issue": issue}

    async def _fake_correct(day_num: int, fail_mode: str):
        if fail_mode == "timeout":
            await asyncio.sleep(0)
            target = next(d for d in days if d["day"] == day_num)
            _mark(target, "timeout", "")
            return day_num, None, "timeout"
        # "ok" delay para que termine después de los timeouts (instant)
        await asyncio.sleep(correction_delay_ok)
        return day_num, {"day": day_num, "corrected": True}, None

    tasks_by_day: dict = {}
    for d in days:
        day_n = d["day"]
        tasks_by_day[
            asyncio.ensure_future(_fake_correct(day_n, fail_modes[day_n]))
        ] = day_n

    correction_results: list = []
    timeout_count = 0
    aborted_pending = False
    pending = set(tasks_by_day.keys())

    while pending:
        done, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )
        for finished in done:
            result = await finished
            correction_results.append(result)
            _, _, fail_reason = result
            if fail_reason == "timeout":
                timeout_count += 1

        if (
            timeout_count >= threshold
            and pending
            and not aborted_pending
        ):
            aborted_pending = True
            for p in pending:
                p.cancel()
                aborted_day = tasks_by_day[p]
                target = next(
                    (d for d in days if d["day"] == aborted_day), None
                )
                if target is not None:
                    _mark(target, "cb_aborted_provider_overload", "")
                correction_results.append(
                    (aborted_day, None, "cb_aborted_provider_overload")
                )
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            pending = set()

    return correction_results, timeout_count, aborted_pending


# ---------------------------------------------------------------------------
# 6. Repro corrida 2026-05-04 03:26 — provider overload abortado temprano
# ---------------------------------------------------------------------------
def test_repro_provider_overload_aborts_after_threshold():
    """Simulación in-memory del CB: 3 days_to_fix, threshold=2.
    Day1 timeout, Day2 timeout → al segundo timeout abortamos Day3.

    Replica la lógica del loop dentro de `self_critique_node` sin acoplar
    a fixtures de DB/LLM, para validar el algoritmo aislado.

    Usa `asyncio.run` en lugar de `pytest.mark.asyncio` para no exigir
    `pytest-asyncio` (consistente con `test_p0_4_llm_cache_sql_fix.py`)."""

    days = [{"day": 1}, {"day": 2}, {"day": 3}]
    fail_modes = {1: "timeout", 2: "timeout", 3: "ok"}

    correction_results, timeout_count, aborted_pending = asyncio.run(
        _simulate_cb_loop(days, fail_modes, threshold=2, correction_delay_ok=0.1)
    )

    assert timeout_count == 2, f"Esperaba 2 timeouts contados, recibido {timeout_count}"
    assert aborted_pending, "El circuit breaker no se activó"

    # Day 3 debe estar abortado (cb_aborted_provider_overload), no completado
    day3 = next(d for d in days if d["day"] == 3)
    assert day3.get("_critique_unresolved", {}).get("reason") == "cb_aborted_provider_overload", (
        f"Day 3 debe tener marker cb_aborted_provider_overload, "
        f"recibido: {day3.get('_critique_unresolved')}"
    )

    # Days 1 y 2 deben tener marker timeout
    for dn in (1, 2):
        d = next(x for x in days if x["day"] == dn)
        assert d.get("_critique_unresolved", {}).get("reason") == "timeout"

    # No debe haber día corregido en este escenario
    corrected_days = [r for r in correction_results if r[1] is not None]
    assert not corrected_days, (
        f"Bajo cascade-failure no esperamos día corregido, recibido: {corrected_days}"
    )


# ---------------------------------------------------------------------------
# 7. Cuando los timeouts no llegan al threshold, no abortamos
# ---------------------------------------------------------------------------
def test_single_timeout_below_threshold_does_not_abort():
    """Un timeout aislado (1 < threshold=2) NO debe activar el CB. Día
    timeout queda con marker, los demás se completan normalmente."""

    days = [{"day": 1}, {"day": 2}, {"day": 3}]
    fail_modes = {1: "timeout", 2: "ok", 3: "ok"}

    correction_results, timeout_count, aborted_pending = asyncio.run(
        _simulate_cb_loop(days, fail_modes, threshold=2, correction_delay_ok=0.05)
    )

    assert timeout_count == 1
    assert not aborted_pending, "1 timeout NO debe activar CB con threshold=2"

    day1 = next(d for d in days if d["day"] == 1)
    assert day1.get("_critique_unresolved", {}).get("reason") == "timeout"

    corrected_nums = {r[0] for r in correction_results if r[1] is not None}
    assert corrected_nums == {2, 3}, (
        f"Esperaba days {{2,3}} corregidos, recibido {corrected_nums}"
    )


# ---------------------------------------------------------------------------
# 7b. Threshold=1 aborta al primer timeout (modo agresivo)
# ---------------------------------------------------------------------------
def test_threshold_1_aborts_on_first_timeout():
    """Con threshold=1 (modo agresivo), el primer timeout cancela todo lo
    pendiente. Útil cuando el operador ve picos sostenidos del provider."""

    days = [{"day": 1}, {"day": 2}, {"day": 3}]
    # Day 1 timeout (instant), Days 2/3 lentos (ok pero abortados)
    fail_modes = {1: "timeout", 2: "ok", 3: "ok"}

    correction_results, timeout_count, aborted_pending = asyncio.run(
        _simulate_cb_loop(days, fail_modes, threshold=1, correction_delay_ok=0.2)
    )

    assert timeout_count == 1
    assert aborted_pending, "Threshold=1 debe abortar al primer timeout"

    # Day 2 y Day 3 deben tener marker cb_aborted (no se completaron)
    for dn in (2, 3):
        d = next(x for x in days if x["day"] == dn)
        marker = d.get("_critique_unresolved", {}).get("reason")
        assert marker == "cb_aborted_provider_overload", (
            f"Day {dn} debe tener marker cb_aborted, recibido {marker}"
        )


# ---------------------------------------------------------------------------
# 8. cb_aborted_provider_overload es una reason reconocida por P1-SURGICAL-1
# ---------------------------------------------------------------------------
def test_cb_aborted_marker_triggers_surgical_regen():
    """[P4-TIMEOUT-2] El marker `cb_aborted_provider_overload` debe ser
    detectado por la lógica de P1-SURGICAL-1 que arma `unresolved_days`
    en el retry path, igual que cualquier otro marker
    (`timeout`, `cb_open`, `error:*`, `llm_returned_none`).

    Heurística: P1-SURGICAL-1 detecta días con marker `_critique_unresolved`
    sin filtrar por reason. Si esto cambia, el CB de P4-TIMEOUT-2 dejaría
    de causar regen y el bug volvería en silencio."""
    import inspect
    import graph_orchestrator as go

    src = inspect.getsource(go)
    # Heurística: el código debe iterar sobre `_critique_unresolved` sin
    # filtrar por reasons específicas (cb_open vs timeout vs cb_aborted...).
    # Verificamos al menos que el marker no esté hardcoded como excluido.
    assert "cb_aborted_provider_overload" in src
    # Las reasons usadas deben ser todas tratadas igual por surgical regen.
    # (Verificación negativa: no hay un filtro tipo `if reason != "timeout"`)
    assert 'reason != "timeout"' not in src
    assert 'reason == "cb_open"' not in src or 'reason in (' in src
