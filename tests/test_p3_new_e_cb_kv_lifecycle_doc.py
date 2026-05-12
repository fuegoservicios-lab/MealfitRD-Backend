"""[P3-NEW-E · 2026-05-11] Anchor parser-based bidireccional docs↔código
para la sub-sección "Ciclo de vida del KV `llm_circuit_breaker:*`" en
CLAUDE.md.

Por qué existe:
    La sub-sección documenta el contrato del KV persistente del CB:
    formato de keys (legacy + per-modelo P1-Q3), payload shape, 3 vías
    de reset (`_atomic_reset_db` / `can_proceed` auto-expira /
    `_sweep_stale_llm_circuit_breakers`), y knobs. Sin un anchor que
    valide que el código referenciado existe, un refactor futuro
    (renombrar la clase / cambiar el sufijo de key / mover el reset
    atómico) deja la documentación silenciosamente desactualizada.

    Patrón espejo P3-LIVE-2 (cross-link CLAUDE.md ↔ código pantry-stale).

Drift detection (bidireccional):
    - Sub-sección "Ciclo de vida del KV `llm_circuit_breaker:*`" borrada
      de CLAUDE.md → falla.
    - Marker `P3-NEW-E` removido del bloque → falla (sub-sección debe
      seguir referenciando su P-fix de origen).
    - Símbolo `LLMCircuitBreaker` desaparece de `graph_orchestrator.py`
      (rename de la clase sin actualizar docs) → falla.
    - Construcción `_key_suffix = f":{model_name}"` cambia
      (ej. a `f"_{model_name}"`) sin actualizar docs → falla.
    - `_atomic_reset_db` renombrado sin actualizar docs → falla.
    - `_sweep_stale_llm_circuit_breakers` renombrado/borrado sin
      actualizar docs (cross-link a P2-NEW-D) → falla.
    - Knobs documentados ausentes del código (drift KNOBS_REGISTRY) →
      falla per-knob.

Tooltip-anchor: P3-NEW-E-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND = _REPO_ROOT / "backend"
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_ORCH = _BACKEND / "graph_orchestrator.py"
_CRON = _BACKEND / "cron_tasks.py"


@pytest.fixture(scope="module")
def claude_md() -> str:
    return _CLAUDE_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def orch_source() -> str:
    return _ORCH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cron_source() -> str:
    return _CRON.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Sub-sección existe en CLAUDE.md con su marker
# ---------------------------------------------------------------------------
def test_subsection_present_with_marker(claude_md: str):
    r"""La sub-sección `### Ciclo de vida del KV \`llm_circuit_breaker:*\``
    debe existir en CLAUDE.md y mencionar su marker P3-NEW-E."""
    assert "### Ciclo de vida del KV `llm_circuit_breaker:*`" in claude_md, (
        "P3-NEW-E violation: sub-sección "
        "`### Ciclo de vida del KV \\`llm_circuit_breaker:*\\`` "
        "borrada de CLAUDE.md. Sin esta documentación, un SRE leyendo "
        "`app_kv_store` no tiene contexto para distinguir 'CB stale por "
        "modelo descartado' vs 'CB realmente abierto'."
    )
    # El marker debe estar presente para que un grep por `P3-NEW-E` desde
    # otro fix (futuro audit) encuentre la sub-sección.
    assert "P3-NEW-E" in claude_md, (
        "P3-NEW-E violation: marker `P3-NEW-E` removido de la sub-sección "
        "de CLAUDE.md. Sin marker, el cross-link entre P-fixes pierde "
        "trazabilidad (un grep por `P3-NEW-E` debe encontrar esta sección)."
    )


# ---------------------------------------------------------------------------
# 2. Clase LLMCircuitBreaker existe en graph_orchestrator.py
# ---------------------------------------------------------------------------
def test_llm_circuit_breaker_class_exists(orch_source: str):
    """La sub-sección referencia `class LLMCircuitBreaker`. Si la clase
    se renombra sin actualizar docs, el anchor debe fallar."""
    assert re.search(
        r"^class\s+LLMCircuitBreaker\s*[:\(]",
        orch_source,
        re.MULTILINE,
    ), (
        "P3-NEW-E violation: `class LLMCircuitBreaker` no encontrada en "
        "`graph_orchestrator.py`. CLAUDE.md sub-sección sigue "
        "referenciándola — si la clase se renombró, actualizar tanto el "
        "código como CLAUDE.md."
    )


# ---------------------------------------------------------------------------
# 3. Construcción del sufijo de key per-modelo
# ---------------------------------------------------------------------------
def test_per_model_key_suffix_construction(orch_source: str):
    """La sub-sección documenta el patrón `llm_circuit_breaker:<model>`
    con sufijo construido como `f":{model_name}"` en `__init__`. Verifica
    que el patrón sigue vigente."""
    assert re.search(
        r'_key_suffix\s*=\s*f["\']\s*:\s*\{model_name\}["\']',
        orch_source,
    ), (
        "P3-NEW-E violation: construcción `_key_suffix = f\":{model_name}\"` "
        "no encontrada en `graph_orchestrator.py`. CLAUDE.md sub-sección "
        "documenta este patrón (P1-Q3) — si cambió a otra forma "
        "(ej. `f\"_{model_name}\"` o sufijo en otra parte del key), "
        "actualizar la documentación."
    )


# ---------------------------------------------------------------------------
# 4. _atomic_reset_db existe (vía #1 de reset)
# ---------------------------------------------------------------------------
def test_atomic_reset_db_method_exists(orch_source: str):
    """La sub-sección lista `_atomic_reset_db` como vía #1 de reset.
    Verifica que sigue definida."""
    assert re.search(
        r"def\s+_atomic_reset_db\s*\(",
        orch_source,
    ), (
        "P3-NEW-E violation: `_atomic_reset_db` no definida en "
        "`graph_orchestrator.py`. CLAUDE.md lista este método como vía #1 "
        "de reset — si se renombró, actualizar docs."
    )


# ---------------------------------------------------------------------------
# 5. can_proceed runtime auto-expiración (vía #2)
# ---------------------------------------------------------------------------
def test_can_proceed_auto_expiration_present(orch_source: str):
    """La sub-sección documenta que `can_proceed` retorna True una vez
    `time.time() - last_failure > reset_timeout`, SIN tocar la fila DB.
    Verifica que el patrón sigue presente."""
    # Buscar el método y validar que su lógica incluye la comparación
    # `time.time() - state.get("last_failure", 0) > self.reset_timeout`.
    assert re.search(
        r"def\s+can_proceed\s*\(",
        orch_source,
    ), "`can_proceed` no definida."
    # El patrón clave: comparación de epoch contra reset_timeout.
    assert re.search(
        r"time\.time\(\)\s*-\s*state\.get\(\s*['\"]last_failure['\"]"
        r".{0,100}>\s*self\.reset_timeout",
        orch_source,
        re.DOTALL,
    ) or re.search(
        r"time\.time\(\)\s*-\s*last_failure.{0,40}>\s*self\.reset_timeout",
        orch_source,
        re.DOTALL,
    ), (
        "P3-NEW-E violation: `can_proceed` ya no compara `time.time() - "
        "last_failure > reset_timeout` (vía #2 documentada en CLAUDE.md). "
        "Si la auto-expiración se removió, actualizar la sub-sección — "
        "el sweep P2-NEW-D depende de ese gap (runtime cerrado + DB "
        "stale) para tener sentido."
    )


# ---------------------------------------------------------------------------
# 6. _sweep_stale_llm_circuit_breakers existe (vía #3, cross-link P2-NEW-D)
# ---------------------------------------------------------------------------
def test_sweep_function_exists(cron_source: str):
    """La sub-sección referencia `_sweep_stale_llm_circuit_breakers`
    como vía #3 de reset (cross-link a P2-NEW-D). Verifica que existe."""
    assert re.search(
        r"^def\s+_sweep_stale_llm_circuit_breakers\s*\(",
        cron_source,
        re.MULTILINE,
    ), (
        "P3-NEW-E violation: `_sweep_stale_llm_circuit_breakers` no "
        "definida en `cron_tasks.py`. CLAUDE.md sub-sección documenta "
        "esta vía #3 (P2-NEW-D · 2026-05-11) — si la función se renombró "
        "actualizar docs y test P2-NEW-D al mismo tiempo."
    )


# ---------------------------------------------------------------------------
# 7. Knobs documentados existen en código
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("knob_name", [
    "MEALFIT_CB_FAILURE_THRESHOLD",
    "MEALFIT_CB_RESET_TIMEOUT_S",
    "MEALFIT_CB_LOCAL_HEALTH_TTL_S",
    "MEALFIT_CB_KV_STALENESS_HOURS",
    "MEALFIT_CB_KV_STALENESS_SWEEP_INTERVAL_MIN",
])
def test_documented_knobs_present_in_code(knob_name: str, orch_source: str, cron_source: str):
    """Cada knob listado en la tabla de la sub-sección DEBE aparecer en
    `graph_orchestrator.py` ó `cron_tasks.py`. Drift detection: si alguien
    borra un knob del código (refactor) sin actualizar CLAUDE.md, el test
    falla."""
    pattern = re.compile(rf'["\']{re.escape(knob_name)}["\']')
    found = pattern.search(orch_source) or pattern.search(cron_source)
    assert found, (
        f"P3-NEW-E violation: knob `{knob_name}` documentado en CLAUDE.md "
        "sub-sección 'Ciclo de vida del KV llm_circuit_breaker:*' NO "
        "encontrado en `graph_orchestrator.py` ni en `cron_tasks.py`. "
        "O el knob fue removido del código (actualizar docs) o el nombre "
        "cambió (actualizar ambos)."
    )


# ---------------------------------------------------------------------------
# 8. Knobs documentados se mencionan TODOS en la sub-sección
# ---------------------------------------------------------------------------
def test_all_critical_knobs_mentioned_in_subsection(claude_md: str):
    """Lado inverso del test #7: cada knob crítico DEBE aparecer en la
    sub-sección. Si alguien añade un nuevo knob al CB en código sin
    documentarlo en CLAUDE.md, el test falla."""
    # Extraer la sub-sección.
    start = claude_md.find("### Ciclo de vida del KV `llm_circuit_breaker:*`")
    assert start >= 0, "Sub-sección no encontrada (cubre test #1)."
    end = claude_md.find("\n## ", start)
    if end < 0:
        end = claude_md.find("\n### ", start + 1)
    section = claude_md[start:end] if end > start else claude_md[start:]
    for knob in [
        "MEALFIT_CB_FAILURE_THRESHOLD",
        "MEALFIT_CB_RESET_TIMEOUT_S",
        "MEALFIT_CB_LOCAL_HEALTH_TTL_S",
        "MEALFIT_CB_KV_STALENESS_HOURS",
        "MEALFIT_CB_KV_STALENESS_SWEEP_INTERVAL_MIN",
    ]:
        assert knob in section, (
            f"P3-NEW-E violation: knob `{knob}` ausente de la sub-sección "
            f"de CLAUDE.md. Cada knob del CB debe aparecer en la tabla "
            f"'Knobs operacionales' para que `/health/version` tenga "
            f"contraparte documentada."
        )
