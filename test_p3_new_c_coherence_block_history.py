"""[P3-NEW-C · 2026-05-08] Tests para telemetría histórica del coherence block.

Bug original (audit 2026-05-07):
  Cuando `MEALFIT_SHOPPING_COHERENCE_GUARD=block` detecta divergencias y
  `review_plan_node` aplica `reject_*` (o `degrade`), el flag transitorio
  `_shopping_coherence_block` se setea pero NO se persistía un registro
  histórico en `meal_plans.plan_data`. Sin telemetría histórica, no se
  podía analizar a largo plazo qué % de planes tropezaron con coh-block,
  qué hipótesis dominaron, ni cuántos retries tomó resolver.

Fix:
  1. `assemble_plan_node` captura el retorno de `run_shopping_coherence_guard`
     y, si hay divergencias, append a `result["_shopping_coherence_block_history"]`.
  2. La entry incluye: `ts`, `attempt`, `divergence_count`, `presence_count`,
     `magnitude_count`, `hypotheses` (counter), `block_set` (bool), `action_taken` (None).
  3. La history se preserva a través de retries leyendo del `state["plan_result"]`
     del attempt previo (LangGraph mantiene state entre nodos).
  4. Cap a 20 entries para evitar bloat en plan_data.
  5. `review_plan_node` rellena `action_taken` de la última entry con el knob
     resuelto (`reject_minor`/`reject_high`/`degrade`) cuando consume el flag.

Cobertura (defensa textual sobre el source — graph_orchestrator es heavy
import, mismo patrón que test_p2_new_d_f_scheduler_telemetry):
  - test_history_append_present_in_assemble
  - test_history_capped_at_20
  - test_history_preserves_prior_across_retries
  - test_history_entry_has_required_fields
  - test_review_marks_action_taken_on_last_entry
  - test_history_only_appends_when_divergences_present
  - test_history_field_name_consistent
"""
import re
from pathlib import Path

import pytest


_GO_PATH = Path(__file__).parent / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def go_source() -> str:
    return _GO_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def assemble_block(go_source) -> str:
    """Aísla el bloque del guard recetas↔lista en `assemble_plan_node`."""
    start = go_source.find("from shopping_calculator import run_shopping_coherence_guard")
    assert start != -1, "No se encontró el import de run_shopping_coherence_guard."
    # Acotamos hasta el siguiente comentario `# [P0-6]` o el `duration =`
    end = go_source.find("duration = round(time.time() - start_time, 2)", start)
    assert end != -1
    return go_source[start:end]


@pytest.fixture(scope="module")
def review_block(go_source) -> str:
    """Aísla el bloque de consumo de `_shopping_coherence_block` en review_plan_node."""
    start = go_source.find('coherence_block = plan.get("_shopping_coherence_block")')
    assert start != -1, "No se encontró el consumer del flag en review_plan_node."
    # Acotar hasta la siguiente sección visible.
    end = go_source.find("# Brechas 1 y 4: Errores deterministas del ensamblador", start)
    assert end != -1
    return go_source[start:end]


# ---------------------------------------------------------------------------
# 1. assemble_plan_node: history append
# ---------------------------------------------------------------------------
def test_history_append_present_in_assemble(assemble_block):
    """assemble_plan_node debe escribir a `_shopping_coherence_block_history`
    en el `result`."""
    assert '_shopping_coherence_block_history' in assemble_block, (
        "assemble_plan_node debe persistir en `_shopping_coherence_block_history`."
    )
    # Debe asignar a result["_shopping_coherence_block_history"].
    assert re.search(
        r'result\[\s*["\']_shopping_coherence_block_history["\']\s*\]\s*=',
        assemble_block,
    ), (
        "assemble_plan_node debe asignar la lista a "
        "`result['_shopping_coherence_block_history']` para que se persista en "
        "meal_plans.plan_data downstream."
    )


def test_history_capped_at_20(assemble_block):
    """La history debe estar acotada a las últimas 20 entries para evitar
    bloat ilimitado en `meal_plans.plan_data` (planes con muchos retries
    o múltiples chunks que comparten plan_data)."""
    assert re.search(r"len\([^)]+\)\s*>\s*20", assemble_block) or "[-20:]" in assemble_block, (
        "Cap explícito de 20 entries no detectado. Sin cap, planes con "
        "muchos retries pueden inflar plan_data ilimitadamente."
    )


def test_history_preserves_prior_across_retries(assemble_block):
    """Cada nueva invocación del guard debe preservar la history previa
    leyendo del `state['plan_result']` del attempt anterior. Sin esto, un
    retry pierde toda la telemetría histórica acumulada."""
    # state.get("plan_result") con _shopping_coherence_block_history.
    assert "state.get(\"plan_result\")" in assemble_block or "state.get('plan_result')" in assemble_block, (
        "Debe leer state['plan_result'] del attempt previo para preservar history."
    )
    # Debe leer la key de history del prior result.
    assert re.search(
        r"\.get\(\s*[\"']_shopping_coherence_block_history[\"']\s*\)",
        assemble_block,
    ), (
        "Debe leer la history previa del plan_result anterior."
    )


def test_history_entry_has_required_fields(assemble_block):
    """Cada entry debe contener los campos canónicos que documenta P3-NEW-C."""
    required_fields = [
        '"ts"', '"attempt"', '"divergence_count"', '"presence_count"',
        '"magnitude_count"', '"hypotheses"', '"block_set"', '"action_taken"',
    ]
    for field in required_fields:
        assert field in assemble_block, (
            f"Entry de history debe incluir el campo {field}. Sin este campo, "
            f"el análisis post-mortem queda ciego en esa dimensión."
        )


def test_history_uses_utc_timestamp(assemble_block):
    """`ts` debe ser ISO format en UTC para correlacionar con logs/Grafana."""
    assert "datetime" in assemble_block and "utc" in assemble_block.lower(), (
        "Timestamp debe ser UTC (datetime.now(timezone.utc).isoformat())."
    )


def test_history_only_appends_when_divergences_present(assemble_block):
    """Solo append cuando el guard retorna divergencias non-empty —
    sin esto, planes "limpios" generarían entries vacías que inflan el
    JSON sin valor analítico."""
    # El bloque de history debe estar dentro de un `if coh_divergences:`.
    assert "if coh_divergences:" in assemble_block, (
        "El append debe estar gated por `if coh_divergences:` — entries "
        "vacías para planes sin divergencias inflan plan_data sin valor."
    )


def test_assemble_does_not_abort_on_history_failure(assemble_block):
    """Si el bloque de telemetría explota por bug interno (e.g., import
    falla, datetime no disponible en algún env), NO debe abortar el assembly.
    La telemetría es best-effort."""
    # Buscar try/except específico al bloque de history.
    history_section = assemble_block[assemble_block.find("if coh_divergences:"):]
    assert "try:" in history_section and "except" in history_section, (
        "El bloque de history debe tener try/except defensivo. La "
        "telemetría es secundaria — un bug en datetime/Counter no debe "
        "tirar el plan completo."
    )


# ---------------------------------------------------------------------------
# 2. review_plan_node: action_taken hidration
# ---------------------------------------------------------------------------
def test_review_marks_action_taken_on_last_entry(review_block):
    """review_plan_node debe rellenar `action_taken` de la última entry
    de history con el knob resuelto (reject_minor/reject_high/degrade)."""
    assert "_shopping_coherence_block_history" in review_block, (
        "review_plan_node debe leer/escribir _shopping_coherence_block_history."
    )
    # Debe asignar action_taken al último elemento.
    assert re.search(
        r'\[-1\]\[["\']action_taken["\']\]\s*=\s*_block_action',
        review_block,
    ), (
        "Debe hidratar `_coh_hist[-1]['action_taken'] = _block_action` para "
        "que la entry refleje qué decisión tomó el reviewer."
    )


def test_review_action_taken_marker_runs_for_all_block_actions(review_block):
    """La hidratación de `action_taken` debe ocurrir ANTES del branching
    `if _block_action == 'degrade': ... else: ...` para que aplique en
    AMBAS ramas (degrade Y reject_*)."""
    mark_pos = review_block.find("action_taken")
    degrade_branch_pos = review_block.find('if _block_action == "degrade":')
    assert mark_pos != -1 and degrade_branch_pos != -1
    assert mark_pos < degrade_branch_pos, (
        "El marker de action_taken debe estar ANTES del branching degrade/reject_*. "
        "Si va dentro de una sola rama, la otra queda con action_taken=None."
    )


def test_review_history_marker_is_defensive(review_block):
    """El marker debe ser defensivo: si la history está vacía o el último
    item no es dict, no debe fallar (compatibilidad con planes legacy
    pre-P3-NEW-C cargados desde cache semántico).

    Verificamos que el bloque tenga: (a) try/except que envuelve el marker;
    y (b) el chequeo `isinstance(..., list)` + truthiness + `isinstance(..., dict)`
    antes de mutar `[-1]`."""
    # Aislar el bloque entre `try:` y el next branching `if _block_action ==`
    try_idx = review_block.find("try:")
    branch_idx = review_block.find('if _block_action == "degrade":')
    assert try_idx != -1 and branch_idx != -1 and try_idx < branch_idx, (
        "Estructura inesperada del bloque: `try:` debe preceder al "
        "branching de _block_action."
    )
    marker_block = review_block[try_idx:branch_idx]
    assert "isinstance" in marker_block, (
        "Marker debe usar isinstance() para validar shape antes de mutar."
    )
    assert "list" in marker_block and "dict" in marker_block, (
        "Marker debe validar que `_coh_hist` es lista no-vacía con dict en "
        "el último slot antes de mutar. Sin esto, un plan legacy con la "
        "key ausente o malformada (e.g. cacheado pre-P3-NEW-C) tira "
        "AttributeError dentro del path crítico de review."
    )
    assert "except" in marker_block, (
        "Marker debe tener except defensivo. La telemetría es secundaria — "
        "un bug en el marker NO debe abortar review_plan_node."
    )


# ---------------------------------------------------------------------------
# 3. Field name consistency
# ---------------------------------------------------------------------------
def test_history_field_name_consistent(go_source):
    """El nombre del campo `_shopping_coherence_block_history` debe ser
    consistente — sin variantes accidentales (`_shopping_coh_history`,
    `_coherence_block_history`, etc.) que romperían la lectura downstream."""
    # Solo debe aparecer la variante canónica.
    canonical = go_source.count("_shopping_coherence_block_history")
    assert canonical >= 3, (
        f"_shopping_coherence_block_history debe aparecer al menos 3× "
        f"(assemble write + assemble read prior + review read; "
        f"review mutation va vía variable local _coh_hist post-read); "
        f"vio {canonical}."
    )
    # No debe haber variantes typo-ed (excluyendo aquellas que son
    # substring de la variante canónica).
    # `_coherence_block_history` es substring del canónico → no chequeable así.
    typo_patterns = [
        "_shopping_coh_history",
        "_coh_block_history",
        "shopping_coherence_history",  # falta prefijo `_` inicial
    ]
    for typo in typo_patterns:
        # Word boundary check: que el typo no aparezca como token aislado.
        if re.search(rf"\b{re.escape(typo)}\b", go_source):
            pytest.fail(
                f"Variante typo '{typo}' encontrada en el source — drift "
                f"de naming rompería la lectura downstream."
            )
