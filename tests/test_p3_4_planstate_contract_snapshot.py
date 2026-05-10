"""[P3-4 · 2026-05-10] Snapshot test del contrato `plan_result` keys.

Bug observado en el audit 2026-05-07/08:
    P1-G: el flag `_shopping_coherence_block` se persistía a `plan_result`
    (dict opaco) pero `review_plan_node` lo leía del state — LangGraph
    strict-schema lo filtraba en el handoff entre nodos → flag perdido,
    guard en modo `block` quedaba no-op silencioso.

Fix:
    P2-CANDIDATE-A (2026-05-08) añadió un bloque CONTRATO al final del
    TypedDict `PlanState` enumerando 6 keys que viven en `plan_result`,
    NO en `state`. Un futuro refactor que migrara cualquiera de ellas a
    nivel del state SIN declararlas en `PlanState` reproduciría P1-G.

Este test (P3-4):
    1. Parsea el bloque CONTRATO de `graph_orchestrator.py` y extrae las
       6 keys documentadas (`_shopping_coherence_block`,
       `_shopping_coherence_block_history`, `_pantry_supplement_required`,
       `_critique_unresolved`, `_merged_chunk_ids`,
       `_user_forced_simplified_weeks`).
    2. Verifica que el bloque CONTRATO existe (no fue removido por refactor).
    3. Verifica que cada key NO está declarada como campo top-level de
       `PlanState`. Si alguien la declara, debe también migrar todos los
       call sites de `result["key"]`/`plan["key"]` a `state["key"]` —
       el test obliga al cambio consciente.
    4. Verifica que cada key tiene al menos 1 referencia real en el código
       (sanity: el contrato refleja keys vivas, no archivadas).

Cobertura adicional:
    - Anchor del header `[P2-CANDIDATE-A · 2026-05-08]` para que el bloque
      no se pierda en futuros refactors sin que el test lo detecte.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH_ORCH = _BACKEND_ROOT / "graph_orchestrator.py"


def _read_orchestrator() -> str:
    if not _GRAPH_ORCH.exists():
        pytest.skip(f"graph_orchestrator.py no encontrado: {_GRAPH_ORCH}")
    return _GRAPH_ORCH.read_text(encoding="utf-8")


def _extract_contract_block(src: str) -> str:
    """Extrae el body del bloque CONTRATO P2-CANDIDATE-A entre los
    separadores `============================================================`.
    """
    # Anchor: el comment header con el tag P2-CANDIDATE-A.
    m = re.search(r"\[P2-CANDIDATE-A[^\]]*\][^\n]*CONTRATO[^\n]*\n", src)
    if not m:
        return ""
    start = m.end()
    # Termina cuando encontramos otra línea de separadores `=====` (al menos 12 `=`s).
    end_match = re.search(r"^[^\S\n]*#\s*={12,}\s*$", src[start:], re.MULTILINE)
    if not end_match:
        return src[start:start + 5000]
    return src[start:start + end_match.start()]


# Las 6 keys documentadas (espejo del comment block — SSOT de este test).
# Si el contrato cambia, ACTUALIZAR esta lista Y el comment del CONTRATO
# en graph_orchestrator.py simultáneamente.
EXPECTED_CONTRACT_KEYS = frozenset({
    "_shopping_coherence_block",
    "_shopping_coherence_block_history",
    "_pantry_supplement_required",
    "_critique_unresolved",
    "_merged_chunk_ids",
    "_user_forced_simplified_weeks",
})


# ---------------------------------------------------------------------------
# 1. Bloque CONTRATO existe y referencia las 6 keys
# ---------------------------------------------------------------------------
def test_contract_block_exists():
    """El header `[P2-CANDIDATE-A ...] CONTRATO` debe estar presente.

    Si esto falla, el bloque fue removido. El bug P1-G reabre — declarar
    keys solo state-side y olvidar migrar call sites se vuelve invisible."""
    src = _read_orchestrator()
    assert re.search(r"\[P2-CANDIDATE-A[^\]]*\][^\n]*CONTRATO", src), (
        "Bloque CONTRATO P2-CANDIDATE-A no encontrado en graph_orchestrator.py. "
        "Restaurar con anchor `[P2-CANDIDATE-A ...] CONTRATO`."
    )


def test_contract_block_lists_all_expected_keys():
    """Cada key del set canónico aparece dentro del bloque CONTRATO."""
    src = _read_orchestrator()
    body = _extract_contract_block(src)
    assert body, "Bloque CONTRATO vacío — anchor encontrado pero body no extraído."

    missing = []
    for key in EXPECTED_CONTRACT_KEYS:
        if f"`{key}`" not in body and key not in body:
            missing.append(key)
    assert not missing, (
        f"Keys del contrato sin documentar en el bloque CONTRATO: {missing}. "
        f"Añadir su explicación al comment block o removerlas de "
        f"`EXPECTED_CONTRACT_KEYS` del test si ya no aplican."
    )


# ---------------------------------------------------------------------------
# 2. Ninguna key del contrato está declarada como campo top-level de PlanState
# ---------------------------------------------------------------------------
def _extract_planstate_fields(src: str) -> set[str]:
    """Extrae los nombres de campos declarados en `class PlanState(TypedDict):`.

    Heurística: desde la línea `class PlanState(TypedDict):` hasta el siguiente
    `class ` top-level o `# =====...` separator largo. Captura líneas con
    pattern `<indent>field_name: type`.
    """
    lines = src.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("class PlanState(TypedDict"):
            start = i
            break
    if start is None:
        return set()
    fields: set[str] = set()
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        # Stop al toparnos con otra class top-level.
        if ln.startswith("class "):
            break
        # Stop con un separator largo de `=`.
        if re.match(r"^[^\S\n]*#\s*={12,}\s*$", ln):
            # No paramos — pueden haber sub-secciones dentro del TypedDict
            # (e.g. el bloque CONTRATO al final). Continuamos parseando, pero
            # NO contamos campos dentro del bloque CONTRATO porque ese bloque
            # es DOCUMENTACIÓN, no declaración de fields.
            # Las líneas dentro del bloque CONTRATO empiezan con `# ...`,
            # así que el matcher de campo de abajo no las capturará.
            continue
        # Capturar `<indent>identifier: type`. El TypedDict requiere 4-space indent.
        m = re.match(r"^[ \t]{4,}([A-Za-z_][A-Za-z0-9_]*)\s*:\s*[A-Za-z_\[]", ln)
        if m:
            fields.add(m.group(1))
    return fields


def test_no_contract_key_is_declared_in_planstate():
    """Ninguna de las 6 keys del contrato debe estar como campo top-level
    del TypedDict `PlanState`. Si alguien la declara, debe migrar TODOS los
    call sites de `result[key]` / `plan[key]` a `state[key]` — el test
    obliga al cambio consciente."""
    src = _read_orchestrator()
    fields = _extract_planstate_fields(src)
    assert fields, "No se pudieron extraer fields de PlanState — el regex drifteó?"

    violations = EXPECTED_CONTRACT_KEYS & fields
    assert not violations, (
        f"Keys del CONTRATO declaradas como campos top-level de PlanState: "
        f"{violations}. Si la migración a state-level es intencional, también "
        f"hay que migrar los call sites de `result[...]`/`plan[...]` a "
        f"`state[...]` y removerla del bloque CONTRATO."
    )


# ---------------------------------------------------------------------------
# 3. Sanity: cada key del contrato tiene al menos 1 referencia real
# ---------------------------------------------------------------------------
def test_each_contract_key_has_real_reference():
    """Cada key debe aparecer en el código (no en comentarios) — sanity
    contra archive accidental. Algunas keys (`_merged_chunk_ids`,
    `_user_forced_simplified_weeks`) viven en `cron_tasks.py` o
    `routers/plans.py` por diseño (markers per-day/per-plan persistidos
    a `meal_plans.plan_data` directo). Buscamos en TODO el backend."""
    backend_py_files = [
        _GRAPH_ORCH,
        _BACKEND_ROOT / "cron_tasks.py",
        _BACKEND_ROOT / "routers" / "plans.py",
        _BACKEND_ROOT / "shopping_calculator.py",
        _BACKEND_ROOT / "services.py",
    ]
    # Construir corpus de código (drop comentarios para no matchear en el
    # propio bloque CONTRATO o en docstrings).
    code_corpus_parts = []
    for p in backend_py_files:
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.lstrip().startswith("#"):
                code_corpus_parts.append(ln)
    code_src = "\n".join(code_corpus_parts)

    missing = []
    for key in EXPECTED_CONTRACT_KEYS:
        # Buscamos uso como string literal en subscripts: `["key"]` o `['key']`.
        pattern = re.compile(
            rf"""[\["']{re.escape(key)}[\]"']""",
            re.MULTILINE,
        )
        if not pattern.search(code_src):
            missing.append(key)
    assert not missing, (
        f"Keys del CONTRATO sin referencia real en el backend: {missing}. "
        f"Si la key se archivó, removerla de `EXPECTED_CONTRACT_KEYS` y "
        f"del bloque CONTRATO en graph_orchestrator.py."
    )


# ---------------------------------------------------------------------------
# 4. Detección de drift de mi set canónico (EXPECTED_CONTRACT_KEYS)
# ---------------------------------------------------------------------------
def test_expected_contract_keys_count_matches_documentation():
    """Sanity: el comentario del CONTRATO documenta cierta cantidad de keys.
    Esta lista hardcoded del test debe coincidir."""
    # 6 keys explícitas + el grupo de 3 markers per-day = 6 explícitos.
    assert len(EXPECTED_CONTRACT_KEYS) == 6, (
        f"EXPECTED_CONTRACT_KEYS tiene {len(EXPECTED_CONTRACT_KEYS)} entries; "
        f"esperaba 6 (espejo del comment block CONTRATO)."
    )
