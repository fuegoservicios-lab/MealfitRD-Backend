"""[P6-CRITIQUE-DAY-FLOOR] Tests para el floor determinístico de días
afectados en self-critique.

Bug observable (PDF 2026-05-05 19:13:16):
  Detector determinístico `_detect_slot_incoherence` listó issues en
  Día 1, Día 2, Día 3 (todos con `'res' aparece en N comidas`). El LLM
  respondió `critique.suggestions` mencionando solo Día 1 y Día 3.
  Cap dinámico=3 tenía capacidad para los 3, pero el parser solo extrajo
  [1, 3] del texto del LLM → Día 2 quedó sin corregir.

  Resultado: usuario recibió Día 2 con 'res' en desayuno+almuerzo+merienda
  (señalado por el detector) sin que el corrector lo tocara.

Causa raíz:
  El parser:
    mentioned = list(dict.fromkeys(
        int(d) for d in _re.findall(r'[Dd]ía\\s*(\\d+)', critique.suggestions)
    ))
  solo mira el texto LIBRE del LLM. Si el LLM omite un día (por brevity,
  por priorización implícita, por hallucination), el parser lo pierde
  silenciosamente.

Fix:
  Usar `slot_issues` (calculado ANTES del LLM, determinístico) como FLOOR.
  Cualquier día mencionado por el detector se añade al `mentioned` aunque
  el LLM no lo haya echo'd back.

Cobertura:
  - LLM omite Día 2 que detector mencionó → floor lo recupera
  - LLM menciona TODOS los días del detector → no duplica
  - LLM menciona días EXTRA al detector → se preservan (LLM puede ser smart)
  - Detector vacío + LLM con días → comportamiento original
  - Sanity: marker `P6-CRITIQUE-DAY-FLOOR` en source
"""
import re
import pytest


def _parse_days_from_text(text: str) -> list:
    """Reproduce el extractor de días del self-critique node."""
    return list(dict.fromkeys(
        int(d) for d in re.findall(r'[Dd]ía\s*(\d+)', text)
    ))


def _apply_day_floor(llm_mentioned: list, slot_issues: list) -> list:
    """Reproduce el floor logic de P6-CRITIQUE-DAY-FLOOR + P2-ORCH-5.

    [P2-ORCH-5 · 2026-05-28] Floor-FIRST: los días con incoherencia
    determinística van al FRENTE (`deterministic_days + mentioned`), no al
    final, para que el slice `[:critique_max_days]` bajo presión de budget no
    los descarte. Antes era `mentioned + missing` (floor al final)."""
    deterministic_days = list(dict.fromkeys(
        int(d) for d in re.findall(r'[Dd]ía\s*(\d+)', "\n".join(slot_issues or []))
    ))
    mentioned = list(llm_mentioned)
    if deterministic_days:
        mentioned = list(dict.fromkeys(deterministic_days + mentioned))
    if not mentioned:
        mentioned = [1]
    return mentioned


# ===========================================================================
# 1. Repro PDF — Día 2 omitido por LLM debe ser recuperado
# ===========================================================================
def test_repro_pdf_dia2_recuperado_por_floor():
    """Caso real corrida 19:13: detector listó días 1/2/3, LLM solo mencionó
    1 y 3. Floor debe añadir 2."""
    slot_issues = [
        "Día 1: almuerzo y cena comparten proteína principal (res). Cambia la proteína de la cena.",
        "Día 1: la proteína 'res' aparece en 4 comidas (almuerzo, cena, desayuno, merienda).",
        "Día 2: la proteína 'res' aparece en 3 comidas (almuerzo, desayuno, merienda).",
        "Día 3: almuerzo y cena comparten carbohidrato principal (batata).",
        "Día 3: la proteína 'res' aparece en 3 comidas (cena, desayuno, merienda).",
        "Día 3: la proteína 'pavo' aparece en 2 comidas (cena, merienda).",
    ]
    llm_suggestions = (
        "Día 1: Cambiar la proteína de la cena para no repetir 'res' y variar el desayuno. "
        "Día 3: Sustituir la batata en la cena por otro carbohidrato y cambiar el pavo de la merienda."
    )
    mentioned = _parse_days_from_text(llm_suggestions)
    assert mentioned == [1, 3], "Pre-floor: LLM omitió Día 2 como en el bug real"

    final = _apply_day_floor(mentioned, slot_issues)
    assert 2 in final, f"Día 2 debe ser añadido por el floor: {final}"
    assert set(final) == {1, 2, 3}, f"Esperado {{1, 2, 3}}, recibido {set(final)}"


# ===========================================================================
# 2. LLM menciona TODOS los días del detector → no duplica
# ===========================================================================
def test_no_duplication_when_llm_mentions_all_days():
    slot_issues = [
        "Día 1: 'res' x4 comidas",
        "Día 2: 'res' x3 comidas",
    ]
    llm_suggestions = "Día 1: cambiar cena. Día 2: cambiar merienda."
    mentioned = _parse_days_from_text(llm_suggestions)
    final = _apply_day_floor(mentioned, slot_issues)
    assert final == [1, 2], f"No debe duplicar; esperado [1, 2], recibido {final}"


# ===========================================================================
# 3. LLM menciona días EXTRA al detector → se preservan
# ===========================================================================
def test_llm_extra_days_preserved():
    """Si el LLM ve algo que el detector no (ej. coherencia cultural), debe
    preservarse junto con los del floor."""
    slot_issues = ["Día 2: 'res' x3 comidas"]
    llm_suggestions = "Día 1: cambiar técnica. Día 2: variar proteína. Día 3: muy frío."
    mentioned = _parse_days_from_text(llm_suggestions)
    final = _apply_day_floor(mentioned, slot_issues)
    assert set(final) == {1, 2, 3}, f"LLM extras deben quedarse: {final}"


# ===========================================================================
# 4. Detector vacío → comportamiento original (solo LLM)
# ===========================================================================
def test_empty_detector_falls_back_to_llm_only():
    slot_issues = []
    llm_suggestions = "Día 2: revisar."
    mentioned = _parse_days_from_text(llm_suggestions)
    final = _apply_day_floor(mentioned, slot_issues)
    assert final == [2]


# ===========================================================================
# 5. Detector vacío + LLM vacío → default Día 1
# ===========================================================================
def test_empty_both_defaults_to_dia1():
    final = _apply_day_floor([], [])
    assert final == [1]


# ===========================================================================
# 6. Detector con día pero LLM vacío → floor lo provee
# ===========================================================================
def test_detector_only_provides_floor():
    slot_issues = ["Día 3: incoherencia"]
    final = _apply_day_floor([], slot_issues)
    assert final == [3]


# ===========================================================================
# 7. [P2-ORCH-5] Orden FLOOR-FIRST: los días determinísticos van primero (su
#    orden), luego los extras del LLM. Antes (pre-P2-ORCH-5) era al revés
#    (LLM primero, floor al final) → bajo cap dinámico=1-2 los días con
#    incoherencia code-cierta se descartaban. Ahora el floor sobrevive al slice.
# ===========================================================================
def test_order_preservation_floor_first():
    """Floor-first: días determinísticos al frente (en su orden), luego extras LLM."""
    slot_issues = ["Día 1: x", "Día 2: y", "Día 3: z"]
    llm_suggestions = "Día 3: ... Día 1: ..."
    mentioned = _parse_days_from_text(llm_suggestions)
    assert mentioned == [3, 1]  # orden del LLM (pre-floor)
    final = _apply_day_floor(mentioned, slot_issues)
    # deterministic_days=[1,2,3] al frente, dedup absorbe los del LLM ya presentes.
    assert final == [1, 2, 3], f"Esperado [1, 2, 3] (floor-first P2-ORCH-5), recibido {final}"


def test_floor_day_survives_budget_slice():
    """[P2-ORCH-5] El día con incoherencia determinística (no mencionado por el
    LLM) DEBE quedar en el head del slice incluso con critique_max_days=1."""
    slot_issues = ["Día 2: 'res' x3 comidas"]  # día 2 = floor, code-cierto
    llm_suggestions = "Día 5: variar técnica."  # LLM menciona otro día
    mentioned = _parse_days_from_text(llm_suggestions)
    final = _apply_day_floor(mentioned, slot_issues)
    # Floor-first → [2, 5]; bajo cap=1 el slice [:1] conserva el día determinístico.
    assert final[0] == 2, f"El floor day 2 debe ir primero, recibido {final}"
    assert final[:1] == [2], f"Bajo cap=1 debe conservar el floor day, recibido {final[:1]}"


# ===========================================================================
# 8. Día con número >9 (regex \d+) — no regresión P1-1
# ===========================================================================
def test_double_digit_days():
    slot_issues = ["Día 10: incoherencia", "Día 11: otra"]
    llm_suggestions = "Día 10: revisar."
    mentioned = _parse_days_from_text(llm_suggestions)
    final = _apply_day_floor(mentioned, slot_issues)
    assert set(final) == {10, 11}, f"Esperado {{10, 11}}, recibido {set(final)}"


# ===========================================================================
# 9. Sanity guard — marker en source code
# ===========================================================================
def test_source_has_critique_day_floor_marker():
    import inspect
    import graph_orchestrator as go
    src = inspect.getsource(go.self_critique_node)
    assert "P6-CRITIQUE-DAY-FLOOR" in src, (
        "Marker debe existir; sin él alguien podría revertir el fix y "
        "reintroducir 'Día 2 sin corregir aunque hay capacidad'"
    )
    assert "deterministic_days" in src, "Variable del fix debe existir"
    assert "slot_issues" in src, "Source debe usar slot_issues como floor"
