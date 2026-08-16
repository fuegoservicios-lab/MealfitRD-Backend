"""[P1-CULINARY-JUDGE · 2026-08-01] Juez LLM culinario (F3). Nace OFF; nunca muta;
fail-open al path (jamás aprueba en silencio lo que capa 1 rechazó — solo
añade señal). Sin llamadas LLM en CI (calibración por script, spec §6)."""
import re
from pathlib import Path

_GO = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_knobs_nacen_apagados_y_en_flash():
    m = re.search(r'CULINARY_JUDGE_GUARD = \(_env_str\("MEALFIT_CULINARY_JUDGE_GUARD", "(\w+)"\)', _GO)
    assert m and m.group(1) == "off", "el juez nace OFF (no gasta tokens sin calibración)"
    assert '_env_str("MEALFIT_CULINARY_JUDGE_MODEL"' in _GO
    m2 = re.search(r'_env_str\("MEALFIT_CULINARY_JUDGE_MODEL",\s*(\w+)\)', _GO)
    assert m2 and m2.group(1) == "_FLASH_MODEL_NAME", "directiva del owner: flash, NO pro"
    assert '_env_int("MEALFIT_CULINARY_JUDGE_TIMEOUT_S", 45' in _GO


def test_el_juez_recibe_recetas_completas():
    i = _GO.index("async def run_culinary_judge")
    win = _GO[i:i + 6000]
    assert '"recipe"' in win, (
        "el input del juez DEBE incluir los pasos de receta — es el único ojo "
        "LLM que los ve (el reviewer médico recibe solo nombre+ingredientes)")


def test_fail_open_devuelve_none():
    i = _GO.index("async def run_culinary_judge")
    win = _GO[i:i + 6000]
    assert "return None" in win and "asyncio.wait_for" in win


def test_schema_tipos_canonicos():
    assert "class CulinaryJudgeReport" in _GO
    for t in ("combo_absurdo", "tecnica_impropia", "paso_incoherente",
              "slot_inapropiado", "nombre_no_corresponde"):
        assert t in _GO, f"tipo canónico ausente del schema: {t}"


def test_culinary_judge_guard_sanea_valor_invalido_a_off():
    """Un valor fuera de {off,warn,block} debe caer a 'off' — NO a 'warn'. El juez es una
    llamada LLM completa por plan (a diferencia del contract-guard determinista, que es
    gratis), así que el único default seguro ante un valor raro es apagado."""
    m = re.search(
        r'if CULINARY_JUDGE_GUARD not in \("off", "warn", "block"\):\s*\n\s*'
        r'CULINARY_JUDGE_GUARD = "(\w+)"',
        _GO,
    )
    assert m and m.group(1) == "off", "valor de MEALFIT_CULINARY_JUDGE_GUARD fuera de rango debe sanear a 'off'"


def test_rubrica_existe_no_vacia_y_estable():
    """La rúbrica se construye UNA vez a nivel de módulo (`_CULINARY_JUDGE_RUBRIC`) para que
    el prefix del prompt sea byte-a-byte idéntico entre invocaciones (cache hits de DeepSeek).
    Dos accesos al atributo del módulo deben devolver el mismo valor — si un refactor futuro lo
    convirtiera en una función que reconstruye el string en cada llamada (p.ej. leyendo el JSON
    de disco de nuevo, o barajando ejemplos), este test lo detectaría."""
    import graph_orchestrator as go

    rubric = go._CULINARY_JUDGE_RUBRIC
    assert isinstance(rubric, str)
    assert len(rubric) > 500, "la rúbrica no debería ser un stub trivial (~1-2k tokens esperados)"
    # Dos accesos al MISMO atributo de módulo: mismo valor (y mismo objeto, al ser un string
    # module-level construido una sola vez a import-time).
    assert go._CULINARY_JUDGE_RUBRIC == rubric
    assert go._CULINARY_JUDGE_RUBRIC is rubric
    # Las 5 definiciones canónicas deben vivir DENTRO de la rúbrica (no solo en el schema) —
    # de lo contrario el LLM nunca recibe la explicación de qué significa cada `tipo`.
    for t in ("combo_absurdo", "tecnica_impropia", "paso_incoherente",
              "slot_inapropiado", "nombre_no_corresponde"):
        assert t in rubric, f"tipo canónico ausente de la RÚBRICA (prompt): {t}"
    assert "violaciones CLARAS" in rubric or "violación" in rubric.lower()


# ============================================================
# SECCIÓN INTEGRACIÓN (Task 12) — `run_culinary_judge` conectado a
# `review_plan_node` + telemetría `_culinary_judge_history`. El juez sigue OFF
# por knob (`CULINARY_JUDGE_GUARD`, T11): con el default, este bloque nuevo es
# un no-op perfecto (el `if CULINARY_JUDGE_GUARD != "off":` nunca entra).
#
# NOTA sobre el tamaño de ventana: el brief original de Task 12 usaba
# `_GO[i:i + 80000]` (pseudocódigo), pero `review_plan_node` mide ~134k chars
# de punta a punta — un window de 80000 corta ANTES del punto de inserción
# real (junto al scan T6, que vive cerca del final del nodo) y daría falsos
# negativos contra una integración correcta. Se usa el límite exacto del nodo
# (`def should_retry`, que es la siguiente función top-level) en su lugar.
#
# [P1-COUNTRY-SYSTEM-F1 · 2026-08-16] `run_culinary_judge` ganó un 2º parámetro
# (`country`, default 'DO' — F1-T3) para poder pasarle el país al armar la rúbrica. El
# callsite productivo pasó de `run_culinary_judge(plan)` a `run_culinary_judge(plan,
# _cj_country)` — los anchors de abajo que buscaban el literal `"run_culinary_judge(plan)"`
# (cierre exacto de paréntesis) se ensancharon a `"run_culinary_judge(plan, "` (con coma):
# sigue probando que `plan` es el 1er argumento Y ahora exige explícitamente que se pase un
# 2º argumento, sin fijarse en su nombre.
# ============================================================

_REVIEW_NODE_START = _GO.index("async def review_plan_node")
_REVIEW_NODE_END = _GO.index("def should_retry")


def test_review_invoca_al_juez_en_paralelo_o_secuencial():
    win = _GO[_REVIEW_NODE_START:_REVIEW_NODE_END]
    assert "run_culinary_judge(" in win


def test_should_retry_no_lee_al_juez_directo():
    """Mismo aislamiento que `_shopping_coherence_block`: should_retry solo ve
    issues/severity traducidos, jamás el reporte crudo del juez."""
    i = _GO.index("def should_retry")
    win = _GO[i:i + 30000]
    assert "culinary_judge" not in win


def test_history_append():
    win = _GO[_REVIEW_NODE_START:_REVIEW_NODE_END]
    assert "_culinary_judge_history" in win


def test_juez_corre_despues_del_scan_deterministico_capa1():
    """Orden medir→juzgar: el bloque del juez LLM (Task 12) debe aparecer
    DESPUÉS del bloque del scan culinario determinista (Task 6,
    P1-CULINARY-CONTRACT) dentro de `review_plan_node` — la capa barata
    (regex/metadata) mide primero; el juez LLM añade juicio holístico encima
    del residuo, nunca al revés."""
    win = _GO[_REVIEW_NODE_START:_REVIEW_NODE_END]
    i_scan = win.index("Scan culinario determinista")
    i_judge = win.index("run_culinary_judge(plan, ")
    assert i_judge > i_scan, "el bloque del juez debe vivir DESPUÉS del scan determinista de capa 1"


def test_juez_corre_tambien_en_rama_bypass_sin_return_intermedio():
    """El reviewer médico tiene una rama bypass ('Sin restricciones
    declaradas') que se salta el LLM reviewer + fact-checker. El juez debe
    correr en AMBAS ramas — para eso el bloque de integración debe vivir
    DESPUÉS de que el if/else bypass-vs-reviewer converja, sin ningún
    `return` intermedio que pueda saltárselo. Ancla contra regresión: si un
    futuro refactor mete un `return` entre el bypass y el bloque del juez,
    la rama bypass dejaría de tener cualquier ojo LLM sobre el plan."""
    j = _GO.index("run_culinary_judge(plan, ", _REVIEW_NODE_START)
    i_bypass = _GO.index("Sin restricciones declaradas", _REVIEW_NODE_START)
    assert _REVIEW_NODE_START < i_bypass < j, "el bloque del juez debe vivir después del gate bypass"
    between = _GO[i_bypass:j]
    # Ningún `return` de nodo (dict literal de LangGraph) entre el bypass y el juez.
    assert not re.search(r"^\s{4,8}return\s", between, re.MULTILINE), (
        "un `return` entre el bypass y el bloque del juez dejaría la rama bypass "
        "sin juicio culinario LLM"
    )


def test_run_culinary_judge_tiene_un_solo_callsite_productivo():
    """Fuera de su propia definición, `run_culinary_judge(` debe tener
    EXACTAMENTE 1 callsite productivo (dentro de `review_plan_node`) — un solo
    punto de inserción, después del merge bypass/reviewer, cubre ambas ramas
    (ver `test_juez_corre_tambien_en_rama_bypass_sin_return_intermedio`), así
    que NO hace falta duplicar la llamada. Si un futuro refactor sí la
    duplica, este test debe subir a 2 y documentar la razón en el comentario
    `[P1-CULINARY-JUDGE]` del bloque de integración."""
    matches = list(re.finditer(r"(async def )?run_culinary_judge\(", _GO))
    callsites = [m for m in matches if not m.group(1)]
    assert len(callsites) == 1, (
        f"se esperaba 1 callsite productivo de run_culinary_judge(, hallados {len(callsites)}"
    )


def test_history_entry_shape_y_action_taken_canonico():
    """El history entry sigue el pattern `_shopping_coherence_block_history`:
    ts/model/violations/action_taken, con action_taken restringido a los
    valores canónicos que Task 12 puede escribir en runtime (`warn_only` /
    `blocked` — `off` nunca se persiste porque el bloque entero está gateado
    por `if CULINARY_JUDGE_GUARD != "off":`)."""
    win = _GO[_REVIEW_NODE_START:_REVIEW_NODE_END]
    j = win.index("run_culinary_judge(plan, ")
    block = win[j:j + 1200]
    assert '"ts":' in block and "datetime.now(timezone.utc).isoformat()" in block
    assert '"model": CULINARY_JUDGE_MODEL' in block
    assert '"violations": _cj_viol' in block
    assert '"action_taken":' in block
    assert '"blocked"' in block and '"warn_only"' in block


def test_violations_pydantic_se_serializan_con_model_dump():
    """`CulinaryViolation` es pydantic — el history es JSON-safe (persistido en
    `plan_data`), así que las violaciones DEBEN pasar por `.model_dump()`
    antes de entrar al history, nunca los objetos pydantic crudos."""
    win = _GO[_REVIEW_NODE_START:_REVIEW_NODE_END]
    assert ".model_dump() for v in" in win


def test_juez_no_op_perfecto_con_default_off():
    """Con el default (`CULINARY_JUDGE_GUARD == "off"`), el bloque de
    integración completo — incluida la llamada a `run_culinary_judge` — vive
    detrás de un solo gate `if CULINARY_JUDGE_GUARD != "off":`. Además,
    `run_culinary_judge` en sí mismo retorna `None` de inmediato cuando el
    guard está OFF (T11) — doble cinturón: aunque algo invocara la función
    directamente fuera de este gate, no golpearía al LLM."""
    win = _GO[_REVIEW_NODE_START:_REVIEW_NODE_END]
    j = win.index("run_culinary_judge(plan, ")
    # La línea del callsite (o la inmediatamente anterior) debe estar dentro
    # de un bloque gateado por el knob.
    pre = win[max(0, j - 400):j]
    assert 'if CULINARY_JUDGE_GUARD != "off":' in pre
    # Y `run_culinary_judge` en sí es fail-safe ante OFF (cinturón + tirantes).
    k = _GO.index("async def run_culinary_judge")
    body = _GO[k:k + 2000]
    assert 'if CULINARY_JUDGE_GUARD == "off":' in body and "return None" in body
