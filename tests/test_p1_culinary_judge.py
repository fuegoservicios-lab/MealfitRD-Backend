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
