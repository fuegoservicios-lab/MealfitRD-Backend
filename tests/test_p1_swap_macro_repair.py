# [P1-SWAP-MACRO-REPAIR · 2026-08-09] El swap le pedía al LLM aritmética de porciones
# multi-restricción — la tarea en la que es MALO y el motor determinista es BUENO. Medido
# (corr=78a438e0, run 31311796944): target carbs 146g, el LLM propuso 110→151→269→332→344→422g
# (cada retry PEOR — la espiral «añade más arroz»), 8/14 swaps muertos en
# SWAP_LLM_RETRIES_EXHAUSTED tras 73-117s. En el intento 2 estuvo a 3,4% en carbs con proteína
# +43%: un escalado determinista de las líneas dominantes lo habría aprobado en milisegundos.
# El fix: ante drift de macros, ANTES de quemar un retry LLM, reparar el candidato con
# `_rebalance_day_macros_to_target` (la MISMA maquinaria que la generación usa hace meses) +
# truth-up + step-sync, y re-validar. La identidad del plato (lo que el LLM hace bien) se
# conserva; las porciones (lo que hace mal) las pone el motor.
import copy
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("MEALFIT_DB_BACKEND", "neon")
os.environ.setdefault("NEON_DATABASE_URL", "postgresql://stub:stub@localhost:5432/stub")
os.environ.setdefault("NEON_DATABASE_URL_UNPOOLED", "postgresql://stub:stub@localhost:5432/stub")

# El venv-test rompe `ChatGLM.__init__` (drift langchain/pydantic) y agent.py lo
# INSTANCIA al importar; además, en la suite completa otros tests dejan un `agent` FALSO
# cacheado en sys.modules (18ª clase: pasa solo, falla en suite). La 1ª versión de este
# arreglo purgaba sys.modules a nivel MÓDULO sin restaurar → re-ejecutó agent con el stub
# para TODA la suite posterior (contaminación reload(): ~600 tests cambiaron de veredicto,
# tree9=113 vs ~717). El contrato correcto: stub + import fresco en setup_module, y
# RESTAURACIÓN COMPLETA del mundo (llm_provider.ChatGLM + sys.modules['agent'] previo,
# fuera FALSO o ausente) en teardown_module — los tests siguientes ven exactamente lo que
# habrían visto sin este archivo.
import importlib  # noqa: E402

import llm_provider as _lp  # noqa: E402


class _StubLLM:
    def __init__(self, *a, **k):
        pass

    def bind_tools(self, *a, **k):
        return self

    def with_structured_output(self, *a, **k):
        return self


ag = None  # asignado en setup_module (import fresco del agent REAL con el stub activo)
_saved = {}


def setup_module(module):
    global ag
    _saved["agent"] = sys.modules.get("agent")
    _saved["llm"] = getattr(_lp, "ChatGLM", None)
    _lp.ChatGLM = _StubLLM
    sys.modules.pop("agent", None)
    ag = importlib.import_module("agent")
    module.ag = ag
    assert hasattr(ag, "_repair_swap_candidate_macros"), (
        "el import fresco no entregó el agent real — revisar el entorno")


def teardown_module(module):
    if _saved.get("llm") is not None:
        _lp.ChatGLM = _saved["llm"]
    if _saved.get("agent") is not None:
        sys.modules["agent"] = _saved["agent"]
    else:
        sys.modules.pop("agent", None)

_AGENT_SRC = open(os.path.join(os.path.dirname(__file__), "..", "agent.py"),
                  encoding="utf-8").read()

# Tabla mínima por-100g (macros reales aproximados) — el FakeDB solo necesita el método
# que la maquinaria consulta (`macros_from_ingredient_string`); rescale/quantize del
# rebalanceador son funciones PURAS de string en nutrition_db (corren de verdad).
_TABLE = {
    "arroz blanco cocido": {"protein": 2.7, "carbs": 28.0, "fats": 0.3},
    "pechuga de pollo": {"protein": 31.0, "carbs": 0.0, "fats": 3.6},
}


class _FakeDB:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g\s+de\s+(.+)", str(s).strip(), re.IGNORECASE)
        if not m:
            return None
        grams, food = float(m.group(1)), m.group(2).strip().lower()
        row = _TABLE.get(food)
        if not row:
            return None
        f = grams / 100.0
        out = {k: v * f for k, v in row.items()}
        out["cals"] = 4 * out["protein"] + 4 * out["carbs"] + 9 * out["fats"]
        return out


def _meal_inflado():
    # el caso medido en miniatura: carbs al DOBLE del target, proteína en target.
    return {
        "name": "Pollo con arroz",
        "meal": "Almuerzo",
        "ingredients": ["300 g de arroz blanco cocido", "150 g de pechuga de pollo"],
        "protein": 51, "carbs": 84, "fats": 6,
        "cals": 4 * 51 + 4 * 84 + 9 * 6,
        "recipe": ["Cocina el arroz.", "Haz la pechuga a la plancha."],
    }


_TARGETS = {"cals": 4 * 51 + 4 * 42 + 9 * 6, "protein": 51, "carbs": 42, "fats": 6}


def test_repara_el_caso_medido_sin_retry_llm():
    meal = _meal_inflado()
    passed, drifts, summary = ag._repair_swap_candidate_macros(meal, dict(_TARGETS), _FakeDB())
    assert passed, f"el motor debe re-porcionar el candidato a banda (drifts={drifts})"
    # la línea carbo-dominante se ESCALÓ (identidad intacta, porción del motor)
    arroz = next(s for s in meal["ingredients"] if "arroz" in s)
    g = float(re.match(r"\s*(\d+(?:\.\d+)?)", arroz).group(1))
    assert g < 300, "el arroz debía bajar hacia el target (~150 g)"
    # la proteína (ya en target) quedó esencialmente intacta
    pollo = next(s for s in meal["ingredients"] if "pechuga" in s)
    gp = float(re.match(r"\s*(\d+(?:\.\d+)?)", pollo).group(1))
    assert 120 <= gp <= 180


def test_sin_lineas_movibles_no_finge_exito():
    meal = {"name": "X", "ingredients": [], "protein": 10, "carbs": 200, "fats": 5, "cals": 885}
    passed, drifts, summary = ag._repair_swap_candidate_macros(meal, dict(_TARGETS), _FakeDB())
    assert passed is False, "sin palanca el repair debe declarar fallo, no éxito"


def test_no_inventa_ingredientes():
    meal = _meal_inflado()
    before = {re.sub(r"^[\d\.\s]+g\s+de\s+", "", s) for s in meal["ingredients"]}
    ag._repair_swap_candidate_macros(meal, dict(_TARGETS), _FakeDB())
    after = {re.sub(r"^[\d\.\s]+g\s+de\s+", "", s) for s in meal["ingredients"]}
    assert after == before, "el repair escala PORCIONES — jamás añade/quita alimentos"


def test_wiring_en_el_validador_antes_del_raise():
    # Estructural: dentro del branch `if not passed:` del validador de macros del swap,
    # el repair corre ANTES de inyectar el summary al prompt y del raise que quema el retry.
    i_val = _AGENT_SRC.index("Drift detectado attempt-pending")
    i_raise = _AGENT_SRC.index("raise ValueError(summary)", i_val)
    win = _AGENT_SRC[i_val:i_raise]
    assert "_repair_swap_candidate_macros(" in win, (
        "el repair determinista debe intentarse ANTES de quemar un retry LLM — sin él, "
        "8/14 swaps murieron en la espiral de porciones (run 31311796944)"
    )
    assert "_swap_macro_repair_enabled()" in win, "kill switch obligatorio"


def test_knob_default_on():
    assert 'MEALFIT_SWAP_MACRO_REPAIR' in _AGENT_SRC
    assert ag._swap_macro_repair_enabled() is True


def test_chat_modify_tiene_el_mismo_repair():
    # tools.execute_modify_single_meal usa el MISMO validador y tenía la MISMA espiral —
    # el repair debe correr también ahí, antes de su raise.
    src = open(os.path.join(os.path.dirname(__file__), "..", "tools.py"),
               encoding="utf-8").read()
    i_val = src.index("Drift en modify_meal")
    i_raise = src.index("raise ValueError(summary)", i_val)
    win = src[i_val:i_raise]
    assert "_repair_swap_candidate_macros" in win or "_mr_repair(" in win, (
        "la espiral del chat-modify es la misma que la del swap — mismo repair"
    )
    assert "_swap_macro_repair_enabled" in win or "_mr_enabled()" in win


def test_writeback_a_res_incluye_los_campos_honestos():
    # el write-back (mismo patrón probado de P2-SWAP-FATS-TRIM) debe propagar ingredientes
    # y macros reparados a `res` — sin esto el repair es cosmético.
    i = _AGENT_SRC.index("_mr_passed, _mr_drifts, _mr_summary = _repair_swap_candidate_macros(")
    win = _AGENT_SRC[i:i + 2200]
    for fk in ("ingredients", "protein", "carbs", "fats", "cals", "macros"):
        assert f'"{fk}"' in win, f"write-back debe cubrir '{fk}'"
