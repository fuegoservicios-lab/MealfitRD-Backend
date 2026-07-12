"""[P1-CONSUMED-BACKDATE · 2026-07-12] "El almuerzo de AYER" no contamina las macros de HOY.

Vivo (owner): subió foto de un plato al chat, el agente preguntó de cuándo era,
respondió "es del almuerzo de ayer" — y la tool lo registró con NOW() → el card
Progreso en Tiempo Real mostró 1600 kcal de HOY con una comida de ayer dentro.

Fix en 3 capas:
  1. tools.log_consumed_meal: `days_ago` (clamp [0,3], garbage→0 — el diario
     nunca registra a futuro) + `meal_type` normalizado + guard de duplicado
     por comida PRINCIPAL/día local RD (una sola desayuno/almuerzo/cena por
     día; merienda/snack se repiten legítimamente) con `force` para override
     confirmado por el usuario.
  2. db_facts.log_consumed_meal: `consumed_at_override` aplica la fecha real.
  3. Prompts (2 builders + instrucción de foto en AgentPage) enseñan al agente
     a extraer el día y el tipo de la conversación.
tooltip-anchor: P1-CONSUMED-BACKDATE
"""
import inspect
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BACKEND)

from tools import _clamp_days_ago, _normalize_meal_type, log_consumed_meal  # noqa: E402
import db_facts  # noqa: E402


def test_clamp_days_ago_never_future_never_far_past():
    assert _clamp_days_ago(0) == 0
    assert _clamp_days_ago(1) == 1
    assert _clamp_days_ago(3) == 3
    assert _clamp_days_ago(9) == 3, "máximo 3 días atrás"
    assert _clamp_days_ago(-1) == 0, "el diario nunca registra a futuro"
    assert _clamp_days_ago("ayer") == 0, "garbage del LLM → hoy"
    assert _clamp_days_ago(None) == 0


def test_normalize_meal_type():
    assert _normalize_meal_type("Almuerzo") == "almuerzo"
    assert _normalize_meal_type("CENA") == "cena"
    assert _normalize_meal_type("brunch") == "snack", "tipo desconocido → snack"
    assert _normalize_meal_type(None) == "snack"


def test_tool_signature_has_backdate_params():
    # `log_consumed_meal` es un StructuredTool — la signature viene del schema.
    params = set(log_consumed_meal.args.keys())
    assert {"meal_type", "days_ago", "force"} <= params, \
        "el LLM necesita los 3 params en el schema de la tool"


def test_db_layer_accepts_consumed_at_override():
    sig = inspect.signature(db_facts.log_consumed_meal)
    assert "consumed_at_override" in sig.parameters
    assert sig.parameters["consumed_at_override"].default is None, \
        "default None: callers existentes (diary.py POST /consumed) intactos"


def test_dup_guard_scoped_to_main_meals_local_date():
    with open(os.path.join(_BACKEND, "tools.py"), encoding="utf-8") as f:
        src = f.read()
    i = src.find("_CONSUMED_MAIN_MEAL_TYPES")
    assert i != -1
    assert '("desayuno", "almuerzo", "cena")' in src[i:i + 120], \
        "solo comidas PRINCIPALES: merienda/snack se repiten legítimamente"
    assert "AT TIME ZONE 'America/Santo_Domingo'" in src, \
        "el 'mismo día' es el día LOCAL RD (UTC-4), no el día UTC del server"
    assert "repite esta herramienta con force=true" in src, \
        "el guard devuelve instrucción de confirmación, no un error mudo"


def test_prompts_teach_backdate_to_agent():
    with open(os.path.join(_BACKEND, "prompts", "chat_agent.py"), encoding="utf-8") as f:
        prompts = f.read()
    assert prompts.count("P1-CONSUMED-BACKDATE") >= 2, \
        "ambos builders (inline y stream) deben enseñar days_ago/meal_type/force"
    assert "days_ago" in prompts and "force=true" in prompts

    with open(os.path.join(_ROOT, "frontend", "src", "pages", "AgentPage.jsx"),
              encoding="utf-8") as f:
        ap = f.read()
    assert "days_ago (1=ayer)" in ap, \
        "la instrucción de foto del chat menciona el registro en otro día"
