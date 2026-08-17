# prompts/__init__.py
"""
Paquete de Prompts de Bioboros.
Re-exporta TODAS las constantes con los mismos nombres originales
para mantener retrocompatibilidad con `from prompts import X`.
"""

# --- Plan Generator ---
from prompts.plan_generator import GENERATOR_SYSTEM_PROMPT

# --- Planner ---
# [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (FINAL-FIX F1a)] build_planner_system_prompt: T2 pattern
# (país-aware render de PLANNER_SYSTEM_PROMPT). PLANNER_SYSTEM_PROMPT en sí NO se re-exportaba
# aquí antes de esta fila — graph_orchestrator.py la importa directo de prompts.planner.
from prompts.planner import build_planner_system_prompt

# --- Medical Reviewer ---
from prompts.medical_reviewer import REVIEWER_SYSTEM_PROMPT

# --- Chat Agent ---
from prompts.chat_agent import (
    CHAT_SYSTEM_PROMPT_BASE,
    CHAT_STREAM_SYSTEM_PROMPT_BASE,
    RAG_ROUTER_PROMPT,
    TITLE_GENERATION_PROMPT,
)

# --- Preferences ---
from prompts.preferences import (
    PREFERENCES_AGENT_PROMPT,
    DETERMINISTIC_VARIETY_PROMPT,
    # [P2-SEEDER-DAYS-COUNT · 2026-08-03] La plantilla parametrizada por días del chunk.
    # `DETERMINISTIC_VARIETY_PROMPT` es su instancia de 3 días y sigue exportándose.
    build_deterministic_variety_prompt,
    # SSOT de la etiqueta del día (0→A). `ai_helpers` la usa para el ancla liviana: dos tablas
    # de letras divergen en el primer cambio.
    option_letter,
)

# --- Meal Operations ---
from prompts.meal_operations import (
    SWAP_MEAL_PROMPT_TEMPLATE,
    MODIFY_MEAL_PROMPT_TEMPLATE,
    RECIPE_EXPANSION_PROMPT,
    # [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (FINAL-FIX F1c)] T2 pattern país-aware. Los templates de
    # arriba siguen exportados intactos (byte-idénticos, consumidos directo por callers legacy);
    # estos builders son el path país-consciente que agent.py::swap_meal / tools.py::
    # execute_modify_single_meal deben usar en su lugar.
    build_swap_meal_prompt_template,
    build_modify_meal_prompt_template,
)
