# prompts/__init__.py
"""
Paquete de Prompts de Bioboros.
Re-exporta TODAS las constantes con los mismos nombres originales
para mantener retrocompatibilidad con `from prompts import X`.
"""

# --- Plan Generator ---
from prompts.plan_generator import GENERATOR_SYSTEM_PROMPT

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
)
