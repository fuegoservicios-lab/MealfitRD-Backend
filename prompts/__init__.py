# prompts/__init__.py
"""
Paquete de Prompts de MealfitRD.
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
)

# --- Meal Operations ---
from prompts.meal_operations import (
    SWAP_MEAL_PROMPT_TEMPLATE,
    MODIFY_MEAL_PROMPT_TEMPLATE,
    RECIPE_EXPANSION_PROMPT,
)
