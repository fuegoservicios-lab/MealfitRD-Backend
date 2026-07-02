import os
import json
import random
import logging
import hashlib
from typing import Optional, List, Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt
import re
from datetime import datetime, timezone
import unicodedata
import concurrent.futures

# Prompts
from prompts import (
    TITLE_GENERATION_PROMPT,
    DETERMINISTIC_VARIETY_PROMPT,
    RECIPE_EXPANSION_PROMPT
)

# Langchain
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] Gemini → DeepSeek.
from llm_provider import ChatDeepSeek, DEEPSEEK_FLASH, model_free_tier
from schemas import ExpandedRecipeModel

from constants import (
    strip_accents,
    DOMINICAN_PROTEINS, DOMINICAN_CARBS, DOMINICAN_VEGGIES_FATS, DOMINICAN_FRUITS,
    PROTEIN_SYNONYMS as protein_synonyms,
    CARB_SYNONYMS as carb_synonyms,
    VEGGIE_FAT_SYNONYMS as veggie_fat_synonyms,
    FRUIT_SYNONYMS as fruit_synonyms,
    _get_fast_filtered_catalogs
)
from db import get_user_profile, update_user_health_profile, update_user_health_profile_atomic, get_user_ingredient_frequencies
from cpu_tasks import _calcular_frecuencias_regex_cpu_bound
from knobs import _env_str, _env_float, _env_bool  # [P3-FLASH-LITE-COST-CUT · 2026-05-21] / [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] / [P3-GAINMUSCLE-PROTEIN-DENSITY · 2026-06-23]

logger = logging.getLogger(__name__)


# [P3-FLASH-LITE-COST-CUT · 2026-05-21] Knob para overridear el modelo del
# generador de títulos de plan sin redeploy (convención P3-PREVIEW-MODEL-KNOB).
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] Default = DeepSeek V4 Flash (tarea aux
# barata, mismo modelo para todos los tiers).
# Tooltip-anchor: P3-FLASH-LITE-COST-CUT.
def _plan_title_model_name() -> str:
    return _env_str("MEALFIT_PLAN_TITLE_MODEL", DEEPSEEK_FLASH)


# [P1-RECIPE-EXPAND-FAILSIGNAL · 2026-05-30] Knob para overridear el modelo del
# "Chef AI" (`expand_recipe_agent`) sin redeploy — mismo patrón que
# `_plan_title_model_name` (P3-FLASH-LITE-COST-CUT). [P0-DEEPSEEK-MIGRATION]
# Default = DeepSeek V4 Flash (expansión de receta es relleno de schema).
# Tooltip-anchor: P1-RECIPE-EXPAND-FAILSIGNAL-MODEL.
def _recipe_expand_model_name() -> str:
    return _env_str("MEALFIT_RECIPE_EXPAND_MODEL", DEEPSEEK_FLASH)


# [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] Timeout per-invoke compartido por los 4
# constructores `ChatGoogleGenerativeAI` de este módulo: `generate_plan_title`
# (callsite síncrono en services.py post-save del plan), `expand_recipe_agent`
# (endpoint síncrono api_expand_recipe), `generate_llm_retrospective` y
# `extract_liked_flavor_profiles` (corren en el thread del chunk-worker /
# nightly cron via _persist_nightly_learning_signals). Pre-fix: ninguno pasaba
# `timeout=`, así que un Gemini colgado bloqueaba indefinidamente el thread del
# threadpool de FastAPI (title/recipe) o el thread del cron (retrospectiva), con
# `max_retries` default del SDK (5) que NO avanza sobre sockets colgados. El
# `timeout=` propaga al deadline gRPC → DeadlineExceeded, capturado por los
# `except Exception` existentes (degradan a fallback determinístico). Default
# 30s; clamp (0, 120]. Knob auto-registrado. Tooltip-anchor: P2-LLM-TIMEOUT-SWEEP.
def _ai_helpers_llm_timeout_s() -> float:
    return _env_float(
        "MEALFIT_AI_HELPERS_LLM_TIMEOUT_S",
        30.0,
        validator=lambda v: 0.0 < v <= 120.0,
    )


def generate_plan_title(plan_data: dict) -> str:
    """Genera un título corto y creativo para un plan nutricional (modelo aux barato)."""
    try:
        # Extraer nombres de comidas para contexto
        meal_names = []
        for d in plan_data.get("days", []):
            for m in d.get("meals", []):
                if m.get("name"):
                    meal_names.append(m["name"])
        
        calories = plan_data.get("calories", 0)
        goal = plan_data.get("goal", plan_data.get("assessment", {}).get("mainGoal", ""))
        
        if not meal_names:
            return f"Plan Evolutivo - {datetime.now().strftime('%d/%m/%Y')}"
        
        meals_summary = ", ".join(meal_names[:6])
        
        goal_map = {
            "lose_weight": "pérdida de grasa",
            "build_muscle": "ganar masa muscular",
            "maintain": "mantenimiento",
            "health": "salud general"
        }
        goal_text = goal_map.get(goal, "nutrición personalizada")
        
        prompt = f"""Genera UN título corto y creativo en español para un plan de comidas. 
REGLAS ESTRICTAS:
- Máximo 5-6 palabras
- Debe sonar motivador, atractivo y premium
- NO incluir calorías, números ni emojis
- NO usar la palabra "Plan" sola
- Puede ser metafórico o usar referencias dominicanas sutiles
- Ejemplos de buenos títulos: "Energía Tropical al Máximo", "Sabor Sin Culpa", "Fuerza y Balance Criollo", "Combustible Para Tu Meta", "Ruta Fit Dominicana", "Poder Verde y Proteína"

Contexto:
- Objetivo: {goal_text}
- Calorías: {calories} kcal
- Platos incluidos: {meals_summary}

Responde SOLO con el título, nada más."""
        
        # [P3-FLASH-LITE-COST-CUT · 2026-05-21] Model via knob (P3-PREVIEW-MODEL-KNOB).
        title_llm = ChatDeepSeek(
            model=_plan_title_model_name(),
            temperature=0.9,
            timeout=_ai_helpers_llm_timeout_s(),  # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30]
        )
        response = title_llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = " ".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])
        title = str(content).replace('"', '').replace("'", "").strip()
        
        # Validar que no sea absurdamente largo
        if len(title) > 50 or len(title) < 3:
            raise ValueError(f"Título inválido: '{title}'")
        
        logger.info(f"✨ [PLAN TITLE] Título creativo generado: {title}")
        return title
        
    except Exception as e:
        logger.error(f"⚠️ [PLAN TITLE] Error generando título creativo, usando fallback: {e}")
        # Fallback determinista
        first_meal = meal_names[0] if meal_names else "Plan Personalizado"
        short_name = first_meal[:20] + "…" if len(first_meal) > 20 else first_meal
        return f"{short_name} — {calories} kcal"


def _apply_recency_fatigue(freq_map, user_id):
    """Ingredientes usados recientemente pesan más que los usados hace 2 semanas."""
    if not freq_map or not user_id or user_id == "guest":
        return freq_map
        
    try:
        # Query: ingredientes de los últimos 3 días pesan x3, últimos 7 días pesan x1.5
        recent_3d = get_user_ingredient_frequencies(user_id, days_limit=3)
        recent_7d = get_user_ingredient_frequencies(user_id, days_limit=7)
        
        fatigued = {}
        for ing, freq in freq_map.items():
            recent_boost = recent_3d.get(ing, 0) * 3.0 + recent_7d.get(ing, 0) * 1.5
            fatigued[ing] = freq + recent_boost
            
        return fatigued
    except Exception as e:
        logger.warning(f"⚠️ [FATIGUE] Error aplicando fatiga temporal: {e}")
        return freq_map


# [P3-GAINMUSCLE-PROTEIN-DENSITY · 2026-06-23 · elevado a módulo P2-9 · 2026-06-23] Proteínas de BAJA
# densidad que NO deben usarse como proteína PRINCIPAL en gain_muscle (piso de proteína alto). Set
# EXPLÍCITO (NO reusar LEGUME_NAMES — omite "habichuelas blancas", cazado por el test trial 7). Elevado
# a nivel módulo para que las superficies de UPDATE (swap_meal, audit inteligencia P2-9) reusen el MISMO
# set que el esqueleto de S1 (get_deterministic_variety_prompt) — un swap/regenerate-day de gain_muscle
# ya no elige Ricotta/Habichuelas/Gandules como main. tooltip-anchor: P2-9-GAINMUSCLE-MAINS
_LOW_DENSITY_AS_MAIN = {
    "habichuelas rojas", "habichuelas negras", "habichuelas blancas",
    "gandules", "lentejas", "garbanzos",
    "queso ricotta", "queso cottage", "queso crema",
    "yogurt",  # regular ~4g prot/100g (NO "yogurt griego" — ése es alto en proteína, exact-match)
}

# [P1-BARIATRIC-DENSE-ANCHOR · 2026-06-28] Quesos de RELLENO / alto-grasa / bajo-valor-proteico-por-porción que el pouch
# bariátrico NO debe usar como proteína PRINCIPAL (corr=3b318e57: el LLM ancló en "Salteado de Queso de Freír" — relleno
# + frito — que el swap NO cazaba porque no estaba en _LOW_DENSITY_AS_MAIN). Se UNE al set global SOLO si _is_bariatric
# (NO global: gain_muscle puede usar queso como main legítimamente). Calibrado por review CLÍNICA adversaria (ASMBS):
# EXCLUYE solo los quesos pobres-como-ancla; deliberadamente NO incluye cottage/ricotta/yogurt griego/CLARAS — esas son
# anclas LEGÍTIMAS post-bariátricas (húmedas, densas en proteína, mejor toleradas que el pollo seco) y NO se degradan
# aquí. Nombres EXACTOS del catálogo (constants.py DOMINICAN_PROTEINS) en minúscula+strip_accents (exact-match).
# tooltip-anchor: P1-BARIATRIC-DENSE-ANCHOR
_BARIATRIC_LOW_DENSITY_AS_MAIN = {
    "queso de freir", "queso blanco", "queso mozzarella",
    "queso de hoja", "queso parmesano", "queso cheddar", "queso gouda",
}


def get_deterministic_variety_prompt(history_text: str, form_data: dict = None, user_id: str = None, rejection_reasons: list = None) -> str:
    """Implementa Inversión de Control Determinista para evitar Mode Collapse en el LLM."""
    logger.debug("🎲 [ANTI MODE-COLLAPSE] Calculando Matriz de Ingredientes (Round-Robin)...")
    history_lower = history_text.lower() if history_text else ""
    history_normalized = strip_accents(history_lower)
    force_variety = bool(form_data.get("_force_variety")) if form_data else False
    
    # --- FILTRO DE RESTRICCIONES MÉDICAS Y DIETÉTICAS ---
    if form_data:
        allergies = tuple([a.lower() for a in form_data.get("allergies", [])])
        
        dislikes_list = [d.lower() for d in form_data.get("dislikes", [])]
        temp_dislikes = form_data.get("temporary_dislikes", {})
        if isinstance(temp_dislikes, dict):
            now = datetime.now(timezone.utc)
            for item, expiry_iso in temp_dislikes.items():
                try:
                    from constants import safe_fromisoformat
                    expiry_dt = safe_fromisoformat(expiry_iso)
                    if expiry_dt.tzinfo is None:
                        expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
                    if now < expiry_dt:
                        dislikes_list.append(item.lower())
                except Exception as _dislike_exc:
                    # [P2-SILENT-DEGRADATION · 2026-05-13] ISO mal-formado / item
                    # corrupto: el dislike temporal se ignora y el item podría
                    # volver al plan. Sin log, un cambio de formato del campo
                    # `temporary_dislikes` o un blip de DB se traduce en "el
                    # usuario marcó X como no-quiero-hoy y reaparece" sin
                    # telemetría. Mantener fallback (no re-raise).
                    logger.debug(
                        "[P2-SILENT-DEGRADATION] temp_dislikes parse falló "
                        "(item=%s): %s: %s",
                        str(item)[:60],
                        type(_dislike_exc).__name__,
                        str(_dislike_exc)[:160],
                    )
        dislikes = tuple(dislikes_list)
        
        diet = form_data.get("diet", form_data.get("dietType", "")).lower()
        
        filtered_proteins, filtered_carbs, filtered_veggies, filtered_fruits = _get_fast_filtered_catalogs(allergies, dislikes, diet)
    else:
        # Guest sin form_data: usar catálogos completos sin filtrar
        filtered_proteins = DOMINICAN_PROTEINS
        filtered_carbs = DOMINICAN_CARBS
        filtered_veggies = DOMINICAN_VEGGIES_FATS
        filtered_fruits = DOMINICAN_FRUITS
    # ----------------------------------------------------
    
    # 1. Analizar qué se ha usado (Optimización O(1) con DB o Fallback a Regex)
    used_proteins = set()
    used_carbs = set()
    used_veggies = set()
    
    protein_freq = {}
    carb_freq = {}
    veggie_freq = {}
    fruit_freq = {}
    
    db_freq_map = {}
    if user_id and user_id != "guest":
        try:
            db_freq_map = get_user_ingredient_frequencies(user_id)
            db_freq_map = _apply_recency_fatigue(db_freq_map, user_id)
        except Exception as e:
            logger.error(f"⚠️ [ANTI MODE-COLLAPSE] Error obteniendo frecuencias de DB: {e}")
            
    if db_freq_map:
        # ======= NUEVO FLUJO OPTIMIZADO O(1) =======
        logger.info(f"⚡ [ANTI MODE-COLLAPSE] Usando Hash Map O(1) de DB con {len(db_freq_map)} métricas pre-calculadas.")
        for p in filtered_proteins:
            syns = protein_synonyms.get(p.lower(), [p.lower()])
            protein_freq[p] = sum(db_freq_map.get(strip_accents(syn.lower()), 0) for syn in syns)
        for c in filtered_carbs:
            syns = carb_synonyms.get(c.lower(), [c.lower()])
            carb_freq[c] = sum(db_freq_map.get(strip_accents(syn.lower()), 0) for syn in syns)
        for v in filtered_veggies:
            syns = veggie_fat_synonyms.get(v.lower(), [v.lower()])
            veggie_freq[v] = sum(db_freq_map.get(strip_accents(syn.lower()), 0) for syn in syns)
        for f in filtered_fruits:
            syns = fruit_synonyms.get(f.lower(), [f.lower()])
            fruit_freq[f] = sum(db_freq_map.get(strip_accents(syn.lower()), 0) for syn in syns)
    else:
        # ======= FALLBACK: Regex en Runtime (O(n×m)) para Invitados =======
        # Truncar historial a los últimos ~5000 chars (~1250 tokens) para proteger de O(N×M) si la sesión guest es larga.
        history_normalized = history_normalized[-5000:] if len(history_normalized) > 5000 else history_normalized
        logger.warning(f"⚠️ [ANTI MODE-COLLAPSE] Fallback Regex en runtime usado para guest o sin historial.")
        
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(
                _calcular_frecuencias_regex_cpu_bound,
                history_normalized,
                filtered_proteins, protein_synonyms,
                filtered_carbs, carb_synonyms,
                filtered_veggies, veggie_fat_synonyms,
                filtered_fruits, fruit_synonyms
            )
            protein_freq, carb_freq, veggie_freq, fruit_freq = future.result()

    # Umbral mínimo: solo considerar "sobreusados" ingredientes con freq >= 3.
    # Con freq=1 o 2 el soft-penalty 1/(freq+1) ya reduce su probabilidad suficientemente;
    # marcarlos como "PROHIBIDOS" en el prompt contradice el modelo de penalización suave.
    OVERUSE_THRESHOLD = 3
    used_proteins = [p for p, freq in protein_freq.items() if freq >= OVERUSE_THRESHOLD]
    used_carbs = [c for c, freq in carb_freq.items() if freq >= OVERUSE_THRESHOLD]
    used_veggies = [v for v, freq in veggie_freq.items() if freq >= OVERUSE_THRESHOLD]
    
    # 2. Construir pools de candidatos con Penalización Suave (Soft Penalty)
    # En vez de un reset total cuando quedan pocos, SIEMPRE usamos toda la lista filtrada
    # pero ponderamos inversamente por frecuencia: 1/(freq+1).
    # Esto evita la desincronización entre available_* y *_freq que causaba contradicciones.

    
    available_proteins = list(filtered_proteins)
    available_carbs = list(filtered_carbs)
    available_veggies = list(filtered_veggies)
    available_fruits = list(filtered_fruits)

    # Guard clause: si las restricciones eliminaron TODOS los ingredientes
    # (ej: vegano con muchas alergias), dejar libertad total al LLM
    if not available_proteins or not available_carbs:
        logger.warning("⚠️ [ANTI MODE-COLLAPSE] No quedan ingredientes disponibles tras filtrar restricciones. Dejando libertad al LLM.")
        return ""
        
    # 3. Restricción para Variedad y Costo: Elegir proteínas y carbohidratos base para rotarlos.
    # Peso inverso: ingredientes menos usados tienen MÁS probabilidad de ser elegidos.
    #
    # 🏷️ FEATURE FLAG: variety_level (ahora expuesto al frontend)
    #   - "standard" (default): 2 proteínas + 2 carbos → optimizado para costo de supermercado.
    #   - "max": 3 proteínas + 3 carbos → máxima variedad (1 distinto por día).
    #   Prioridad: form_data > health_profile en DB > "standard"
    #   Frontend: exponer como toggle en Settings del usuario con key "variety_level".
    variety_level = form_data.get("variety_level", "") if form_data else ""
    
    # Si no viene en form_data, intentar leer del perfil persistido en DB
    if not variety_level and user_id and user_id != "guest":
        try:
            profile = get_user_profile(user_id)
            if profile:
                hp = profile.get("health_profile") or {}
                variety_level = hp.get("variety_level", "standard")
        except Exception as _var_exc:
            # [P2-SILENT-DEGRADATION · 2026-05-13] DB blip / pool exhaustion:
            # variety_level cae al default "standard" sin que SRE vea correlate
            # entre planes con variedad baja y degradación operacional. Log
            # debug permite grep `[P2-SILENT-DEGRADATION] variety_level` para
            # contar incidentes. Fallback intacto (no re-raise).
            logger.debug(
                "[P2-SILENT-DEGRADATION] variety_level profile fetch falló "
                "(user_id=%s): %s: %s",
                str(user_id)[:36],
                type(_var_exc).__name__,
                str(_var_exc)[:160],
            )
    variety_level = variety_level or "standard"
    if force_variety:
        variety_level = "max"
        logger.warning("🎯 [P0-3] _force_variety=true -> elevando variety_level a max para el siguiente chunk.")

    # Auto-promoción a "max" para objetivos que se benefician de mayor diversidad
    # de proteínas. Razón: con 'standard' el sistema elige solo 2 proteínas base y
    # las cicla (P[0], P[1], P[0]) — eso fuerza repetición almuerzo↔cena del mismo
    # día y dispara incoherencias de slot. Para gain_muscle/lose_weight el aporte
    # de aminoácidos completos y la variedad de fuentes importa más que optimizar
    # el costo del supermercado (3 proteínas vs 2 al mes es marginal en costo).
    _GOALS_FORCE_MAX_VARIETY = {"gain_muscle", "lose_weight"}
    _main_goal_for_variety = (form_data.get("mainGoal") or "").strip().lower() if form_data else ""
    # [P1-REVIEWER-TRANSIENT-RETRY · 2026-06-27] (FASE C) Bariátrica también auto-promueve a variety_level=max:
    # 6 comidas pequeñas con solo 2 proteínas base → repetición same-day (el reviewer y el gate same-day-protein
    # lo penalizan); más proteínas distintas = menos monotonía + mejor reparto del piso proteico en volumen pequeño.
    _baria_for_variety = False
    try:
        from constants import BARIATRIC_CONDITION_TERMS as _BT_V, strip_accents as _sa_v
        _cbv = _sa_v(" ".join(str(x) for x in ((form_data.get("medicalConditions") or []) if form_data else []))
                     + " " + str((form_data.get("otherConditions") or "") if form_data else "")).lower()
        _baria_for_variety = any(t in _cbv for t in _BT_V)
    except Exception:
        _baria_for_variety = False
    if variety_level != "max" and (_main_goal_for_variety in _GOALS_FORCE_MAX_VARIETY or _baria_for_variety):
        variety_level = "max"
        _vary_reason = "bariátrica" if _baria_for_variety else f"goal='{_main_goal_for_variety}'"
        logger.info(
            f"🎯 [GOAL VARIETY] Auto-promovido a variety_level=max por {_vary_reason} "
            f"(más proteínas distintas = menos repetición almuerzo↔cena)."
        )
    
    if variety_level == "max":
        num_proteins_to_pick = min(3, len(available_proteins))   # 1 proteína distinta por día
        num_carbs_to_pick = min(3, len(available_carbs))         # 1 carb distinto por día
        num_veggies_to_pick = min(6, len(available_veggies))   # 2 vegetales distintos por día
        logger.info(f"🎯 [ANTI MODE-COLLAPSE] variety_level=max → distribución máxima (3P/3C/3V)")
    else:
        num_proteins_to_pick = min(2, len(available_proteins))
        num_carbs_to_pick = min(2, len(available_carbs))
        num_veggies_to_pick = min(6, len(available_veggies))   # 2 vegetales distintos por día
    num_fruits_to_pick = min(2, len(available_fruits)) if available_fruits else 0
    
    # Pesos inversos: ingredientes menos usados tienen más probabilidad de ser elegidos.
    # Fórmula: 1 / (freq + 1)  →  freq 0 = peso 1.0, freq 1 = 0.5, freq 3 = 0.25, ...
    # Esta fórmula da una penalización consistente e independiente del max_freq del dataset.
    protein_weights = [1.0 / (protein_freq.get(p, 0) + 1) for p in available_proteins]
    carb_weights = [1.0 / (carb_freq.get(c, 0) + 1) for c in available_carbs]
    veggie_weights = [1.0 / (veggie_freq.get(v, 0) + 1) for v in available_veggies]

    # [P2-TRANSFORM-BASE-BOOST · 2026-07-02] (audit v3 creatividad GAP-2) Las bases TRANSFORMABLES
    # (harinas/maíz, P1-FLOURS-POOLS) compiten con ~15 carbs por 2-3 cupos por chunk con peso solo de
    # frecuencia-inversa → los transforms insignia del owner (panqueques/arepitas/bollitos) casi nunca
    # podían emerger porque la base jamás ganaba un cupo. Boost multiplicativo del peso (default 2.0×) —
    # sigue siendo sorteo ponderado (no forzado); el efecto se valida con la serie del KPI de creatividad.
    # Rollback sin redeploy: MEALFIT_TRANSFORM_BASE_BOOST=1.0. tooltip-anchor: P2-TRANSFORM-BASE-BOOST
    try:
        import os as _os_tb
        _tb_boost = max(1.0, min(5.0, float(_os_tb.environ.get("MEALFIT_TRANSFORM_BASE_BOOST", "2.0"))))
    except Exception:
        _tb_boost = 2.0
    if _tb_boost > 1.0 and available_carbs:
        try:
            from constants import strip_accents as _sa_tb
            _TRANSFORM_BASE_TOKENS = ("harina", "maiz", "tortilla de trigo")
            carb_weights = [
                w * (_tb_boost if any(t in _sa_tb(str(c).lower()) for t in _TRANSFORM_BASE_TOKENS) else 1.0)
                for w, c in zip(carb_weights, available_carbs)
            ]
        except Exception:
            pass

    # [P1-BUDGET-TIER-LEVERS · 2026-07-02] (audit v4 presupuesto) Ponderación ECONÓMICA del sorteo:
    # el tier del formulario era señal solo-prompt (advisory). Cuando el presupuesto pide economía
    # (low "Económico" o custom ajustado — SSOT nutrition_calculator.budget_prefers_economy), las
    # proteínas/carbos del TERCIO MÁS BARATO del pool (precio/lb-equivalente del catálogo master)
    # reciben boost multiplicativo (default 2.0×). Sigue siendo sorteo ponderado — jamás remueve
    # ítems ni toca filtros clínicos/alergias (esos ya corrieron aguas arriba). Ítems sin precio
    # resoluble → peso intacto. Rollback sin redeploy: MEALFIT_BUDGET_POOL_WEIGHT=1.0.
    # tooltip-anchor: P1-BUDGET-TIER-LEVERS (pool weighting)
    try:
        import os as _os_bw
        _bud_boost = max(1.0, min(5.0, float(_os_bw.environ.get("MEALFIT_BUDGET_POOL_WEIGHT", "2.0"))))
    except Exception:
        _bud_boost = 2.0
    if _bud_boost > 1.0:
        try:
            from nutrition_calculator import budget_prefers_economy as _bpe_bw
            if _bpe_bw(form_data or {}):
                from shopping_calculator import get_master_ingredients as _gmi_bw
                from constants import strip_accents as _sa_bw

                def _bw_price_map() -> dict:
                    _out = {}
                    for _row in _gmi_bw() or []:
                        try:
                            _price = float(_row.get("price_per_lb") or 0)
                        except (TypeError, ValueError):
                            _price = 0.0
                        if _price <= 0:
                            try:
                                _ppu = float(_row.get("price_per_unit") or 0)
                                _basis = (float(_row.get("density_g_per_unit") or 0)
                                          or float(_row.get("container_weight_g") or 0))
                                if _ppu > 0 and _basis > 0:
                                    _price = _ppu * 453.592 / _basis
                            except (TypeError, ValueError):
                                _price = 0.0
                        if _price <= 0:
                            continue
                        _names = [_row.get("name") or ""]
                        if isinstance(_row.get("aliases"), list):
                            _names.extend(str(_a) for _a in _row.get("aliases"))
                        for _n in _names:
                            _k = _sa_bw(str(_n).strip().lower())
                            if _k:
                                _out.setdefault(_k, _price)
                    return _out

                def _bw_resolve(_name, _pmap):
                    _k = _sa_bw(str(_name or "").strip().lower())
                    if not _k:
                        return None
                    if _k in _pmap:
                        return _pmap[_k]
                    _padded = f" {_k} "
                    _best = None
                    for _mk in _pmap:
                        if len(_mk) >= 4 and f" {_mk} " in _padded:
                            if _best is None or len(_mk) > len(_best):
                                _best = _mk
                    return _pmap.get(_best) if _best else None

                def _bw_boost_cheapest(_names, _weights, _pmap):
                    _prices = [_bw_resolve(_n, _pmap) for _n in _names]
                    _valid = sorted(_p for _p in _prices if _p and _p > 0)
                    if len(_valid) < 4:
                        return _weights
                    _p33 = _valid[max(0, int(0.33 * (len(_valid) - 1)))]
                    return [
                        _w * (_bud_boost if (_p and _p <= _p33) else 1.0)
                        for _w, _p in zip(_weights, _prices)
                    ]

                _pmap_bw = _bw_price_map()
                if _pmap_bw:
                    protein_weights = _bw_boost_cheapest(available_proteins, protein_weights, _pmap_bw)
                    carb_weights = _bw_boost_cheapest(available_carbs, carb_weights, _pmap_bw)
                    logger.info(
                        f"💰 [P1-BUDGET-TIER-LEVERS] Presupuesto económico → boost {_bud_boost:.1f}× "
                        f"a proteínas/carbos del tercio más barato del catálogo en el sorteo."
                    )
        except Exception as _bw_e:
            logger.debug(f"[P1-BUDGET-TIER-LEVERS] pool weighting no-op: {_bw_e}")

    # Penalización de embutidos según objetivo nutricional.
    # Salami/longaniza/jamón/chorizo/tocineta/salchichón son procesados con sodio
    # alto y grasas saturadas — apropiados ocasionalmente en perfiles 'balanced'
    # pero contraindicados como base recurrente en perfiles que buscan ganancia
    # muscular limpia, pérdida de peso o mejora de salud cardiovascular.
    # Multiplicador 0.1 = 90% menos probabilidad de ser elegido (no eliminado:
    # puede aparecer ocasionalmente como variación cultural).
    _PROCESSED_MEAT_KEYWORDS = (
        "salami", "longaniza", "jamón", "jamon", "chorizo",
        "tocineta", "tocino", "salchichón", "salchichon", "salchicha",
        "mortadela", "embutido",
    )
    _GOALS_PENALIZE_PROCESSED = {
        "gain_muscle", "lose_weight", "health_improvement",
    }
    # [P2-PROTEIN-PENALTY-FATTY-MEAT · 2026-05-16] Categoría adicional:
    # carnes frescas grasas (NO procesadas) que para gain_muscle son
    # subóptimas por su ratio proteína/grasa.
    #   - Chuleta de cerdo: ~250kcal/100g, 20g grasa vs pechuga pollo
    #     165kcal/100g, 3.6g grasa.
    #   - Costilla, panceta, lechón, pernil: similar perfil graso.
    # Bug observable (plan_id=fbd014b2 2026-05-16): planner eligió
    # `Chuleta` en pool gain_muscle → receta Día 2 generada con cerdo →
    # PROTEIN-RECIPE-VIOLATION strippeó chuleta → cena sin proteína →
    # revisor médico rechazó.
    # Penalty ×0.3 (menos agresivo que embutidos ×0.1) porque son fresh
    # meat con valor nutricional legítimo en perfiles 'balanced'/cultural,
    # solo subóptimas para gain_muscle específicamente.
    _FATTY_FRESH_MEAT_KEYWORDS = (
        "chuleta", "costilla", "panceta", "lechón", "lechon",
        "pernil", "cerdo asado",
    )
    _GOALS_PENALIZE_FATTY_FRESH = {"gain_muscle"}
    _main_goal = (form_data.get("mainGoal") or "").strip().lower() if form_data else ""
    # [P1-BARIATRIC-PROTEIN-DENSITY · 2026-06-27] El paciente bariátrico necesita proteína ANIMAL densa en
    # comidas pequeñas: una base de leguminosa (baja densidad) no alcanza el piso de proteína en el volumen del
    # pouch y el revisor médico la rechaza por fibra/FODMAPs + déficit (visto corr=5b30b71f: Garbanzos×2). Por eso
    # bariátrica recibe el MISMO trato que gain_muscle: NO forzar leguminosa como proteína-main + sustituir las
    # proteínas-main de baja densidad por animal. tooltip-anchor: P1-BARIATRIC-PROTEIN-DENSITY
    _is_bariatric = False
    try:
        from constants import BARIATRIC_CONDITION_TERMS as _BARIA_T2, strip_accents as _sa_b2
        _cb = _sa_b2(
            " ".join(str(x) for x in ((form_data.get("medicalConditions") or []) if form_data else []))
            + " " + str((form_data.get("otherConditions") or "") if form_data else "")
        ).lower()
        _is_bariatric = any(t in _cb for t in _BARIA_T2)
    except Exception:
        _is_bariatric = False
    # [P1-BARIATRIC-PROTEIN-DENSITY · 2026-06-27] (iter 5) Nueces/semillas ENTERAS → riesgo OBSTRUCTIVO del pouch
    # (el revisor rechazó crítico 'maní/chía/pistachos enteros'). Penalizar fuerte (×0.1) en el pool veg/grasa para
    # bariátrica → preferir mantequillas/molidas. Las formas molidas/mantequilla/fileteada NO se penalizan.
    if _is_bariatric and veggie_weights:
        _WHOLE_NUT_SEED_TOKENS = ("mani", "almendra", "nuez", "nueces", "pistacho", "maranon", "merey",
                                  "avellana", "semilla", "pepita", "chia", "ajonjoli", "sesamo", "linaza")
        for _vi, _v in enumerate(available_veggies):
            _vn = strip_accents(str(_v).lower())
            if "mantequilla" in _vn or "molid" in _vn or "fileteada" in _vn:
                continue  # ya en forma segura (molida / mantequilla / fileteada)
            if any(_t in _vn for _t in _WHOLE_NUT_SEED_TOKENS):
                veggie_weights[_vi] *= 0.1
    if _main_goal in _GOALS_PENALIZE_PROCESSED or _is_bariatric:
        # [P1-BARIATRIC-PROTEIN-DENSITY] bariátrica penaliza embutidos grasos como proteína-main: el revisor
        # médico los rechaza (grasa saturada/sodio/aditivos → dumping + intolerancia). Visto corr=5ffd78cf:
        # el pool eligió 'Longaniza' → rechazo crítico.
        _penalized_count = 0
        for i, p in enumerate(available_proteins):
            p_norm = strip_accents(p.lower())
            if any(kw in p_norm for kw in _PROCESSED_MEAT_KEYWORDS):
                protein_weights[i] *= 0.1
                _penalized_count += 1
        if _penalized_count:
            logger.info(
                f"🥩 [GOAL PENALTY] Embutidos penalizados ×0.1 ({_penalized_count} items) "
                f"por goal='{_main_goal}'."
            )
    if _main_goal in _GOALS_PENALIZE_FATTY_FRESH:
        _fatty_penalized_count = 0
        for i, p in enumerate(available_proteins):
            p_norm = strip_accents(p.lower())
            if any(kw in p_norm for kw in _FATTY_FRESH_MEAT_KEYWORDS):
                protein_weights[i] *= 0.3
                _fatty_penalized_count += 1
        if _fatty_penalized_count:
            logger.info(
                f"🥩 [GOAL PENALTY-FATTY] Carnes grasas frescas (chuleta/costilla/panceta) "
                f"penalizadas ×0.3 ({_fatty_penalized_count} items) por goal='{_main_goal}'."
            )
    
    fruit_weights = []
    if available_fruits:
        fruit_weights = [1.0 / (fruit_freq.get(f, 0) + 1) for f in available_fruits]
        # [P1-BARIATRIC-PROTEIN-DENSITY · 2026-06-27] Bariátrica: penaliza frutas de ALTO índice glucémico
        # (guineo/mango/uva/piña/plátano) → prefiere bajo-IG (fresa/lechosa/mandarina/manzana). El revisor médico
        # rechazaba mango (clash) y guineo en porción grande por dumping (corr=5ffd78cf). Penalty ×0.15 (graceful:
        # si solo hay alto-IG disponible, igual se eligen). tooltip-anchor: P1-BARIATRIC-PROTEIN-DENSITY
        if _is_bariatric:
            _HIGH_GI_FRUITS = ("guineo", "banana", "mango", "uva", "pina", "platano", "melon", "sandia", "tamarindo")
            for _i, _f in enumerate(available_fruits):
                if any(_g in strip_accents(_f.lower()) for _g in _HIGH_GI_FRUITS):
                    fruit_weights[_i] *= 0.15

    # random.choices puede dar duplicados, así que aseguramos unicidad
    unique_proteins = []
    _pool_p = list(zip(available_proteins, protein_weights))
    while len(unique_proteins) < num_proteins_to_pick and _pool_p:
        pick = random.choices([x[0] for x in _pool_p], weights=[x[1] for x in _pool_p], k=1)[0]
        unique_proteins.append(pick)
        _pool_p = [(p, w) for p, w in _pool_p if p != pick]
    
    # 🥗 GARANTÍA NUTRICIONAL: Asegurar al menos 1 leguminosa en la selección.
    # [P1-LEGUME-GUARANTEE-GOAL-AWARE · 2026-06-16] Goal-aware: para gain_muscle NO
    # se fuerza una leguminosa como proteína PRINCIPAL de un día. Una base de
    # leguminosa (lentejas/garbanzos + almidón) no alcanza el piso de proteína (90%
    # de un target alto — p.ej. 108g de 120g) con porciones cocinables → el revisor
    # médico rechaza por DÉFICIT DE PROTEÍNA → retry-storm + entrega degradada.
    # Observado en vivo (corr 13117aff, 2026-06-16: la garantía forzó 'Lentejas' y
    # 'Garbanzos' como proteína principal → días 84-107g vs piso 108g; peor aún,
    # forzaba la leguminosa INCLUSO cuando la directiva de retry decía explícitamente
    # "NO dependas solo de leguminosas"). Las leguminosas siguen apareciendo como
    # acompañante en la generación del día; solo no se IMPONEN como proteína
    # principal del esqueleto para los objetivos de este set.
    _GOALS_SKIP_LEGUME_GUARANTEE = {"gain_muscle"}  # tooltip-anchor: legume_guarantee_goal_gate
    LEGUME_NAMES = {"habichuelas rojas", "habichuelas negras", "gandules", "lentejas", "garbanzos"}
    has_legume = any(p.lower() in LEGUME_NAMES for p in unique_proteins)
    if not has_legume and (_main_goal in _GOALS_SKIP_LEGUME_GUARANTEE or _is_bariatric):
        logger.info(
            f"🥩 [GARANTÍA NUTRICIONAL] Omitida para goal='{_main_goal}'{' (bariátrica)' if _is_bariatric else ''} — "
            f"la leguminosa no se impone como proteína principal (manda el piso de proteína)."
        )
    elif not has_legume:
        available_legumes = [p for p in available_proteins if p.lower() in LEGUME_NAMES]
        if available_legumes:
            legume_pick = random.choice(available_legumes)
            if len(unique_proteins) >= 2:
                freqs = [(p, protein_freq.get(p, 0)) for p in unique_proteins]
                freqs.sort(key=lambda x: x[1], reverse=True)
                replaced = freqs[0][0]
                idx = unique_proteins.index(replaced)
                unique_proteins[idx] = legume_pick
                logger.info(f"🥗 [GARANTÍA NUTRICIONAL] Leguminosa '{legume_pick}' reemplaza a '{replaced}'")
            else:
                unique_proteins.append(legume_pick)
                logger.info(f"🥗 [GARANTÍA NUTRICIONAL] Leguminosa '{legume_pick}' añadida")

    # [P3-GAINMUSCLE-PROTEIN-DENSITY · 2026-06-23] Para gain_muscle las proteínas PRINCIPALES del
    # esqueleto deben ser de ALTA densidad (animal). Sin esto el selector podía elegir 3 proteínas de
    # BAJA densidad (visto en vivo corr=f36bd39f: Queso Ricotta + Habichuelas Rojas + Gandules) → días
    # bajo el piso de proteína (124g) → el LLM rellena con huevo → choca con el cap de huevo Y el piso
    # a la vez → 3 rechazos del revisor → entrega DEGRADADA. Reemplazamos las proteínas-main de baja
    # densidad (leguminosas + ricotta/cottage/crema) por alta densidad usando el pool ponderado (que ya
    # penaliza embutidos/grasas). Las leguminosas/ricotta siguen apareciendo como ACOMPAÑANTE en la
    # generación del día (no se IMPONEN como main). Knob rollback: MEALFIT_GAINMUSCLE_HIGH_DENSITY_PROTEIN.
    # Tooltip-anchor: P3-GAINMUSCLE-PROTEIN-DENSITY.
    # Set EXPLÍCITO `_LOW_DENSITY_AS_MAIN` elevado a nivel módulo (P2-9) — reusado por swap_meal.
    if (_main_goal == "gain_muscle" or _is_bariatric) and _env_bool("MEALFIT_GAINMUSCLE_HIGH_DENSITY_PROTEIN", True):
        # [P1-BARIATRIC-PROTEIN-DENSITY] para bariátrica el set "reemplazable como main" incluye TAMBIÉN los
        # embutidos grasos (no solo baja densidad) → garantiza proteína animal magra en las comidas principales.
        def _should_replace_main(_p):
            _pl = strip_accents(_p.lower())
            if _pl in _LOW_DENSITY_AS_MAIN:
                return True
            if _is_bariatric and _pl in _BARIATRIC_LOW_DENSITY_AS_MAIN:  # [P1-BARIATRIC-DENSE-ANCHOR] quesos-relleno
                return True
            if _is_bariatric and any(_kw in _pl for _kw in _PROCESSED_MEAT_KEYWORDS):
                return True
            return False
        _low_mains = [p for p in unique_proteins if _should_replace_main(p)]
        if _low_mains:
            _hd_pool = [(p, w) for p, w in zip(available_proteins, protein_weights)
                        if p not in unique_proteins and not _should_replace_main(p)]
            for _rep in _low_mains:
                if not _hd_pool:
                    break  # sin alta-densidad disponible → conservar el de baja densidad (graceful)
                _new = random.choices([x[0] for x in _hd_pool], weights=[x[1] for x in _hd_pool], k=1)[0]
                unique_proteins[unique_proteins.index(_rep)] = _new
                _hd_pool = [(p, w) for p, w in _hd_pool if p != _new]
                logger.info(
                    f"💪 [GAIN-MUSCLE PROTEIN-DENSITY] '{_new}' (alta densidad) reemplaza a "
                    f"'{_rep}' (baja densidad como proteína principal)"
                )

    unique_carbs = []
    _pool_c = list(zip(available_carbs, carb_weights))
    while len(unique_carbs) < num_carbs_to_pick and _pool_c:
        pick = random.choices([x[0] for x in _pool_c], weights=[x[1] for x in _pool_c], k=1)[0]
        unique_carbs.append(pick)
        _pool_c = [(c, w) for c, w in _pool_c if c != pick]
        
    unique_veggies = []
    _pool_v = list(zip(available_veggies, veggie_weights))
    while len(unique_veggies) < num_veggies_to_pick and _pool_v:
        pick = random.choices([x[0] for x in _pool_v], weights=[x[1] for x in _pool_v], k=1)[0]
        unique_veggies.append(pick)
        _pool_v = [(v, w) for v, w in _pool_v if v != pick]
    
    unique_fruits = []
    if available_fruits and fruit_weights:
        _pool_f = list(zip(available_fruits, fruit_weights))
        while len(unique_fruits) < num_fruits_to_pick and _pool_f:
            pick = random.choices([x[0] for x in _pool_f], weights=[x[1] for x in _pool_f], k=1)[0]
            unique_fruits.append(pick)
            _pool_f = [(f, w) for f, w in _pool_f if f != pick]
            
    # ======= GROCERY CYCLE LOCK (Ahorro de Supermercado) =======
    # [P1-VARIETY-RENEWAL-NO-CYCLE-LOCK · 2026-06-27] El cycle-lock reutiliza los ingredientes base del ciclo de
    # compras (quincenal/mensual) para que el usuario NO tenga que re-comprar mid-ciclo. Efecto colateral: cada
    # renovación dentrega LOS MISMOS alimentos (solo varían los platos). El owner pidió priorizar VARIEDAD de
    # ingredientes sobre el reuso ("no me des los mismos a menos que lo necesite"). Default OFF → cada renovación
    # elige ingredientes nuevos del pool (202) y actualiza la base del ciclo (el shopping list refleja lo nuevo).
    # Flip a True (MEALFIT_GROCERY_CYCLE_LOCK=true) restaura el ahorro (reuso de las compras del ciclo).
    GROCERY_CYCLE_LOCK_ENABLED = _env_bool("MEALFIT_GROCERY_CYCLE_LOCK", False)
    grocery_duration = form_data.get("groceryDuration", "weekly") if form_data else "weekly"
    grocery_days = 7
    if grocery_duration == "biweekly": grocery_days = 15
    elif grocery_duration == "monthly": grocery_days = 30
    
    cycle_locked = False
    new_cycle_started = False
    
    # Excepción: la regla no aplica si grocery_days es 7 y no queremos complicar o si es guest
    if grocery_days > 7 and user_id and user_id != "guest":
        try:
            profile = get_user_profile(user_id)
            if profile:
                hp = profile.get("health_profile") or {}
                if not isinstance(hp, dict): hp = {}
                grocery_cycle = hp.get("grocery_cycle")
                
                now = datetime.now(timezone.utc)
                
                if grocery_cycle and "start_date" in grocery_cycle:
                    try:
                        from constants import safe_fromisoformat
                        cycle_start = safe_fromisoformat(grocery_cycle["start_date"])
                        if cycle_start.tzinfo is None:
                            cycle_start = cycle_start.replace(tzinfo=timezone.utc)
                        days_elapsed = (now - cycle_start).days
                        
                        # Si es < 2 días, es regeneración del mismo plan base, actualizaremos el ciclo.
                        if 2 <= days_elapsed < grocery_days and GROCERY_CYCLE_LOCK_ENABLED:
                            # ¡BLOQUEO ACTIVO! Forzamos la reutilización de ingredientes (ahorro de supermercado).
                            cycle_locked = True
                            unique_proteins = grocery_cycle.get("base_proteins", unique_proteins)
                            unique_carbs = grocery_cycle.get("base_carbs", unique_carbs)
                            unique_veggies = grocery_cycle.get("base_veggies", unique_veggies)
                            logger.info(f"🔒 [GROCERY CYCLE LOCK] Reutilizando ingredientes del ciclo (Día {days_elapsed} de {grocery_days}).")
                        elif days_elapsed >= grocery_days:
                            logger.info(f"🔓 [GROCERY CYCLE] Ciclo expirado ({days_elapsed} >= {grocery_days} días). Iniciando nuevo ciclo.")
                            new_cycle_started = True
                        else:
                            # [P1-VARIETY-RENEWAL-NO-CYCLE-LOCK] Día 2..N con lock OFF (default) → variety-first:
                            # NO reutilizamos; se eligen ingredientes nuevos y se actualiza la base del ciclo.
                            if 2 <= days_elapsed < grocery_days:
                                logger.info(f"🎨 [GROCERY CYCLE] Variety-first (lock OFF) en Día {days_elapsed}/{grocery_days} → ingredientes NUEVOS (no reuso).")
                            else:
                                logger.info(f"🔄 [GROCERY CYCLE] Regeneración en Día {days_elapsed} del ciclo. Actualizando Plan Base.")
                            new_cycle_started = True
                    except Exception as e:
                        logger.error(f"Error parseando fecha del ciclo: {e}")
                        new_cycle_started = True
                else:
                    new_cycle_started = True
                    
                # Si se necesita un nuevo ciclo o regeneración, guardamos los ingredientes recién elegidos
                if new_cycle_started:
                    start_date_to_save = now.isoformat()
                    # Si es regeneración (< 2 días), mantener el start_date original
                    if grocery_cycle and "start_date" in grocery_cycle and not (days_elapsed >= grocery_days if 'days_elapsed' in locals() else True):
                        start_date_to_save = grocery_cycle["start_date"]

                    # [P1-2] Write atómico vía advisory lock (FOR UPDATE). Antes,
                    # `get_user_profile + mutate + update_user_health_profile`
                    # eran 2 roundtrips no atómicos: bajo concurrencia del mismo
                    # user_id (regenerar mismo plan en 2 tabs, cron paralelo),
                    # dos writers leían el mismo snapshot de hp, cada uno
                    # appendeaba/mutaba localmente, y el último UPDATE pisaba al
                    # primero — perdiendo silenciosamente fields como
                    # `frictions`, `weight_history`, `reflection_history`,
                    # `lifetime_lessons_history` que otro path estuviera mutando
                    # entre el read y el write. Ahora el mutator SOLO toca
                    # `grocery_cycle`; los demás campos persisten intactos bajo
                    # FOR UPDATE.
                    new_grocery_cycle = {
                        "start_date": start_date_to_save,
                        "duration_days": grocery_days,
                        "base_proteins": unique_proteins,
                        "base_carbs": unique_carbs,
                        "base_veggies": unique_veggies,
                    }

                    def _grocery_cycle_mutator(_hp):
                        _hp["grocery_cycle"] = new_grocery_cycle
                        return None

                    update_user_health_profile_atomic(user_id, _grocery_cycle_mutator)
                    logger.info("💾 [GROCERY CYCLE] Guardados nuevos ingredientes base del ciclo.")
        except Exception as e:
            logger.error(f"Error procesando Grocery Cycle Lock: {e}")
    # ==========================================================

    # ======= CURRENT PANTRY INGREDIENTS INJECTION (ROTATION MODE) =======
    current_pantry_ingredients = (form_data.get("current_pantry_ingredients") or form_data.get("current_shopping_list", [])) if form_data else []
    if current_pantry_ingredients:
        logger.info(f"🔄 [ROTATION MODE] Extrayendo ingredientes base de la lista actual.")
        extracted_p, extracted_c, extracted_v, extracted_f = [], [], [], []
        csl_lower = [strip_accents(i.lower()) for i in current_pantry_ingredients]
        
        for item in csl_lower:
            for p in DOMINICAN_PROTEINS:
                syns = protein_synonyms.get(p.lower(), [p.lower()])
                if any(strip_accents(s) in item for s in syns) and p not in extracted_p: 
                    extracted_p.append(p)
            for c in DOMINICAN_CARBS:
                syns = carb_synonyms.get(c.lower(), [c.lower()])
                if any(strip_accents(s) in item for s in syns) and c not in extracted_c: 
                    extracted_c.append(c)
            for v in DOMINICAN_VEGGIES_FATS:
                syns = veggie_fat_synonyms.get(v.lower(), [v.lower()])
                if any(strip_accents(s) in item for s in syns) and v not in extracted_v: 
                    extracted_v.append(v)
            for f in DOMINICAN_FRUITS:
                syns = fruit_synonyms.get(f.lower(), [f.lower()])
                if any(strip_accents(s) in item for s in syns) and f not in extracted_f: 
                    extracted_f.append(f)
                
        if extracted_p: unique_proteins = extracted_p
        if extracted_c: unique_carbs = extracted_c
        if extracted_v: unique_veggies = extracted_v
        if extracted_f: unique_fruits = extracted_f
        cycle_locked = True # We force cycle locked mode to ensure pure rotation
        
    # ======= FORCED INGREDIENT INJECTION (FROM RAG/HISTORY) =======
    if form_data and "_force_base_proteins" in form_data:
        _forced_p = form_data.get("_force_base_proteins", unique_proteins)
        _forced_c = form_data.get("_force_base_carbs", unique_carbs)
        _forced_v = form_data.get("_force_base_veggies", unique_veggies)
        
        # --- FILTAR INGREDIENTES RECHAZADOS (EVITAR LOOP DE REVISOR MÉDICO) ---
        _banned_strings = []
        for pm in form_data.get("previous_meals", []):
            _banned_strings.append(strip_accents(pm.lower()))
        for dm in form_data.get("disliked_meals", []):
            _banned_strings.append(strip_accents(dm.lower()))
            
        def _is_forced_allowed(item):
            item_n = strip_accents(item.lower())
            for banned in _banned_strings:
                if item_n in banned or banned in item_n:
                    return False
                # Keywords fuertes
                words = item_n.split()
                banned_words = banned.split()
                if "pollo" in item_n and "pollo" in banned: return False
                if "res" in words and "res" in banned_words: return False
                if "cerdo" in item_n and "cerdo" in banned: return False
                if "pescado" in item_n and "pescado" in banned: return False
                if "habichuelas" in item_n and "habichuelas" in banned: return False
            return True
        
        unique_proteins = [p for p in _forced_p if _is_forced_allowed(p)]
        if len(unique_proteins) < 3: unique_proteins = _forced_p # Fallback de seguridad
        
        unique_carbs = [c for c in _forced_c if _is_forced_allowed(c)]
        if len(unique_carbs) < 3: unique_carbs = _forced_c
        
        forced_veg = [v for v in _forced_v if _is_forced_allowed(v)]
        if len(forced_veg) < 6: forced_veg = _forced_v
        
        # Frutas usualmente entran como vegetales desde el prompt, las filtramos manualmente
        fruit_names_lower = [strip_accents(f.strip().lower()) for f in DOMINICAN_FRUITS]
        extracted_fruits = []
        extracted_veggies = []
        
        for v in forced_veg:
            if strip_accents(v.strip().lower()) in fruit_names_lower:
                extracted_fruits.append(v)
            else:
                extracted_veggies.append(v)
                
        unique_veggies = extracted_veggies if extracted_veggies else unique_veggies
        if extracted_fruits:
            unique_fruits = extracted_fruits
            
        logger.info(f"🔒 [FORCE LOCK + FILTRADO] Proteínas: {unique_proteins}")
        logger.info(f"🔒 [FORCE LOCK + FILTRADO] Carbos: {unique_carbs}")
        logger.info(f"🔒 [FORCE LOCK + FILTRADO] Vegetales: {unique_veggies}")
        logger.info(f"🔒 [FORCE LOCK + FILTRADO] Frutas Extraídas: {unique_fruits}")
    # ==========================================================

    # Dedupicamos usando minúsculas normalizadas para evitar seleccionar "Huevos" y "Huevo s" en la misma corrida
    def _dedup_list(items):
        seen = set()
        out = []
        for i in items:
            # Remover espacios extra (ej. "Huevo s" -> "Huevos") solo como parche seguro si es común
            norm = i.lower().strip().replace(" s", "s")
            if norm not in seen:
                seen.add(norm)
                out.append(i)
        return out
        
    unique_proteins = _dedup_list(unique_proteins)
    unique_carbs = _dedup_list(unique_carbs)
    unique_veggies = _dedup_list(unique_veggies)
    if unique_fruits:
        unique_fruits = _dedup_list(unique_fruits)

    # Mezclar ANTES de rellenar o truncar, para asegurar rotación de todos los items en la lista de ingredientes base
    random.shuffle(unique_proteins)
    random.shuffle(unique_carbs)
    random.shuffle(unique_veggies)
    if unique_fruits:
        random.shuffle(unique_fruits)

    # Cada día recibe una proteína, un carbohidrato, y un vegetal únicos (sin repeticiones entre días)
    # Si no se pudieron elegir 3, rellenamos ciclando lo que hay
    _base_proteins = list(unique_proteins)
    while len(unique_proteins) < 3:
        unique_proteins.append(_base_proteins[len(unique_proteins) % len(_base_proteins)])
    _base_carbs = list(unique_carbs)
    while len(unique_carbs) < 3:
        unique_carbs.append(_base_carbs[len(unique_carbs) % len(_base_carbs)])
    _base_veggies = list(unique_veggies)
    while len(unique_veggies) < 6:
        unique_veggies.append(_base_veggies[len(unique_veggies) % len(_base_veggies)])
    # [CROSS-DAY-FRUIT-DIVERSITY 2026-05-07] Bug observable plan 55da8e9b:
    # cuando el picker dejaba <3 frutas únicas, el padding usaba siempre
    # `unique_fruits[0]` → fruta repetida en múltiples días (caso real:
    # `['Melón', 'Naranja', 'Melón']`). El LLM luego metía Melón en
    # 5 meals across Día 1 + Día 3, gatillando rechazo médico por
    # "carotenemia (melón+auyama excesivo)".
    #
    # Fix: padding inteligente en 2 niveles:
    #   1. Round-robin sobre _base_fruits si hay >=2 (paridad con
    #      proteínas/carbos/veggies — line 588-595).
    #   2. Si solo hay 1 fruta base, complementar desde un pool DR default
    #      con frutas que NO estén ya presentes (garantiza 3 distintas).
    if unique_fruits:
        _base_fruits = list(unique_fruits)
        _DEFAULT_DR_FRUITS = (
            'Limón', 'Lechosa', 'Mango', 'Piña', 'Guineo', 'Naranja',
            'Fresas', 'Chinola', 'Melón', 'Manzana',
        )
        while len(unique_fruits) < 3:
            existing = {f.lower() for f in unique_fruits}
            # Prioridad 1: añadir una fruta DR default que NO esté ya presente.
            # Garantiza variedad cross-day (cada día recibe fruta distinta).
            _added = False
            for _df in _DEFAULT_DR_FRUITS:
                if _df.lower() not in existing:
                    unique_fruits.append(_df)
                    _added = True
                    break
            if _added:
                continue
            # Prioridad 2 (rara): todas las default ya presentes — round-robin
            # del base como último recurso.
            if _base_fruits:
                unique_fruits.append(_base_fruits[len(unique_fruits) % len(_base_fruits)])
            else:
                break  # Salida segura
    
    chosen_proteins = unique_proteins[:3]
    chosen_carbs = unique_carbs[:3]
    chosen_veggies = unique_veggies[:6]
    chosen_fruits = unique_fruits[:3] if unique_fruits else []
    
    # Repetimos mezcla final de los 3 días elegidos para distribuir el orden
    random.shuffle(chosen_proteins)
    random.shuffle(chosen_carbs)
    random.shuffle(chosen_veggies)
    if chosen_fruits:
        random.shuffle(chosen_fruits)
    
    blocked_text = ""
    if used_proteins or used_carbs or used_veggies:
        # Solo bloquear ingredientes sobreusados (freq >= OVERUSE_THRESHOLD) que NO fueron elegidos por el determinismo.
        # Esto elimina la contradicción: si el picker eligió "Pollo", no le decimos al LLM que está prohibido.
        chosen_set = set(p.lower() for p in chosen_proteins + chosen_carbs + chosen_veggies + chosen_fruits)
        blocked_items = [item for item in (used_proteins + used_carbs + used_veggies)
                         if item.lower() not in chosen_set]
        if blocked_items:
            blocked_text = f"⚠️ EVITA usar como base principal estos ingredientes sobreusados (el usuario ya los ha comido frecuentemente): {', '.join(blocked_items)}. Prioriza alternativas frescas."
    
    # Nota de conservación de alimentos según frecuencia de compras
    grocery_duration = form_data.get("groceryDuration", "weekly") if form_data else "weekly"
    if grocery_duration == "monthly":
        blocked_text += "\n🛒 COMPRAS MENSUALES: El usuario compra para 30 días. PRIORIZA ingredientes no perecederos o fácilmente congelables, granos secos, proteínas congelables. Evita depender de perecederos de vida corta."
    elif grocery_duration == "biweekly":
        blocked_text += "\n🛒 COMPRAS QUINCENALES: El usuario compra para 15 días. PRIORIZA ingredientes de duración media o congelables."
        
    if cycle_locked:
        # Use a safe fallback for days_elapsed in case it wasn't defined perfectly
        d_elapsed = locals().get('days_elapsed', '?')
        blocked_text += f"\n\n🚨 [REGLA DE AHORRO EXTREMA]: El usuario está en el Día {d_elapsed} de su ciclo de compras de {grocery_days} días. TIENES LA OBLIGACIÓN ESTRICTA de basar todas las comidas en usar EXACTAMENTE las proteínas, carbohidratos y vegetales asignados explícitamente en el prompt. Usa diferentes preparaciones y técnicas de cocción para que no se aburra, pero NO SUGIERAS ALIMENTOS BASE NUEVOS."
        
    is_plan_expired = form_data.get("is_plan_expired", False) if form_data else False
    if is_plan_expired:
        blocked_text += "\n\n♻️ [NUEVO CICLO DE COMPRAS]: El plan anterior del usuario ha expirado. Este es un ciclo de compras completamente nuevo. TIENES PERMISO PARA SUGERIR NUEVOS INGREDIENTES BASE. Ignora las restricciones de ahorro extremo del ciclo anterior."    
    if user_id and user_id != "guest":
        try:
            profile = get_user_profile(user_id)
            if profile:
                hp = profile.get("health_profile") or {}
                persisted_rejections = hp.get("rejection_patterns", [])
                if persisted_rejections:
                    blocked_text += "\n\n🧠 [MEMORIA DEL REVISOR MÉDICO - EVITA ESTOS ERRORES HISTÓRICOS]:"
                    for r in persisted_rejections[-5:]: # Solo los últimos 5 para no sobrecargar el prompt
                        blocked_text += f"\n - {r}"
        except Exception as _rej_exc:
            # [P2-SILENT-DEGRADATION · 2026-05-13] DB blip / pool exhaustion:
            # el agente pierde memoria histórica de rechazos del Revisor Médico
            # → puede repetir errores ya corregidos en planes anteriores.
            # Impacto mayor que los otros 2 silent-paths: la cadena
            # rejection→retry→aprendizaje se rompe. Log debug permite
            # detectar burst de fallos durante incidentes de DB.
            logger.debug(
                "[P2-SILENT-DEGRADATION] rejection_patterns fetch falló "
                "(user_id=%s): %s: %s",
                str(user_id)[:36],
                type(_rej_exc).__name__,
                str(_rej_exc)[:160],
            )

    # Inyectar razones de rechazo del intento anterior (Mutación de Retry - GAP 1)
    if rejection_reasons:
        blocked_text += "\n\n🚨 [REVISIÓN RECHAZADA] El Revisor Médico rechazó tu intento anterior por los siguientes motivos. MUTA TU ESTRATEGIA INMEDIATAMENTE Y EVITA:"
        for reason in rejection_reasons:
            blocked_text += f"\n - {reason}"
            
    update_reason = form_data.get("update_reason") if form_data else None
    
    # ======= [GAP 1] PERSISTENCIA DE SEÑALES DE APRENDIZAJE =======
    # Guardamos los "dislikes" y "skips" como patrones de rechazo permanentes
    if form_data and user_id and user_id != 'guest':
        disliked_m = form_data.get("disliked_meals", [])
        skipped_m = form_data.get("skipped_meals", [])
        
        # Si se genera con update_reason == 'dislike', también consideramos previous_meals como disliked
        if update_reason == 'dislike':
            prev_m = form_data.get("previous_meals", [])
            if isinstance(prev_m, list):
                # [P0 FIX GAP 2] Evitar Mode Collapse por baneo masivo.
                # Solo persistimos las primeras 3 comidas (ej. el día actual) para aprender la señal
                # sin agotar la base de ingredientes permitidos a largo plazo.
                disliked_m.extend(prev_m[:3])
                
        meals_to_ban = set()
        if isinstance(disliked_m, list): meals_to_ban.update(disliked_m)
        if isinstance(skipped_m, list): meals_to_ban.update(skipped_m)
        
        if meals_to_ban:
            try:
                # [P1-2] Mutator atómico. La lectura de `rejection_patterns`
                # ocurre DENTRO del mutator (bajo FOR UPDATE), así que dos
                # invocaciones concurrentes con el mismo user_id se serializan
                # y NUNCA pierden bans entre sí. Antes, dos writers leían la
                # misma lista, cada uno appendeaba localmente, y el último
                # UPDATE pisaba al primero — un dislike/skip simultáneo del
                # cron y del manual disparaba la regeneración con la lista de
                # bans más reciente del último UPDATE, descartando los del
                # otro. El sentinel mutable `bans_count_box` lleva el contador
                # de nuevas adiciones afuera del mutator para el log.
                bans_count_box = {"count": 0}

                def _rejection_mutator(_hp):
                    _rejected = list(_hp.get("rejection_patterns", []) or [])
                    if not isinstance(_rejected, list):
                        _rejected = []
                    _new_bans = []
                    for _m in meals_to_ban:
                        if _m and isinstance(_m, str) and _m not in _rejected:
                            _new_bans.append(_m)
                            _rejected.append(_m)
                    if not _new_bans:
                        return False  # nada que persistir
                    _hp["rejection_patterns"] = _rejected[-50:]  # cap anti-bloat
                    bans_count_box["count"] = len(_new_bans)
                    return None

                update_user_health_profile_atomic(user_id, _rejection_mutator)
                if bans_count_box["count"] > 0:
                    logger.info(
                        f"🧠 [GAP 1] Aprendizaje Continuo: Persistidos "
                        f"{bans_count_box['count']} platos en rejection_patterns "
                        f"por acciones 'dislike'/'skip'."
                    )
            except Exception as e:
                logger.error(f"❌ [GAP 1] Error persistiendo señales de dislike/skip: {e}")
    # ==============================================================

    if update_reason == 'variety':
        blocked_text += "\n\n💡 [INTENCIÓN DEL USUARIO]: El usuario solicitó explícitamente MAYOR VARIEDAD al actualizar el plan. Ofrece combinaciones creativas, diferentes técnicas de cocción y perfiles de sabor novedosos."
    elif update_reason == 'dislike':
        blocked_text += "\n\n🚨 [INTENCIÓN DEL USUARIO]: El usuario solicitó actualizar el plan porque NO LE GUSTARON las opciones generadas. EVITA los perfiles de sabor de los platos anteriores y cambia radicalmente la estrategia."
    elif update_reason == 'time':
        blocked_text += "\n\n⏱️ [INTENCIÓN DEL USUARIO]: El usuario NO TIENE TIEMPO HOY. Obligatorio: propón recetas extremadamente rápidas (menos de 20 min) y que requieran muy poca preparación."
    elif update_reason == 'similar':
        blocked_text += "\n\n🍽️ [INTENCIÓN DEL USUARIO]: El usuario ya comió algo similar recientemente. Ofrece un perfil de sabor o técnica de cocción COMPLETAMENTE DISTINTA a lo que normalmente sugiere."
    # [P3-NEWPLAN-NO-BUDGET-MODAL · 2026-05-23] Branch `update_reason ==
    # 'budget'` eliminado. La opción "Opciones económicas" del modal
    # new-plan (Dashboard.jsx) se removió porque el comportamiento ya
    # respeta la nevera + lista de compras por default (el frontend
    # pasa `current_pantry_ingredients` a `/api/plans/generate`). El
    # hint "ECONÓMICAS" era ortogonal a esa restricción real y sugería
    # al usuario que los demás reasons NO usaban su nevera.
    # `pantry_first` se preserva para back-compat con callers legacy.
    elif update_reason == 'pantry_first':
        if not cycle_locked:
            blocked_text += "\n\n📦 [INTENCIÓN DEL USUARIO]: El usuario quiere MAXIMIZAR EL USO DE SU INVENTARIO. Las recetas deben depender exclusivamente de ingredientes base comunes de despensa sin requerir compras exóticas."
    elif update_reason == 'cravings':
        blocked_text += "\n\n🤤 [INTENCIÓN DEL USUARIO]: El usuario tiene un ANTOJO. Ofrece opciones más indulgentes, comfort food dominicano o versiones saludables de platos tipo cheat-meal, pero manteniendo los macros."
    elif update_reason == 'weekend':
        blocked_text += "\n\n🎉 [INTENCIÓN DEL USUARIO]: El usuario busca algo para un FIN DE SEMANA ESPECIAL. Propón platos más elaborados, con presentación premium, ideales para disfrutar con tiempo o en familia."
    
    # Construir parámetros de frutas para el prompt
    fruit_params = {}
    if chosen_fruits and len(chosen_fruits) == 3:
        fruit_params = {
            "fruit_0": chosen_fruits[0], "fruit_1": chosen_fruits[1], "fruit_2": chosen_fruits[2]
        }
    else:
        _fallback_fruit = "elige la fruta dominicana que mejor combine con la preparación"
        fruit_params = {"fruit_0": _fallback_fruit, "fruit_1": _fallback_fruit, "fruit_2": _fallback_fruit}
        
    prompt = DETERMINISTIC_VARIETY_PROMPT.format(
        protein_0=chosen_proteins[0], carb_0=chosen_carbs[0],
        veggie_0=chosen_veggies[0], veggie_0b=chosen_veggies[3],
        protein_1=chosen_proteins[1], carb_1=chosen_carbs[1],
        veggie_1=chosen_veggies[1], veggie_1b=chosen_veggies[4],
        protein_2=chosen_proteins[2], carb_2=chosen_carbs[2],
        veggie_2=chosen_veggies[2], veggie_2b=chosen_veggies[5],
        blocked_text=blocked_text,
        **fruit_params
    )
    logger.info(f"✅ [ANTI MODE-COLLAPSE] Proteínas elegidas para 3 días (rotadas si es necesario): {chosen_proteins}")
    logger.info(f"✅ [ANTI MODE-COLLAPSE] Carbohidratos elegidos para 3 días (rotados si es necesario): {chosen_carbs}")
    logger.info(f"✅ [ANTI MODE-COLLAPSE] Vegetales/Grasas elegidos (2 distintos por día): {chosen_veggies}")
    logger.info(f"✅ [ANTI MODE-COLLAPSE] Fruta sugerida: {chosen_fruits}")
    return prompt

def expand_recipe_agent(meal_data: dict) -> Optional[list[str]]:
    """Expande una receta genérica en instrucciones súper detalladas actuando
    como un Chef Instructor Premium.

    [P1-RECIPE-EXPAND-FAILSIGNAL · 2026-05-30] Devuelve `None` cuando la
    expansión NO produce contenido nuevo válido (excepción LLM, circuit
    breaker abierto, respuesta vacía o no-lista). Pre-fix: este helper
    devolvía SILENCIOSAMENTE la receta original (`meal_data.get("recipe")`)
    en cualquier fallo. El endpoint `/recipe/expand` interpretaba ese eco como
    éxito → marcaba `isExpanded=True` + persistía + cobraba cuota, y el guard
    del frontend (`if (meal.isExpanded) return`) jamás reintentaba — un único
    blip de Gemini dejaba la comida (y, vía Camino-2, toda ocurrencia con la
    misma receta) atrapada permanentemente en sus pasos tersos sin vía de
    retry. Señalizar fallo con `None` permite al endpoint NO marcar el flag,
    NO persistir y NO cobrar cuota, devolviendo la original solo para display.

    Validación de salida (cierra schema gap P2 #9): la lista debe ser no-vacía
    y contener al menos un paso string no-blank. Una respuesta degenerada
    (lista vacía, o pasos todos blancos) se trata como fallo, NO como
    expansión válida.

    Tooltip-anchor: P1-RECIPE-EXPAND-FAILSIGNAL-AGENT
    """
    logger.info(f"👨‍🍳 [CHEF AGENT] Expandiendo instrucciones para: {meal_data.get('name', 'Receta')}")

    prompt = RECIPE_EXPANSION_PROMPT.format(
        name=meal_data.get("name", "Receta sin nombre"),
        desc=meal_data.get("desc", ""),
        ingredients_json=json.dumps(meal_data.get("ingredients", []), ensure_ascii=False),
        recipe_json=json.dumps(meal_data.get("recipe", []), ensure_ascii=False)
    )

    try:
        llm = ChatDeepSeek(
            model=_recipe_expand_model_name(),  # [P1-RECIPE-EXPAND-FAILSIGNAL] knob, era hardcoded
            temperature=0.7,
            timeout=_ai_helpers_llm_timeout_s(),  # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30]
        ).with_structured_output(ExpandedRecipeModel)

        @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
        def _invoke():
            return llm.invoke(prompt)

        response = _invoke()
        steps = getattr(response, "recipe", None) if response is not None else None
        # [P2-RECIPE-STEP-CONTRACT-GATE · 2026-07-01] (audit recetas P2-2) El check aceptaba ≥1 paso pese a
        # que ExpandedRecipeModel exige exactamente 3 → una "expansión" degenerada de 1 paso REEMPLAZABA una
        # receta completa y quedaba isExpanded=true (sin retry posible). Ahora exige ≥3 pasos sustantivos;
        # menos → fallo señalizado (None: sin cobro, sin persist, el cliente puede reintentar).
        if isinstance(steps, list):
            clean_steps = [s for s in steps if isinstance(s, str) and s.strip()]
            if len(clean_steps) >= 3:
                logger.info("✅ [CHEF AGENT] Receta expandida con éxito.")
                return clean_steps
            if clean_steps:
                logger.warning(f"⚠️ [CHEF AGENT] Expansión degenerada ({len(clean_steps)} paso(s) < 3) → "
                               f"fallo señalizado (no reemplaza la receta completa).")
                return None
        logger.warning("⚠️ [CHEF AGENT] El modelo no regresó una lista 'recipe' válida. Señalizando fallo (None).")
        return None

    except Exception as e:
        logger.error(f"❌ [CHEF AGENT] Falla al expandir receta: {e}")
        return None


def generate_llm_retrospective(user_id: str, plan_data: dict, consumed_records: list, recent_likes: list, recent_rejections: list) -> str:
    """
    [MEJORA 5] LLM-as-Judge Offline: Analiza la dieta planificada vs ejecutada y genera
    lecciones aprendidas cualitativas sobre por qué el usuario tuvo éxito o fracasó.
    """
    logger.info(f"🧠 [LLM-as-Judge] Generando retrospectiva semanal para user: {user_id}")
    
    try:
        # Simplificar datos para no ahogar la ventana de contexto
        planned_meals = []
        for day in plan_data.get("days", []):
            for m in day.get("meals", []):
                planned_meals.append(f"{m.get('meal')}: {m.get('name')}")
                
        consumed_meals = [cm.get("meal_name", "") for cm in consumed_records if cm.get("meal_name")]
        liked_names = [l.get("meal_name", "") for l in recent_likes] if recent_likes else []
        rejected_names = [r.get("meal_name", "") for r in recent_rejections] if recent_rejections else []
        
        prompt = f"""Eres el Juez Clínico Nutricional (LLM-as-Judge). 
Tu trabajo es analizar el plan de comidas de la última semana de un usuario, qué comió realmente y qué rechazó o le gustó.
A partir de estos datos, extrae EXACTAMENTE 3 lecciones cualitativas altamente accionables sobre su comportamiento.

DATOS:
- Comidas planificadas: {', '.join(planned_meals[:15])} (truncado a 15)
- Comidas REALMENTE consumidas (adherencia): {', '.join(consumed_meals[:15])}
- Comidas a las que dio "Me Gusta": {', '.join(liked_names)}
- Comidas que rechazó explícitamente: {', '.join(rejected_names)}

REGLAS DE SALIDA:
- Escribe una lista de 3 puntos (bullet points).
- Cada punto debe explicar POR QUÉ algo funcionó o falló.
- Usa un tono clínico pero directo.
- NO ofrezcas consejos futuros, SOLO hechos observados (ej: "El usuario respondió excelente a desayunos salados, pero rechazó todos los batidos dulces").
"""
        llm = ChatDeepSeek(
            model=model_free_tier(),  # [P0-DEEPSEEK-MIGRATION] aux barato (knob MEALFIT_MODEL_FREE_TIER)
            temperature=0.2,
            timeout=_ai_helpers_llm_timeout_s(),  # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30]
        )
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = " ".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])

        retrospective = str(content).strip()
        logger.info(f"✅ [LLM-as-Judge] Retrospectiva generada: {retrospective[:100]}...")
        return retrospective
    except Exception as e:
        logger.error(f"❌ [LLM-as-Judge] Error generando retrospectiva: {e}")
        return ""

def extract_liked_flavor_profiles(recent_likes: list) -> list[str]:
    """Extrae características subyacentes (perfiles de sabor, ingredientes clave, técnicas) de los likes del usuario."""
    if not recent_likes:
        return []
        
    try:
        from pydantic import BaseModel, Field
        class FlavorProfiles(BaseModel):
            profiles: list[str] = Field(description="Lista de 2-3 perfiles de sabor, ingredientes o técnicas que el usuario disfruta explícitamente.")
            
        liked_names = [l.get("meal_name", "") for l in recent_likes if l.get("meal_name")]
        if not liked_names:
            return []
            
        prompt = f"""Analiza los siguientes platos a los que el usuario dio "Me Gusta".
Extrae 2 o 3 características subyacentes (ej: ingredientes clave, perfiles de sabor, tipo de preparación) que tengan en común o que definan sus gustos.
Platos: {', '.join(liked_names)}
Ejemplos de características: "Prefiere desayunos salados con plátano", "Le gustan los guisos tradicionales dominicanos con salsa", "Disfruta de proteínas a la plancha"
"""
        
        llm = ChatDeepSeek(
            model=model_free_tier(),  # [P0-DEEPSEEK-MIGRATION] aux barato (knob MEALFIT_MODEL_FREE_TIER)
            temperature=0.2,
            timeout=_ai_helpers_llm_timeout_s(),  # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30]
        ).with_structured_output(FlavorProfiles)
        
        response = llm.invoke(prompt)
        logger.info(f"❤️ [FEATURE EXTRACTION] Perfiles de sabor extraídos: {response.profiles}")
        return response.profiles
    except Exception as e:
        logger.error(f"❌ [FEATURE EXTRACTION] Error extrayendo perfiles de sabor: {e}")
        return []
