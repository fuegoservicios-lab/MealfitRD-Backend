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
from langchain_google_genai import ChatGoogleGenerativeAI
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
from db import get_user_profile, update_user_health_profile, get_user_ingredient_frequencies
from cpu_tasks import _calcular_frecuencias_regex_cpu_bound

logger = logging.getLogger(__name__)


def generate_plan_title(plan_data: dict) -> str:
    """Genera un título corto y creativo para un plan nutricional usando Gemini Flash-Lite."""
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
        
        title_llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0.9,
            google_api_key=os.environ.get("GEMINI_API_KEY")
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


def get_deterministic_variety_prompt(history_text: str, form_data: dict = None, user_id: str = None, rejection_reasons: list = None) -> str:
    """Implementa Inversión de Control Determinista para evitar Mode Collapse en el LLM."""
    logger.debug("🎲 [ANTI MODE-COLLAPSE] Calculando Matriz de Ingredientes (Round-Robin)...")
    history_lower = history_text.lower() if history_text else ""
    history_normalized = strip_accents(history_lower)
    
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
                except Exception:
                    pass
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
    skip_lunch = form_data.get("skipLunch", False) if form_data else False
    variety_level = form_data.get("variety_level", "") if form_data else ""
    
    # Si no viene en form_data, intentar leer del perfil persistido en DB
    if not variety_level and user_id and user_id != "guest":
        try:
            profile = get_user_profile(user_id)
            if profile:
                hp = profile.get("health_profile") or {}
                variety_level = hp.get("variety_level", "standard")
        except Exception:
            pass
    variety_level = variety_level or "standard"
    
    if skip_lunch:
        num_proteins_to_pick = min(1, len(available_proteins))  # 1 proteína (solo Cena la necesita fuerte)
        num_carbs_to_pick = min(2, len(available_carbs))         # 2 carbos (Desayuno y Cena)
        num_veggies_to_pick = min(4, len(available_veggies))     # 4 vegetales (2 por día × 2 días con comida principal)
        logger.info(f"⚡ [ANTI MODE-COLLAPSE] skipLunch=true → distribución reducida (1P/2C/2V)")
    elif variety_level == "max":
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
    
    fruit_weights = []
    if available_fruits:
        fruit_weights = [1.0 / (fruit_freq.get(f, 0) + 1) for f in available_fruits]
    
    # random.choices puede dar duplicados, así que aseguramos unicidad
    unique_proteins = []
    _pool_p = list(zip(available_proteins, protein_weights))
    while len(unique_proteins) < num_proteins_to_pick and _pool_p:
        pick = random.choices([x[0] for x in _pool_p], weights=[x[1] for x in _pool_p], k=1)[0]
        unique_proteins.append(pick)
        _pool_p = [(p, w) for p, w in _pool_p if p != pick]
    
    # 🥗 GARANTÍA NUTRICIONAL: Asegurar al menos 1 leguminosa en la selección
    LEGUME_NAMES = {"habichuelas rojas", "habichuelas negras", "gandules", "lentejas", "garbanzos"}
    has_legume = any(p.lower() in LEGUME_NAMES for p in unique_proteins)
    if not has_legume and not skip_lunch:
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
                        if 2 <= days_elapsed < grocery_days:
                            # ¡BLOQUEO ACTIVO! Forzamos la reutilización de ingredientes.
                            cycle_locked = True
                            unique_proteins = grocery_cycle.get("base_proteins", unique_proteins)
                            unique_carbs = grocery_cycle.get("base_carbs", unique_carbs)
                            unique_veggies = grocery_cycle.get("base_veggies", unique_veggies)
                            logger.info(f"🔒 [GROCERY CYCLE LOCK] Reutilizando ingredientes del ciclo (Día {days_elapsed} de {grocery_days}).")
                        elif days_elapsed >= grocery_days:
                            logger.info(f"🔓 [GROCERY CYCLE] Ciclo expirado ({days_elapsed} >= {grocery_days} días). Iniciando nuevo ciclo.")
                            new_cycle_started = True
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
                        
                    hp["grocery_cycle"] = {
                        "start_date": start_date_to_save,
                        "duration_days": grocery_days,
                        "base_proteins": unique_proteins,
                        "base_carbs": unique_carbs,
                        "base_veggies": unique_veggies
                    }
                    update_user_health_profile(user_id, hp)
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
    if unique_fruits:
        while len(unique_fruits) < 3:
            unique_fruits.append(unique_fruits[0])
    
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
    
    # Si skipLunch y solo 1 proteína base, agregar nota al blocked_text
    # para que el LLM sepa que debe variar la PREPARACIÓN, no el ingrediente.
    if skip_lunch and len(_base_proteins) == 1:
        blocked_text += f"\n💡 NOTA: Solo hay 1 proteína base ({_base_proteins[0]}) para los 3 días porque el usuario omite Almuerzo. Varía la PREPARACIÓN y TÉCNICA de cocción cada día, no el ingrediente."
    
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
        except Exception:
            pass

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
                # Usar los métodos ya importados al inicio de ai_helpers.py
                profile = get_user_profile(user_id)
                if profile:
                    hp = profile.get("health_profile") or {}
                    if not isinstance(hp, dict): hp = {}
                    
                    rejected = hp.get("rejection_patterns", [])
                    if not isinstance(rejected, list): rejected = []
                    
                    new_bans = []
                    for m in meals_to_ban:
                        if m and isinstance(m, str) and m not in rejected:
                            new_bans.append(m)
                            rejected.append(m)
                    
                    if new_bans:
                        hp["rejection_patterns"] = rejected[-50:]  # Limitar para no saturar JSON
                        update_user_health_profile(user_id, hp)
                        logger.info(f"🧠 [GAP 1] Aprendizaje Continuo: Persistidos {len(new_bans)} platos en rejection_patterns por acciones 'dislike'/'skip'.")
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
    elif update_reason == 'budget':
        blocked_text += "\n\n💰 [INTENCIÓN DEL USUARIO]: El usuario busca opciones ECONÓMICAS. Prioriza ingredientes de muy bajo costo y alto rendimiento, evitando proteínas premium o ingredientes importados costosos."
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

def expand_recipe_agent(meal_data: dict) -> list[str]:
    """Expande una receta genérica en instrucciones súper detalladas actuando como un Chef Instructor Premium."""
    logger.info(f"👨‍🍳 [CHEF AGENT] Expandiendo instrucciones para: {meal_data.get('name', 'Receta')}")
    
    prompt = RECIPE_EXPANSION_PROMPT.format(
        name=meal_data.get("name", "Receta sin nombre"),
        desc=meal_data.get("desc", ""),
        ingredients_json=json.dumps(meal_data.get("ingredients", []), ensure_ascii=False),
        recipe_json=json.dumps(meal_data.get("recipe", []), ensure_ascii=False)
    )
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0.7,
            google_api_key=os.environ.get("GEMINI_API_KEY")
        ).with_structured_output(ExpandedRecipeModel)
        
        @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
        def _invoke():
            return llm.invoke(prompt)
            
        response = _invoke()
        if hasattr(response, "recipe") and response.recipe:
            logger.info("✅ [CHEF AGENT] Receta expandida con éxito.")
            return response.recipe
        else:
            logger.warning("⚠️ [CHEF AGENT] El modelo no regresó la lista 'recipe'. Usando original.")
            return meal_data.get("recipe", [])
            
    except Exception as e:
        logger.error(f"❌ [CHEF AGENT] Falla al expandir receta: {e}")
        return meal_data.get("recipe", [])


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
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0.2,
            google_api_key=os.environ.get("GEMINI_API_KEY")
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
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0.2,
            google_api_key=os.environ.get("GEMINI_API_KEY")
        ).with_structured_output(FlavorProfiles)
        
        response = llm.invoke(prompt)
        logger.info(f"❤️ [FEATURE EXTRACTION] Perfiles de sabor extraídos: {response.profiles}")
        return response.profiles
    except Exception as e:
        logger.error(f"❌ [FEATURE EXTRACTION] Error extrayendo perfiles de sabor: {e}")
        return []
