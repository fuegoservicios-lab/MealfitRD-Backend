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
from schemas import SemanticDedupResult, ExpandedRecipeModel

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

def semantic_merge_items(survivors: list) -> list:
    """Usa LLM para agrupar ingredientes que son el mismo producto (ej: 'Pollo desmenuzado' y 'Pechuga de pollo')."""
    if not survivors or len(survivors) < 2:
        return []
        
    logger.info(f"🧠 [SEMANTIC DEDUP] Analizando {len(survivors)} ingredientes únicos con LLM...")
    
    # Preparamos una lista simplificada para el LLM para ahorrar tokens
    items_for_prompt = []
    for s in survivors:
        items_for_prompt.append({
            "id": s["item"]["id"],
            "name": s["name"],
            "qty": s["item"].get("qty", "")
        })
    
    prompt = f"""Dado este listado exacto de ingredientes con sus IDs y cantidades, agrupa ESTRICTAMENTE aquellos que sean variantes del mismo producto alimenticio base en la vida real (ej: 'Pollo desmenuzado' y 'Pechuga de pollo' -> 'Pechuga de pollo', 'Huevos' y 'Huevos grandes' -> 'Huevos', 'Cebolla roja' y 'Cebolla blanca' -> 'Cebolla').
REGLAS:
- SOLO AGRUPA si son variaciones del MISMO producto y pueden comprarse/usarse como lo mismo.
- NO agrupes 'Manzana verde' con 'Banana' bajo 'Frutas'. NO agrupes 'Cerdo' con 'Pollo'.
- Calcula el "merged_qty" de manera inteligente: Si ambos tienen unidades compatibles (ej "1 lb" y "2 lb"), súmalos ("3 lb"). Si las unidades son incompatibles o vagas (ej "1 lb" y "2 unidades"), concaténalos ("1 lb + 2 unidades").
- Si un ingrediente NO tiene otros pares similares en esta lista, NO lo incluyas en tu respuesta.
- Retorna SOLO los clusters que resulten en la fusión de 2 o más ítems de la lista.

Lista de ingredientes:
{json.dumps(items_for_prompt, ensure_ascii=False)}"""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0.0,
            google_api_key=os.environ.get("GEMINI_API_KEY")
        ).with_structured_output(SemanticDedupResult)
        
        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=3)
        )
        def _invoke():
            return llm.invoke(prompt)
            
        response = _invoke()
        if hasattr(response, "clusters") and response.clusters:
            clusters = [c.model_dump() for c in response.clusters if len(c.item_ids_to_merge) > 1]
            logger.info(f"✅ [SEMANTIC DEDUP] Encontrados {len(clusters)} grupos semánticos aptos para fusión.")
            return clusters
        return []
    except Exception as e:
        logger.error(f"❌ [SEMANTIC DEDUP] Error de LLM agrupando ingredientes: {e}")
        return []

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


def get_deterministic_variety_prompt(history_text: str, form_data: dict = None, user_id: str = None) -> str:
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
                    expiry_dt = datetime.fromisoformat(expiry_iso.replace("Z", "+00:00"))
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
            
    # ======= SHOPPING CYCLE LOCK (Ahorro de Supermercado) =======
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
                shopping_cycle = hp.get("shopping_cycle")
                
                now = datetime.now(timezone.utc)
                
                if shopping_cycle and "start_date" in shopping_cycle:
                    try:
                        cycle_start = datetime.fromisoformat(shopping_cycle["start_date"].replace("Z", "+00:00"))
                        days_elapsed = (now - cycle_start).days
                        
                        # Si es < 2 días, es regeneración del mismo plan base, actualizaremos el ciclo.
                        if 2 <= days_elapsed < grocery_days:
                            # ¡BLOQUEO ACTIVO! Forzamos la reutilización de ingredientes.
                            cycle_locked = True
                            unique_proteins = shopping_cycle.get("base_proteins", unique_proteins)
                            unique_carbs = shopping_cycle.get("base_carbs", unique_carbs)
                            unique_veggies = shopping_cycle.get("base_veggies", unique_veggies)
                            logger.info(f"🔒 [SHOPPING CYCLE LOCK] Reutiizando ingredientes del ciclo (Día {days_elapsed} de {grocery_days}).")
                        elif days_elapsed >= grocery_days:
                            logger.info(f"🔓 [SHOPPING CYCLE] Ciclo expirado ({days_elapsed} >= {grocery_days} días). Iniciando nuevo ciclo.")
                            new_cycle_started = True
                        else:
                            logger.info(f"🔄 [SHOPPING CYCLE] Regeneración en Día {days_elapsed} del ciclo. Actualizando Plan Base.")
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
                    if shopping_cycle and "start_date" in shopping_cycle and not (days_elapsed >= grocery_days if 'days_elapsed' in locals() else True):
                        start_date_to_save = shopping_cycle["start_date"]
                        
                    hp["shopping_cycle"] = {
                        "start_date": start_date_to_save,
                        "duration_days": grocery_days,
                        "base_proteins": unique_proteins,
                        "base_carbs": unique_carbs,
                        "base_veggies": unique_veggies
                    }
                    update_user_health_profile(user_id, hp)
                    logger.info("💾 [SHOPPING CYCLE] Guardados nuevos ingredientes base del ciclo.")
        except Exception as e:
            logger.error(f"Error procesando Shopping Cycle Lock: {e}")
    # ==========================================================

    # ======= FORCED SHOPPING LIST INJECTION =======
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

    # Mezclar ANTES de rellenar o truncar, para asegurar rotación de todos los items en la lista de compras
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
        blocked_text += "\n🛒 COMPRA MENSUAL: El usuario compra para 30 días. Prioriza ingredientes de larga duración (tubérculos, granos secos, proteínas congelables). Evita depender de perecederos de vida corta."
    elif grocery_duration == "biweekly":
        blocked_text += "\n🛒 COMPRA QUINCENAL: El usuario compra para 15 días. Equilibra frescos con ingredientes duraderos. Sugiere congelación para proteínas frescas."
        
    if cycle_locked:
        # Use a safe fallback for days_elapsed in case it wasn't defined perfectly
        d_elapsed = locals().get('days_elapsed', '?')
        blocked_text += f"\n\n🚨 [REGLA DE AHORRO EXTREMA]: El usuario está en el Día {d_elapsed} de su ciclo de compras de {grocery_days} días. TIENES LA OBLIGACIÓN ESTRICTA de basar todas las comidas en usar EXACTAMENTE las proteínas, carbohidratos y vegetales asignados explícitamente en el prompt. Usa diferentes preparaciones y técnicas de cocción para que no se aburra, pero NO MANDES A COMPRAR ALIMENTOS BASE NUEVOS."
    
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




