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
    # [P2-SEEDER-DAYS-COUNT · 2026-08-03] plantilla parametrizada por días del chunk. Sustituye
    # al import de `DETERMINISTIC_VARIETY_PROMPT`, que se quedó sin consumo aquí (sigue exportado
    # por `prompts/__init__` como la instancia de 3 días).
    build_deterministic_variety_prompt,
    option_letter as _prompt_option_letter,
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
    _get_fast_filtered_catalogs,
    # [P2-SEEDER-DAYS-COUNT · 2026-08-03] techo del reparto = el mismo cap que el orquestador
    # aplica a `_days_to_generate` (2×PLAN_CHUNK_SIZE).
    PLAN_CHUNK_SIZE,
)
from db import (get_user_profile, update_user_health_profile, update_user_health_profile_atomic,
                get_user_ingredient_frequencies,
                # [P1-CYCLE-BASE-AFFINITY · 2026-08-02] la compra persistida del ciclo vigente.
                get_latest_meal_plan_with_id)
from cpu_tasks import _calcular_frecuencias_regex_cpu_bound
from knobs import _env_str, _env_float, _env_bool, _env_int  # [P3-FLASH-LITE-COST-CUT · 2026-05-21] / [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] / [P3-GAINMUSCLE-PROTEIN-DENSITY · 2026-06-23] / [P2-PANTRY-ROTATION-FLOOR · 2026-07-29]

logger = logging.getLogger(__name__)

# [P2-FREQ-LOOKUP-CANONICAL · 2026-07-29] (audit solver+seeder v4) El lookup de frecuencias del
# sorteo usa TAMBIÉN la clave canónica de `normalize_ingredient_for_tracking` (la misma con la que
# se ESCRIBE la tabla), no solo los alias. Sin esto, 19 de los 145 ítems de los pools tenían
# `freq=0` permanente — el peso MÁXIMO — porque su clave de tracking es OTRO ítem del pool.
# Rollback sin redeploy: MEALFIT_FREQ_LOOKUP_CANONICAL=false → lookup solo-por-alias (previo).
FREQ_LOOKUP_CANONICAL = _env_bool("MEALFIT_FREQ_LOOKUP_CANONICAL", True)
# [P2-FREQ-TRACKING-CHUNKED · 2026-07-29] (audit solver+seeder v4) El tracking de frecuencias vivía
# SOLO en la rama no-chunked. Como `use_chunking = perfil AND días > PLAN_CHUNK_SIZE(3)`, TODO plan
# de 7 días de un usuario con perfil caía en la rama que NO trackea → la tabla nunca crecía y el
# sorteo perdía su señal. Medido en Neon 2026-07-29: 0 filas cruzando con planes; 30 planes creados
# desde el último `last_used`. Rollback sin redeploy: MEALFIT_TRACK_FREQ_ON_CHUNKED=false.
TRACK_FREQ_ON_CHUNKED = _env_bool("MEALFIT_TRACK_FREQ_ON_CHUNKED", True)
# [P2-PANTRY-ROTATION-FLOOR · 2026-07-29] (audit solver+seeder v4) Mínimo de proteínas DISTINTAS que
# la nevera debe aportar para que el modo rotación reemplace el pool y active `cycle_locked`. Por
# debajo, la nevera va primero pero se completa con el sorteo (y sin lock) — si no, una nevera con
# una sola proteína produce los 3 días con la misma base, violando el cap de huevo del propio prompt.
# `=1` restaura exactamente el comportamiento previo. Rollback sin redeploy.
PANTRY_ROTATION_MIN_PROTEINS = max(1, min(3, _env_int("MEALFIT_PANTRY_ROTATION_MIN_PROTEINS", 2,
                                                      lambda v: 1 <= v <= 3)))
# [P1-PANTRY-FLOOR-CLINICAL-FILTER · 2026-08-02] (audit solver+seeder v7) Lo extraído de la nevera
# pasa por los MISMOS filtros de "puede ser base principal" que el sorteo (embutidos por goal,
# baja densidad, curados en sal, frutas alto-IG en bariátrica) ANTES de decidir si sostiene la
# rotación. Sin esto, una nevera de salami + longaniza se convertía en la base OBLIGATORIA de los
# 3 días con `cycle_locked` puesto → rechazo clínico determinista y retries quemados.
# `False` restaura exactamente el comportamiento previo. Rollback sin redeploy.
PANTRY_FLOOR_CLINICAL_FILTER = _env_bool("MEALFIT_PANTRY_FLOOR_CLINICAL_FILTER", True)
# [P2-LIGHT-PROTEIN-SEED · 2026-07-29] (audit solver+seeder v4) Sortea el ancla proteica de
# desayuno/merienda (última categoría sin sorteo: 5 viñetas literales iguales para todos los planes).
# Nace OFF esperando A/B — con el knob apagado el prompt queda BYTE-IDÉNTICO (bloque vacío).
LIGHT_PROTEIN_SEED = _env_bool("MEALFIT_LIGHT_PROTEIN_SEED", False)


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


def _build_expand_llm(modelo: str, **kw):
    """[P1-RECIPE-EXPAND-MODEL-PROVIDER · 2026-07-26] Cliente del proveedor que corresponde al
    modelo, para que `MEALFIT_RECIPE_EXPAND_MODEL` pueda apuntar a un modelo OpenAI.

    Este nodo es el mejor sitio del pipeline para pagar un modelo mejor, y por eso se abre:
      · lo dispara el USUARIO sobre UN plato ("regenera para más detalle"), no corre en cada plan;
      · es exactamente donde sale el badge `_dish_quality_degraded` — la receta que quedó pobre;
      · la llamada es diminuta (una receta), así que el premium se mide en céntimos.

    ⚠️ Sin este dispatch, apuntar el knob a `gpt-5.6-luna` mandaba el modelo al base_url de DeepSeek
    con la key equivocada. Es el mismo fallo que P1-LUNA-USAGE-BLIND cerró en el day-gen; aquí el
    `ChatDeepSeek` a secas lo tenía latente desde que el knob existe.

    ⚠️ `with_structured_output` NO se aplica aquí a propósito: `ChatDeepSeek` lo override-a para las
    rarezas de DeepSeek (`function_calling` en vez de `json_schema`) y OpenAI quiere el default de
    langchain. Lo pone el caller sobre el cliente que reciba.
    """
    try:
        from llm_provider import is_openai_model
        if is_openai_model(modelo):
            from graph_orchestrator import ChatOpenAIInstrumented
            return ChatOpenAIInstrumented(model=modelo, **kw)
    except Exception as _e:
        # fail-cheap: ante cualquier duda, el proveedor barato de siempre
        logger.warning(f"[P1-RECIPE-EXPAND-MODEL-PROVIDER] dispatch falló ({type(_e).__name__}), "
                       f"usando DeepSeek: {str(_e)[:120]}")
    return ChatDeepSeek(model=modelo, **kw)


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


# [P2-OVERUSE-RAW-FREQ · 2026-08-03] El veto textual «EVITA usar como base principal»
# (`OVERUSE_THRESHOLD`, más abajo en `get_deterministic_variety_prompt`) comparaba
# `freq >= 3` contra `db_freq_map` YA fatigado (`_apply_recency_fatigue`,
# `recent_3d×3.0 + recent_7d×1.5`): un ingrediente comido UNA sola vez ayer da
# `1 + 1*3.0 + 1*1.5 = 5.5 >= 3` y entra al veto textual, aunque el comentario de
# calibración del umbral diga explícitamente que 1-2 usos NO deben marcarse "PROHIBIDOS"
# (el soft-penalty `1/(freq+1)` de los PESOS ya castiga lo suficiente). En planes de
# 15/30 días esto vetaba en el prompt los staples RECIÉN COMPRADOS.
#
# `True` (default): `used_proteins/used_carbs/used_veggies` se computan desde la
# frecuencia CRUDA (snapshot pre-fatiga). Los PESOS del sorteo (`1/(freq+1)`) siguen
# leyendo el mapa fatigado SIN CAMBIOS — la fatiga sigue siendo correcta para sesgar la
# lotería; lo incorrecto era usarla también para el umbral binario de veto. `False`
# restaura el comportamiento previo exacto (veto sobre fatigado) — rollback sin redeploy.
# tooltip-anchor: P2-OVERUSE-RAW-FREQ
OVERUSE_ON_RAW_FREQ = _env_bool("MEALFIT_OVERUSE_ON_RAW_FREQ", True)


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

# ─────────── vocabularios clínicos del seeder (nivel módulo = SSOT único) ───────────
# [P1-PANTRY-FLOOR-CLINICAL-FILTER · 2026-08-02] (audit solver+seeder v7) Estos cuatro
# vocabularios vivían DENTRO de `get_deterministic_variety_prompt`, y dos de ellos dentro de un
# `if` (`_SALT_CURED_PROTEIN_TOKENS` bajo `if _sb_penalty < 1.0`, `_HIGH_GI_FRUITS` bajo
# `if _is_bariatric`), así que el bloque de la NEVERA —600 líneas más abajo, misma función— ni
# siquiera podía leerlos sin arriesgar un `NameError`. Subirlos aquí es lo que permite que el
# filtro de la nevera reuse los MISMOS conjuntos que los penalties del sorteo en vez de escribir
# una quinta lista a mano (este repo ya arrastra cuatro copias del vocabulario curado y el
# historial de drift que eso produce). Cero cambio de contenido en la subida.

# Embutidos: procesados con sodio alto y grasas saturadas. Apropiados ocasionalmente en perfiles
# 'balanced', contraindicados como base recurrente en ganancia muscular limpia / pérdida de grasa.
_PROCESSED_MEAT_KEYWORDS = (
    "salami", "longaniza", "jamón", "jamon", "chorizo",
    "tocineta", "tocino", "salchichón", "salchichon", "salchicha",
    "mortadela", "embutido",
)
# `_GOALS_PENALIZE_PROCESSED` NO sube aquí a propósito: sigue viviendo dentro de
# `get_deterministic_variety_prompt`, junto al penalty del sorteo que decide con él y junto a
# `_GOALS_FORCE_MAX_VARIETY`, que `test_p2_seeder_pairs_goals` valida contra `_MAIN_GOAL_ENUM`
# leyendo el cuerpo de esa función. Por eso `_pantry_clinical_main_filter` recibe DECISIONES ya
# tomadas (`penaliza_procesados` / `exige_densidad`) en vez de re-derivarlas: las dos condiciones
# quedan escritas una al lado de la otra en el mismo cuerpo y no pueden drifear entre capas.
# [P1-SODIUM-BOMB-POOL · 2026-07-05] Proteínas CURADAS EN SAL — la proteína ES sal: un solo día
# con bacalao o salami revienta el techo OMS de 2000mg. Penalty universal en el sorteo (ver el
# call site) y, desde P1-PANTRY-FLOOR-CLINICAL-FILTER, criterio del filtro de la nevera.
_SALT_CURED_PROTEIN_TOKENS = ("bacalao", "arenque", "salami", "salchichon", "pepperoni",
                              "mortadela", "tocino", "panceta", "longaniza", "chorizo",
                              "salchicha", "embutido", "jamon")
# [P1-BARIATRIC-PROTEIN-DENSITY · 2026-06-27] Frutas de ALTO índice glucémico: el revisor médico
# rechazaba mango (clash) y guineo en porción grande por dumping (corr=5ffd78cf).
_HIGH_GI_FRUITS = ("guineo", "banana", "mango", "uva", "pina", "platano", "melon", "sandia",
                   "tamarindo")
# [P1-PANTRY-FLOOR-CLINICAL-FILTER · 2026-08-02] "Curado o embutido" como UN concepto, DERIVADO
# de los dos vocabularios de arriba en vez de escrito por quinta vez. Los dos se solapan casi por
# completo pero no del todo (`tocineta` solo está en el de embutidos), y una lista nueva a mano
# garantizaría que la próxima incorporación entre en una y no en la otra.
_CURED_OR_PROCESSED_TOKENS = tuple(sorted(
    {strip_accents(str(t).lower()) for t in _SALT_CURED_PROTEIN_TOKENS}
    | {strip_accents(str(t).lower()) for t in _PROCESSED_MEAT_KEYWORDS}
))


def _token_matches_wb(name, tokens) -> bool:
    """¿`name` contiene alguno de `tokens` como PALABRA COMPLETA?

    [P1-PANTRY-FLOOR-CLINICAL-FILTER · 2026-08-02] Word-boundary, no subcadena. El repo lleva
    13+ incidentes de esta clase (`"sal"`⊂`"Salami"`, `"res"`⊂`"fresas"`, `"pollo"`⊂`"repollo"`,
    `"molida"`⊂`"linaza molida"`) y aquí hay uno REAL medido sobre el catálogo:
    `"pina"` (token de «Piña», alto IG) ⊂ `"es**pina**cas"`. Un filtro por subcadena marcaría
    las Espinacas como fruta de alto índice glucémico. Mismo patrón canónico que ya usan
    `cpu_tasks.py`, el `fast_regex` de `constants.py` y `_pantry_pick` de este módulo:
    `strip_accents` en los DOS lados + `\\b…\\b`."""
    _n = strip_accents(str(name or "").lower())
    if not _n:
        return False
    for t in (tokens or ()):
        _t = strip_accents(str(t or "").lower()).strip()
        if _t and re.search(r'\b' + re.escape(_t) + r'\b', _n):
            return True
    return False


def _is_low_density_main(name, _is_bariatric: bool) -> bool:
    """¿`name` es una proteína que NO debe ocupar el slot de proteína PRINCIPAL?

    [P1-PANTRY-FLOOR-CLINICAL-FILTER · 2026-08-02] Cuerpo subido desde el closure
    `_should_replace_main` de `get_deterministic_variety_prompt` (que ahora delega aquí) para
    que el filtro de la nevera aplique EXACTAMENTE el mismo criterio que el sorteo — eran las
    dos mitades de la misma regla y solo una corría sobre lo extraído de la nevera.
    Los dos sets son de EXACT-MATCH a propósito (P2-9 / P1-BARIATRIC-DENSE-ANCHOR: 'Yogurt' sí,
    'Yogurt griego entero' no); solo los embutidos matchean por token — y ese match pasa de `in`
    crudo a word-boundary. Medido sobre `DOMINICAN_PROTEINS` completo: cero divergencia entre los
    dos operadores, así que es no-op para el sorteo (que solo ve nombres del catálogo) y cierra
    la superficie de subcadena para el filtro de la nevera.
    El parámetro se llama `_is_bariatric` para conservar intacto el anclaje textual de
    `test_p1_bariatric_dense_anchor::test_branch_present_and_knob_reused`."""
    _pl = strip_accents(str(name or "").lower())
    if _pl in _LOW_DENSITY_AS_MAIN:
        return True
    if _is_bariatric and _pl in _BARIATRIC_LOW_DENSITY_AS_MAIN:  # [P1-BARIATRIC-DENSE-ANCHOR] quesos-relleno
        return True
    if _is_bariatric and _token_matches_wb(_pl, _PROCESSED_MEAT_KEYWORDS):
        return True
    return False


def _pantry_clinical_main_filter(extracted_p, extracted_f, *, penaliza_procesados: bool = False,
                                 exige_densidad: bool = False, is_bariatric: bool = False):
    """[P1-PANTRY-FLOOR-CLINICAL-FILTER · 2026-08-02] (audit solver+seeder v7)

    Devuelve `(proteinas, frutas)` de la nevera que SÍ pueden ser BASE del día.

    El seeder aplica sus penalties clínicos sobre los PESOS del sorteo (embutidos ×0.1 por goal,
    curados en sal ×0.1 universal, reemplazo de mains de baja densidad para gain_muscle /
    bariátrica, frutas alto-IG ×0.15). Cientos de líneas después el modo rotación REEMPLAZA el
    pool por lo extraído de la nevera y activa `cycle_locked` ("NO SUGIERAS ALIMENTOS BASE
    NUEVOS"): los penalties quedaban íntegramente bypaseados. Una nevera con «Salami
    Dominicano» + «Longaniza» producía literalmente `['Salami Dominicano', 'Longaniza', 'Salami
    Dominicano']` como bases obligatorias de los 3 días → rechazo determinista del revisor
    clínico → el retry re-corre el seeder con la MISMA nevera → retries quemados. Misma clase
    que P1-SODIUM-BOMB-POOL y P1-FRUIT-SEEDER-GATE-CONTRACT.

    `penaliza_procesados` y `exige_densidad` los decide el CALLER, con las mismas condiciones
    exactas que gobiernan los penalties del sorteo (que están escritos en el mismo cuerpo, unos
    cientos de líneas más arriba). Re-derivarlas aquí crearía dos copias de la misma regla
    clínica en capas distintas — justo el drift que este P-fix viene a cerrar.

    Tres reglas, en este orden:

      1. Goal que penaliza procesados (o perfil bariátrico) ⇒ embutidos y proteínas de baja
         densidad salen del pool de MAINS. NO desaparecen de la nevera: el prompt de nevera
         los sigue ofreciendo como acompañante/saborizante — solo dejan de ser la base del día.
      2. 100% de lo extraído curado/embutido ⇒ pool de nevera VACÍO, para NINGÚN goal se activa
         el lock. El presupuesto de sodio de la OMS no depende del objetivo, y un lock sobre
         puros embutidos es el peor caso posible (obligatorio + irreparable por el LLM).
      3. Espejo en frutas alto-IG SOLO si bariátrica, que es el único perfil donde hoy existe
         ese penalty. No se inventa una regla nueva para otros perfiles.

    Solo QUITA de lo que recibe (subconjunto ordenado): jamás puede reintroducir un alimento
    que alergia/dieta/dislike ya excluyeron aguas arriba (`_pantry_pick` filtra contra los
    `filtered_*`). Nunca reordena — la prioridad de ahorro de la nevera se conserva."""
    _p = list(extracted_p or [])
    _f = list(extracted_f or [])

    # 1 · embutidos / baja densidad fuera del slot de MAIN. Cada mitad se aplica solo si el
    #     caller la activó — extenderlas por cuenta propia (p.ej. sacar leguminosas también en
    #     `lose_fat`, donde el sorteo NO las reemplaza) sería inventar una regla clínica nueva
    #     por el camino en vez de cerrar el bypass.
    if _p and (penaliza_procesados or exige_densidad):
        _kept = [x for x in _p
                 if not (penaliza_procesados and _token_matches_wb(x, _PROCESSED_MEAT_KEYWORDS))
                 and not (exige_densidad and _is_low_density_main(x, is_bariatric))]
        if len(_kept) < len(_p):
            logger.info(
                f"🩺 [P1-PANTRY-FLOOR-CLINICAL-FILTER] {len(_p) - len(_kept)} proteína(s) de la "
                f"nevera fuera de las BASES clínicas"
                f"{' (bariátrica)' if is_bariatric else ''}: "
                f"{[x for x in _p if x not in _kept]} — siguen disponibles como acompañante.")
            _p = _kept

    # 2 · 100% curado/embutido ⇒ sin bases propias y sin lock, para cualquier goal
    if _p and all(_token_matches_wb(x, _CURED_OR_PROCESSED_TOKENS) for x in _p):
        logger.info(
            f"🧂 [P1-PANTRY-FLOOR-CLINICAL-FILTER] la nevera solo aporta proteína curada/embutida "
            f"({_p}) → NO sostiene la rotación: el pool se sortea completo y no se activa el "
            f"cycle-lock (evita 3 días de embutido obligatorio).")
        _p = []

    # 3 · espejo de frutas alto-IG (solo bariátrica)
    if _f and is_bariatric:
        _kept_f = [x for x in _f if not _token_matches_wb(x, _HIGH_GI_FRUITS)]
        if len(_kept_f) < len(_f):
            logger.info(
                f"🍌 [P1-PANTRY-FLOOR-CLINICAL-FILTER] {len(_f) - len(_kept_f)} fruta(s) de alto "
                f"índice glucémico fuera de las asignadas por perfil bariátrico: "
                f"{[x for x in _f if x not in _kept_f]}.")
            _f = _kept_f

    return _p, _f


# ─────── [P3-SEEDER-TEMPLATE-COVERAGE · 2026-08-04] cobertura de plantillas por base ───────
# (audit solver+seeder v7 · Task 19)
#
# El sorteo repartía bases sin saber cuántas variantes de plato REALES soporta cada una.
# `dish_library` le ofrece al day-generator ~87 plantillas dominicanas curadas, pero
# `_protein_matches_pool` sólo deja pasar las que NOMBRAN la base asignada. Bases como
# Chivo/Conejo/Pulpo/Bacalao no aparecen en NINGUNA plantilla de almuerzo: cuando el sorteo
# —o peor, la nevera bajo `cycle_locked`— se las impone 2-3 días del chunk, el day-gen compone
# esos platos SIN recombinación y tiende a la fórmula clonada (que hoy sólo se mide DESPUÉS,
# con el detector de P1-CONTAINER-SERVABLE).
#
# El audit lo ajustó a la baja a propósito: las plantillas de clase genérica aplican a todo
# pool, así que esto es **degradación suave, no bloqueo**. Por eso el cierre es un multiplicador
# ×0.5 sobre el peso del sorteo —jamás una exclusión— más un WARNING cuando la base viene de la
# nevera y va a ocupar ≥2 días. tooltip-anchor: P3-SEEDER-TEMPLATE-COVERAGE

# Slot PRINCIPAL: el almuerzo es donde aterriza la base proteica/carbo del día (la cena reusa el
# mismo par, ver P1-CARB-SEEDER-PAIRS). Medir ahí es medir dónde duele la falta de plantilla.
_TEMPLATE_COVERAGE_MAIN_SLOT = "almuerzo"
# 'none'/'mixta' aplican a TODO pool y no nombran ningún alimento: no distinguen a nadie, así que
# no pueden contar como cobertura de nada (si contaran, ninguna base daría 0 y el fix sería inerte).
_TEMPLATE_CLASS_NAMELESS = ("none", "mixta")
# Las otras tres clases genéricas de `_protein_matches_pool` SÍ nombran una clase concreta: una
# plantilla de huevo habla del huevo. 'huevo'/'queso' alcanzan sus bases por límite de palabra;
# 'legumbre' es un nombre de CATEGORÍA, no de alimento, así que se puentea con el SSOT
# `constants.NUTRITIONAL_CATEGORIES` en vez de escribir una lista de leguminosas a mano (que
# drifearía contra `LEGUME_NAMES` y contra la GARANTÍA de leguminosa del propio seeder).
_TEMPLATE_CLASS_CATEGORY = {
    "huevo": "huevos y lácteos",
    "queso": "huevos y lácteos",
    "legumbre": "legumbres",
}
# (slot, campo) -> (id(lista_de_plantillas), {token: nº}). La clave lleva el `id` de la lista que
# `dish_library` cachea, así que un reload/stub de la biblioteca invalida la entrada solo.
_TEMPLATE_COVERAGE_CACHE: dict = {}


def _template_token_names_base(token: str, base_ascii: str, base_lower: str) -> bool:
    """¿La clase de plato `token` NOMBRA a esta base?

    MISMA forma que la rama no-genérica de `dish_library._protein_matches_pool`: `\\b` al inicio y
    NADA al final ('huevo' debe alcanzar 'Huevos' y 'papa' a 'Papas'). Word-boundary y no
    subcadena — este repo lleva 14+ incidentes de esa clase y aquí hay dos MEDIDOS contra los
    catálogos vivos: `'pollo'` ⊂ `'re**pollo**'` (18 plantillas heredadas) y `'res'` ⊂
    `'f**res**a'` (6). La paridad con el matcher de la biblioteca está anclada por test: si
    divergen, esta cobertura mediría una biblioteca que el day-gen no ve.
    tooltip-anchor: P3-SEEDER-TEMPLATE-COVERAGE"""
    if not token or not base_ascii:
        return False
    if re.search(r"\b" + re.escape(token), base_ascii):
        return True
    _cat = _TEMPLATE_CLASS_CATEGORY.get(token)
    if not _cat:
        return False
    try:
        from constants import get_nutritional_category
        return get_nutritional_category(base_lower) == _cat
    except Exception:
        return False


def _dish_template_class_counts(slot: str, field: str) -> dict:
    """{token_de_clase: nº de plantillas} del slot, DERIVADO del JSON vivo. Fail-open → {}."""
    try:
        from dish_library import load_dish_templates
        _tpls = load_dish_templates() or []
    except Exception as _tc_e:
        logger.debug("[P3-SEEDER-TEMPLATE-COVERAGE] biblioteca no disponible: %s: %s",
                     type(_tc_e).__name__, str(_tc_e)[:160])
        return {}
    _key = (slot, field)
    _hit = _TEMPLATE_COVERAGE_CACHE.get(_key)
    if _hit is not None and _hit[0] == id(_tpls):
        return _hit[1]
    _counts: dict = {}
    for _t in _tpls:
        if not isinstance(_t, dict) or slot not in (_t.get("slots") or []):
            continue
        _tok = strip_accents(str(_t.get(field) or "none").lower()).strip()
        if not _tok or _tok in _TEMPLATE_CLASS_NAMELESS:
            continue
        _counts[_tok] = _counts.get(_tok, 0) + 1
    if len(_TEMPLATE_COVERAGE_CACHE) > 32:      # cota anti-bloat si alguien stubea la biblioteca
        _TEMPLATE_COVERAGE_CACHE.clear()
    _TEMPLATE_COVERAGE_CACHE[_key] = (id(_tpls), _counts)
    return _counts


def _template_coverage(base: str, field: str = "protein", slot: str = None) -> int:
    """Nº de plantillas del slot principal que NOMBRAN a `base`.

    `field` es `'protein'` para el pool de proteínas y `'base'` para el de carbos (los dos campos
    que la plantilla declara). Devuelve **-1** cuando no hay biblioteca legible: "no sé" NO es
    "cero" — un 0 ahí penalizaría el pool entero por igual y disfrazaría la avería de decisión."""
    _counts = _dish_template_class_counts(slot or _TEMPLATE_COVERAGE_MAIN_SLOT, field)
    if not _counts:
        return -1
    _ascii = strip_accents(str(base or "").lower()).strip()
    if not _ascii:
        return -1
    _lower = str(base or "").strip().lower()
    return sum(_n for _tok, _n in _counts.items()
               if _template_token_names_base(_tok, _ascii, _lower))


def _low_template_coverage_penalty() -> float:
    """Multiplicador del sorteo para las bases sin ninguna plantilla propia. `1.0` = OFF.

    Se lee en CADA llamada (no a nivel módulo) para que el rollback no necesite redeploy, igual
    que `MEALFIT_CYCLE_BASE_AFFINITY`. Clamp [0.1, 1.0]: el piso impide que alguien convierta un
    sesgo en una exclusión de facto escribiendo 0."""
    try:
        return min(1.0, max(0.1, _env_float(
            "MEALFIT_LOW_TEMPLATE_COVERAGE_PENALTY", 0.5, lambda v: 0.1 <= v <= 1.0)))
    except Exception:
        return 0.5


def _apply_low_template_coverage_penalty(names, weights, field: str, factor: float):
    """Devuelve `(pesos, nº penalizados)`. Multiplicador SUAVE, jamás exclusión.

    Sólo baja el peso de las bases con cobertura EXACTAMENTE 0; el resto queda byte-idéntico (es
    un sesgo, no una redistribución). Sin biblioteca legible es no-op: `_template_coverage`
    devuelve -1 y ninguna base entra."""
    _w = list(weights)
    if factor >= 1.0 or not names:
        return _w, 0
    if not _dish_template_class_counts(_TEMPLATE_COVERAGE_MAIN_SLOT, field):
        return _w, 0
    _out, _n = [], 0
    for _name, _wi in zip(names, _w):
        if _template_coverage(_name, field) == 0:
            _out.append(_wi * factor)
            _n += 1
        else:
            _out.append(_wi)
    return _out, _n


# ─────── [P3-BUDGET-POOL-FLOOR · 2026-08-04] piso del boost económico del sorteo ───────
# (audit solver+seeder v7 · Task 20)
#
# `P1-BUDGET-TIER-LEVERS` multiplica ×2 el peso de las proteínas/carbos del TERCIO MÁS BARATO
# cuando el presupuesto pide economía. Pero el peso base es `1/(freq+1)`: un staple barato comido
# 33 veces pesa 1/34 = 0,029 y ×2 sigue siendo 0,059 contra el 1,0 de un premium fresco — 17:1 EN
# CONTRA del barato. El caso general está sano (un barato NO reciente pesa 2,0 y gana); el residuo
# es el ciclo de 15/30 días con pool barato chico, donde el tercio barato ENTERO se fatiga y el
# sorteo se va a los premium — que el cheapen-pass de `assemble` corrige después con churn y
# colapso de variedad. El cheapen-pass sigue ahí como RED; esto evita tener que usarla.
#
# El piso es RELATIVO a la mediana de los pesos del pool, no absoluto: los pesos dependen del
# historial de cada usuario, así que un número fijo no significaría lo mismo en dos pools.
# tooltip-anchor: P3-BUDGET-POOL-FLOOR


def _bw_median(values) -> float:
    """Mediana (idéntica a `statistics.median`, sin importar el módulo por un solo uso)."""
    _v = sorted(float(x) for x in (values or []))
    if not _v:
        return 0.0
    _n = len(_v)
    return _v[_n // 2] if _n % 2 else (_v[_n // 2 - 1] + _v[_n // 2]) / 2.0


def _budget_pool_floor() -> float:
    """Fracción de la mediana del pool por debajo de la cual no cae un ítem del tercio barato.

    `0.0` = OFF (el sorteo vuelve a ser byte-idéntico al de P1-BUDGET-TIER-LEVERS). El tope 1.5
    existe para poder pasar la mediana cuando el dueño quiera reuso agresivo; con el default 0.8
    el piso queda POR DEBAJO de la mediana por construcción, así que un barato fatigado NO
    adelanta a un premium fresco — sesgo, no lock. Se lee en CADA llamada (rollback sin redeploy).
    """
    try:
        return min(1.5, max(0.0, _env_float(
            "MEALFIT_BUDGET_POOL_FLOOR", 0.8, lambda v: 0.0 <= v <= 1.5)))
    except Exception:
        return 0.8


def _budget_boost_with_floor(weights, prices, boost: float, floor: float):
    """Boost económico del tercio más barato CON piso relativo. Devuelve pesos nuevos.

    `prices` viene alineado con `weights` (None = precio no resoluble ⇒ peso intacto: sin precio
    no se puede afirmar que el ítem sea barato). Conserva intacto el contrato heredado de
    P1-BUDGET-TIER-LEVERS: con menos de 4 precios resolubles no hay tercil que calcular y los
    pesos se devuelven sin tocar.

    El piso se calcula sobre la mediana de los pesos ANTES del boost — o sea, sobre la escala de
    fatiga del pool tal como quedó. Y es un `max`, nunca un reemplazo: un barato FRESCO conserva
    su ×2 en vez de perderlo contra el piso."""
    _w = list(weights)
    _valid = sorted(_p for _p in prices if _p and _p > 0)
    if len(_valid) < 4:
        return _w
    _p33 = _valid[max(0, int(0.33 * (len(_valid) - 1)))]
    _floor_w = (_bw_median(_w) * floor) if floor > 0 else 0.0
    return [
        (max(_wi * boost, _floor_w) if (_p and _p <= _p33) else _wi)
        for _wi, _p in zip(_w, prices)
    ]


# [P1-FRUIT-SEEDER-GATE-CONTRACT · 2026-07-26] El seeder y el gate de variedad hablaban vocabularios
# distintos: de las 30 frutas del catálogo el gate reconocía 16, así que un pool de 3 salía sin
# ninguna reconocida el 9% de las veces y con ≤1 el 44,8%. Estos dos helpers cierran el contrato por
# el lado del seeder. El vocabulario vive en `graph_orchestrator._FEATURED_FRUITS` (SSOT del gate):
# se importa en lugar de duplicarlo, porque una copia divergiría en el primer alimento nuevo.
def _n_gate_fruits(fruits) -> int:
    """Cuántas frutas del pool cuenta el gate de repetición intra-día. Fail-safe → 0."""
    try:
        from graph_orchestrator import _featured_fruits_in_name as _ffn
        return sum(1 for f in (fruits or []) if _ffn(f))
    except Exception:
        return 0


def _rotate_fruit_pairs(fruits, days: int = 3):
    """Reparte el pool en (fruta_a, fruta_b) por día — DOS distintas cada día.

    Día i recibe `(fruits[i], fruits[i+1])` sobre la rotación, así que 4 frutas distintas cubren 3
    días con dos por día en vez de comprar 6. Prioriza las que el gate reconoce: si el pool trae
    níspero y guineo, el guineo va primero porque una repetición de níspero era INVISIBLE para el
    gate y por tanto no ayudaba a satisfacerlo.

    Devuelve `None` si no hay al menos 2 frutas utilizables — el caller cae al texto libre en vez de
    inventar un pool. tooltip-anchor: P1-FRUIT-SEEDER-GATE-CONTRACT"""
    try:
        from graph_orchestrator import _featured_fruits_in_name as _ffn
    except Exception:
        def _ffn(_x):
            return {"?"}
    base = [f for f in (fruits or []) if f and str(f).strip()]
    if len(base) < 2:
        return None
    # reconocidas primero, conservando el orden relativo (el shuffle previo ya dio la aleatoriedad)
    ordenadas = [f for f in base if _ffn(f)] + [f for f in base if not _ffn(f)]
    # [P2-SEEDER-DAYS-COUNT · 2026-08-03] `days` se delega íntegro al helper compartido.
    return _rotate_pairs(ordenadas, days=days)


# [P1-BARIATRIC-PROTEIN-DENSITY] Nueces/semillas ENTERAS: riesgo OBSTRUCTIVO del pouch. Las formas
# molida / mantequilla / fileteada NO cuentan (ya son seguras).
# [P2-LIGHT-PROTEIN-POOL · 2026-07-30] Promovido a módulo: era local del pool de variedad y el
# sorteo del ancla liviana necesitaba el MISMO universo. Dos listas paralelas de tokens sobre el
# mismo concepto clínico es la receta para que el próximo endurecimiento aterrice en una sola.
_WHOLE_NUT_SEED_TOKENS = ("mani", "almendra", "nuez", "nueces", "pistacho", "maranon", "merey",
                          "avellana", "semilla", "pepita", "chia", "ajonjoli", "sesamo", "linaza")
_SAFE_NUT_FORM_TOKENS = ("mantequilla", "molid", "filetead")
# Tokens del ancla liviana. Los 3 primeros son LÁCTEOS y viven en DOMINICAN_PROTEINS, no en
# DOMINICAN_VEGGIES_FATS — por eso el pool tiene que mirar las DOS listas.
_LIGHT_ANCHOR_TOKENS = ("queso", "yogur", "ricotta", "almendra", "nuez", "nueces",
                        "mani", "mantequilla")


# [P3-CYCLE-BASE-FLOOR · 2026-07-31] (audit v6 · C2) Mínimo de bases DISTINTAS que la intersección
# del grocery-cycle debe sostener para que valga la pena imponer el lock. Bajo esto, el ahorro no
# compensa entregar el mismo alimento todos los días del ciclo. Espejo del floor de la nevera
# (`PANTRY_ROTATION_MIN_PROTEINS`, P2-PANTRY-ROTATION-FLOOR).
_CYCLE_BASE_MIN_ITEMS = _env_int("MEALFIT_CYCLE_BASE_MIN_ITEMS", 2, lambda v: 1 <= v <= 6)


def _intersect_cycle_base(persisted, allowed) -> "list | None":
    """[P3-GROCERY-LOCK-REFILTER · 2026-07-30] (audit solver+seeder v5) Intersecta las bases que
    el grocery-cycle persistió al INICIO del ciclo con el pool permitido de HOY.

    Devuelve `None` (no una lista vacía) cuando no queda nada utilizable, para que el caller
    conserve su sorteo con un `or` y el lock degrade a variedad en vez de imponer un alimento
    que el usuario ya no puede comer. Comparación normalizada sin acentos.
    tooltip-anchor: P3-GROCERY-LOCK-REFILTER"""
    try:
        if not persisted or not isinstance(persisted, (list, tuple)):
            return None
        _ok = {strip_accents(str(a).lower()).strip() for a in (allowed or [])}
        if not _ok:
            return None
        _out = [p for p in persisted if strip_accents(str(p).lower()).strip() in _ok]
        # [P3-CYCLE-BASE-FLOOR · 2026-07-31] (audit solver+seeder v6 · C2) El fail-open solo
        # disparaba con intersección VACÍA. Con UN superviviente la lista de 1 gana el `or` del
        # caller, el padding cíclico la replica a los 3 días y el prompt del lock PROHÍBE bases
        # nuevas: los días restantes del ciclo salen todos con la misma proteína. El bloque de
        # nevera, 140 líneas más abajo, ya había aprendido esta lección (`_floor_pool`,
        # P2-PANTRY-ROTATION-FLOOR) — el floor no llegó a esta superficie hermana, que es la
        # asimetría dominante de este audit. Bajo el mínimo se degrada igual que con vacío.
        # tooltip-anchor: P3-CYCLE-BASE-FLOOR
        if len(_out) < _CYCLE_BASE_MIN_ITEMS:
            return None
        return _out or None
    except Exception:
        return None


def _catalog_pick_wb(item_norm: str, full_catalog, syn_map, allowed) -> "str | None":
    """[P1-PANTRY-EXTRACT-FILTERED-WB · 2026-07-30] Resuelve una línea de texto libre
    ("2 lb de filete de pescado") al alimento del catálogo que nombra.

    [P1-CYCLE-BASE-AFFINITY · 2026-08-02] Subida a nivel de módulo (SSOT único). Vivía como
    closure `_pantry_pick` DENTRO del bloque de nevera, ~280 líneas DESPUÉS del sorteo, así que
    la afinidad de ciclo —que corre ANTES del sorteo— no podía usarla y habría necesitado su
    propia comparación de nombres. Esa cuarta implementación es exactamente la trampa que este
    repo lleva 13+ incidentes pagando (`"sal"`⊂`"Salami"`, `"res"`⊂`"fresas"`,
    `"pollo"`⊂`"repollo"`, `"pina"`⊂`"Espinacas"`). El closure queda delegando aquí.

    Las tres reglas, que hacen falta JUNTAS:
      1. LÍMITE DE PALABRA (`\\b`) sobre texto sin acentos — no subcadena cruda.
      2. Gana el alias MÁS ESPECÍFICO (el más largo) evaluado sobre el catálogo COMPLETO —
         `PROTEIN_SYNONYMS['res']` incluye el alias genérico `'filete'`, que ES palabra completa
         dentro de "filete de pescado".
      3. El ganador solo se devuelve si está en `allowed` (el pool YA filtrado por
         alergia/dieta/dislike). Si el ítem nombra un alimento excluido, NO se degrada a un
         match genérico más débil: simplemente no aporta base.
    tooltip-anchor: P1-CYCLE-BASE-AFFINITY (matcher SSOT)"""
    _best, _len = None, 0
    for _food in full_catalog:
        for s in syn_map.get(_food.lower(), [_food.lower()]):
            _s = strip_accents(str(s).lower()).strip()
            if _s and len(_s) > _len and re.search(r'\b' + re.escape(_s) + r'\b', item_norm):
                _best, _len = _food, len(_s)
    return _best if (_best is not None and _best in allowed) else None


# ======= [P1-CYCLE-BASE-AFFINITY · 2026-08-02] (audit solver+seeder v7 · Task 9) =======
# La lista quincenal/mensual extrapola las bases de la ventana visible (3-4 días) a 15/30 días,
# pero la fatiga por recencia (`recent_3d×3.0 + recent_7d×1.5` = ×4,5 EN CONTRA de lo recién
# comido) empuja los chunks siguientes hacia OTRAS bases. El usuario compra un saco de arroz
# para 30 días y desde el chunk 2 el plan le pide pasta: el sistema manda comprar una cosa y
# cocinar otra.
#
# El `MEALFIT_GROCERY_CYCLE_LOCK` que existía para esto es BINARIO ("REGLA DE AHORRO EXTREMA…
# usa EXACTAMENTE las proteínas asignadas") y está OFF por decisión explícita del dueño
# (P1-VARIETY-RENEWAL-NO-CYCLE-LOCK: "no me des los mismos a menos que lo necesite"). Esto es la
# fuerza INTERMEDIA que faltaba: un multiplicador SUAVE sobre los pesos del sorteo. La variedad
# sigue siendo posible; solo se sesga hacia lo que el usuario ya tiene en casa.
#
# **DEFAULT 1.0 = APAGADO, y es deliberado.** El dueño eligió variedad sobre reuso a sabiendas;
# encender esto es decisión suya, no nuestra. Con el default el sorteo es byte-idéntico al
# anterior y ni siquiera se lee el plan (cero I/O añadida) — ver
# `test_default_apagado_no_cambia_el_sorteo_ni_hace_io`.
_CYCLE_AFFINITY_MIN, _CYCLE_AFFINITY_MAX = 1.0, 6.0


def _cycle_base_affinity_factor() -> float:
    """Multiplicador del peso de las bases compradas. `1.0` = apagado. Clamp [1.0, 6.0].

    Se lee en CADA llamada (no se cachea) para conservar el rollback sin redeploy, y vía
    `_env_float` para auto-registrarse en `_KNOBS_REGISTRY` — un knob que no aparece en el
    snapshot de `/health/version` es invisible durante un incidente de sorteo sesgado
    (P3-SEEDER-KNOBS-REGISTRY)."""
    try:
        return min(_CYCLE_AFFINITY_MAX, max(_CYCLE_AFFINITY_MIN, _env_float(
            "MEALFIT_CYCLE_BASE_AFFINITY", 1.0,
            lambda v: _CYCLE_AFFINITY_MIN <= v <= _CYCLE_AFFINITY_MAX)))
    except Exception:
        return 1.0


def _shopping_item_is_stable(item: dict) -> bool:
    """¿Este ítem de la lista se compró para TODO el ciclo (no perecedero)?

    Fuente primaria: el flag `is_perishable` que `_build_hybrid_shopping_list` (VISIÓN-C) ya
    persiste en cada ítem — es LA decisión que el sistema tomó al armar la compra: los staples
    llevan cantidad del periodo completo y los perecederos, cantidad semanal. Reusarla en vez de
    re-derivarla evita que las dos superficies opinen distinto.

    Fallback para planes viejos sin el flag: el clasificador canónico `is_perishable_category`.
    Sin ninguna de las dos señales → False (fail-closed): "no sé" no es "no perecedero"."""
    if not isinstance(item, dict):
        return False
    _flag = item.get("is_perishable")
    if isinstance(_flag, bool):
        return _flag is False
    _cat, _name = item.get("category"), item.get("name")
    if not _cat:
        return False
    try:
        from shopping_calculator import is_perishable_category
        return not is_perishable_category(_cat, item.get("shelf_life_days"), _name)
    except Exception as _exc:
        # [P2-SILENT-DEGRADATION] sin clasificador no hay afinidad (fail-closed), pero deja traza.
        logger.debug("[P1-CYCLE-BASE-AFFINITY] clasificación de perecibilidad falló (%s): %s: %s",
                     str(_name)[:60], type(_exc).__name__, str(_exc)[:160])
        return False


def _purchased_cycle_bases(user_id: str, grocery_days: int, grocery_duration: str,
                           allowed_proteins, allowed_carbs) -> "tuple[set, set]":
    """Bases NO perecederas que el usuario YA compró en el ciclo vigente.

    **Fuente: la lista de compras persistida del propio plan**, no `grocery_cycle.base_*`.
    Medido contra producción (2026-08-02): 0 de 23 planes tienen `plan_data->'grocery_cycle'` y
    no existe columna `grocery*`/`cycle*` en el esquema — leer de ahí habría entregado código
    inerte que consulta una clave que nunca está. `aggregated_shopping_list*` sí existe (22 de
    23 planes) y es LITERALMENTE lo que se le mandó a comprar: la ground truth de "lo que ya
    tiene en casa".

    Fail-open en bloque: cualquier fallo devuelve conjuntos vacíos y el sorteo queda intacto.
    tooltip-anchor: P1-CYCLE-BASE-AFFINITY (fuente de datos)"""
    _empty = (set(), set())
    try:
        record = get_latest_meal_plan_with_id(user_id)
        if not record or not isinstance(record.get("plan_data"), dict):
            return _empty
        # El ciclo tiene que seguir VIVO: un plan más viejo que su propia duración significa que
        # el usuario ya volvió al súper, y sesgar hacia una compra vencida es peor que no sesgar.
        _created = record.get("created_at")
        if _created is not None:
            if isinstance(_created, str):
                from constants import safe_fromisoformat
                _created = safe_fromisoformat(_created)
            if _created.tzinfo is None:
                _created = _created.replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - _created).days >= grocery_days:
                logger.info("🛒 [P1-CYCLE-BASE-AFFINITY] ciclo vencido (plan de hace ≥%s días) → "
                            "sin afinidad.", grocery_days)
                return _empty
        _plan = record["plan_data"]
        # La lista del PERIODO es la que lleva el `is_perishable` de la híbrida; el top-level es
        # su espejo para la duración elegida y sirve de fallback.
        _key = {"monthly": "aggregated_shopping_list_monthly",
                "biweekly": "aggregated_shopping_list_biweekly"}.get(grocery_duration)
        _items = (_plan.get(_key) if _key else None) or _plan.get("aggregated_shopping_list") or []
        if not isinstance(_items, list) or not _items:
            return _empty

        _allow_p, _allow_c = set(allowed_proteins), set(allowed_carbs)
        _bases_p, _bases_c = set(), set()
        for _it in _items:
            if not _shopping_item_is_stable(_it):
                continue
            _norm = strip_accents(str(_it.get("name") or "").lower()).strip()
            if not _norm:
                continue
            # `allowed` = el pool YA filtrado por alergia/dieta/dislike y por los penalties
            # clínicos: la afinidad NUNCA puede resucitar una base excluida (decisión #4).
            _p = _catalog_pick_wb(_norm, DOMINICAN_PROTEINS, protein_synonyms, _allow_p)
            if _p:
                _bases_p.add(_p)
            _c = _catalog_pick_wb(_norm, DOMINICAN_CARBS, carb_synonyms, _allow_c)
            if _c:
                _bases_c.add(_c)
        return _bases_p, _bases_c
    except Exception as _exc:
        logger.warning("[P1-CYCLE-BASE-AFFINITY] lectura de la compra del ciclo falló → sorteo "
                       "sin afinidad (fail-open): %s: %s", type(_exc).__name__, str(_exc)[:200])
        return _empty


def _apply_cycle_affinity(names, weights, bases, factor) -> list:
    """Multiplica el peso de las bases compradas. Pura, sin I/O, index-alineada con `names`."""
    if not bases or factor <= 1.0:
        return weights
    return [w * (factor if n in bases else 1.0) for n, w in zip(names, weights)]


def _build_light_protein_pool(veggies, proteins, *, bariatric: bool = False) -> list:
    """[P2-LIGHT-PROTEIN-POOL · 2026-07-30] (audit solver+seeder v5) Pool del "ancla liviana"
    sorteada por día para desayuno/merienda.

    El filtro original corría SOLO sobre `filtered_veggies` con 8 tokens, y 3 de ellos
    ('queso', 'yogur', 'ricotta') NO existen en `DOMINICAN_VEGGIES_FATS`: los lácteos viven en
    `DOMINICAN_PROTEINS`. Resultado: esos 3 tokens no podían matchear jamás y el ancla quedaba
    reducida a {Nueces/Almendras, Maní, Mantequilla de maní, Mantequilla de almendras} — o sea
    reforzando el "maní en 4 de 12" que el knob venía a curar, y sin la diversificación láctea
    que su propio comentario prometía.

    Segundo defecto: el callsite hacía `_light_pool[:4]`, un slice POSICIONAL que nunca consulta
    `veggie_weights`, así que el penalty ×0.1 por riesgo obstructivo del pouch no llegaba a
    aplicarse — el ancla podía priorizar frutos secos ENTEROS para un perfil post-quirúrgico.
    Aquí se filtran explícitamente (las formas molida/mantequilla/fileteada sí son seguras).

    Determinista: preserva el orden de los catálogos de entrada. tooltip-anchor: P2-LIGHT-PROTEIN-POOL"""
    _out, _seen = [], set()
    for _src in (veggies or [], proteins or []):
        for _item in _src:
            _n = strip_accents(str(_item).lower())
            if not any(t in _n for t in _LIGHT_ANCHOR_TOKENS):
                continue
            if bariatric and any(t in _n for t in _WHOLE_NUT_SEED_TOKENS) \
                    and not any(t in _n for t in _SAFE_NUT_FORM_TOKENS):
                continue      # forma ENTERA: riesgo obstructivo del pouch
            if _n not in _seen:
                _seen.add(_n)
                _out.append(_item)
    return _out


def _pick_light_anchor_candidates(pool, k: int = 4) -> list:
    """[P2-SEEDER-PAIRS-GOALS · 2026-07-31] (audit v6 · F18) Elige `k` candidatos del pool del
    ancla liviana SIN REEMPLAZO y de verdad al azar.

    El callsite hacía `_light_pool[:4]`, un slice POSICIONAL. `_build_light_protein_pool`
    concatena los vegetales ANTES que las proteínas, y los vegetales aportan exactamente 5 frutos
    secos, así que los 11 lácteos del pool (índices 5..15) NO entraban jamás y el "sorteo" del
    ancla era una constante. O sea: el fix v5 que amplió el pool para incluir lácteos quedó INERTE
    en su consumidor — el pool creció y nadie llegaba a mirarlo.

    Muestreo UNIFORME a propósito, no ponderado por la frecuencia inversa del seeder: los pesos de
    vegetales y proteínas no son comparables entre sí (`veggie_weights` lleva el penalty de frutos
    secos y `protein_weights` no), así que mezclarlos sesgaría el ancla por un artefacto del
    penalty en vez de por la historia del usuario. tooltip-anchor: P2-SEEDER-PAIRS-GOALS"""
    _items = [x for x in (pool or []) if x]
    if not _items:
        return []
    return random.sample(_items, min(int(k), len(_items)))


def _rotate_pairs(items, days: int = 3):
    """[P1-CARB-SEEDER-PAIRS · 2026-07-27] Núcleo de la rotación en pares, extraído de
    `_rotate_fruit_pairs` para que carbos y frutas usen LA MISMA: día i recibe
    `(items[i], items[i+1])` sobre la rotación circular.

    La gracia es el ahorro en la compra: 4 elementos distintos cubren 3 días con dos por día en
    vez de comprar 6, porque se reutilizan entre días.

    Se extrae en vez de copiarse: dos implementaciones del mismo reparto divergen — es el patrón
    que P1-PANTRY-GATE-SSOT y P1-CLOSER-INTO-MONTAJE costaron cerrar en esta misma base de código.

    Devuelve `None` con menos de 2 elementos utilizables; el caller decide el fallback.

    [P1-ROTATE-PAIRS-DEDUPE · 2026-07-28] Deduplica preservando orden ANTES de contar:
    cazado en vivo (05:48) un pool colapsado a ['Arroz Blanco']×3 — len 3 pasaba el guard
    y cada día recibía (Arroz, Arroz): la "2ª base distinta" era LA MISMA, derrotando el
    propósito del par. Con dedupe, el degenerado cae al fallback del caller ("otra base
    distinta del catálogo"), que sí empuja variedad.

    [P2-SEEDER-DAYS-COUNT · 2026-08-03] `days` (default 3 = comportamiento previo para los
    callers no migrados) en vez del `range(3)` hardcodeado. El chunk DOMINANTE de los planes
    largos es de 4 días (`split_with_absorb`: 15d → [3,4,4,4]) y el estampado al esqueleto
    reparte por módulo, así que con 3 pares el día índice 3 recibía el reparto del día 0.
    Con el pool más corto que `days` la rotación se repite (degradación deliberada, ver el
    docstring de `get_deterministic_variety_prompt`), pero nunca produce dos días CONSECUTIVOS
    con el mismo par: con n≥2 elementos `(base[i], base[i+1]) != (base[i+1], base[i+2])`.
    tooltip-anchor: P2-SEEDER-DAYS-COUNT"""
    base = [x for x in (items or []) if x and str(x).strip()]
    _vistos: set = set()
    base = [x for x in base if not (x in _vistos or _vistos.add(x))]
    if len(base) < 2:
        return None
    n = len(base)
    days = max(1, int(days or 1))
    return [(base[i % n], base[(i + 1) % n]) for i in range(days)]


def get_deterministic_variety_prompt(history_text: str, form_data: dict = None, user_id: str = None,
                                     rejection_reasons: list = None,
                                     out_assignment: dict = None,
                                     *, days_count: int = 3) -> str:
    # [P2-VEGGIE-CHANNEL-DAYGEN · 2026-07-30] `out_assignment` (opcional) recibe el reparto que el
    # seeder decide, como DATO además de como prosa. Hoy transporta `veggie_pairs`: el
    # day-generator es quien escribe los ingredientes reales y no tenía forma de recibir esa
    # rotación (el esqueleto tipado no la declaraba), así que generaba ciego y caía en su default.
    # Opcional y no-mutante si el caller no lo pasa → cero impacto en los callers existentes.
    """Implementa Inversión de Control Determinista para evitar Mode Collapse en el LLM.

    [P2-SEEDER-DAYS-COUNT · 2026-08-03] (audit solver+seeder v7) `days_count` = días del chunk que
    se está generando. Keyword-only con default 3 (= comportamiento previo) para que ningún caller
    no migrado cambie de conducta.

    ## Por qué existe

    Todo el reparto estaba fijado a 3 días: `_rotate_pairs` con `range(3)`, tres opciones A/B/C
    en el prompt y `num_proteins_to_pick = min(3, ...)`. Pero `constants.split_with_absorb`
    reparte 15d → [3,4,4,4] y 30d → [3,4,4,4,4,4,4,3]: la forma DOMINANTE de chunk es de 4 días.
    Como el orquestador estampa el reparto al esqueleto por módulo
    (`_pairs_all[_di % len(_pairs_all)]`), el día índice 3 recibía EXACTAMENTE el reparto del
    día 0 — misma proteína, mismos carbos, mismos vegetales, misma fruta. En un plan de 30 días
    son ~6 pares de días clonados por construcción, y el contrato «1 proteína distinta por día»
    de `variety_level=max` (auto-promovido para gain_muscle/lose_fat y bariátrica) era
    aritméticamente insatisfacible en el 4º día de cada chunk.

    ## Degradación deliberada con pool corto

    Si el pool filtrado (alergias + dislikes + dieta + filtros clínicos) tiene menos elementos que
    `days_count` —el caso real de los chunks de 6 días de un plan de 21— el reparto DEGRADA al
    módulo de siempre: los días extra reciclan bases ya asignadas. No se emite ningún gate ni se
    exige lo imposible; es la lección de P1-FRUIT-SEEDER-GATE-CONTRACT (una instrucción
    insatisfacible no produce mejores planes, produce retries quemados). El prompt tampoco
    PROMETE N proteínas distintas: enumera el reparto y ya.

    Rollback sin redeploy: `MEALFIT_SEEDER_DAYS_COUNT=false` → 3 días fijos, byte-idéntico al
    comportamiento previo. tooltip-anchor: P2-SEEDER-DAYS-COUNT
    """
    logger.debug("🎲 [ANTI MODE-COLLAPSE] Calculando Matriz de Ingredientes (Round-Robin)...")
    # El knob se lee en CADA llamada (no a nivel módulo) para que el rollback no necesite
    # redeploy ni reimport, igual que MEALFIT_GAINMUSCLE_HIGH_DENSITY_PROTEIN.
    if not _env_bool("MEALFIT_SEEDER_DAYS_COUNT", True):
        days_count = 3
    try:
        _dc = int(days_count or 3)
    except (TypeError, ValueError):
        _dc = 3
    # Mismo techo que `graph_orchestrator._MAX_DAYS_TO_GENERATE` (2×PLAN_CHUNK_SIZE): un
    # `_days_to_generate` corrupto no debe inflar el prompt ni el sorteo. Piso 1 (chunk de 1 día
    # existe: `split_with_absorb` no lo produce, pero `/regenerate-day` sí genera de a uno).
    _dc = max(1, min(_dc, PLAN_CHUNK_SIZE * 2))
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
        
        # [P2-SEEDER-DIET-NONE · 2026-07-31] (audit solver+seeder v6 · F32) Era
        # `form_data.get("diet", form_data.get("dietType", "")).lower()`, que explota con
        # AttributeError si la clave EXISTE con valor None — y `dietType: null` es entrada válida:
        # el campo es presence-optional en el boundary (decisión documentada en formValidation.js)
        # y un health_profile rehidratado puede traerlo así. El crash era determinista: el retry
        # del pipeline repetía el mismo AttributeError y la generación quedaba bloqueada para ese
        # usuario, sin mensaje accionable. `or` en vez de default posicional + `str()` defensivo.
        # tooltip-anchor: P2-SEEDER-DIET-NONE
        diet = str(form_data.get("diet") or form_data.get("dietType") or "").lower()
        
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
    # [P2-OVERUSE-RAW-FREQ · 2026-08-03] Snapshot ANTES de la fatiga de recencia: el veto
    # textual (`used_proteins/used_carbs/used_veggies`, más abajo) lee de aquí cuando el
    # knob está encendido (default). Los pesos del sorteo siguen leyendo `db_freq_map`
    # (fatigado) sin cambios.
    raw_freq_map = {}
    if user_id and user_id != "guest":
        try:
            db_freq_map = get_user_ingredient_frequencies(user_id)
            raw_freq_map = dict(db_freq_map)
            db_freq_map = _apply_recency_fatigue(db_freq_map, user_id)
        except Exception as e:
            logger.error(f"⚠️ [ANTI MODE-COLLAPSE] Error obteniendo frecuencias de DB: {e}")

    if db_freq_map:
        # ======= NUEVO FLUJO OPTIMIZADO O(1) =======
        logger.info(f"⚡ [ANTI MODE-COLLAPSE] Usando Hash Map O(1) de DB con {len(db_freq_map)} métricas pre-calculadas.")

        def _freq_for_pool_item(item: str, syn_map: dict, freq_map: dict = None) -> int:
            """[P2-FREQ-LOOKUP-CANONICAL · 2026-07-29] (audit solver+seeder v4) El seeder LEÍA por
            alias y la tabla se ESCRIBE por la BASE canónica de `normalize_ingredient_for_tracking`
            (n-gramas contra `GLOBAL_REVERSE_MAP`). Para todo ítem del pool que es VARIANTE de otra
            base, la clave consultada NUNCA existía en la tabla → `freq=0` permanente → peso
            `1/(0+1)=1.0`, el MÁXIMO, para siempre.

            Medido sobre los 4 pools (145 ítems): **19 ciegos** — Maní y Almendras fileteadas
            resuelven a `nueces/almendras`; Salmón/Tilapia/Mero/Bacalao/Arenque a `pescado`; Muslo
            de pollo a `pollo`; Hígado de res a `res`… La asimetría se REFUERZA sola: comer salmón
            incrementa `pescado`, lo que castiga al ítem genérico del pool y NO al salmón, así que
            el sorteo se desplaza hacia el pez específico que el usuario acaba de comer. Es la
            explicación del maní en 4 de 12 comidas que el owner venía viendo.

            UNIÓN, no reemplazo: sumamos alias + clave canónica (si un ítem ya es su propia base, no
            pierde señal; el `set` evita contar dos veces cuando coinciden).

            [P2-OVERUSE-RAW-FREQ · 2026-08-03] `freq_map` opcional (default `db_freq_map`,
            fatigado): permite reutilizar el mismo lookup contra `raw_freq_map` (pre-fatiga)
            para el veto textual, sin duplicar la lógica de alias/canónico.
            tooltip-anchor: P2-FREQ-LOOKUP-CANONICAL"""
            _keys = {strip_accents(s.lower()) for s in syn_map.get(item.lower(), [item.lower()])}
            if FREQ_LOOKUP_CANONICAL:
                try:
                    from constants import normalize_ingredient_for_tracking as _norm_freq
                    _canon = _norm_freq(item)
                    if _canon:
                        _keys.add(strip_accents(str(_canon).lower()))
                except Exception as _exc:
                    # [P2-SILENT-DEGRADATION] Si esto falla, `_keys` se queda SOLO con los alias y
                    # el lookup vuelve a fallar al 100% para los 19 ítems ciegos — o sea, degrada
                    # exactamente al bug que P2-FREQ-LOOKUP-CANONICAL vino a cerrar, y en silencio.
                    logger.debug(
                        "[P2-SILENT-DEGRADATION] freq lookup canónico de %r: %s: %s",
                        item, type(_exc).__name__, str(_exc)[:160])
            _map = db_freq_map if freq_map is None else freq_map
            return sum(_map.get(_k, 0) for _k in _keys)

        for p in filtered_proteins:
            protein_freq[p] = _freq_for_pool_item(p, protein_synonyms)
        for c in filtered_carbs:
            carb_freq[c] = _freq_for_pool_item(c, carb_synonyms)
        for v in filtered_veggies:
            veggie_freq[v] = _freq_for_pool_item(v, veggie_fat_synonyms)
        for f in filtered_fruits:
            fruit_freq[f] = _freq_for_pool_item(f, fruit_synonyms)

        # [P2-OVERUSE-RAW-FREQ · 2026-08-03] El veto textual (más abajo, `OVERUSE_THRESHOLD`)
        # usa la frecuencia CRUDA (pre-fatiga) cuando el knob está encendido (default); los
        # `*_freq` de arriba (fatigados) siguen alimentando SOLO los pesos del sorteo.
        if OVERUSE_ON_RAW_FREQ:
            protein_freq_for_veto = {p: _freq_for_pool_item(p, protein_synonyms, raw_freq_map)
                                      for p in filtered_proteins}
            carb_freq_for_veto = {c: _freq_for_pool_item(c, carb_synonyms, raw_freq_map)
                                   for c in filtered_carbs}
            veggie_freq_for_veto = {v: _freq_for_pool_item(v, veggie_fat_synonyms, raw_freq_map)
                                     for v in filtered_veggies}
        else:
            protein_freq_for_veto, carb_freq_for_veto, veggie_freq_for_veto = (
                protein_freq, carb_freq, veggie_freq)
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
        # [P2-OVERUSE-RAW-FREQ · 2026-08-03] Rama regex (guest / sin `db_freq_map`): no hay
        # fatiga de recencia aplicada aquí (`_calcular_frecuencias_regex_cpu_bound` no la
        # conoce), así que `protein_freq` etc. YA son la frecuencia cruda — el veto usa
        # directamente lo mismo que los pesos, sin necesidad de un mapa separado.
        protein_freq_for_veto, carb_freq_for_veto, veggie_freq_for_veto = (
            protein_freq, carb_freq, veggie_freq)

    # Umbral mínimo: solo considerar "sobreusados" ingredientes con freq >= 3.
    # Con freq=1 o 2 el soft-penalty 1/(freq+1) ya reduce su probabilidad suficientemente;
    # marcarlos como "PROHIBIDOS" en el prompt contradice el modelo de penalización suave.
    # [P2-OVERUSE-RAW-FREQ · 2026-08-03] Comparación contra `*_freq_for_veto` (crudo por
    # default, ver `OVERUSE_ON_RAW_FREQ`), NO contra `*_freq` (que sigue fatigado para los
    # pesos del sorteo, más abajo).
    OVERUSE_THRESHOLD = 3
    used_proteins = [p for p, freq in protein_freq_for_veto.items() if freq >= OVERUSE_THRESHOLD]
    used_carbs = [c for c, freq in carb_freq_for_veto.items() if freq >= OVERUSE_THRESHOLD]
    used_veggies = [v for v, freq in veggie_freq_for_veto.items() if freq >= OVERUSE_THRESHOLD]
    
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
    # [P2-VEGGIE-POOL-EMPTY-GUARD · 2026-07-30] (audit solver+seeder v5) El guard cubría solo
    # proteínas y carbos. Con `available_veggies` vacío el padding de más abajo ejecuta
    # `_base_veggies[len(unique_veggies) % len(_base_veggies)]` → **ZeroDivisionError**, y ningún
    # caller lo envuelve (`_build_shared_context` → nodo planner/day-gen): la excepción tumba el
    # nodo y el retry repite el MISMO crash de forma determinista, dejando la generación
    # BLOQUEADA para ese usuario —sin mensaje accionable— hasta que edite sus dislikes. Las
    # frutas ya estaban protegidas (`if unique_fruits:` + break). Misma semántica que las otras
    # dos categorías: sin pool utilizable, libertad total al LLM (el day-gen funciona sin este
    # bloque de variedad; lo que NO funciona es un plan que nunca llega a generarse).
    if not available_proteins or not available_carbs or not available_veggies:
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
    # día y dispara incoherencias de slot. Para gain_muscle/lose_fat el aporte
    # de aminoácidos completos y la variedad de fuentes importa más que optimizar
    # el costo del supermercado (3 proteínas vs 2 al mes es marginal en costo).
    # [P2-SEEDER-PAIRS-GOALS · 2026-07-31] (audit v6 · F28) Era `"lose_weight"`, un valor que NO
    # existe en `_MAIN_GOAL_ENUM` ({lose_fat, gain_muscle, maintenance, performance}) y que el
    # router rechaza con 422 ⇒ el token era INALCANZABLE y la auto-promoción no disparaba nunca
    # para quien quiere perder grasa — justo el perfil que más sufre la repetición de proteína.
    # tooltip-anchor: P2-SEEDER-PAIRS-GOALS
    _GOALS_FORCE_MAX_VARIETY = {"gain_muscle", "lose_fat"}
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
    
    # [P2-SEEDER-DAYS-COUNT · 2026-08-03] `_dc` (días del chunk) en vez del 3 literal. En
    # `standard` el 2 se QUEDA a propósito: ahí "2 proteínas" es una decisión de COSTO de
    # supermercado (documentada arriba), no el techo aritmético que este fix corrige — el padding
    # de más abajo las cicla hasta cubrir los `_dc` días.
    if variety_level == "max":
        num_proteins_to_pick = min(_dc, len(available_proteins))   # 1 proteína distinta por día
        num_carbs_to_pick = min(_dc, len(available_carbs))         # 1 carb distinto por día
        num_veggies_to_pick = min(2 * _dc, len(available_veggies))   # 2 vegetales distintos por día
        logger.info(f"🎯 [ANTI MODE-COLLAPSE] variety_level=max → distribución máxima "
                    f"({num_proteins_to_pick}P/{num_carbs_to_pick}C/{num_veggies_to_pick}V "
                    f"para {_dc} día(s))")
    else:
        num_proteins_to_pick = min(2, len(available_proteins))
        num_carbs_to_pick = min(2, len(available_carbs))
        num_veggies_to_pick = min(2 * _dc, len(available_veggies))   # 2 vegetales distintos por día
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
    # [P3-SEEDER-KNOBS-REGISTRY · 2026-07-31] (audit v6 · F30) Leía `os.environ` en crudo, así que
    # el knob NO se auto-registraba en `_KNOBS_REGISTRY` y era invisible en `/health/version`: durante
    # un incidente de sorteo sesgado el operador consulta el snapshot, no lo ve, y concluye que no
    # existe tal palanca o que su override no aplica. `_env_float` lee en CADA llamada igual que
    # antes (no cachea), así que el rollback sin redeploy se conserva.
    # tooltip-anchor: P3-SEEDER-KNOBS-REGISTRY
    try:
        _tb_boost = min(5.0, max(1.0, _env_float(
            "MEALFIT_TRANSFORM_BASE_BOOST", 2.0, lambda v: 1.0 <= v <= 5.0)))
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
        except Exception as _exc:
            # [P2-SILENT-DEGRADATION] best-effort: la falla no debe romper el flujo,
            # pero sí dejar traza (antes: pass silencioso).
            logger.debug(
                "[P2-SILENT-DEGRADATION] transform-base boost de carb_weights no aplicado (pesos sin ponderar): %s: %s",
                type(_exc).__name__, str(_exc)[:160])

    # [P1-BUDGET-TIER-LEVERS · 2026-07-02] (audit v4 presupuesto) Ponderación ECONÓMICA del sorteo:
    # el tier del formulario era señal solo-prompt (advisory). Cuando el presupuesto pide economía
    # (low "Económico" o custom ajustado — SSOT nutrition_calculator.budget_prefers_economy), las
    # proteínas/carbos del TERCIO MÁS BARATO del pool (precio/lb-equivalente del catálogo master)
    # reciben boost multiplicativo (default 2.0×). Sigue siendo sorteo ponderado — jamás remueve
    # ítems ni toca filtros clínicos/alergias (esos ya corrieron aguas arriba). Ítems sin precio
    # resoluble → peso intacto. Rollback sin redeploy: MEALFIT_BUDGET_POOL_WEIGHT=1.0.
    # tooltip-anchor: P1-BUDGET-TIER-LEVERS (pool weighting)
    # [P3-SEEDER-KNOBS-REGISTRY · 2026-07-31] ver la nota del gemelo MEALFIT_TRANSFORM_BASE_BOOST.
    try:
        _bud_boost = min(5.0, max(1.0, _env_float(
            "MEALFIT_BUDGET_POOL_WEIGHT", 2.0, lambda v: 1.0 <= v <= 5.0)))
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

                # [P3-BUDGET-POOL-FLOOR · 2026-08-04] El tercil + el boost viven ahora en
                # `_budget_boost_with_floor` (nivel módulo): el piso relativo necesita la mediana
                # del pool y probarlo por sorteo sería adivinar. Con `MEALFIT_BUDGET_POOL_FLOOR=0`
                # devuelve exactamente `w × boost`, o sea el comportamiento previo byte a byte.
                # El gating NO cambia: esto sigue corriendo sólo dentro del `budget_prefers_economy`
                # de P1-BUDGET-TIER-LEVERS. tooltip-anchor: P3-BUDGET-POOL-FLOOR
                _bp_floor = _budget_pool_floor()

                def _bw_boost_cheapest(_names, _weights, _pmap):
                    _prices = [_bw_resolve(_n, _pmap) for _n in _names]
                    return _budget_boost_with_floor(_weights, _prices, _bud_boost, _bp_floor)

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
    # [P1-PANTRY-FLOOR-CLINICAL-FILTER · 2026-08-02] `_PROCESSED_MEAT_KEYWORDS` vive ahora a
    # nivel módulo (SSOT único): el filtro de la nevera de más abajo aplica el MISMO vocabulario
    # que este penalty del sorteo, en vez de una quinta copia a mano.
    # [P2-SEEDER-PAIRS-GOALS · 2026-07-31] (audit v6 · F28) Eran `"lose_weight"` y
    # `"health_improvement"`, ninguno en `_MAIN_GOAL_ENUM` ⇒ el penalty de embutidos solo aplicaba
    # a gain_muscle. `health_improvement` se ELIMINA en vez de remapearse: no tiene equivalente
    # en el enum (ni `maintenance` ni `performance` significan "mejorar salud"), y mapearlo a uno
    # sería inventar una intención que el usuario no declaró.
    _GOALS_PENALIZE_PROCESSED = {
        "gain_muscle", "lose_fat",
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
    # [P1-SODIUM-BOMB-POOL · 2026-07-05] Proteínas CURADAS EN SAL (bacalao/arenque/salami/tocino/
    # longaniza...) — penalty UNIVERSAL en el sorteo (todos los goals: el presupuesto OMS de
    # 2000mg de sodio no depende del objetivo). La proteína ES sal: un solo día con bacalao o
    # salami revienta el techo aunque el §17 del prompt y el autofix de sodio hagan todo bien
    # (medido en vivo: plan 3aa6e58a con pools Salami Dominicano + Bacalao → 4,576mg + banner
    # micro_worst_day_ceiling + 3 intentos). Se APILA con el penalty de embutidos por goal
    # (salami en gain_muscle queda ×0.01 — prácticamente excluido). Graceful: si el catálogo/
    # gustos solo dejan curados, igual pueden salir. Rollback sin redeploy:
    # MEALFIT_SODIUM_BOMB_POOL_PENALTY=1.0. tooltip-anchor: P1-SODIUM-BOMB-POOL
    try:
        from knobs import _env_float as _sb_envf
        _sb_penalty = max(0.0, min(1.0, _sb_envf("MEALFIT_SODIUM_BOMB_POOL_PENALTY", 0.1)))
    except Exception:
        _sb_penalty = 0.1
    if _sb_penalty < 1.0:
        # [P1-PANTRY-FLOOR-CLINICAL-FILTER · 2026-08-02] La tupla vivía AQUÍ dentro, o sea que
        # con el knob en 1.0 ni siquiera existía; ahora es constante de módulo (SSOT único,
        # compartida con el filtro de la nevera).
        _salt_penalized = 0
        for i, p in enumerate(available_proteins):
            p_norm = strip_accents(p.lower())
            if any(kw in p_norm for kw in _SALT_CURED_PROTEIN_TOKENS):
                protein_weights[i] *= _sb_penalty
                _salt_penalized += 1
        if _salt_penalized:
            logger.info(
                f"🧂 [P1-SODIUM-BOMB-POOL] {_salt_penalized} proteína(s) curada(s) en sal "
                f"penalizada(s) ×{_sb_penalty} en el sorteo (presupuesto de sodio OMS: "
                f"bacalao/salami revientan el techo del día ellos solos)."
            )
    
    fruit_weights = []
    if available_fruits:
        fruit_weights = [1.0 / (fruit_freq.get(f, 0) + 1) for f in available_fruits]
        # [P1-BARIATRIC-PROTEIN-DENSITY · 2026-06-27] Bariátrica: penaliza frutas de ALTO índice glucémico
        # (guineo/mango/uva/piña/plátano) → prefiere bajo-IG (fresa/lechosa/mandarina/manzana). El revisor médico
        # rechazaba mango (clash) y guineo en porción grande por dumping (corr=5ffd78cf). Penalty ×0.15 (graceful:
        # si solo hay alto-IG disponible, igual se eligen). tooltip-anchor: P1-BARIATRIC-PROTEIN-DENSITY
        # [P1-PANTRY-FLOOR-CLINICAL-FILTER · 2026-08-02] `_HIGH_GI_FRUITS` subida a nivel módulo
        # (SSOT único): el espejo del filtro de la nevera usa la MISMA tupla.
        if _is_bariatric:
            for _i, _f in enumerate(available_fruits):
                if any(_g in strip_accents(_f.lower()) for _g in _HIGH_GI_FRUITS):
                    fruit_weights[_i] *= 0.15

    # ======= [P3-SEEDER-TEMPLATE-COVERAGE · 2026-08-04] COBERTURA DE PLANTILLAS =======
    # Multiplicador SUAVE sobre las bases que no aparecen en NINGUNA plantilla de la biblioteca
    # de platos. Va aquí, entre los demás penalties y antes del sorteo: todos son multiplicativos
    # (conmutan), así que el orden no altera el resultado, pero la regla "la afinidad de ciclo es
    # SIEMPRE la última antes del sorteo" se conserva intacta.
    # NUNCA excluye — con pool corto la base sin plantillas sigue pudiendo salir, que es lo
    # correcto: mejor un plato improvisado que un pool vacío. Rollback sin redeploy:
    # MEALFIT_LOW_TEMPLATE_COVERAGE_PENALTY=1.0. tooltip-anchor: P3-SEEDER-TEMPLATE-COVERAGE
    _tpl_factor = _low_template_coverage_penalty()
    if _tpl_factor < 1.0:
        try:
            protein_weights, _tpl_np = _apply_low_template_coverage_penalty(
                available_proteins, protein_weights, "protein", _tpl_factor)
            carb_weights, _tpl_nc = _apply_low_template_coverage_penalty(
                available_carbs, carb_weights, "base", _tpl_factor)
            if _tpl_np or _tpl_nc:
                logger.info(
                    "🍽️ [P3-SEEDER-TEMPLATE-COVERAGE] %d proteína(s) y %d carbo(s) sin ninguna "
                    "plantilla de %s en la biblioteca → peso ×%.2f en el sorteo (sesgo, no "
                    "exclusión: el day-gen las improvisa y tiende a clonar la fórmula).",
                    _tpl_np, _tpl_nc, _TEMPLATE_COVERAGE_MAIN_SLOT, _tpl_factor)
        except Exception as _tpl_e:
            logger.debug("[P3-SEEDER-TEMPLATE-COVERAGE] penalty no-op: %s: %s",
                         type(_tpl_e).__name__, str(_tpl_e)[:160])

    # ======= [P1-CYCLE-BASE-AFFINITY · 2026-08-02] AFINIDAD CON LA COMPRA DEL CICLO =======
    # Va AQUÍ a propósito: DESPUÉS de la fatiga por recencia y DESPUÉS de todos los filtros y
    # penalties (alergia/dieta/dislike + embutidos + curados en sal + carnes grasas + alto-IG +
    # el filtro clínico de la nevera), y JUSTO ANTES del sorteo. Un orden distinto la volvería
    # peligrosa: aplicada antes de los filtros, resucitaría un alimento excluido.
    #
    # `available_proteins`/`available_carbs` YA son el pool filtrado, así que el matching contra
    # ellos no puede reintroducir nada: una alergia registrada a mitad de ciclo sigue ganándole a
    # la lista de compras aunque la base esté comprada.
    _aff_bases_p, _aff_bases_c = set(), set()
    _aff_factor = _cycle_base_affinity_factor()
    if _aff_factor > 1.0 and form_data and user_id and user_id != "guest":
        _aff_duration = form_data.get("groceryDuration", "weekly")
        _aff_days = {"biweekly": 15, "monthly": 30}.get(_aff_duration, 7)
        # Dos gates que definen "existe la divergencia":
        #   · ciclo > 7 días — en el semanal se re-compra cada semana y no hay nada que honrar.
        #   · `_days_offset > 0` — chunk de CONTINUACIÓN. En un plan fresco (chunk 1) no hay nada
        #     comprado todavía; sesgarlo hacia la compra del plan ANTERIOR sería reimplantar por
        #     la puerta de atrás el cycle-lock que el dueño apagó (P1-VARIETY-RENEWAL-NO-CYCLE-LOCK
        #     se tomó justamente para las RENOVACIONES). El refill JIT sí corre con offset > 0.
        try:
            _aff_offset = int(form_data.get("_days_offset") or 0)
        except (TypeError, ValueError):
            _aff_offset = 0
        if _aff_days > 7 and _aff_offset > 0:
            _aff_bases_p, _aff_bases_c = _purchased_cycle_bases(
                user_id, _aff_days, _aff_duration, available_proteins, available_carbs)
            if _aff_bases_p or _aff_bases_c:
                logger.info(
                    "🛒 [P1-CYCLE-BASE-AFFINITY] afinidad ×%.1f hacia lo YA comprado del ciclo "
                    "(%s días, día %s): proteínas=%s carbos=%s. Sesgo, no lock: el resto del pool "
                    "sigue pudiendo salir.",
                    _aff_factor, _aff_days, _aff_offset,
                    sorted(_aff_bases_p) or "—", sorted(_aff_bases_c) or "—")
    protein_weights = _apply_cycle_affinity(
        available_proteins, protein_weights, _aff_bases_p, _aff_factor)
    # ======================================================================================

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
        # [P1-PANTRY-FLOOR-CLINICAL-FILTER · 2026-08-02] Cuerpo subido a `_is_low_density_main`
        # (nivel módulo). Este closure queda como delegación para que el filtro de la nevera
        # aplique EXACTAMENTE el mismo criterio: eran las dos mitades de una sola regla y solo
        # esta corría, así que lo extraído de la nevera la esquivaba entera.
        def _should_replace_main(_p):
            return _is_low_density_main(_p, _is_bariatric)
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

    # [P1-CATALOG-VARIETY-OPENED · 2026-07-26] `P1-SODIUM-BOMB-POOL` sólo pesa el pool de PROTEÍNAS,
    # así que al abrir el pool de carbos a `Galletas de soda` (941 mg Na/100 g, medido en el catálogo)
    # un día podía gastar media cuota OMS de sodio en su base de carbohidrato, sin que nada lo frenara.
    # Mismo patrón y mismo knob-spirit que el de proteínas: penalty en el SORTEO, no exclusión — el
    # alimento sigue existiendo y sale de vez en cuando, que es lo que el owner pidió al aprobar los
    # cuatro. `Granola` y `Durazno en almíbar` entran por azúcar añadido (19,8 y 14 g/100 g).
    # Rollback sin redeploy: MEALFIT_SALTY_SWEET_CARB_PENALTY=1.0.
    # tooltip-anchor: P1-CATALOG-VARIETY-OPENED
    try:
        from knobs import _env_float as _ss_envf
        _ss_penalty = max(0.0, min(1.0, _ss_envf("MEALFIT_SALTY_SWEET_CARB_PENALTY", 0.15)))
    except Exception:
        _ss_penalty = 0.15
    if _ss_penalty < 1.0:
        _SALTY_SWEET_CARB_TOKENS = ("galleta", "granola")
        _ss_n = 0
        for _i_c, _c in enumerate(available_carbs):
            if any(t in strip_accents(str(_c).lower()) for t in _SALTY_SWEET_CARB_TOKENS):
                carb_weights[_i_c] *= _ss_penalty
                _ss_n += 1
        if _ss_n:
            logger.info(f"🧂 [P1-CATALOG-VARIETY-OPENED] {_ss_n} base(s) de carbohidrato alta(s) en "
                        f"sodio/azúcar penalizada(s) ×{_ss_penalty} en el sorteo (salen, pero raro).")

    # [P1-CYCLE-BASE-AFFINITY · 2026-08-02] Segundo punto de aplicación. El penalty de carbos
    # altos en sodio/azúcar (arriba) corre DESPUÉS del sorteo de proteínas, así que la afinidad de
    # carbos tiene que aplicarse aquí para conservar la regla "después de todos los penalties,
    # justo antes del sorteo". El conjunto ya se resolvió arriba (una sola lectura del plan).
    carb_weights = _apply_cycle_affinity(
        available_carbs, carb_weights, _aff_bases_c, _aff_factor)

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
    # [P3-CYCLE-PERSIST-AFTER-POOLS · 2026-07-31] (audit v6 · F20) La base del ciclo se
    # persistía DENTRO del bloque del lock, antes de que la nevera, el dedupe y el padding
    # reescribieran esos mismos pools ⇒ el ciclo guardaba bases que el plan entregado no usa,
    # y al regenerar el lock las re-imponía con “REGLA DE AHORRO EXTREMA”: proteínas que no
    # están ni en la nevera del usuario ni en su lista de compras del ciclo. Aquí solo se toma
    # la DECISIÓN; la ESCRITURA se hace abajo con los pools ya definitivos.
    # tooltip-anchor: P3-CYCLE-PERSIST-AFTER-POOLS
    _cycle_persist_pending = None
    
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
                            # [P3-GROCERY-LOCK-REFILTER · 2026-07-30] (audit solver+seeder v5)
                            # El restore re-imponía las bases que se persistieron al INICIO del
                            # ciclo sin re-filtrarlas contra las alergias/dislikes/dieta de HOY —
                            # y el prompt del lock añade "REGLA DE AHORRO EXTREMA… usa EXACTAMENTE
                            # las proteínas asignadas". Un usuario que empieza un ciclo mensual con
                            # ['Pollo','Salmón'] y al día 10 registra alergia a pescado recibía
                            # Salmón como base OBLIGATORIA el resto del ciclo: el allergen-guard lo
                            # convierte en rechazo→retry en CADA regeneración (hasta 20 días de
                            # fricción). Se intersecta con el pool filtrado actual; si la
                            # intersección queda vacía se conserva el sorteo ya computado
                            # (fail-open: mejor perder el ahorro que imponer un alérgeno).
                            unique_proteins = (_intersect_cycle_base(
                                grocery_cycle.get("base_proteins"), filtered_proteins)
                                or unique_proteins)
                            unique_carbs = (_intersect_cycle_base(
                                grocery_cycle.get("base_carbs"), filtered_carbs)
                                or unique_carbs)
                            unique_veggies = (_intersect_cycle_base(
                                grocery_cycle.get("base_veggies"), filtered_veggies)
                                or unique_veggies)
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

                    # [P3-CYCLE-PERSIST-AFTER-POOLS · 2026-07-31] Se difiere la escritura: `now` y
                    # `days_elapsed` son locales de este try, así que la decisión (start_date +
                    # duración) se captura AQUÍ; las bases se toman abajo, ya definitivas.
                    _cycle_persist_pending = {"start_date": start_date_to_save,
                                              "duration_days": grocery_days}
                    del new_grocery_cycle
        except Exception as e:
            logger.error(f"Error procesando Grocery Cycle Lock: {e}")
    # ==========================================================

    # ======= CURRENT PANTRY INGREDIENTS INJECTION (ROTATION MODE) =======
    # [P3-SEEDER-TEMPLATE-COVERAGE · 2026-08-04] Las bases que salen de la NEVERA se guardan aparte
    # para la telemetría de cobertura de más abajo: `extracted_p`/`extracted_c` viven dentro del
    # `if` y el chequeo necesita además `chosen_proteins`/`chosen_carbs`, que se calculan 300
    # líneas después (tras dedupe, shuffle y padding). Sin nevera quedan vacías y el chequeo es
    # no-op — el WARNING es sobre la base IMPUESTA, no sobre cualquier base sorteada.
    _tpl_pantry_p, _tpl_pantry_c = [], []
    current_pantry_ingredients = (form_data.get("current_pantry_ingredients") or form_data.get("current_shopping_list", [])) if form_data else []
    if current_pantry_ingredients:
        logger.info(f"🔄 [ROTATION MODE] Extrayendo ingredientes base de la lista actual.")
        extracted_p, extracted_c, extracted_v, extracted_f = [], [], [], []
        csl_lower = [strip_accents(i.lower()) for i in current_pantry_ingredients]
        
        # [P1-PANTRY-EXTRACT-FILTERED-WB · 2026-07-30] (audit solver+seeder v5) Dos defectos en el
        # mismo loop, y hacían falta las DOS correcciones:
        #
        #   (1) EL OPERADOR era `in` crudo (subcadena). `PROTEIN_SYNONYMS['res']` trae el alias
        #       'res' y `['pollo']` trae 'pollo', así que una nevera de vegetales extraía carne:
        #           'res'   ⊂ '**fres**as'   → Res
        #           'pollo' ⊂ 're**pollo**'  → Pollo
        #       Dos proteínas fantasma ≥ PANTRY_ROTATION_MIN_PROTEINS ⇒ el pool se REEMPLAZA por
        #       ellas y se activa `cycle_locked` ("REGLA DE AHORRO EXTREMA… EXACTAMENTE las
        #       proteínas asignadas"): 3 días de pollo y res que el usuario NO tiene. Es el 13º
        #       incidente de subcadena del repo; el patrón canónico (word-boundary sobre el mismo
        #       synonym map, con strip_accents en los dos lados) ya vive en `cpu_tasks.py` y en el
        #       `fast_regex` de `constants.py` — esta era la tercera implementación, la única cruda.
        #
        #   (2) EL UNIVERSO eran los catálogos COMPLETOS (`DOMINICAN_*`) en vez de los `filtered_*`
        #       que este mismo seeder ya calculó arriba. Aunque el match sea exacto, eso deja a la
        #       nevera resucitar un alimento que la alergia/dieta/dislike había excluido del pool —
        #       y con `cycle_locked` el prompt lo vuelve OBLIGATORIO. Para un vegetariano el pool
        #       filtrado se descartaba entero y el backstop de dieta lo convertía en retry-storm.
        #
        # Con solo (1) o solo (2) queda medio agujero abierto.
        #   (3) EL ALIAS GANADOR. Con (1)+(2) todavía quedaba un fantasma: `PROTEIN_SYNONYMS['res']`
        #       incluye el alias genérico `'filete'`, que ES una palabra completa dentro de
        #       "Filete de pescado" ⇒ una nevera con PESCADO extraía RES. El word-boundary no puede
        #       verlo (el alias no es subcadena de otra palabra: es genuinamente ambiguo) y el pool
        #       filtrado tampoco (Res sí está permitido). La regla que lo cierra: gana el alias MÁS
        #       ESPECÍFICO (el más largo) sobre el catálogo COMPLETO, y solo se extrae si ese
        #       ganador está permitido. Si el ítem nombra un alimento excluido (alergia/dieta/
        #       dislike), NO se degrada a un match genérico más débil — ese ítem simplemente no
        #       aporta base. Medido: 'filete de pescado' → Pescado(17) gana a Res(6); 'salmon
        #       fresco' → Pescado; 'repollo'/'fresas' → ningún match proteico.
        # [P1-CYCLE-BASE-AFFINITY · 2026-08-02] Cuerpo subido a `_catalog_pick_wb` (nivel módulo).
        # Este closure queda como delegación para que la afinidad de ciclo —que corre ANTES del
        # sorteo, ~280 líneas más arriba— aplique EXACTAMENTE el mismo matching y no nazca una
        # cuarta implementación de comparación de nombres.
        def _pantry_pick(item_norm: str, full_catalog, syn_map, allowed) -> str | None:
            return _catalog_pick_wb(item_norm, full_catalog, syn_map, allowed)

        _allow_p, _allow_c = set(filtered_proteins), set(filtered_carbs)
        _allow_v, _allow_f = set(filtered_veggies), set(filtered_fruits)
        for item in csl_lower:
            _p = _pantry_pick(item, DOMINICAN_PROTEINS, protein_synonyms, _allow_p)
            if _p and _p not in extracted_p:
                extracted_p.append(_p)
            _c = _pantry_pick(item, DOMINICAN_CARBS, carb_synonyms, _allow_c)
            if _c and _c not in extracted_c:
                extracted_c.append(_c)
            _v = _pantry_pick(item, DOMINICAN_VEGGIES_FATS, veggie_fat_synonyms, _allow_v)
            if _v and _v not in extracted_v:
                extracted_v.append(_v)
            _f = _pantry_pick(item, DOMINICAN_FRUITS, fruit_synonyms, _allow_f)
            if _f and _f not in extracted_f:
                extracted_f.append(_f)

        # [P1-PANTRY-FLOOR-CLINICAL-FILTER · 2026-08-02] (audit solver+seeder v7) Lo extraído pasa
        # por los MISMOS filtros de "puede ser base principal" que el sorteo — que corren 400
        # líneas más arriba y solo tocan PESOS, así que el reemplazo del pool por la nevera los
        # bypaseaba enteros. Corre AQUÍ, ANTES del `len(extracted_p) >= _min_p` de abajo, a
        # propósito: un pool que queda corto tras filtrar cae por la rama que YA existe (nevera
        # primero + sorteo completa, sin lock) en vez de por una rama nueva. Los embutidos
        # filtrados NO desaparecen de la nevera — dejan de ser candidatos a base del día y
        # siguen ofrecidos como acompañante por el prompt de nevera. Rollback:
        # MEALFIT_PANTRY_FLOOR_CLINICAL_FILTER=false.
        # Las dos condiciones son COPIA LITERAL de las que gobiernan los penalties del sorteo
        # (`_main_goal in _GOALS_PENALIZE_PROCESSED or _is_bariatric` para el ×0.1 de embutidos;
        # `_main_goal == "gain_muscle" or _is_bariatric` + su knob para el reemplazo de mains de
        # baja densidad), a propósito y sin ampliarlas: el bug era que la nevera las esquivaba,
        # no que fueran insuficientes.
        if PANTRY_FLOOR_CLINICAL_FILTER:
            extracted_p, extracted_f = _pantry_clinical_main_filter(
                extracted_p, extracted_f,
                penaliza_procesados=(_main_goal in _GOALS_PENALIZE_PROCESSED or _is_bariatric),
                exige_densidad=((_main_goal == "gain_muscle" or _is_bariatric)
                                and _env_bool("MEALFIT_GAINMUSCLE_HIGH_DENSITY_PROTEIN", True)),
                is_bariatric=_is_bariatric)

        # [P3-SEEDER-TEMPLATE-COVERAGE · 2026-08-04] Snapshot POST-filtro clínico: lo que el filtro
        # de arriba descartó ya no puede ocupar días, así que tampoco debe generar telemetría.
        _tpl_pantry_p, _tpl_pantry_c = list(extracted_p), list(extracted_c)

        # [P2-PANTRY-ROTATION-FLOOR · 2026-07-29] (audit solver+seeder v4) Esto REEMPLAZABA los pools
        # por lo extraído de la nevera y forzaba `cycle_locked = True` INCONDICIONALMENTE — sin sorteo,
        # sin cap y sin mínimo. `useRegeneratePlan.js` manda `current_pantry_ingredients` en TODA
        # renovación, así que si tras el `/inventory/consume` la nevera conserva UNA sola proteína
        # reconocible (p.ej. Huevos), el padding la cicla y `chosen_proteins = [Huevos]×3`: los 3 días
        # con huevo mientras el MISMO prompt dice "el HUEVO no debe aparecer en más de 2-3 comidas de
        # todo el plan" y `cycle_locked` prohíbe introducir otra base. El plan nacía obligado a violar
        # su propio cap, y `build_variety_report` lo cazaba después.
        # Además abría una puerta lateral al bloqueo que `MEALFIT_GROCERY_CYCLE_LOCK` tiene APAGADO
        # por default desde P1-VARIETY-RENEWAL-NO-CYCLE-LOCK, justo porque el owner pidió variedad
        # de ingredientes sobre reuso.
        # Ahora la nevera es un PISO, no una camisa de fuerza: va PRIMERO (conserva la prioridad de
        # ahorro) y se completa con el sorteo ponderado hasta el mínimo; solo con suficientes bases
        # propias se activa el lock. tooltip-anchor: P2-PANTRY-ROTATION-FLOOR
        _min_p = PANTRY_ROTATION_MIN_PROTEINS
        if extracted_p:
            if len(extracted_p) >= _min_p:
                unique_proteins = extracted_p
            else:
                _rest = [p for p in unique_proteins if p not in extracted_p]
                unique_proteins = extracted_p + _rest
                logger.info(
                    f"🧊 [P2-PANTRY-ROTATION-FLOOR] la nevera aportó {len(extracted_p)} proteína(s) "
                    f"(<{_min_p}) → se completa con el sorteo ponderado y NO se activa cycle-lock "
                    f"(evita los 3 días con la misma base).")
        # [P2-PANTRY-FLOOR-ALL-CATS · 2026-07-30] (audit solver+seeder v5) El floor de v4 aterrizó
        # SOLO en proteínas; estas 3 categorías conservaban el reemplazo incondicional, que es
        # literalmente el bug que aquel fix describe: con UN carbo reconocible tras
        # `/inventory/consume`, el padding lo cicla y los 3 días salen con la misma base (la
        # monotonía medida en el 39% de los días). Con vegetales es peor — 1 extraído ⇒ los 6
        # slots del prompt idénticos. Mismo patrón que las proteínas: la nevera va PRIMERO
        # (conserva la prioridad de ahorro) y se completa con el sorteo hasta el mínimo.
        # Umbral 2 para carbos/frutas; 3 para vegetales, que llenan 6 slots (2 por día).
        def _floor_pool(_extracted, _sorteo, _min):
            if len(_extracted) >= _min:
                return _extracted
            return _extracted + [x for x in _sorteo if x not in _extracted]

        if extracted_c:
            unique_carbs = _floor_pool(extracted_c, unique_carbs, 2)
        if extracted_v:
            unique_veggies = _floor_pool(extracted_v, unique_veggies, 3)
        if extracted_f:
            unique_fruits = _floor_pool(extracted_f, unique_fruits, 2)
        # El lock solo si la nevera sostiene la rotación por sí sola.
        # [P3-CYCLE-LOCK-ADDITIVE · 2026-07-31] (audit v6 · F19) Era una asignación directa que
        # PISABA el `cycle_locked = True` del grocery-cycle: dos guardas sobre el mismo campo,
        # la segunda siempre gana. Con el knob del ciclo encendido y una nevera sin proteínas
        # tras `/inventory/consume`, el plan usaba la base intersectada del ciclo pero SIN la
        # regla de ahorro, o sea el coste del lock sin su beneficio. Acumulativo.
        # tooltip-anchor: P3-CYCLE-LOCK-ADDITIVE
        _pantry_sustains_rotation = bool(extracted_p) and len(extracted_p) >= _min_p
        cycle_locked = cycle_locked or _pantry_sustains_rotation
        
    # ======= FORCED INGREDIENT INJECTION (FROM RAG/HISTORY) =======
    # [P3-FORCED-ALLOWED-CLEANUP · 2026-07-30] (audit solver+seeder v5) BLOQUE ELIMINADO.
    #
    # Era una rama con CERO productores en todo el repo (grep repo-wide: `_force_base_proteins`
    # solo aparecia aqui, como CONSUMIDOR — ni el backend ni el frontend lo emiten nunca) y con
    # tres defectos dentro, listos para el primer caller que la reactivara:
    #
    #   1. `if item_n in banned or banned in item_n` — DOBLE substring sin word-boundary:
    #      'res' esta dentro de 'ensalada fresca' y 'pollo' dentro de 'repollo guisado', asi que
    #      un dislike de "Ensalada fresca" baneaba la Res antes de llegar a las keywords. Es la
    #      misma clase que ya mordio 13 veces en este repo.
    #   2. Los fallbacks (`if len(...) < 3: unique_proteins = _forced_p`) restauraban la lista
    #      forzada COMPLETA — bans legitimos incluidos — cuando el filtro dejaba menos del
    #      minimo: un filtro simultaneamente sobre-inclusivo (por 1) y no-op (por esto), donde
    #      cual de los dos gana depende del CONTEO.
    #   3. Con `_force_base_proteins=[]` el padding de mas abajo hacia `% len([])` →
    #      ZeroDivisionError.
    #
    # Codigo muerto con bugs es deuda que solo espera a un caller nuevo para volverse incidente,
    # y ademas era una puerta lateral al bloqueo que MEALFIT_GROCERY_CYCLE_LOCK tiene APAGADO por
    # default desde P1-VARIETY-RENEWAL-NO-CYCLE-LOCK (el owner pidio variedad sobre reuso).
    # Si vuelve a hacer falta forzar bases desde RAG/historial, se escribe de nuevo con
    # word-boundary y sin fallback-que-restaura-bans desde el dia 1.
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
    # [P3-CYCLE-PERSIST-AFTER-POOLS · 2026-07-31] (audit v6 · F20) Escritura del ciclo con los pools
    # YA definitivos (post-nevera, post-dedupe) y ANTES del shuffle/padding, que introduce duplicados
    # cíclicos que no deben persistirse como “base”. Así lo que el ciclo guarda es lo que el plan
    # realmente usa. tooltip-anchor: P3-CYCLE-PERSIST-AFTER-POOLS
    if _cycle_persist_pending and user_id:
        try:
            _new_cycle = dict(_cycle_persist_pending)
            _new_cycle.update({"base_proteins": list(unique_proteins),
                               "base_carbs": list(unique_carbs),
                               "base_veggies": list(unique_veggies)})

            def _grocery_cycle_mutator(_hp):
                _hp["grocery_cycle"] = _new_cycle
                return None

            update_user_health_profile_atomic(user_id, _grocery_cycle_mutator)
            logger.info(f"💾 [GROCERY CYCLE] Base del ciclo guardada con los pools definitivos: "
                        f"{len(_new_cycle['base_proteins'])}P/{len(_new_cycle['base_carbs'])}C/"
                        f"{len(_new_cycle['base_veggies'])}V.")
        except Exception as _cp_e:
            logger.error(f"[P3-CYCLE-PERSIST-AFTER-POOLS] no se pudo guardar el ciclo: "
                         f"{type(_cp_e).__name__}: {_cp_e}")

    random.shuffle(unique_proteins)
    random.shuffle(unique_carbs)
    random.shuffle(unique_veggies)
    if unique_fruits:
        random.shuffle(unique_fruits)

    # Cada día recibe una proteína, un carbohidrato, y un vegetal únicos (sin repeticiones entre días)
    # Si no se pudieron elegir 3, rellenamos ciclando lo que hay
    # [P2-SEEDER-DAYS-COUNT · 2026-08-03] Los objetivos del padding escalan con el chunk: `_dc`
    # bases (1/día) y `2*_dc` vegetales (2/día). Sin esto el reparto seguiría teniendo material
    # para 3 días aunque el helper pidiera 4 pares.
    _base_proteins = list(unique_proteins)
    while len(unique_proteins) < _dc:
        unique_proteins.append(_base_proteins[len(unique_proteins) % len(_base_proteins)])
    _base_carbs = list(unique_carbs)
    while len(unique_carbs) < _dc:
        unique_carbs.append(_base_carbs[len(unique_carbs) % len(_base_carbs)])
    _base_veggies = list(unique_veggies)
    while len(unique_veggies) < 2 * _dc:
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
        # [P1-FRUIT-SEEDER-GATE-CONTRACT · 2026-07-26] Fuera `Limón` y `Naranja` de la lista de
        # relleno: el gate las excluye A PROPÓSITO (ralladura/aderezo, no "la fruta del plato"), así
        # que ocupaban un slot del pool sin poder satisfacer nunca "una fruta distinta por comida".
        # El pool del caso vivo era `['Níspero','Toronja','Limón']` — un slot gastado en un
        # condimento. Todas las de esta lista SÍ están en `_FEATURED_FRUITS`.
        _DEFAULT_DR_FRUITS = (
            'Lechosa', 'Mango', 'Piña', 'Guineo', 'Fresas', 'Chinola',
            'Melón', 'Manzana', 'Guayaba', 'Mandarina',
        )
        # [P1-FRUIT-PAD-FILTERED · 2026-07-30] (audit solver+seeder v5) El relleno recorría
        # `_DEFAULT_DR_FRUITS` con un ÚNICO chequeo ("¿ya está?") y jamás consultaba
        # `filtered_fruits` — deshaciendo el filtro de alergias/dislikes/dieta que este mismo
        # seeder aplicó al principio. Y la fruta rechazada era la MÁS probable de entrar:
        # precisamente por estar excluida del pool no aparecía en `existing`, así que el padding
        # la elegía. Con `num_fruits_to_pick = min(2, ...)` el while añade ≥2 defaults en CADA
        # plan con frutas, así que no era un borde: era el caso normal.
        # `P1-FRUIT-SEEDER-GATE-CONTRACT` ya había curado esta lista por el lado del GATE; esto
        # la cura por el lado del USUARIO. Comparación normalizada (sin acentos, tolerando
        # singular/plural tipo Fresa/Fresas) porque el catálogo y la tupla no coinciden en forma.
        def _fruit_key(_s: str) -> str:
            _k = strip_accents(str(_s).lower()).strip()
            return _k[:-1] if _k.endswith("s") else _k

        _allowed_fruit_keys = {_fruit_key(_f) for _f in (filtered_fruits or [])}
        # [P1-FRUIT-SEEDER-GATE-CONTRACT] 4 y no 3: el reparto da 2 frutas distintas por día
        # rotando sobre 4, así la semana usa 4 frutas en vez de las 6 que costaría 2×3 sin reutilizar.
        # [P2-SEEDER-DAYS-COUNT · 2026-08-03] `_dc + 1` generaliza ese "4 para 3 días": con n+1
        # frutas (n = _dc días) la rotación da n pares `(f[i], f[i+1])` para i en 0..n-1 SIN
        # reciclar índices (n+1 > n, así que nunca hace falta el módulo) — el último es
        # [FINAL-REVIEW-P2 · 2026-08-03] (f[n-1], f[n]), no (f[n-1], f[0]) como decía este
        # comentario (verificado ejecutando `_rotate_pairs`). La razón real sigue siendo la
        # economía de lista: n+1 frutas cubren n días con 2 distintas cada uno, en vez de las
        # 2n que costaría sin reutilizar — y el día n-1 nunca clona el par del día 0 porque su
        # segundo elemento es f[n], no f[0]. Con `_dc=3` sigue siendo 4, byte-idéntico.
        while len(unique_fruits) < _dc + 1:
            # [P3-FRUIT-PAD-KEY-SYMMETRY · 2026-07-31] (audit v6 · F21) Era `{f.lower() ...}`: el
            # chequeo de PERMITIDAS normalizaba singular/plural con `_fruit_key` y el de DUPLICADAS
            # no, así que con 'Fresa' ya en la lista el pad añadía 'Fresas' y un día recibía
            # (Fresa, Fresas) como sus "dos frutas distintas". Dos comparaciones sobre el mismo
            # concepto con criterios distintos. El bug está en el chequeo, no en el dato: renombrar
            # 'Fresas' en `_DEFAULT_DR_FRUITS` cerraría el caso de hoy y dejaría la clase abierta.
            # tooltip-anchor: P3-FRUIT-PAD-KEY-SYMMETRY
            existing = {_fruit_key(f) for f in unique_fruits}
            # Prioridad 1: añadir una fruta DR default que NO esté ya presente Y que el usuario
            # pueda comer. Garantiza variedad cross-day (cada día recibe fruta distinta).
            _added = False
            for _df in _DEFAULT_DR_FRUITS:
                if _fruit_key(_df) in existing:
                    continue
                if _allowed_fruit_keys and _fruit_key(_df) not in _allowed_fruit_keys:
                    continue   # alergia / dislike / dieta la excluyó del pool
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
    
    # [P2-SEEDER-DAYS-COUNT · 2026-08-03] Los cortes escalan con el chunk (ver el padding arriba).
    chosen_proteins = unique_proteins[:_dc]
    chosen_carbs = unique_carbs[:_dc]
    chosen_veggies = unique_veggies[:2 * _dc]
    # [P1-FRUIT-SEEDER-GATE-CONTRACT · 2026-07-26] 4 frutas: `_rotate_fruit_pairs` las reparte en
    # 2 por día reutilizándolas entre días.
    chosen_fruits = unique_fruits[:_dc + 1] if unique_fruits else []

    # [P3-SEEDER-TEMPLATE-COVERAGE · 2026-08-04] TELEMETRÍA, no guard. Una base de la nevera sin
    # ninguna plantilla propia que además ocupa ≥2 de los días del chunk es el caso que el audit
    # describe: bajo `cycle_locked` el prompt prohíbe introducir bases nuevas, así que esos dos
    # días se componen enteros por improvisación y acaban pareciéndose. Se registra ANTES de
    # decidir un guard más duro (rotar la base, relajar el lock) porque no hay serie que diga
    # cuántas veces pasa de verdad; el sorteo NO la excluye y este bloque no cambia nada.
    # tooltip-anchor: P3-SEEDER-TEMPLATE-COVERAGE
    try:
        for _tpl_cat, _tpl_field, _tpl_bases, _tpl_chosen in (
                ("proteína", "protein", _tpl_pantry_p, chosen_proteins),
                ("carbohidrato", "base", _tpl_pantry_c, chosen_carbs)):
            for _tpl_b in _tpl_bases:
                _tpl_days = sum(1 for _x in _tpl_chosen if _x == _tpl_b)
                if _tpl_days >= 2 and _template_coverage(_tpl_b, _tpl_field) == 0:
                    logger.warning(
                        "🍽️ [P3-SEEDER-TEMPLATE-COVERAGE] la %s '%s' viene de la NEVERA, no tiene "
                        "NINGUNA plantilla de %s en la biblioteca y ocupa %d de %d día(s) del "
                        "chunk: esos días se componen sin recombinación y tienden a clonar la "
                        "fórmula. Telemetría — el sorteo NO la excluye.",
                        _tpl_cat, _tpl_b, _TEMPLATE_COVERAGE_MAIN_SLOT, _tpl_days, _dc)
    except Exception as _tpl_w_e:
        logger.debug("[P3-SEEDER-TEMPLATE-COVERAGE] telemetría de nevera no-op: %s: %s",
                     type(_tpl_w_e).__name__, str(_tpl_w_e)[:160])

    # Repetimos mezcla final de los días elegidos para distribuir el orden
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
    
    # [P1-FRUIT-SEEDER-GATE-CONTRACT · 2026-07-26] DOS frutas por día, no una.
    #
    # El contrato estaba roto por los dos lados y por eso "fruta repetida el mismo día" era la
    # razón de rechazo dominante (67% de los planes de la línea base reintentaban):
    #
    #   · el seeder asignaba UNA fruta por día (`fruit_0/1/2`)
    #   · el gate exige una fruta dulce DISTINTA por COMIDA dentro del día
    #
    # Un día con fruta en el desayuno Y en la merienda —la forma más común— no puede cumplir eso
    # desde el pool: el day-gen repite la única que tiene (el gate rechaza) o improvisa una de
    # fuera del pool (y choca cross-día). Caso vivo del 07:47: pool `['Níspero','Toronja','Limón']`
    # → usó guineo, que no estaba asignado. NO es un fallo del modelo; la instrucción era
    # insatisfacible, y un modelo mejor se estrella igual.
    #
    # Reparto CONSERVADOR con la lista de compras en mente: 4 frutas distintas por semana en vez de
    # 3, y cada día recibe dos consecutivas de la rotación (día i → frutas[i], frutas[i+1]), así
    # que las frutas se reutilizan entre días en lugar de comprar 6.
    _fruit_slots = _rotate_fruit_pairs(chosen_fruits, days=_dc)
    if _fruit_slots:
        fruit_params = {f"fruit_{i}": _fruit_slots[i][0] for i in range(_dc)}
        fruit_params.update({f"fruit_{i}b": _fruit_slots[i][1] for i in range(_dc)})
    else:
        _fallback_fruit = "elige la fruta dominicana que mejor combine con la preparación"
        fruit_params = {}
        for _i in range(_dc):
            fruit_params[f"fruit_{_i}"] = _fallback_fruit
            fruit_params[f"fruit_{_i}b"] = _fallback_fruit

    # [P1-CARB-SEEDER-PAIRS · 2026-07-27] SEGUNDA base por día — el mismo contrato roto que
    # P1-FRUIT-SEEDER-GATE-CONTRACT cerró ayer para la fruta, con la última categoría que quedaba.
    #
    # El reparto era: veggie_i + veggie_ib (dos), fruit_i + fruit_ib (dos, desde ayer)… y carb_i
    # SOLO UNO. Pero un día tiene almuerzo Y cena, y ambos llevan base. Con una sola asignada, el
    # day-gen la usa en las dos. No es un fallo del modelo: la asignación no daba para otra cosa.
    #
    # Medido sobre 23 días de 8 planes vivos: **9 días (39%)** repiten una base en 2+ comidas. En
    # el plan del owner, los 3 días de 3 — papa (599 g el lunes), papa (824 g el martes), yautía.
    #
    # Se reparte con `_rotate_pairs` (día i → carbos[i], carbos[i+1]), el MISMO helper de la fruta:
    # las bases se reutilizan entre días en vez de comprar 6 distintas, así que la lista de compras
    # crece en una como mucho. Sin lista utilizable → se omite el segundo y el prompt cae a su
    # redacción previa (fail-open: nunca peor que hoy).
    # [P2-SEEDER-PAIRS-GOALS · 2026-07-31] (audit v6 · F17) Los DOS slots del día salen ahora de la
    # MISMA rotación. Antes el primero venía de `chosen_carbs` (lista PADDED a 3 por repetición) y
    # el segundo de `_carb_slots` (rotación DEDUPLICADA): dos listas derivadas por separado pareadas
    # por índice. Con 2 bases únicas —el caso POR DEFECTO, `num_carbs_to_pick = min(2, ...)`— el
    # índice i deja de señalar al mismo elemento y sale `carb_i == carb_ib`: el prompt ASIGNA una
    # base y en la misma frase PROHÍBE repetirla. Medido: 29 de 90 días colisionaban.
    # Frutas y vegetales ya tomaban ambos slots del mismo `_slots[i]`; los carbos eran la única
    # categoría fuera del contrato. tooltip-anchor: P2-SEEDER-PAIRS-GOALS
    _carb_slots = _rotate_pairs(chosen_carbs, days=_dc)
    if _carb_slots:
        carb_params = {f"carb_{i}": _carb_slots[i][0] for i in range(_dc)}
        carb_params.update({f"carb_{i}b": _carb_slots[i][1] for i in range(_dc)})
    else:
        # Una sola base única: `_rotate_pairs` devuelve None y pedir "no repitas" es insatisfacible.
        # Fail-open deliberado (P1-CARB-BASE-NO-REPEAT), no tocar.
        # [P2-SEEDER-DAYS-COUNT] `% len(chosen_carbs)` porque el padding puede haber quedado corto
        # si el pool llegó vacío por otra vía; un IndexError aquí tumbaría el nodo entero.
        carb_params = {f"carb_{i}": chosen_carbs[i % len(chosen_carbs)] for i in range(_dc)}
        carb_params.update({f"carb_{i}b": "otra base distinta del catálogo" for i in range(_dc)})
    logger.info(f"✅ [P1-CARB-SEEDER-PAIRS] par de bases por día (las dos del mismo reparto): "
                f"{[(carb_params[f'carb_{i}'], carb_params[f'carb_{i}b']) for i in range(_dc)]}")

    # [P2-LIGHT-PROTEIN-SEED · 2026-07-29] (audit solver+seeder v4) El seeder reparte por día proteína
    # principal, carbo (×2, P1-CARB-SEEDER-PAIRS), vegetal/grasa (×2) y fruta (×2,
    # P1-FRUIT-SEEDER-GATE-CONTRACT). El ANCLA PROTEICA de desayuno/merienda era la última categoría
    # sin sorteo: 5 viñetas literales, idénticas para los 3 días de TODOS los planes. En un plan de
    # 4×3, eso son 6 de 12 comidas eligiendo de un menú fijo donde 2 opciones son frutos secos y 1 es
    # huevo — el origen medido del "maní en 4 de 12" y del huevo topando su cap. El advisory
    # `cross_day_snack_pair_repeats` que se añadió hoy está midiendo el síntoma de este literal.
    #
    # Nace OFF (mismo criterio que MEALFIT_FAT_LEAN_SWAP / MEALFIT_REFINE_HOUSEHOLD_LINES): con el
    # knob apagado el prompt es BYTE-IDÉNTICO al actual (protege el prompt-cache) porque el bloque
    # inyectado es la cadena vacía. tooltip-anchor: P2-LIGHT-PROTEIN-SEED
    _light_block = ""
    if LIGHT_PROTEIN_SEED:
        try:
            # [P2-LIGHT-PROTEIN-POOL · 2026-07-30] (audit solver+seeder v5) el pool se construía
            # SOLO desde `filtered_veggies`, donde no existe ningún lácteo → 3 de los 8 tokens
            # ('queso', 'yogur', 'ricotta') no podían matchear jamás y el ancla acababa siendo
            # 100% frutos secos, reforzando el "maní en 4 de 12" que el knob venía a curar.
            # [P3-LIGHT-ANCHOR-NOT-BLOCKED · 2026-07-31] (audit v6 · C1) El ancla no puede nombrar
            # lo que el MISMO prompt prohibe tres bloques antes: `blocked_text` lista los
            # sobreusados y el ancla los podía sortear igual ("EVITA... Maní" + "día B → Maní").
            # Igualdad EXACTA, no subcadena: 'Maní' y 'Mantequilla de maní' son ítems distintos del
            # catálogo y bloquear uno no debe borrar el otro; ambos lados salen de las mismas listas.
            # tooltip-anchor: P3-LIGHT-ANCHOR-NOT-BLOCKED
            _over_light = {str(x).lower() for x in (list(used_proteins) + list(used_veggies))}
            _light_pool = [x for x in _build_light_protein_pool(
                filtered_veggies, filtered_proteins, bariatric=_is_bariatric)
                if str(x).lower() not in _over_light]
            if len(_light_pool) >= 2:
                # [P2-SEEDER-PAIRS-GOALS · 2026-07-31] (audit v6 · F18) era `_light_pool[:4]`,
                # un slice posicional que cortaba antes del primer lácteo. Ver el helper.
                # [P2-SEEDER-DAYS-COUNT · 2026-08-03] El ancla se sortea para los `_dc` días del
                # chunk, no para 3: con 4 días el día D se quedaba sin línea propia (el prompt
                # solo nombraba A/B/C) mientras el resto del reparto sí escalaba.
                _light_slots = _rotate_pairs(
                    _pick_light_anchor_candidates(_light_pool, max(4, _dc + 1)), days=_dc)
                if _light_slots:
                    _l = [" o ".join(s) if isinstance(s, (list, tuple)) else str(s)
                          for s in _light_slots[:_dc]]
                    while len(_l) < _dc:
                        _l.append(_l[-1] if _l else "")
                    # SSOT de la etiqueta del día: la misma función que numera las OPCIONES del
                    # prompt (importada arriba). Una segunda tabla de letras diverge en cuanto
                    # alguien toque el alfabeto, y las dos frases hablarían de días distintos.
                    _light_dias = " · ".join(
                        f"día {_prompt_option_letter(_i)} → {_l[_i]}" for _i in range(_dc))
                    _light_block = (
                        f"\n     ⭐ ANCLA LIVIANA SORTEADA POR DÍA (prioriza ESTA sobre las viñetas "
                        f"genéricas de abajo): {_light_dias}.")
                    logger.info(f"🥚 [P2-LIGHT-PROTEIN-SEED] anclas livianas por día: {_l}")
        except Exception as _lp_e:
            _light_block = ""   # fail-open: el literal de siempre nunca es peor que hoy
            logger.warning(f"[P2-LIGHT-PROTEIN-SEED] sorteo no-op: {type(_lp_e).__name__}: {_lp_e}")

    # [P2-VEGGIE-PAIRS-ROTATE · 2026-07-30] (audit solver+seeder v5) El par del día se derivaba
    # por OFFSET POSICIONAL (`chosen_veggies[i]` + `chosen_veggies[i+3]`) sin ninguna garantía de
    # que fueran distintos. Con 3 vegetales únicos el padding produce [a,b,c,a,b,c] y, tras el
    # shuffle, ~20% de los días reciben el MISMO vegetal dos veces (40% con 2 únicos, 100% con 1):
    # "Día A: Zanahoria y Zanahoria", contradiciendo la instrucción de "2 vegetales distintos" del
    # propio prompt. El day-gen entonces o repite (y el gate de repetición intra-día rechaza) o
    # improvisa fuera del pool — el mismo mecanismo insatisfacible que P1-FRUIT-SEEDER-GATE-CONTRACT
    # documentó: no es fallo del modelo, la asignación no da para otra cosa.
    # Frutas (P1-FRUIT-SEEDER-GATE-CONTRACT) y carbos (P1-CARB-SEEDER-PAIRS) ya usaban el helper
    # con dedupe; los vegetales eran la última categoría fuera del contrato.
    # [P3-VEGGIE-SLOTS-CONSUME-SIX · 2026-07-31] (audit v6 · F33) `_rotate_pairs` produce
    # (v[i], v[i+1]) rotando, o sea consume 4 de los 6 vegetales sorteados: 2 picks quedaban
    # INERTES —persistidos en `base_veggies` y eximidos del blocked_text, pero jamás ofrecidos— y
    # el log decía 6. Con 6 únicos se reparten en 3 pares DISJUNTOS (los 6 se usan y ningún
    # vegetal se repite entre días, que es la intención documentada en `num_veggies_to_pick`);
    # con menos de 6 se cae al helper compartido, mismo contrato que fruta y carbo.
    # tooltip-anchor: P3-VEGGIE-SLOTS-CONSUME-SIX
    # [P2-SEEDER-DAYS-COUNT · 2026-08-03] `2*_dc` en vez del 6 literal: la intención documentada
    # es "2 vegetales distintos por día", que con 4 días son 8 y no 6.
    _vg6 = list(dict.fromkeys(chosen_veggies[:2 * _dc]))
    if len(_vg6) >= 2 * _dc:
        _veg_slots = [(_vg6[2 * _i], _vg6[2 * _i + 1]) for _i in range(_dc)]
    else:
        _veg_slots = _rotate_pairs(chosen_veggies[:2 * _dc], days=_dc)
    if _veg_slots:
        veggie_params = {}
        for _i in range(_dc):
            veggie_params[f"veggie_{_i}"] = _veg_slots[_i][0]
            veggie_params[f"veggie_{_i}b"] = _veg_slots[_i][1]
        # [P2-VEGGIE-CHANNEL-DAYGEN · 2026-07-30] (audit solver+seeder v5) El reparto se publica
        # como DATO, no solo interpolado en la prosa del prompt. El day-generator lo necesita
        # (es quien escribe los ingredientes reales) y la salida tipada del planner no puede
        # transportarlo; parsear el prompt de vuelta sería frágil —los dos vegetales del día
        # viven en frases distintas— y se rompería con cualquier reescritura del copy.
        if isinstance(out_assignment, dict):
            # [P2-SEEDER-DAYS-COUNT · 2026-08-03] `[:_dc]`: las listas viajan con un par POR DÍA
            # del chunk. El estampado del orquestador sigue siendo `% len(pares)` sin cambios —
            # con largo == nº de días del esqueleto el módulo es la identidad, y si el pool
            # degradó a menos pares el módulo sigue siendo el fallback correcto.
            out_assignment["veggie_pairs"] = [tuple(p) for p in _veg_slots[:_dc]]
            # [P2-SEEDER-PAIRS-CHANNEL · 2026-07-31] (audit v6 · F23) Los repartos de carbo y fruta
            # se calculaban igual de determinísticamente que el de vegetales pero solo salían como
            # PROSA del prompt del planner: llegaban al day-gen únicamente si el LLM se molestaba
            # en copiarlos a `carb_pool`/`fruit_pool`. Publicarlos como DATO permite completarlos
            # en el esqueleto cuando el planner los deja cortos, que es cuando se pierde la regla
            # anti-repetición y el gate quema un retry. tooltip-anchor: P2-SEEDER-PAIRS-CHANNEL
            if _carb_slots:
                out_assignment["carb_pairs"] = [tuple(p) for p in _carb_slots[:_dc]]
            if _fruit_slots:
                out_assignment["fruit_pairs"] = [tuple(p) for p in _fruit_slots[:_dc]]
    else:
        # <2 únicos: fallback textual (mismo patrón que carb_ib) en vez de repetir el mismo nombre.
        veggie_params = {f"veggie_{_i}": chosen_veggies[_i % len(chosen_veggies)]
                         for _i in range(_dc)}
        veggie_params.update({f"veggie_{_i}b": "otro vegetal distinto del catálogo"
                              for _i in range(_dc)})
    # [P2-SEEDER-DAYS-COUNT · 2026-08-03] La plantilla se CONSTRUYE con el nº de días del chunk
    # (`build_deterministic_variety_prompt`) en vez de ser el literal de 3 opciones. Con `_dc=3`
    # devuelve byte a byte el prompt histórico.
    prompt = build_deterministic_variety_prompt(_dc).format(
        light_protein_block=_light_block,
        blocked_text=blocked_text,
        **{f"protein_{_i}": chosen_proteins[_i % len(chosen_proteins)] for _i in range(_dc)},
        **veggie_params,
        **carb_params,
        **fruit_params
    )
    logger.info(f"✅ [ANTI MODE-COLLAPSE] Proteínas elegidas para {_dc} día(s) (rotadas si es necesario): {chosen_proteins}")
    logger.info(f"✅ [ANTI MODE-COLLAPSE] Carbohidratos elegidos para {_dc} día(s) (rotados si es necesario): {chosen_carbs}")
    logger.info(f"✅ [ANTI MODE-COLLAPSE] Vegetales/Grasas elegidos (2 distintos por día): {chosen_veggies}")
    # [P1-FRUIT-SEEDER-GATE-CONTRACT · 2026-07-26] Se loguea el reparto POR DÍA y cuántas de las
    # frutas cuentan para el gate: el log anterior ("Fruta sugerida: [...]") no permitía ver que el
    # pool era insatisfacible — el forense del 07:47 tuvo que cruzarlo a mano con `_FEATURED_FRUITS`.
    logger.info(f"🍓 [ANTI MODE-COLLAPSE] Frutas por día (2 distintas c/u): {_fruit_slots or 'fallback'}"
                f" | reconocidas por el gate: {_n_gate_fruits(chosen_fruits)}/{len(chosen_fruits or [])}")
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
        # [P1-RECIPE-EXPAND-MODEL-PROVIDER · 2026-07-26] Proveedor por prefijo del modelo: el knob
        # `MEALFIT_RECIPE_EXPAND_MODEL` ya existía pero `ChatDeepSeek` a secas mandaba cualquier
        # valor al base_url de DeepSeek. Ahora puede apuntar a `gpt-5.6-luna` y funcionar.
        llm = _build_expand_llm(
            _recipe_expand_model_name(),  # [P1-RECIPE-EXPAND-FAILSIGNAL] knob, era hardcoded
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
