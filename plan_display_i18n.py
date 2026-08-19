# backend/plan_display_i18n.py
"""Motor de enriquecimiento de la capa de display i18n del plan.

tooltip-anchor: P1-PLAN-DISPLAY-I18N

Decisión arquitectónica (spec `docs/superpowers/specs/2026-08-19-plan-display-i18n-design.md`):
el plan se GENERA y PERSISTE siempre en español canónico — los nombres de alimentos y platos
son identificadores del sistema (`pantry_names_match`, coherence guard, backstop de alergias
resuelven por esos strings exactos; P1-I18N-DASHBOARD). Este módulo NUNCA lee ese campo de
vuelta ni condiciona su propia conducta a él — solo ESCRIBE un campo paralelo de solo-lectura:

    meal["_display"][locale] = {"name", "description", "recipe", "ingredients"}

que el frontend consume con fallback campo a campo al original si falta (fase 2, fuera de
este módulo). es-DO jamás lleva `_display` — byte-idéntico a hoy.

Contrato:
    enrich_plan_display(plan_id, user_id, locale, day_indices=None) -> dict
        {"enriched_meals": int, "skipped": str | None}. Jamás lanza (fail-open TOTAL — spec
        "Global Constraints": cualquier excepción del enriquecimiento es un warning + el plan
        se queda sin `_display` + el frontend cae a español, nunca rompe generación/swap/lectura).

    schedule_plan_display_enrichment(plan_id, user_id, locale, day_indices=None) -> None
        Wrapper fire-and-forget: thread background + dedupe. Los 5 disparadores (Task 2/3)
        llaman a este, no a `enrich_plan_display` directo.

Validación determinista (spec, sección "Contrato de datos"):
    - `recipe` e `ingredients` de salida deben tener la MISMA longitud que el original, en el
      mismo orden (alineados por índice) — si no, el meal ENTERO se descarta (no se persiste
      _display para ese meal).
    - Cada línea de `ingredients` debe contener el nombre canónico español del alimento como
      substring accent-insensitive (extraído del original — ver `_extract_canonical_name`). Si
      la línea traducida lo pierde, SOLO esa línea cae de vuelta al original español (fallback
      per-línea, NO descarta el meal — "un gloss que pierde el identificador es peor que no
      tener gloss"). Si el original no tiene canónico identificable, la línea pasa sin check.

[Fix round 1 · 2026-08-19, review `task-1-review.md`] Tres cambios de contrato sobre Task 1:
    - Lotes: el plan se trocea en lotes de `MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS` días por
      llamada LLM (Finding 6) — un solo lote gigante (todo el plan en una llamada) truncaba la
      salida y quemaba el costo sin recuperación parcial ni registro.
    - TOCTOU: `targets` se lee FUERA del row-lock (no se puede sostener un `FOR UPDATE` durante
      los ~segundos que dura la llamada LLM). El mutator de `_persist_batch` compara el nombre
      canónico del meal en esa posición contra el snapshot leído ANTES de escribir — si difiere
      (swap/regenerate-day concurrente), SALTA ese meal (Finding 5). Sin esto, la promesa de la
      spec ("stale display imposible por construcción") era falsa bajo carrera.
    - Lock cross-worker: se libera SIEMPRE en el `finally` (antes solo tenía TTL, Finding 4) y su
      clave incluye los `day_indices` del lote (dos disparadores sobre días distintos del mismo
      plan+locale ya no se bloquean entre sí).

Costo: NUNCA a `api_usage` (cero crédito de usuario) — telemetría a `llm_usage_events` vía
`db.log_llm_usage_event(node="plan_display_i18n")`, emitida INMEDIATAMENTE tras cada `invoke`
exitoso (Finding 7) — antes solo se emitía en el camino feliz completo, dejando ciegos los
modos de fallo post-invoke (JSON roto, 0 meals válidos, persist fallido).

Knobs (auto-registrados en `_KNOBS_REGISTRY` vía `knobs._env_bool/_env_str/_env_float/_env_int`):
    MEALFIT_PLAN_DISPLAY_I18N                default True  — kill switch total.
    MEALFIT_PLAN_DISPLAY_I18N_MODEL          default flash — convención P3-PREVIEW-MODEL-KNOB.
    MEALFIT_PLAN_DISPLAY_I18N_TIMEOUT_S      default 60.0  — timeout del cliente LLM.
    MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS     default 4     — días por llamada LLM (clamp 1-30).
    MEALFIT_PLAN_DISPLAY_I18N_MAX_OUTPUT_TOKENS default 8000 — cota de salida por lote.
"""

import json
import logging
import re
import threading
import time
from typing import Optional

from knobs import _env_bool, _env_str, _env_float, _env_int
from constants import strip_accents
from prompts.chat_agent import _COACH_LANGUAGE_NAMES
from llm_provider import build_chat_llm, DEEPSEEK_FLASH
from langchain_core.messages import SystemMessage
from db import update_plan_data_atomic, log_llm_usage_event, execute_sql_query, execute_sql_write

logger = logging.getLogger(__name__)

# ============================================================
# Knobs
# ============================================================


def _plan_display_i18n_enabled() -> bool:
    return _env_bool("MEALFIT_PLAN_DISPLAY_I18N", True)


def _plan_display_i18n_model_name() -> str:
    return _env_str("MEALFIT_PLAN_DISPLAY_I18N_MODEL", DEEPSEEK_FLASH)


def _plan_display_i18n_timeout_s() -> float:
    return _env_float("MEALFIT_PLAN_DISPLAY_I18N_TIMEOUT_S", 60.0)


def _plan_display_i18n_batch_days() -> int:
    """[Finding 6 · fix round 1] Días por llamada LLM. `day_indices=None` (todo
    el plan, disparador 4 de la spec — cambio de idioma con plan ya creado) es
    justo el caso que motivó la feature Y el que más probablemente reventaba:
    ~28 días × 4-5 meals en UNA sola llamada trunca la salida."""
    return _env_int(
        "MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS",
        4,
        validator=lambda v: 1 <= v <= 30,
    )


def _plan_display_i18n_max_output_tokens() -> int:
    return _env_int(
        "MEALFIT_PLAN_DISPLAY_I18N_MAX_OUTPUT_TOKENS",
        8000,
        validator=lambda v: 500 <= v <= 32000,
    )


def _circuit_breaker_can_proceed(model_name: str) -> bool:
    """[Finding 10 · fix round 1] Consulta al circuit breaker ANTES de gastar
    threads/latencia en un provider ya caído — mismo patrón que el callsite de
    referencia (`agent.py:4771-4779`, título del chat). Import LOCAL: evitar que
    `graph_orchestrator` (módulo pesado) entre en el costo de import de este
    módulo (contrato verificado: import sin efectos DB/LLM).

    Fail-open: si el CB no se puede resolver (import falla, DB caída), asumimos
    que se puede proceder — un circuit breaker que bloquea por su propia
    indisponibilidad sería peor que no tenerlo.
    """
    try:
        from graph_orchestrator import _get_circuit_breaker
        return _get_circuit_breaker(model_name).can_proceed()
    except Exception as e:
        logger.debug(f"[P1-PLAN-DISPLAY-I18N] circuit breaker check falló (fail-open): {e!r}")
        return True


# ============================================================
# Dedupe: lock in-process + marker cross-worker en app_kv_store
# (mismo patrón que `agent.py::_try_claim_title_lock_cross_worker`)
#
# [Finding 4 · fix round 1] Dos cambios sobre Task 1:
#   1. El marker cross-worker se LIBERA en el `finally` de `enrich_plan_display`
#      (antes solo expiraba por TTL a los 5 min) — con 5 disparadores sobre el
#      mismo (plan_id, locale) el TTL viejo bloqueaba un re-enriquecimiento
#      legítimo (p.ej. swap 3 min después del persist inicial) durante toda la
#      ventana, sin reintento ni alerta.
#   2. La clave incluye los `day_indices` del lote — dos disparadores sobre
#      DÍAS DISTINTOS del mismo plan+locale ya no se bloquean entre sí.
# El set in-process (`_INFLIGHT`) es un pre-filtro barato de proceso (NO
# incluye day_indices a propósito: es solo una optimización para evitar un
# roundtrip a la DB cuando el mismo worker ya tiene ALGO en vuelo para ese
# plan+locale; la autoridad real es el marker cross-worker de abajo).
# ============================================================

_INFLIGHT: set = set()
_INFLIGHT_LOCK = threading.Lock()
_ENRICH_LOCK_TTL_S = 300  # Red de seguridad si el proceso muere ANTES del finally.


def _enrich_lock_kv_key(plan_id: str, locale: str, day_indices: list) -> str:
    """Determinista: el mismo lote de días (en cualquier orden de entrada,
    `enrich_plan_display` ya normaliza a sorted+dedup antes de llamar aquí)
    produce siempre la misma key."""
    scope = ",".join(str(i) for i in day_indices) if day_indices else "none"
    if len(scope) > 200:
        import hashlib
        scope = "h" + hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16]
    return f"plan_display_enrich:{plan_id}:{locale}:{scope}"


def _try_claim_enrich_lock_cross_worker(plan_id: str, locale: str, day_indices: list) -> bool:
    """Claim atómico cross-worker vía UPSERT en `app_kv_store`. Espejo de
    `agent.py::_try_claim_title_lock_cross_worker`. Best-effort: si la DB no
    responde, retorna True (fail-open) — preferimos duplicar una llamada
    flash barata a bloquear el enriquecimiento por un outage del KV.
    """
    try:
        _now_ts = time.time()
        _kv_key = _enrich_lock_kv_key(plan_id, locale, day_indices)
        result = execute_sql_query(
            """
            INSERT INTO app_kv_store (key, value)
            VALUES (%s, jsonb_build_object('started_at', %s::float))
            ON CONFLICT (key) DO UPDATE SET
                value = jsonb_build_object('started_at', %s::float),
                updated_at = NOW()
            WHERE COALESCE((app_kv_store.value->>'started_at')::float, 0)
                  < %s::float
            RETURNING key
            """,
            (_kv_key, _now_ts, _now_ts, _now_ts - _ENRICH_LOCK_TTL_S),
            fetch_all=True,
        )
        return bool(result)
    except Exception as e:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] claim de lock cross-worker falló (fail-open) "
            f"plan={plan_id} locale={locale}: {e!r}"
        )
        return True


def _release_enrich_lock_cross_worker(plan_id: str, locale: str, day_indices: list) -> None:
    """[Finding 4 · fix round 1] DELETE best-effort del marker — libera el slot
    para el PRÓXIMO disparador legítimo dentro de la ventana TTL en vez de
    obligarlo a esperar hasta 5 min. El TTL se conserva como red de seguridad
    para el caso donde el proceso muere ANTES de llegar a este `finally`."""
    try:
        _kv_key = _enrich_lock_kv_key(plan_id, locale, day_indices)
        execute_sql_write("DELETE FROM app_kv_store WHERE key = %s", (_kv_key,))
    except Exception as e:
        logger.debug(
            f"[P1-PLAN-DISPLAY-I18N] release de lock cross-worker falló (best-effort) "
            f"plan={plan_id} locale={locale}: {e!r}"
        )


# ============================================================
# Lectura del plan (ownership AND user_id, mismo patrón de routers/plans.py)
# ============================================================


def _fetch_plan_data(plan_id: str, user_id: str) -> Optional[dict]:
    # tooltip-anchor: I2 user_id filter (test_p1_plan_display_i18n.py sección "ownership").
    try:
        row = execute_sql_query(
            "SELECT plan_data FROM meal_plans WHERE id = %s AND user_id = %s",
            (plan_id, user_id),
            fetch_one=True,
        )
    except Exception as e:
        logger.warning(f"[P1-PLAN-DISPLAY-I18N] SELECT plan falló plan={plan_id}: {e!r}")
        return None
    if not row:
        return None
    pd = row.get("plan_data")
    if isinstance(pd, str):
        try:
            pd = json.loads(pd)
        except Exception:
            return None
    if not isinstance(pd, dict):
        return None
    return pd


def _normalize_day_indices(day_indices, days_len: int) -> list:
    """`None` -> todos los días del plan. Lista arbitraria -> dedup + sorted +
    solo enteros reales (nunca bool, que es subclase de int en Python)."""
    if day_indices is None:
        return list(range(days_len))
    try:
        cleaned = {int(i) for i in day_indices if isinstance(i, int) and not isinstance(i, bool)}
    except TypeError:
        return []
    return sorted(cleaned)


# ============================================================
# Extracción del nombre canónico.
#
# [Finding 1 · fix round 1] La v1 de Task 1 devolvía "todo el resto de la
# línea" tras quitar cantidad+unidad — contra el corpus real (formato
# dominante "<cantidad> <unidad> de <alimento> <modificador>") eso arrastraba
# el "de " conector Y los modificadores de estado de cocción ("secas",
# "cocidas", "picadas", "rallada", "al gusto"), que el LLM NUNCA reproduce
# dentro de un gloss porque no son parte del nombre del alimento. El prompt
# pedía "Nombre canónico" y el validador comparaba contra "de Nombre canónico
# modificador" — descartaba ~10 de cada 11 líneas reales con gloss perfecto.
#
# Fix en 3 pasos, medido contra las 11 líneas reales del review (11/11,
# mínimo pactado 10/11 — ver test_canonical_extraction_validates_against_real_corpus):
#   1. Prefijo de cantidad/unidad (igual que v1, + "lonjas/rebanadas/rodajas",
#      que faltaban — day_generator.py:48 las usa).
#   2. UN SOLO conector ("de"/"del"/"la"/"el"/"los"/"las") inmediatamente tras
#      la cantidad/unidad — nombres compuestos como "semillas de linaza" o
#      "Pechuga de pollo" conservan su "de" INTERNO porque solo se quita el
#      PRIMER match.
#   3. Modificador de estado de cocción/preparación al FINAL de la línea
#      ("secas", "cocidas", "picadas", "rallada", "al gusto"...). Los
#      modificadores que SÍ cambian la identidad del alimento ("blanco" en
#      "arroz blanco", "pintas" en "judías pintas", "dominicano" en "Oregano
#      dominicano") NO están en esta lista — se conservan a propósito.
# ============================================================

_QTY_UNIT_PREFIX_RE = re.compile(
    r"^\s*[\d.,/½¼¾\s]*\s*"
    r"(kg|kilos?|g|gr|gramos?|ml|mls?|l|litros?|tazas?|taza|cditas?|cdtas?|cdas?|cucharadas?|"
    r"cucharaditas?|unidad(?:es)?|unids?|piezas?|pza|oz|onzas?|lb|lbs|libras?|"
    r"lonjas?|rebanadas?|rodajas?|lascas?)?\b\s*",
    # [Finding 1 fix, iteración 2] `\b` tras el grupo de unidad: sin esto la
    # alternativa CORTA "l"/"g"/"lb" (que aparece antes en la lista por orden
    # alfabético natural) consume solo el prefijo de una palabra más larga
    # ("l" de "lonjas" -> deja "onjas de queso"), el mismo defecto de
    # alternación-no-longest-match que ya mordió a "unidades?" vs "unid" en
    # la ronda anterior. `\b` fuerza backtracking hacia la alternativa que
    # consume la PALABRA completa.
    re.IGNORECASE,
)
_LEADING_LINKER_RE = re.compile(r"^(?:de|del|la|el|los|las)\s+", re.IGNORECASE)
_PARENTHETICAL_RE = re.compile(r"\([^)]*\)")
_TRAILING_MODIFIER_RE = re.compile(
    r"\s+(?:al gusto"
    r"|en cubos|en trozos|en rodajas|en tiras|en juliana"
    r"|secas?|secos?"
    r"|cocidas?|cocidos?"
    r"|picadas?|picados?"
    r"|ralladas?|rallados?"
    r"|molidas?|molidos?"
    r"|trituradas?|triturados?"
    r"|cortadas?|cortados?"
    r"|deshuesadas?|deshuesados?"
    r"|desmenuzadas?|desmenuzados?"
    r"|peladas?|pelados?"
    r"|lavadas?|lavados?)\s*$",
    re.IGNORECASE,
)


def _extract_canonical_name(ingredient_line: str) -> str:
    if not isinstance(ingredient_line, str):
        return ""
    s = ingredient_line.strip()
    if not s:
        return ""
    s = _QTY_UNIT_PREFIX_RE.sub("", s, count=1).strip()
    s = _LEADING_LINKER_RE.sub("", s, count=1).strip()
    s = _PARENTHETICAL_RE.sub("", s).strip()
    s = _TRAILING_MODIFIER_RE.sub("", s).strip()
    return s


# ============================================================
# Directivas de idioma NATIVAS por locale + prompt (UN lote por llamada)
#
# [Finding 3 · fix round 1] La v1 de Task 1 escribía la directiva ENTERA en
# español y decía el literal "English gloss" a los 4 locales — contradictorio
# para pt-BR/fr-FR/it-IT. Lección ya pagada dos veces el 2026-08-18
# (P1-COACH-LANGUAGE-NATIVE): una directiva escrita en español pidiendo otro
# idioma es la señal más débil posible; se reescribe EN el idioma destino,
# instrucción Y demostración (el ejemplo de gloss) a la vez. Mismo patrón que
# `prompts/chat_agent.py::_TITLE_LANGUAGE_DIRECTIVES`.
#
# El bloque de DATOS originales (nombres/descripciones/recetas del plan) se
# mantiene en español — eso es correcto, es lo que hay que traducir.
# ============================================================

_DISPLAY_LANGUAGE_DIRECTIVES = {
    "en-US": (
        "Translate these Dominican meal-plan dishes into English, for the USER to read. "
        "The system keeps operating in canonical Spanish internally — this is ONLY a "
        "parallel display layer.\n\n"
        "STRICT RULES:\n"
        "1. Write 'name'/'description'/'recipe' EXCLUSIVELY in English. EXCEPTION inside "
        "'ingredients': each line carries the food name in the format \"English gloss "
        "(Nombre canónico en español)\" — example: \"30 g dried red beans (Habichuelas "
        "rojas)\". The Spanish canonical name MUST appear literally, untranslated, exactly "
        "as in the original (it is a system identifier).\n"
        "2. The output arrays 'recipe' and 'ingredients' MUST have EXACTLY the same number "
        "of elements as the original, in the SAME order (aligned by index).\n"
        "3. Reply with ONLY valid JSON, no markdown, no text outside the JSON, with this "
        "exact contract:\n"
        '{"meals":[{"i":0,"name":"...","description":"...","recipe":["...","..."],'
        '"ingredients":["...","..."]}]}'
    ),
    "pt-BR": (
        "Traduza estes pratos de um plano alimentar dominicano para o Português, para "
        "LEITURA do usuário. O sistema continua operando em espanhol canônico internamente "
        "— isto é APENAS uma camada de exibição paralela.\n\n"
        "REGRAS ESTRITAS:\n"
        "1. Escreva 'name'/'description'/'recipe' EXCLUSIVAMENTE em Português. EXCEÇÃO "
        "dentro de 'ingredients': cada linha traz o nome do alimento no formato \"gloss em "
        "português (Nome canônico em espanhol)\" — exemplo: \"30 g feijão vermelho seco "
        "(Habichuelas rojas)\". O nome canônico em espanhol DEVE aparecer literalmente, sem "
        "traduzir, exatamente como no original (é um identificador do sistema).\n"
        "2. Os arrays 'recipe' e 'ingredients' de saída DEVEM ter EXATAMENTE a mesma "
        "quantidade de elementos que o original, na MESMA ordem (alinhados por índice).\n"
        "3. Responda APENAS com JSON válido, sem markdown, sem texto fora do JSON, com este "
        "contrato exato:\n"
        '{"meals":[{"i":0,"name":"...","description":"...","recipe":["...","..."],'
        '"ingredients":["...","..."]}]}'
    ),
    "fr-FR": (
        "Traduis ces plats d'un plan alimentaire dominicain en Français, pour la LECTURE de "
        "l'utilisateur. Le système continue de fonctionner en espagnol canonique en interne "
        "— ceci est SEULEMENT une couche d'affichage parallèle.\n\n"
        "RÈGLES STRICTES :\n"
        "1. Rédige 'name'/'description'/'recipe' EXCLUSIVEMENT en Français. EXCEPTION à "
        "l'intérieur de 'ingredients' : chaque ligne porte le nom de l'aliment au format "
        "« gloss en français (Nom canonique en espagnol) » — exemple : « 30 g de haricots "
        "rouges secs (Habichuelas rojas) ». Le nom canonique espagnol DOIT apparaître "
        "littéralement, sans traduction, exactement comme dans l'original (c'est un "
        "identifiant du système).\n"
        "2. Les tableaux 'recipe' et 'ingredients' de sortie DOIVENT avoir EXACTEMENT le "
        "même nombre d'éléments que l'original, dans le MÊME ordre (alignés par indice).\n"
        "3. Réponds UNIQUEMENT avec du JSON valide, sans markdown, sans texte hors du JSON, "
        "avec ce contrat exact :\n"
        '{"meals":[{"i":0,"name":"...","description":"...","recipe":["...","..."],'
        '"ingredients":["...","..."]}]}'
    ),
    "it-IT": (
        "Traduci questi piatti di un piano alimentare dominicano in Italiano, per la "
        "LETTURA dell'utente. Il sistema continua a operare in spagnolo canonico "
        "internamente — questo è SOLO uno strato di visualizzazione parallelo.\n\n"
        "REGOLE RIGIDE:\n"
        "1. Scrivi 'name'/'description'/'recipe' ESCLUSIVAMENTE in Italiano. ECCEZIONE "
        "dentro 'ingredients': ogni riga porta il nome dell'alimento nel formato \"gloss in "
        "italiano (Nome canonico in spagnolo)\" — esempio: \"30 g di fagioli rossi secchi "
        "(Habichuelas rojas)\". Il nome canonico spagnolo DEVE apparire letteralmente, "
        "senza tradurlo, esattamente come nell'originale (è un identificatore del "
        "sistema).\n"
        "2. Gli array 'recipe' e 'ingredients' in uscita DEVONO avere ESATTAMENTE lo stesso "
        "numero di elementi dell'originale, nello STESSO ordine (allineati per indice).\n"
        "3. Rispondi SOLO con JSON valido, senza markdown, senza testo fuori dal JSON, con "
        "questo contratto esatto:\n"
        '{"meals":[{"i":0,"name":"...","description":"...","recipe":["...","..."],'
        '"ingredients":["...","..."]}]}'
    ),
}

_DISPLAY_DATA_HEADER = {
    "en-US": "ORIGINAL DISHES (Spanish — do not translate food names outside the "
    "ingredients gloss):",
    "pt-BR": "PRATOS ORIGINAIS (espanhol — não traduza os nomes dos alimentos fora do "
    "gloss de ingredients):",
    "fr-FR": "PLATS ORIGINAUX (espagnol — ne traduis pas les noms des aliments en dehors "
    "du gloss d'ingredients) :",
    "it-IT": "PIATTI ORIGINALI (spagnolo — non tradurre i nomi degli alimenti al di fuori "
    "del gloss di ingredients):",
}

_JSON_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _build_prompt(targets: list, locale: str) -> str:
    directive = _DISPLAY_LANGUAGE_DIRECTIVES.get(locale)
    header = _DISPLAY_DATA_HEADER.get(locale)
    if directive is None or header is None:
        # `enrich_plan_display` ya descarta locales fuera de `_COACH_LANGUAGE_NAMES`
        # antes de llegar aquí — este fallback es solo para que `_build_prompt` no
        # reviente si algún día se llama aislado con un locale sin directiva nativa.
        idioma = _COACH_LANGUAGE_NAMES.get(locale, locale)
        directive = (
            f"Translate into {idioma}. Reply with ONLY valid JSON matching the meals "
            f"contract: "
            '{"meals":[{"i":0,"name":"...","description":"...","recipe":["..."],'
            '"ingredients":["..."]}]}'
        )
        header = "ORIGINAL DISHES (Spanish):"

    lines = []
    for i, t in enumerate(targets):
        lines.append(f"{i}. NAME: {t['name']}")
        lines.append(f"   DESCRIPTION: {t['description']}")
        lines.append("   RECIPE:")
        for step_idx, step in enumerate(t["recipe"]):
            lines.append(f"     [{step_idx}] {step}")
        lines.append("   INGREDIENTS:")
        for ing_idx, ing in enumerate(t["ingredients"]):
            lines.append(f"     [{ing_idx}] {ing}")
    meals_block = "\n".join(lines)

    return f"{directive}\n\n{header}\n{meals_block}"


def _parse_json_response(raw: str) -> Optional[dict]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    cleaned = _JSON_CODE_FENCE_RE.sub("", raw).strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


# ============================================================
# Recolección de targets (SIN row-lock — ver aviso TOCTOU en `_persist_batch`)
# ============================================================


def _collect_targets(days: list, day_indices_batch: list) -> list:
    idx_set = set(day_indices_batch)
    targets = []
    for day_idx, day in enumerate(days):
        if day_idx not in idx_set:
            continue
        if not isinstance(day, dict):
            continue
        meals = day.get("meals")
        if not isinstance(meals, list):
            continue
        for meal_idx, meal in enumerate(meals):
            if not isinstance(meal, dict):
                continue
            recipe = meal.get("recipe")
            ingredients = meal.get("ingredients")
            targets.append(
                {
                    "day_idx": day_idx,
                    "meal_idx": meal_idx,
                    "name": meal.get("name") or "",
                    "description": meal.get("description") or "",
                    "recipe": recipe if isinstance(recipe, list) else [],
                    "ingredients": ingredients if isinstance(ingredients, list) else [],
                }
            )
    return targets


# ============================================================
# Validación por meal + construcción del `_display` final
# ============================================================


def _validate_and_build_display(original: dict, item: dict) -> Optional[dict]:
    name = item.get("name")
    description = item.get("description")
    recipe = item.get("recipe")
    ingredients = item.get("ingredients")

    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(description, str) or not description.strip():
        return None
    if not isinstance(recipe, list) or len(recipe) != len(original["recipe"]):
        return None
    if not isinstance(ingredients, list) or len(ingredients) != len(original["ingredients"]):
        return None

    final_ingredients = []
    for idx, translated_line in enumerate(ingredients):
        translated_line = translated_line if isinstance(translated_line, str) else ""
        original_line = original["ingredients"][idx]
        original_line = original_line if isinstance(original_line, str) else str(original_line)
        canonical = _extract_canonical_name(original_line)
        if not canonical:
            # Sin canónico identificable en el original: la línea pasa sin check
            # (spec: "si el original no tiene canónico identificable, la línea
            # pasa sin check").
            final_ingredients.append(translated_line.strip() or original_line)
            continue
        if strip_accents(canonical).lower() in strip_accents(translated_line).lower():
            final_ingredients.append(translated_line)
        else:
            # Un gloss que pierde el identificador es peor que no tener gloss:
            # se descarta ESA línea (no el meal) -> fallback al original español.
            final_ingredients.append(original_line)

    final_recipe = [step if isinstance(step, str) else str(step) for step in recipe]

    return {
        "name": name.strip(),
        "description": description.strip(),
        "recipe": final_recipe,
        "ingredients": final_ingredients,
    }


def _emit_usage_telemetry(plan_id: str, user_id: str, model_name: str, response) -> None:
    """Best-effort. NUNCA toca `api_usage` — solo `llm_usage_events` (libro de
    costo, node="plan_display_i18n"). Cualquier fallo se traga en silencio."""
    try:
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        log_llm_usage_event(
            user_id=user_id,
            plan_id=plan_id,
            model=model_name,
            node="plan_display_i18n",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        logger.debug(f"[P1-PLAN-DISPLAY-I18N] telemetría falló (best-effort): {e!r}")


# ============================================================
# Persistencia de UN lote validado
# ============================================================


def _persist_batch(
    plan_id: str,
    user_id: str,
    locale: str,
    targets: list,
    valid_by_index: dict,
) -> tuple:
    """Persiste un lote ya validado. Retorna `(written_count, mismatch_count)`.

    El `mutator` corre DENTRO del `SELECT … FOR UPDATE` de `update_plan_data_atomic`
    (contrato P2-MUTATOR-PURITY, `db_plans.py:563`): pura, CPU-only, sin IO, sin
    llamadas LLM/DB, sin re-entrada al pool. Acumular en los dicts `counters`
    (closures Python en memoria) NO viola esa pureza.

    [Finding 5 · fix round 1] TOCTOU: `targets` se construyó en `enrich_plan_display`
    ANTES de este lock (leyendo `plan_data` fuera de cualquier lock — sostener un
    `FOR UPDATE` durante los ~segundos que dura la llamada LLM no es viable). Si el
    meal en esa posición CAMBIÓ entre la lectura y este persist (un
    swap/regenerate-day concurrente escribió ahí primero), el `name` canónico ya no
    coincide con el snapshot leído — escribir de todos modos pegaría la traducción
    del plato VIEJO sobre el plato NUEVO, exactamente el "stale display" que la spec
    declara imposible por construcción. Se compara el `name` (identificador estable
    del meal, en español canónico) y se SALTA si difiere — el meal se queda sin
    `_display` para este locale hasta el próximo disparador legítimo.
    """
    counters = {"written": 0, "mismatch": 0}

    def _mutator(pd: dict):
        pd_days = pd.get("days") if isinstance(pd, dict) else None
        if not isinstance(pd_days, list):
            return pd
        for i, display in valid_by_index.items():
            t = targets[i]
            try:
                day = pd_days[t["day_idx"]]
                meal = day.get("meals", [])[t["meal_idx"]]
            except (IndexError, TypeError, KeyError, AttributeError):
                continue
            if not isinstance(meal, dict):
                continue
            if meal.get("name") != t["name"]:
                counters["mismatch"] += 1
                continue
            disp_map = meal.get("_display")
            if not isinstance(disp_map, dict):
                disp_map = {}
            disp_map[locale] = display
            meal["_display"] = disp_map
            counters["written"] += 1
        return pd

    try:
        persisted = update_plan_data_atomic(plan_id, _mutator, user_id=user_id)
    except Exception as e:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] persist falló plan={plan_id} locale={locale}: {e!r}"
        )
        return 0, 0
    if not persisted:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] persist retornó vacío (plan desapareció o ownership "
            f"no matcheó) plan={plan_id} locale={locale}"
        )
        return 0, 0
    return counters["written"], counters["mismatch"]


# ============================================================
# Motor
# ============================================================


def enrich_plan_display(
    plan_id: str,
    user_id: str,
    locale: str,
    day_indices: Optional[list] = None,
) -> dict:
    """Enriquece `_display[locale]` de los meals de `plan_id` (ownership
    `AND user_id`). JAMÁS lanza — fail-open total (ver docstring del módulo).

    Trocea en lotes de `MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS` días (Finding 6):
    cada lote es UNA llamada LLM + UN persist independiente, así que un lote que
    falla (JSON roto, timeout) no tumba a los demás — recuperación parcial.

    Returns:
        {"enriched_meals": int, "skipped": str | None}
    """
    key = (plan_id, locale)
    try:
        if not _plan_display_i18n_enabled():
            return {"enriched_meals": 0, "skipped": "knob_off"}
        if not isinstance(locale, str) or locale not in _COACH_LANGUAGE_NAMES:
            # Cubre es-DO (nunca en el dict) y cualquier locale inválido.
            return {"enriched_meals": 0, "skipped": "locale"}

        plan_data = _fetch_plan_data(plan_id, user_id)
        if plan_data is None:
            return {"enriched_meals": 0, "skipped": "not_found"}

        days = plan_data.get("days")
        if not isinstance(days, list) or not days:
            return {"enriched_meals": 0, "skipped": "no_days"}

        requested_day_indices = _normalize_day_indices(day_indices, len(days))
        if not requested_day_indices:
            return {"enriched_meals": 0, "skipped": "no_meals"}

        with _INFLIGHT_LOCK:
            if key in _INFLIGHT:
                return {"enriched_meals": 0, "skipped": "dedupe_inprocess"}
            _INFLIGHT.add(key)

        cross_worker_claimed = False
        try:
            if not _try_claim_enrich_lock_cross_worker(plan_id, locale, requested_day_indices):
                return {"enriched_meals": 0, "skipped": "dedupe_locked"}
            cross_worker_claimed = True

            model_name = _plan_display_i18n_model_name()
            if not _circuit_breaker_can_proceed(model_name):
                logger.info(
                    f"[P1-PLAN-DISPLAY-I18N] circuit breaker abierto model={model_name!r} "
                    f"plan={plan_id} locale={locale} — skip silente."
                )
                return {"enriched_meals": 0, "skipped": "circuit_breaker_open"}

            batch_size = _plan_display_i18n_batch_days()
            max_tokens = _plan_display_i18n_max_output_tokens()
            timeout_s = _plan_display_i18n_timeout_s()
            day_batches = [
                requested_day_indices[i : i + batch_size]
                for i in range(0, len(requested_day_indices), batch_size)
            ]

            total_written = 0
            mismatch_total = 0
            last_skip_reason = "no_meals"

            for batch_day_indices in day_batches:
                targets = _collect_targets(days, batch_day_indices)
                if not targets:
                    continue

                prompt = _build_prompt(targets, locale)
                try:
                    llm = build_chat_llm(
                        model_name,
                        temperature=0.2,
                        timeout=timeout_s,
                        max_output_tokens=max_tokens,
                    )
                    response = llm.invoke([SystemMessage(content=prompt)])
                except Exception as e:
                    logger.warning(
                        f"[P1-PLAN-DISPLAY-I18N] LLM invoke falló plan={plan_id} "
                        f"locale={locale} days={batch_day_indices}: {e!r}"
                    )
                    last_skip_reason = "llm_exception"
                    continue

                # [Finding 7 · fix round 1] Telemetría INMEDIATAMENTE tras el invoke
                # exitoso — el gasto ya ocurrió aquí, sin importar que el parseo,
                # la validación o el persist de abajo fallen después.
                _emit_usage_telemetry(plan_id, user_id, model_name, response)

                raw_content = getattr(response, "content", "") or ""
                parsed = _parse_json_response(raw_content)
                if parsed is None or not isinstance(parsed.get("meals"), list):
                    last_skip_reason = "json_parse_error"
                    continue

                valid_by_index = {}
                for item in parsed["meals"]:
                    if not isinstance(item, dict):
                        continue
                    i = item.get("i")
                    if not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(targets):
                        continue
                    display = _validate_and_build_display(targets[i], item)
                    if display is not None:
                        valid_by_index[i] = display

                if not valid_by_index:
                    last_skip_reason = "no_valid_meals"
                    continue

                written, mismatches = _persist_batch(
                    plan_id, user_id, locale, targets, valid_by_index
                )
                total_written += written
                mismatch_total += mismatches
                if written:
                    last_skip_reason = None

            if mismatch_total:
                logger.warning(
                    f"[P1-PLAN-DISPLAY-I18N] {mismatch_total} meal(s) omitidos por TOCTOU "
                    f"(cambiaron entre lectura y persist) plan={plan_id} locale={locale}"
                )

            if total_written > 0:
                logger.info(
                    f"[P1-PLAN-DISPLAY-I18N] enriquecidos {total_written} meal(s) en "
                    f"{len(day_batches)} lote(s) plan={plan_id} locale={locale}"
                )
                return {"enriched_meals": total_written, "skipped": None}

            if mismatch_total:
                last_skip_reason = "persist_stale_mismatch"
            return {"enriched_meals": 0, "skipped": last_skip_reason}
        finally:
            with _INFLIGHT_LOCK:
                _INFLIGHT.discard(key)
            if cross_worker_claimed:
                _release_enrich_lock_cross_worker(plan_id, locale, requested_day_indices)
    except Exception as e:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] enrich_plan_display excepción no controlada "
            f"(fail-open) plan={plan_id} locale={locale}: {e!r}"
        )
        return {"enriched_meals": 0, "skipped": "exception"}


def schedule_plan_display_enrichment(
    plan_id: str,
    user_id: str,
    locale: str,
    day_indices: Optional[list] = None,
) -> None:
    """Wrapper fire-and-forget: dispara `enrich_plan_display` en un thread
    background. Nunca lanza — cualquier fallo (incluso al lanzar el thread)
    queda en warning. Los 5 disparadores de la spec llaman a este helper,
    nunca al motor directo, así el caller no bloquea su propio request.
    """
    try:
        if not _plan_display_i18n_enabled():
            return
        if not isinstance(locale, str) or locale not in _COACH_LANGUAGE_NAMES:
            return

        # Pre-check barato in-process: evita levantar un thread si ya hay uno
        # en vuelo para el mismo (plan_id, locale). No sustituye el dedupe
        # real (cross-worker + re-check) que vive dentro de `enrich_plan_display`.
        with _INFLIGHT_LOCK:
            if (plan_id, locale) in _INFLIGHT:
                logger.debug(
                    f"[P1-PLAN-DISPLAY-I18N] schedule skip — ya en vuelo "
                    f"plan={plan_id} locale={locale}"
                )
                return

        def _run():
            try:
                result = enrich_plan_display(
                    plan_id, user_id, locale, day_indices=day_indices
                )
                logger.info(
                    f"[P1-PLAN-DISPLAY-I18N] background enrich plan={plan_id} "
                    f"locale={locale} result={result}"
                )
            except Exception as e:
                # enrich_plan_display ya es fail-open y no debería lanzar, pero
                # el thread background no debe morir ruidoso bajo ningún caso.
                logger.warning(
                    f"[P1-PLAN-DISPLAY-I18N] background enrich excepción "
                    f"(fail-open) plan={plan_id} locale={locale}: {e!r}"
                )

        threading.Thread(target=_run, daemon=True).start()
    except Exception as e:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] schedule_plan_display_enrichment falló "
            f"(fail-open) plan={plan_id} locale={locale}: {e!r}"
        )
