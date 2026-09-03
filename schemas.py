from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal

class MacrosModel(BaseModel):
    protein: str = Field(description="Gramos de proteína totales, ej: '150g'")
    carbs: str = Field(description="Gramos de carbohidratos totales, ej: '200g'")
    fats: str = Field(description="Gramos de grasas totales, ej: '60g'")

class MealModel(BaseModel):
    meal: str = Field(description="Momento del día, Ej: 'Desayuno', 'Almuerzo', 'Merienda', 'Cena'")
    # [Z3-SCHEMA-DROP-TIME · 2026-05-28] Optional para que el LLM deje de emitirlo
    # (campo no-renderizado: solo passthrough backend). Backfill 'Flexible' garantiza
    # un valor downstream (ver graph_orchestrator backfill con `not m.get("time")`).
    time: Optional[str] = Field(default=None, description="Hora sugerida (opcional, se rellena a 'Flexible' si falta)")
    name: str = Field(description="Nombre creativo y descriptivo del plato")
    desc: str = Field(description="Descripción apetitosa y profesional de la receta")
    prep_time: str = Field(description="Tiempo estimado de preparación, Ej: '15 min'")
    difficulty: str = Field(default="Fácil", description="Nivel de dificultad, Ej: 'Fácil', 'Intermedio', 'Difícil'")
    cals: int = Field(description="Calorías aproximadas de este plato")
    protein: int = Field(default=0, description="Gramos de proteína estimados en esta porción, Ej: 30")
    carbs: int = Field(default=0, description="Gramos de carbohidratos estimados en esta porción, Ej: 45")
    fats: int = Field(default=0, description="Gramos de grasas estimados en esta porción, Ej: 15")
    # [Z2-SCHEMA-DROP-MACROS · 2026-05-28] Optional: array de tags por-meal sin
    # consumidor (el frontend lee macros a nivel-plan; un backfill lo sobreescribe).
    # Output muerto pagado a $9/M. Optional → el LLM deja de emitirlo de forma fiable.
    macros: Optional[List[str]] = Field(default=None, description="Lista rápida de macros (opcional, tags informativos)")
    ingredients: List[str] = Field(description="Lista de ingredientes consolidados sin clonar y con unidades comerciales exactas (texto simple), Ej:['1 plátano verde maduro', '2 huevos', '1/2 aguacate']")
    recipe: List[str] = Field(description="Pasos de preparación. DEBES usar los prefijos: 'Mise en place: ...', 'El Toque de Fuego: ...' y 'Montaje: ...'")

class SupplementModel(BaseModel):
    name: str = Field(description="Nombre del suplemento, Ej: 'Creatina Monohidrato'")
    dose: str = Field(description="Dosis recomendada, Ej: '5g (1 cucharadita)'")
    timing: str = Field(description="Momento del día para tomarlo, Ej: 'Post-entreno', 'Con el desayuno'")
    reason: str = Field(description="Justificación breve de por qué se recomienda para el usuario")

class DailyPlanModel(BaseModel):
    day: int = Field(description="Identificador del día o alternativa (1 al 3)")
    day_name: Optional[str] = Field(default=None, description="Nombre del día de la semana (ej: Lunes, Martes)")
    meals: List[MealModel] = Field(description="Lista de comidas de esta alternativa en orden cronológico. MUY IMPORTANTE: Si el usuario omite el almuerzo, genera SOLO 3 comidas: Desayuno, Merienda, Cena.")
    supplements: Optional[List[SupplementModel]] = Field(default=None, description="Lista de suplementos para esta alternativa. Solo se incluye si el usuario activó includeSupplements: true.")

class PlanModel(BaseModel):
    main_goal: str = Field(description="El objetivo principal identificado. Ej: 'Pérdida de Peso (Déficit)'")
    calories: int = Field(description="Total de calorías estrictas planificadas sumando todas las comidas")
    macros: MacrosModel = Field(description="Distribución matemática de macronutrientes para el día")
    insights: List[str] = Field(description="Lista EXACTA de 3 frases: 1. Inicia con 'Diagnóstico: ', 2. Inicia con 'Estrategia: ', 3. Inicia con 'Tip del Chef: '")
    days: List[DailyPlanModel] = Field(description="Lista de 3 días o alternativas continuas con sus respectivas comidas")

class ExpandedRecipeModel(BaseModel):
    recipe: List[str] = Field(description="Lista de EXACTAMENTE 3 pasos: Mise en place, El Toque de Fuego y Montaje, magistralmente detallados.")


# ============================================================
# SCHEMAS PARA PIPELINE MAP-REDUCE (Paralelización por Día)
# ============================================================

class DaySkeletonModel(BaseModel):
    """Asignación liviana de un solo día producida por el nodo Planificador."""
    day: int = Field(description="Número de alternativa (1 al 3)")
    assigned_technique: str = Field(description="Técnica de cocción principal asignada a la comida principal de este día, Ej: 'Guisado', 'Al Horno', 'Salteado'")
    protein_pool: List[str] = Field(description="Proteínas base asignadas a este día, Ej: ['Pechuga de pollo', 'Huevos']")
    carb_pool: List[str] = Field(description="Carbohidratos base asignados a este día, Ej: ['Arroz integral', 'Batata']")
    fruit_pool: List[str] = Field(description="Frutas asignadas a este día, Ej: ['Guineo', 'Manzana']")
    # [P2-VEGGIE-CHANNEL-DAYGEN · 2026-07-30] (audit solver+seeder v5) El seeder sorteaba
    # vegetales/grasas por día (pool filtrado por alergias + pareo con dedupe) y los interpolaba
    # SOLO en el prompt del ESQUELETO. El day-generator —quien escribe los ingredientes reales—
    # no tenía forma de recibir esa decisión: este modelo no declaraba el campo, así que ni
    # siquiera podía transportarla. Toda la maquinaria de vegetales del seeder producía una
    # decisión sin consumidor estructural, y el day-gen caía en su default (la misma ensalada
    # verde/aguacate los 3 días). Misma clase que P1-CARB-BASE-NO-REPEAT, que cerró esto SOLO
    # para carbos. Default [] → los planners que no lo emitan siguen validando.
    veggie_pool: List[str] = Field(default_factory=list,
                                   description="Vegetales/grasas asignados a este día, Ej: ['Brócoli', 'Aguacate']")
    meal_types: List[str] = Field(description="Tipos de comidas a generar en orden, Ej: ['Desayuno', 'Almuerzo', 'Merienda', 'Cena']")
    breakfast_category: str = Field(default="Libre", description="Categoría base del desayuno asignada a este día. DEBE ser diferente para cada día. Valores: 'Mangú/Tubérculos', 'Avena/Cereales', 'Pan/Tostadas', 'Batido/Bowl', 'Revoltillo/Tortilla'")
    brief_concept: str = Field(description="Concepto temático breve de este día, Ej: 'Día Caribeño con enfoque en proteína magra y tubérculos'")

class PlanSkeletonModel(BaseModel):
    """Esqueleto liviano del plan producido por el nodo Planificador (fase map)."""
    main_goal: str = Field(description="El objetivo principal identificado. Ej: 'Pérdida de Peso (Déficit)'")
    insights: List[str] = Field(description="Lista EXACTA de 3 frases: 1. Inicia con 'Diagnóstico: ', 2. Inicia con 'Estrategia: ', 3. Inicia con 'Tip del Chef: '")
    days: List[DaySkeletonModel] = Field(description="Lista de 3 asignaciones, una por cada día u opción alternativa")

class SingleDayPlanModel(BaseModel):
    """Plan detallado de un solo día, producido por cada worker paralelo."""
    day: int = Field(description="Identificador del día (e.g. 1 para Día 1)")
    day_name: Optional[str] = Field(default=None, description="Nombre del día de la semana (ej: Lunes, Martes)")
    meals: List[MealModel] = Field(description="Lista de comidas completas con ingredientes y recetas")
    supplements: Optional[List[SupplementModel]] = Field(default=None, description="Suplementos si aplica")

from typing import Dict, Any


# ============================================================
# [P1-11] Schema explícito de eventos SSE para `/api/plans/analyze/stream`.
# ------------------------------------------------------------
# Antes el endpoint reenviaba CUALQUIER evento que llegara del
# `progress_callback` al cliente — incluyendo eventos internos como `metric`,
# `token`, `tool_call`, `token_reset` que el frontend silenciosamente ignora
# (~50 eventos extras × ~200 bytes = ~10 KB de bandwidth desperdiciado por
# request, más ruido en logs de DevTools del cliente).
#
# Además había un bug latente: el orquestador emite `day_completed` pero el
# frontend (`Plan.jsx`) escucha `day_complete` — el progreso por día nunca
# se actualizaba vía SSE (solo vía polling de /chunk-status cada 5s).
#
# Esta lista declarativa centraliza el contrato público SSE; cualquier evento
# fuera del set se filtra antes de yieldearse al cliente. El alias
# `day_completed` → `day_complete` se aplica en el filtro (ver
# `routers/plans.py:event_generator`) para fixear el bug sin tocar las ~50
# llamadas a `_emit_progress("day_completed", ...)` en el orquestador.
# ============================================================
SseEventName = Literal[
    "phase",         # cambio de fase del pipeline (skeleton, parallel_generation, ...)
    "day_started",   # un worker paralelo inició la generación de un día
    "day_complete",  # un día terminó (alias canónico; backend emite "day_completed")
    "complete",      # plan final listo (contiene el plan JSON completo)
    "error",         # error durante la generación
    "heartbeat",     # keep-alive emitido cada ~5s por el SSE handler
]

# Set para lookup O(1) en el filtro. Incluye también `day_completed` como
# entrada legítima para que el filtro no la tire antes de renombrarla; el
# rename a `day_complete` ocurre en el handler SSE.
PUBLIC_SSE_EVENTS = frozenset({
    "phase", "day_started", "day_complete", "day_completed",
    "complete", "error", "heartbeat",
})


class HealthProfileSchema(BaseModel):
    meal_adherence_weekday: Dict[str, float] = Field(default_factory=dict)
    meal_adherence_weekend: Dict[str, float] = Field(default_factory=dict)
    quality_history: List[Any] = Field(default_factory=list)
    emergency_backup_plan: List[Any] = Field(default_factory=list)
    mainGoal: Optional[str] = None
    activityLevel: Optional[str] = None
    dietTypes: List[str] = Field(default_factory=list)
    country: Optional[str] = None
    inventoryMode: Optional[str] = 'indulgent'
    # [P3-TZ-FALLBACK-SSOT · 2026-08-22] `tzOffset` pasa de `0` a `None`, y el campo `timezone`
    # —que traía una zona IANA dominicana clavada como default— se BORRA. (El literal no se
    # reproduce en este comentario a propósito: el guard lo busca, y citarlo aquí pondría en rojo
    # al propio arreglo — comentario-vence-guard, once veces ya en esta ola.)
    #
    # Medido: ese campo tenía CERO lectores en todo el backend y CERO filas en la base. No era un
    # default, era una invitación: quien escribiera `profile.timezone` creyendo leer el huso del
    # usuario obtendría una zona caribeña para un noruego, y el código parecería correcto en RD,
    # que es donde se prueba. Borrarlo es más seguro que corregirlo — con `extra = 'allow'`, si
    # algún día una fila trae la clave, el validador sigue conservándola.
    #
    # `tzOffset` sí existe en datos reales (5 filas), así que se queda declarada; lo que se le
    # quita es el default que fabricaba un huso plausible. `0` se confunde con UTC; `None` no se
    # confunde con nada. La resolución de verdad vive en `_resolve_request_tz_offset` y
    # `user_tz_offset_min`, sobre `constants.DEFAULT_TZ_OFFSET_MIN`.
    tzOffset: Optional[int] = None
    householdSize: Optional[int] = 1
    groceryDuration: Optional[str] = 'weekly'
    totalDays: Optional[int] = 3
    # [2026-08-23] `class Config` esta deprecado desde Pydantic V2 y emite
    # PydanticDeprecatedSince20 en cada arranque y en cada corrida del gate.
    # `ConfigDict` es su reemplazo exacto; se retira en V3.
    model_config = ConfigDict(extra='allow')
