from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class MacrosModel(BaseModel):
    protein: str = Field(description="Gramos de proteína totales, ej: '150g'")
    carbs: str = Field(description="Gramos de carbohidratos totales, ej: '200g'")
    fats: str = Field(description="Gramos de grasas totales, ej: '60g'")

class MealModel(BaseModel):
    meal: str = Field(description="Momento del día, Ej: 'Desayuno', 'Almuerzo', 'Merienda', 'Cena'")
    time: str = Field(description="Hora sugerida, Ej: '8:00 AM'")
    name: str = Field(description="Nombre creativo y descriptivo del plato")
    desc: str = Field(description="Descripción apetitosa y profesional de la receta")
    prep_time: str = Field(description="Tiempo estimado de preparación, Ej: '15 min'")
    difficulty: str = Field(default="Fácil", description="Nivel de dificultad, Ej: 'Fácil', 'Intermedio', 'Difícil'")
    cals: int = Field(description="Calorías aproximadas de este plato")
    protein: int = Field(default=0, description="Gramos de proteína estimados en esta porción, Ej: 30")
    carbs: int = Field(default=0, description="Gramos de carbohidratos estimados en esta porción, Ej: 45")
    fats: int = Field(default=0, description="Gramos de grasas estimados en esta porción, Ej: 15")
    macros: List[str] = Field(description="Lista rápida de macros, Ej:['Alto en proteína', 'Bajo en carbohidratos']")
    ingredients: List[str] = Field(description="Lista de ingredientes consolidados sin clonar y con unidades comerciales exactas (texto simple), Ej:['1 plátano verde maduro', '2 huevos', '1/2 aguacate']")
    recipe: List[str] = Field(description="Pasos de preparación. DEBES usar los prefijos: 'Mise en place: ...', 'El Toque de Fuego: ...' y 'Montaje: ...'")

class SupplementModel(BaseModel):
    name: str = Field(description="Nombre del suplemento, Ej: 'Creatina Monohidrato'")
    dose: str = Field(description="Dosis recomendada, Ej: '5g (1 cucharadita)'")
    timing: str = Field(description="Momento del día para tomarlo, Ej: 'Post-entreno', 'Con el desayuno'")
    reason: str = Field(description="Justificación breve de por qué se recomienda para el usuario")

class DailyPlanModel(BaseModel):
    day: int = Field(description="Identificador de la Opción/Alternativa (ej. 1 para Opción A, 2 para Opción B, 3 para Opción C)")
    meals: List[MealModel] = Field(description="Lista de comidas de esta alternativa en orden cronológico. MUY IMPORTANTE: Si el usuario omite el almuerzo, genera SOLO 3 comidas: Desayuno, Merienda, Cena.")
    supplements: Optional[List[SupplementModel]] = Field(default=None, description="Lista de suplementos para esta alternativa. Solo se incluye si el usuario activó includeSupplements: true.")

class PlanModel(BaseModel):
    main_goal: str = Field(description="El objetivo principal identificado. Ej: 'Pérdida de Peso (Déficit)'")
    calories: int = Field(description="Total de calorías estrictas planificadas sumando todas las comidas")
    macros: MacrosModel = Field(description="Distribución matemática de macronutrientes para el día")
    insights: List[str] = Field(description="Lista EXACTA de 3 frases: 1. Inicia con 'Diagnóstico: ', 2. Inicia con 'Estrategia: ', 3. Inicia con 'Tip del Chef: '")
    days: List[DailyPlanModel] = Field(description="Lista de 3 OPCIONES DIARIAS INTERCAMBIABLES (plantillas de un mismo día) con comidas variadas")

class ExpandedRecipeModel(BaseModel):
    recipe: List[str] = Field(description="Lista de EXACTAMENTE 3 pasos: Mise en place, El Toque de Fuego y Montaje, magistralmente detallados.")
