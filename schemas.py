from pydantic import BaseModel, Field
from typing import List

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
    cals: int = Field(description="Calorías aproximadas de este plato")
    macros: List[str] = Field(description="Lista rápida de macros, Ej:['Alto en proteína', 'Bajo en carbohidratos']")
    ingredients: List[str] = Field(description="Lista de ingredientes con cantidades (texto simple), Ej:['1 plátano verde maduro', '2 huevos', '1/2 aguacate']")
    recipe: List[str] = Field(description="Pasos de preparación. DEBES usar los prefijos: 'Mise en place: ...', 'El Toque de Fuego: ...' y 'Montaje: ...'")

class DailyPlanModel(BaseModel):
    day: int = Field(description="Número de día (1, 2, o 3)")
    meals: List[MealModel] = Field(description="Lista de comidas en orden cronológico. MUY IMPORTANTE: Si el usuario omite el almuerzo, genera SOLO 3 comidas: Desayuno, Merienda, Cena.")

class PlanModel(BaseModel):
    main_goal: str = Field(description="El objetivo principal identificado. Ej: 'Pérdida de Peso (Déficit)'")
    calories: int = Field(description="Total de calorías estrictas planificadas sumando todas las comidas")
    macros: MacrosModel = Field(description="Distribución matemática de macronutrientes para el día")
    insights: List[str] = Field(description="Lista EXACTA de 3 frases: 1. Inicia con 'Diagnóstico: ', 2. Inicia con 'Estrategia: ', 3. Inicia con 'Tip del Chef: '")
    days: List[DailyPlanModel] = Field(description="Lista de 3 días con planes de comida variados")

class ShoppingItemModel(BaseModel):
    category: str = Field(description="Nombre de la categoría principal, Ej: 'Carnes', 'Frutas y Verduras'")
    emoji: str = Field(description="Un emoji representativo de la categoría, Ej: '🥩', '🥬'")
    name: str = Field(description="Nombre del ingrediente o producto, Ej: 'Pechuga de pollo'")
    qty: str = Field(description="Cantidad redondeada a unidades reales de supermercado, Ej: '2 Unidades', '1 Libra', '250g'")

class ShoppingListModel(BaseModel):
    items: List[ShoppingItemModel] = Field(description="Lista de ingredientes parseados y estructurados individualmente.")
