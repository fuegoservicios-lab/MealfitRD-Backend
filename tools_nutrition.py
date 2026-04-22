import json
from langchain_core.tools import tool

# Base de datos simulada (Mock) de macros de ingredientes comunes (por 100g)
MOCK_NUTRITION_DB = {
    "pollo": {"calories": 165, "protein": 31, "carbs": 0, "fats": 3.6},
    "pechuga": {"calories": 165, "protein": 31, "carbs": 0, "fats": 3.6},
    "arroz": {"calories": 130, "protein": 2.7, "carbs": 28, "fats": 0.3}, # Arroz blanco cocido
    "platano": {"calories": 122, "protein": 1.3, "carbs": 32, "fats": 0.4}, # Plátano verde
    "huevo": {"calories": 155, "protein": 13, "carbs": 1.1, "fats": 11},
    "aguacate": {"calories": 160, "protein": 2, "carbs": 8.5, "fats": 15},
    "avena": {"calories": 389, "protein": 16.9, "carbs": 66.3, "fats": 6.9},
    "batata": {"calories": 86, "protein": 1.6, "carbs": 20, "fats": 0.1},
    "queso": {"calories": 402, "protein": 25, "carbs": 1.3, "fats": 33}, # Queso cheddar genérico
    "pescado": {"calories": 105, "protein": 20, "carbs": 0, "fats": 2.7}, # Tilapia/Pescado blanco
    "salmon": {"calories": 208, "protein": 20, "carbs": 0, "fats": 13},
    "res": {"calories": 250, "protein": 26, "carbs": 0, "fats": 15}, # Carne de res molida magra
    "habichuela": {"calories": 127, "protein": 8.7, "carbs": 22.8, "fats": 0.5}, # Habichuela roja cocida
    "lechosa": {"calories": 43, "protein": 0.5, "carbs": 11, "fats": 0.3},
    "leche": {"calories": 42, "protein": 3.4, "carbs": 5, "fats": 1}, # Leche baja en grasa
}

@tool
def consultar_nutricion(ingrediente: str, gramos: float) -> str:
    """
    Consulta la base de datos interna de nutrición para obtener las calorías y macronutrientes exactos.
    
    Args:
        ingrediente: El nombre del ingrediente a consultar (ej. "pollo", "arroz"). Usa palabras clave simples.
        gramos: La cantidad en gramos (ej. 150).
        
    Returns:
        Un string con las calorías y macros calculados para esa porción, o un mensaje de error si no se encuentra.
    """
    # Limpiar búsqueda
    query = ingrediente.lower().strip()
    
    # Buscar coincidencia (muy simple)
    match = None
    for key, data in MOCK_NUTRITION_DB.items():
        if key in query or query in key:
            match = data
            break
            
    if not match:
        return f"No se encontró información para '{ingrediente}' en la base de datos local. Usa estimaciones estándar."
        
    # Calcular macros para la cantidad solicitada (regla de 3)
    multiplier = gramos / 100.0
    
    cals = round(match["calories"] * multiplier)
    pro = round(match["protein"] * multiplier, 1)
    carbs = round(match["carbs"] * multiplier, 1)
    fats = round(match["fats"] * multiplier, 1)
    
    result = {
        "ingredient": ingrediente,
        "portion_grams": gramos,
        "calories": cals,
        "protein_g": pro,
        "carbs_g": carbs,
        "fats_g": fats
    }
    
    return json.dumps(result)

# Lista de herramientas para exportar y "bindear" al LLM
NUTRITION_TOOLS = [consultar_nutricion]
