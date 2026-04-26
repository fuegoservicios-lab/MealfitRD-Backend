import sys
import os
import logging
from unittest.mock import MagicMock

# Mocking external dependencies that might not be in the environment
sys.modules["langchain_google_genai"] = MagicMock()
sys.modules["langchain_core"] = MagicMock()
sys.modules["langchain_core.embeddings"] = MagicMock()

# Añadir el path del backend para poder importar constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import validate_ingredients_against_pantry

# Configurar logging para ver los mensajes de error
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_p1_1_validation():
    # Pantry: solo 100g de pollo disponibles
    pantry = ["100g pollo"]
    
    # Receta solicita 110g de pollo
    # Con tolerancia 1.0 (strict) -> debe FALLAR
    # Con tolerancia 1.2 (hybrid) -> debe PASAR
    
    recipe_110 = ["110g pollo"]
    
    print("\n--- Test 1: 110g vs 100g (Tolerance 1.0 - Strict) ---")
    res_strict = validate_ingredients_against_pantry(recipe_110, pantry, strict_quantities=True, tolerance=1.0)
    if isinstance(res_strict, str):
        print(f"✅ OK: Falló como se esperaba en modo estricto.\n{res_strict[:100]}...")
    else:
        print("❌ ERROR: Debió fallar en modo estricto.")

    print("\n--- Test 2: 110g vs 100g (Tolerance 1.2 - Hybrid) ---")
    res_hybrid = validate_ingredients_against_pantry(recipe_110, pantry, strict_quantities=True, tolerance=1.2)
    if res_hybrid is True:
        print("✅ OK: Pasó correctamente en modo híbrido (dentro del 20%).")
    else:
        print(f"❌ ERROR: Debió pasar en modo híbrido.\n{res_hybrid}")

    # Receta solicita 250g de pollo
    # Con tolerancia 1.2 (hybrid) -> debe FALLAR (excede el 20%)
    recipe_250 = ["250g pollo"]
    
    print("\n--- Test 3: 250g vs 100g (Tolerance 1.2 - Hybrid) ---")
    res_hybrid_250 = validate_ingredients_against_pantry(recipe_250, pantry, strict_quantities=True, tolerance=1.2)
    if isinstance(res_hybrid_250, str):
        print(f"✅ OK: Falló como se esperaba (250g excede 120g).\n{res_hybrid_250[:100]}...")
    else:
        print("❌ ERROR: Debió fallar incluso en modo híbrido.")

if __name__ == "__main__":
    test_p1_1_validation()
