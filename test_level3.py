"""
Test suite para Nivel 3: Motor Data-Driven, Mínimos Comprables, Categorías Server-Side
Ejecutable sin dependencias externas (mock de imports)
"""
import sys
import types

# Mock de imports para ejecutar sin Supabase/LangChain
mock_db = types.ModuleType('db_core')
mock_db.supabase = None
sys.modules['db_core'] = mock_db

# Mock de constants
mock_const = types.ModuleType('constants')
mock_const.UNIT_WEIGHTS = {
    'huevo': 60, 'guineo': 120, 'platano': 250, 'limon': 50,
    'tomate': 150, 'cebolla': 180, 'ajo': 5, 'naranja': 200,
    'manzana': 180, 'aguacate': 250, 'papa': 150, 'zanahoria': 75
}
mock_const.normalize_ingredient_for_tracking = lambda x: x.lower()
mock_const.strip_accents = lambda x: x
mock_const.TECHNIQUE_FAMILIES = {}
mock_const.ALL_TECHNIQUES = []
mock_const.TECH_TO_FAMILY = {}
mock_const.SUPPLEMENT_NAMES = {}
sys.modules['constants'] = mock_const

# Mock langchain
for mod_name in ['langchain_google_genai', 'langgraph', 'langgraph.graph', 'tenacity']:
    sys.modules[mod_name] = types.ModuleType(mod_name)

sys.path.insert(0, '.')

from shopping_calculator import (
    apply_smart_market_units, 
    get_plural_unit, 
    MARKET_MINIMUMS, 
    DISPLAY_CATEGORY_MAP,
    _get_display_category
)

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}")
        failed += 1

# ============================================
# TEST 1: Market Minimums Enforcement
# ============================================
print("\n🔒 TEST 1: Mínimos Comprables")

result = apply_smart_market_units("Pechuga de pollo", 0.1, "lb", 0.0, {"category": "Proteínas"})
test("Pollo 0.1lb → mínimo 1/4 lb", "1/4" in str(result["market_qty"]))

result2 = apply_smart_market_units("Orégano", 0.005, "lb", 0.0, {"category": "Especias", "market_container": "Sobre", "container_weight_g": 10})
test("Orégano 0.005lb → mínimo 1 Sobre", result2["market_qty"] >= 1 and result2["market_unit"] == "Sobre")

# ============================================
# TEST 2: Data-Driven Container Resolution
# ============================================
print("\n⚡ TEST 2: Resolución Data-Driven (Sección 0.5)")

result3 = apply_smart_market_units("Avena", 1.1, "lb", 0.0, {"category": "Despensa y Granos", "market_container": "Paquete", "container_weight_g": 450})
test(f"Avena 1.1lb + Paquete 450g → 2 Paquetes (got qty={result3['market_qty']}, unit={result3['market_unit']})", 
     result3["market_qty"] == 2 and result3["market_unit"] == "Paquete")

result4 = apply_smart_market_units("Almendras", 0.44, "lb", 0.0, {"category": "Despensa", "market_container": "Fundita", "container_weight_g": 200})
test(f"Almendras 0.44lb + Fundita 200g → 1 Fundita (got qty={result4['market_qty']}, unit={result4['market_unit']})", 
     result4["market_qty"] == 1 and result4["market_unit"] == "Fundita") 

result5 = apply_smart_market_units("Arroz blanco", 2.2, "lb", 0.0, {"category": "Despensa y Granos", "market_container": "Paquete", "container_weight_g": 453})
test(f"Arroz 2.2lb + Paquete 453g → 3 Paquetes (got qty={result5['market_qty']}, unit={result5['market_unit']})", 
     result5["market_qty"] == 3 and result5["market_unit"] == "Paquete")

# Espinaca: Mazo de 280g
result6 = apply_smart_market_units("Espinaca", 0.5, "lb", 0.0, {"category": "Vegetales", "market_container": "Mazo", "container_weight_g": 280})
test(f"Espinaca 0.5lb + Mazo 280g → 1 Mazo (got qty={result6['market_qty']}, unit={result6['market_unit']})", 
     result6["market_qty"] == 1 and result6["market_unit"] == "Mazo")

# Proteína whey: Pote de 907g. 2lb = 907.18g → ceil(907.18/907) = 2 (punto flotante correcto)
result7 = apply_smart_market_units("Proteína whey", 2.0, "lb", 0.0, {"category": "Suplementos", "market_container": "Pote", "container_weight_g": 907})
test(f"Whey 2lb + Pote 907g → 2 Potes (got qty={result7['market_qty']}, unit={result7['market_unit']})", 
     result7["market_qty"] == 2 and result7["market_unit"] == "Pote")

# ============================================
# TEST 3: Display Category Mapping
# ============================================
print("\n🏷️ TEST 3: Categorías Server-Side")

test("Proteínas → 🥩 PROTEÍNAS", _get_display_category("Proteínas") == "🥩 PROTEÍNAS")
test("Lácteos → 🥛 LÁCTEOS", _get_display_category("Lácteos") == "🥛 LÁCTEOS")
test("Frutas → 🍎 FRUTAS", _get_display_category("Frutas") == "🍎 FRUTAS")
test("Vegetales → 🥗 VEGETALES", _get_display_category("Vegetales") == "🥗 VEGETALES")
test("Víveres → 🥔 VÍVERES", _get_display_category("Víveres") == "🥔 VÍVERES")
test("Despensa y Granos → 🥫 DESPENSA Y GRANOS", _get_display_category("Despensa y Granos") == "🥫 DESPENSA Y GRANOS")
test("Especias → 🧂 ESPECIAS", _get_display_category("Especias") == "🧂 ESPECIAS")
test("Suplementos → 💊 SUPLEMENTOS", _get_display_category("Suplementos") == "💊 SUPLEMENTOS")

# Fallback NLP
test("NLP Fallback: 'Pollo' → 🥩", _get_display_category("", "Pollo deshuesado") == "🥩 PROTEÍNAS")
test("NLP Fallback: 'Tomate' → 🥗", _get_display_category("", "Tomate rojo") == "🥗 VEGETALES")
test("NLP Fallback: 'Plátano' → 🥔", _get_display_category("", "Plátano verde") == "🥔 VÍVERES")
test("NLP Fallback: 'Arroz' → 🥫", _get_display_category("", "Arroz blanco") == "🥫 DESPENSA Y GRANOS")
test("NLP Fallback: 'Manzana' → 🍎", _get_display_category("", "Manzana verde") == "🍎 FRUTAS")
test("NLP Fallback: unknown → 🛒", _get_display_category("", "xyz desconocido") == "🛒 OTROS")

# ============================================
# TEST 4: Pluralización 
# ============================================
print("\n📦 TEST 4: Pluralización")
test("3 → Funditas", get_plural_unit(3, "Fundita") == "Funditas")
test("1 → Fundita", get_plural_unit(1, "Fundita") == "Fundita")
test("2 → Mazos", get_plural_unit(2, "Mazo") == "Mazos")
test("2 → sobres", get_plural_unit(2, "sobre") == "sobres")
test("2 → Paquetes", get_plural_unit(2, "Paquete") == "Paquetes")

# ============================================
# TEST 5: Market Minimums Coverage
# ============================================
print("\n📋 TEST 5: Cobertura MARKET_MINIMUMS")
expected = ["lb", "lbs", "Pote", "Paquete", "Fundita", "Mazo", "Lata", "Sobre", "Sobrecito", "Frasco", "Botella", "Cartón", "Envase", "Cabeza", "Ud."]
for unit in expected:
    test(f"'{unit}' en MARKET_MINIMUMS", unit in MARKET_MINIMUMS)

# ============================================
# RESUMEN
# ============================================
print(f"\n{'='*50}")
print(f"📊 RESULTADOS: {passed} PASSED / {failed} FAILED / {passed+failed} TOTAL")
if failed == 0:
    print("🎉 ¡TODOS LOS TESTS DEL NIVEL 3 PASARON!")
else:
    print(f"⚠️  {failed} test(s) fallaron.")
print(f"{'='*50}")
