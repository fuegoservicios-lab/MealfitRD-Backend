import json
import math
from diary_micros import micros_de_ingredientes, resumen_micros, metas_micros
from diary_nutrients import NUTRIENTS, life_stage
from nutrition_db import IngredientNutritionDB

def test_unknown_zero_invalid_and_partial_are_distinct():
    class DB:
        def micros_from_ingredient_string(self, ing):
            return {'magnesium_mg': 25, 'b12_mcg': 0, 'zinc_mg': -1, 'iodine_mcg': math.nan} if ing == 'known' else None
    meal = micros_de_ingredientes(['known', 'unknown'], DB())
    assert meal['values']['magnesium_mg'] == 25
    assert meal['values']['b12_mcg'] == 0
    assert meal['values']['zinc_mg'] is None
    assert meal['values']['iodine_mcg'] is None
    assert meal['coverage']['b12_mcg'] == {'known': 1, 'total': 2, 'status': 'partial'}
    day = resumen_micros([{'micros': meal}, {'micros': None}])
    assert day['micros']['b12_mcg'] == 0
    assert day['micros']['vit_b6_mg'] is None
    assert day['micros_coverage']['by_nutrient']['magnesium_mg']['status'] == 'partial'
    json.dumps(day, allow_nan=False)

def test_portions_units_and_dfe_with_catalog_evidence():
    db = IngredientNutritionDB(rows=[{'name': 'Alimento de prueba', 'kcal_per_100g': 200,
        'vitamin_b6_mg_per_100g': .4, 'iodine_mcg_per_100g': 12, 'folate_mcg_dfe_per_100g': 80,
        'vitamin_b12_mcg_per_100g': .01, 'micronutrient_provenance': {key: {'source': 'USDA FDC', 'fdc_id': 123, 'reported': True} for key in ('vit_b6_mg', 'iodine_mcg', 'folate_mcg', 'b12_mcg')}}])
    result = micros_de_ingredientes(['150 g de Alimento de prueba'], db)
    assert result['values']['vit_b6_mg'] == .6
    assert result['values']['iodine_mcg'] == 18
    assert result['values']['folate_mcg'] == 120
    assert result['values']['b12_mcg'] == .015
    assert result['provenance']['vit_b6_mg'][0]['fdc_id'] == 123
    assert db.micros_from_ingredient_string('150 g de Alimento de prueba')['provenance']['catalog']['source']
    assert resumen_micros([{'micros': result}])['micros_coverage']['by_nutrient']['vit_b6_mg']['status'] == 'complete'

def test_references_age_stage_and_missing_profile():
    assert len(metas_micros('male', 30)) == 17
    assert metas_micros('male', 30)['magnesium_mg']['target'] == 400
    assert metas_micros('male', 31)['magnesium_mg']['target'] == 420
    assert metas_micros('masculina', 30)['zinc_mg']['target'] == 11
    assert metas_micros('female', 25, pregnant=True)['magnesium_mg']['target'] == 350
    breast = metas_micros('female', 35, lactating=True)
    assert breast['b12_mcg']['target'] == 2.8
    assert breast['folate_mcg']['target'] == 500
    assert breast['iodine_mcg']['target'] == 290
    assert breast['vit_e_mg']['target'] == 19
    assert metas_micros('female', 52)['vit_b6_mg']['target'] == 1.5
    assert metas_micros(None, 30) == metas_micros('male', None) == metas_micros('male', 17) == {}
    assert breast['magnesium_mg']['upper_limit'] is None
    assert breast['folate_mcg']['unit'] == 'mcg DFE'
    assert life_stage({'medicalConditions': ['Lactancia']}) == (False, True)
    assert life_stage({'isPregnant': 'sí'}) == (True, False)
    assert life_stage({'medicalConditions': ['Grossesse']}) == (True, False)
    assert life_stage({'medicalConditions': ['Allattamento']}) == (False, True)

def test_empty_day_and_legacy_evidence():
    empty = resumen_micros([])
    assert len(empty['micros']) == len(NUTRIENTS)
    assert all(v is None for v in empty['micros'].values())
    old = resumen_micros([{'micros': {'values': {'fiber_g': 0}}}])
    assert old['micros']['fiber_g'] == 0
    assert old['micros_coverage']['by_nutrient']['fiber_g']['status'] == 'partial'


def test_parenthesized_qty_unit_and_compound_head_before_con():
    from nutrition_db import _split_qty_unit_name

    assert _split_qty_unit_name('aceite vegetal (≈1 cucharada)') == (1.0, 'cucharada', 'aceite vegetal')
    assert _split_qty_unit_name('mantequilla (1 cucharadita)') == (1.0, 'cucharadita', 'mantequilla')

    db = IngredientNutritionDB(rows=[
        {'name': 'Zanahoria', 'aliases': ['zanahoria', 'zanahorias'], 'kcal_per_100g': 44.2, 'carbs_g_per_100g': 10.0},
        {'name': 'Aceite de oliva', 'aliases': ['aceite de oliva'], 'kcal_per_100g': 900.0, 'fats_g_per_100g': 100.0},
        {'name': 'Aceite vegetal', 'aliases': ['aceite vegetal'], 'kcal_per_100g': 884.0, 'fats_g_per_100g': 100.0, 'density_g_per_cup': 218.0},
        {'name': 'Arroz con leche', 'aliases': ['arroz con leche'], 'kcal_per_100g': 118.0},
        {'name': 'Arroz blanco', 'aliases': ['arroz'], 'kcal_per_100g': 360.0},
        {'name': 'Leche', 'aliases': ['leche'], 'kcal_per_100g': 61.0},
        {'name': 'Pollo', 'aliases': ['pollo'], 'kcal_per_100g': 165.0, 'protein_g_per_100g': 31.0},
    ])

    carrots = db.lookup('zanahorias baby cocidas con aceite de oliva (≈120 g; 1 cucharadita aceite)')
    assert carrots is not None and carrots.name == 'Zanahoria'
    carrots_macros = db.macros_from_ingredient_string('zanahorias baby cocidas con aceite de oliva (≈120 g; 1 cucharadita aceite)')
    assert carrots_macros is not None and carrots_macros['kcal'] < 100.0

    oil_g = db.grams_from_ingredient_string('aceite vegetal (≈1 cucharada)')
    assert oil_g is not None and 12.0 <= oil_g <= 15.0

    assert db.lookup('Arroz con leche').name == 'Arroz con leche'
    assert db.lookup('ensalada con pollo').name == 'Pollo'

