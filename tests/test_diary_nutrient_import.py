from scripts.backfill_diary_nutrients import extract

def test_import_requires_correct_unit_and_dfe_and_does_not_manufacture_zero():
    food = {'foodNutrients': [
        {'nutrient': {'id': 1175, 'unitName': 'mg'}, 'amount': .4},
        {'nutrient': {'id': 1100, 'unitName': 'µg'}, 'amount': 0},
        {'nutrient': {'id': 1177, 'unitName': 'µg'}, 'amount': 90},
        {'nutrient': {'id': 1178, 'unitName': 'mg'}, 'amount': 1},
        {'nutrient': {'id': 1095, 'unitName': 'mg'}, 'amount': -1},
    ]}
    values = extract(food)
    assert values['vit_b6_mg'] == .4
    assert values['iodine_mcg'] == 0
    assert values['folate_mcg'] is None
    assert values['b12_mcg'] is None
    assert values['zinc_mg'] is None

def test_unverified_legacy_fill_is_not_exposed_as_new_evidence():
    from nutrition_db import IngredientNutritionDB
    from diary_micros import micros_de_ingredientes
    db = IngredientNutritionDB(rows=[{'name':'Prueba', 'kcal_per_100g':100,
        'zinc_mg_per_100g':0, 'fiber_g_per_100g':0}])
    result = micros_de_ingredientes(['100 g de Prueba'], db)
    assert result['values']['zinc_mg'] is None
    assert result['values']['fiber_g'] == 0
