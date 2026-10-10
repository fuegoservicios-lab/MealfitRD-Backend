import json
import math
import pytest
from supplement_labels import analyze, normalize

FACTS = {'serving_text': '3 capsules', 'units_per_serving': 3, 'serving_unit': 'capsules',
         'amount_basis': 'per_serving', 'table_readable': True, 'nutrients': [
             {'name': 'Magnesium', 'amount': 300, 'unit': 'mg', 'form': 'citrate', 'basis': 'elemental',
              'label_text': 'Magnesium 300 mg (as citrate)'},
             {'name': 'Vitamin B12', 'amount': 2.4, 'unit': 'mcg', 'basis': 'declared', 'label_text': 'Vitamin B12 2.4 mcg'}]}


def test_one_capsule_is_a_third_of_the_label_serving():
    result = analyze(FACTS, units_taken=1, taken_unit='cápsula')
    assert result['servings_taken'] == pytest.approx(1/3)
    assert result['dose'][0]['amount'] == 100
    assert result['dose'][1]['amount'] == .8
    assert not result['needs']
    assert result['scope'] == 'label_interpretation_only_no_diary_write'


def test_portion_arithmetic_is_not_capsule_arithmetic():
    assert analyze(FACTS, servings_taken=2)['dose'][0]['amount'] == 600
    assert analyze(FACTS, units_taken=2, taken_unit='g')['dose'][0]['amount'] is None
    assert analyze(FACTS, units_taken=2)['dose'][0]['amount'] is None
    assert analyze(FACTS, units_taken=2, servings_taken=2, taken_unit='capsule')['dose'][0]['amount'] is None


def test_missing_amount_basis_or_serving_size_is_never_assumed():
    assert analyze(FACTS)['dose'][0]['amount'] is None
    for overrides in ({'units_per_serving': None}, {'table_readable': False}, {'amount_basis': 'per_100g'}):
        r = analyze({**FACTS, **overrides}, units_taken=1, taken_unit='capsule')
        assert r['dose'][0]['amount'] is None and r['needs']


@pytest.mark.parametrize('value', [None, -1, math.inf, math.nan, True, [], {}])
def test_absent_or_invalid_minerals_are_not_zero(value):
    facts = {**FACTS, 'nutrients': [{**FACTS['nutrients'][0], 'amount': value}]}
    r = analyze(facts, servings_taken=1)
    assert r['label']['nutrients'][0]['amount'] is None
    assert r['dose'][0]['amount'] is None
    json.dumps(r, allow_nan=False)


def test_zero_is_valid_but_percent_or_unsubstantiated_numbers_are_unknown():
    for amount, unit, evidence, expected in [(0, 'mg', 'Magnesium 0 mg', 0), (50, '%', '50% DV', None), (400, 'mg', '', None)]:
        r = normalize({**FACTS, 'nutrients': [{'name': 'Magnesium', 'amount': amount, 'unit': unit, 'label_text': evidence}]})
        assert r['nutrients'][0]['amount'] == expected


def test_compound_iu_and_folic_acid_are_not_converted():
    rows = [{'name': 'Magnesium citrate', 'amount': 1000, 'unit': 'mg', 'basis': 'compound', 'label_text': 'Magnesium citrate 1000 mg'},
            {'name': 'Vitamin D', 'amount': 400, 'unit': 'IU', 'label_text': 'Vitamin D 400 IU'},
            {'name': 'Folic acid', 'amount': 400, 'unit': 'mcg', 'label_text': 'Folic acid 400 mcg'}]
    r = analyze({**FACTS, 'nutrients': rows}, servings_taken=1)
    assert r['dose'][0] == {'name': 'Magnesium citrate', 'amount': 1000, 'unit': 'mg', 'form': '', 'basis': 'compound'}
    assert r['dose'][1]['unit'] == 'IU'
    assert r['dose'][2]['name'] == 'Folic acid' and r['dose'][2]['unit'] == 'mcg'


def test_photo_of_zero_calorie_supplement_is_a_readable_label_and_reaches_chat():
    from vision_agent import _coerce_meal_scan, _MealVisionResult
    from prompts.chat_agent import build_vision_context, build_tools_instructions, build_tools_instructions_stream
    payload = _MealVisionResult(photo_kind='etiqueta', is_food=True, description='Magnesium label', supplement_facts=FACTS).model_dump()
    result = _coerce_meal_scan(payload)
    assert result['label_read'] and result['calories'] == 0
    assert 'No se lee' not in result['description']
    for photo in ({'kind': 'etiqueta', **result}, {'kind': 'multi', 'items': [{'kind': 'etiqueta', **result}]}):
        text = build_vision_context(photo)
        assert '300' in text and 'units_per_serving' in text
    for builder in (build_tools_instructions, build_tools_instructions_stream):
        assert 'analizar_suplemento' in builder('test-user')


def test_saved_label_keeps_minerals_without_changing_legacy_whey_macros():
    import suplementos
    whey = {'gramos_porcion': 31, 'kcal': 120, 'protein_g': 24, 'carbs_g': 3, 'fats_g': 1.5}
    assert suplementos.etiqueta_valida(whey) == whey
    saved = suplementos.etiqueta_valida({**whey, 'supplement_facts': FACTS})
    assert saved['supplement_facts']['nutrients'][0]['amount'] == 300
    assert suplementos.macros_de_porciones(saved, 2) == {'kcal': 240, 'protein_g': 48, 'carbs_g': 6, 'fats_g': 3}


def test_analysis_tool_reads_only_the_current_user_saved_label(monkeypatch):
    import tools
    import suplementos
    queries = []
    def lookup(uid, name):
        queries.append((uid, name))
        return {'serving_label': {'supplement_facts': FACTS}}
    def no_write(*args, **kwargs):
        pytest.fail('Analysis must never write inventory or diary')
    monkeypatch.setattr(suplementos, 'buscar', lookup)
    monkeypatch.setattr(suplementos, '_upsert', no_write)
    monkeypatch.setattr(suplementos, 'descontar', no_write)
    result = json.loads(tools.analizar_suplemento.invoke({'user_id': 'current-user', 'nombre': 'Magnesium',
                                                        'unidades_tomadas': 1, 'unidad_tomada': 'capsula'}))
    assert queries == [('current-user', 'Magnesium')]
    assert result['dose'][0]['amount'] == 100
    assert 'analizar_suplemento' in [t.name for t in tools.agent_tools]
    guest = json.loads(tools.analizar_suplemento.invoke({'user_id': 'guest', 'nombre': 'Magnesium'}))
    assert guest['status'] == 'missing_label' and len(queries) == 1


def test_supplement_only_label_is_preserved_without_fabricated_macros():
    import suplementos
    result = suplementos.etiqueta_valida({'supplement_facts': FACTS})
    assert result['supplement_facts']['units_per_serving'] == 3
    assert result['gramos_porcion'] == 0
    assert all(result[key] == 0 for key in ('kcal', 'protein_g', 'carbs_g', 'fats_g'))
