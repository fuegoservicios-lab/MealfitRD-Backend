"""Declared food-label nutrients; never infer missing values or serving counts."""
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict
from diary_nutrients import NUTRIENTS, valid_amount

KEYS = {key: unit for key, _, unit, _ in NUTRIENTS}


class FoodLabelNutrient(BaseModel):
    model_config = ConfigDict(extra='forbid')
    key: Literal['fiber_g', 'sodium_mg', 'potassium_mg', 'calcium_mg', 'iron_mg',
                 'vit_c_mg', 'vit_a_mcg', 'vit_d_mcg', 'magnesium_mg', 'zinc_mg',
                 'b12_mcg', 'folate_mcg', 'vit_e_mg', 'vit_k_mcg', 'selenium_mcg',
                 'vit_b6_mg', 'iodine_mcg']
    amount: float = Field(ge=0, le=100000, allow_inf_nan=False, strict=True)
    unit: Literal['g', 'mg', 'mcg', 'mcg RAE', 'mcg DFE']


class FoodLabel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    serving_text: str = Field(min_length=1, max_length=200)
    nutrients: list[FoodLabelNutrient] = Field(max_length=17)


def normalize(label):
    if not label:
        return None
    data = FoodLabel.model_validate(label).model_dump()
    seen = set()
    for item in data['nutrients']:
        key = item['key']
        # No implicit %DV, IU, retinol or folic-acid -> DFE conversions.
        expected = 'g' if key == 'fiber_g' else 'mcg' if key.endswith('_mcg') else 'mg'
        if key == 'vit_a_mcg': expected = 'mcg RAE'
        if key == 'folate_mcg': expected = 'mcg DFE'
        if key in seen or item['unit'] != expected:
            raise ValueError('Nutriente duplicado o unidad incompatible con el contador')
        seen.add(key)
    return data


def snapshot(label, servings):
    if label is None:
        return None
    label = normalize(label)
    count = valid_amount(servings)
    if isinstance(servings, bool) or count is None or not 0 < count <= 100:
        raise ValueError('Indica cuántas porciones de esa etiqueta consumiste; no se supone una porción')
    values = {n['key']: round(n['amount'] * count, 3) for n in label['nutrients']}
    return {'source': 'food_label', 'label': label, 'servings': count, 'values': values}


def micros_from_snapshot(data):
    if not isinstance(data, dict) or data.get('source') != 'food_label':
        return None
    try:
        data = snapshot(data.get('label'), data.get('servings'))
    except (TypeError, ValueError):
        return None
    if not data or not data['values']:
        return None
    values = {key: data['values'].get(key) for key in KEYS}
    return {'values': values, 'resolved': 1, 'total': 1,
            'coverage': {key: {'known': int(value is not None), 'total': 1,
                              'status': 'complete' if value is not None else 'none'} for key, value in values.items()},
            'provenance': {key: [{'source': 'Etiqueta del producto', 'serving': data['label']['serving_text']}]
                           if value is not None else [] for key, value in values.items()}}


def context(label):
    import json
    try:
        label = normalize(label)
    except (TypeError, ValueError):
        return ''
    return ('\nNUTRIENTES LEÍDOS POR PORCIÓN (no por paquete): ' + json.dumps(label, ensure_ascii=False)
            + '. Si confirma consumo, conserva food_facts y usa label_servings según lo que consumió; '
              'si no se conoce la cantidad, pregunta antes de registrar.\n') if label else ''
