import ast
import logging
from pathlib import Path
from typing import Optional
from unittest.mock import Mock
import pytest
from meal_registration_time import infer_current_meal_slot, explicit_repeat_consumption, confirmed_repeat_consumption

@pytest.mark.parametrize('text,slot', [
 ('Me comí una galleta Oreo','snack'), ('Me comí una Oreo','snack'),
 ('Me comí un paquete de galletas Oreo','snack'), ('Me comí dos galletas','snack'),
 ('Me comí un plátano grande maduro, un pedazo de aguacate, y dos huevos','almuerzo'),
 ('Me comí arroz con pollo','almuerzo'), ('Me comí galletas y dos huevos','almuerzo'),
 ('Me comí una Oreo de almuerzo',None), ('Me comí unas galletas ayer',None),
])
def test_snack_and_meal_slots_follow_actual_description(text,slot):
 assert infer_current_meal_slot([{'role':'user','content':text}],13)==slot

@pytest.mark.parametrize('text,allowed', [
 ('Me comí otra porción de arroz',True), ('Repetí el plato',True),
 ('I ate another serving',True), ('No me comí otra porción',False),
 ('¿Puedo comer otro plato?',False), ('Me comeré otra porción',False),
 ('Me comí huevos',False), ('Me serví otra porción',False),
 ('¿Repetí el plato?',False), ('Repetí la pregunta',False),
])
def test_only_fresh_consumption_authorizes_an_identical_repeat(text,allowed):
 assert explicit_repeat_consumption([{'role':'assistant','content':'Repetí el plato'},
                                     {'role':'user','content':text}])==allowed


@pytest.mark.parametrize('text,warning,allowed', [
 ('sí',True,True), ('sí, comí dos',True,True), ('sí',False,False),
 ('no, solo uno',True,False), ('cámbialo a cena',True,False),
 ('Sí quiero comer otra porción mañana',True,False),
 ('Me comí otra porción',False,True),
])
def test_duplicate_warning_does_not_authorize_every_next_response(text,warning,allowed):
 assert confirmed_repeat_consumption([{'role':'user','content':text}],warning)==allowed


def test_photo_wrapper_is_context_not_repeat_authorization():
 text='Contexto: si repetí el plato. Mensaje del usuario: "Me comí una Oreo suelta"'
 messages=[{'role':'user','content':[{'type':'text','text':text}]}]
 assert not explicit_repeat_consumption(messages)
 assert infer_current_meal_slot(messages,13)=='snack'

def _db_function():
 source=ast.parse((Path(__file__).parents[1]/'db_facts.py').read_text(encoding='utf-8'))
 fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='log_consumed_meal')
 env={'Optional':Optional,'connection_pool':True,'Jsonb':lambda x:x,
      'logger':logging.getLogger('test'),'execute_sql_query':Mock(),
      'execute_sql_write':Mock(return_value=[{'id':'new'}])}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'<diary>','exec'),env)
 return env

def test_retry_does_not_insert_and_uses_food_and_quantity_not_slot(monkeypatch):
 monkeypatch.setenv('MEALFIT_CONSUMED_MEAL_DEDUP_WINDOW_S','60')
 env=_db_function(); env['execute_sql_query'].return_value={'id':'first'}
 assert env['log_consumed_meal']('owner','Huevos',140,12,1,10,['2 huevos'],meal_type='almuerzo')=='deduped'
 env['execute_sql_write'].assert_not_called()
 sql,params=env['execute_sql_query'].call_args.args
 assert params[:7]==('owner','Huevos',140,12,1,10,['2 huevos'])
 assert 'meal_type = %s' not in sql
 assert all(name+' = %s' in sql for name in ('calories','protein','carbs','healthy_fats','ingredients'))

def test_changed_quantity_and_confirmed_repeat_can_be_inserted(monkeypatch):
 monkeypatch.setenv('MEALFIT_CONSUMED_MEAL_DEDUP_WINDOW_S','60')
 env=_db_function(); env['execute_sql_query'].return_value=None
 assert env['log_consumed_meal']('owner','Huevos',210,18,1,15,['3 huevos'],meal_type='almuerzo')=='new'
 env['execute_sql_query'].reset_mock();env['execute_sql_query'].return_value={'id':'old'}
 assert env['log_consumed_meal']('owner','Huevos',210,18,1,15,['3 huevos'],meal_type='almuerzo',allow_repeat=True)=='new'
 env['execute_sql_query'].assert_not_called()
