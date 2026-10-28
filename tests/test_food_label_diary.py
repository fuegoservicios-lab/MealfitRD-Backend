import ast
import logging
from pathlib import Path
from unittest.mock import Mock
import unittest
from food_label_nutrients import snapshot, micros_from_snapshot, normalize, context
from diary_micros import resumen_micros
from packaged_food_portions import guard

LABEL = {'serving_text': '3 galletas (34 g)', 'nutrients': [
    {'key': 'fiber_g', 'amount': 1, 'unit': 'g'},
    {'key': 'sodium_mg', 'amount': 135, 'unit': 'mg'},
    {'key': 'vit_d_mcg', 'amount': 0, 'unit': 'mcg'},
]}


class FoodDiaryTests(unittest.TestCase):
    def test_label_quantity_and_unknown(self):
        totals = resumen_micros([{'micros': micros_from_snapshot(snapshot(LABEL, 2))}])
        self.assertEqual(totals['micros']['fiber_g'], 2)
        self.assertEqual(totals['micros']['sodium_mg'], 270)
        self.assertEqual(totals['micros']['vit_d_mcg'], 0)
        self.assertIsNone(totals['micros']['magnesium_mg'])
        self.assertEqual(totals['micros_coverage']['con_datos'], 1)

    def test_fraction_and_mixed_coverage(self):
        totals = resumen_micros([{'micros': micros_from_snapshot(snapshot(LABEL, .5))}, {'micros': None}])
        self.assertEqual(totals['micros']['sodium_mg'], 67.5)
        self.assertEqual(totals['micros_coverage']['by_nutrient']['sodium_mg']['status'], 'partial')

    def test_quantity_is_required(self):
        for count in (None, 0, -1, True, float('inf'), 101):
            with self.subTest(count=count), self.assertRaises(ValueError): snapshot(LABEL, count)

    def test_wrong_units_duplicates_and_nonfinite_rejected(self):
        for nutrient in ({'key':'fiber_g','amount':1,'unit':'mg'}, {'key':'folate_mcg','amount':100,'unit':'mcg'},
                         {'key':'iron_mg','amount':float('nan'),'unit':'mg'}, {'key':'sodium_mg','amount':-1,'unit':'mg'}):
            with self.subTest(nutrient=nutrient), self.assertRaises(ValueError):
                normalize({'serving_text':'1 pieza','nutrients':[nutrient]})
        with self.assertRaises(ValueError): normalize({**LABEL,'nutrients':LABEL['nutrients']*2})

    def test_does_not_trust_stored_totals(self):
        value = snapshot(LABEL, 1)
        value['values']['sodium_mg'] = 999999
        self.assertEqual(micros_from_snapshot(value)['values']['sodium_mg'], 135)
        self.assertIsNone(micros_from_snapshot({'source':'food_label'}))
        self.assertIn('no por paquete', context(LABEL))

    def test_cookie_clarification(self):
        for text in ('Me comí una galleta Oreo','Me comí un paquete de galletas'):
            self.assertTrue(guard('log_consumed_meal', {'meal_name':'Galletas Oreo'}, [{'role':'user','content':text}]))
        for text in ('Me comí una galleta Oreo suelta', 'Me comí un paquete de 4 galletas Oreo',
                     'Me comí 34 g de galletas', 'Registra un estimado del paquete de galletas'):
            self.assertIsNone(guard('log_consumed_meal', {'meal_name':'Galletas Oreo'}, [{'role':'user','content':text}]))

    def test_calorie_correction_not_packet(self):
        msgs = [{'role':'user','content':'Esa galleta Oreo tiene 170 calorías'}]
        self.assertTrue(guard('correct_consumed_meal', {'meal_name':'Paquete Oreo','ingredients':['34 g Oreo']}, msgs))
        self.assertIsNone(guard('correct_consumed_meal', {'calories':170}, msgs))

    def test_correction_clears_old_snapshot_and_scopes_owner(self):
        # Execute the real persistence function with a fake SQL executor, never a live database.
        source = ast.parse((Path(__file__).parents[1]/'db_facts.py').read_text(encoding='utf-8'))
        fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name=='update_consumed_meal')
        env = {'Optional': __import__('typing').Optional,'connection_pool':True,'Jsonb':lambda x:x,
               'execute_sql_write':Mock(return_value=[{'id':'meal'}]),'logger':logging.getLogger('test')}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<db-function>','exec'),env)
        for kwargs, expected in [({'calories':170},'attachment_ids'),
                                 ({'ingredients':['34 g galletas']},'ELSE NULL END'),
                                 ({'nutrition':snapshot(LABEL,2)},'nutrition = %s')]:
            env['update_consumed_meal']('owner','meal',**kwargs)
            sql, params = env['execute_sql_write'].call_args.args
            self.assertIn(expected,sql)
            self.assertIn('WHERE id = %s AND user_id = %s',sql)
            self.assertEqual(params[-2:],('meal','owner'))
        env['update_consumed_meal']('owner','meal',meal_type='cena')
        self.assertNotIn('nutrition',env['execute_sql_write'].call_args.args[0])

    def test_nutrition_with_attachment_ids_only_returns_none_for_label_micros(self):
        # [P1-ATTACHMENT-MICROS-FALLBACK · 2026-10-27] Comidas con adjuntos de foto guardan
        # nutrition = {'attachment_ids': [...]}. micros_from_snapshot debe dar None para que el diario
        # recurra limpiamente a los micros de los ingredientes del plato.
        self.assertIsNone(micros_from_snapshot({'attachment_ids': ['fake-uuid-1', 'fake-uuid-2']}))

    def test_insert_persists_snapshot_atomically(self):
        source = ast.parse((Path(__file__).parents[1]/'db_facts.py').read_text(encoding='utf-8'))
        fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name=='log_consumed_meal')
        write = Mock(return_value=[{'id':'meal'}])
        env = {'Optional':__import__('typing').Optional,'connection_pool':True,'Jsonb':lambda x:x,
               'execute_sql_write':write,'execute_sql_query':Mock(return_value=None),'logger':logging.getLogger('test')}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<db-insert>','exec'),env)
        nutrition = snapshot(LABEL,2)
        self.assertEqual(env['log_consumed_meal']('owner','Galletas',340,4,50,14,source='chat',nutrition=nutrition),'meal')
        sql, params = write.call_args.args
        self.assertIn('nutrition',sql)
        self.assertIn(nutrition,params)
        self.assertEqual(sql.count('%s'),len(params))

    def test_voice_unlimited_start_skips_spend_query_but_cap_is_enforced(self):
        import coach_live as live
        from unittest.mock import patch
        import types
        client = Mock(); client.post.return_value.status_code=201
        client.post.return_value.json.return_value={'session':{'id':'test-live'},'transport':{'sdp':'answer'}}
        with patch.object(live,'_session_http',return_value=client), patch.object(live,'gastado_usd',return_value=2) as spend, \
             patch.object(live,'tope_usd',return_value=0) as cap, patch.object(live.threading,'Thread'), \
             patch.dict('sys.modules',{'ios_free':types.SimpleNamespace(inherit_context=lambda fn:fn)}), \
             patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic-not-a-key'}):
            self.assertEqual(live.crear_sesion('owner','offer','chat'),('test-live','answer'))
            spend.assert_not_called()
            cap.return_value=1
            with self.assertRaises(live.LiveNoDisponible): live.crear_sesion('owner','offer','chat')
            spend.assert_called_once()
            self.assertEqual(client.post.call_count,1)
            live.SESIONES.pop('test-live',None)


if __name__ == '__main__': unittest.main()
