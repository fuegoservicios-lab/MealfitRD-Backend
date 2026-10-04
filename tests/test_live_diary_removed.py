import ast
import json
import logging
from pathlib import Path
import pytest
import coach_live as live

@pytest.fixture
def sessions(monkeypatch):
    mine=live.SesionLive('live-own','u1','chat-own')
    other=live.SesionLive('live-other','u2','chat-other')
    closed=live.SesionLive('live-closed','u1','chat-closed',cerrada=True)
    monkeypatch.setattr(live,'SESIONES',{s.live_id:s for s in [mine,other,closed]})
    return mine,other,closed

def test_only_active_owned_sessions_receive_the_verified_delete_once(sessions):
    mine,other,closed=sessions
    meal={'id':'meal-1','meal_name':'Lasaña','meal_type':'cena','consumed_at':'2026-10-01'}
    live.notificar_borrado('u1',meal)
    live.notificar_borrado('u1',meal)
    assert len(mine._cambios_app)==1
    assert not other._cambios_app and not closed._cambios_app

def test_without_new_user_speech_the_notice_updates_live_and_chat(sessions,monkeypatch):
    import consentimientos,db_chat
    monkeypatch.setattr(consentimientos,'permite_ia',lambda *a:True)
    saved=[]
    monkeypatch.setattr(db_chat,'save_message',lambda *a,**k:saved.append((a,k)))
    s=sessions[0]
    live.notificar_borrado('u1',{'id':'meal-1','meal_name':'Lasaña','meal_type':'cena'})
    events=[]
    class WS:
        def send(self,event): events.append(json.loads(event))
    live._enviar_cambios_app(WS(),s)
    assert [e['type'] for e in events]==['session.thinking.append','session.commentary.append']
    assert all(e['delegation_id'] is None for e in events)
    assert 'Ya no cuenta' in events[0]['content'] and 'Lasaña' in events[0]['content']
    assert 'Eliminaste' in events[1]['content'] and 'ya no cuenta' in events[1]['content']
    assert saved[0][0][0:2]==('chat-own','model') and saved[0][1]['user_id']=='u1'
    assert s.novedades[0]['aviso_diario'] and s.novedades[0]['diario']
    live._enviar_cambios_app(WS(),s)
    assert len(events)==2

def test_failed_delivery_is_retried_and_does_not_claim_a_spoken_notice(sessions,monkeypatch):
    import consentimientos
    monkeypatch.setattr(consentimientos,'permite_ia',lambda *a:True)
    s=sessions[0]
    live.notificar_borrado('u1',{'id':'meal-1','meal_name':'Lasaña'})
    class WS:
        def send(self,event): raise RuntimeError('socket unavailable')
    live._enviar_cambios_app(WS(),s)
    assert len(s._cambios_app)==1 and s.novedades==[]

def test_revoked_consent_does_not_send_diary_context(sessions,monkeypatch):
    import consentimientos
    monkeypatch.setattr(consentimientos,'permite_ia',lambda *a:False)
    s=sessions[0]
    live.notificar_borrado('u1',{'id':'meal-1','meal_name':'Lasaña'})
    class WS:
        def send(self,event): pytest.fail('revoked consent')
    live._enviar_cambios_app(WS(),s)

@pytest.mark.parametrize('rows,expected', [([],False),([{'id':'meal-1','meal_name':'Lasaña'}],True)])
def test_the_actual_delete_notifies_only_after_a_successful_owned_write(monkeypatch,rows,expected):
    path=Path(live.__file__).parent/'db_facts.py'
    fn=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='delete_consumed_meal')
    writes=[]; notices=[]
    def write(sql,args,**kwargs):
        assert 'user_id = %s' in sql and args[:2]==('meal-1','u1')
        writes.append('committed'); return rows
    monkeypatch.setattr(live,'notificar_borrado',lambda *a:notices.append((list(writes),a)))
    ns={'logger':logging.getLogger('test'),'connection_pool':True,'execute_sql_write':write}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    assert ns['delete_consumed_meal']('u1','meal-1') is expected
    assert bool(notices)==expected
    if expected: assert notices==[(['committed'],('u1',rows[0]))]
