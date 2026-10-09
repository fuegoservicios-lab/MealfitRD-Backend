from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from meal_registration_time import infer_current_meal_slot, normalize_live_offset


@pytest.mark.parametrize('legacy,fixed', [(-240, 240), (330, -330), (0, 0), (840, -840)])
def test_both_live_client_generations_use_chat_clock(legacy, fixed):
    assert normalize_live_offset(legacy) == fixed
    assert normalize_live_offset(fixed, 'utc_minus_local') == fixed


@pytest.mark.parametrize('invalid', [True, '240', None, 900, -900, float('nan'), float('inf'), 30.5])
def test_bad_offsets_fall_back_to_profile(invalid):
    assert normalize_live_offset(invalid) is None


@pytest.mark.parametrize('hour,slot', [(9, 'desayuno'), (13, 'almuerzo'), (16, 'merienda'),
                                     (18.49, 'merienda'), (18.5, 'cena'), (21 + 56/60, 'cena')])
def test_fresh_consumption_uses_clock_not_previous_topic(hour, slot):
    messages = [HumanMessage(content='Háblame de la merienda'), AIMessage(content='¿Quieres merendar?'),
                HumanMessage(content='Me comí tres pasteles en hoja grandes'), AIMessage(content='')]
    assert infer_current_meal_slot(messages, hour) == slot


@pytest.mark.parametrize('text', [
    'Me comí tres pasteles en hoja de merienda', 'Me comí tres pasteles en hoja en la cena',
    'Me comí tres pasteles ayer', 'Me comí tres pasteles a las 10',
    'Me comí tres pasteles a las 10:30 pm', 'Me comí tres pasteles esta tarde',
    'Sí, tres grandes', '¿Puedo comer tres pasteles?', 'No me comí tres pasteles',
    'I ate breakfast', 'I ate three rolls yesterday', 'Comi no jantar', 'Ho mangiato a pranzo',
])
def test_explicit_meal_time_and_clarifications_are_not_overwritten(text):
    assert infer_current_meal_slot([HumanMessage(content=text)], 21.9) is None


@pytest.mark.parametrize('schedule', ['night_shift', 'variable'])
def test_shift_worker_schedule_is_preserved(schedule):
    assert infer_current_meal_slot([HumanMessage(content='Me comí tres pasteles')], 21.9, schedule) is None


def test_clock_unknown_does_not_invent_slot():
    assert infer_current_meal_slot([HumanMessage(content='Me comí tres pasteles')], None) is None


def test_cooking_method_is_not_a_clock_reference():
    assert infer_current_meal_slot([HumanMessage(content='Me comí pollo a la plancha')], 21.9) == 'cena'


@pytest.mark.parametrize('offset,convention,expected', [(-240, None, 240),
    (240, 'utc_minus_local', 240), (330, None, -330), (0, 'utc_minus_local', 0)])
def test_live_endpoint_passes_canonical_offset_to_session(monkeypatch, offset, convention, expected):
    import asyncio
    import coach_live
    import db_chat
    from routers import chat
    calls = []
    monkeypatch.setattr(db_chat, 'get_session_owner', lambda *args: 'owner')
    def create(*args):
        calls.append(args)
        return 'live', 'answer-sdp'
    monkeypatch.setattr(coach_live, 'crear_sesion', create)
    result = asyncio.run(chat.api_chat_live_sesion({
        'sdp': 'offer-sdp', 'session_id': '12345678-1234-1234-1234-123456789abc',
        'tz_offset': offset, 'tz_offset_convention': convention,
    }, verified_user_id='owner'))
    assert result['sdp'] == 'answer-sdp'
    assert calls[0][-1] == expected


def test_actual_agent_rewrites_guessed_merienda_before_invoking_diary(monkeypatch):
    import agent
    import prompts.chat_agent
    monkeypatch.setattr(prompts.chat_agent, 'hora_local_del_chat', lambda offset: 21 + 56/60)
    calls = []
    monkeypatch.setattr(agent, 'agent_tools', [SimpleNamespace(
        name='log_consumed_meal', invoke=lambda args: calls.append(dict(args)) or 'Guardado como cena')])
    args = {'meal_name': '3 pasteles en hoja grandes', 'calories': 900, 'protein': 30,
            'carbs': 120, 'healthy_fats': 30, 'meal_type': 'merienda', 'days_ago': 0}
    state = {'user_id': 'owner', 'session_id': 'chat', 'form_data': {}, 'tz_offset': 240,
             'messages': [HumanMessage(content='Me comí tres pasteles en hoja grandes'),
                          AIMessage(content='', tool_calls=[{'name': 'log_consumed_meal',
                              'args': args, 'id': 'write', 'type': 'tool_call'}])]}
    result = agent.execute_tools(state)
    assert calls[0]['user_id'] == 'owner'
    assert calls[0]['meal_type'] == 'cena'
    assert calls[0]['calories'] == 900 and calls[0]['meal_name'] == '3 pasteles en hoja grandes'
    assert 'cena' in result['messages'][0].content


def test_long_live_session_refreshes_local_date_each_turn(monkeypatch):
    import coach_live
    from routers import chat
    captured = []
    async def stream():
        yield 'data: {"type":"done","response":"Listo."}\n\n'
    def api(tasks, data, user):
        captured.append(data)
        return SimpleNamespace(body_iterator=stream())
    monkeypatch.setattr(chat, 'api_chat_stream', api)
    session = coach_live.SesionLive('live', 'owner', 'chat', local_date='2020-01-01', tz_offset=240)
    coach_live.correr_turno_del_coach(session, 'Me comí tres pasteles')
    expected = (datetime.now(timezone.utc) - timedelta(minutes=240)).date().isoformat()
    assert captured[0]['local_date'] == expected
    assert captured[0]['tz_offset'] == 240


def test_2156_rd_is_not_0556_when_legacy_voice_offset_is_corrected(monkeypatch):
    from prompts import chat_agent
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 9, 1, 56, tzinfo=timezone.utc)
    monkeypatch.setattr(chat_agent, 'datetime', Clock)
    offset = normalize_live_offset(-240)
    assert chat_agent.hora_local_del_chat(offset) == pytest.approx(21 + 56/60)
    assert '21:56' in chat_agent.build_temporal_context('2026-10-08', offset)
