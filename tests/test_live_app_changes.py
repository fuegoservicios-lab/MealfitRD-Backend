import asyncio
import json
import threading
from types import SimpleNamespace

import coach_live


def test_el_canal_despierta_cuando_se_guarda_un_cambio_y_respeta_el_cursor():
    sesion = coach_live.SesionLive('live', 'user', 'chat')
    respuesta = []
    entrando = threading.Event()

    def esperar():
        entrando.set()
        respuesta.append(sesion.esperar_novedades(0, 20))

    hilo = threading.Thread(target=esperar, daemon=True)
    hilo.start()
    assert entrando.wait(1)
    sesion.publicar({'agua': True})
    hilo.join(1)
    assert not hilo.is_alive(), 'no espera al siguiente intervalo de polling'
    assert respuesta[0]['novedades'] == [{'n': 1, 'agua': True}]
    assert sesion.esperar_novedades(1)['novedades'] == []
    sesion.finalizar()
    assert sesion.esperar_novedades(1, 20)['cerrada']


def test_el_done_fragmentado_publica_antes_de_background_y_sin_etiquetas_llm(monkeypatch):
    from routers import chat
    sesion = coach_live.SesionLive('live', 'user', 'chat')
    observado = []
    final = {'type': 'done', 'response': 'Anoté el vaso.', 'updated_fields': {'weight': 70},
             'new_plan': {'days': []}, 'pantry_modified_at': 123,
             'ajustes_de_app': {'tema': 'dark'}}
    evento = ('data: ' + json.dumps(final, ensure_ascii=False) + '\n\n').encode()

    async def stream():
        yield evento[:17]
        yield evento[17:39]
        yield evento[39:]

    def api(tareas, datos, user):
        assert datos['is_call_mode'] and user == 'user'
        assert coach_live.turno_live_sin_cuota(), 'voz no consume créditos del chat'
        tareas.add_task(lambda: observado.append(sesion.esperar_novedades(0)))
        return SimpleNamespace(body_iterator=stream())

    monkeypatch.setattr(chat, 'api_chat_stream', api)
    respuesta, cambios = coach_live.correr_turno_del_coach(sesion, 'Me bebí un vaso de agua')
    assert not coach_live.turno_live_sin_cuota(), 'la exención no se filtra a mensajes escritos'
    assert respuesta == 'Anoté el vaso.'
    assert cambios == {'cambios_publicados': True}
    n = observado[0]['novedades'][0]
    assert n['agua'] and n['diario'] and n['plan'] and n['perfil'] and n['nevera']
    assert n['ajustes_de_app'] == {'tema': 'dark'}
    assert n['turno_completo'] is False


def test_error_del_coach_no_publica_un_cambio_inventado(monkeypatch):
    from routers import chat
    sesion = coach_live.SesionLive('live', 'user', 'chat')

    async def stream():
        yield 'data: {"type":"error","message":"sin red"}\n\n'

    monkeypatch.setattr(chat, 'api_chat_stream', lambda *a: SimpleNamespace(body_iterator=stream()))
    coach_live.correr_turno_del_coach(sesion, 'anota agua')
    assert sesion.novedades == []


def test_novedades_se_publican_antes_de_que_empiece_la_respuesta_hablada(monkeypatch):
    import consentimientos
    sesion = coach_live.SesionLive('live', 'user', 'chat')
    monkeypatch.setattr(consentimientos, 'permite_ia', lambda *a: True)
    monkeypatch.setattr(coach_live, 'correr_turno_del_coach', lambda *a: ('Listo.', {'agua': True}))

    class WS:
        def send(self, mensaje):
            assert json.loads(mensaje)['type'] == 'session.commentary.append'
            assert sesion.esperar_novedades(0)['novedades'][0]['agua']

    coach_live._delegar(WS(), sesion, 'd1', 'un vaso')


def test_exencion_de_cuota_se_limpia_si_falla_el_coach(monkeypatch):
    import pytest
    from routers import chat
    def api(*args):
        assert coach_live.turno_live_sin_cuota()
        raise RuntimeError('sin conexión')
    monkeypatch.setattr(chat, 'api_chat_stream', api)
    with pytest.raises(RuntimeError):
        coach_live.correr_turno_del_coach(coach_live.SesionLive('live', 'user', 'chat'), 'hola')
    assert not coach_live.turno_live_sin_cuota()
