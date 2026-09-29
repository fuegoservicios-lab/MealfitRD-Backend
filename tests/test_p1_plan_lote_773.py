"""[P1-PLAN-LOTE-773 · 2026-09-28] El aviso de un regalo sale en el idioma de la persona. `push_i18n` traduce por texto
español exacto y no admite cifras ni fechas: la frase se arma aquí, en los cinco idiomas."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

import avisos_regalo as av

RD = ZoneInfo("America/Santo_Domingo")
FIN_MES = datetime(2026, 10, 1, tzinfo=timezone.utc)      # exclusivo: vale hasta el 30-sep


def test_la_fecha_es_el_ultimo_dia_que_vale():
    assert av.fecha_corta(FIN_MES, "es-DO") == "30/09"
    assert av.fecha_corta(FIN_MES, "en-US") == "09/30"
    assert av.fecha_corta(datetime(2026, 11, 1, tzinfo=RD), "fr-FR") == "31/10"   # «hasta el 31-oct» incluido


@pytest.mark.parametrize("idioma,titulo,cuerpo", [
    ("es-DO", "Tienes un regalo 🎁", "Te regalamos 20 créditos para crear planes, válidos hasta el 30/09."),
    ("en-US", "You have a gift 🎁", "We gave you 20 credits to create plans, valid until 09/30."),
    ("pt-BR", "Você ganhou um presente 🎁", "Você ganhou 20 créditos para criar planos, válidos até 30/09."),
    ("fr-FR", "Vous avez un cadeau 🎁", "Nous vous offrons 20 crédits pour créer des plans, valables jusqu’au 30/09."),
    ("it-IT", "Hai un regalo 🎁", "Ti regaliamo 20 crediti per creare piani, validi fino al 30/09."),
])
def test_creditos_en_cada_idioma(idioma, titulo, cuerpo):
    regalo = {"id": "g", "kind": "creditos_generacion", "amount": 20, "plan": None, "ends_at": FIN_MES}
    assert av.texto_del_aviso(regalo, idioma) == (titulo, cuerpo)


def test_singular_del_coach():
    regalo = {"id": "g", "kind": "creditos_coach", "amount": 1, "plan": None, "ends_at": FIN_MES}
    assert av.texto_del_aviso(regalo, "es-DO")[1] == "Te regalamos 1 mensaje más con tu coach, válido hasta el 30/09."


def test_plan_con_y_sin_fecha():
    fin = datetime(2026, 11, 1, tzinfo=RD)
    assert av.texto_del_aviso({"kind": "plan", "plan": "ultra", "ends_at": fin}, "es-DO")[1] == \
        "Tienes Max de cortesía hasta el 31/10."
    assert av.texto_del_aviso({"kind": "plan", "plan": "basic", "ends_at": None}, "it-IT")[1] == "Ora hai Base in omaggio."


def test_idioma_desconocido_cae_al_espanol():
    assert av.texto_del_aviso({"kind": "plan", "plan": "plus", "ends_at": None}, "de-DE")[0] == "Tienes un regalo 🎁"


def test_avisar_usa_el_idioma_y_nunca_lanza(monkeypatch):
    import utils_push
    enviados = []
    monkeypatch.setattr(av, "execute_sql_query", lambda *a, **k: {"locale": "en-US"})
    monkeypatch.setattr(utils_push, "send_push_notification",
                        lambda uid, t, b, **k: enviados.append((uid, t, b, k)) or True)
    assert av.avisar("u1", {"id": "g1", "kind": "plan", "plan": "plus", "ends_at": None}) is True
    assert enviados == [("u1", "You have a gift 🎁", "You now have Plus on us.",
                         {"url": "/dashboard", "tag": "regalo-g1"})]

    def _rota(*a, **k):
        raise RuntimeError("sin red")
    monkeypatch.setattr(utils_push, "send_push_notification", _rota)
    assert av.avisar("u1", {"id": "g1", "kind": "plan", "plan": "plus", "ends_at": None}) is False


def test_en_segundo_plano_no_bloquea(monkeypatch):
    hilos = []

    class _Hilo:
        def __init__(self, target, args, daemon, name):
            hilos.append((target, args, daemon))

        def start(self):
            hilos.append("start")
    monkeypatch.setattr(av.threading, "Thread", _Hilo)
    av.avisar_en_segundo_plano("u1", {"id": "g"})
    assert hilos[0][0] is av.avisar and hilos[0][2] is True and hilos[1] == "start"
