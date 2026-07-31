"""[P1-CANARY-RETRY-ONLY · 2026-07-26] El modelo caro sólo donde el barato ya falló.

## La aritmética que cambia el default

    gpt-5.6-luna   USD 0,102 por plan de 3 días   (11,6× flash por token, medido 2026-07-26)
    flash          USD 0,010

En el **intento 1** se paga en TODOS los planes, incluidos los ~2 de cada 3 que el modelo barato
resuelve bien. Y los datos de hoy no muestran que lo valga: con el contrato de fruta arreglado
(`P1-FRUIT-SEEDER-GATE-CONTRACT`), DeepSeek entregó banda 1.00 sin reintentos.

En el **reintento** sólo llegan los planes donde el barato YA demostró que no pudo, así que el
sobrecoste esperado cae de 0,09 en cada plan a ~0,03 de media. Y hay un argumento de calidad además
del de precio: el reintento llega con una directiva CONCRETA de qué corregir («no repitas fruta»,
«no pongas avena de cena», «usa las proteínas asignadas»). Eso es seguir instrucciones bajo
restricciones — donde un modelo que razona debería rendir más que en la generación libre.

## Lo que NO cambia

El canario sigue **antepuesto**, no sustituyendo: si Luna falla o su circuit breaker abre, la cadena
cae a pro/flash y el plan se genera igual. Y sigue apagado por defecto
(`MEALFIT_DAYGEN_CANARY_MODEL` vacío) — este fix decide DÓNDE se aplica cuando se encienda, no lo
enciende.
"""
import pytest

import graph_orchestrator as go


@pytest.fixture(autouse=True)
def _sin_tier_routing(monkeypatch):
    """[P1-DAYGEN-TIER-MODEL · 2026-07-31] Estos tests anclan el CANARIO sobre
    la cadena base [flash, red]. El routing por tier (Luna primario) tiene su
    ancla en test_p1_daygen_tier_model.py; aquí se neutraliza para que las
    aserciones midan el canario y no el tier/API-key del entorno."""
    monkeypatch.setattr(go, "_daygen_tier_profile", lambda: (None, ""))


@pytest.fixture
def canario_on(monkeypatch):
    monkeypatch.setattr(go, "DAYGEN_CANARY_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(go, "DAYGEN_CANARY_PCT", 100)
    monkeypatch.setattr(go, "DAYGEN_CANARY_SCOPE", "retry")


def test_el_default_es_retry():
    assert go.DAYGEN_CANARY_SCOPE == "retry"


def test_intento_1_NO_paga_el_modelo_caro(canario_on):
    """El 100% de los planes pasa por aquí; ~2 de cada 3 no necesitan más."""
    ch = go._day_model_chain({"user_id": "u"}, 1)
    assert "gpt-5.6-luna" not in ch
    assert ch[0].startswith("deepseek")


@pytest.mark.parametrize("intento", [2, 3, 4])
def test_el_reintento_SI_lo_usa(canario_on, intento):
    ch = go._day_model_chain({"user_id": "u"}, intento)
    assert ch[0] == "gpt-5.6-luna"


def test_el_reintento_conserva_la_red(canario_on):
    """Anteponer, no sustituir: si Luna falla o su CB abre, el plan se genera igual."""
    ch = go._day_model_chain({"user_id": "u"}, 2)
    assert len(ch) >= 2 and any("deepseek" in m for m in ch[1:])


def test_scope_all_restaura_el_comportamiento_anterior(canario_on, monkeypatch):
    """Para poder medir el intento 1 cuando haga falta, sin redeploy."""
    monkeypatch.setattr(go, "DAYGEN_CANARY_SCOPE", "all")
    assert go._day_model_chain({"user_id": "u"}, 1)[0] == "gpt-5.6-luna"


def test_valor_raro_del_knob_cae_al_mas_barato(monkeypatch):
    """Fail-safe: un typo en el env var no debe encender el modelo caro en todos los planes."""
    monkeypatch.setattr(go, "DAYGEN_CANARY_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(go, "DAYGEN_CANARY_PCT", 100)
    monkeypatch.setattr(go, "DAYGEN_CANARY_SCOPE", "TODO-MAL")
    assert "gpt-5.6-luna" not in go._day_model_chain({"user_id": "u"}, 1)


def test_apagado_sigue_apagado_en_los_dos_intentos(monkeypatch):
    monkeypatch.setattr(go, "DAYGEN_CANARY_MODEL", "")
    monkeypatch.setattr(go, "DAYGEN_CANARY_PCT", 0)
    for a in (1, 2):
        assert all(not m.startswith("gpt") for m in go._day_model_chain({"user_id": "u"}, a))


def test_el_knob_esta_registrado():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'MEALFIT_DAYGEN_CANARY_SCOPE' in src
    assert 'DAYGEN_CANARY_SCOPE = ' in src
