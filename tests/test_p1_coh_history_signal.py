"""[P1-COH-HISTORY-SIGNAL · 2026-08-14] El historial del guard deja de tirar la señal.

`_shopping_coherence_block_history` es el registro post-mortem del coherence guard: lo
que un operador abre para responder «¿este plan tuvo una incoherencia de verdad?».
Tiene cap 20 y era FIFO puro.

Medido en producción (journal del VPS, 7 días):

    395  COH-GUARD/block
    302  de ellos con hipótesis EXCLUSIVAMENTE `recipe_unquantified`  (sal, pimienta,
         comino: el LLM escribió «al gusto» y no puso gramos)
    234  COH-HISTORY-TRUNCATED

O sea: el registro se llenaba de condimentos sin cuantificar y, al llenarse, EXPULSABA
las entradas accionables — un `cap_swallowed_modifier` es «la receta pide pollo y la
lista no lo trae», exactamente lo que el guard existe para cazar. *El registro que
existe para diagnosticar acababa guardando veinte veces «sal al gusto».*

Este fix NO cambia la severidad (esas divergencias ya son no-severas desde
P1-COHERENCE-INF-NOT-SEVERE) ni lo que se reporta: cambia a QUIÉN se desaloja cuando el
cap aprieta. Primero el ruido más antiguo; si no hay bastante, FIFO como siempre.
"""
from __future__ import annotations

import graph_orchestrator as go


def _entrada(hipotesis: dict, ts: str = "2026-08-14T00:00:00+00:00") -> dict:
    return {
        "ts": ts,
        "attempt": 1,
        "divergence_count": sum(hipotesis.values()),
        "hypotheses": dict(hipotesis),
        "block_set": True,
        "action_taken": None,
    }


_RUIDO = {"recipe_unquantified": 2}
_SENAL = {"cap_swallowed_modifier": 1}


def test_el_ruido_se_reconoce_y_la_senal_no():
    assert go._entrada_de_historial_es_ruido(_entrada(_RUIDO)) is True
    assert go._entrada_de_historial_es_ruido(_entrada(_SENAL)) is False
    # mezcla: si hay UNA accionable, la entrada entera es señal
    assert go._entrada_de_historial_es_ruido(
        _entrada({"recipe_unquantified": 3, "cap_swallowed_modifier": 1})
    ) is False


def test_ante_la_duda_una_entrada_cuenta_como_senal():
    """Conservador por diseño: perder una entrada accionable es el fallo caro."""
    assert go._entrada_de_historial_es_ruido({}) is False
    assert go._entrada_de_historial_es_ruido({"hypotheses": {}}) is False
    assert go._entrada_de_historial_es_ruido({"hypotheses": None}) is False
    assert go._entrada_de_historial_es_ruido("no soy un dict") is False
    assert go._entrada_de_historial_es_ruido(
        _entrada({"hipotesis_que_nadie_ha_visto": 1})
    ) is False


def test_el_desalojo_tira_el_ruido_y_conserva_la_senal():
    """El caso real: una entrada accionable antigua rodeada de sal y pimienta."""
    historial = [_entrada(_SENAL, ts="T0")] + [_entrada(_RUIDO, ts=f"T{i}") for i in range(1, 25)]
    recortado = go._desalojar_priorizando_ruido(historial, 20)

    assert len(recortado) == 20
    assert any(h["hypotheses"] == _SENAL for h in recortado), (
        "se desalojó la ÚNICA entrada accionable teniendo 24 de ruido para tirar — "
        "es exactamente el fallo que este fix cierra"
    )
    assert recortado[-1]["ts"] == "T24", "la entrada recién añadida nunca se desaloja"


def test_la_entrada_recien_anadida_nunca_se_desaloja():
    """[mutación M3, 2026-08-14] El test de arriba AFIRMABA esto y no lo verificaba:
    con 24 entradas de ruido delante, el desalojo se sacia mucho antes de llegar a la
    última, así que mutar el rango a `range(len(historial))` no rompía nada. Este caso
    lo expone: la nueva es la ÚNICA de ruido y todo lo demás es señal, así que un
    desalojo que la considere candidata la elegirá A ELLA — y el evento que se acaba de
    registrar desaparecería del registro que existe para registrarlo."""
    historial = [_entrada(_SENAL, ts="s1"), _entrada(_SENAL, ts="s2"), _entrada(_RUIDO, ts="nueva")]
    recortado = go._desalojar_priorizando_ruido(historial, 2)
    assert len(recortado) == 2
    assert recortado[-1]["ts"] == "nueva", (
        f"se desalojó la entrada recién añadida: {[h['ts'] for h in recortado]}"
    )
    assert recortado[0]["ts"] == "s2", "debió caer la señal MÁS ANTIGUA, no la reciente"


def test_sin_ruido_suficiente_cae_al_FIFO_de_siempre():
    """Si todo es señal, el comportamiento no cambia: se van los más antiguos."""
    historial = [_entrada(_SENAL, ts=f"T{i}") for i in range(25)]
    recortado = go._desalojar_priorizando_ruido(historial, 20)
    assert len(recortado) == 20
    assert [h["ts"] for h in recortado] == [f"T{i}" for i in range(5, 25)]


def test_el_helper_publico_aplica_la_politica():
    """`_apply_coherence_history_cap` es el SSOT que usan los dos call sites."""
    previo = [_entrada(_SENAL, ts="viejo-accionable")] + [
        _entrada(_RUIDO, ts=f"ruido-{i}") for i in range(30)
    ]
    nuevo = go._apply_coherence_history_cap(previo, _entrada(_RUIDO, ts="nueva"))
    cap = go._coherence_block_history_cap()
    assert len(nuevo) == cap
    assert nuevo[-1]["ts"] == "nueva"
    assert any(h["ts"] == "viejo-accionable" for h in nuevo), (
        "el helper SSOT no está aplicando la prioridad: la entrada accionable murió"
    )


def test_no_se_pierde_nada_si_el_historial_cabe():
    previo = [_entrada(_RUIDO, ts=f"T{i}") for i in range(3)]
    nuevo = go._apply_coherence_history_cap(previo, _entrada(_SENAL, ts="ultima"))
    assert len(nuevo) == 4 and nuevo[-1]["ts"] == "ultima"
