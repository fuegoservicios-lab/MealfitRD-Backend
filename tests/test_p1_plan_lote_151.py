# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-151 · 2026-09-21] El aviso que llega ANTES no puede abrir preguntando si ya comiste.

El lote 150 movió el recordatorio a 15 min ANTES de la hora habitual y le dijo al prompt que animara. Pero le dejó
una salida escrita: «si aun así necesitas preguntar, el verbo sigue siendo el suyo: "¿Ya {verbo}?"». Medido en
producción el mismo día (21-sep, 11:30 local, usuario del dueño): el mensaje que salió empezaba por
«¿Ya desayunaste? Si aún no, este es el momento ideal…». El modelo tomó la excepción como la norma.

*Una excepción escrita en un prompt no es una excepción: es una opción, y el modelo la elige.* Si de verdad no
quieres una conducta, el prompt tiene que prohibirla, no permitirla «solo si hace falta».

El verbo propio de cada comida (lote 73) se conserva: sigue en el prompt, ahora dentro de la prohibición, así que
el ancla de aquel test sigue hablando del mismo contrato.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(nombre: str) -> str:
    return (_BACKEND / nombre).read_text(encoding="utf-8")


def test_el_prompt_prohibe_preguntar_si_ya_comio():
    prompt = _src("prompts/proactive.py")
    assert "PROHIBIDO abrir preguntando «¿Ya {verbo}?»" in prompt
    # la salida que el modelo usaba como norma ya no está
    assert "Si aun así necesitas preguntar" not in prompt
    assert "si aún así necesitas preguntar" not in prompt.lower()


def test_sigue_mandando_animar_con_el_verbo_de_esa_comida():
    """Lo del lote 73 no se pierde al cerrar la salida: el verbo sigue siendo el de SU comida."""
    prompt = _src("prompts/proactive.py")
    assert "«{infinitivo}»" in prompt
    assert "nunca el de otra" in prompt
    assert "anímale a comer ahora" in prompt
    # y no vuelve a dar por hecho que ya comió
    assert "No des por hecho que ya comió" not in prompt or "PROHIBIDO" in prompt


@pytest.mark.parametrize("comida,infinitivo", [
    ("Desayuno", "desayunar"), ("Almuerzo", "almorzar"), ("Merienda", "merendar"), ("Cena", "cenar"),
])
def test_el_prompt_se_arma_sin_huecos(comida, infinitivo):
    """`.format()` con todas las claves: un placeholder nuevo sin su argumento reventaría el cron a las 6 a.m."""
    import proactive_agent as pa
    from prompts.proactive import PROACTIVE_PROMPT

    assert pa.INFINITIVO_DE_COMIDA[comida] == infinitivo
    # Se rellenan TODOS los huecos que declare la plantilla, no una lista a mano: así, si mañana nace otro
    # placeholder, este test sigue midiendo lo suyo en vez de romperse por un `KeyError` ajeno.
    import string

    claves = {n for _l, n, _s, _c in string.Formatter().parse(PROACTIVE_PROMPT) if n}
    valores = {k: "" for k in claves}
    valores.update(
        missing_meal=comida,
        verbo=pa.VERBO_DE_COMIDA[comida],
        infinitivo=infinitivo,
        trigger_time="8:45 AM",
    )
    texto = PROACTIVE_PROMPT.format(**valores)
    assert "{" not in texto.replace("{{", "").replace("}}", ""), "quedó un placeholder sin rellenar"
    assert infinitivo in texto


def test_el_aviso_no_puede_salirse_por_arriba_de_la_franja_de_su_comida(monkeypatch):
    """Medido en la cuenta del dueño el 21-sep: almuerzo 15:06 y merienda 15:45 — 39 min de diferencia.

    La hora sale de la MEDIA de lo que registra, y con 4-5 muestras una comida tardía la arrastra fuera de rango.
    El dueño eligió acotar. Comprobado con sus números reales: 15,35 de media ⇒ aviso 14:15, y la merienda queda
    a hora y media, no a media hora.

    [P1-PLAN-LOTE-220] El acotado vive ahora en el camino por historial (knob `MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY`):
    por defecto la hora ya no sale de lo registrado."""
    import db_facts
    import proactive_agent as pa

    monkeypatch.setenv("MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY", "1")
    monkeypatch.setattr(pa, "get_nudge_response_rate", lambda _u, _m=None: (1.0, 0))
    monkeypatch.setattr(db_facts, "get_avg_meal_hour", lambda _u, _m, ventana=None: 15.35)
    assert pa.hora_de_aviso("u", "Almuerzo", 13.0)[0] == pytest.approx(14.25)
    # dentro de la franja no se toca nada
    monkeypatch.setattr(db_facts, "get_avg_meal_hour", lambda _u, _m, ventana=None: 13.8)
    assert pa.hora_de_aviso("u", "Almuerzo", 13.0)[0] == pytest.approx(13.55)


def test_el_lado_TEMPRANO_no_se_acota_a_proposito(monkeypatch):
    """Acotar por abajo devolvería el defecto que el lote 150 vino a cerrar: el aviso DESPUÉS de la comida.

    Quien desayuna de verdad a las 7:05 sigue recibiéndolo a las 6:50, no a las 7:15. El acotado, por diseño, solo
    puede ADELANTAR un aviso; nunca retrasarlo. [P1-PLAN-LOTE-220] Camino por historial, detrás de su knob."""
    import db_facts
    import proactive_agent as pa

    monkeypatch.setenv("MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY", "1")
    monkeypatch.setattr(pa, "get_nudge_response_rate", lambda _u, _m=None: (1.0, 0))
    monkeypatch.setattr(db_facts, "get_avg_meal_hour", lambda _u, _m, ventana=None: 7.083)
    assert pa.hora_de_aviso("u", "Desayuno", 9.0)[0] == pytest.approx(6.833, abs=1e-3)


def test_la_distancia_se_mide_en_el_reloj_no_restando():
    """Para una cena de las 19:30, las 00:06 están 4,6 h DESPUÉS, no 19,4 h antes.

    Restar daría un número negativo enorme, el acotado no se aplicaría y el aviso se quedaría de madrugada — que es
    exactamente el caso del lote 83."""
    import proactive_agent as pa

    assert pa._acotar_a_su_franja(0.1, 19.5, "Cena") == pytest.approx(21.0)
    assert pa._acotar_a_su_franja(23.0, 19.5, "Cena") == pytest.approx(21.0)
    assert pa._acotar_a_su_franja(18.0, 19.5, "Cena") == pytest.approx(18.0), "temprano: intacto"
    assert pa._acotar_a_su_franja(2.0, 19.5, "Cena") == pytest.approx(21.0), "las 2:00 son 6,5 h después"


def test_la_banda_es_un_knob_y_abrirla_devuelve_la_conducta_anterior(monkeypatch):
    import proactive_agent as pa

    assert pa._banda_del_aviso_h() == pytest.approx(1.5)
    monkeypatch.setenv("MEALFIT_PROACTIVE_NUDGE_BAND_H", "12")
    assert pa._acotar_a_su_franja(15.35, 13.0, "Almuerzo") == pytest.approx(15.35)
    monkeypatch.setenv("MEALFIT_PROACTIVE_NUDGE_BAND_H", "0.1")   # fuera del clamp [0.5, 12]
    assert pa._banda_del_aviso_h() == pytest.approx(1.5), "un valor imposible cae al default, no rompe el cron"


def test_el_marcador_va_con_su_lote():
    app = _src("app.py")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"P1-PLAN-LOTE-(\d+) · \d{4}-\d{2}-\d{2}"', app)
    assert m, "el marcador cambió de forma"
    assert int(m.group(1)) >= 151, "el marcador nunca baja"
