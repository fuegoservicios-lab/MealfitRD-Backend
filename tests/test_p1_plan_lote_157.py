"""[P1-PLAN-LOTE-157 · 2026-09-22] El turno mudo tiene techo, y el knob que decía acotarlo dice
la verdad sobre lo que hace.

TERCERA PASADA sobre el coach. Dos hallazgos, ninguno de ellos un fallo de lógica — los dos son
**defensas que se anuncian y no están**, que es la familia que este repo lleva meses cerrando.

1 · NINGÚN TECHO MIENTRAS EL SERVIDOR ESTÁ MUDO. El presupuesto total del stream (120 s) se
    comprueba al principio del `for event in stream_iter`, así que **solo corre cuando llega un
    evento**. Es la misma limitación que `P2-CHAT-STREAM-INACTIVITY-POSTHOC` documentó para el
    chequeo de inactividad, y que allí se aceptó a propósito. Consecuencia: ante un cuelgue de
    verdad (cero eventos) no hay nada que lo dispare. Y el cliente tampoco tenía tope: su
    `fetchWithAuth` limpia el temporizador en cuanto llegan las cabeceras —correcto, si no
    rompería el SSE—, de modo que el usuario podía quedarse en «Pensando…» hasta rendirse.
    El techo vive ahora en el CLIENTE (`utils/silencioDelStream.js`, 5 min de silencio absoluto),
    que es el único lado que puede mirar el reloj mientras el otro no habla.

2 · UN KNOB QUE PROMETE LO QUE YA NO HACE. `MEALFIT_CHAT_STREAM_INACTIVITY_TIMEOUT_S` decía en su
    comentario «abortamos el stream». Era cierto hasta el 14-sep. Desde entonces solo decide
    cuándo se registra un WARNING — y el knob se auto-registra en `_KNOBS_REGISTRY` y se publica
    en `/health/version`, así que un operador podía subirlo esperando alargar una protección que
    ya no existe.

      *Un knob que describe una conducta retirada es peor que no tenerlo: promete una defensa y
      además consume la atención de quien la busca.*

Por qué el techo del cliente no es el error que el POSTHOC deshizo: allí se abortaba por un hueco
YA PASADO (la llegada del evento probaba que el turno estaba vivo) y con 25 s de umbral. Aquí se
corta solo tras 5 minutos de silencio ABSOLUTO, cualquier evento reinicia el reloj, y —esto es lo
que lo hace seguro— el corte empuja un `502`, que desde el lote 156 es la puerta del rescate: si
el servidor terminó igual, el cliente adopta la respuesta en vez de perderla.

Tooltip-anchor: P1-PLAN-LOTE-157
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AGENT = _BACKEND_ROOT / "agent.py"
_FRONT = _BACKEND_ROOT.parent / "frontend"


def _inactivity_block() -> str:
    """El comentario del knob de inactividad + su función."""
    src = _AGENT.read_text(encoding="utf-8")
    ini = src.find("# [P1-CHAT-STREAM-INACTIVITY")
    assert ini != -1, "Desapareció el bloque del knob de inactividad."
    fin = src.find("def _chat_stream_inactivity_timeout_s", ini)
    assert fin > ini
    return src[ini:fin]


def test_el_knob_de_inactividad_no_dice_que_aborta():
    """Decía «abortamos el stream» y no aborta desde el 14-sep."""
    bloque = _inactivity_block()
    assert "abortamos el stream" not in bloque, (
        "El comentario volvió a prometer que aborta. No lo hace: la comprobación solo corre "
        "cuando llega un evento (P2-CHAT-STREAM-INACTIVITY-POSTHOC)."
    )


def test_el_knob_de_inactividad_dice_lo_que_SI_hace():
    """Que el siguiente operador lo lea y sepa que subirlo no alarga ninguna protección."""
    bloque = _inactivity_block()
    assert "WARNING" in bloque, "Tiene que decir que hoy solo gobierna un registro."
    assert "solo corre cuando LLEGA un evento" in bloque, (
        "La razón importa más que el hecho: sin ella alguien lo «arregla» reintroduciendo el "
        "aborto post-hoc que mataba turnos vivos."
    )


def test_el_aborto_post_hoc_sigue_sin_existir():
    """La regresión concreta que no puede volver: cortar el turno por un hueco ya pasado."""
    src = _AGENT.read_text(encoding="utf-8")
    ini = src.find("if _gap_since_last > _stream_inactivity_budget:")
    assert ini != -1, "Desapareció el chequeo de inactividad."
    cuerpo = src[ini:ini + 900]
    assert "logger.warning" in cuerpo, "Hoy es telemetría."
    assert "raise TimeoutError" not in cuerpo, (
        "Volvió a abortar por inactividad: eso mataba turnos vivos con el trabajo ya persistido."
    )


def test_el_cliente_tiene_su_propio_techo_de_silencio():
    """El único reloj que corre mientras el servidor no habla."""
    util = _FRONT / "src" / "utils" / "silencioDelStream.js"
    if not util.exists():
        pytest.skip(f"El árbol del frontend no está junto a este backend ({util}).")
    src = util.read_text(encoding="utf-8")
    assert "SILENCIO_MAXIMO_MS = 300_000" in src, (
        "El umbral baja de 5 min: una tool legítima calla 2-4 y volveríamos a cortar turnos vivos."
    )
    assert "hayQueCortarPorSilencio" in src


def test_el_techo_del_cliente_se_apoya_en_el_rescate_del_156():
    """Cortar sin rescate sería cambiar una espera infinita por una pérdida.

    El backend persiste la respuesta en su `done` aunque el cliente se haya ido; el 502 del corte
    es lo que hace que el cliente vaya a buscarla.
    """
    agente = _FRONT / "src" / "pages" / "AgentPage.jsx"
    rescate = _FRONT / "src" / "utils" / "rescateDelTurno.js"
    if not agente.exists() or not rescate.exists():
        pytest.skip("El árbol del frontend no está junto a este backend.")
    src = agente.read_text(encoding="utf-8")
    ini = src.find("if (error.name === 'AbortError') {")
    assert ini != -1
    bloque = src[ini:ini + 1400]
    assert "_cortadoPorSilencio" in bloque and "status: 502" in bloque, (
        "El corte del vigilante tiene que pintar un 502 (la puerta del rescate), no callarse "
        "como el botón Detener."
    )
    assert "ESTADOS_DE_CORTE = [0, 502]" in rescate.read_text(encoding="utf-8")


def test_el_presupuesto_total_sigue_siendo_la_capa_del_servidor():
    """No se toca: lo que se añade es un techo en el cliente, no otro en el servidor."""
    src = _AGENT.read_text(encoding="utf-8")
    assert re.search(r'MEALFIT_CHAT_STREAM_TOTAL_TIMEOUT_S"?,\s*\n?\s*120\.0', src), (
        "El presupuesto total del stream cambió de valor sin querer."
    )
