"""[P1-CHAT-STOP-POWER · 2026-07-12] Botón Detener universal + "Recuperando" ya no se traba.

Pedido del owner: "quiero el poder para detener el modelo cuando está pensando
como Gemini/OpenAI" + "lo de recuperando respuesta se quedó trabado".

El ■ rojo existía SOLO durante el stream. Ahora cubre las 3 fases:
  1. Stream (como antes — AbortController).
  2. Análisis de foto: el controller nace al inicio del try y su signal viaja
     en el fetch de /diary/upload (gemma 30-90s cancelable).
  3. Recuperación de turno huérfano: stop cancela el episodio (doneSig) y no
     se relanza para el mismo huérfano.

Bug del trabado: el poll contaba filas CRUDAS del server (títulos de sistema
incluidos) → rehidratación prematura SIN respuesta → el huérfano renacía y
attempts se reseteaba en bucle infinito. Fix: filtrado como display + exigir
último=model + firma de episodio (mismo huérfano = mismos intentos).
tooltip-anchor: P1-CHAT-STOP-POWER
"""

import pytest
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

@pytest.fixture(scope="module", autouse=True)
def _load_frontend_sibling_sources(frontend_repo_path):
    # La fixture compartida salta el módulo antes de cualquier I/O si falta el hermano.
    _ = frontend_repo_path
    global _AP, f
    with open(os.path.join(_ROOT, "frontend", "src", "pages", "AgentPage.jsx"),
              encoding="utf-8") as f:
        _AP = f.read()



def test_stop_button_covers_recovery_phase():
    """[actualizado P1-CHAT-TURN-ACTIVE · 2026-08-10] Antes este caso afirmaba la
    cadena literal `(isLoading || recoveringTurn) ? (`. Su INTENCIÓN —que el ■ rojo
    cubra todas las fases en que hay algo que detener— se cumple ahora MEJOR, y por
    eso el literal cambia en vez de conservarse.

    `isLoading` se apaga en el PRIMER token, así que el gate viejo hacía desaparecer
    el botón justo al empezar a escribirse la respuesta: la fase más larga del turno
    y la que el usuario más quiere poder cortar. `isTurnActive` dura hasta el
    `finally`, o sea hasta que el turno termina de verdad."""
    assert "(isTurnActive || recoveringTurn) ? (" in _AP, \
        "el ■ debe verse durante TODO el turno (incluido el streaming) y la recuperación"
    assert "(isLoading || recoveringTurn) ? (" not in _AP, \
        "volvió el gate que ocultaba el ■ en cuanto llegaba el primer token"
    i = _AP.find("const handleStopGeneration")
    win = _AP[i:i + 2400]
    assert "setRecoveringTurn(false)" in win


def test_un_turno_en_vuelo_bloquea_el_siguiente():
    """[P1-CHAT-TURN-ACTIVE · 2026-08-10] `isLoading` significa «está pensando», no
    «hay un turno en vuelo»: deja de ser cierto en el primer token. Como gobernaba
    también el guard de entrada, desde ese instante se podía lanzar un SEGUNDO stream
    que escribe sobre la MISMA burbuja que el primero sigue llenando — conversación
    corrupta y sin aviso.

    El guard lee el REF y no el state a propósito: dos toques dentro del mismo frame
    de React verían ambos el valor viejo (misma lección que P1-FORM-4)."""
    i = _AP.find("const handleSend = async")
    win = _AP[i:i + 1800]
    assert re.search(r"\|\|\s*isTurnActiveRef\.current\)\s*return;", win), \
        "el guard de entrada debe mirar el turno, no el 'pensando'"

    # Encendido único y apagado único: si alguien apaga el turno en una rama
    # concreta del stream, el hueco se reabre por esa puerta.
    assert _AP.count("_setTurnActive(true)") == 1, \
        "el turno debe encenderse en un solo sitio (handleSend)"
    i_fin = _AP.find("} finally {", i)
    assert i_fin > 0 and "_setTurnActive(false)" in _AP[i_fin:i_fin + 400], \
        "el apagado autoritativo va en el finally: cubre done, error, abort y excepción"

    # «Nuevo chat» era el único camino sin guard alguno.
    i_new = _AP.find("const handleNewChat")
    assert "isTurnActiveRef.current" in _AP[i_new:i_new + 700], \
        "abrir un chat nuevo a mitad de un turno debe cortar el stream anterior"


def test_detener_cierra_la_burbuja():
    """Detener es TERMINAR el turno, no dejarlo colgado: si la burbuja se queda con
    `isStreaming: true`, nunca ofrece Copiar/Regenerar sobre lo que sí llegó y el
    efecto de caché se salta la persistencia."""
    i = _AP.find("const handleStopGeneration")
    win = _AP[i:i + 1600]
    assert "isStreaming: false" in win, \
        "el stop debe cerrar la burbuja en curso"


def test_stop_dismissal_survives_refresh_and_leaves_feedback():
    """[v2] Vivo: 'cuando la detengo y refresco vuelve a estar igual' — el
    descarte vivía en un ref (muere con la página). Ahora se persiste en
    localStorage por firma del huérfano Y el stop deja constancia visible."""
    i = _AP.find("const handleStopGeneration")
    win = _AP[i:i + 4200]
    assert "safeLocalStorageSet(_orphanDismissKey(currentSessionId), _sig)" in win, \
        "el descarte debe sobrevivir al refresh"
    # [P2-CHAT-ERROR-MINIMAL · 2026-09-04] copias cortas sin emojis: el feedback
    # es el `content` de la burbuja de cierre («Detenido. …»), ya sin el «⏹».
    assert re.search(r"content:\s*t\('Detenido", win), \
        "feedback visible al detener (pedido del owner)"
    assert "_stoppedByUser: true" in win
    # El efecto respeta el descarte persistido:
    assert "safeLocalStorageGet(_orphanDismissKey(currentSessionId), null) === _sig" in _AP
    # La firma ignora burbujas locales (estable ante refetches que las quitan):
    assert "!m._isErrorBubble && !m._stoppedByUser" in _AP
    # El agotamiento también persiste:
    assert _AP.count("safeLocalStorageSet(_orphanDismissKey(currentSessionId)") >= 2


def test_stop_bubble_survives_server_rehydration():
    """[v3] Vivo: 'el mensaje de que está detenido desapareció' — la burbuja es
    CLIENT-ONLY y el replace del refetch la borraba. fetchSessionMessages la
    RECONSTRUYE desde el marcador persistente (fuente de verdad)."""
    i = _AP.find("Reconstruir la burbuja")
    assert i != -1, "la reconstrucción post-rehidratación desapareció"
    win = _AP[i:i + 1500]
    assert "_orphanDismissKey(sessionId)" in win and "_orphanSig(_mappedMsgs)" in win, \
        "la burbuja se reconstruye SOLO si el descarte persistido matchea la firma"
    assert "_lastMapped.role === 'user'" in win, \
        "solo cuando el último real es del user (turno detenido sin respuesta)"
    assert "⏹ Detenido" in win


def test_stop_covers_photo_analysis():
    # El controller nace ANTES del upload y su signal viaja en el fetch.
    i = _AP.find("El AbortController nace ANTES")
    assert i != -1
    j = _AP.find("signal: controller.signal", i)
    assert j != -1 and j - i < 4000, \
        "el fetch de /diary/upload debe llevar el signal del controller"


def test_recovery_episode_signature_no_infinite_loop():
    i = _AP.find("Firma del episodio")
    assert i != -1
    win = _AP[i:i + 1200]
    assert "st.doneSig === _sig) return" in win, "huérfano descartado no relanza"
    assert "st.sig !== _sig" in win, \
        "attempts solo se resetean para un huérfano NUEVO (no en bucle)"
    assert "st.attempts > 30) return" in win, "episodio agotado no renace"
