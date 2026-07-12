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
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

with open(os.path.join(_ROOT, "frontend", "src", "pages", "AgentPage.jsx"),
          encoding="utf-8") as f:
    _AP = f.read()


def test_stop_button_covers_recovery_phase():
    assert "(isLoading || recoveringTurn) ? (" in _AP, \
        "el ■ debe verse también durante 'Recuperando tu respuesta…'"
    i = _AP.find("const handleStopGeneration")
    win = _AP[i:i + 2400]
    assert "setRecoveringTurn(false)" in win


def test_stop_dismissal_survives_refresh_and_leaves_feedback():
    """[v2] Vivo: 'cuando la detengo y refresco vuelve a estar igual' — el
    descarte vivía en un ref (muere con la página). Ahora se persiste en
    localStorage por firma del huérfano Y el stop deja constancia visible."""
    i = _AP.find("const handleStopGeneration")
    win = _AP[i:i + 2400]
    assert "safeLocalStorageSet(_orphanDismissKey(currentSessionId), _sig)" in win, \
        "el descarte debe sobrevivir al refresh"
    assert "⏹ Detenido" in win, "feedback visible al detener (pedido del owner)"
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
    assert "st.attempts > 6) return" in win, "episodio agotado no renace"
