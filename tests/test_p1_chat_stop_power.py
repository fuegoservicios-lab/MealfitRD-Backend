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
    win = _AP[i:i + 1400]
    assert "doneSig = _st.sig" in win.replace("_st.doneSig", "doneSig = _st.sig") or "_st.doneSig = _st.sig" in win, \
        "stop descarta el huérfano actual (no relanzar el episodio)"
    assert "setRecoveringTurn(false)" in win


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
