"""[P1-CHAT-REFRESH-RECOVER · 2026-07-12] Refresh a mitad de respuesta ya no deja el chat mudo.

Vivo (owner): envió "actualiza el desayuno", refrescó la página y el chat
quedó con su mensaje huérfano — sin indicador "pensando", sin respuesta y sin
retry, para siempre (el estado del turno murió con la página; el server pudo
o no haber completado).

Recuperación en AgentPage: al detectar "último mensaje = user y nada en vuelo"
→ (1) indicador 'Recuperando tu respuesta…', (2) sondeo del historial cada 4s
(máx 6 intentos) que solo REHIDRATA cuando el server va igual o adelante del
estado local (no pierde la burbuja huérfana), (3) al agotar, burbuja retryable
que reenvía el prompt. Mismo espíritu que P1-SWAP-REGEN-RESUME.
tooltip-anchor: P1-CHAT-REFRESH-RECOVER
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

with open(os.path.join(_ROOT, "frontend", "src", "pages", "AgentPage.jsx"),
          encoding="utf-8") as f:
    _AP = f.read()


def _block():
    i = _AP.find("P1-CHAT-REFRESH-RECOVER")
    assert i != -1, "el bloque de recuperación desapareció"
    return _AP[i:i + 5200]


def test_orphan_detection_and_indicator():
    blk = _block()
    assert "last.role === 'user' && !isLoading && !isLoadingHistory" in blk, \
        "huérfano = último mensaje del user sin nada en vuelo"
    assert "(isLoading || recoveringTurn) && (" in _AP, \
        "el indicador pensando también se muestra durante la recuperación"
    assert "Recuperando tu respuesta…" in _AP


def test_poll_is_conservative():
    blk = _block()
    # [P1-CHAT-STOP-POWER v2] El conteo filtra filas de sistema y exige que el
    # ÚLTIMO sea del modelo — contar filas crudas rehidrataba prematuro sin
    # respuesta y el episodio renacía en bucle ("Recuperando" trabado, vivo).
    assert "srv.length >= localCount + 1" in blk, \
        "solo rehidratar cuando el server va ADELANTE (filtrado)"
    assert "srvLast.role === 'model'" in blk, \
        "rehidratar SOLO si hay respuesta real del modelo al final"
    assert "cur.attempts > 6" in blk, "el sondeo tiene tope (~26s), no es infinito"


def test_timeout_leaves_retryable_bubble():
    blk = _block()
    assert "retryPrompt: canRetry ? lastPrev.content : null" in blk
    assert "_isErrorBubble: true" in blk
    assert "!lastPrev.isImage" in blk, \
        "los huérfanos con foto NO auto-reintentan (el dataURL contaminaría el prompt) — se pide reenviar"


def test_cleanup_on_unmount():
    blk = _block()
    assert "clearTimeout(st.timer)" in blk, "el timer no debe sobrevivir al unmount"
