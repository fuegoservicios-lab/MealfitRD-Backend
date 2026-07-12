"""[P2-CHAT-HISTORY-CLEAN · 2026-07-12] El historial del chat muestra al usuario SU texto, no el prompt interno.

Vivo (owner): la burbuja con su foto "apareció unos segundos y cuando cargó la
respuesta del agente desapareció y se volvió texto" — texto que era el PROMPT
ENRIQUECIDO crudo ([Sistema: El usuario subió una foto de ALIMENTOS SUELTOS...]
+ Instrucción: ... completa).

Dos causas encadenadas en AgentPage.fetchSessionMessages:
  1. El refetch del historial corría con el turno EN VUELO (sus deps cambian
     cuando el contexto refresca post-tool) y pisaba el estado local — que iba
     ADELANTE del server — con lo persistido crudo. Guard: isLoadingRef.
  2. El limpiador de display solo conocía 2 variantes VIEJAS de wrapper; las
     nuevas de P1-CHAT-VISION-GEMMA (items/otro/caído y el kindHint con
     paréntesis) no matcheaban → prompt interno visible. Detector único ancho.

Arquitectura (decidida, no accidental): el historial PERSISTE el prompt
completo — el LLM necesita el análisis de la foto en turnos futuros ("agrégalo
a la nevera" refiere al análisis previo). La limpieza es SOLO de display.
tooltip-anchor: P2-CHAT-HISTORY-CLEAN
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

with open(os.path.join(_ROOT, "frontend", "src", "pages", "AgentPage.jsx"),
          encoding="utf-8") as f:
    _AP = f.read()


def _fetch_block():
    i = _AP.find("const fetchSessionMessages = useCallback")
    assert i != -1, "fetchSessionMessages desapareció"
    return _AP[i:i + 9000]


def test_refetch_never_clobbers_inflight_turn():
    blk = _fetch_block()
    assert "isLoadingRef.current" in blk, \
        "sin este guard, el refetch pisa la burbuja recién enviada con el prompt crudo"
    # El ref espejo debe existir y seguirse del state.
    assert "isLoadingRef.current = isLoading" in _AP


def test_wrapper_detector_covers_all_variants():
    blk = _fetch_block()
    assert "acaba de subir|subió" in blk and "una imagen|una foto" in blk, \
        "el detector único debe cubrir plato/items/otro/caído — no solo las 2 variantes viejas"
    assert "Instrucción:[\\s\\S]*$" in blk, \
        "la cola 'Instrucción: ...' del prompt interno no debe mostrarse jamás"
    assert "Mensaje del usuario:\\s*([\\s\\S]*)$" in blk, \
        "si el usuario escribió texto junto a la foto, se muestra SOLO su texto"


def test_local_thumbs_survive_rehydration():
    blk = _fetch_block()
    assert "_canMergeThumbs" in blk and "_localUsers" in blk, \
        "el server no persiste imágenes (sin storage): los thumbs dataURL locales se conservan"
    assert "startsWith('blob:')" in blk, \
        "los blob: muertos no se re-inyectan — solo dataURL/URLs estables"
