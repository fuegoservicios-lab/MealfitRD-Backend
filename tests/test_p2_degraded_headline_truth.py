"""[P2-DEGRADED-HEADLINE-TRUTH · 2026-07-31] El banner de calidad degradada
dejaba de acusar a la IA cuando la IA no falló.

CASO REAL (plan d476023a, cuenta gratis del owner): el revisor devolvió
"APROBADO" en el intento #1, la calidad holística fue 0.925 con `retry=1.00` y
`review=1.00`, y aun así el banner decía "La IA no logró un plan óptimo tras 1
intento". Lo que había pasado era otra cosa: una auditoría POSTERIOR al visto
bueno (`P2-PANEL-SOFT-REJECT`) marcó el plan por `micro_worst_day_ceiling` —
el Día 1 se pasó del techo de sodio (gouda en desayuno + almuerzo, camarones de
cena). Acusar al motor de un fallo que no cometió confunde el diagnóstico (yo
mismo leí el banner y busqué el fallo en el sitio equivocado) y erosiona la
confianza en el producto sin motivo.

Los motivos son de dos familias:
  A) La IA no convergió de verdad → agotó intentos / presupuesto / contexto.
  B) La IA entregó y el revisor APROBÓ → una auditoría marcó un detalle.
Sólo la familia A merece el titular de agotamiento.

Parser-based sobre el JSX (no hay runner de JS en CI): ancla la REGLA — que el
titular se derive del motivo y no esté hardcodeado en la vista.
"""
from __future__ import annotations

import re
from pathlib import Path

_DASH = (
    Path(__file__).resolve().parent.parent.parent
    / "frontend" / "src" / "pages" / "Dashboard.jsx"
).read_text(encoding="utf-8")


def test_retry_exhaustion_family_is_explicit():
    m = re.search(r"Q_DEGRADED_RETRY_EXHAUSTION = new Set\(\[(.*?)\]\)", _DASH, re.DOTALL)
    assert m, "falta el set de motivos de agotamiento de intentos"
    cuerpo = m.group(1)
    for motivo in ("high_contextual", "max_attempts", "invalid_pipeline_start",
                   "budget_exhausted"):
        assert motivo in cuerpo, (
            f"'{motivo}' SÍ es agotamiento de la IA y debe llevar ese titular"
        )
    # Los de auditoría post-aprobación NO pueden estar aquí: el plan fue aprobado.
    for motivo in ("micro_worst_day_ceiling", "micro_worst_day", "low_band_score",
                   "high_sodium_sugar", "low_micros"):
        assert motivo not in cuerpo, (
            f"'{motivo}' lo marca una auditoría DESPUÉS de aprobar — no es un "
            f"fallo de convergencia de la IA"
        )


def test_headline_is_derived_not_hardcoded():
    assert "export function resolveQualityDegradedHeadline" in _DASH
    # La frase acusatoria vive SOLO dentro del helper (que decide cuándo aplica).
    # Se cuentan CÓDIGO, no comentarios: los comentarios que explican el arreglo
    # citan la frase y hacían fallar al test contra su propio fix (segunda vez
    # en esta sesión — auditar lo que ejecuta, no la prosa de al lado).
    codigo = re.sub(r"/\*.*?\*/", "", _DASH, flags=re.DOTALL)
    codigo = re.sub(r"^\s*//.*$", "", codigo, flags=re.MULTILINE)
    ocurrencias = codigo.count("La IA no logró un plan óptimo")
    assert ocurrencias <= 2, (
        f"'La IA no logró un plan óptimo' aparece {ocurrencias} veces en código — "
        f"debe vivir sólo dentro de resolveQualityDegradedHeadline (con y sin "
        f"conteo de intentos), nunca escrita en la vista"
    )
    # Y la vista debe llamar al helper, no componer el texto.
    assert "resolveQualityDegradedHeadline(" in _DASH.split("export function resolveQualityDegradedHeadline")[1], (
        "el banner debe consumir el helper"
    )


def test_unknown_reason_net_is_not_inert():
    """La red para motivos NUEVOS (sin clasificar) se mide contra el MAPA.
    Primer intento usaba `resolveQualityDegradedLabel`, que jamás devuelve null
    para un motivo con texto (cae a un genérico) — la red quedaba INERTE y sólo
    se supo EJECUTÁNDOLA: `motivo_desconocido` con 3 intentos salía como "Plan
    listo, con un aviso" en vez de reconocer el agotamiento."""
    # Ventana anclada a la ESTRUCTURA (hasta la siguiente función exportada), no a
    # un conteo de bytes: con `i + 1800` la ventana se derramaba dentro de
    # `resolveQualityDegradedLabel`, cuya propia firma contiene la cadena que este
    # test prohíbe — el test se acusaba a sí mismo.
    i = _DASH.index("export function resolveQualityDegradedHeadline")
    j = _DASH.index("export function resolveQualityDegradedLabel", i)
    win = _DASH[i:j]
    assert "hasOwnProperty.call(Q_DEGRADED_REASON_MAP" in win, (
        "la pertenencia debe consultarse contra el mapa explícito"
    )
    assert "resolveQualityDegradedLabel(reason)" not in win, (
        "usar el resolver aquí deja la red inerte: nunca devuelve null"
    )


def test_notification_shares_the_same_source():
    """La notificación decía 'Plan no óptimo (1 intento)' con el mismo defecto:
    dos superficies contando la misma historia deben leer del mismo sitio."""
    i = _DASH.index("const buildQualityNotification")
    win = _DASH[i:i + 2000]
    assert "resolveQualityDegradedHeadline(" in win, (
        "la notificación debe derivar su título del mismo helper que el banner"
    )
    assert "_head.exhausted" in win, (
        "el título de agotamiento sólo aplica si el helper lo confirma"
    )
