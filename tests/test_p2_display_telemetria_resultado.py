"""[P2-DISPLAY-SIN-TELEMETRIA-RESULTADO · 2026-08-21] El enriquecimiento instrumentaba
lo que GASTA y nada de lo que PASA.

`_emit_usage_telemetry` escribe el coste en `llm_usage_events` desde el día uno. Pero el
módulo entero tenía CERO referencias a `pipeline_metrics`, CERO a `system_alerts` y CERO
`logger.error`. Así que un enriquecimiento descartado entero —JSON malformado, lote
pasado del tope de salida, todos los meals con mismatch TOCTOU— era indistinguible de
uno que nunca se disparó: el usuario ve su plan en español y en el servidor no queda
rastro de por qué.

Y el nivel de log no es un detalle de estilo: con Sentry en `DEFAULT_EVENT_LEVEL=ERROR`,
un `logger.warning` no sube. Elegir `warning` para una degradación es decidir que nadie
se entere.

QUÉ ANCLA. Que exista la emisión de RESULTADO además de la de coste, que se llame
SIEMPRE (no sólo en el camino feliz), que sea best-effort de verdad —un fallo suyo no
puede tumbar el enriquecimiento— y que el caso «cero escrituras habiendo despachado
lotes» salga por `error`.

Es parser-based a propósito: montar el ciclo completo exige el LLM stubbeado y el pool
de DB, y lo que hay que fijar aquí es estructural.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_MOD = _BACKEND / "plan_display_i18n.py"

_MARKER = "P2-DISPLAY-SIN-TELEMETRIA-RESULTADO"


def _fuente() -> str:
    if not _MOD.exists():
        pytest.skip(f"{_MOD} no existe en este checkout")
    return _MOD.read_text(encoding="utf-8")


def _sin_comentarios(src: str) -> str:
    """Sólo líneas de código.

    Este repo tiene siete precedentes documentados en agosto de «un comentario derrotó
    al guard», y esta clase de test —«esto TIENE que aparecer»— es la que se los come al
    revés: un nombre citado en prosa daría el guard por satisfecho con el código borrado.
    """
    return "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))


def test_existe_una_emision_de_resultado_ademas_de_la_de_coste() -> None:
    codigo = _sin_comentarios(_fuente())
    assert "pipeline_metrics" in codigo, (
        "El módulo sigue sin escribir a `pipeline_metrics`. Un enriquecimiento "
        "descartado entero es indistinguible de uno que nunca se disparó. "
        f"[{_MARKER}]"
    )
    assert "def _emit_result_telemetry" in codigo, (
        f"falta el emisor de resultado [{_MARKER}]"
    )
    assert "def _emit_usage_telemetry" in codigo, (
        "desapareció el emisor de COSTE. Son complementarios, no alternativos: uno "
        f"mide lo que se gasta y el otro lo que pasa. [{_MARKER}]"
    )


def test_la_emision_de_resultado_corre_en_el_camino_de_fallo_tambien() -> None:
    """Si sólo se emitiera al escribir algo, el silencio volvería a significar dos
    cosas distintas — que es justo el defecto."""
    codigo = _sin_comentarios(_fuente())
    m = re.search(r"_emit_result_telemetry\(plan_id, user_id, locale, [^)]*\)", codigo)
    assert m, f"no encuentro la llamada al emisor de resultado [{_MARKER}]"
    resto = codigo[m.end():]
    # La llamada tiene que preceder a los DOS returns finales del ciclo, no colgar de
    # la rama del éxito.
    assert resto.count('return {"enriched_meals"') >= 2, (
        "La emisión de resultado quedó DESPUÉS de algún return, así que no cubre "
        f"todos los desenlaces. [{_MARKER}]"
    )


def test_es_best_effort() -> None:
    """Nunca puede tumbar el enriquecimiento: la telemetría es para saber qué pasó,
    no una razón más para que no pase."""
    src = _fuente()
    cuerpo = src[src.index("def _emit_result_telemetry"):]
    cuerpo = cuerpo[: cuerpo.index("\ndef ", 1)]
    assert "try:" in cuerpo and "except Exception" in cuerpo, (
        f"el emisor de resultado no está envuelto en un try/except [{_MARKER}]"
    )


def test_cero_escrituras_con_lotes_despachados_sale_por_error() -> None:
    """Con Sentry en DEFAULT_EVENT_LEVEL=ERROR, un `warning` no sube. Elegir el nivel
    ES decidir si alguien se entera."""
    codigo = _sin_comentarios(_fuente())
    assert re.search(r"logger\.error\(\s*\n?\s*f?\"\[P1-PLAN-DISPLAY-I18N\] enriquecimiento SIN escrituras", codigo), (
        "La degradación «cero escrituras habiendo despachado lotes» no sale por "
        f"`logger.error`, así que Sentry no la recoge. [{_MARKER}]"
    )
