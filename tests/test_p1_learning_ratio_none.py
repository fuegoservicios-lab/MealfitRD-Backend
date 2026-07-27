"""[P1-LEARNING-RATIO-NONE · 2026-07-27] Un logger.warning mataba la semana del plan.

## El caso, sacado de los logs del VPS

    File "/opt/mealfit/backend/cron_tasks.py", line 27439, in _chunk_worker
    TypeError: unsupported format string passed to NoneType.__format__

Seis veces en tres horas, chunks 4 / 6 / 9 del plan `69f9e03d` — el mismo cuyas semanas 4-11
llevaban días sin avanzar. La línea era:

    f"[P0-1/LEARNING-FLEXIBLE] chunk {week_number} ratio={learning_ready_ratio:.0%}, "

`learning_ready.get("ratio")` devuelve **None** cuando no hay datos de aprendizaje, y
`None:.0%` levanta TypeError.

## Por qué duele tanto para ser una línea de log

La excepción sale de DENTRO del `logger.warning`, o sea que **el mensaje de diagnóstico mata al
chunk que intentaba explicar**. `_chunk_worker` aborta, el chunk queda `failed` y esa SEMANA del
plan no se genera nunca. El usuario pidió 30 días y se queda sin ellos por un formato de string.

Y solo dispara en el camino `flexible_mode` con ratio ausente — la combinación menos probada.

## Lo que lo hace un descuido y no un diseño

El idioma seguro `(learning_ready_ratio or 0):.0%` ya se usa BIEN en las otras cuatro
interpolaciones de esta misma variable, y hay un guard `is not None` antes de la quinta. Esta
era la única sin proteger.

tooltip-anchor: P1-LEARNING-RATIO-NONE
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_CRON = Path(__file__).resolve().parent.parent / "cron_tasks.py"
_SRC = _CRON.read_text(encoding="utf-8")


# ───────────── 1. la reproducción del crash ─────────────

def test_none_con_formato_porcentaje_revienta():
    """La premisa: esto es lo que hacía la línea, y por qué mataba el chunk."""
    ratio = None
    with pytest.raises(TypeError):
        f"{ratio:.0%}"


def test_el_idioma_seguro_no_revienta():
    for ratio in (None, 0, 0.5, 1.0):
        assert f"{(ratio or 0):.0%}".endswith("%")


# ───────────── 2. el contrato: NINGUNA interpolación sin proteger ─────────────

def test_ninguna_interpolacion_de_ratio_queda_desprotegida():
    """Ancla de la CLASE, no del caso. Cada `{learning_ready_ratio...:.0%}` debe ir con
    `(… or 0)` o estar dentro de un guard `is not None`.

    Se busca la variable formateada como PORCENTAJE sin el `or 0` inmediato. La quinta
    aparición (línea ~26685) vive bajo `if learning_ready_ratio is not None:` — se acepta
    comprobando que ese guard exista justo antes.
    """
    desprotegidas = []
    for m in re.finditer(r"\{learning_ready_ratio:\.\d*%\}", _SRC):
        # ¿hay un guard `is not None` en las 12 líneas previas?
        antes = _SRC[:m.start()].splitlines()[-12:]
        if not any("learning_ready_ratio is not None" in l for l in antes):
            linea = _SRC[:m.start()].count("\n") + 1
            desprotegidas.append(linea)
    assert not desprotegidas, (
        f"interpolación de `learning_ready_ratio` como % sin `(… or 0)` ni guard, en la(s) "
        f"línea(s) {desprotegidas}. `learning_ready.get('ratio')` devuelve None sin datos de "
        f"aprendizaje y el TypeError ABORTA `_chunk_worker`: esa semana del plan no se genera."
    )


def test_el_idioma_seguro_sigue_en_uso():
    """Si alguien 'limpia' los `(… or 0)` por verbosos, vuelve el crash."""
    n = len(re.findall(r"\(learning_ready_ratio or 0\)", _SRC))
    assert n >= 4, (
        f"solo {n} usos del idioma seguro `(learning_ready_ratio or 0)`; se esperaban >= 4. "
        f"Quitarlos reintroduce el TypeError que mata el chunk."
    )


# ───────────── 3. el sitio concreto que fallaba ─────────────

def test_la_rama_flexible_mode_esta_protegida():
    """Era la única sin proteger, y encima la menos transitada (flexible_mode + ratio ausente)."""
    i = _SRC.index("[P0-1/LEARNING-FLEXIBLE]")
    bloque = _SRC[i:_SRC.index(")", _SRC.index("logger.warning", i - 400))]
    assert "learning_ready_ratio:.0%" not in bloque, "la rama flexible_mode volvió a formatear None"


def test_el_marker_esta_puesto():
    assert "P1-LEARNING-RATIO-NONE" in _SRC
