"""[P1-PAPER-BENCHMARK · 2026-08-02] Las cifras del benchmark viven en UN solo
sitio, y lo derivable se deriva.

QUÉ PASÓ. Las mismas métricas estaban escritas a mano en TRES ficheros del
frontend — `components/home/BenchmarkShowcase.jsx`, `pages/PrecisionPage.jsx` y
`pages/Engine.jsx` — y ya habían drifteado: la fila «macros que cuadran al
recalcular» decía `llm: 0` en dos de ellos y `llm: 55` en el tercero. No había
forma de saber cuál era la buena leyendo el código, así que el rediseño de la
sección quedó bloqueado esperando al dueño. El 2026-08-02 fijó el valor: es
**0**; `PrecisionPage` era el que mentía.

QUÉ ANCLA ESTE TEST, y por qué cada cosa:

  1. Que el SSOT existe.
  2. Que **ningún consumidor declara cifras propias**. Un SSOT al que alguien
     puede añadirle un "solo este número" al lado deja de serlo en el primer
     merge. Se comprueba con el patrón `mealfit: <dígito>`, que es la forma
     literal que tenían las tres tablas duplicadas, y con las cifras de titular
     (`±3,2 %`) que vivían por duplicado en `HIGHLIGHTS` y `HERO_STATS`.
  3. Que **la Δ de la Fig. 04.2 se calcula, jamás se escribe**. Escribir
     `+67,7` habría creado un TERCER sitio de drift dentro del mismo P-fix que
     cierra los otros dos — el riesgo 3(c) del spec, textualmente.
  4. Que la fila binaria **no volvió al gráfico**. `100 vs 0` es una capacidad,
     no una magnitud; dibujarla en el mismo eje 0-100 que un porcentaje
     fabricaba un «Δ 100 pts» inventado (spec §5.1). Si alguien la reintroduce
     en `VERSUS`, las dos figuras vuelven a mentir sin que nada más lo note.
  5. Que el número que se decidió **no se puede reintroducir**: `llm: 55` no
     puede reaparecer en ningún consumidor.

POR QUÉ ES UN TEST PARSER-BASED Y NO UN TEST DE RENDER: el drift no rompe nada
en ejecución. Dos ficheros con números distintos renderizan perfectamente; el
fallo es que el sitio afirma dos cosas incompatibles a dos visitantes distintos.
Eso solo se ve leyendo los tres ficheros a la vez, que es justo lo que nadie
hace.

Tooltip-anchor: P1-PAPER-BENCHMARK
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"

_SSOT = _FRONTEND_SRC / "data" / "benchmark.js"

# Los TRES consumidores. `Engine.jsx` no estaba en el plan original: apareció
# en la revisión final de la rama (2026-08-02) al descubrir que la cifra vivía
# en tres ficheros y no en dos. Dejarlo fuera habría dado por cerrado un drift
# a dos bandas que en realidad era a tres.
_CONSUMERS = [
    _FRONTEND_SRC / "components" / "home" / "BenchmarkShowcase.jsx",
    _FRONTEND_SRC / "pages" / "PrecisionPage.jsx",
    _FRONTEND_SRC / "pages" / "Engine.jsx",
]

_FIGURES = _FRONTEND_SRC / "components" / "home" / "figures"

# Comentarios: un comentario que EXPLICA el drift que se cerró no es una
# violación. Misma lección que `test_p2_paper_no_ink.py::_strip_comments`
# (2026-07-31): un test de este tipo llegó a fallar contra su propio arreglo
# porque el comentario que lo documentaba contenía la cadena prohibida. Aquí
# muerde el doble, porque la cabecera de cada fichero tocado CITA los valores
# viejos (`llm: 0` vs `llm: 55`) para que nadie los reintroduzca por error.
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"^\s*//.*$", re.MULTILINE)


def _strip_comments(text: str) -> str:
    """Sustituye por espacios (no borra) para no descuadrar los números de línea."""
    text = _BLOCK_COMMENT.sub(lambda m: " " * len(m.group(0)), text)
    return _LINE_COMMENT.sub(lambda m: " " * len(m.group(0)), text)


def _code_of(path: Path) -> str:
    return _strip_comments(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. El SSOT existe y expone el contrato que las tres superficies consumen
# ---------------------------------------------------------------------------
def test_ssot_exists():
    assert _SSOT.exists(), (
        "P1-PAPER-BENCHMARK: falta `frontend/src/data/benchmark.js`."
    )


def test_ssot_exports_the_agreed_surface():
    """Los seis nombres son el contrato. Renombrar uno rompe silenciosamente a
    un consumidor que no se esté mirando en ese momento."""
    text = _SSOT.read_text(encoding="utf-8")
    for name in ("SERIES", "MACROS", "BANDS", "VERSUS", "CAPS", "HEADLINE_FIGURES"):
        assert re.search(rf"export const {name}\b", text), (
            f"P1-PAPER-BENCHMARK: `benchmark.js` debe exportar `{name}`."
        )


def test_the_binary_capability_is_not_a_magnitude_anymore():
    """`100 vs 0` («macros que cuadran al recalcular») es una CAPACIDAD.

    Vivía en `VERSUS`, dibujada en el mismo eje 0-100 que un porcentaje de
    acierto y un porcentaje de planes — tres magnitudes incommensurables sobre
    una misma regla, lo que fabricaba la impresión de que la tercera mejora era
    7× la primera, y un «Δ 100 pts» que no significa nada (spec §5.1).

    Ahora vive en `CAPS` como `■ SÍ / □ NO`. Este test falla si alguien la
    devuelve a `VERSUS`: la corrección es estructural, no cosmética, y desde
    fuera no se nota hasta que la figura vuelve a mentir.
    """
    code = _code_of(_SSOT)
    versus_block = code.split("export const VERSUS")[1].split("export const CAPS")[0]
    assert "recalcular" not in versus_block.lower(), (
        "P1-PAPER-BENCHMARK: la fila binaria («cuadran al recalcular») volvió a "
        "`VERSUS`. Es una capacidad, no una magnitud: su sitio es `CAPS`."
    )
    caps_block = code.split("export const CAPS")[1].split("export const HEADLINE")[0]
    assert "recalcular" in caps_block.lower(), (
        "P1-PAPER-BENCHMARK: la capacidad «cuadran al recalcular» desapareció de "
        "`CAPS`. Bajó del gráfico a la tabla — no se borró."
    )


# ---------------------------------------------------------------------------
# 2. Ningún consumidor declara cifras propias
# ---------------------------------------------------------------------------
def test_consumers_import_the_ssot_and_hold_no_numbers():
    for path in _CONSUMERS:
        assert path.exists(), f"P1-PAPER-BENCHMARK: no existe {path}."
        text = _code_of(path)
        assert "data/benchmark" in text, (
            f"P1-PAPER-BENCHMARK: {path.name} no importa el SSOT "
            "(`frontend/src/data/benchmark.js`)."
        )
        assert not re.search(r"mealfit:\s*\d", text), (
            f"P1-PAPER-BENCHMARK: {path.name} sigue declarando cifras a mano "
            "(`mealfit: <número>`). Ese es exactamente el drift que este P-fix "
            "cierra: tres ficheros y dos verdades."
        )
        # Las cifras de TITULAR también estaban duplicadas: `±3.2%` vivía en
        # `HIGHLIGHTS` (landing), en `HERO_STATS` (/precision) y en la prosa de
        # /motor. Las ETIQUETAS pueden diferir entre superficies; el número no.
        for figure in ("3.2", "3,2"):
            assert f"±{figure}" not in text, (
                f"P1-PAPER-BENCHMARK: {path.name} escribe ±{figure}% a mano. "
                "Esa cifra se deriva de `MACROS` en `HEADLINE_FIGURES`."
            )


def test_the_discarded_value_cannot_come_back():
    """`llm: 55` fue la mentira concreta. El dueño fijó 0 el 2026-08-02.

    Un test que solo comprobara «hay SSOT» pasaría con el 55 reinstalado en
    cualquier consumidor mañana.
    """
    for path in _CONSUMERS + [_SSOT]:
        text = _code_of(path)
        assert not re.search(r"llm:\s*55\b", text), (
            f"P1-PAPER-BENCHMARK: {path.name} reintrodujo `llm: 55`. El valor "
            "decidido por el dueño (2026-08-02) es 0, y por ser binario la fila "
            "vive en `CAPS` como □ NO."
        )


# ---------------------------------------------------------------------------
# 3. La Δ se deriva, jamás se escribe
# ---------------------------------------------------------------------------
def test_delta_is_derived_never_written():
    """La cota de la Fig. 04.2 vale `mealfit − llm` = 67,7 pp.

    Escribirla crea un TERCER sitio de drift dentro del mismo P-fix que cierra
    los otros dos — el riesgo 3(c) del spec, textualmente. Se comprueba en
    TODAS las figuras y en el propio SSOT, no solo en `InBandBar.jsx`: el sitio
    donde alguien la escribiría "solo para no recalcularla" es imprevisible.
    """
    targets = [_SSOT] + sorted(_FIGURES.glob("*.jsx")) + _CONSUMERS
    for path in targets:
        text = _code_of(path)
        for written in ("67.7", "67,7"):
            assert written not in text, (
                f"P1-PAPER-BENCHMARK: la Δ está escrita a mano en {path.name} "
                f"(`{written}`). Tiene que derivarse de `VERSUS`."
            )


def test_in_band_figure_computes_the_delta_from_the_ssot():
    """El complemento del test anterior: que no esté escrita no prueba que se
    calcule — podría simplemente haber desaparecido de la figura, y con ella la
    única cifra que convierte la comparación en una MEDICIÓN.
    """
    fig = _FIGURES / "InBandBar.jsx"
    assert fig.exists(), "P1-PAPER-BENCHMARK: falta `figures/InBandBar.jsx`."
    text = _code_of(fig)
    assert re.search(r"\.mealfit\s*-\s*\w*\.?\w*llm", text, re.IGNORECASE), (
        "P1-PAPER-BENCHMARK: `InBandBar.jsx` no calcula la Δ como "
        "`mealfit − llm`. La cota tiene que derivarse, no desaparecer."
    )
    assert "pp" in text, (
        "P1-PAPER-BENCHMARK: la cota perdió su unidad. `pp` (puntos "
        "porcentuales) + «de planes» es la única diferencia entre una medición "
        "y una afirmación (spec §5.3)."
    )
