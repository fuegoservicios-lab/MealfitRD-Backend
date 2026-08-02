"""[P2-PAPER-NO-INK · 2026-08-01/02] Ningún color saturado sobrevive en la
superficie papel del landing — en NINGUNA notación CSS, ni en CSS ni en JSX.

Motivación (por qué este test no existía y por qué ahora sí):
    El rediseño `2026-08-01-landing-papel-tecnico` cambió la política de tema
    de 6 rutas de marketing (`/`, `/precios`, `/como-funciona`, `/funciones`,
    `/precision`, `/motor` — SSOT en `frontend/src/utils/paperSurface.js`) de
    "oscuro siempre" a "papel: blanco y negro estricto, cuadrícula
    milimetrada, reglas de 1px". Verificado antes de escribir este test:
    CERO tests anclaban esa regla. Un futuro edit (o un merge de otra rama)
    podía reintroducir un acento de color sin que nada lo notara.

Por qué el JSX y no solo el CSS:
    Los 9 acentos pastel del landing pre-rediseño NO vivían en el CSS.
    Vivían en `HowItWorks.jsx` (`STEPS[].color` inyectado como
    `style={{'--accent': step.color}}`) y en `DashboardShowcase.jsx`
    (`FEATURES[].color`, mismo patrón). Un escaneo solo-CSS los habría dado
    por muertos — de ahí que `_iter_paper_sources` incluya `.jsx`/`.js`.

Por qué CUATRO notaciones y no solo `#RRGGBB` [P2-PAPER-NO-INK · 2026-08-02]:
    La primera versión de este test solo miraba hex de 6 dígitos. Verificar
    `/precios` en un navegador real (no solo grep) destapó un blob de fondo
    multicolor (`radial-gradient(..., rgba(79, 70, 229, 0.18) ...)`) viviendo
    a toda anchura detrás del header — con el test original en VERDE. Un
    guard ciego a una notación da MÁS confianza de la que merece: es peor
    que no tener guard, porque nadie vuelve a mirar a mano lo que el test ya
    dice cubrir. `_saturated_colors_in` ahora mira:
      - `#RRGGBB` (hex de 6 dígitos) — canal por canal, max−min > 24.
      - `#RGB` (hex de 3 dígitos, `#F00` es tan rojo como `#FF0000`) —
        expandiendo cada dígito antes de aplicar el mismo criterio.
      - `rgb()`/`rgba()` — mismo criterio de canal, leído de los 3 primeros
        argumentos numéricos.
      - `hsl()`/`hsla()` — por SATURACIÓN declarada (el segundo argumento):
        >10% ya no es un neutro. No hay canales R/G/B que leer aquí, así que
        el criterio de "canal" no aplica — se usa el propio nombre del
        espacio de color.
    `test_the_scanner_catches_every_color_notation` ancla las 4 a la vez:
    sin ese caso, un agujero de notación no lo ve nadie — exactamente el
    modo de fallo que este mismo apartado describe haber sufrido.

Por qué un test y no una revisión visual:
    Un `#34D399` (o un `rgba(52, 211, 153, .4)`, o un `hsl(160, 84%, 60%)`)
    superviviente en un estado que solo se ve en un breakpoint, o detrás de
    un `:hover`, no lo detecta nadie mirando la pantalla.

Por qué `components/layout/` queda FUERA:
    Header, Footer y Layout se montan en las 21 rutas de la app (no solo las
    6 de papel). Sus bloques `data-theme="dark"`/`"light"` con `#ef4444`,
    `#818CF8` y compañía siguen sirviendo a las 15 rutas que NO son
    marketing (legales, /novedades, /supermercado, /configuracion). Borrarlos
    de ahí rompería esas páginas. Su corrección papel (¿tiene bloque `paper`
    donde tiene bloque `dark`?) la cubre `test_p2_paper_theme_blocks.py` —
    esa es la pregunta correcta para un módulo compartido, esta no.

Por qué se filtran comentarios ANTES de escanear:
    Un comentario que EXPLICA por qué se borró un color no es una violación.
    Lección de 2026-07-31, repetida aquí porque cuesta poco y muerde caro: un
    test de este tipo en este mismo repo llegó a fallar contra su propio
    arreglo porque el comentario que documentaba el fix contenía la cadena
    prohibida. `_strip_comments` sustituye por espacios (no borra) para no
    descuadrar los números de línea del informe. `DashboardShowcase.module.css`
    y `HowItWorks.jsx` tienen ambos un bloque `/* ... */` que documenta,
    a propósito, los colores que existían ANTES del rediseño — sin el
    stripper, este test los marcaría como violaciones vivas.

`test_the_comment_stripper_actually_strips` existe porque, sin ese caso, un
fallo silencioso del stripper convierte el test de arriba en un test que no
mira nada — y pasaría en verde para siempre.

Excepciones activas (verdaderos positivos, NO relajar el test para
acomodarlos — ver `_PENDING_REWORK` abajo):
    - `BenchmarkShowcase.jsx` / `.module.css`: Task 11 del plan
      ("SSOT del benchmark") está BLOQUEADA esperando una decisión del dueño
      (benchmark LLM: 0 vs 55 — ver `.superpowers/sdd/2026-08-01-landing-
      papel-tecnico/progress.md`). La sección sigue siendo el instrumento
      oscuro de junio con sus acentos (`#2DD4BF`, `#34D399`, `#A78BFA`, etc.)
      intactos.
    - `Pricing.module.css`: Task 13 ("Las páginas token-driven") todavía no
      se ha ejecutado (sin `task-13-report.md` en el ledger SDD al momento
      de escribir este test). El archivo retiene hex saturados reales
      (`#047857`, `#8B5CF6` ×2) Y un blob de fondo entero en `rgba()`
      (`.pricing::before`, 3 `radial-gradient` con índigo/rosa/esmeralda) Y
      una píldora tintada (`.badge`, `rgba(79, 70, 229, ...)`) heredados del
      tema oscuro/claro pre-rediseño. `Pricing.jsx` en sí SÍ está limpio
      (0 color) — la exclusión es solo del módulo CSS.

Tooltip-anchor: P2-PAPER-NO-INK
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"

# SOLO módulos EXCLUSIVOS de la superficie papel.
#
# `components/layout/` (Header, Footer, Layout) queda FUERA a propósito: esos
# tres se montan en 21 rutas, y sus bloques `data-theme="dark"`/`"light"` con
# #ef4444, #818CF8 y compañía siguen sirviendo a las 15 rutas que NO son
# marketing (legales, novedades, /supermercado, /configuracion). Bajo `paper`
# no aplican, pero borrarlos rompería esas páginas. Su corrección papel la
# cubre test_p2_paper_theme_blocks.py, que es la pregunta correcta para un
# módulo compartido: «¿tiene bloque paper donde tiene bloque dark?».
_PAPER_DIRS = [_FRONTEND_SRC / "components" / "home"]
_PAPER_FILES = [
    _FRONTEND_SRC / "pages" / "HowItWorksPage.module.css",
    _FRONTEND_SRC / "pages" / "Engine.module.css",
    _FRONTEND_SRC / "pages" / "PrecisionPage.module.css",
]

# TODO(P2-PAPER-NO-INK): dos exclusiones temporales, verdaderos positivos
# ambas — NO añadir más sin documentar aquí y en el docstring de arriba.
#
#   1. BenchmarkShowcase.jsx / .module.css — Task 11 del plan de rediseño
#      ("SSOT del benchmark") está BLOQUEADA esperando decisión del dueño
#      (benchmark LLM: 0 vs 55). Quitar de esta lista en cuanto Task 11
#      cierre y la sección quede repintada en papel.
#   2. Pricing.module.css — Task 13 ("Las páginas token-driven") todavía no
#      se ejecutó; el módulo retiene color saturado real en 3 notaciones
#      (hex, rgb() del blob de fondo, rgb() de la píldora de badge) del
#      tema oscuro/claro pre-rediseño. Quitar de esta lista en cuanto
#      Task 13 cierre.
_PENDING_REWORK = {
    _FRONTEND_SRC / "components" / "home" / "BenchmarkShowcase.jsx",
    _FRONTEND_SRC / "components" / "home" / "BenchmarkShowcase.module.css",
    _FRONTEND_SRC / "components" / "home" / "Pricing.module.css",
}

# --- Las 4 notaciones de color que CSS/JSX pueden usar --------------------
_HEX6 = re.compile(r"#(?P<hex6>[0-9A-Fa-f]{6})\b")
# `#RGB` es tan rojo como `#RRGGBB` (`#F00` == `#FF0000`) — el lookahead
# negativo (además del \b) evita que esto matchee los primeros 3 dígitos de
# un hex de 6 (ahí el char siguiente sigue siendo hex, así que \b ya falla
# por sí solo; el lookahead lo deja explícito).
_HEX3 = re.compile(r"#(?P<hex3>[0-9A-Fa-f]{3})\b(?![0-9A-Fa-f])")
_RGB_FN = re.compile(r"\brgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", re.IGNORECASE)
_HSL_FN = re.compile(r"\bhsla?\(\s*[\d.]+(?:deg|turn|rad)?\s*,\s*([\d.]+)%", re.IGNORECASE)

# Un neutro tiene los tres canales casi iguales. 24/255 deja pasar los grises
# ligeramente cálidos del papel (#FBFBFA, #F4F4F2) y nada más — y sí rechaza
# los slate azulados heredados (#0F172A da 27, #475569 da 34), que es lo que
# queremos: en papel esos deben ser --pa-ink / --pa-ink-2. Mismo criterio
# para hex3 y rgb()/rgba() — son la misma noción de "canal", solo cambia de
# dónde se leen los 3 números.
_SATURATION_LIMIT = 24
# hsl()/hsla() no tiene canales R/G/B — tiene un eje de saturación propio.
# >10% ya no es un gris: hsl(60, 3%, 98%) es un blanco cálido de papel;
# hsl(250, 84%, 60%) es un índigo saturado.
_HSL_SATURATION_LIMIT = 10.0

# Comentarios CSS y JS. Se eliminan ANTES de escanear.
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"^\s*//.*$", re.MULTILINE)


def _is_saturated_rgb(r: int, g: int, b: int) -> bool:
    return (max(r, g, b) - min(r, g, b)) > _SATURATION_LIMIT


def _saturated_colors_in(text: str) -> list[str]:
    """Devuelve TODOS los aciertos de color saturado en `text`, en su
    notación original — para que el mensaje de error diga QUÉ encontró
    (`#8B5CF6`, `rgb(79, 70, 229)`, `hsl(250, 84%`...), no solo DÓNDE.

    Cubre las 4 notaciones que CSS/JSX usan en este repo: hex de 6 dígitos,
    hex de 3 dígitos, rgb()/rgba() por canal, hsl()/hsla() por saturación.
    Ver docstring del módulo, sección "Por qué CUATRO notaciones", para el
    incidente que motivó ampliar esto más allá de `#RRGGBB`.
    """
    hits: list[str] = []

    for m in _HEX6.finditer(text):
        r, g, b = (int(m.group("hex6")[i:i + 2], 16) for i in (0, 2, 4))
        if _is_saturated_rgb(r, g, b):
            hits.append(m.group(0))

    for m in _HEX3.finditer(text):
        r, g, b = (int(c * 2, 16) for c in m.group("hex3"))
        if _is_saturated_rgb(r, g, b):
            hits.append(m.group(0))

    for m in _RGB_FN.finditer(text):
        r, g, b = (int(v) for v in m.groups())
        if _is_saturated_rgb(r, g, b):
            hits.append(f"rgb({r}, {g}, {b})")

    for m in _HSL_FN.finditer(text):
        if float(m.group(1)) > _HSL_SATURATION_LIMIT:
            hits.append(m.group(0))

    return hits


def _strip_comments(text: str) -> str:
    """Un comentario que EXPLICA por qué se borró un color no es una violación.

    Lección de 2026-07-31, repetida aquí porque cuesta poco y muerde caro: un
    test de este tipo llegó a fallar contra su propio arreglo porque el
    comentario que documentaba el fix contenía la cadena prohibida. Se
    sustituye por espacios (no se borra) para no descuadrar los números de
    línea del informe.
    """
    text = _CSS_COMMENT.sub(lambda m: " " * len(m.group(0)), text)
    return _LINE_COMMENT.sub(lambda m: " " * len(m.group(0)), text)


def _iter_paper_sources():
    for d in _PAPER_DIRS:
        for f in d.rglob("*"):
            if f.is_file() and f.suffix in {".css", ".jsx", ".js"} and f not in _PENDING_REWORK:
                yield f
    for f in _PAPER_FILES:
        if f.exists():
            yield f


def test_no_saturated_color_in_paper_surfaces():
    violations = []
    for path in _iter_paper_sources():
        clean = _strip_comments(path.read_text(encoding="utf-8"))
        for lineno, line in enumerate(clean.splitlines(), 1):
            for hit in _saturated_colors_in(line):
                rel = path.relative_to(_REPO_ROOT)
                violations.append(f"{rel}:{lineno} → {hit}  |  {line.strip()[:90]}")
    assert not violations, (
        "P2-PAPER-NO-INK: color saturado (hex/rgb/hsl) en la superficie papel:\n  "
        + "\n  ".join(violations)
        + "\n\nArreglar el color (o, si es un archivo genuinamente pendiente "
        "de otra tarea del plan, añadirlo a _PENDING_REWORK con su razón) — "
        "NO relajar los límites de saturación ni sacar el directorio del escaneo."
    )


def test_the_comment_stripper_actually_strips():
    """Sin este caso, un fallo del stripper convierte el test de arriba en un
    test que no mira nada — y pasaría en verde para siempre."""
    assert "#FF0000" not in _strip_comments("/* ojo: antes era #FF0000 */")
    assert "#FF0000" not in _strip_comments("// antes era #FF0000")
    assert "#FF0000" in _strip_comments("color: #FF0000;")


def test_the_scanner_catches_every_color_notation():
    """[P2-PAPER-NO-INK · 2026-08-02] Ancla las 4 notaciones a la vez.

    Por qué existe: la primera versión de este escáner solo miraba
    `#RRGGBB`. Verificar `/precios` en un navegador real (no solo grep)
    destapó un blob de fondo multicolor entero construido en `rgba()` que
    el test original, en verde, jamás iba a ver — porque no miraba esa
    notación. Un guard ciego a una notación da MÁS confianza de la que
    merece: es peor que no tener guard, porque nadie vuelve a mirar a mano
    lo que el test ya dice cubrir. Sin este caso, un agujero de notación
    (el próximo `hsl(...)` o `#RGB` que alguien pegue) no lo ve nadie hasta
    que alguien vuelva a abrir un navegador de casualidad.
    """
    assert _saturated_colors_in("color: #34D399;")
    assert _saturated_colors_in("color: #F00;")
    assert _saturated_colors_in("background: rgba(79, 70, 229, 0.08);")
    assert _saturated_colors_in("background: hsl(250, 84%, 60%);")
    assert not _saturated_colors_in("color: #FBFBFA;")
    assert not _saturated_colors_in("color: #3D3D3B;")
    assert not _saturated_colors_in("background: rgba(11, 11, 11, 0.06);")
    assert not _saturated_colors_in("background: hsl(60, 3%, 98%);")


def test_pending_rework_exclusions_are_still_pending():
    """Guard inverso: si BenchmarkShowcase o Pricing.module.css alguna vez
    quedan limpios de color saturado en CUALQUIER notación (Task 11 / Task
    13 cerradas), esta excepción debe RETIRARSE de `_PENDING_REWORK` — de lo
    contrario queda una excepción documentada que ya no aplica, invitando a
    colgar un color nuevo detrás de ella sin que el escáner lo vea (el mismo
    modo de fallo que I6 del CLAUDE.md ya describe para otra whitelist de
    este repo).

    Si este test falla, es BUENA noticia: significa que alguien cerró Task 11
    o Task 13. Acción: borrar la entrada correspondiente de `_PENDING_REWORK`
    (y de los docstrings que la documentan), no marcar este test como xfail.
    """
    still_saturated = []
    for path in sorted(_PENDING_REWORK):
        if not path.exists():
            continue
        clean = _strip_comments(path.read_text(encoding="utf-8"))
        if any(_saturated_colors_in(line) for line in clean.splitlines()):
            still_saturated.append(path.relative_to(_REPO_ROOT))
    assert still_saturated == [
        p.relative_to(_REPO_ROOT) for p in sorted(_PENDING_REWORK) if p.exists()
    ], (
        "P2-PAPER-NO-INK: al menos uno de los archivos en _PENDING_REWORK ya "
        "NO tiene color saturado. Retiralo de la lista de exclusion — de lo "
        "contrario es una excepcion fantasma que ya no protege nada."
    )
