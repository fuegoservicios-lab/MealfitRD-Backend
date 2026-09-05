"""[P1-MOBILE-FIT · 2026-08-09] El landing en móvil: que quepa, que se lea, que
se pueda tocar.

Tres defectos medidos con Playwright sobre producción a 320/360/390/430 px
(`isMobile`, dSF 2), y una guarda de proceso.

1. LA FRANJA DEL HERO SE CORTABA. Las dos celdas que sobreviven en móvil piden
   180,4 + 195,3 = 375,7 px. Con `overflow-x: clip` en html/body no hay scroll
   que lo revele: el texto se corta y no hay forma de leerlo. A 320 px se
   perdían 55,7 px.

   Era una regresión de P1-HERO-DEDUP-ACCENT (mismo día): el bloque móvil de
   `Hero.module.css` llevaba escrita la medición «83,4 + 145,8 = 229 de los
   360», cierta cuando la celda 1 era el wordmark. El P-fix la cambió a un
   literal de 180,4 px y no se volvió a medir el móvil.

   El corte está en 410 px y no en 383 a propósito: 383 dejaría 0,3 px de
   holgura, a merced de cómo renderice la fuente cada dispositivo.

   [P2-HERO-VANGUARDIA · 2026-09-05] Eran 400 y 376. La celda 1 pasó de «DE
   PRECISIÓN» a «DE VANGUARDIA» (+1 carácter ≈ +6,9 px), la pareja pide ahora
   ~382,6 px y el corte sube con ella. La aritmética viva la ancla
   `test_p2_hero_vanguardia.py`; este fichero solo exige que el apilado
   exista y cubra el corte.

2. HOWITWORKS TRUNCABA SIN EXPANSOR. Las 4 descripciones piden 6, 9, 5 y 4
   líneas a 360 px y el recorte mostraba 3: 12 líneas escondidas (18 a 320) y
   ningún control para pedirlas. El coste medido de mostrarlas enteras son
   +279 px sobre una página de 10.598 (+2,6 %), no «de 1 a 3 pantallas» como
   temía el comentario que lo justificaba.

3. OBJETIVOS TÁCTILES CORTOS. `SeeMoreLink` a 32 px de alto (4 instancias) y
   los iconos sociales a 29×29 bajo papel / 35,2 en el footer oscuro. El
   mínimo es 44 (Apple) / 48 (Google). Se agrandan con un pseudo-elemento
   absoluto: no mueve tinta, no desplaza el subrayado del enlace (su
   `border-bottom` ES el subrayado) y no engorda el anillo de foco.

4. NADA MEDÍA EL LANDING A 320 px. Por eso el defecto 1 llegó a producción.
   Este fichero ancla las decisiones contra su fuente (corre en cada pytest);
   la medición REAL vive en `frontend/e2e/mobile_no_overflow.spec.js`, y el
   último test de aquí exige que siga existiendo.

MÉTODO — lo que este fichero NO puede ver: es un parser, no un navegador. No
mide anchos; ancla las decisiones que los produjeron. Por eso el e2e es su
pareja obligatoria y no un extra.

Nota de implementación: todo escaneo pasa antes por `_strip_comments`. Sin eso,
una nota que DESCRIBA el recorte prohibido cuenta como una violación del
recorte — el test caería contra su propio arreglo, que es un modo de fallo que
este repo ya pagó una vez.

Tooltip-anchor: P1-MOBILE-FIT
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "frontend" / "src"
_HERO_CSS = _SRC / "components" / "home" / "Hero.module.css"
_HOWITWORKS_CSS = _SRC / "components" / "home" / "HowItWorks.module.css"
_SEEMORE_CSS = _SRC / "components" / "home" / "SeeMoreLink.module.css"
_FOOTER_CSS = _SRC / "components" / "layout" / "Footer.module.css"
_BENCHMARK_CSS = _SRC / "components" / "home" / "BenchmarkShowcase.module.css"
_BENCHMARK_JSX = _SRC / "components" / "home" / "BenchmarkShowcase.jsx"
_SHOWCASE_JSX = _SRC / "components" / "home" / "DashboardShowcase.jsx"
_SERVICE_WORKER = _SRC / "custom-sw.js"
_E2E_SPEC = _REPO_ROOT / "frontend" / "e2e" / "mobile_no_overflow.spec.js"

# El ancho de contrato más estrecho del repo. Un iPhone con Display Zoom
# activado renderiza aquí, así que no es un ancho teórico.
_CONTRACT_WIDTH = 320


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _strip_js_comments(src: str) -> str:
    """Bloque Y línea. La versión de CSS no basta aquí: las notas de este repo
    explican qué literal se retiró, y un escáner que las lea acusa al arreglo de
    ser el defecto. (Un `//` dentro de una URL también cae; para buscar una
    frase prohibida da igual.)"""
    return re.sub(r"//[^\n]*", "", _strip_comments(src))


def _media_blocks(css: str) -> list[tuple[int | None, str]]:
    """[(max-width, cuerpo)] emparejando llaves de verdad — un `@media` con
    reglas anidadas rompe cualquier regex perezosa."""
    out: list[tuple[int | None, str]] = []
    for m in re.finditer(r"@media[^{]*\{", css):
        head = m.group(0)
        mw = re.search(r"max-width:\s*(\d+)px", head)
        depth, i = 1, m.end()
        while i < len(css) and depth:
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1
        out.append((int(mw.group(1)) if mw else None, css[m.end():i - 1]))
    return out


def _rule_body(css: str, selector: str) -> list[str]:
    """Cuerpos de todas las reglas cuyo selector empieza por `selector`."""
    pattern = rf"(?:^|\}}|\{{)\s*{re.escape(selector)}(?:[^{{}}]*)?\{{([^{{}}]*)\}}"
    return re.findall(pattern, css, re.MULTILINE)


# ── 1. La franja ────────────────────────────────────────────────────────────

def test_cartridge_stacks_before_it_runs_out_of_room():
    """La fila de dos celdas necesita ~382,6 px. Bajo 410 px el cajetín TIENE que
    apilar en una columna — es la única salida que no pierde información ni
    rompe el piso tipográfico de 11 px que el repo fijó por escrito."""
    css = _strip_comments(_HERO_CSS.read_text(encoding="utf-8"))
    stacked = [
        (mw, body) for mw, body in _media_blocks(css)
        if mw is not None and _CONTRACT_WIDTH <= mw <= 420
        and any(
            re.search(r"grid-template-columns:\s*1fr\s*;", b)
            for b in _rule_body(body, ".cartridge")
        )
    ]
    assert stacked, (
        "P1-MOBILE-FIT: no hay ningún bloque @media (entre 320 y 420px) donde "
        "`.cartridge` pase a una sola columna. Sin él las dos celdas piden "
        "~382,6px y a 320px se cortan ~62,6px SIN scroll que lo revele "
        "(html/body llevan overflow-x: clip)."
    )
    assert any(mw >= 383 for mw, _ in stacked), (
        "P1-MOBILE-FIT: el apilado arranca por debajo de 383px, justo donde el "
        "corte YA ocurre. Tiene que cubrir 383px hacia abajo."
    )


def test_stacked_cartridge_turns_its_hairline_and_drops_the_right_align():
    """Apilar sin girar la regla deja un `border-left` colgando al inicio de la
    segunda fila, y el `text-align: right` que la fila usaba para empujar el
    lugar contra el margen deja la fila 2 desalineada de la 1."""
    css = _strip_comments(_HERO_CSS.read_text(encoding="utf-8"))
    bodies = [
        body for mw, body in _media_blocks(css)
        if mw is not None and _CONTRACT_WIDTH <= mw <= 420
        and any(
            re.search(r"grid-template-columns:\s*1fr\s*;", b)
            for b in _rule_body(body, ".cartridge")
        )
    ]
    joined = "\n".join(bodies)
    assert re.search(r"border-top:\s*1px", joined), (
        "P1-MOBILE-FIT: el bloque apilado no gira la hairline a `border-top`. "
        "La separación entre celdas es vertical cuando apilan."
    )
    assert re.search(r"border-left:\s*(0|none)", joined), (
        "P1-MOBILE-FIT: el bloque apilado no anula el `border-left` de la fila. "
        "Quedaría una regla vertical suelta a la izquierda de la fila 2."
    )
    assert re.search(r"text-align:\s*left", joined), (
        "P1-MOBILE-FIT: el bloque apilado no revierte el `text-align: right` "
        "que la fila usaba para la celda del lugar."
    )


# ── 2. El truncado ──────────────────────────────────────────────────────────

def test_the_step_descriptions_are_not_truncated():
    """Un recorte con «…» y sin expansor es contenido inaccesible: promete un
    resto que nadie puede cobrar. Medido a 360px, escondía 12 líneas de 4
    descripciones (18 a 320px)."""
    css = _strip_comments(_HOWITWORKS_CSS.read_text(encoding="utf-8"))
    prop = "-webkit-line" + "-clamp"  # partido: ver la nota de método del módulo
    for body in _rule_body(css, ".cellDesc"):
        assert prop not in body, (
            "P1-MOBILE-FIT: `.cellDesc` vuelve a recortar líneas. Las 4 "
            "descripciones piden 6, 9, 5 y 4 líneas a 360px; recortarlas a 3 "
            "esconde 12 sin ningún control que las pida. El coste de mostrarlas "
            "son +279px sobre 10.598 (+2,6%), medido, no estimado."
        )


def test_a_fold_is_only_legitimate_when_something_can_unfold_it():
    """LA REGLA GENERAL detrás del caso de arriba, y la que se va a erosionar.

    Recortar texto no está prohibido; recortarlo SIN CONTROL sí. La metodología
    del benchmark (663 caracteres = 20 líneas a 320px) se pliega a 4 en móvil, y
    es legítimo porque un botón la devuelve entera. Si alguien deja el recorte y
    se lleva el botón, esto cae — que es el único momento en que el plegado se
    convierte en el defecto que este P-fix vino a quitar."""
    css = _strip_comments(_BENCHMARK_CSS.read_text(encoding="utf-8"))
    prop = "-webkit-line" + "-clamp"
    recorta = any(prop in body for body in _rule_body(css, ".footnoteText"))
    if not recorta:
        return  # sin recorte no hay nada que exigir
    assert _rule_body(css, ".footnoteToggle"), (
        "P1-MOBILE-FIT: la metodología se recorta pero su control desapareció. "
        "Un recorte sin expansor promete un resto que nadie puede cobrar — es "
        "exactamente el defecto que este P-fix quitó de HowItWorks."
    )
    jsx = _BENCHMARK_JSX.read_text(encoding="utf-8")
    assert "aria-expanded" in jsx and "aria-controls" in jsx, (
        "P1-MOBILE-FIT: el control del plegado perdió su semántica. Sin "
        "`aria-expanded`/`aria-controls`, un lector de pantalla anuncia un botón "
        "que no dice qué gobierna ni si está abierto."
    )


# ── 3. Los objetivos táctiles ───────────────────────────────────────────────

def _has_touch_pseudo(css: str, selector: str) -> bool:
    """El selector se posiciona y tiene un `::after` absoluto con inset negativo
    — la técnica que agranda el área sin mover la tinta."""
    positioned = any(
        re.search(r"position:\s*relative", b) for b in _rule_body(css, selector)
    )
    pseudo = _rule_body(css, f"{selector}::after")
    grown = any(
        re.search(r"position:\s*absolute", b) and re.search(r"inset:\s*-", b)
        for b in pseudo
    )
    return positioned and grown


def test_see_more_link_is_reachable_with_a_thumb():
    """32px de alto × 4 instancias (HowItWorks, DashboardShowcase,
    BenchmarkShowcase, NewsHighlight). Se agranda con pseudo-elemento y no con
    padding: el `border-bottom` de este enlace ES su subrayado, así que padding
    lo desplazaría, y el anillo de `:focus-visible` dejaría de ceñir el texto."""
    css = _strip_comments(_SEEMORE_CSS.read_text(encoding="utf-8"))
    assert _has_touch_pseudo(css, ".link"), (
        "P1-MOBILE-FIT: `SeeMoreLink` perdió su área táctil. Mide 32px de alto "
        "de tinta; el mínimo es 44 (Apple) / 48 (Google)."
    )


def test_social_icons_are_reachable_with_a_thumb():
    """29×29 bajo papel (18 de glifo + 5,6 de padding), 35,2 en el footer oscuro
    de las otras 21 rutas. Un solo `inset` cubre los dos temas."""
    css = _strip_comments(_FOOTER_CSS.read_text(encoding="utf-8"))
    assert _has_touch_pseudo(css, ".socialIcon"), (
        "P1-MOBILE-FIT: los iconos sociales perdieron su área táctil. Miden "
        "29×29 bajo papel — el footer se monta en 21 rutas, así que esto no es "
        "solo del landing."
    )


# ── Nomenclatura: un título regulado no nombra una feature ──────────────────

def test_no_regulated_title_names_a_feature():
    """Directiva permanente del dueño: es creador único y sin credenciales
    clínicas, así que ninguna feature puede llamarse con un título regulado.
    «Nutricionista IA» vivía en DOS sitios donde nombra al producto: el rótulo
    de la sección 04 del landing —contradiciendo a su propia tarjeta, que ya
    decía la fórmula correcta 20px más abajo— y el título de respaldo de las
    notificaciones push, que es lo que se lee en una pantalla de bloqueo, sin
    nada alrededor que lo matice.

    LO QUE ESTE TEST NO PROHÍBE, y por eso mira solo dos ficheros: el sustantivo
    común. «Consulta con tu nutricionista» en un aviso legal o en un banner
    clínico es correcto y necesario — ahí la palabra señala a un profesional
    HUMANO al que remitimos, que es justo lo contrario de apropiarse del título.
    Y `utils/recipeSteps.js` la usa para PARSEAR texto que genera el backend:
    renombrarla ahí no cambia nomenclatura, rompe el parser."""
    prohibido = re.compile(r"nutricionista\s+(ia|ai)\b", re.IGNORECASE)
    for ruta in (_SHOWCASE_JSX, _SERVICE_WORKER):
        limpio = _strip_js_comments(ruta.read_text(encoding="utf-8"))
        assert not prohibido.search(limpio), (
            f"P1-MOBILE-FIT: {ruta.name} vuelve a nombrar una feature con un "
            "título regulado. La fórmula acordada describe la función sin "
            "reclamar la credencial."
        )


# ── 4. La guarda que faltaba ────────────────────────────────────────────────

def test_the_real_measurement_guard_still_exists():
    """Este fichero es un parser: ancla decisiones, no mide anchos. Quien mide
    es el e2e. Si alguien lo borra, el defecto 1 puede volver sin que ninguna
    suite se entere — que es exactamente como llegó a producción."""
    assert _E2E_SPEC.exists(), (
        f"P1-MOBILE-FIT: falta {_E2E_SPEC.name}. Es la ÚNICA sonda que mide el "
        "landing a 320px de verdad; sin ella este fichero ancla decisiones que "
        "nadie comprueba contra un navegador."
    )
    spec = _E2E_SPEC.read_text(encoding="utf-8")
    assert str(_CONTRACT_WIDTH) in spec, (
        f"P1-MOBILE-FIT: el e2e ya no mide a {_CONTRACT_WIDTH}px, que es el "
        "ancho de contrato más estrecho (un iPhone con Display Zoom activado "
        "renderiza ahí)."
    )
