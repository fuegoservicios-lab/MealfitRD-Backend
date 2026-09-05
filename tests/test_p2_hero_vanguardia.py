"""[P2-HERO-VANGUARDIA · 2026-09-05] El rótulo comercial del landing, y lo que
cuesta en píxeles.

La celda 1 del cajetín del hero decía «NUTRICIÓN DE PRECISIÓN» y ahora dice
«NUTRICIÓN DE VANGUARDIA». El cambio es de registro: «precisión» es la
categoría técnica y sigue nombrándose donde describe (el pie, /about); en la
primera línea de la página manda el gancho.

POR QUÉ ESTE FICHERO EXISTE. El cambio vale UN carácter, y en esa fila un
carácter no es gratis. `P1-MOBILE-FIT` midió que las dos celdas que sobreviven
en móvil pedían 180,4 + 195,3 = 375,7 px y que por eso el cajetín apila bajo un
corte elegido con 24 px de cushion sobre el punto exacto donde deja de caber.
Ese cushion es un número escrito en un comentario: nada lo defendía. La
regresión de agosto entró justo así — se cambió el literal de la celda 1, la
nota de medición no se actualizó y a 320 px se cortaban 55,7 px SIN scroll que
lo revelara, porque html/body llevan `overflow-x: clip`.

QUÉ ANCLA: la aritmética, no el literal. Si mañana alguien alarga cualquiera de
las dos celdas —o baja el corte del apilado— y la pareja se come el cushion,
este test cae antes que producción. Cubre la clase entera, no la instancia.

QUÉ NO PUEDE VER: es un parser, no un navegador. El paso por carácter sale
calibrado de las dos mediciones que `P1-MOBILE-FIT` dejó escritas con emulación
real de dispositivo —(180,4−28)/22 = 6,93 y (195,3−28)/24 = 6,97 px— que
coinciden a 0,04 px. La medición REAL sigue viviendo en
`frontend/e2e/mobile_no_overflow.spec.js`.

Tooltip-anchor: P2-HERO-VANGUARDIA
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "frontend" / "src"
_HERO_JSX = _SRC / "components" / "home" / "Hero.jsx"
_HERO_CSS = _SRC / "components" / "home" / "Hero.module.css"
_FOOTER_JSX = _SRC / "components" / "layout" / "Footer.jsx"
_LOCALES_DIR = _SRC / "i18n" / "locales"

# Los cuatro catálogos. es-DO NO lleva fichero a propósito: la clave ES el texto
# español, así que la base son 0 bytes (ver CLAUDE.md → P1-I18N-DASHBOARD).
_LOCALES = ("en-US", "pt-BR", "fr-FR", "it-IT")

# Calibración de `P1-MOBILE-FIT`, medida con emulación real de dispositivo:
# 11 px, letter-spacing 0,08em, padding 0,875rem a cada lado.
_PX_PER_CHAR = 6.95
_CELL_PADDING_PX = 28.0

# El cushion que el bloque móvil de Hero.module.css declara por escrito. No es
# «lo que cabe»: es lo que sobra para no quedar a merced de cómo renderice la
# fuente cada dispositivo.
_MIN_CUSHION_PX = 24.0
_CONTRACT_WIDTH = 320


def _fold(text: str) -> str:
    """Minúsculas sin acentos. El francés obliga: buscar el rastro de
    «precisión» con una subcadena literal deja pasar «précision», que es
    justo el idioma donde nadie de este equipo lo leería."""
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfd if not unicodedata.combining(c))


def _strip_comments(src: str) -> str:
    """Bloque y línea. Este repo documenta en el propio fichero qué literal se
    retiró; un escáner que lea las notas acusa al arreglo de ser el defecto."""
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL))


def _cartridge_cells() -> list[str]:
    """Los literales de texto plano del cajetín, en orden.

    La celda del medio no matchea a propósito: su contenido sale de
    `MICROS_TRACKED` por template literal y su className es compuesto. Es
    también la que se oculta bajo 600 px, así que no participa de la pareja.
    """
    jsx = _strip_comments(_HERO_JSX.read_text(encoding="utf-8"))
    return re.findall(r"<span className=\{styles\.cartridgeCell\}>([^<{]+)</span>", jsx)


def _media_blocks(css: str) -> list[tuple[int | None, str]]:
    """[(max-width, cuerpo)] emparejando llaves de verdad.

    Recortar N caracteres tras el `@media` NO sirve: la primera versión de este
    parser leía 400 y se comía el bloque siguiente, así que el bloque de 599 px
    «contenía» la regla de apilado del de 409 y el test daba verde contra una
    mutación que debía tumbarlo. Un parser que no puede fallar no informa.
    """
    out: list[tuple[int | None, str]] = []
    for m in re.finditer(r"@media[^{]*\{", css):
        mw = re.search(r"max-width:\s*(\d+)px", m.group(0))
        depth, i = 1, m.end()
        while i < len(css) and depth:
            depth += (css[i] == "{") - (css[i] == "}")
            i += 1
        out.append((int(mw.group(1)) if mw else None, css[m.end(): i - 1]))
    return out


def _stacking_breakpoint() -> int:
    """El `max-width` del @media donde `.cartridge` pasa a una sola columna."""
    css = _strip_comments(_HERO_CSS.read_text(encoding="utf-8"))
    hits = [
        mw for mw, body in _media_blocks(css)
        if mw is not None and any(
            re.search(r"grid-template-columns:\s*1fr\s*;", rule)
            for rule in re.findall(r"\.cartridge\s*\{([^{}]*)\}", body)
        )
    ]
    assert len(hits) == 1, (
        "P2-HERO-VANGUARDIA: se esperaba UN @media donde `.cartridge` apile a "
        f"una columna y se encontraron {hits!r}. Si el selector o la propiedad "
        "cambiaron de nombre, este test deja de medir nada — arregla el parser "
        "antes que producción."
    )
    return hits[0]


def _cell_width_px(text: str) -> float:
    return len(text) * _PX_PER_CHAR + _CELL_PADDING_PX


# ── 1. El rótulo ────────────────────────────────────────────────────────────

def test_the_cartridge_leads_with_the_commercial_label():
    """La celda 1 rotula qué es esto, y desde hoy lo hace en registro comercial."""
    cells = _cartridge_cells()
    assert len(cells) == 2, (
        "P2-HERO-VANGUARDIA: se esperaban 2 celdas de texto plano en el cajetín "
        f"(qué-es y dónde) y se encontraron {len(cells)}: {cells!r}. La del medio "
        "no cuenta — sale de MICROS_TRACKED por template literal."
    )
    assert cells[0] == "NUTRICIÓN DE VANGUARDIA", (
        "P2-HERO-VANGUARDIA: la celda 1 del cajetín dice "
        f"{cells[0]!r}. Si es un cambio deliberado de copy, actualiza este test "
        "Y vuelve a correr la aritmética de abajo: el ancho móvil depende de su "
        "longitud, y esa es exactamente la comprobación que se saltó la "
        "regresión de P1-MOBILE-FIT."
    )


def test_precision_survives_where_it_describes():
    """«De precisión» sale del gancho, no del vocabulario. Sigue siendo la
    categoría técnica en el pie y en /about — retirarla de todas partes
    cambiaría de qué dice el producto que es, que no es lo que se pidió."""
    about = (_SRC / "pages" / "AboutPage.jsx").read_text(encoding="utf-8")
    assert "Precisión nutricional" in about, (
        "P2-HERO-VANGUARDIA: el H1 de /about ya no reivindica precisión. Es la "
        "afirmación que el motor respalda con números medidos (banda de macros, "
        "piso de proteína, micronutrientes vs DRI); «vanguardia» no respalda "
        "nada. El gancho se cambió; la evidencia no se retira."
    )


# ── 2. Lo que el carácter extra cuesta ──────────────────────────────────────

def test_the_mobile_pair_keeps_the_cushion_it_declares():
    """Entre el corte del apilado y 599 px el cajetín es UNA fila de dos celdas.
    Esa fila tiene que caber en el ancho más estrecho de esa banda con el
    cushion que el CSS declara — no «casi caber», que es lo que este repo ya
    rechazó por escrito cuando midió 0,3 px de holgura."""
    cells = _cartridge_cells()
    pair_px = sum(_cell_width_px(c) for c in cells)
    narrowest = _stacking_breakpoint() + 1  # el primer ancho que NO apila
    cushion = narrowest - pair_px

    assert cushion >= _MIN_CUSHION_PX, (
        f"P2-HERO-VANGUARDIA: a {narrowest}px la fila pide {pair_px:.1f}px y solo "
        f"sobran {cushion:.1f}px, por debajo de los {_MIN_CUSHION_PX:.0f}px que el "
        "bloque móvil de Hero.module.css declara.\n"
        f"Celdas: {[(c, round(_cell_width_px(c), 1)) for c in cells]}\n"
        "Salidas, en el orden en que este repo ya las evaluó: subir el "
        "`max-width` del apilado (solo afecta a la banda que dejas de cubrir, "
        "donde no aterriza ningún teléfono común), o comprar aire en el "
        "tracking/padding del bloque móvil — NUNCA bajando de 11px, que es el "
        "piso tipográfico sin excepciones de rol."
    )


def test_each_stacked_row_fits_the_contract_width():
    """Bajo el corte el cajetín apila, y entonces cada celda es una fila suya.
    La más ancha tiene que caber a 320 px — un iPhone con Display Zoom."""
    widest = max(_cell_width_px(c) for c in _cartridge_cells())
    assert widest <= _CONTRACT_WIDTH, (
        f"P2-HERO-VANGUARDIA: apilado, la celda más ancha pide {widest:.1f}px y el "
        f"suelo del contrato son {_CONTRACT_WIDTH}px. Apilar ya no salva este "
        "literal: hay que acortarlo."
    )


# ── 3. El pie: cambiar el copy huérfana su traducción en silencio ───────────

def test_the_footer_line_is_translated_in_every_catalog():
    """La clave ES el texto español. Cambiarlo sin reescribir la clave en los
    cuatro catálogos deja la traducción huérfana y el pie cae a español en
    inglés, portugués, francés e italiano — sin error, sin log, sin nada."""
    footer = _FOOTER_JSX.read_text(encoding="utf-8")
    literal = re.search(r"\{t\('(Nutrición de [^']+?)'\)\}", footer)
    assert literal, (
        "P2-HERO-VANGUARDIA: no se encontró la línea `t('Nutrición de …')` del "
        "pie. Si se retiró o se partió en varias claves, actualiza este test."
    )
    key = literal.group(1)

    for loc in _LOCALES:
        catalog = json.loads((_LOCALES_DIR / f"{loc}.json").read_text(encoding="utf-8"))
        assert key in catalog, (
            f"P2-HERO-VANGUARDIA: {loc}.json no tiene la clave del pie:\n  {key!r}\n"
            "Es el modo de fallo silencioso del motor i18n propio: sin clave, el "
            "texto cae a español y nadie se entera. Reescribe la clave Y el valor "
            "(`npm run i18n:check` también lo paga)."
        )
        assert "precis" not in _fold(catalog[key]), (
            f"P2-HERO-VANGUARDIA: la traducción de {loc} sigue diciendo «precisión» "
            f"mientras el español dice «vanguardia»:\n  {catalog[key]!r}\n"
            "Reescribir la clave sin traducir el valor deja los cinco idiomas "
            "diciendo dos cosas distintas."
        )


def test_no_catalog_keeps_the_retired_key():
    """La clave vieja no puede sobrevivir en ningún catálogo: sería una entrada
    que ya nadie pide, y el próximo que audite el i18n no sabría si es deuda o
    una superficie que se le escapó."""
    retired = (
        "Nutrición de precisión potenciada por Inteligencia Artificial. "
        "Tu camino hacia una vida más saludable empieza aquí."
    )
    for loc in _LOCALES:
        raw = (_LOCALES_DIR / f"{loc}.json").read_text(encoding="utf-8")
        assert retired not in json.loads(raw), (
            f"P2-HERO-VANGUARDIA: {loc}.json conserva la clave retirada del pie. "
            "Bórrala: una traducción huérfana no falla, solo miente en la "
            "próxima auditoría."
        )
