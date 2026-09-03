"""[P1-HERO-DEDUP-ACCENT · 2026-08-09] El hero decía cuatro veces lo mismo.

4 de las 5 celdas de la franja eran una reformulación en mono de los dos
párrafos que tenían encima; `BIOBOROS` salía en el wordmark y otra vez 40 px
debajo; y `Crear mi Plan Ahora` aparecía idéntico en header y hero sin
scrollear.

El acento `--pa-accent` marca LA CIFRA de un SSOT — condición necesaria, no
suficiente. Es la parte del diseño que un editor futuro erosionaría primero
(un rojo es cómodo de reutilizar), así que va anclada, no solo comentada.

Tooltip-anchor: P1-HERO-DEDUP-ACCENT
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "frontend" / "src"
_INDEX_CSS = _SRC / "index.css"
_HERO_JSX = _SRC / "components" / "home" / "Hero.jsx"
_HERO_CSS = _SRC / "components" / "home" / "Hero.module.css"
_FIG_JSX = _SRC / "components" / "home" / "figures" / "PlateExploded.jsx"
_FIG_CSS = _SRC / "components" / "home" / "figures" / "PlateExploded.module.css"

_ACCENT = "#C1200E"


def test_accent_token_declared_once_with_exact_value():
    """Un segundo sitio de declaración es un segundo sitio de drift: el día que
    alguien retoque el rojo, la mitad del landing se queda en el viejo."""
    css = _INDEX_CSS.read_text(encoding="utf-8")
    decls = re.findall(r"--pa-accent\s*:\s*([^;]+);", css)
    assert len(decls) == 1, (
        f"P1-HERO-DEDUP-ACCENT: --pa-accent declarado {len(decls)} vez/veces "
        f"en index.css: {decls}. Debe declararse exactamente una."
    )
    assert decls[0].strip().upper() == _ACCENT, (
        f"P1-HERO-DEDUP-ACCENT: --pa-accent vale {decls[0].strip()!r}, se esperaba "
        f"{_ACCENT}. Ese valor no es estético: da 5,83:1 contra --pa-paper en AMBAS "
        "direcciones (AA texto normal como tinta y como contratinta). #D6301A pasa "
        "por 0,22 y #E24A15 no pasa."
    )


def test_accent_declared_inside_the_paper_block():
    """Fuera del bloque papel el token aplicaría también a dashboard/light/dark,
    donde no hay sistema que lo dote de significado."""
    css = _INDEX_CSS.read_text(encoding="utf-8")
    start = css.index('html[data-theme="paper"]')
    end = css.index("}", css.index("--pa-grid-major", start))
    assert "--pa-accent" in css[start:end], (
        "P1-HERO-DEDUP-ACCENT: --pa-accent no está dentro del bloque de tokens de "
        'html[data-theme="paper"] en index.css.'
    )


def test_figure_cotas_carry_the_accent():
    """Las dos cotas son el único sitio de la figura donde el acento es legítimo:
    ya significan «esto lo medimos»."""
    jsx = _FIG_JSX.read_text(encoding="utf-8")
    assert "styles.cota" in jsx, (
        "P1-HERO-DEDUP-ACCENT: el <g> de COTAS de PlateExploded.jsx debe llevar "
        "className={styles.cota} para que el acento se scopee a él."
    )
    css = _FIG_CSS.read_text(encoding="utf-8")
    scoped = re.findall(r"\.cota\s+\.(stroke|arrow|value)\s*\{[^}]*--pa-accent[^}]*\}", css)
    assert len(scoped) == 3, (
        "P1-HERO-DEDUP-ACCENT: se esperan 3 reglas `.cota .stroke|.arrow|.value` "
        f"con var(--pa-accent) en PlateExploded.module.css; hay {len(scoped)}. "
        "Geometría, flechas y valores de la cota van los tres al acento o la cota "
        "queda a dos tintas."
    )


def test_accent_never_touches_the_material_encoding():
    """`sólido/contorno/trama45°/punteado` codifican material en 7 superficies del
    sistema. Teñir una pieza rompe la codificación, no solo esta figura."""
    css = _FIG_CSS.read_text(encoding="utf-8")
    for cls in ("hatchFill", "solidFill", "guide", "node", "hatchLine", "edgeLight"):
        block = re.search(rf"^\.{cls}\s*\{{([^}}]*)\}}", css, re.MULTILINE)
        assert block, f"P1-HERO-DEDUP-ACCENT: no encuentro la regla .{cls} en {_FIG_CSS.name}"
        assert "--pa-accent" not in block.group(1), (
            f"P1-HERO-DEDUP-ACCENT: .{cls} usa el acento. Esa clase codifica MATERIAL "
            "(§5.5), no una cifra — el acento solo va sobre las cotas."
        )


# ── De-duplicación (D1-D7 del spec) ──────────────────────────────────────────

def test_hero_does_not_repeat_the_wordmark():
    """D1: `BIOBOROS` salía en el wordmark del header y otra vez 40 px debajo,
    en la celda 1 del cartucho."""
    jsx = _HERO_JSX.read_text(encoding="utf-8")
    assert "BIOBOROS" not in jsx, (
        "P1-HERO-DEDUP-ACCENT: `BIOBOROS` volvió a Hero.jsx. El wordmark del "
        "header lo dice 40 px más arriba; el cartucho tiene 3 celdas para decir "
        "3 cosas distintas."
    )


def test_hero_cta_is_not_the_headers_twin():
    """D2: el CTA del header es permanente en landing (decisión del dueño,
    P3-HEADER-FLOAT-REDESIGN). No se revierte — lo que se rompe es que el hero
    repita su literal exacto en la misma pantalla.

    ESCANEA EL FICHERO ENTERO, comentarios incluidos, igual que
    `test_p1_paper_hero_fig00.py`. El primer intento traía un filtro que
    descartaba líneas de comentario, y no funcionaba: las líneas de
    continuación de un bloque `/* … */` empiezan por espacios, no por `*`.
    Un stripper de comentarios a medias es peor que ninguno — da la
    impresión de precisión y deja pasar justo el caso que motiva el test.
    La convención del repo (y la de la cabecera de `Hero.jsx`) es la
    contraria y es más simple: al documentar lo que se borró, DESCRÍBELO,
    no lo cites."""
    jsx = _HERO_JSX.read_text(encoding="utf-8")
    assert "Crear mi Plan Ahora" not in jsx, (
        "P1-HERO-DEDUP-ACCENT: el hero volvió al literal exacto del header "
        "('Crear mi Plan Ahora'). Dos rectángulos negros idénticos sin scrollear "
        "es el defecto que este P-fix cerró. El literal del hero es 'Crear mi plan'."
    )


def test_the_strip_that_repeated_the_paragraph_is_gone():
    """D3-D6: 4 de las 5 celdas eran una reformulación en mono del párrafo que
    tenían encima. La franja no añadía información, añadía líneas."""
    jsx = _HERO_JSX.read_text(encoding="utf-8")
    for label in ("MÉTODO", "PERFIL", "REVISIÓN", "COCINA"):
        assert label not in jsx, (
            f"P1-HERO-DEDUP-ACCENT: volvió la celda `{label}` de la franja. Su "
            "contenido ya lo dice el párrafo — ver la tabla D3-D6 del spec."
        )
    assert "const STRIP" not in jsx, (
        "P1-HERO-DEDUP-ACCENT: volvió `const STRIP`. El dato que sí valía "
        "(los créditos) vive ahora en la línea `.datum`."
    )


def test_hero_still_derives_both_numbers_from_their_ssot():
    """El de-drift no puede pagarse con la de-duplicación: al borrar la franja
    es tentador escribir el número a mano. `test_p1_landing_bench_1_anchors`
    ya vigila TIER_CREDITS; esto añade MICROS_TRACKED y lo hace explícito."""
    jsx = _HERO_JSX.read_text(encoding="utf-8")
    assert "TIER_CREDITS" in jsx, (
        "P1-HERO-DEDUP-ACCENT: Hero.jsx dejó de derivar los créditos de "
        "config/plans.js. test_p1_landing_bench_1_anchors.py:384-388 también cae."
    )
    assert "MICROS_TRACKED" in jsx, (
        "P1-HERO-DEDUP-ACCENT: la celda del cartucho debe derivar el 17 de "
        "data/systemFacts.js, no escribirlo a mano."
    )
    assert not re.search(r"\b17 MICRONUTRIENTES", jsx), (
        "P1-HERO-DEDUP-ACCENT: el 17 está escrito a mano en el cartucho."
    )


def test_hero_has_a_single_body_paragraph():
    """Los dos párrafos decían ambos «qué recibes». Se fusionaron en uno que
    además absorbe las 4 celdas borradas."""
    css = _HERO_CSS.read_text(encoding="utf-8")
    assert not re.search(r"^\.promise\s*\{", css, re.MULTILINE), (
        "P1-HERO-DEDUP-ACCENT: `.promise` sigue en Hero.module.css. El párrafo se "
        "fusionó con `.lead` — una clase sin consumidor invita a recablearla."
    )
    for cls in ("strip", "stripCell", "stripLabel", "stripValue"):
        assert not re.search(rf"^\.{cls}\s*[,{{]", css, re.MULTILINE), (
            f"P1-HERO-DEDUP-ACCENT: `.{cls}` sigue en Hero.module.css tras borrar "
            "la franja."
        )


def test_accent_stays_off_the_controls_and_the_cartridge():
    """LA REGLA QUE MÁS SE VA A EROSIONAR. Los casos de arriba se autoprotegen
    (si vuelve el duplicado, caen), pero «el acento solo sobre cifras» es una
    convención — y una convención sin ancla dura hasta el siguiente con prisa.

    El CTA es el caso importante: es el sitio más tentador para un rojo y el
    peor. El botón ya es el único rectángulo de tinta sólida de la pantalla;
    teñirlo REBAJA la jerarquía de tinta en vez de subirla, y contradice el
    §1.1 el primer día (un control no es una cifra)."""
    css = _HERO_CSS.read_text(encoding="utf-8")
    for cls in ("primaryBtn", "ghostBtn", "cartridgeCell", "title", "lead"):
        for block in re.findall(rf"^\.{cls}(?:[:\s,][^{{]*)?\{{([^}}]*)\}}", css, re.MULTILINE):
            assert "--pa-accent" not in block, (
                f"P1-HERO-DEDUP-ACCENT: `.{cls}` usa el acento. Solo lo llevan las "
                "cotas de la Fig. 00 y el numeral de créditos (`.datumNum`). "
                "Ver §1.4 del spec."
            )
    assert re.search(r"^\.datumNum\s*\{[^}]*--pa-accent", css, re.MULTILINE | re.DOTALL), (
        "P1-HERO-DEDUP-ACCENT: `.datumNum` debe llevar el acento — es el segundo "
        "y último call site."
    )


# ── Escala (§3 del spec) ─────────────────────────────────────────────────────

def test_title_grows_at_the_top_but_not_at_the_bottom():
    """El diagnóstico era que el H1 no dominaba su propia pantalla: 64 px a peso
    400 con tres franjas de mono gris compitiendo.

    El PISO se queda en 2.5rem y no es timidez: `Nutrición calculada,` son 20
    caracteres y a 375 px solo hay 327 px de columna. Subir el piso desborda la
    línea. Lo que sobraba era techo, no suelo."""
    css = _HERO_CSS.read_text(encoding="utf-8")
    m = re.search(r"^\.title\s*\{([^}]*)\}", css, re.MULTILINE)
    assert m, "P1-HERO-DEDUP-ACCENT: no encuentro la regla .title"
    block = m.group(1)
    clamp = re.search(r"font-size:\s*clamp\(([^,]+),([^,]+),([^)]+)\)", block)
    assert clamp, "P1-HERO-DEDUP-ACCENT: .title debe seguir usando clamp()"
    floor, _, cap = (v.strip() for v in clamp.groups())
    assert floor == "2.5rem", (
        f"P1-HERO-DEDUP-ACCENT: el piso del titular es {floor}, debe ser 2.5rem. "
        "A 375 px no caben 20 caracteres por encima de eso."
    )
    assert cap == "6.5rem", (
        f"P1-HERO-DEDUP-ACCENT: el techo del titular es {cap}, debe ser 6.5rem "
        "(era 4rem = 64 px, que en escritorio no dominaba la pantalla)."
    )
    assert "font-weight: 500" in block, (
        "P1-HERO-DEDUP-ACCENT: el titular debe ir a peso 500 (era 400)."
    )


def test_the_title_balances_its_lines_instead_of_breaking_by_hand():
    """`text-wrap: balance` es LOAD-BEARING, no cosmético: quitarlo devuelve el
    corte natural y con él la huérfana.

    El titular llevaba un `<br />` fijo. A 104 px eso dejaba «no» solo en su
    propia línea — el pivote de la afirmación, en el peor sitio. Y NO se
    arregla con un techo más bajo: MEDIDO, «no improvisada» pide 6,36 × el
    tamaño de fuente contra una columna de 625 px, así que con `8.2vw` esa
    línea solo cabe por encima de ~1209 px de viewport; y a 1200 px exactos la
    columna ENCOGE 46 px (el padding del contenedor salta de 2rem a 4rem)
    mientras la fuente sigue creciendo. No hay número fijo que estabilice el
    corte en toda la banda.

    Reparto verificado con Range.getClientRects() en 320/375/900/1024/1200/
    1440/1920: 2-3 líneas, cero desbordes, cero scroll horizontal."""
    css = _HERO_CSS.read_text(encoding="utf-8")
    m = re.search(r"^\.title\s*\{([^}]*)\}", css, re.MULTILINE)
    assert m and "text-wrap: balance" in m.group(1), (
        "P1-HERO-DEDUP-ACCENT: `.title` perdió `text-wrap: balance`. Sin él el "
        "titular vuelve al corte natural y a la línea huérfana."
    )
    jsx = _HERO_JSX.read_text(encoding="utf-8")
    h1 = re.search(r"<motion\.h1[^>]*>(.*?)</motion\.h1>", jsx, re.DOTALL)
    assert h1, "P1-HERO-DEDUP-ACCENT: no encuentro el <motion.h1> del hero"
    assert "<br" not in h1.group(1), (
        "P1-HERO-DEDUP-ACCENT: volvió un salto de línea fijo dentro del titular. "
        "Un corte a mano compite con `balance` y reintroduce la huérfana en la "
        "banda donde la línea no cabe."
    )


def test_the_rule_under_the_title_is_a_rule_not_a_hairline():
    css = _HERO_CSS.read_text(encoding="utf-8")
    m = re.search(r"^\.titleRule\s*\{([^}]*)\}", css, re.MULTILINE)
    assert m, "P1-HERO-DEDUP-ACCENT: no encuentro la regla .titleRule"
    assert "height: 3px" in m.group(1), (
        "P1-HERO-DEDUP-ACCENT: .titleRule debe medir 3px. A 1px remataba un "
        "titular de 104 px con una hairline gris — subrayado tímido."
    )
    assert "transform-origin: left" in m.group(1), (
        "P1-HERO-DEDUP-ACCENT: .titleRule perdió su transform-origin. El trazado "
        "scaleX de 520 ms se dibujaría desde el centro."
    )


def test_figure_and_caption_grow_together():
    """`.caption` tiene max-width igual al encuadre del dibujo a propósito: su
    regla superior tiene que coincidir con el ancho de la figura. Subir una sin
    la otra deja el pie desalineado — y es el fallo que no se ve en el diff."""
    fig = _FIG_CSS.read_text(encoding="utf-8")
    m = re.search(r"^\.fig00\s*\{([^}]*)\}", fig, re.MULTILINE)
    assert m and "max-width: 560px" in m.group(1), (
        "P1-HERO-DEDUP-ACCENT: .fig00 debe topar en 560px (era 420px)."
    )
    css = _HERO_CSS.read_text(encoding="utf-8")
    cap = re.search(r"^\.caption\s*\{([^}]*)\}", css, re.MULTILINE)
    assert cap and "max-width: 560px" in cap.group(1), (
        "P1-HERO-DEDUP-ACCENT: .caption debe topar en 560px, igual que .fig00. "
        "El comentario de esa regla explica por qué van atadas."
    )
