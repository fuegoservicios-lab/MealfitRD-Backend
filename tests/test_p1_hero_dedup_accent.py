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
