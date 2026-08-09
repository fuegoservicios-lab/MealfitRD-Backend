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
