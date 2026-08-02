"""[P2-PAPER-NO-INK · 2026-08-01] Paridad de bloques de tema.

(i) Ningún `:global(html[data-theme="dark"])` REAL sobrevive en
`components/home/*.module.css` — esas secciones ya solo se renderizan bajo
`paper` (la home no lleva toggle de tema), así que un bloque `dark` ahí es
código que nadie sabe si aplica: ni el navegador lo dispara nunca, ni un
lector puede saber si sigue vivo o es basura de una migración a medias.

(ii) `Header.module.css` y `Footer.module.css` (los dos módulos COMPARTIDOS
que se montan en las 21 rutas de la app, no solo en las 6 de papel) tienen
bloque `paper` en cuanto tienen bloque `dark`: si una clase tiene regla
`dark` y no tiene su equivalente `paper`, en las rutas de papel esa clase cae
a su base clara — que en el header es invisible (contraste medido 1.027:1
en una revisión anterior de este mismo rediseño).

Por qué (i) filtra comentarios y el brief original NO lo hacía:
    `DashboardShowcase.module.css` lleva, a propósito, un bloque de comentario
    `/* ... */` que documenta qué bloques `data-theme="dark"` existían ANTES
    del rediseño y por qué se borraron (incluye la línea literal
    `` `html[data-theme="dark"] .featureItem` `` dentro de backticks, como
    prosa). Un chequeo de substring ingenuo sobre el texto crudo del archivo
    encuentra esa cadena y marca el archivo como "todavía tiene bloque dark"
    — un falso positivo, y exactamente la misma clase de error que
    `test_p2_paper_no_ink.py` ya documenta haber sufrido en este repo
    (2026-07-31: un test falló contra su propio arreglo porque el comentario
    que explicaba el fix contenía la cadena prohibida). Verificado
    EJECUTANDO antes de fijar esta versión: el chequeo sin stripping falla
    hoy sobre `DashboardShowcase.module.css`; con `_strip_comments` pasa.

Por qué (i) YA NO excluye `Pricing.module.css` [P2-PAPER-NO-INK · 2026-08-02]:
    Mientras Task 13 del plan de rediseño ("Las páginas token-driven") no
    corrió, el archivo retenía DOS bloques `:global(html[data-theme="dark"])`
    reales — no en comentario — heredados del tema oscuro pre-rediseño
    (`.pricing::after` y `.btnOutline:hover`), y quedaba excluido con el mismo
    criterio que `test_p2_paper_no_ink.py::_PENDING_REWORK`. Task 13 reescribió
    el módulo a papel y borró los dos bloques en el mismo commit, así que la
    exclusión desaparece: `_PENDING_REWORK` queda VACÍO aquí. Una excepción
    documentada que ya no aplica es peor que ninguna — invita a reañadir un
    bloque dark citando la doc (mismo razonamiento que I6 del CLAUDE.md).

Tooltip-anchor: P2-PAPER-NO-INK
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "frontend" / "src"
_HOME = _SRC / "components" / "home"
_SHARED = [
    _SRC / "components" / "layout" / "Header.module.css",
    _SRC / "components" / "layout" / "Footer.module.css",
]

# [P2-PAPER-NO-INK · 2026-08-02] VACÍO a propósito. Contenía
# `Pricing.module.css` mientras Task 13 ("Las páginas token-driven") no
# migraba el módulo a papel: retenía 2 bloques dark reales (`.pricing::after`,
# `.btnOutline:hover`). Task 13 los borró en el mismo commit que reescribió el
# archivo, así que la entrada se retira. El set se conserva (en vez de borrar
# el mecanismo) porque Task 11 podría necesitarlo para BenchmarkShowcase si esa
# sección acaba llevando bloque dark — hoy no lo lleva.
_PENDING_REWORK: set[Path] = set()

# Mismo par de regexes y misma razón que test_p2_paper_no_ink.py::_strip_comments:
# un comentario que EXPLICA un bloque dark borrado no es un bloque dark vivo.
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"^\s*//.*$", re.MULTILINE)


def _strip_comments(text: str) -> str:
    text = _CSS_COMMENT.sub(lambda m: " " * len(m.group(0)), text)
    return _LINE_COMMENT.sub(lambda m: " " * len(m.group(0)), text)


def test_no_dark_blocks_left_in_home_modules():
    offenders = []
    for f in _HOME.rglob("*.module.css"):
        if f in _PENDING_REWORK:
            continue
        clean = _strip_comments(f.read_text(encoding="utf-8"))
        if 'data-theme="dark"' in clean:
            offenders.append(f.relative_to(_REPO_ROOT))
    assert not offenders, (
        f"P2-PAPER-NO-INK: quedan bloques dark REALES (no en comentario) en "
        f"components/home: {offenders}. Esas secciones solo se renderizan "
        "bajo `paper`; el bloque dark es codigo que nadie sabe si aplica. "
        "Si el archivo es un verdadero pendiente de otra tarea del plan "
        "(ver Pricing.module.css / Task 13), documentarlo en _PENDING_REWORK "
        "con su razon — no borrar este assert."
    )


def test_comment_stripper_prevents_false_positive_on_explanatory_comments():
    """Ancla la razón de (i): sin stripping, un comentario que documenta un
    bloque dark borrado se lee como un bloque dark vivo. Este caso reproduce
    en miniatura el patrón real de DashboardShowcase.module.css."""
    doc_comment = (
        "/* Antes existía `html[data-theme=\"dark\"] .featureItem` con "
        "override de color; se borró porque la sección ya no cambia de "
        "tema. */\n.featureItem { color: var(--pa-ink); }"
    )
    assert 'data-theme="dark"' in doc_comment, "el fixture no reproduce el patrón real"
    assert 'data-theme="dark"' not in _strip_comments(doc_comment)


def test_shared_modules_have_paper_where_they_have_dark():
    for path in _SHARED:
        text = path.read_text(encoding="utf-8")
        if 'data-theme="dark"' in text:
            assert 'data-theme="paper"' in text, (
                f"P2-PAPER-NO-INK: {path.name} tiene bloque dark y no tiene "
                "bloque paper. Se monta en 21 rutas: bajo paper esas clases "
                "caen a su base clara."
            )
