# -*- coding: utf-8 -*-
"""[P1-MICRO-CRISP-SURFACES · 2026-08-12] Paridad claro↔oscuro del panel de
Micronutrientes.

El owner pidió «que se vea más nítido» en los dos temas. Medir primero dijo que
no era un color feo: el CLARO vivía a la mitad de separación que el oscuro en
todas las relaciones (chip vs panel 4,7 contra 8,1 · tarjeta vs panel 7,1 contra
11,7 · chevron 3,07:1 contra 5,06:1), y en AMBOS temas la tarjeta de atención se
disolvía por su mitad inferior porque el gradiente terminaba exactamente en el
color del panel (dL* 1,8 en claro, 0,0 en oscuro).

Este guard afirma la PARIDAD, no un número — un tema puede reafinarse mientras
arrastre a su gemelo (la lección de P1-NOTEBOOK-MARGIN-LIGHT). Recalcula de
verdad desde los tokens del DS, así que también salta si alguien cambia
`--surface-sunken` o `--border` en index.css y rompe el panel de rebote.
"""
import re
from pathlib import Path

import pytest

FRONT = Path(__file__).resolve().parents[2] / "frontend" / "src"
CSS_MOD = FRONT / "components" / "dashboard" / "MicronutrientMeter.module.css"
CSS_DS = FRONT / "index.css"

# Tono de estado «far» (el de las tarjetas del reporte): el claro lo reasigna.
TONO = {"claro": "#EA580C", "oscuro": "#FB923C"}


# ── color utils (sRGB, igual que el navegador para color-mix in srgb) ──
def _px(c):
    h = c.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _mix(c1, pct, c2):
    a = _px(c1) if isinstance(c1, str) else c1
    b = _px(c2) if isinstance(c2, str) else c2
    k = pct / 100.0
    return tuple(round(x * k + y * (1 - k)) for x, y in zip(a, b))


def _lin(v):
    v /= 255.0
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _Y(c):
    r, g, b = (_lin(v) for v in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _L(c):
    y = _Y(c)
    return 116 * (y ** (1 / 3)) - 16 if y > 0.008856 else 903.3 * y


def _ratio(a, b):
    x, y = sorted((_Y(a), _Y(b)), reverse=True)
    return (x + 0.05) / (y + 0.05)


def _tokens(tema):
    src = CSS_DS.read_text(encoding="utf-8")
    sel = ":root {" if tema == "claro" else 'html[data-theme="dark"] {'
    i = src.index(sel)
    blk = src[i:src.index("\n}", i)]
    out = {}
    for k in ("--bg-card", "--surface-sunken", "--border", "--text-main", "--text-muted", "--text-light"):
        m = re.search(re.escape(k) + r"\s*:\s*(#[0-9A-Fa-f]{6})\s*;", blk)
        if not m:
            pytest.fail(f"token {k} no resoluble como hex en el bloque {tema} de index.css")
        out[k] = m.group(1)
    return out


def _recipe(nombre, tema):
    """Lee el porcentaje de una receta color-mix del módulo, por tema."""
    src = CSS_MOD.read_text(encoding="utf-8")
    if tema == "oscuro":
        i = src.index('html[data-theme="dark"]) .panel')
        src_scope = src[i:src.index("}", i)]
    else:
        i = src.index(".panel {")
        src_scope = src[i:src.index("}", i)]
    m = re.search(re.escape(nombre) + r":\s*([^;]+);", src_scope)
    if not m:
        pytest.fail(f"{nombre} no declarado para el tema {tema} — la paridad se declara en AMBOS")
    return m.group(1).strip()


def _resolve(expr, tk):
    """Resuelve `var(--x)` o `color-mix(in srgb, var(--a) N%, var(--b))`."""
    expr = expr.strip()
    m = re.fullmatch(r"var\((--[\w-]+)\)", expr)
    if m:
        return _px(tk[m.group(1)])
    m = re.fullmatch(r"color-mix\(in srgb,\s*var\((--[\w-]+)\)\s*(\d+)%,\s*var\((--[\w-]+)\)\s*\)", expr)
    if m:
        return _mix(tk[m.group(1)], int(m.group(2)), tk[m.group(3)])
    pytest.fail(f"receta no reconocida: {expr!r}")


def _hex(rgb):
    return "#%02X%02X%02X" % tuple(rgb)


def _panel(tema):
    tk = _tokens(tema)
    sunken = _resolve(_recipe("--mn-sunken", tema), tk)
    line = _resolve(_recipe("--mn-line", tema), tk)
    ink = _resolve(_recipe("--mn-ink-soft", tema), tk)
    att_base = _resolve(_recipe("--mn-att-base", tema), {**tk, "--mn-sunken": _hex(sunken)})
    att = _mix(TONO[tema], 9, att_base)        # .att: 9% de tono sobre la base del tema
    att_border = _mix(TONO[tema], 52, tk["--border"])
    card = _px(tk["--bg-card"])
    return {
        "_L_att": _L(att),
        "_L_chip": _L(sunken),
        "chip vs panel": abs(_L(sunken) - _L(card)),
        "linea vs chip": abs(_L(line) - _L(sunken)),
        "tarjeta vs panel": abs(_L(att) - _L(card)),
        "borde tarjeta vs tarjeta": abs(_L(att_border) - _L(att)),
        "nombre sobre tarjeta": _ratio(_px(tk["--text-main"]), att),
        "apagado sobre tarjeta": _ratio(ink, att),
        "apagado sobre chip": _ratio(_px(tk["--text-muted"]), sunken),
    }


def test_paridad_claro_oscuro():
    """Cada relación mide parecido en los dos temas. El umbral (72%) permite
    afinar un tema sin arrastrar milimétricamente al otro, pero no que uno viva
    a la mitad del otro — que era el estado medido antes del fix."""
    c, o = _panel("claro"), _panel("oscuro")
    despares = []
    for k in c:
        # La tarjeta y su borde NO entran en la paridad: su magnitud y su
        # direccion las decide cada tema. En claro la tarjeta se HUNDE bajo el
        # panel blanco; en oscuro no puede hundirse mas (seria el marron sucio de
        # agosto) ni aclararse (la queja del dueno), y al quedarse oscura su
        # borde resalta MAS por consecuencia — que es el efecto buscado, no un
        # defecto. Las dos tienen abajo su regla propia: un SUELO, no un espejo.
        if k.startswith("_") or k in ("tarjeta vs panel", "borde tarjeta vs tarjeta"):
            continue
        par = min(c[k], o[k]) / max(c[k], o[k])
        if par < 0.72:
            despares.append(f"{k}: claro {c[k]:.2f} vs oscuro {o[k]:.2f} ({par*100:.0f}%)")
    assert not despares, "paridad rota:\n  " + "\n  ".join(despares)


def test_texto_del_panel_cumple_aa():
    for tema in ("claro", "oscuro"):
        m = _panel(tema)
        for k in ("nombre sobre tarjeta", "apagado sobre tarjeta", "apagado sobre chip"):
            assert m[k] >= 4.5, f"{tema} · {k} = {m[k]:.2f} (<4.5 AA)"


def test_superficies_separadas_de_verdad():
    """Piso absoluto: por debajo de dL* 6 dos superficies contiguas se funden
    (el panel entero se ve «lavado», que es el reporte original)."""
    for tema in ("claro", "oscuro"):
        m = _panel(tema)
        for k in ("chip vs panel", "tarjeta vs panel"):
            assert m[k] >= 5.5, f"{tema} · {k} = dL* {m[k]:.1f} (<5,5 = se funden)"
        # El borde es lo que define la tarjeta cuando su relleno es discreto
        # (justo el caso del tema oscuro): suelo propio, sin techo.
        assert m["borde tarjeta vs tarjeta"] >= 12.0, (
            f"{tema} · borde de la tarjeta = dL* {m['borde tarjeta vs tarjeta']:.1f} "
            f"(<12 = la tarjeta pierde su recorte)"
        )


def test_en_oscuro_la_tarjeta_se_queda_oscura():
    """[P1-MICRO-DARK-STAYS-DARK · 2026-08-12] La correccion del dueno a mi
    primera version: yo optimice la PARIDAD DE MAGNITUD (misma separacion que
    en claro) y en oscuro eso dio una tarjeta de L* 21,1 — mas clara que el
    punto mas claro del degradado que sustituia, y mas clara que los chips que
    tiene debajo. «Se ve muy claro y eso es lo que no queria, y mas que esta en
    modo oscuro».

    La regla que queda escrita no es un numero mio: en oscuro la tarjeta de
    atencion NO puede ser mas clara que la superficie hundida de los chips. Un
    tema oscuro que aclara sus tarjetas deja de ser oscuro."""
    m = _panel("oscuro")
    assert m["_L_att"] <= m["_L_chip"], (
        f"la tarjeta de atencion (L* {m['_L_att']:.1f}) quedo MAS CLARA que los "
        f"chips (L* {m['_L_chip']:.1f}): en oscuro la tarjeta se aclara y el panel "
        f"deja de leerse como tema oscuro."
    )


def test_la_tarjeta_no_vuelve_a_disolverse():
    """`.att` SÓLIDA: el gradiente moría en el color del panel y la mitad de
    abajo desaparecía. Y una sola declaración sirve a los dos temas (mezcla
    sobre `--mn-sunken`), así que tampoco puede volver un override por tema."""
    src = CSS_MOD.read_text(encoding="utf-8")
    i = src.index("\n.att {")
    bloque = src[i:src.index("}", i)]
    assert "linear-gradient" not in bloque, "volvió el gradiente que disolvía la tarjeta"
    assert "var(--mn-att-base)" in bloque, "la tarjeta debe mezclarse sobre la base que cada tema define"
    assert not re.search(r'html\[data-theme="dark"\]\)\s*\.att\s*\{[^}]*background', src), \
        "reapareció un override oscuro de .att: la regla base ya cubre ambos temas"


def test_las_superficies_del_panel_pasan_por_las_variables_locales():
    """Chips/límites/resumen consumen `--mn-sunken`/`--mn-line`. Si alguien
    vuelve a `--surface-sunken` crudo, el tema claro pierde la mitad de su
    separación sin que nadie lo note."""
    src = CSS_MOD.read_text(encoding="utf-8")
    for clase in (".stat {", ".q {", ".lim {"):
        i = src.index(clase)
        bloque = src[i:src.index("}", i)]
        assert "var(--mn-sunken)" in bloque, f"{clase} no usa --mn-sunken"
        assert "var(--mn-line)" in bloque, f"{clase} no usa --mn-line"
