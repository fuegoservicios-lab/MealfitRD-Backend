# -*- coding: utf-8 -*-
"""[P1-MICRO-CRISP-SURFACES · 2026-08-12] Las superficies del panel de Micronutrientes.

HISTORIA DEL CONTRATO, porque es la lección: nació afirmando PARIDAD DE
MAGNITUD (que cada relación midiese lo mismo en los dos temas) y hubo que
parchearlo dos veces seguidas, siempre en la misma dirección — el dueño
rechazando superficies que en oscuro salían claras:

  1. la tarjeta de atención en L* 21,1 («se ve muy claro y eso es lo que no
     quería, y más que está en modo oscuro»),
  2. los tres chips de resumen en L* 16,4 («el fondo gris hace que se vea muy
     claro en el modo oscuro y el contexto de adentro lo opaca»).

Dos parches en la misma dirección no son dos incidentes: son un contrato
equivocado. La simetría que perseguía obligaba al tema oscuro a ACLARAR sus
superficies para igualar la separación del claro, que es justo lo que un tema
oscuro no debe hacer. El contrato de hoy dice lo que el dueño dijo:

  · En CLARO las superficies se HUNDEN bajo el panel blanco (suelo de dL*).
  · En OSCURO no se aclaran: los chips quedan por debajo del panel, y la única
    que se levanta es la tarjeta de atención —porque lleva estado— con un techo
    ANCLADO A SU GEMELA: no puede levantarse en oscuro más de lo que se hunde
    en claro.
  · Y NO se mide paridad de nada: cada relación tiene su SUELO y cada tema
    llega a él por su camino (ver el bloque «Sobre la paridad» más abajo).

Recalcula todo desde los tokens del DS, así que salta también si alguien cambia
`--bg-page` o `--border` en index.css y rompe el panel de rebote.
"""
import re
from pathlib import Path

import pytest

FRONT = Path(__file__).resolve().parents[2] / "frontend" / "src"
CSS_MOD = FRONT / "components" / "dashboard" / "MicronutrientMeter.module.css"
CSS_DS = FRONT / "index.css"

TONO = {"claro": "#EA580C", "oscuro": "#FB923C"}   # el estado «far», el de las tarjetas
TONO_PCT_ATT = 9        # .att: color-mix(--tone 9%, --mn-att-base)
TONO_PCT_BORDE = 52     # borde de .att


def _px(c):
    h = c.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _hex(rgb):
    return "#%02X%02X%02X" % tuple(rgb)


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
    for k in ("--bg-page", "--bg-card", "--surface-sunken", "--border",
              "--text-main", "--text-muted", "--text-light"):
        m = re.search(re.escape(k) + r"\s*:\s*(#[0-9A-Fa-f]{6})\s*;", blk)
        if not m:
            pytest.fail(f"token {k} no resoluble como hex en el bloque {tema} de index.css")
        out[k] = m.group(1)
    return out


def _recipe(nombre, tema):
    src = CSS_MOD.read_text(encoding="utf-8")
    ancla = 'html[data-theme="dark"]) .panel' if tema == "oscuro" else ".panel {"
    i = src.index(ancla)
    scope = src[i:src.index("}", i)]
    m = re.search(re.escape(nombre) + r":\s*([^;]+);", scope)
    if not m:
        pytest.fail(f"{nombre} no declarado para el tema {tema} — se declara en AMBOS")
    return m.group(1).strip()


def _resolve(expr, tk):
    expr = expr.strip()
    m = re.fullmatch(r"var\((--[\w-]+)\)", expr)
    if m:
        return _px(tk[m.group(1)])
    m = re.fullmatch(r"color-mix\(in srgb,\s*var\((--[\w-]+)\)\s*(\d+)%,\s*var\((--[\w-]+)\)\s*\)", expr)
    if m:
        return _mix(tk[m.group(1)], int(m.group(2)), tk[m.group(3)])
    pytest.fail(f"receta no reconocida: {expr!r}")


def _panel(tema):
    tk = _tokens(tema)
    chip = _resolve(_recipe("--mn-sunken", tema), tk)
    tk2 = {**tk, "--mn-sunken": _hex(chip)}
    line = _resolve(_recipe("--mn-line", tema), tk)
    ink = _resolve(_recipe("--mn-ink-soft", tema), tk)
    att = _mix(TONO[tema], TONO_PCT_ATT, _resolve(_recipe("--mn-att-base", tema), tk2))
    borde_att = _mix(TONO[tema], TONO_PCT_BORDE, tk["--border"])
    panel = _px(tk["--bg-card"])
    return {
        "L_panel": _L(panel), "L_chip": _L(chip), "L_att": _L(att),
        "linea vs chip": abs(_L(line) - _L(chip)),
        "borde de la tarjeta": abs(_L(borde_att) - _L(att)),
        "nombre sobre tarjeta": _ratio(_px(tk["--text-main"]), att),
        "apagado sobre tarjeta": _ratio(ink, att),
        "apagado sobre chip": _ratio(_px(tk["--text-muted"]), chip),
    }


# ── El contrato por tema ────────────────────────────────────────────────────

def test_en_claro_las_superficies_se_hunden_bajo_el_panel():
    m = _panel("claro")
    for nombre, valor in (("chips", m["L_chip"]), ("tarjeta", m["L_att"])):
        hundido = m["L_panel"] - valor
        assert hundido >= 5.5, (
            f"en claro los {nombre} deben hundirse bajo el panel blanco: "
            f"dL* {hundido:.1f} (<5,5 = se funden con el panel)"
        )


def test_en_oscuro_las_superficies_no_se_aclaran():
    """La corrección del dueño, dos veces seguidas y en la misma dirección.
    Los chips se hunden o se quedan; jamás flotan por encima del panel."""
    m = _panel("oscuro")
    assert m["L_chip"] <= m["L_panel"] + 0.5, (
        f"los chips de resumen quedaron MÁS CLAROS que el panel (L* {m['L_chip']:.1f} "
        f"vs {m['L_panel']:.1f}): en oscuro eso son cajas grises flotando, y el "
        f"contenido de dentro se apaga en vez de resaltar."
    )


def test_en_oscuro_la_tarjeta_se_levanta_menos_de_lo_que_se_hunde_en_claro():
    """La tarjeta de atención SÍ puede levantarse —lleva estado— pero su techo
    no es un número mío: es lo que su gemela del tema claro se hunde. Con L*
    21,1 (la versión que el dueño rechazó) se levantaba 12,8 contra los 11,3
    que baja en claro; hoy se levanta 6,0."""
    c, o = _panel("claro"), _panel("oscuro")
    levanta = o["L_att"] - o["L_panel"]
    hunde_gemela = c["L_panel"] - c["L_att"]
    assert levanta <= hunde_gemela, (
        f"la tarjeta se levanta {levanta:.1f} en oscuro pero su gemela solo se hunde "
        f"{hunde_gemela:.1f} en claro: en oscuro se está aclarando de más."
    )


def test_la_tarjeta_oscura_no_se_apoya_en_el_fondo_de_pagina():
    """El suelo del otro lado (P1-MICRO-DARK-SURFACES, agosto): tenir el naranja
    sobre `--bg-page` da rgb(37,31,35), un marron apagado que en una paleta de
    slates frios se ve SUCIO, no alarmante. Se comprueba por RESOLUCION y no por
    el nombre de la variable: al hundir los chips, `--mn-sunken` paso a valer
    `--bg-page` en oscuro, asi que apoyar la tarjeta en el (por indireccion)
    vuelve a caer en la trampa sin escribir su nombre en ningun sitio."""
    tk = _tokens("oscuro")
    base = _resolve(_recipe("--mn-att-base", "oscuro"),
                    {**tk, "--mn-sunken": _hex(_resolve(_recipe("--mn-sunken", "oscuro"), tk))})
    assert _hex(base).upper() != tk["--bg-page"].upper(), (
        f"la base de la tarjeta oscura resuelve al fondo de pagina ({tk['--bg-page']}): "
        f"el naranja sobre ese casi-negro da el marron sucio que P1-MICRO-DARK-SURFACES "
        f"cerro en agosto. Da igual que se llegue por indireccion."
    )


# ── Sobre la paridad, que ya no se mide ─────────────────────────────────────
#
# Aqui vivia `test_paridad_de_lo_simetrico`, ultimo resto del contrato original.
# Se retira, y el motivo es la conclusion de todo este P-fix: cada vez que el
# tema oscuro mejoro DE VERDAD, la paridad lo marco como defecto. Con los chips
# hundidos, su linea los recorta a dL* 21,2 contra los 9,2 del claro, y su texto
# llega a 7,34:1 contra 4,56 — el oscuro DUPLICA al claro en las dos, y las dos
# son mejores. Un guard que hay que ir excluyendo fila a fila hasta quedarse sin
# ninguna no esta midiendo un contrato: esta midiendo mi idea equivocada.
#
# Lo que de verdad protegia —que ningun tema viva a la mitad del otro— lo cubren
# los SUELOS de arriba, y con el ejemplo real: los chips claros a dL* 4,7 caen
# por el suelo de 5,5, y el chevron a 3,07:1 cae por AA. Un suelo por relacion
# es mas honesto que un espejo entre temas, porque cada tema llega a el por su
# camino.


def test_el_chevron_no_vuelve_a_la_tinta_debil():
    """Concreto, porque fue uno de los hallazgos: el chevron de los chips usaba
    `--text-light` y en claro se quedaba en 3,07:1 (5,06 en oscuro) — la mitad,
    otra vez. Es un control, no una decoracion."""
    src = CSS_MOD.read_text(encoding="utf-8")
    i = src.index(".qChev {")
    bloque = src[i:src.index("}", i)]
    assert "var(--text-muted)" in bloque, "el chevron volvio a una tinta mas debil que --text-muted"


# ── Anclas estructurales ────────────────────────────────────────────────────

def test_la_tarjeta_no_vuelve_a_disolverse():
    """`.att` SÓLIDA y con UNA declaración: el gradiente moría en el color del
    panel (dL* 1,8 en claro, 0,0 en oscuro) y la mitad de abajo desaparecía."""
    src = CSS_MOD.read_text(encoding="utf-8")
    i = src.index("\n.att {")
    bloque = src[i:src.index("}", i)]
    assert "linear-gradient" not in bloque, "volvió el gradiente que disolvía la tarjeta"
    assert "var(--mn-att-base)" in bloque, "la tarjeta debe mezclarse sobre la base que cada tema define"
    assert not re.search(r'html\[data-theme="dark"\]\)\s*\.att\s*\{[^}]*background', src), \
        "reapareció un override oscuro de .att: la regla base ya cubre ambos temas"


def test_las_superficies_pasan_por_las_variables_locales():
    src = CSS_MOD.read_text(encoding="utf-8")
    for clase in (".stat {", ".q {", ".lim {"):
        i = src.index(clase)
        bloque = src[i:src.index("}", i)]
        assert "var(--mn-sunken)" in bloque, f"{clase} no usa --mn-sunken"
        assert "var(--mn-line)" in bloque, f"{clase} no usa --mn-line"
