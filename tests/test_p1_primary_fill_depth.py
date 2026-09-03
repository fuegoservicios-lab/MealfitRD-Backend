"""[P1-PRIMARY-FILL-DEPTH · 2026-08-14] Un botón relleno de `--primary` deslumbra
en modo oscuro, porque en oscuro `--primary` es TINTA, no relleno.

EL SÍNTOMA. El dueño señaló «Reanudar el plan»: «este botón está muy brilloso
para el modo oscuro». Medido, no era una impresión — era una INVERSIÓN:

    tema claro   relleno #4F46E5  L* 40,7   blanco encima 6,29:1
    tema oscuro  relleno #818CF8  L* 61,8   blanco encima 2,98:1  ← AA falla

El botón era MÁS luminoso en modo oscuro que en modo claro, sobre un panel de
L* 8,3 (+53,5 de salto), y el texto blanco encima no llegaba al 4,5:1 de AA.

LA CAUSA. `html[data-theme="dark"]` aclara los índigos a propósito («indigo/
emerald aclarados para contraste sobre fondos oscuros»): son TINTA sobre fondo
oscuro. Usados como RELLENO invierten su intención. Usar un token no basta —
hay que usar el token PARA LO QUE FUE DISEÑADO.

LO QUE YA EXISTÍA. `P1-CTA-FILL-DEPTH` (mismo día) resolvió exactamente esto
para el CTA con `--cta-fill`, rebajando al 75% contra el panel. Su receta era
correcta y su alcance, uno. Este P-fix la convierte en token plano reutilizable
(`--primary-fill`) y lo aplica a las 14 superficies que quedaron fuera.

POR QUÉ UN TOKEN NUEVO Y NO `--cta-fill`: `--cta-fill` es un DEGRADADO. Estos 14
call sites son rellenos PLANOS; meterles un degradado cambiaría el modo claro,
que hoy está bien. `--primary-fill` resuelve a `var(--primary)` en claro y en
papel (cero cambio visual ahí) y solo corrige el oscuro.

POR QUÉ SIN EXCEPCIONES: `.ctaPrimary` de `/como-funciona` vive en el tema papel,
donde `--primary` es #0B0B0B y el blanco encima está perfecto. Podría excluirse,
pero como el token es no-op en papel, migrarlo también deja UNA regla sin
excepciones que recordar. Una excepción innecesaria es una invitación a copiarla.

Tooltip-anchor: P1-PRIMARY-FILL-DEPTH
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "frontend" / "src"
_INDEX = _SRC / "index.css"

# Superficie de papel: ahí `--primary` es casi negro y el blanco encima está bien.
# Se migra igual (el token es no-op en papel), pero no se MIDE contra AA-oscuro.
_AA_TEXTO_NORMAL = 4.5


# --------------------------------------------------------------------------
# utilidades de color (sRGB / WCAG) — el guard mide el RESULTADO, no el cómo
# --------------------------------------------------------------------------

def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _lin(c: float) -> float:
    c /= 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _luminancia(h: str) -> float:
    r, g, b = _rgb(h)
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _contraste(a: str, b: str) -> float:
    ya, yb = _luminancia(a), _luminancia(b)
    hi, lo = max(ya, yb), min(ya, yb)
    return (hi + 0.05) / (lo + 0.05)


def _mezcla_srgb(a: str, b: str, pct_a: float) -> str:
    """Equivalente a `color-mix(in srgb, a pct%, b)` — mezcla sRGB directa."""
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    f = lambda x, y: round(x * pct_a + y * (1 - pct_a))
    return "#%02X%02X%02X" % (f(ra, rb), f(ga, gb), f(ba, bb))


def _css() -> str:
    return _INDEX.read_text(encoding="utf-8")


def _bloque(selector: str) -> str:
    """El cuerpo del bloque de tema (`:root`, `html[data-theme="dark"]`, …)."""
    css = _css()
    i = css.find(selector + " {")
    assert i >= 0, f"[P1-PRIMARY-FILL-DEPTH] No existe el bloque `{selector}` en index.css"
    j = css.find("\n}", i)
    return css[i:j]


def _valor(bloque: str, prop: str) -> str | None:
    m = re.search(rf"^\s*{re.escape(prop)}:\s*([^;]+);", bloque, re.MULTILINE)
    return m.group(1).strip() if m else None


# --------------------------------------------------------------------------
# 1. El token existe en los tres temas
# --------------------------------------------------------------------------

@pytest.mark.parametrize("selector", [':root', 'html[data-theme="dark"]', 'html[data-theme="paper"]'])
def test_el_token_de_relleno_existe_en_cada_tema(selector):
    valor = _valor(_bloque(selector), "--primary-fill")
    assert valor, (
        f"[P1-PRIMARY-FILL-DEPTH] `{selector}` no define `--primary-fill`.\n"
        "El token se declara EXPLÍCITO en cada bloque de tema (igual que "
        "`--cta-fill`). Declararlo solo en `:root` como `var(--primary)` "
        "funcionaría hoy por accidente —los tres selectores son el mismo "
        "elemento <html>— pero es justo la forma en que `--cta-tint` nació "
        "INERTE: una custom property se resuelve donde se DECLARA."
    )


# --------------------------------------------------------------------------
# 2. En oscuro el relleno está REBAJADO y el blanco encima pasa AA
# --------------------------------------------------------------------------

def test_en_oscuro_el_relleno_no_es_la_tinta_cruda():
    valor = _valor(_bloque('html[data-theme="dark"]'), "--primary-fill")
    assert valor != "var(--primary)", (
        "[P1-PRIMARY-FILL-DEPTH] En oscuro `--primary-fill` es `var(--primary)` a "
        "secas — que es el bug. En oscuro `--primary` (#818CF8) es TINTA sobre "
        "fondo oscuro; como RELLENO da L* 61,8 sobre un panel de L* 8,3 y deja el "
        "texto blanco en 2,98:1. El botón acababa MÁS brillante en modo oscuro "
        "que en modo claro."
    )


def _relleno_oscuro_resuelto(bloque: str) -> str:
    """`--primary-fill` en oscuro como hex: acepta la receta `color-mix(in srgb,
    var(--primary) N%, var(--bg-card))` de P1-CTA-FILL-DEPTH o un literal.
    [P2-PRIMARY-FILL-INK · 2026-09-03] La mezcla con el panel desaturaba (índigo
    grisáceo, «muy claro» para el dueño); ahora es índigo 700 literal. Lo que
    este guard protege es el RESULTADO (contraste y salto), no la receta."""
    valor = _valor(bloque, "--primary-fill") or ""
    m = re.search(
        r"color-mix\(in srgb,\s*var\(--primary\)\s*(\d+)%,\s*var\(--bg-card\)\s*\)", valor
    )
    if m:
        primary = _valor(bloque, "--primary")
        bg_card = _valor(bloque, "--bg-card")
        assert primary and bg_card, "Faltan --primary/--bg-card en el bloque oscuro"
        return _mezcla_srgb(primary, bg_card, int(m.group(1)) / 100)
    m = re.fullmatch(r"#[0-9A-Fa-f]{6}", valor)
    assert m, (
        "[P1-PRIMARY-FILL-DEPTH] No se pudo leer `--primary-fill` en oscuro "
        f"(valor: {valor!r}). Se espera un hex literal o "
        "`color-mix(in srgb, var(--primary) N%, var(--bg-card))`. Si cambias de "
        "mecanismo, actualiza este guard para que siga midiendo el CONTRASTE."
    )
    return valor.upper()


def test_el_blanco_sobre_el_relleno_oscuro_pasa_AA():
    """La aserción que importa: el RESULTADO medido, no la receta."""
    bloque = _bloque('html[data-theme="dark"]')
    relleno = _relleno_oscuro_resuelto(bloque)
    contraste = _contraste("#FFFFFF", relleno)
    assert contraste >= _AA_TEXTO_NORMAL, (
        f"[P1-PRIMARY-FILL-DEPTH] El relleno oscuro resultante ({relleno}) deja el "
        f"texto blanco en {contraste:.2f}:1, por debajo del {_AA_TEXTO_NORMAL}:1 de "
        "AA para texto normal. Estos botones llevan texto de ~0,85rem: no califican "
        "como 'texto grande'."
    )


def test_el_relleno_oscuro_sigue_destacando_sobre_el_panel():
    """Rebajar no es apagar: un CTA tiene que seguir leyéndose como el principal."""
    bloque = _bloque('html[data-theme="dark"]')
    relleno = _relleno_oscuro_resuelto(bloque)

    def lstar(h):
        y = _luminancia(h)
        return 116 * (y ** (1 / 3)) - 16 if y > 0.008856 else 903.3 * y

    salto = lstar(relleno) - lstar(_valor(bloque, "--bg-card"))
    assert salto >= 25, (
        f"[P1-PRIMARY-FILL-DEPTH] El relleno quedó a solo {salto:.1f} puntos de L* "
        "sobre el panel. Se pasó de frenada: el arreglo es que deje de DESLUMBRAR, "
        "no que deje de verse. Es el botón principal de su tarjeta."
    )


# --------------------------------------------------------------------------
# 3. Ninguna superficie rellena vuelve al `--primary` crudo con texto blanco
# --------------------------------------------------------------------------

def test_ninguna_superficie_rellena_usa_primary_crudo_con_texto_blanco():
    ofensores = []
    for ruta in list(_SRC.rglob("*.module.css")) + [_INDEX]:
        texto = ruta.read_text(encoding="utf-8")
        # Solo bloques de regla; se ignoran comentarios para no cazar prosa.
        limpio = re.sub(r"/\*.*?\*/", "", texto, flags=re.DOTALL)
        for m in re.finditer(r"([.#][\w-]+)\s*\{([^}]*)\}", limpio):
            sel, cuerpo = m.group(1), m.group(2)
            relleno_crudo = re.search(r"background:\s*var\(--primary[,)]", cuerpo)
            if not relleno_crudo or re.search(r"--primary-(light|dark|fill)", cuerpo):
                continue
            if re.search(r"color:\s*(#fff\b|#ffffff\b|white\b)", cuerpo, re.I):
                ofensores.append(f"{ruta.relative_to(_SRC)} → {sel}")

    assert not ofensores, (
        "[P1-PRIMARY-FILL-DEPTH] Superficies rellenas con `var(--primary)` crudo y "
        "texto blanco encima (deslumbran en modo oscuro y fallan AA):\n  "
        + "\n  ".join(ofensores)
        + "\n\nUsa `var(--primary-fill)`. Es no-op en claro y en papel; solo corrige "
        "el oscuro, donde `--primary` es tinta y no relleno."
    )
