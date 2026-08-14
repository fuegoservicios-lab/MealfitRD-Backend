"""[P1-LANDING-SW-DEFER · 2026-08-14] El precache del Service Worker no debe
competir con el primer paint del landing.

EL DEFECTO QUE CIERRA. `main.jsx` registraba el SW con `immediate: true`. En
workbox-window (`Workbox.ts`, guard `if (!immediate && document.readyState !==
'complete') await new Promise(res => addEventListener('load', res))`) ese flag
significa literalmente «no esperes a `window.load`»: el install arranca al
EVALUAR el chunk de entrada, que es exactamente cuando el navegador está
pidiendo el chunk del hero. Su propio JSDoc lo marca «(not recommended)».

Medido sobre `dist/custom-sw.js` el 2026-08-14: el manifest inyectado trae 96
entradas / 3.837 KiB, y tras el filtro por host de `custom-sw.js`
(P2-SW-PRECACHE-HOST-SPLIT) un visitante del APEX precacheaba 73 entradas =
2.557 KiB crudos ≈ 988 KiB por la red, en 73 fetches paralelos sobre la misma
conexión que el LCP. Con `immediate` en su default esos bytes salen de la
ventana del primer paint sin perder una sola garantía: los tres markers que
rodean el registro (P2-PWA-SKIPWAITING, P1-SW-AUTO-APPLY-SAFE,
P2-PWA-UPDATE-POLL) operan sobre callbacks POSTERIORES al registro.

POR QUÉ ADEMÁS SE TOCA EL PRECACHE. Dentro de esos 988 KiB viajaban iconos que
el navegador sólo pide cuando el usuario INSTALA el PWA (`apple-touch-icon*`,
los pide el sistema operativo, nunca se renderizan en la app) y un PNG cuyo
único consumidor era un componente sin importadores. Se excluyen del PRECACHE,
que no es lo mismo que borrarlos del árbol: `manifest.json` referencia
`apple-touch-icon.png` cuatro veces y BRAND-FAVICON-B los declara por escrito
fallback de root. Borrarlos rompería los iconos del PWA instalado.

LO QUE ESTE TEST NO AFIRMA. No mide bytes (eso depende del build) ni afirma que
el precache sea pequeño. Ancla las DOS decisiones que un refactor puede
deshacer sin darse cuenta: el momento del registro y qué familias de assets
quedan fuera del precache.

Tooltip-anchor: P1-LANDING-SW-DEFER
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND = _REPO_ROOT / "frontend"
_MAIN_JSX = _FRONTEND / "src" / "main.jsx"
_VITE_CONFIG = _FRONTEND / "vite.config.js"
_PUBLIC = _FRONTEND / "public"


def _read(path: Path) -> str:
    if not path.exists():
        pytest.fail(f"[P1-LANDING-SW-DEFER] No existe {path.relative_to(_REPO_ROOT)}")
    return path.read_text(encoding="utf-8")


def _register_sw_options(text: str) -> str:
    """Devuelve el cuerpo del objeto de opciones de `registerSW({...})`.

    Se corta en el primer `onNeedRefresh` / `onRegisteredSW` porque esos
    callbacks traen sus propias llaves y un balanceo ingenuo se pasaría de
    largo. Las opciones escalares (las que nos importan) van antes.
    """
    m = re.search(r"registerSW\(\s*\{", text)
    if not m:
        pytest.fail(
            "[P1-LANDING-SW-DEFER] No se encontró la llamada `registerSW({` en main.jsx. "
            "Si el registro del SW se movió de sitio, mueve también este guard."
        )
    tail = text[m.end():]
    stop = re.search(r"\bon[A-Z]\w*\s*\(", tail)
    return tail[: stop.start()] if stop else tail[:2000]


# ---------------------------------------------------------------------------
# 1. El registro del SW espera a `window.load`
# ---------------------------------------------------------------------------

def test_registro_del_sw_no_usa_immediate():
    """`immediate: true` mete ~988 KiB de precache dentro de la ventana del LCP."""
    opciones = _register_sw_options(_read(_MAIN_JSX))
    encontrado = re.search(r"\bimmediate\s*:\s*(\w+)", opciones)
    assert not (encontrado and encontrado.group(1) == "true"), (
        "[P1-LANDING-SW-DEFER] `registerSW` volvió a `immediate: true`.\n"
        "Con ese flag el install del Service Worker arranca al evaluar el chunk de "
        "entrada — es decir, compitiendo por el ancho de banda con el chunk del hero "
        "del landing. Medido: 73 entradas / ~988 KiB en el apex.\n"
        "El default (`false`) espera a `window.load` y NO pierde ninguna garantía: "
        "P2-PWA-SKIPWAITING, P1-SW-AUTO-APPLY-SAFE y P2-PWA-UPDATE-POLL operan sobre "
        "callbacks posteriores al registro."
    )


def test_el_motivo_del_defer_esta_escrito_junto_al_registro():
    """Sin la razón anotada, el siguiente que lea esto vuelve a poner `immediate`."""
    texto = _read(_MAIN_JSX)
    assert "P1-LANDING-SW-DEFER" in texto, (
        "[P1-LANDING-SW-DEFER] Falta el marker en main.jsx. La ausencia de "
        "`immediate: true` es invisible: sin un comentario que diga POR QUÉ el "
        "registro espera a `load`, el flag vuelve en el próximo toqueteo del SW."
    )


# ---------------------------------------------------------------------------
# 2. El precache no carga con iconos que sólo pide el sistema operativo
# ---------------------------------------------------------------------------

def _glob_ignores(text: str) -> list[str]:
    """Entradas de `globIgnores`, emparejando corchetes y saltando comentarios.

    [2026-08-14] Antes era `\\[(.*?)\\]` no-codicioso: se detenía en el PRIMER `]`, que
    hoy es el del marker `[P2-LANDING-PRERENDER-META · 2026-08-14]` dentro de un
    comentario de la propia lista. Resultado: la función devolvía `[]` y los cinco
    tests de iconos gritaban «no está en globIgnores» sobre cinco entradas que sí
    están. Es la misma trampa que ya mordió a este archivo: *la prosa que explica una
    estructura contiene los caracteres de esa estructura.*
    """
    i = text.find("globIgnores:")
    if i == -1:
        pytest.fail(
            "[P1-LANDING-SW-DEFER] No se encontró `globIgnores` en vite.config.js "
            "(lo introdujo P2-PWA-PRECACHE-TRIM). Si el recorte del precache se movió "
            "a otro mecanismo, actualiza este guard."
        )
    a = text.index("[", i)
    prof = 0
    fin = None
    for k in range(a, len(text)):
        if text[k] == "[":
            prof += 1
        elif text[k] == "]":
            prof -= 1
            if prof == 0:
                fin = k
                break
    assert fin is not None, "globIgnores sin cierre — ¿config a medio editar?"
    cuerpo = "\n".join(
        ln for ln in text[a:fin + 1].splitlines() if not ln.strip().startswith("//")
    )
    return re.findall(r"['\"]([^'\"]+)['\"]", cuerpo)


@pytest.mark.parametrize(
    "patron",
    [
        # Los que enlaza `manifest.json` (icono del PWA instalado + shortcuts).
        "apple-touch-icon.png",
        "apple-touch-icon-180.png",
        "apple-touch-icon-192.png",
        # Los que enlaza `index.html` (BRAND-FAVICON-B les dio nombre nuevo porque
        # iOS cachea el apple-touch-icon a nivel de SO e ignora el `?v=`).
        "apple-touch-icon-v2.png",
        "apple-touch-icon-180-v2.png",
    ],
)
def test_los_iconos_del_sistema_operativo_no_se_precachean(patron):
    """Los pide el SO al INSTALAR el PWA; nunca se renderizan dentro de la app."""
    ignorados = _glob_ignores(_read(_VITE_CONFIG))
    assert patron in ignorados, (
        f"[P1-LANDING-SW-DEFER] `{patron}` no está en `globIgnores`.\n"
        "Es un icono que sólo pide el sistema operativo cuando el usuario instala "
        "el PWA — un visitante anónimo del landing lo descarga en el precache sin "
        "que nada llegue a mostrarlo nunca.\n"
        "OJO: excluir del PRECACHE, NO borrar del árbol. `manifest.json` referencia "
        "`apple-touch-icon.png` 4 veces y BRAND-FAVICON-B los declara fallback de root."
    )


def test_globignores_no_lista_ficheros_inexistentes():
    """Una entrada que no puede casar nunca se lee como «esto se está excluyendo»."""
    huerfanas = []
    for patron in _glob_ignores(_read(_VITE_CONFIG)):
        if "*" in patron:
            continue  # los globs se resuelven contra dist/, no contra public/
        if not (_PUBLIC / patron).exists():
            huerfanas.append(patron)
    assert not huerfanas, (
        f"[P1-LANDING-SW-DEFER] `globIgnores` lista ficheros que no existen: {huerfanas}.\n"
        "Una exclusión que no puede casar con nada no es inofensiva: quien depure el "
        "precache la leerá como «ese asset ya está excluido» y buscará el peso en otra "
        "parte. Bórrala o corrige el nombre."
    )


# ---------------------------------------------------------------------------
# 3. Assets muertos fuera del árbol (y por tanto fuera del precache)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "relativo,motivo",
    [
        (
            "src/components/common/Logo.jsx",
            "cero importadores en todo `frontend/src` — Rollup lo tree-shakea del "
            "bundle, pero su `<img src=\"/mealfit-mark-dark.png\">` mantenía vivo un "
            "PNG de 47,7 KB que SÍ viajaba en el precache",
        ),
        (
            "public/mealfit-mark-dark.png",
            "47,7 KB precacheados cuyo único consumidor era el `Logo.jsx` huérfano",
        ),
        (
            "public/og-image-v3.jpg",
            "67,8 KB de la imagen social anterior al rebrand; cero referencias tras "
            "P2-WORDMARK-BIOBOROS (el HTML sirve `og-image-v4.jpg`)",
        ),
    ],
)
def test_assets_muertos_del_landing_borrados(relativo, motivo):
    assert not (_FRONTEND / relativo).exists(), (
        f"[P1-LANDING-SW-DEFER] `frontend/{relativo}` sigue en el árbol.\n"
        f"Motivo para borrarlo: {motivo}.\n"
        "Si lo has resucitado a propósito, borra su fila de este test explicando "
        "quién lo consume ahora."
    )
