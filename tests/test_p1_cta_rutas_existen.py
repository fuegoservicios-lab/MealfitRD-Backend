"""[P1-CTA-RUTAS-EXISTEN · 2026-08-21] Cinco CTA del Dashboard apuntaban a una ruta
que no existe.

QUÉ PASÓ. Los banners de plan pausado de `Dashboard.jsx` definen `url: '/inventory'` y
el consumidor hace `navigate(_copy.url)`. Volcando los `path=` de `App.jsx`: existen
`/pantry` y `/mi-nevera` (alias que redirigen) y `/dashboard/pantry` (la ruta real).
`/inventory` no está en ninguno, así que cae en el catch-all `path="*"` → NotFound.

NO ES UN FALLO DE i18n. Salió de la auditoría de idiomas de rebote y afecta a TODOS los
idiomas, `es-DO` incluido. Y toca justo los banners que piden una acción urgente: plan
pausado por nevera vacía, primera compra pendiente, inventario sin validar. El usuario
lee «Actualizar nevera», pulsa, y aterriza en una página de «no existe».

QUÉ ANCLA, y por qué así. No estas cinco cadenas: la CLASE. Se extraen todos los `path=`
declarados en `App.jsx` y todos los destinos literales que el código emite (`url:`,
`to=`, `navigate('…')`), y se exige que cada destino resuelva. Un guard sobre
`'/inventory'` concreto no habría visto este —nació con el banner— y no vería el
siguiente.

La resolución respeta los prefijos dinámicos (`/novedades/:slug`) y los parámetros de
ruta, porque una ruta paramétrica es una ruta que existe.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONT_SRC = _ROOT / "frontend" / "src"
_APP = _FRONT_SRC / "App.jsx"

_MARKER = "P1-CTA-RUTAS-EXISTEN"

# Destinos que NO son rutas de la SPA: los resuelve el navegador, no el router.
_NO_ES_RUTA = re.compile(r"^(?:https?:|mailto:|tel:|#|\.|/api/|/assets/|/[\w.-]+\.\w{2,5}$)")


def _leer(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"{p} no existe en este checkout (repos hermanos)")
    return p.read_text(encoding="utf-8")


def _rutas_declaradas() -> set[str]:
    app = _leer(_APP)
    return {m.group(1) for m in re.finditer(r'<Route\s+path=\{?["\']([^"\']+)["\']', app)}


def _resuelve(destino: str, rutas: set[str]) -> bool:
    if destino in rutas:
        return True
    # Una ruta paramétrica (`/novedades/:slug`) cubre a sus hijos concretos.
    for r in rutas:
        if ":" not in r:
            continue
        patron = "^" + re.sub(r":[^/]+", r"[^/]+", re.escape(r).replace(r"\:", ":")) + "$"
        if re.match(patron, destino):
            return True
    # Un `path="*"` existe, pero aterrizar ahí es precisamente el defecto: no cuenta.
    return False


def _destinos_literales() -> list[tuple[str, int, str]]:
    """(fichero, línea, destino) de cada navegación con destino literal."""
    patrones = (
        re.compile(r"""\burl:\s*['"](/[^'"]*)['"]"""),
        re.compile(r"""\bto=\{?['"](/[^'"]*)['"]"""),
        re.compile(r"""\bnavigate\(\s*['"](/[^'"]*)['"]"""),
    )
    fuera = []
    for p in sorted(_FRONT_SRC.rglob("*.jsx")) + sorted(_FRONT_SRC.rglob("*.js")):
        rel = str(p.relative_to(_FRONT_SRC)).replace("\\", "/")
        if "__tests__" in rel or ".test." in rel:
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        for pat in patrones:
            for m in pat.finditer(txt):
                destino = m.group(1)
                if _NO_ES_RUTA.match(destino):
                    continue
                linea = txt[: m.start()].count("\n") + 1
                fuera.append((rel, linea, destino))
    return fuera


def test_toda_navegacion_literal_apunta_a_una_ruta_que_existe() -> None:
    rutas = _rutas_declaradas()
    assert rutas, "no pude extraer ningún `path=` de App.jsx — ¿cambió el estilo?"

    rotos = [(f, ln, d) for f, ln, d in _destinos_literales() if not _resuelve(d, rutas)]
    assert not rotos, (
        "Estas navegaciones apuntan a rutas que NO existen en App.jsx y caen en el "
        "catch-all `path=\"*\"` (NotFound):\n"
        + "\n".join(f"  {f}:{ln} → {d!r}" for f, ln, d in rotos)
        + f"\n[{_MARKER}]"
    )


def test_el_guard_detecta_una_ruta_inventada() -> None:
    """MUTACIÓN DE CONTROL. Sin esto, un extractor que no encontrara NADA pasaría el
    test de arriba y el fichero entero sería decorativo."""
    rutas = _rutas_declaradas()
    assert not _resuelve("/no-existe-esta-ruta", rutas)
    # Y una que sí existe tiene que resolver, o el guard estaría roto al revés.
    assert _resuelve("/dashboard/pantry", rutas), (
        "`/dashboard/pantry` no resuelve: el extractor de rutas de App.jsx está roto y "
        "el test de arriba estaría dando falsos positivos."
    )


def test_los_banners_de_pausa_llevan_a_la_nevera_real() -> None:
    """El caso concreto, nombrado, porque es el que motivó el guard: son los CTA de
    los banners que piden una acción urgente."""
    dashboard = _leer(_FRONT_SRC / "pages" / "Dashboard.jsx")
    assert "url: '/inventory'" not in dashboard, (
        "Vuelve a haber CTA apuntando a `/inventory`, que no existe. La ruta real es "
        f"`/dashboard/pantry`. [{_MARKER}]"
    )
