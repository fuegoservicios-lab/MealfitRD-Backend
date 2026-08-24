"""[P2-BETA-MATCH-SIN-COUNTRY · 2026-08-23] `P1-BETA-PRICE-LEAKS` nombró tres PANTALLAS y
parcheó tres CALL SITES. Hay CUATRO.

El cuarto es `SupermarketBrands.jsx` —el panel «Marcas del súper» del Dashboard—, que mandaba
`body: JSON.stringify({ names })` a secas. El helper del servidor decide con
`pricing_mode_for_country(canonicalize_country(country))`, y su fail-safe ante país AUSENTE es
DEVOLVER LOS PRECIOS: omitir el campo no degrada, filtra pesos dominicanos a un usuario de
España.

Lo tapaba que el panel se oculte cuando `planData._pricing_mode === 'beta_no_prices'`. Eso
convierte una defensa en profundidad en UNA sola puerta, y esa puerta cuelga de una clave del
plan que ya se demostró NO durable (un plan beta vivo perdió su `_pricing_mode` en un
recálculo, con 49 precios en RD$ recuperados y sin una alerta).

POR QUÉ ESTE GUARD ENUMERA EN VEZ DE MIRAR UN FICHERO. El defecto original no fue escribir mal
un call site: fue **contar pantallas y creer que eran call sites**. Un guard que compruebe
«SupermarketBrands manda country» repetiría exactamente ese error el día que aparezca el
quinto. Así que la propiedad es de conjunto: *todo* POST a `/api/supermarket/match` en
`frontend/src` manda el país, y el conteo vivo se declara para que añadir uno sin país sea
imposible de pasar por alto.

Los comentarios se ELIMINAN antes de medir: tres de los cuatro call sites llevan encima una
nota que cita `country: formData?.country`, y un guard al que le vale un comentario es un
guard que aprueba el defecto que persigue.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO / "frontend" / "src"
_ENDPOINT = "/api/supermarket/match"


def _sin_comentarios(src: str) -> str:
    """Quita comentarios `//` y `/* */` de JS/JSX respetando cadenas y plantillas.

    No intenta entender expresiones regulares literales: dentro del alcance medido (los
    ficheros que hablan con este endpoint) no hay ninguna, y el propio test lo comprueba
    verificando que sigue viendo los mismos call sites después del barrido.
    """
    out = []
    i, n = 0, len(src)
    comilla = None
    while i < n:
        c = src[i]
        if comilla:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(src[i + 1])
                i += 2
                continue
            if c == comilla:
                comilla = None
            i += 1
            continue
        if c in "\"'`":
            comilla = c
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            i = n if j == -1 else j + 2
            # Preserva los saltos de línea para no mover los números de línea.
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _argumento_de(src: str, desde: int, funcion: str) -> str:
    """Texto del argumento de la primera llamada a `funcion` a partir de `desde`."""
    k = src.index(funcion + "(", desde)
    k += len(funcion) + 1
    profundidad = 1
    j = k
    while j < len(src) and profundidad:
        if src[j] == "(":
            profundidad += 1
        elif src[j] == ")":
            profundidad -= 1
        j += 1
    return src[k:j - 1]


def _call_sites():
    """[(ruta_relativa, texto_del_body)] de cada POST a /api/supermarket/match en src/."""
    encontrados = []
    for ruta in sorted(_SRC.rglob("*.js")) + sorted(_SRC.rglob("*.jsx")):
        if "__tests__" in ruta.parts:
            continue
        limpio = _sin_comentarios(ruta.read_text(encoding="utf-8", errors="replace"))
        for m in re.finditer(re.escape(_ENDPOINT), limpio):
            encontrados.append((
                ruta.relative_to(_REPO).as_posix(),
                _argumento_de(limpio, m.end(), "JSON.stringify"),
            ))
    return encontrados


def test_los_call_sites_del_match_siguen_siendo_los_medidos():
    """Ancla el CONTEO. Si mañana hay cinco, este test lo dice antes de que el quinto llegue
    a producción sin país — que es exactamente cómo se coló el cuarto."""
    sitios = _call_sites()
    ficheros = sorted({f for f, _ in sitios})
    assert len(sitios) == 4, (
        f"cambió el número de POST a {_ENDPOINT}: {sitios}. Si es uno nuevo, tiene que "
        f"mandar `country` y sumarse a este conteo EN EL MISMO cambio."
    )
    assert ficheros == [
        "frontend/src/components/assessment/questions/QPantryBuilder.jsx",
        "frontend/src/components/dashboard/SupermarketBrands.jsx",
        "frontend/src/pages/Pantry.jsx",
    ], ficheros


@pytest.mark.parametrize("idx", range(4))
def test_todo_call_site_del_match_manda_el_pais(idx):
    """La propiedad de conjunto: NINGUNO puede preguntar sin país, porque el fail-safe del
    servidor ante país ausente es devolver los precios en RD$."""
    fichero, body = _call_sites()[idx]
    assert "country" in body, (
        f"{fichero} pregunta a {_ENDPOINT} sin `country`: el servidor devolvería precios "
        f"en pesos dominicanos a un usuario de país beta. Body medido: {body!r}"
    )


def test_el_cuarto_call_site_saca_el_pais_del_contexto_y_no_de_un_prop_inexistente():
    """`SupermarketBrands` no recibe `formData` por props. El fix de P1-BETA-PRICE-LEAKS en
    `Pantry.jsx` falló la primera vez justo por esto: `formData?.country` con `formData` sin
    declarar es un ReferenceError —el `?.` protege contra `undefined`, no contra un símbolo
    que no existe—, así que la ruta de marcas REVENTABA en vez de degradar."""
    ruta = _SRC / "components" / "dashboard" / "SupermarketBrands.jsx"
    limpio = _sin_comentarios(ruta.read_text(encoding="utf-8", errors="replace"))
    assert re.search(r"import\s*\{[^}]*\buseAssessment\b[^}]*\}\s*from", limpio), (
        "SupermarketBrands usa `formData` sin importar `useAssessment`"
    )
    assert re.search(r"\bformData\b[^\n]*=\s*useAssessment\(\)", limpio), (
        "`formData` no se liga desde el contexto: sería un ReferenceError en tiempo de render"
    )


def test_el_pais_es_dependencia_del_efecto_que_dispara_la_peticion():
    """Cambiar de país en Configuración y volver al Dashboard tiene que repreguntar. Sin la
    dependencia, el panel se quedaría con la respuesta del país anterior — precios incluidos."""
    ruta = _SRC / "components" / "dashboard" / "SupermarketBrands.jsx"
    limpio = _sin_comentarios(ruta.read_text(encoding="utf-8", errors="replace"))
    deps = re.search(r"\}\s*,\s*\[([^\]]*)\]\s*\)\s*;\s*\n\s*const persistPref", limpio)
    assert deps, "no se localizó el array de dependencias del `load`"
    assert "formData?.country" in deps.group(1).replace(" ", ""), deps.group(1)
