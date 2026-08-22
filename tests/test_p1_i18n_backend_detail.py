"""[P1-I18N-BACKEND-DETAIL · 2026-08-21] El `detail` español del servidor ganaba sobre el
fallback traducido.

El patrón era `toast.error(data?.detail || t('…'))`. El `||` hace que el texto **español
del servidor** se pinte siempre que exista, y el fallback traducido sólo se vea cuando el
backend **no** explica qué pasó. O sea: la traducción estaba escrita, revisada y presente
en los cuatro catálogos — y era exactamente la rama que no llegaba nunca.

Y el gate lo daba verde, porque la clave existe y está traducida. **Nadie medía cuál de
las dos ramas del `||` gana.** Ese es el hueco que este fichero cierra: un test de
cobertura no puede verlo, porque el defecto no está en el catálogo sino en el operador.

MEDIDO: 22 usos de `?.detail ||` en `frontend/src`, de los cuales **5** en posición
inequívoca de copy (`toast.error(...)`, `description:`) y 10 en `throw new Error(...)`.

POR QUÉ NO BASTA CON INVERTIR EL `||`: `t('…') || data.detail` pintaría siempre el
fallback y tiraría lo que el servidor sí sabe —«te faltan 3 ingredientes» degradado a
«inténtalo de nuevo»—. Lo correcto es traducir lo que el servidor sabe, y para eso el
canal es el CÓDIGO, no la prosa. El backend ya emite ocho `error_code` canónicos.

LO QUE NO ENTRA, y por qué no es un olvido: los 10 `throw new Error(detail || …)`. Ahí el
string viaja a un `catch` cuyo destino varía —unos lo pintan, otros lo registran, otros
sólo miran `err.code`—, así que migrarlos a ciegas cambiaría comportamiento que nadie ha
medido. Se documentan aquí como deuda con nombre en vez de dejarlos parecer cubiertos.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_MARKER = "P1-I18N-BACKEND-DETAIL"

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_SRC = _ROOT / "frontend" / "src"

# `toast.error(…)` y `description:` son posición de COPY sin ambigüedad: lo que va ahí lo
# lee el usuario. `throw new Error(…)` no lo es — ver el docstring.
_POSICION_DE_COPY = re.compile(
    # `(?:\?\.|\.)` y no `\?\.?`: lo segundo exige el signo de interrogacion, asi que
    # `err.detail ||` —que existe en el arbol— era invisible. Lo cazo la mutacion de
    # control; el test principal estaba pasando sin ver una de las dos formas.
    r"(?:toast\.(?:error|warning|success)\s*\(|description:\s*)[^;\n]*?(?:\?\.|\.)detail\s*\|\|"
)


def _ficheros():
    if not (_ROOT / "backend").is_dir() or not _SRC.is_dir():
        pytest.skip(f"{_ROOT} no es la raíz del repo (¿worktree?)")
    return [p for p in list(_SRC.rglob("*.jsx")) + list(_SRC.rglob("*.js"))
            if "__tests__" not in p.parts]


def _sin_comentarios(js: str) -> str:
    """Un `detail ||` citado en un comentario no es código.

    Este repo lleva siete guards derrotados por prosa en dos días, varios con el
    comentario escrito por quien escribía el guard.
    """
    js = re.sub(r"/\*.*?\*/", " ", js, flags=re.S)
    return re.sub(r"//[^\n]*", " ", js)


# ============================================================
# 1 · Ninguna posición de copy pinta el `detail` crudo
# ============================================================

def test_ninguna_posicion_de_copy_pinta_el_detail_del_servidor() -> None:
    culpables = []
    for p in _ficheros():
        s = _sin_comentarios(p.read_text(encoding="utf-8"))
        for m in _POSICION_DE_COPY.finditer(s):
            linea = s[:m.start()].count("\n") + 1
            culpables.append(f"{p.relative_to(_SRC).as_posix()}:{linea}")

    assert not culpables, (
        "Estos sitios pintan el `detail` del servidor —que viene en ESPAÑOL siempre— y "
        f"dejan el fallback traducido como rama muerta: {culpables}. Usa "
        f"`mensajeDeError(data, t('…'), t)`: traduce por CÓDIGO lo que el servidor sabe y "
        f"manda el `detail` crudo a la consola, no a la cara del usuario. [{_MARKER}]"
    )


def test_el_detector_veria_el_patron_original() -> None:
    """MUTACIÓN DE CONTROL. Si el regex dejara de casar, el test de arriba daría verde
    pasando en vacío — el modo de fallo de `P1-CULINARY-METADATA-BETA`."""
    for muestra in (
        "toast.error(data?.detail || t('Algo falló.'));",
        "toast.error(err.detail || t('Algo falló.'));",
        "  description: data?.detail || t('Reintenta.'),",
    ):
        assert _POSICION_DE_COPY.search(muestra), f"el detector no ve: {muestra!r}"


def test_el_detector_no_marca_un_throw() -> None:
    """La otra mitad del control: si marcara los `throw`, el guard pediría migrar 10
    sitios cuyo destino varía, y un guard que pide lo que no se puede dar se desactiva."""
    assert not _POSICION_DE_COPY.search(
        "throw new Error(data?.detail || t('No se pudo registrar.'));"
    )


# ============================================================
# 2 · El helper traduce por código y no pierde el detalle
# ============================================================

def _errorcopy() -> str:
    p = _SRC / "utils" / "errorCopy.js"
    if not p.exists():
        pytest.fail(
            f"No existe `utils/errorCopy.js`. Sin él, cada call site vuelve a decidir "
            f"por su cuenta si pinta el español del servidor. [{_MARKER}]"
        )
    return p.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "codigo",
    [
        "ai_unavailable", "ai_exhausted_retries", "swap_ai_unavailable",
        "swap_llm_retries_exhausted", "swap_clinical_violation",
        "swap_strict_pantry_no_inventory", "pantry_insufficient_for_goal",
        "budget_insufficient", "budget_below_goal_floor",
    ],
)
def test_los_codigos_que_el_backend_emite_tienen_copy(codigo: str) -> None:
    """Los ocho `error_code` canónicos del backend más `budget_insufficient`, que viaja
    como `detail.code`. Si el backend añade uno y aquí no está, ese error cae al fallback
    genérico — degradación aceptable, pero el usuario pierde el motivo concreto."""
    assert f"{codigo}:" in _errorcopy(), (
        f"`{codigo}` no tiene copy traducible. El backend lo emite y el usuario recibirá "
        f"un mensaje genérico en su lugar. [{_MARKER}]"
    )


def test_el_copy_es_una_funcion_de_t_y_no_una_cadena() -> None:
    """LA TRAMPA DEL CONGELADO. Un `t('…')` evaluado en ámbito de módulo se resuelve al
    IMPORTAR y se queda en el idioma de arranque — y en es-DO parece correcto. Por eso la
    tabla guarda funciones."""
    src = _errorcopy()
    m = re.search(r"const COPY_POR_CODIGO = \{(.*?)\n\};", src, re.S)
    assert m, "no encontré la tabla de copy"
    cuerpo = m.group(1)
    assert "(t) => t(" in cuerpo, (
        f"la tabla guarda cadenas y no funciones de `t`: se congelarían en el idioma de "
        f"arranque. [{_MARKER}]"
    )
    for linea in cuerpo.strip().split("\n"):
        if ":" in linea and "t(" in linea:
            assert "(t) =>" in linea, f"esta entrada no es función de `t`: {linea.strip()[:70]}"


def test_el_detail_crudo_no_se_pierde_va_a_la_consola() -> None:
    """El servidor SÍ sabe cosas que el cliente no. Traducir por código no puede
    significar tirar el diagnóstico — sólo sacarlo de la pantalla."""
    src = _errorcopy()
    assert "console.error" in src, (
        f"el `detail` sin código traducible se descarta en silencio. Tiene que ir a la "
        f"consola: los guards del repo preservan `console.error` en producción a "
        f"propósito, y Sentry lo recoge. [{_MARKER}]"
    )


def test_la_deuda_de_los_throw_esta_declarada() -> None:
    """Un alcance que se decide y no se anota vuelve como hallazgo en la siguiente
    auditoría."""
    src = _errorcopy()
    assert "throw new Error" in src and "catch" in src, (
        f"no está escrito por qué los `throw new Error(detail || …)` quedan fuera. Sin "
        f"esa razón, el siguiente auditor los cuenta como olvido — o los migra a ciegas. "
        f"[{_MARKER}]"
    )
