"""[P2-I18N-GATE-ENDURECIDO · 2026-08-22] Seis formas que tenia el gate de i18n de decir
que si sin haber comprobado nada.

El motor usa el texto espanol COMO clave, asi que cambiar un copy huerfana su traduccion EN
SILENCIO. Este gate es la unica defensa que existe contra eso -- y por eso lo que le falle a
EL cuesta mas que lo que le falle a una pantalla.

LAS SEIS, y lo que cada una dejaba pasar:

  1. P2-I18N-GATE-CIEGO-PLACEHOLDER. Medía que el valor existiera y SIRVIERA, nunca que
     conservara los `{placeholders}` de la clave. Un placeholder PERDIDO borra el dato de la
     pantalla («Te quedan {n} comidas» sin `{n}` deja «Te quedan comidas»: parece correcta y
     no dice nada). Uno INVENTADO se pinta LITERAL, porque `_interpolate` solo sustituye las
     claves que le pasan: el usuario ve `{dias}` en crudo.

  2. P3-I18N-CHECK-SIN-MARCADO. Nueve claves llevan HTML que entra al `innerHTML` del PDF.
     Una traduccion que se deja un `</strong>` rompe el documento a partir de ahi. El
     validador de secuencia de tags existia SOLO en la herramienta de merge -- vigilaba la
     puerta por la que entran las traducciones nuevas, no el estado del catalogo.

  3. P2-I18N-TRINQUETE-DESAPARECE-EN-SILENCIO. Si el fichero de baseline faltaba o no
     parseaba, el trinquete se apagaba y el gate salia VERDE. La forma mas facil de
     desactivar la defensa era BORRAR un fichero, y nada lo decia. Es la misma clase que
     `P1-CI-GATE-INCONCLUSIVE`: «no concluyente» no puede colapsar a ningun lado.

  4. P3-I18N-TRINQUETE-SIN-COMPROBACION-DE-DIRECCION. «Puede BAJAR, nunca subir» vivia SOLO
     en un comentario: `--update-baseline` reescribia el valor sin mirar. O sea que la forma
     de convertir un rojo en verde era ejecutar el comando que el propio mensaje de error
     sugiere. No se prohibe subirlo --hay casos legitimos-- pero deja de ser gratis y
     silencioso.

  5. P3-I18N-COBERTURA-REDONDEA-A-100. `toFixed(1)` REDONDEA: con 2.524 de 2.525 imprimia
     «100.0%» junto a un «1» en la columna FALTAN. La cobertura es el unico numero con el
     que alguien decide si un idioma esta listo.

  6. P3-I18N-KEYDECL-TRAS-EL-FILTRO-BARATO. La extraccion de `i18nKey()` estaba DETRAS del
     filtro `if (!/\\bt\\(|\\btn\\(/.test(src)) continue`, asi que un fichero que solo DECLARA
     claves ni se abria: su traduccion salia HUERFANA en los cuatro idiomas y el mensaje del
     gate invita literalmente a borrarla. Mismo defecto que `P1-I18N-GATE-CIEGO-SIN-T`, un
     nivel mas abajo.

TODAS SE VERIFICARON POR MUTACION antes de escribir esto: borrar un `{n}`, quitar un
`</strong>`, borrar el baseline, corromperlo, y bajarlo a mano para forzar una subida. Un
guard que nunca se ha visto rojo no ha probado nada.

tooltip-anchor: P2-I18N-GATE-ENDURECIDO
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_CHECKER = _ROOT / "frontend" / "scripts" / "i18n-check.mjs"

_MARKER = "P2-I18N-GATE-ENDURECIDO"


def _fuente() -> str:
    if not _CHECKER.exists():
        pytest.skip(f"no existe {_CHECKER} (¿repo hermano sin clonar?)")
    return io.open(_CHECKER, encoding="utf-8").read()


def test_valida_los_placeholders_de_la_clave() -> None:
    src = _fuente()
    assert re.search(r"placeholders distintos de la clave", src), (
        f"el gate dejó de comparar los `{{placeholders}}` del valor contra los de la clave. "
        f"Un placeholder perdido borra el dato de la pantalla y uno inventado se pinta "
        f"literal; las dos formas se cuelan enteras por el check de cobertura. [{_MARKER}]"
    )


def test_valida_la_secuencia_de_etiquetas_html() -> None:
    src = _fuente()
    assert re.search(r"secuencia de etiquetas HTML distinta de la clave", src), (
        f"el gate dejó de comparar el marcado. Nueve claves llevan HTML que entra al "
        f"`innerHTML` del PDF: una traducción que se deja un `</strong>` rompe el documento "
        f"a partir de ahí. [{_MARKER}]"
    )


def test_el_trinquete_ausente_o_corrupto_es_fallo_duro() -> None:
    """La forma más fácil de desactivar la defensa era borrar un fichero."""
    src = _fuente()
    assert "Falta el trinquete de español sin envolver" in src, (
        f"un baseline AUSENTE vuelve a ser silencioso: sin él no hay detección de "
        f"retrocesos y el gate sale verde por omisión. [{_MARKER}]"
    )
    assert "no es JSON válido" in src, (
        f"un baseline CORRUPTO vuelve a ser silencioso. [{_MARKER}]"
    )
    # …y sólo `--update-baseline` lo perdona, que es el comando que existe para crearlo.
    assert re.search(r"UPDATE_BASELINE\s*\)\s*\{", src) or "!UPDATE_BASELINE" in src, (
        f"la excepción para `--update-baseline` desapareció: crear el fichero por primera "
        f"vez sería imposible. [{_MARKER}]"
    )


def test_el_trinquete_no_sube_sin_decirlo() -> None:
    src = _fuente()
    assert "ALLOW_RATCHET_UP" in src and "--allow-ratchet-up" in src, (
        f"`--update-baseline` vuelve a poder SUBIR el trinquete en silencio. Era la forma "
        f"de convertir un rojo en verde ejecutando el comando que el propio mensaje de "
        f"error sugiere. [{_MARKER}]"
    )
    assert "SUBIRÍA el trinquete" in src, (
        f"desapareció el aviso que enumera los ficheros que subirían. [{_MARKER}]"
    )


def test_la_cobertura_no_redondea_hasta_100() -> None:
    src = _fuente()
    assert "Math.floor(r.coverage * 1000)" in src, (
        f"la columna COBERTURA vuelve a redondear con `toFixed`: 2.524 de 2.525 imprimía "
        f"«100.0%» junto a un «1» en FALTAN, y es el único número con el que alguien decide "
        f"si un idioma está listo. [{_MARKER}]"
    )


def test_las_declaraciones_de_clave_entran_en_el_filtro() -> None:
    src = _fuente()
    m = re.search(r"if \(!/([^/]+)/\.test\(src\)\) continue;", src)
    assert m, f"no encontré el filtro de ficheros del escáner [{_MARKER}]"
    filtro = m.group(1)
    assert "i18nKey" in filtro, (
        f"`i18nKey\\(` volvió a quedarse fuera del filtro (`{filtro}`), así que un fichero "
        f"que sólo DECLARA claves no se abre y su traducción sale huérfana en los cuatro "
        f"idiomas — con el mensaje del gate invitando a borrarla. [{_MARKER}]"
    )
    for esperado in ("t\\(", "tn\\("):
        assert esperado in filtro, (
            f"el filtro dejó de reconocer `{esperado}`: se saltaría ficheros con "
            f"traducciones reales. [{_MARKER}]"
        )
