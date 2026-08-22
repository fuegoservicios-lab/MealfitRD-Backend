"""[P2-I18N-GATE-ACTIONS-INALCANZABLE · 2026-08-22] El gate de i18n de GitHub Actions
llevaba 21 corridas SIN EJECUTARSE, y toda la ola que cerro los 58 gaps de i18n se
mergeo mientras tanto.

QUE PASO, medido con `gh run list` sobre el repo del frontend:

    ultimo verde   2026-08-21T23:10:49Z
    21 corridas consecutivas en `failure` despues, hasta 2026-08-22T06:09:31Z
    en TODAS ellas: `npm run i18n:check:strict` -> conclusion `skipped`

El job `quality` es una cadena SERIAL de doce pasos sin `if: always()` ni
`continue-on-error`. El paso de i18n era el DECIMO, detras de cinco pasos bloqueantes
(eslint, lint:count, typecheck, huerfanos, npm test). Cualquiera de los cinco lo
saltaba junto con los siete que venian detras.

LA CAUSA DE AQUELLAS 21 FUE UN IMPORT MUERTO. `vi` importado y sin usar en
`src/__tests__/legal_links_apex.test.jsx:24` -> `no-unused-vars`, que es un ERROR de
eslint, no un aviso. Y esto importa mas que la anecdota:

    ✖ 66 problems (1 error, 66 warnings)

`--max-warnings 66` NO puede absorber un error. Quien viera «67 > 66» y subiera el
techo a 67 no habria arreglado nada -- los avisos estaban justo en el tope y el
bloqueante era el error. Es la tercera vez que este job se rompe por un paso de
adelante (`P1-CI-QUALITY-ABORTADO` fue la anterior, y REGRESO en menos de 27 horas),
asi que el arreglo no puede ser volver a quitar el error del dia: tiene que ser que el
gate de traducciones deje de colgar de los demas.

EL ARREGLO: el paso de i18n va PRIMERO, justo despues de instalar dependencias. Tarda
~2 s y su fallo es de una linea, asi que ademas la causa sale antes que los ~2 min de
vitest. Es el MISMO orden que ya usa el gate de release (`scripts/run_ci.ps1`, «Frontend
i18n» antes de «Frontend vitest») -- que es exactamente el que NO se rompio en esas 21
corridas, y por eso el margen de 66/66 de hoy no vuelve a dejar el repo sin defensa.

POR QUE NO `if: always()`: el paso seguiria corriendo, si, pero DESPUES de vitest, y con
el job ya en rojo por otra causa. Lo que se quiere no es que se ejecute igualmente, es
que la unica defensa contra una traduccion huerfana no dependa de que nadie deje un
import muerto en ningun sitio del repo.

ESTE GUARD MIDE LA PROPIEDAD, NO LA GRAFIA. No comprueba «el paso 2 se llama X»: recoge
los pasos `run:` del job en ORDEN y exige que lo unico que preceda al de i18n sea
instalacion de dependencias. Sigue vivo si manana se anaden pasos, se renombran o se
reordenan los demas.

tooltip-anchor: P2-I18N-GATE-ACTIONS-INALCANZABLE
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_WF = _ROOT / "frontend" / ".github" / "workflows" / "ci.yml"

_MARKER = "P2-I18N-GATE-ACTIONS-INALCANZABLE"

# Lo unico que puede preceder al gate de i18n: traerse el codigo y las dependencias.
# Cualquier otra cosa es un paso que puede fallar y saltarselo.
_PERMITIDO_ANTES = re.compile(r"^npm ci$|^npm install\b")


def _fuente() -> str:
    if not _WF.exists():  # el frontend es un repo hermano; puede no estar clonado
        pytest.skip(f"no existe {_WF}")
    return io.open(_WF, encoding="utf-8").read()


def _pasos_run_del_job(fuente: str, job: str) -> list[str]:
    """Los comandos `- run:` del job `job`, EN ORDEN.

    Sin dependencia de YAML: el fichero fija la indentacion (`  <job>:` a dos espacios)
    y el job termina en la siguiente clave al mismo nivel.
    """
    lineas = fuente.splitlines()
    ini = None
    for i, l in enumerate(lineas):
        if re.match(rf"^  {re.escape(job)}:\s*$", l):
            ini = i
            break
    assert ini is not None, f"no encontre el job `{job}` en {_WF.name} [{_MARKER}]"

    fin = len(lineas)
    for i in range(ini + 1, len(lineas)):
        if re.match(r"^  \S", lineas[i]):  # siguiente job al mismo nivel
            fin = i
            break

    pasos: list[str] = []
    for l in lineas[ini:fin]:
        m = re.match(r"^\s*-?\s*run:\s*(?:\|\s*)?(.*)$", l)
        if m and m.group(1).strip():
            pasos.append(m.group(1).strip())
    return pasos


def _indice_i18n(pasos: list[str]) -> int:
    for i, p in enumerate(pasos):
        if "i18n:check" in p:
            return i
    return -1


def test_el_job_quality_tiene_el_gate_de_i18n() -> None:
    pasos = _pasos_run_del_job(_fuente(), "quality")
    assert _indice_i18n(pasos) >= 0, (
        "el job `quality` no ejecuta `npm run i18n:check*`. Es la unica defensa contra "
        f"que un cambio de copy huerfane su traduccion en los 4 idiomas. [{_MARKER}]"
    )


def test_el_gate_de_i18n_corre_en_estricto() -> None:
    pasos = _pasos_run_del_job(_fuente(), "quality")
    paso = pasos[_indice_i18n(pasos)]
    assert "i18n:check:strict" in paso, (
        f"el gate corre en modo permisivo (`{paso}`). Sin `--strict` una clave sin "
        "traduccion NO tumba nada: cae al espanol y la pantalla queda a medias en "
        f"silencio, que es justo el caso que motivo ponerlo. [{_MARKER}]"
    )


def test_nada_que_pueda_fallar_precede_al_gate_de_i18n() -> None:
    """LA invariante. Lo unico antes del gate puede ser instalar dependencias.

    Si esto falla, el gate ha vuelto a quedar detras de un paso bloqueante y una
    corrida roja por CUALQUIER otra causa lo salta -- que es exactamente lo que
    dejo 21 corridas sin verificar una sola traduccion.
    """
    pasos = _pasos_run_del_job(_fuente(), "quality")
    i = _indice_i18n(pasos)
    assert i >= 0, f"sin paso de i18n en `quality` [{_MARKER}]"

    intrusos = [p for p in pasos[:i] if not _PERMITIDO_ANTES.match(p)]
    assert not intrusos, (
        "estos pasos van ANTES del gate de i18n y pueden saltarselo al fallar: "
        f"{intrusos}. El job es una cadena serial sin `if: always()`, asi que "
        "cualquiera de ellos deja las traducciones sin verificar -- medido: 21 "
        f"corridas consecutivas con el paso en `skipped`. [{_MARKER}]"
    )


def test_el_orden_espeja_al_gate_de_release() -> None:
    """El gate de release ya pone i18n antes de vitest y por eso NO se rompio.

    Anclar la paridad evita que los dos caminos vuelvan a divergir: durante esas 21
    corridas, `run_ci.ps1` seguia verificando las traducciones y Actions no.
    """
    pasos = _pasos_run_del_job(_fuente(), "quality")
    i = _indice_i18n(pasos)
    tests = [j for j, p in enumerate(pasos) if re.match(r"^npm (run )?test\b", p)]
    if not tests:
        pytest.skip("el job no corre vitest; la paridad no aplica")
    assert i < min(tests), (
        f"i18n (paso {i}) va DESPUES de vitest (paso {min(tests)}). El gate de release "
        "lo pone antes a proposito: tarda ~2 s y su fallo es de una linea, mientras la "
        f"suite tarda ~2 min. [{_MARKER}]"
    )


def test_el_fichero_explica_por_que_va_primero() -> None:
    """Sin la razon escrita, el proximo que ordene los pasos por estetica lo mueve.

    Es la misma clase de perdida que `P1-CI-QUALITY-ABORTADO`, que regreso en 27 h.
    """
    fuente = _fuente()
    assert _MARKER in fuente, (
        f"el workflow no cita `{_MARKER}`, asi que nada explica por que el paso de "
        f"i18n va el primero. [{_MARKER}]"
    )
