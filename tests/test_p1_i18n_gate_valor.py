"""[P1-I18N-GATE-VALOR · 2026-08-21] El validador medía que la CLAVE existiera,
nunca que el valor SIRVIERA.

QUÉ PASÓ. `i18n-check.mjs` calculaba `missing` con `!catKeys.has(k)` —presencia de
la clave— y `translated = catKeys.size - orphans.length` sin mirar el valor ni una
vez. Reproducido sobre el script real: poner `""` en una clave viva de `en-US.json`
daba `2339 traducidas / 0 faltan / 100.0% / ✅ Catálogos coherentes` y exit 0 en
modo ESTRICTO. Pasaban por traducidos `""`, `"   "`, `null`, `0`, `[]`, `{}`,
`{one:"",other:""}` y un plural `{other:"x"}` sin `one`.

Y no era una hipótesis de laboratorio: **las herramientas del propio repo escriben
exactamente esos valores**. `--write-template` (misma línea de este script) rellena
las faltantes con `''` o `{one:'',other:''}` DIRECTAMENTE en el catálogo que se
despacha, y `i18n-batches.mjs` lo invoca como primer paso de `split` mientras su
`merge` DESCARTA los vacíos en vez de completarlos. Toda clave que el traductor no
devolviera quedaba en `""` con `missing = 0` para siempre.

POR QUÉ ES P1 Y NO P2. El usuario no ve un hueco: el motor cae al español
(`typeof hit === 'string' && hit !== ''`, src/i18n/index.js). El defecto es el
FALSO VERDE — la cifra de cobertura, que es el único número con el que alguien
decide si un idioma está listo, no puede distinguir «traducido» de «hueco relleno
por una herramienta». Un 100% que incluye blancos no mide cobertura.

QUÉ ANCLA. Que el modo estricto rechace un valor inservible con la misma dureza
con que rechaza una clave ausente, en las dos formas (cadena vacía y plural mal
formado) y en las dos direcciones del desajuste de forma. Y la mutación de
control: con los valores buenos, verde — sin ella el test no distinguiría «el
checker detecta blancos» de «el checker falla siempre».

MÉTODO. Se copia el script REAL a un tmpdir con un `src/` mínimo y se ejecuta con
node. Reimplementar su lógica en el test mediría el test, no el validador — el
error que este repo ya registró como «un fake que aplica la regla en vez de
modelar el SQL no ve el bug del SQL».
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_FRONTEND = _ROOT / "frontend"
_CHECKER = _FRONTEND / "scripts" / "i18n-check.mjs"

_MARKER = "P1-I18N-GATE-VALOR"

# El fixture declara DOS idiomas: el base (sin catálogo, es el fallback) y uno
# objetivo. Los dos regex que el script usa para leer el SSOT son
# `DEFAULT_LOCALE = '...'` y `{ code: '...'`, así que el fixture los reproduce.
_LOCALES_JS = """
export const DEFAULT_LOCALE = 'es-DO';
export const LOCALES = [
    { code: 'es-DO', native: 'Español' },
    { code: 'en-US', native: 'English' },
];
"""

# Un componente con una clave simple y una de plural. Va fuera de `src/i18n/`
# porque el `walk` del script salta ese directorio a propósito.
_COMPONENTE = """
import { useT } from '../i18n';
export function Demo({ n }) {
    const { t, tn } = useT();
    return <p>{t('Hola')} {tn(n, 'plato', 'platos')}</p>;
}
"""


def _tiene_node() -> bool:
    return shutil.which("node") is not None


def _montar(tmp: Path, catalogo: dict) -> Path:
    """Un checkout mínimo: el script real + un `src/` de tres ficheros."""
    if not _CHECKER.exists():
        pytest.skip(f"{_CHECKER} no existe en este checkout (repos hermanos)")

    (tmp / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(_CHECKER, tmp / "scripts" / "i18n-check.mjs")

    src = tmp / "src"
    (src / "i18n" / "locales").mkdir(parents=True, exist_ok=True)
    (src / "i18n" / "locales.js").write_text(_LOCALES_JS, encoding="utf-8")
    (src / "componentes").mkdir(parents=True, exist_ok=True)
    (src / "componentes" / "Demo.jsx").write_text(_COMPONENTE, encoding="utf-8")

    (src / "i18n" / "locales" / "en-US.json").write_text(
        json.dumps(catalogo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return tmp / "scripts" / "i18n-check.mjs"


def _correr(tmp: Path, catalogo: dict, estricto: bool = True):
    script = _montar(tmp, catalogo)
    cmd = ["node", str(script)] + (["--strict"] if estricto else [])
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


_BUENO = {"Hola": "Hello", "platos": {"one": "dish", "other": "dishes"}}


@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
def test_control_un_catalogo_completo_pasa(tmp_path: Path) -> None:
    """MUTACIÓN DE CONTROL. Sin esto, un checker que falle SIEMPRE pasaría todos
    los casos de abajo y el fichero entero sería una coartada."""
    r = _correr(tmp_path, _BUENO)
    assert r.returncode == 0, (
        "El fixture BUENO tiene las dos claves bien traducidas y aun así el "
        f"checker falla. Entonces los casos de este fichero no prueban nada.\n"
        f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )


@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
def test_control_una_clave_ausente_falla(tmp_path: Path) -> None:
    """El caso que el checker YA detectaba. Fija la referencia: lo que sigue tiene
    que fallar con la misma dureza que esto."""
    catalogo = dict(_BUENO)
    del catalogo["Hola"]
    r = _correr(tmp_path, catalogo)
    assert r.returncode != 0, "Una clave ausente tiene que fallar en --strict."


@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
@pytest.mark.parametrize(
    "valor,etiqueta",
    [
        ("", "cadena vacía"),
        ("   ", "solo espacios"),
        (None, "null"),
        (0, "cero"),
        ([], "array vacío"),
    ],
)
def test_un_valor_inservible_no_cuenta_como_traducido(tmp_path: Path, valor, etiqueta: str) -> None:
    """Una clave presente con valor inservible es indistinguible de una ausente
    para el usuario: el motor cae al español en los dos casos. El gate tiene que
    tratarlas igual, o la cobertura miente."""
    catalogo = dict(_BUENO)
    catalogo["Hola"] = valor
    r = _correr(tmp_path, catalogo)
    assert r.returncode != 0, (
        f"El checker acepta {etiqueta} ({valor!r}) como traducción válida y reporta "
        "cobertura completa. Es el falso verde que `--write-template` fabrica en "
        f"cada `split` de i18n-batches. [{_MARKER}]\nstdout:\n{r.stdout}"
    )


@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
@pytest.mark.parametrize(
    "valor,etiqueta",
    [
        ({}, "objeto vacío"),
        ({"one": "", "other": ""}, "ambas formas en blanco"),
        ({"one": "dish"}, "sin la forma `other`"),
        ({"other": "dishes"}, "sin la forma `one`"),
    ],
)
def test_un_plural_incompleto_no_cuenta_como_traducido(tmp_path: Path, valor, etiqueta: str) -> None:
    """`badPlurals` solo miraba la dirección objeto→cadena: cazaba el plural
    declarado como texto plano y dejaba pasar el objeto a medio rellenar, que es
    justo la forma que escribe `--write-template`."""
    catalogo = dict(_BUENO)
    catalogo["platos"] = valor
    r = _correr(tmp_path, catalogo)
    assert r.returncode != 0, (
        f"El checker acepta un plural con {etiqueta} ({valor!r}). Un plural sin "
        "sus dos formas traduce en singular SIEMPRE, sin avisar — que es la razón "
        f"por la que `badPlurals` existe. [{_MARKER}]\nstdout:\n{r.stdout}"
    )


@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
def test_una_clave_simple_declarada_como_objeto_falla(tmp_path: Path) -> None:
    """La dirección contraria del desajuste de forma. `t()` con una entrada objeto
    cae al español (el motor lo dice: «es más honesto que pintar
    [object Object]»), así que es un hueco silencioso más."""
    catalogo = dict(_BUENO)
    catalogo["Hola"] = {"one": "Hi", "other": "Hi"}
    r = _correr(tmp_path, catalogo)
    assert r.returncode != 0, (
        "El checker acepta una clave NO plural declarada como objeto. `t()` la "
        f"descarta y pinta español. [{_MARKER}]\nstdout:\n{r.stdout}"
    )


@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
def test_la_cobertura_descuenta_los_blancos(tmp_path: Path) -> None:
    """No basta con el exit code: la CIFRA es lo que alguien mira para decidir si
    un idioma está listo. Un 100% que incluye blancos es peor que un fallo, porque
    se cita en una decisión."""
    catalogo = dict(_BUENO)
    catalogo["Hola"] = ""
    r = _correr(tmp_path, catalogo)
    salida = r.stdout + r.stderr
    assert "100.0%" not in salida, (
        "Con una de dos claves en blanco, el informe sigue diciendo 100.0%. La "
        f"cobertura tiene que descontar los valores inservibles. [{_MARKER}]\n{salida}"
    )


@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
def test_el_modo_permisivo_tambien_rechaza_un_blanco(tmp_path: Path) -> None:
    """Un blanco NO es «todavía no traducido» —eso es la clave ausente, y en modo
    permisivo se tolera a propósito durante una migración—. Un blanco es una clave
    que YA se procesó y salió vacía: siempre es un defecto, con o sin `--strict`."""
    catalogo = dict(_BUENO)
    catalogo["Hola"] = ""
    r = _correr(tmp_path, catalogo, estricto=False)
    assert r.returncode != 0, (
        "En modo permisivo un blanco pasa. Pero permisivo tolera lo AUSENTE "
        "(migración a medias, cae a español y es coherente), no lo procesado-y-"
        f"vacío, que es siempre un error de la herramienta o del traductor. [{_MARKER}]"
    )
