"""[P1-I18N-EXTRACTOR-AST · 2026-08-21] El guard de «`t()` en ámbito de módulo»
era ciego al ejemplo que la propia documentación usa para explicar la trampa.

QUÉ PASÓ. `i18n-check.mjs` decidía si una llamada estaba en ámbito de módulo
contando llaves: `scanAt` acumulaba `{` y `}` y `isModuleScopeCode` exigía
`depth === 0`. Pero un literal de objeto o de array TAMBIÉN abre llave, así que
cualquier tabla de copy —la forma exacta del bug— salía con `depth >= 1` y no se
reportaba.

Medido con un fixture de cinco casos contra el script real, ANTES del arreglo:

    const TABS = [{ label: t('X') }];      → NO reportado   ← el ejemplo de la doc
    const COPY = { titulo: t('X') };       → NO reportado
    const SUELTO = t('X');                 → reportado
    function ok() { return t('X'); }       → no reportado (correcto)
    function ok2() { return `${t('X')}`; } → no reportado (correcto)

Uno de tres. Y el que se escapaba es literalmente el que `backend/docs/i18n_dashboard.md`
y la cabecera de `src/i18n/index.js` ponen como ejemplo canónico:

    const TABS = [{ label: t('Plan') }];   // ❌ congelado en español para siempre

POR QUÉ IMPORTA. Es el bug más difícil de ver del sistema —la doc lo dice— porque
en `es-DO` se ve perfecto y pasa cualquier revisión visual: el array se evalúa al
importar, antes de que exista el catálogo, y queda en español para los otros
cuatro idiomas para siempre. El guard existe SOLO para eso y no cubría su caso
principal.

QUÉ CAMBIA. La pregunta «¿esto corre al importar?» es «¿hay una función entre
esta llamada y la raíz del módulo?», y eso no lo puede responder un contador de
llaves: lo responde un AST. Se pasa a `@babel/parser`, que además se DECLARA como
devDependency — hasta hoy llegaba de rebote por `@vitejs/plugin-react`, y un
script de gate apoyado en un transitivo se rompe en el siguiente bump del
lockfile sin que nadie lo haya decidido.

La extracción de CLAVES sigue por regex a propósito: una clave tiene que ser un
literal estático para poder existir en un catálogo, así que el regex es exacto
para eso y no hay nada que ganar cambiándolo.
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
_PKG = _FRONTEND / "package.json"

_MARKER = "P1-I18N-EXTRACTOR-AST"

_LOCALES_JS = """
export const DEFAULT_LOCALE = 'es-DO';
export const LOCALES = [
    { code: 'es-DO', native: 'Español' },
    { code: 'en-US', native: 'English' },
];
"""


def _tiene_node() -> bool:
    return shutil.which("node") is not None


def _correr(tmp: Path, fuente: str, catalogo: dict | None = None):
    """Ejecuta el checker REAL contra un `src/` mínimo con la fuente dada."""
    if not _CHECKER.exists():
        pytest.skip(f"{_CHECKER} no existe en este checkout (repos hermanos)")

    (tmp / "scripts").mkdir(parents=True, exist_ok=True)
    # [P1-I18N-GATE-CIEGO-SIN-T · 2026-08-21] Copiar SÓLO `i18n-check.mjs` dejó de
    # bastar: desde ese P-fix importa `i18n-sin-envolver.mjs` (el detector de literales
    # nunca envueltos), `i18n-alcance.mjs` (qué ficheros están dentro del alcance) y
    # `lib/grafo-modulos.mjs` (el grafo de imports, compartido con `huerfanos.mjs`).
    # Con un solo fichero el `node` del tmpdir muere en ERR_MODULE_NOT_FOUND y los
    # asertos fallan por una razón que no tiene nada que ver con lo que miden.
    #
    # Se copia el SET completo, no se relaja el aserto: el arnés dice «ejecuta el
    # checker REAL», y el checker real tiene dependencias.
    for _rel in ("i18n-check.mjs", "i18n-sin-envolver.mjs", "i18n-alcance.mjs"):
        _origen = _CHECKER.parent / _rel
        if _origen.exists():
            shutil.copy2(_origen, tmp / "scripts" / _rel)
    _lib = _CHECKER.parent / "lib"
    if _lib.exists():
        (tmp / "scripts" / "lib").mkdir(parents=True, exist_ok=True)
        for _f in _lib.glob("*.mjs"):
            shutil.copy2(_f, tmp / "scripts" / "lib" / _f.name)

    src = tmp / "src"
    (src / "i18n" / "locales").mkdir(parents=True, exist_ok=True)
    (src / "i18n" / "locales.js").write_text(_LOCALES_JS, encoding="utf-8")
    (src / "c").mkdir(parents=True, exist_ok=True)
    (src / "c" / "Demo.jsx").write_text(fuente, encoding="utf-8")
    (src / "i18n" / "locales" / "en-US.json").write_text(
        json.dumps(catalogo if catalogo is not None else {}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # `node_modules` del frontend real, para que el script resuelva sus imports
    # sin instalar nada en el tmpdir.
    enlace = tmp / "node_modules"
    real = _FRONTEND / "node_modules"
    if real.exists() and not enlace.exists():
        try:
            enlace.symlink_to(real, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("no puedo enlazar node_modules en esta plataforma")

    return subprocess.run(
        ["node", str(tmp / "scripts" / "i18n-check.mjs")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def _reporta(salida: str, clave: str) -> bool:
    return "ÁMBITO DE MÓDULO" in salida and json.dumps(clave, ensure_ascii=False) in salida


@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
def test_babel_parser_esta_declarado_como_dependencia() -> None:
    """Un gate no puede apoyarse en un paquete que le llega de rebote.

    `@babel/parser` entraba por `@vitejs/plugin-react` → `@babel/core`. Funciona
    hasta que un bump del lockfile lo deduplica de otra forma, y entonces el gate
    de i18n muere con un `ERR_MODULE_NOT_FOUND` que nadie relacionó con ese bump.
    """
    if not _PKG.exists():
        pytest.skip("package.json no existe en este checkout")
    pkg = json.loads(_PKG.read_text(encoding="utf-8"))
    declaradas = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    assert "@babel/parser" in declaradas, (
        "`scripts/i18n-check.mjs` importa @babel/parser y package.json no lo "
        f"declara: hoy llega como transitivo de @vitejs/plugin-react. [{_MARKER}]"
    )


# ─────────────────────── los tres casos de ámbito de módulo ───────────────────────
#
# Los tres se evalúan al IMPORTAR y quedan congelados en español. El guard sólo
# cazaba el tercero, porque los dos primeros abren una llave que su contador
# interpretaba como «estoy dentro de algo».

@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
def test_caza_el_ejemplo_canonico_de_la_doc(tmp_path: Path) -> None:
    """`const TABS = [{ label: t('X') }]` — la forma que la documentación usa para
    explicar la trampa, y la única que de verdad aparece en el código real: nadie
    escribe `const X = t('...')` suelto, todo el mundo escribe tablas de copy."""
    fuente = (
        "import { t } from '../i18n';\n"
        "const TABS = [{ label: t('CongeladoEnArray') }];\n"
        "export default TABS;\n"
    )
    r = _correr(tmp_path, fuente)
    assert _reporta(r.stdout + r.stderr, "CongeladoEnArray"), (
        "El guard no reporta el ejemplo canónico de la doc. Un array de copy en "
        "ámbito de módulo corre ANTES de que el catálogo exista y queda en "
        f"español para siempre. [{_MARKER}]\n{r.stdout}\n{r.stderr}"
    )


@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
def test_caza_un_objeto_de_copy_en_ambito_de_modulo(tmp_path: Path) -> None:
    fuente = (
        "import { t } from '../i18n';\n"
        "const COPY = { titulo: t('CongeladoEnObjeto') };\n"
        "export default COPY;\n"
    )
    r = _correr(tmp_path, fuente)
    assert _reporta(r.stdout + r.stderr, "CongeladoEnObjeto"), (
        f"Un objeto de copy en ámbito de módulo no se reporta. [{_MARKER}]\n{r.stdout}"
    )


@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
def test_sigue_cazando_la_asignacion_desnuda(tmp_path: Path) -> None:
    """NO REGRESIÓN. Es el único caso que el contador de llaves sí veía; el AST
    tiene que seguir viéndolo o el arreglo cambia un agujero por otro."""
    fuente = (
        "import { t } from '../i18n';\n"
        "const SUELTO = t('CongeladoSuelto');\n"
        "export default SUELTO;\n"
    )
    r = _correr(tmp_path, fuente)
    assert _reporta(r.stdout + r.stderr, "CongeladoSuelto"), (
        f"Se perdió el caso que el guard viejo SÍ cazaba. [{_MARKER}]\n{r.stdout}"
    )


@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
def test_caza_un_array_anidado_en_una_constante_exportada(tmp_path: Path) -> None:
    """La forma real del repo: una tabla de navegación con anidamiento de verdad."""
    fuente = (
        "import { t } from '../i18n';\n"
        "export const NAV = [\n"
        "    { id: 'plan', label: t('CongeladoAnidado'), hijos: [{ label: t('CongeladoHondo') }] },\n"
        "];\n"
    )
    r = _correr(tmp_path, fuente)
    salida = r.stdout + r.stderr
    assert _reporta(salida, "CongeladoAnidado"), f"[{_MARKER}]\n{salida}"
    assert _reporta(salida, "CongeladoHondo"), (
        f"El anidamiento profundo se escapa. [{_MARKER}]\n{salida}"
    )


# ───────────────────────── lo que NO se debe reportar ─────────────────────────
#
# Un guard que grita con todo se apaga en una semana. Estos casos son CORRECTOS y
# tienen que salir en silencio: si el arreglo los reporta, cambia un falso
# negativo por un falso positivo, que no es arreglar el guard.

@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
@pytest.mark.parametrize(
    "etiqueta,fuente",
    [
        (
            "función declarada",
            "import { t } from '../i18n';\n"
            "export function getTabs() { return [{ label: t('Bien') }]; }\n",
        ),
        (
            "flecha",
            "import { t } from '../i18n';\n"
            "export const getTabs = () => [{ label: t('Bien') }];\n",
        ),
        (
            "template literal dentro de función",
            "import { t } from '../i18n';\n"
            "export function saludo() { return `${t('Bien')}`; }\n",
        ),
        (
            "componente con hook",
            "import { useT } from '../i18n';\n"
            "export function C() { const { t } = useT(); return <p>{t('Bien')}</p>; }\n",
        ),
        (
            "método de objeto",
            "import { t } from '../i18n';\n"
            "export const api = { etiqueta() { return t('Bien'); } };\n",
        ),
        (
            "callback dentro de una constante de módulo",
            "import { t } from '../i18n';\n"
            "export const COLS = [{ render: () => t('Bien') }];\n",
        ),
        (
            "citado en un comentario",
            "import { t } from '../i18n';\n"
            "// Ojo: `const X = t('Bien')` en ámbito de módulo se congela.\n"
            "export function f() { return t('Bien'); }\n",
        ),
    ],
)
def test_no_reporta_lo_que_esta_bien(tmp_path: Path, etiqueta: str, fuente: str) -> None:
    r = _correr(tmp_path, fuente, catalogo={"Bien": "Fine"})
    salida = r.stdout + r.stderr
    assert "ÁMBITO DE MÓDULO" not in salida, (
        f"Falso positivo con {etiqueta}: esa llamada corre en render, no al "
        f"importar. Un guard que grita con lo correcto se acaba apagando. "
        f"[{_MARKER}]\n{salida}"
    )


@pytest.mark.skipif(not _tiene_node(), reason="node no está en PATH")
def test_el_repo_real_no_tiene_ninguna(tmp_path: Path) -> None:
    """Con el guard ya viendo las tres formas, el repo tiene que seguir limpio.

    Si esto falla tras el arreglo, NO relajes el guard: acaba de destapar copy
    congelado de verdad, que es exactamente para lo que se escribió.
    """
    if not shutil.which("npm"):
        pytest.skip("npm no está en PATH")
    r = subprocess.run(
        ["npm", "run", "i18n:check", "--silent"],
        cwd=str(_FRONTEND), capture_output=True, text=True,
        encoding="utf-8", errors="replace", shell=True,
    )
    assert "ÁMBITO DE MÓDULO" not in (r.stdout + r.stderr), (
        "El guard, ya arreglado, encuentra `t()` en ámbito de módulo en el repo "
        "real. Eso es copy congelado en español para los 4 idiomas. Conviértelo "
        f"en función y llámalo en render. [{_MARKER}]\n{r.stdout}\n{r.stderr}"
    )
