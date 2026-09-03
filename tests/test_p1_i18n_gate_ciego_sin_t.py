"""[P1-I18N-GATE-CIEGO-SIN-T · 2026-08-21] El gate cantaba 100 % sobre el
denominador que él mismo definía.

QUÉ PASÓ. `i18n-check.mjs` extrae las claves de las llamadas `t()`/`tn()` y las
coteja contra los catálogos. Es exacto para lo que mide — y lo que mide es sólo lo
que YA está envuelto: **una cadena que nunca pasó por `t()` no entra en
`liveKeys`, luego no puede faltar**. Encima, el bucle de ficheros empezaba con

    if (!/\\bt\\(|\\btn\\(/.test(src)) continue;

así que un fichero sin UNA SOLA llamada `t()` ni siquiera se abría. Medido: ocho
utils de etiquetas (`planWeeks`, `shelfLife`, `authErrors`, `chunkStatus`,
`chunkKinds`, `foodSearch`, `routeMeta`, `todayRemaining`) con `t()/tn()` = 0 y
español puro dentro, invisibles para un gate que reportaba «100,0 % en los 4
idiomas».

POR QUÉ ES LA CAUSA RAÍZ. El comentario del gate en `run_ci.ps1` atribuía las ocho
superficies en español del 2026-08-20 a que «i18n:check sólo corría cuando alguien
se acordaba» y concluía que ponerlo en estricto lo cerraba. No lo cierra: aquellas
eran de la forma «nunca fue clave» —`P1-HIST-DIAS-I18N`: `History.jsx` era el
único sitio con el array de días crudo— y estricto no puede ver eso. Sin este
detector, cada arreglo de superficie del plan se reabre solo.

MEDIDO AL CABLEARLO. 1.149 literales en español en `src/`; 153 dentro del alcance
declarado, en 23 ficheros. La diferencia es landing y legales, que se descuentan
DERIVÁNDOLOS del SSOT de rutas + el grafo de imports, no con una lista a mano.
El reparto de los 153 casa uno a uno con las tareas del plan: AssessmentContext
46, History 31, PendingPipelineRecovery 13, authErrors 9 (sus nueve `return`).

POR QUÉ TRINQUETE Y NO ERROR. Con 153 vivas, ponerlo en rojo el día uno deja el
gate rojo y entrena a saltárselo — la lección literal de `P1-CI-GATE-PASSABLE`. El
número puede bajar, nunca subir, y por FICHERO además de por total: si uno mejora
y otro empeora, la suma puede quedar igual y el retroceso pasar inadvertido. Un
trinquete que sólo mira la suma no es un trinquete, es un promedio.
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
_CHECK = _FRONTEND / "scripts" / "i18n-check.mjs"
_DETECTOR = _FRONTEND / "scripts" / "i18n-sin-envolver.mjs"
_ALCANCE = _FRONTEND / "scripts" / "i18n-alcance.mjs"
_GRAFO = _FRONTEND / "scripts" / "lib" / "grafo-modulos.mjs"
_BASELINE = _FRONTEND / "scripts" / "i18n-sin-envolver.baseline.json"

_MARKER = "P1-I18N-GATE-CIEGO-SIN-T"


def _hay_node() -> bool:
    return shutil.which("node") is not None


def _saltar_si_falta(*rutas: Path) -> None:
    for r in rutas:
        if not r.exists():
            pytest.skip(f"{r} no existe en este checkout (repos hermanos)")


def _node(codigo: str) -> str:
    """Ejecuta un módulo ESM efímero dentro de `frontend/scripts/`.

    Tiene que vivir ahí para que sus imports relativos resuelvan igual que los del
    script real; un tmpdir necesitaría replicar `node_modules` y mediría otra cosa.
    """
    tmp = _FRONTEND / "scripts" / "_t_p1_gate_ciego.mjs"
    tmp.write_text(codigo, encoding="utf-8")
    try:
        r = subprocess.run(
            ["node", str(tmp)], cwd=str(_FRONTEND),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    finally:
        tmp.unlink(missing_ok=True)
    assert r.returncode == 0, f"el sondeo de node falló:\n{r.stdout}\n{r.stderr}"
    return r.stdout


# ───────────────────────────── el detector ─────────────────────────────

@pytest.mark.skipif(not _hay_node(), reason="node no está en PATH")
def test_detecta_las_posiciones_de_alto_rendimiento() -> None:
    """Las cinco formas en que el copy en español llega al usuario sin pasar por
    `t()`. Cada una salió de un hallazgo real de la auditoría."""
    _saltar_si_falta(_DETECTOR)
    fuente = json.dumps(
        "import { toast } from 'sonner';\n"
        "export function C() {\n"
        "  toast.error('No pudimos guardar los cambios');\n"
        "  return <div title=\"Abre tu nevera\"><p>Tu plan está listo</p></div>;\n"
        "}\n"
        "export const META = { label: 'Añadir alimento' };\n"
        "export function msg() { return 'Revisa los valores del formulario'; }\n"
    )
    salida = _node(
        "import { detectarEnFuente } from './i18n-sin-envolver.mjs';\n"
        f"const h = detectarEnFuente({fuente});\n"
        "console.log(JSON.stringify(h.map(x => x.posicion).sort()));\n"
    )
    posiciones = set(json.loads(salida.strip()))
    for esperada in {"toast", "jsx-text", "attr:title", "prop:label", "return"}:
        assert esperada in posiciones, (
            f"El detector no ve la posición {esperada!r}; encontró {sorted(posiciones)}. "
            f"[{_MARKER}]"
        )


@pytest.mark.skipif(not _hay_node(), reason="node no está en PATH")
def test_no_reporta_lo_que_ya_pasa_por_t() -> None:
    """MUTACIÓN DE CONTROL. Si el detector marcase también lo ya traducido, todos
    los tests de arriba pasarían y no probarían nada.

    El caso con `<strong>` es el que obliga a filtrar por ESTRUCTURA y no por
    texto: un filtro de línea vería el literal español dentro del `t()` y lo
    contaría. Este repo tiene seis precedentes en agosto de guards derrotados por
    mirar texto donde había que mirar el árbol.
    """
    _saltar_si_falta(_DETECTOR)
    fuente = json.dumps(
        "import { useT } from '../i18n';\n"
        "export function C() {\n"
        "  const { t } = useT();\n"
        "  toast.error(t('No pudimos guardar los cambios'));\n"
        "  return <div title={t('Abre tu nevera')}><p>{t('<strong>Aviso:</strong> revisa tu plan')}</p></div>;\n"
        "}\n"
    )
    salida = _node(
        "import { detectarEnFuente } from './i18n-sin-envolver.mjs';\n"
        f"console.log(JSON.stringify(detectarEnFuente({fuente})));\n"
    )
    assert json.loads(salida.strip()) == [], (
        f"El detector reporta cadenas que YA pasan por t(). Salida: {salida} [{_MARKER}]"
    )


@pytest.mark.skipif(not _hay_node(), reason="node no está en PATH")
def test_no_reporta_identificadores_ni_rutas() -> None:
    """Ni `es-DO`, ni una URL, ni una clase CSS, ni una sigla son copy."""
    _saltar_si_falta(_DETECTOR)
    fuente = json.dumps(
        "export const A = { label: 'es-DO' };\n"
        "export const B = { label: 'https://ejemplo.com/de/la/ruta' };\n"
        "export const C = { label: '/dashboard/pantry' };\n"
        "export const D = { label: 'SLA' };\n"
        "export const E = { label: 'PDF' };\n"
        "export function f() { return 'kcal'; }\n"
    )
    salida = _node(
        "import { detectarEnFuente } from './i18n-sin-envolver.mjs';\n"
        f"console.log(JSON.stringify(detectarEnFuente({fuente})));\n"
    )
    assert json.loads(salida.strip()) == [], (
        f"Falsos positivos sobre identificadores. Un guard que grita con lo que no "
        f"es copy se apaga en una semana. Salida: {salida} [{_MARKER}]"
    )


@pytest.mark.skipif(not _hay_node(), reason="node no está en PATH")
def test_la_escotilla_exige_una_razon() -> None:
    """`// [I18N-EXEMPT: <razón>]` silencia; sin razón, no. Mismo trato que
    `P2-LOGGER-EXEMPT` da a los `print()`: una excepción sin motivo escrito es
    indistinguible de un silenciamiento por prisa."""
    _saltar_si_falta(_DETECTOR)
    con_razon = json.dumps(
        "export function f() {\n"
        "  // [I18N-EXEMPT: nombre de alimento, SSOT del motor]\n"
        "  return 'Habichuelas con dulce de la abuela';\n"
        "}\n"
    )
    sin_razon = json.dumps(
        "export function f() {\n"
        "  // [I18N-EXEMPT: ]\n"
        "  return 'Habichuelas con dulce de la abuela';\n"
        "}\n"
    )
    salida = _node(
        "import { detectarEnFuente } from './i18n-sin-envolver.mjs';\n"
        f"console.log(JSON.stringify(detectarEnFuente({con_razon}).length));\n"
        f"console.log(JSON.stringify(detectarEnFuente({sin_razon}).length));\n"
    )
    con, sin = [int(x) for x in salida.strip().splitlines()]
    assert con == 0, f"La escotilla CON razón no silencia. [{_MARKER}]"
    assert sin == 1, (
        f"La escotilla SIN razón silencia igual: entonces no es una escotilla, es "
        f"un `# noqa`. [{_MARKER}]"
    )


# ───────────────────────────── el alcance ─────────────────────────────

@pytest.mark.skipif(not _hay_node(), reason="node no está en PATH")
def test_el_alcance_se_deriva_y_deja_fuera_landing_y_legales() -> None:
    """Landing y legales están FUERA por decisión escrita, y eso se calcula del
    SSOT de rutas + el grafo de imports, no de una lista a mano que envejece."""
    _saltar_si_falta(_ALCANCE, _GRAFO)
    salida = _node(
        "import { clasificarAlcance } from './i18n-alcance.mjs';\n"
        "console.log(JSON.stringify(clasificarAlcance()));\n"
    )
    d = json.loads(salida)
    fuera, dentro = set(d["fuera"]), set(d["dentro"])

    for esperado in ["pages/legal/LegalPages.jsx", "pages/Home.jsx", "pages/Engine.jsx",
                     "components/home/Hero.jsx", "data/benchmark.js"]:
        assert esperado in fuera, (
            f"{esperado} debería estar FUERA de alcance (landing/legales). [{_MARKER}]"
        )
    assert not (fuera & {"pages/Dashboard.jsx", "pages/History.jsx", "pages/Pantry.jsx"}), (
        f"Una página del dashboard salió fuera de alcance. [{_MARKER}]"
    )


@pytest.mark.skipif(not _hay_node(), reason="node no está en PATH")
def test_el_chrome_compartido_queda_DENTRO() -> None:
    """El matiz que hace correcta la regla, y el que la primera versión falló.

    `Layout`, `Header` y `Footer` los pintan el landing Y el dashboard. Cortar el
    grafo en el primer componente del `element` de cada ruta los dejaba fuera —
    medido: «2 páginas resueltas» y Header/Footer/Layout marcados como marketing,
    exactamente al revés—. El corte tiene que ser la PÁGINA, que es la hoja del
    envoltorio. Una regla por directorio (`components/home/**`) tampoco sabría
    distinguirlo.
    """
    _saltar_si_falta(_ALCANCE)
    salida = _node(
        "import { clasificarAlcance } from './i18n-alcance.mjs';\n"
        "console.log(JSON.stringify(clasificarAlcance()));\n"
    )
    dentro = set(json.loads(salida)["dentro"])
    for compartido in ["components/layout/Header.jsx", "components/layout/Footer.jsx",
                       "components/layout/Layout.jsx"]:
        assert compartido in dentro, (
            f"{compartido} es chrome COMPARTIDO: el usuario logueado lo ve, así que "
            f"hay que traducirlo aunque el landing también lo pinte. [{_MARKER}]"
        )


# ───────────────────────────── el trinquete ─────────────────────────────

def test_el_trinquete_existe_y_esta_desglosado_por_fichero() -> None:
    """Sólo el total no basta: si un fichero mejora y otro empeora, la suma puede
    quedar igual y el retroceso pasar inadvertido."""
    _saltar_si_falta(_BASELINE)
    b = json.loads(_BASELINE.read_text(encoding="utf-8"))
    assert isinstance(b.get("total"), int), f"El trinquete no declara `total`. [{_MARKER}]"
    por = b.get("porArchivo") or {}
    # [P2-I18N-ESCANER-RECALL · 2026-08-22] El desglose se exige mientras HAYA deuda. El
    # trinquete llegó a CERO el 2026-08-22, y ahí un desglose vacío es el estado correcto —
    # no la ausencia de la defensa. Condicionarlo al total endurece además el test: ahora
    # también caza un `total > 0` que se quede sin desglose, que antes pasaba si alguien
    # escribía la cifra a mano.
    if b["total"] > 0:
        assert por, f"El trinquete declara {b['total']} pero no tiene desglose. [{_MARKER}]"
    else:
        assert not por, (
            f"El trinquete dice 0 y trae desglose: {por}. [{_MARKER}]"
        )
    assert sum(por.values()) == b["total"], (
        f"El total del trinquete ({b['total']}) no cuadra con la suma del desglose "
        f"({sum(por.values())}). [{_MARKER}]"
    )


@pytest.mark.skipif(not _hay_node(), reason="node no está en PATH")
def test_el_gate_ve_un_fichero_sin_una_sola_llamada_a_t() -> None:
    """El corazón del punto ciego, anclado por CONDUCTA.

    La primera versión de este test buscaba la ausencia del token `continue;` en
    el bucle de ficheros. Estaba mal por dos motivos, y el segundo es el que
    importa: (1) el `continue` sigue ahí y debe seguir, porque la EXTRACCIÓN de
    claves sí que sólo aplica a ficheros con `t()`; (2) un test que busca un
    token no distingue el código correcto del roto — es exactamente el defecto
    que P1-I18N-BOOT-DEFAULT-INDEX0 documenta, donde el guard anclaba la
    expresión defectuosa por su literal.

    Lo que hay que medir es que el DETECTOR alcance un fichero cuyo contenido no
    contiene `t(` por ningún lado: la forma exacta de los ocho utils de etiquetas.
    """
    _saltar_si_falta(_DETECTOR)
    sin_t = json.dumps(
        "export const ESTADOS = {\n"
        "  pendiente: 'Todavía no tiene tu plan del día',\n"
        "  listo: 'Tu plan del día está listo',\n"
        "};\n"
    )
    salida = _node(
        "import { detectarEnFuente } from './i18n-sin-envolver.mjs';\n"
        f"const src = {sin_t};\n"
        "if (/\\bt\\(|\\btn\\(/.test(src)) { console.error('el fixture contiene t(, no sirve'); process.exit(1); }\n"
        "console.log(JSON.stringify(detectarEnFuente(src).length));\n"
    )
    assert int(salida.strip()) >= 1, (
        "Un fichero sin UNA SOLA llamada a `t()` no produce hallazgos: sigue "
        "siendo invisible. Es la forma de los ocho utils de etiquetas "
        f"(planWeeks, shelfLife, authErrors, …). [{_MARKER}]"
    )


@pytest.mark.skipif(not _hay_node(), reason="node no está en PATH")
def test_el_repo_real_no_ha_retrocedido() -> None:
    """El trinquete, ejercido contra el árbol de verdad.

    Si esto falla, alguien añadió copy en español sin envolver. El arreglo es
    envolverlo en `t()` —o marcarlo con `[I18N-EXEMPT: razón]` si de verdad no debe
    traducirse—, nunca regenerar el trinquete para tapar el retroceso.
    """
    if not shutil.which("npm"):
        pytest.skip("npm no está en PATH")
    r = subprocess.run(
        ["npm", "run", "i18n:check", "--silent"],
        cwd=str(_FRONTEND), capture_output=True, text=True,
        encoding="utf-8", errors="replace", shell=True,
    )
    assert "MÁS español sin envolver" not in (r.stdout + r.stderr), (
        f"Retroceso del trinquete de i18n. [{_MARKER}]\n{r.stdout}\n{r.stderr}"
    )
