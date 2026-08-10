"""[P1-SETTINGS-DIALOG · 2026-08-10] La configuración se abre como VENTANA sobre
el dashboard, no como página aparte.

La configuración es un desvío —vas, cambias algo, vuelves—, y navegar fuera
costaba el scroll, la vista y volver a montar todo al regresar.

Lo que este fichero protege NO es el marco: es lo que el marco puso en riesgo.

── 1. LA SALIDA ────────────────────────────────────────────────────────────
Settings protege borradores de peso y altura con un aviso de descarte. Como
página tenía UNA sola forma de salir (el botón «Volver») y el guard vivía en su
`onClick`. La ventana añade TRES: la X, el clic en el fondo y la tecla ESC. Una
regla que vive en un manejador solo protege a quien pasa por ese manejador —el
mismo modo de fallo que ya costó un P-fix cuando una validación de un paso del
formulario no cubría a quien se lo saltaba—, y aquí lo que se pierde son los
números que el usuario acaba de escribir.

Por eso hay UNA puerta (`requestExit`) publicada en `exitGateRef`, y el diálogo
no cierra por su cuenta: pide permiso.

── 2. EL HISTORIAL ─────────────────────────────────────────────────────────
`navigateToSection` sincroniza la sección con el hash usando `replaceState`.
Pasaba `null` como state, lo que VACÍA la entrada del historial — y ahí vive
`backgroundLocation`, la marca que convierte la ruta en ventana. Con `null`,
pulsar la primera sección borraba la marca y el diálogo se cerraba solo.

── 3. LAS CAPAS ────────────────────────────────────────────────────────────
Settings abre confirmaciones por dentro. Los listeners del hook de a11y viven en
`document`, así que sin una noción de «quién está encima» una sola tecla ESC
cerraba la confirmación Y la ventana.

MÉTODO — lo que este fichero no puede ver: es un parser. Que las tres vías
CIERREN de verdad se mide en un navegador (harness con router real: abrir,
cambiar de sección, ESC, atrás, clic en el fondo, entrada directa) y el ESC
anidado se mide en `src/__tests__/useModalAccessibility.nested.test.jsx`. Aquí
se ancla que las decisiones sigan escritas donde se tomaron.

Tooltip-anchor: P1-SETTINGS-DIALOG
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "frontend" / "src"
_DIALOG = _SRC / "components" / "dashboard" / "SettingsDialog.jsx"
_SETTINGS = _SRC / "pages" / "Settings.jsx"
_APP = _SRC / "App.jsx"
_LAYOUT = _SRC / "components" / "dashboard" / "DashboardLayout.jsx"
_HOOK = _SRC / "hooks" / "useModalAccessibility.js"
_NESTED_TEST = _SRC / "__tests__" / "useModalAccessibility.nested.test.jsx"


def _strip_comments(src: str) -> str:
    """Bloque y línea. Sin esto, una nota que EXPLIQUE el patrón retirado cuenta
    como el patrón retirado y el guard cae contra su propio arreglo."""
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL))


# ── 1. La salida ────────────────────────────────────────────────────────────

def test_the_exit_gate_is_one_and_checks_for_unsaved_drafts():
    """`requestExit` es la puerta. Si alguien la vacía o le quita la
    comprobación, cerrar la ventana con datos a medias deja de avisar."""
    src = _strip_comments(_SETTINGS.read_text(encoding="utf-8"))
    gate = re.search(r"const requestExit = \(\) => \{(.*?)\n    \};", src, re.DOTALL)
    assert gate, (
        "P1-SETTINGS-DIALOG: no existe `requestExit` en Settings. Es la puerta "
        "que comparten el botón de salida y las tres formas de cerrar la ventana."
    )
    cuerpo = gate.group(1)
    assert "bodyMetricsChanged" in cuerpo and "setShowDiscardConfirm" in cuerpo, (
        "P1-SETTINGS-DIALOG: la puerta de salida ya no comprueba los borradores "
        "de peso/altura. Cerrar con la X, el fondo o ESC los tiraría en silencio."
    )


def test_the_exit_button_goes_through_the_gate_instead_of_checking_by_itself():
    """El botón tenía la comprobación inline. Si vuelve ahí, vuelve a haber dos
    reglas: la del botón y la de las otras tres vías, que es como se separan."""
    src = _strip_comments(_SETTINGS.read_text(encoding="utf-8"))
    boton = re.search(
        r"className=\{styles\.exitSettingsBtn\}(.*?)>", src, re.DOTALL
    )
    assert boton, "P1-SETTINGS-DIALOG: no se encuentra el botón de salida de Settings."
    assert "requestExit" in boton.group(1), (
        "P1-SETTINGS-DIALOG: el botón de salida no usa `requestExit`. Si vuelve a "
        "comprobar los borradores por su cuenta, la ventana y el botón pueden "
        "divergir sin que nada lo note."
    )


def test_settings_publishes_its_gate_for_the_dialog():
    src = _strip_comments(_SETTINGS.read_text(encoding="utf-8"))
    assert "exitGateRef" in src and "exitGateRef.current = requestExit" in src, (
        "P1-SETTINGS-DIALOG: Settings dejó de publicar su puerta. Sin eso el "
        "diálogo no tiene a quién preguntar y cerraría directamente."
    )


def test_the_dialog_asks_before_closing_on_all_three_paths():
    """Las tres vías del diálogo tienen que pasar por la puerta publicada, no
    por el cierre directo."""
    src = _strip_comments(_DIALOG.read_text(encoding="utf-8"))
    assert re.search(r"exitGateRef\.current\(\)", src), (
        "P1-SETTINGS-DIALOG: el diálogo ya no invoca la puerta de Settings."
    )
    # ESC y fondo deben apuntar a `requestClose` (la que consulta la puerta),
    # nunca a `closeDialog` (la que cierra sin preguntar).
    assert re.search(r"onClose:\s*requestClose", src), (
        "P1-SETTINGS-DIALOG: ESC ya no pasa por la puerta de salida."
    )
    assert re.search(r"onClick=\{requestClose\}", src), (
        "P1-SETTINGS-DIALOG: el clic en el fondo ya no pasa por la puerta."
    )


def test_the_backdrop_does_not_duplicate_an_accessible_name():
    """El fondo fue un `<button>` con nombre propio y creó DOS controles
    llamados igual dentro del mismo diálogo (él y el «Cerrar» de la cabecera).
    Es un atajo de ratón redundante —ESC y el botón visible hacen lo mismo—, así
    que no se anuncia."""
    src = _strip_comments(_DIALOG.read_text(encoding="utf-8"))
    backdrop = re.search(r"className=\{styles\.backdrop\}(.*?)/>", src, re.DOTALL)
    assert backdrop, "P1-SETTINGS-DIALOG: no se encuentra el fondo del diálogo."
    assert "aria-hidden" in backdrop.group(1), (
        "P1-SETTINGS-DIALOG: el fondo volvió a anunciarse. Duplica el nombre del "
        "botón «Cerrar» de la cabecera dentro del mismo diálogo."
    )


# ── 2. El historial ─────────────────────────────────────────────────────────

def test_section_navigation_preserves_the_history_state():
    """La marca `backgroundLocation` vive en el state de la entrada actual.
    Reemplazarla con `null` la borra y el diálogo se cierra en el primer clic."""
    src = _strip_comments(_SETTINGS.read_text(encoding="utf-8"))
    llamadas = re.findall(r"history\.replaceState\(([^,]+),", src)
    assert llamadas, "P1-SETTINGS-DIALOG: no hay llamadas a replaceState en Settings."
    for arg in llamadas:
        assert arg.strip() != "null", (
            "P1-SETTINGS-DIALOG: `replaceState` vuelve a vaciar el state del "
            "historial. Ahí vive la marca que sostiene la ventana abierta (y el "
            "`key`/`idx` que React Router usa para su historial)."
        )


def test_the_dialog_route_only_mounts_over_a_background_location():
    """Sin ubicación de fondo NO hay ventana: la ruta cae a la página completa.
    Eso es lo que hace que un enlace directo o un refresco sigan funcionando sin
    mantener un segundo diseño."""
    src = _strip_comments(_APP.read_text(encoding="utf-8"))
    assert "backgroundLocation" in src, (
        "P1-SETTINGS-DIALOG: App ya no lee `backgroundLocation`."
    )
    assert re.search(r"<Routes location=\{backgroundLocation \|\| location\}>", src), (
        "P1-SETTINGS-DIALOG: el árbol de rutas dejó de resolverse contra la "
        "ubicación de fondo — la página de detrás no se pintaría."
    )
    assert re.search(r"\{backgroundLocation && \(", src), (
        "P1-SETTINGS-DIALOG: la ruta del diálogo ya no está condicionada a la "
        "ubicación de fondo; se montaría también en la entrada directa."
    )


def test_the_entry_points_pass_the_background_location():
    """Un punto de entrada que no la manda abre la PÁGINA, no la ventana — y el
    usuario ve dos comportamientos distintos para el mismo botón según por dónde
    entró."""
    layout = _strip_comments(_LAYOUT.read_text(encoding="utf-8"))
    assert layout.count("backgroundLocation") >= 2, (
        "P1-SETTINGS-DIALOG: alguno de los dos accesos del dashboard (menú de "
        "cuenta y menú «más» móvil) dejó de pasar la ubicación de fondo."
    )


# ── 3. Las capas ────────────────────────────────────────────────────────────

def test_the_a11y_hook_can_yield_to_a_layer_above_it():
    hook = _strip_comments(_HOOK.read_text(encoding="utf-8"))
    assert "isTopmost" in hook, (
        "P1-SETTINGS-DIALOG: el hook de a11y perdió `isTopmost`. Sin él, ESC "
        "cierra la confirmación Y la ventana de una sola tecla."
    )
    assert re.search(r"isTopmostRef\.current\(\)\) return;", hook), (
        "P1-SETTINGS-DIALOG: el hook ya no cede el ESC/Tab a la capa de encima."
    )
    assert _NESTED_TEST.exists(), (
        "P1-SETTINGS-DIALOG: falta el test de comportamiento de las capas "
        "anidadas. Este fichero es un parser: no puede pulsar ESC."
    )
