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


# ── 5. Acabado visual ───────────────────────────────────────────────────────

def test_the_panel_does_not_paint_a_focus_ring_around_itself():
    """[P1-SETTINGS-DIALOG-POLISH · 2026-08-10] EL CERCO BLANCO AL REFRESCAR, que
    reportó el dueño.

    El hook de a11y enfoca el panel al abrirlo para que un lector de pantalla lo
    anuncie. Refrescar es una acción de TECLADO (F5 / Ctrl+R), así que la
    heurística `:focus-visible` de Chrome da por hecho que el usuario navega con
    teclado y le pinta al panel su anillo por defecto: blanco, rodeando la
    ventana entera.

    Quitarlo NO es perder accesibilidad: es un destino de foco programático
    (`tabindex="-1"`) que nadie puede tabular. El anillo significa algo en los
    controles de dentro, y ahí sigue intacto."""
    css = _strip_comments(
        (_SRC / "components" / "dashboard" / "SettingsDialog.module.css").read_text(encoding="utf-8")
    )
    bloque = re.search(r"\.panel:focus[^{]*\{([^}]*)\}", css)
    assert bloque and "outline: none" in bloque.group(1), (
        "P1-SETTINGS-DIALOG: el panel vuelve a pintar su anillo de foco. Como se "
        "enfoca solo al abrir, cualquier apertura con el teclado (refrescar, por "
        "ejemplo) dibuja un cerco blanco alrededor de toda la ventana."
    )


def test_the_dialog_width_does_not_depend_on_what_the_section_contains():
    """[P1-SETTINGS-DIALOG-STABLE · 2026-08-10] «Cada vez que cambio de sección se
    mueve todo», reportado por el dueño. No era una impresión: MEDIDO, el
    contenido de la ventana valía 1040px en Privacidad, 941 en Súper
    Personalización y 698 en Plan & Objetivo, y la navegación saltaba 171px.

    La causa es una herencia sutil. Como página, `.wrapper` lleva `margin: 0
    auto` para centrarse en pantallas anchas. Dentro del diálogo es hijo de un
    flex en columna, y **un margen automático en el eje cruzado anula el
    `stretch`**: el wrapper deja de ocupar su contenedor y pasa a medir su
    CONTENIDO. Cada sección tiene un contenido distinto, así que cada sección
    tenía una ventana distinta.

    Por eso se ancla `margin: 0` explícito y no basta con el `width`: mientras el
    margen automático siga ahí, el ancho declarado convive con un elemento que
    sigue autodimensionándose."""
    css = _strip_comments((_SRC / "pages" / "Settings.module.css").read_text(encoding="utf-8"))
    bloque = re.search(r"\.inDialog\s*\{([^}]*)\}", css)
    assert bloque, "P1-SETTINGS-DIALOG: desapareció el bloque `.inDialog`."
    cuerpo = bloque.group(1)
    assert re.search(r"width:\s*100%", cuerpo), (
        "P1-SETTINGS-DIALOG: el contenido de la ventana dejó de fijar su ancho; "
        "volverá a medir lo que cada sección tenga dentro."
    )
    assert re.search(r"margin:\s*0\s*;", cuerpo), (
        "P1-SETTINGS-DIALOG: volvió el margen automático heredado de la página. "
        "En un flex anula el stretch y el ancho pasa a depender del contenido — "
        "que es exactamente el defecto que este bloque cerró."
    )


def test_the_scrollbar_never_moves_the_content():
    """Unas secciones desbordan y otras no (1358px de alto contra 774). Sin hueco
    reservado, la barra aparece y desaparece al navegar y desplaza el contenido su
    ancho en cada cambio — el mismo defecto que arriba, a menor escala."""
    # [reapuntado 2026-08-14] La reserva se MUDÓ (no se borró) del contenedor del
    # diálogo a la columna de contenido — P1-SETTINGS-SCROLL-HUGS-EDGE puso el
    # scroller donde debía estar, y con él el gutter; el CSS del diálogo lo documenta
    # («viajan»). El test miraba solo el archivo viejo y declaraba que la defensa
    # había desaparecido. Lo que importa es que EL SCROLLER, viva donde viva, reserve
    # el hueco: si no, la barra aparece y desaparece al navegar y desplaza el contenido.
    _hojas = [
        _SRC / "components" / "dashboard" / "SettingsDialog.module.css",
        _SRC / "pages" / "Settings.module.css",
    ]
    _con_gutter = [
        p.name for p in _hojas
        if "scrollbar-gutter: stable" in _strip_comments(p.read_text(encoding="utf-8"))
    ]
    assert _con_gutter, (
        "P1-SETTINGS-DIALOG: el hueco de la barra de scroll dejó de reservarse en "
        f"NINGUNA de las hojas de Configuración ({[p.name for p in _hojas]}). En "
        "Windows la barra ocupa sitio real: el contenido salta ~15px entre una "
        "sección larga y una corta."
    )


def test_the_dialog_flattens_the_page_chrome():
    """Una superficie elevada no necesita elevar a sus hijos. Dentro de la
    ventana, la lista de secciones y el panel de contenido dejan de ser tarjetas
    con borde, sombra y blur — si no, son tres fronteras dibujadas una encima de
    otra, que es exactamente lo que el dueño describió como recargado.

    Se comprueba el PREFIJO DE TEMA además de la clase: sin él, el override se
    aplica en claro y no en oscuro (`html[data-theme="dark"] .sidebarNav` pesa
    más), que fue el primer intento fallido."""
    jsx = _strip_comments(_SETTINGS.read_text(encoding="utf-8"))
    assert "styles.inDialog" in jsx, (
        "P1-SETTINGS-DIALOG: Settings dejó de marcar su raíz en modo ventana; sin "
        "esa clase no hay nada de lo que colgar el aplanado."
    )
    css = _strip_comments((_SRC / "pages" / "Settings.module.css").read_text(encoding="utf-8"))
    assert re.search(r'html\[data-theme="dark"\]\)\s*\.inDialog\s+\.sidebarNav', css), (
        "P1-SETTINGS-DIALOG: el aplanado perdió su variante para tema oscuro. La "
        "regla por tema pesa más y las tarjetas reaparecen SOLO en oscuro — que "
        "es el tema del dashboard."
    )


# ── 4. UNA sola configuración ───────────────────────────────────────────────

def test_there_is_exactly_one_settings_surface():
    """[P1-SETTINGS-ONE-SURFACE · 2026-08-10] Había DOS: el panel del dashboard y
    una página liviana en `/configuracion` (apariencia + cuenta + contraseña +
    borrar cuenta) que era un subconjunto de la otra, mantenida aparte. Dos
    configuraciones divergen con el primer cambio que alguien haga en una sola.

    Se comprueba la ausencia del FICHERO y no del literal de la ruta: la ruta
    sigue viva como redirect, a propósito, porque llevaba meses enlazada y un
    404 sería el final del camino para quien tenga el marcador."""
    vieja = _SRC / "pages" / "AccountSettings.jsx"
    assert not vieja.exists(), (
        "P1-SETTINGS-ONE-SURFACE: volvió a existir una segunda página de "
        "configuración. Lo que necesite vivir ahí va como sección de "
        "`/dashboard/settings`, que es la única superficie."
    )
    app = _strip_comments(_APP.read_text(encoding="utf-8"))
    assert re.search(
        r'path="/configuracion"\s+element=\{<Navigate to="/dashboard/settings" replace', app
    ), (
        "P1-SETTINGS-ONE-SURFACE: `/configuracion` dejó de redirigir. Los "
        "marcadores y pestañas abiertas de meses caerían al catch-all."
    )


def test_an_account_without_a_plan_can_still_reach_its_settings():
    """LA CONDICIÓN QUE HACÍA IMPOSIBLE BORRAR LA PÁGINA VIEJA, y la que se
    romperá si alguien «limpia» la exención por parecer una excepción rara.

    Una cuenta recién creada (OTP/OAuth) sin assessment NO puede entrar a
    `/dashboard/*`: el gate la manda al formulario. Si esta ruta no está exenta,
    esa cuenta se queda sin apariencia, sin cambiar contraseña y sin poder
    BORRAR SU CUENTA — requisito de la App Store, no una comodidad."""
    src = _strip_comments(
        (_SRC / "components" / "layout" / "ProtectedRoute.jsx").read_text(encoding="utf-8")
    )
    exencion = re.search(r"const isOnAccountSettings = location\.pathname === '([^']+)'", src)
    assert exencion, (
        "P1-SETTINGS-ONE-SURFACE: desapareció la exención del gate de assessment "
        "para la configuración."
    )
    assert exencion.group(1) == "/dashboard/settings", (
        "P1-SETTINGS-ONE-SURFACE: la exención apunta a "
        f"`{exencion.group(1)}`, que ya no es la configuración. Una cuenta sin "
        "plan rebotaría al formulario y no podría llegar a borrar su cuenta."
    )
    assert "!isOnAccountSettings" in src, (
        "P1-SETTINGS-ONE-SURFACE: la exención existe pero el gate ya no la "
        "consulta — es una variable inerte con aspecto de defensa."
    )


def test_the_header_entry_points_at_the_surviving_settings():
    """El ⚙ del Header es el ÚNICO acceso de una cuenta sin plan (no puede entrar
    al dashboard por su propio pie). Si apunta a la ruta vieja, el redirect lo
    salva; si apunta a cualquier otra cosa, esa cuenta se queda sin camino."""
    header = _strip_comments(
        (_SRC / "components" / "layout" / "Header.jsx").read_text(encoding="utf-8")
    )
    assert 'to="/configuracion"' not in header, (
        "P1-SETTINGS-ONE-SURFACE: el Header vuelve a enlazar la ruta vieja. "
        "Funcionaría por el redirect, pero enseña como destino algo que ya no "
        "es una página."
    )
    assert header.count('to="/dashboard/settings"') >= 2, (
        "P1-SETTINGS-ONE-SURFACE: falta alguno de los dos accesos del Header "
        "(menú de cuenta de escritorio y menú móvil) a la configuración."
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


# ── 4. El marco: tamaño y canto ─────────────────────────────────────────────

_DIALOG_CSS = _SRC / "components" / "dashboard" / "SettingsDialog.module.css"


def test_the_panel_has_a_real_border_and_not_just_a_shadow():
    """[P1-SETTINGS-PANEL-CRISP · 2026-08-15] El panel nació SIN borde.

    Su único canto era una sombra de 70px de difuminado, y sobre un backdrop
    oscuro eso no delimita nada: panel oscuro contra fondo oscuro, sin línea que
    los separe. El dueño lo describió como que «se veía blando», y tenía razón —
    no era la sombra, era la ausencia de borde.
    """
    css = _DIALOG_CSS.read_text(encoding="utf-8")
    panel = re.search(r"\.panel\s*\{(.*?)\n\}", css, re.DOTALL)
    assert panel, "P1-SETTINGS-PANEL-CRISP: no encuentro la regla `.panel`."
    assert re.search(r"border:\s*1px solid", panel.group(1)), (
        "P1-SETTINGS-PANEL-CRISP: `.panel` perdió su borde. Una sombra difusa da "
        "profundidad pero NO da canto; sin el borde, la ventana vuelve a fundirse "
        "con el fondo en tema oscuro."
    )


def test_the_dark_border_is_stronger_than_the_global_token_on_purpose():
    """El `#334155` NO es drift: es una medición.

    `var(--border)` (#1F2937) contra el panel #0B1120 da 1,28 de contraste — a
    1px, invisible. #334155 da 1,82, que se ve sin parecer un contorno duro.
    El token global está calibrado para superficies DENTRO de la página, donde el
    fondo circundante es el mismo; aquí la ventana flota sobre un backdrop
    atenuado que apaga todo alrededor.

    Este test existe porque el color a pelo INVITA a «limpiarlo» devolviéndolo al
    token, que es precisamente el cambio que reintroduce el problema.
    """
    css = _DIALOG_CSS.read_text(encoding="utf-8")
    assert re.search(
        r'html\[data-theme="dark"\]\)\s*\.panel\s*\{[^}]*border-color:\s*#334155',
        css,
        re.DOTALL | re.IGNORECASE,
    ), (
        "P1-SETTINGS-PANEL-CRISP: el override de borde en tema oscuro desapareció "
        "o volvió a `var(--border)`. Medido: var(--border) da 1,28 de contraste "
        "contra el panel (invisible a 1px) y #334155 da 1,82. Si quieres cambiar "
        "el valor, mídelo antes — no lo devuelvas al token global, que está "
        "calibrado para tarjetas sobre la página y no para una ventana flotante."
    )


def test_the_panel_is_not_full_screen_on_desktop():
    """Que quepa aire alrededor es lo que la hace leerse como ventana.

    Se redujo de 1040×780 a 960×720 por petición del dueño. El guard no fija las
    cifras exactas —eso convertiría cualquier ajuste fino en un fallo— sino el
    TECHO: si alguien vuelve a subirla por encima de lo que un portátil de 13"
    puede enmarcar, la ventana deja de parecer una ventana.
    """
    css = _DIALOG_CSS.read_text(encoding="utf-8")
    panel = re.search(r"\.panel\s*\{(.*?)\n\}", css, re.DOTALL)
    assert panel
    ancho = re.search(r"width:\s*min\((\d+)px", panel.group(1))
    alto = re.search(r"height:\s*min\((\d+)vh,\s*(\d+)px\)", panel.group(1))
    assert ancho and alto, (
        "P1-SETTINGS-PANEL-CRISP: `.panel` ya no declara width/height con `min()`. "
        "Ese `min()` es lo que impide que la ventana desborde en pantallas chicas."
    )
    assert int(ancho.group(1)) <= 1000, (
        f"P1-SETTINGS-PANEL-CRISP: la ventana mide {ancho.group(1)}px de ancho. "
        "Por encima de ~1000px vuelve a ocupar casi todo el escritorio y deja de "
        "leerse como ventana (era 1040 y por eso se bajó)."
    )
    assert int(alto.group(1)) <= 86, (
        f"P1-SETTINGS-PANEL-CRISP: la ventana ocupa {alto.group(1)}vh de alto. "
        "Sin aire arriba y abajo el marco desaparece en pantallas bajas."
    )


def test_no_border_on_mobile_where_the_panel_is_the_screen():
    """En móvil el panel ES la pantalla: ese 1px no separa de nada.

    Dibujaría una línea pegada a los cantos del teléfono y le robaría un píxel al
    contenido por los cuatro lados. Es la misma razón por la que ese bloque ya
    ponía `border-radius: 0`.
    """
    css = _DIALOG_CSS.read_text(encoding="utf-8")
    movil = re.search(r"@media \(max-width: 768px\)\s*\{(.*)", css, re.DOTALL)
    assert movil, "P1-SETTINGS-PANEL-CRISP: falta el bloque móvil."
    panel_movil = re.search(r"\.panel\s*\{(.*?)\n    \}", movil.group(1), re.DOTALL)
    assert panel_movil, "P1-SETTINGS-PANEL-CRISP: falta `.panel` dentro del bloque móvil."
    assert re.search(r"border:\s*0", panel_movil.group(1)), (
        "P1-SETTINGS-PANEL-CRISP: el panel móvil recuperó borde. A pantalla "
        "completa ese 1px no delimita nada: solo roba un píxel de contenido por "
        "cada lado."
    )
