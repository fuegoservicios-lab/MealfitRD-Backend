"""[P3-I18N-METRICA-COMA-CLAVADA + 14 gaps mas · 2026-08-22] La ultima ola de i18n.

Este fichero ancla los quince gaps del cierre del 22-ago. Lleva el nombre del marker que
bumpea `_LAST_KNOWN_PFIX` porque el cross-link (`test_p2_hist_audit_14_marker_test_link.py`)
exige que el slug del marker case con un fichero de test; los otros catorce viven aqui con
su propia seccion.

LO QUE ESTA OLA ENSENA

1. EL SEPARADOR DECIMAL DEL IDIOMA BASE ESTABA MAL. `_etiqueta_metrica`
   (`shopping_calculator.py`) escribe «1,4 kg» con coma a mano, y su comentario lo justifica
   con «Coma decimal: la lista se lee en espanol». Medido: `Intl` en **es-DO** devuelve
   `1.4`, con PUNTO — la Republica Dominicana escribe el decimal como Estados Unidos, no
   como Espana. O sea que la coma clavada no era solo un descuido con los otros cuatro
   idiomas: estaba mal tambien en el idioma en el que se escribio. El comentario confundio
   «espanol» con «Espana».

2. UN CUARTO SITIO DE PRECIO QUE NO APARECIO EN LA BUSQUEDA. Los tres importes del checkout
   llevaban `.toFixed(2)` detras y el grep los encontro; el cuarto — la linea de impuesto,
   `US$0.00` — no lleva ninguno, asi que el patron paso por encima. Volvi a buscar por la
   FORMA que imaginaba en vez de por la propiedad («un importe en pantalla»). El guard de
   abajo cuenta `US$` en el fichero entero, que es la unica pregunta que no se puede
   contestar a medias.

3. COMENTARIO-VENCE-GUARD, otra vez y otra vez mio. Al arreglar `hour12: true` deje un
   comentario que lo CITA, asi que un `'hour12' not in src` se pondria rojo por mi propia
   prosa. No se censura el comentario: se exige que las apariciones esten TODAS en lineas de
   comentario, que es la afirmacion real.

4. EL RATCHET DEL GLOSARIO NO SE MOVIO, Y ESO ERA EL DATO. Ampliar el glosario a las dos
   formas del plural daba, medido antes de escribir nada, exactamente los mismos 16 desvios.
   Stock oculto: cero. Se arregla el MECANISMO — la proxima traduccion que pierda el termino
   en el singular ya no pasa — no una deuda que no existia.

tooltip-anchor: P3-I18N-METRICA-COMA-CLAVADA
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONT = _ROOT / "frontend"
_BACK = _ROOT / "backend"


def _leer(p: Path) -> str:
    return io.open(p, encoding="utf-8").read()


def _sin_lineas_de_comentario(src: str) -> str:
    """Quita las lineas cuyo primer token es `//`.

    Se filtran las LINEAS enteras que empiezan por `//`, no todo lo que siga a un `//` en
    cualquier posicion: un `//` dentro de una cadena (una URL, sin ir mas lejos) haria que
    el filtro se comiera codigo real, y en una asercion de tipo «esto ya no aparece»
    comerse codigo es un falso VERDE.
    """
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("//"))


# ───────────────────────── 1. P3-I18N-METRICA-COMA-CLAVADA ─────────────────────────

def test_el_separador_decimal_se_pregunta_a_intl_y_no_se_tabula():
    src = _leer(_FRONT / "src" / "utils" / "shoppingHelpers.js")
    assert "const _separadorDecimal = () =>" in src, (
        "desaparecio `_separadorDecimal`: la cantidad de la lista vuelve a salir con la "
        "coma decimal clavada del backend en los cinco idiomas"
    )
    assert "formatNumber(1.1)" in src, (
        "el separador dejo de preguntarse a `Intl`. Si alguien lo sustituyo por una tabla "
        "`locale -> separador`, es la enesima tabla a mano de este repo y drifeara igual "
        "que las tres que P1-DIET-CANON-SSOT tuvo que fusionar"
    )
    assert "_separadorDecimal()" in src.split("export const glossShoppingQty")[1], (
        "`_separadorDecimal` existe pero `glossShoppingQty` ya no lo llama: el helper vive "
        "y el glosado no lo usa, que es peor que no tenerlo"
    )


def test_el_glosado_toca_el_separador_pero_jamas_la_agrupacion():
    """Reagrupar convertiria «1250 g» en «1.250 g», y en una lista de la compra eso se
    puede leer como mil doscientos cincuenta veces mas. El gap dice «separador»."""
    cuerpo = _leer(_FRONT / "src" / "utils" / "shoppingHelpers.js")
    cuerpo = cuerpo.split("export const glossShoppingQty")[1].split("export const")[0]
    cuerpo = _sin_lineas_de_comentario(cuerpo)
    assert "formatNumber(" not in cuerpo.replace("formatNumber(1.1)", ""), (
        "`glossShoppingQty` esta pasando la cantidad entera por `formatNumber`. Eso "
        "reagrupa los millares ademas de cambiar el separador"
    )


def test_el_numero_crudo_sigue_siendo_el_que_se_parsea():
    """La red doble: `parseMarketQty` lee `display_qty` CRUDO en el camino de `/restock`.
    Glosar al renderizar no puede desviar ni un gramo de lo que se descuenta de la nevera —
    pero solo mientras nadie meta el glosado en ese camino."""
    dash = _leer(_FRONT / "src" / "pages" / "Dashboard.jsx")
    assert "parseMarketQty(ing.display_qty)" in dash
    assert "parseMarketQty(glossShoppingQty" not in dash, (
        "alguien metio el texto GLOSADO en el parser de cantidades. Ese numero decide "
        "cuanto se descuenta de la Nevera"
    )


# ───────────────────────── 2. P3-I18N-CHECKOUT-MONEDA-CLAVADA ──────────────────────

def test_el_checkout_no_escribe_ni_un_solo_importe_a_mano():
    """Cuenta `US$` en el fichero ENTERO.

    Tres de los cuatro importes llevaban `.toFixed(2)` y el grep los encontro; el cuarto
    (la linea de impuesto, `US$0.00`) no lleva ninguno y se escapo. La pregunta correcta no
    es «cuantos `toFixed` hay» sino «queda algun simbolo de moneda escrito a mano»."""
    src = _leer(_FRONT / "src" / "components" / "dashboard" / "PaymentModal.jsx")
    assert "US$" not in src, (
        "vuelve a haber un importe con el simbolo escrito a mano en el checkout: la unica "
        "pantalla donde el usuario decide gastar dinero saldra con el formato anglosajon "
        "en los cinco idiomas"
    )
    assert "formatCurrency" in src


def test_formatcurrency_existe_y_degrada_sin_dejar_el_precio_en_blanco():
    src = _leer(_FRONT / "src" / "i18n" / "index.js")
    assert "export function formatCurrency(" in src
    cuerpo = src.split("export function formatCurrency(")[1].split("export function")[0]
    assert "catch" in cuerpo, (
        "`formatCurrency` perdio su fail-soft: un `Intl` que lance dejaria el precio en "
        "blanco justo en el modal de pago"
    )


def test_la_moneda_que_viaja_a_paypal_no_se_toca():
    """Esto traduce como se ESCRIBE un importe, no en que se cobra. El precio es el mismo
    para todo el mundo."""
    src = _leer(_FRONT / "src" / "components" / "dashboard" / "PaymentModal.jsx")
    assert "currency_code: 'USD'" in src or 'currency_code: "USD"' in src, (
        "el `currency_code` que viaja a PayPal cambio. Formatear un importe en el idioma "
        "del usuario NO es cobrarle en su moneda"
    )


# ───────────────────────── 3. P2-I18N-PUSH-ERROR-COPY ──────────────────────────────

def test_los_throws_de_push_llevan_codigo_y_no_copy_en_espanol():
    src = _leer(_FRONT / "src" / "utils" / "pushNotifications.js")
    codigo = _sin_lineas_de_comentario(src)
    assert "No hay Service Worker registrado." not in codigo, (
        "el `throw` volvio a llevar la frase en espanol. Como llega al usuario por "
        "`err.message || t('…')` y `err.message` SIEMPRE es verdad, el `|| t(…)` de al "
        "lado es una rama muerta: la traduccion existe y no se pinta jamas"
    )
    assert '_e.code = "sw_missing"' in codigo
    assert '_e.code = "server_error"' in codigo
    # `err?.code` aparece DOS veces en ese return (`code:` y el ternario de `error:`), asi
    # que buscar la subcadena la satisface el otro sitio: la mutacion que anula el campo
    # `code` dejaba este assert VERDE. Se ancla el campo, que es lo que el llamador lee.
    assert "code: err?.code" in codigo, (
        "el catch se volvio a tragar el codigo: sin el, el copy no se puede resolver por "
        "locale y volvemos al mensaje en espanol"
    )


@pytest.mark.parametrize("locale", ["en-US", "pt-BR", "fr-FR", "it-IT"])
def test_los_dos_codigos_nuevos_tienen_copy_en_los_cuatro_catalogos(locale):
    """Test de DATO. Si la traduccion falta, `t()` cae al espanol en silencio y el sintoma
    es exactamente el que este gap cierra."""
    cat = json.loads(_leer(_FRONT / "src" / "i18n" / "locales" / f"{locale}.json"))
    for clave in (
        "No hay Service Worker registrado en este navegador.",
        "El servidor rechazó la suscripción (error {codigo}).",
    ):
        assert cat.get(clave), f"{locale} no traduce {clave!r}"


def test_settings_mapea_los_dos_codigos():
    src = _leer(_FRONT / "src" / "pages" / "Settings.jsx")
    assert "sw_missing:" in src and "server_error:" in src, (
        "Settings dejo de mapear los codigos nuevos: el usuario vera el codigo crudo"
    )


# ───────────────────────── 4. P3-I18N-CONTADOR-SIN-SEPARADOR ───────────────────────

def test_el_contador_de_macros_formatea_las_dos_cifras():
    src = _leer(_FRONT / "src" / "components" / "dashboard" / "TrackingProgress.jsx")
    assert "formatNumber(consumed)" in src and "formatNumber(goal)" in src, (
        "el contador volvio a pintar la cifra cruda. Las DOS importan: formatear solo la "
        "meta deja «2100 / 2,100», que es peor que no formatear ninguna"
    )


# ───────────────────────── 5. P3-I18N-ORDEN-NOMBRES-ES-CLAVADO ─────────────────────

def test_el_unico_ordenador_de_texto_sigue_al_idioma_activo():
    src = _leer(_FRONT / "src" / "components" / "history" / "HistoryDesktopPanel.jsx")
    codigo = _sin_lineas_de_comentario(src)
    assert 'localeCompare(b.name, "es")' not in codigo, (
        "el orden alfabetico volvio a clavar «es»: ordena la n y los digrafos con las "
        "reglas del castellano para un frances o un italiano"
    )
    assert "localeCompare(b.name, getLocale())" in codigo


# ───────────────────────── 6. P3-I18N-HORA-COACH-12H ───────────────────────────────

def test_la_hora_del_coach_la_decide_el_idioma_y_no_un_hour12_forzado():
    src = _leer(_FRONT / "src" / "pages" / "AgentPage.jsx")
    codigo = _sin_lineas_de_comentario(src)
    assert "hour12" not in codigo, (
        "`hour12` volvio al codigo: ANULA al formateador que si lee el locale y fuerza "
        "AM/PM a los cinco idiomas, cuando el frances, el italiano y el espanol usan 24 h"
    )
    assert "timeStyle: 'short'" in codigo
    # Comentario-vence-guard, en la direccion «mi prosa dispara el guard»: el comentario
    # que explica el arreglo CITA `hour12: true`. Se exige que las apariciones esten todas
    # en comentarios, no que la palabra desaparezca del fichero.
    assert "hour12" in src, (
        "desaparecio tambien el comentario que explica por que no se usa `hour12`. Sin el, "
        "el proximo que quiera «arreglar» el formato de hora lo reanade"
    )


# ───────────────────────── 7-10. el grupo `_display` del backend ───────────────────

def test_el_docstring_ya_no_afirma_que_el_modulo_nunca_lee_display():
    """Decirlo mal importa: un lector que crea que la regla es «nadie lo lee» borrara la
    comprobacion de `_ya_traducido_*` creyendo que restaura una invariante."""
    src = _leer(_BACK / "plan_display_i18n.py")
    assert "NUNCA lee ese campo de vuelta" not in src, (
        "volvio la afirmacion falsa. Desde P2-DISPLAY-REDESPACHO-SIN-FILTRO el modulo SI "
        "lee su propio `_display`, y con razon: para no re-pagar una traduccion que ya esta"
    )
    assert "jamás influye en el dato canónico" in src, (
        "se perdio la frontera REAL, que es la que sigue intacta"
    )


def test_la_poda_de_idiomas_cubre_tambien_el_display_de_nivel_plan():
    src = _leer(_BACK / "plan_display_i18n.py")
    codigo = "\n".join(
        l for l in src.splitlines() if not l.lstrip().startswith("#")
    )
    crudos = re.findall(r'pd\["_display"\]\s*=\s*plan_disp\b', codigo)
    assert not crudos, (
        f"{len(crudos)} escritura(s) del `_display` de nivel plan sin podar: acumula los "
        "cinco idiomas y nada lo evacua nunca"
    )
    assert codigo.count('_podar_locales(plan_disp') >= 2


@pytest.mark.parametrize("razon", ["dedupe_locked", "circuit_breaker_open"])
def test_los_dos_abandonos_mudos_dejan_fila(razon):
    """El breaker abierto significa que el proveedor esta caido, y era justo el estado que
    no dejaba rastro en `pipeline_metrics`."""
    src = _leer(_BACK / "plan_display_i18n.py")
    assert f'"reason": "{razon}"' in src, (
        f"el camino `{razon}` volvio a salir mudo: en la telemetria es indistinguible de "
        "un plan que nunca se pidio"
    )


def test_los_cinco_knobs_del_display_se_declaran_en_el_import():
    """CONDUCTA, no forma: se importa el modulo y se pregunta al registry vivo.

    `knobs._env_*` registra al ser LLAMADO, y los cinco viven dentro de funciones que solo
    corren cuando hay algo que traducir. Esta capa se ha ejecutado CINCO veces en toda su
    historia (medido 2026-08-22), asi que en la practica eran invisibles siempre para el
    operador que consulta que puede tocar sin redeploy.
    """
    import plan_display_i18n  # noqa: F401 — importarlo ES el sujeto del test
    from graph_orchestrator import get_knobs_registry_snapshot

    snap = get_knobs_registry_snapshot()
    vivos = set(snap) if isinstance(snap, dict) else {k.get("name") for k in snap}
    esperados = {
        "MEALFIT_PLAN_DISPLAY_I18N",
        "MEALFIT_PLAN_DISPLAY_I18N_MODEL",
        "MEALFIT_PLAN_DISPLAY_I18N_TIMEOUT_S",
        "MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS",
        "MEALFIT_PLAN_DISPLAY_I18N_MAX_OUTPUT_TOKENS",
    }
    faltan = sorted(esperados - vivos)
    assert not faltan, (
        f"{faltan} no estan en el registry tras importar el modulo. Un knob que el "
        "operador no ve no es un knob: es una constante que alguien puede cambiar por "
        "accidente"
    )


# ───────────────────────── 11. P3-I18N-GLOSARIO-ALCANCE-RECORTADO ──────────────────

def test_el_glosario_mira_las_dos_formas_del_plural():
    src = _leer(_FRONT / "scripts" / "i18n-check.mjs")
    assert "v && v.one" in src, (
        "el glosario volvio a mirar solo `other`. Una traduccion puede usar el termino "
        "pactado en «2 platos» y otro distinto en «1 plato», y el desvio no existiria"
    )
    assert "formas.some(" in src, (
        "se exige el termino en la CONCATENACION en vez de en cada forma. Unir los dos "
        "textos y buscar una vez deja pasar justo el caso que esto busca"
    )


# ───────────────────────── 12. P3-I18N-CATALOGVERSION-ANCLA-LA-FORMA ───────────────

def test_existe_el_guard_de_que_el_contador_SUBE():
    ruta = _FRONT / "src" / "__tests__" / "I18nProvider.p2_ready_load_bearing.test.jsx"
    src = _leer(ruta)
    assert "el contador SUBE cuando entra un catalogo" in src, (
        "se borro el guard de conducta de `catalogVersion`. El que queda mide el TIPO, y "
        "un contador congelado en 0 es un numero perfectamente valido que no repinta nada"
    )


# ───────────────────────── 13. P3-I18N-MIGRACION-ESPEJO-CRLF ───────────────────────

def test_el_espejo_de_la_migracion_compara_contenido_no_bytes():
    src = _leer(_BACK / "tests" / "test_p1_i18n_dashboard.py")
    assert "_MIGRATION_ROOT.read_bytes() == _MIGRATION_BACKEND.read_bytes()" not in src, (
        "volvio la comparacion de bytes crudos entre dos repos hermanos. Git reescribe los "
        "finales de linea al pasar por el indice, asi que ese assert se pone rojo sin "
        "defecto — y un guard que hace eso ensena a ignorarlo"
    )
    assert 'replace("\\r\\n", "\\n")' in src


# ───────────────────────── 14. P3-I18N-AUTOLOCALE-INDESCUBRIBLE ────────────────────

def test_el_knob_de_autodeteccion_esta_documentado_donde_se_busca():
    src = _leer(_FRONT / ".env.example")
    assert "VITE_AUTO_LOCALE" in src, (
        "el knob volvio a vivir SOLO en un comentario del codigo. Un knob que no esta en "
        "el `.env.example` no es un knob: es una constante que alguien cambia sin saber "
        "que existia"
    )


# ───────────────────────── 15. P3-I18N-PROFILE-EXENCION-SIN-FILA ───────────────────

@pytest.mark.parametrize("md", ["CLAUDE.md", "backend/CLAUDE.md"])
def test_patch_profile_tiene_fila_en_la_tabla_de_exenciones(md):
    """Es el UNICO exento que puede acabar gastando en el LLM: cambiar `locale` dispara el
    enriquecimiento de `_display`. La exencion es correcta — al cap, un 402 aqui dejaria al
    usuario sin poder volver a su idioma — pero sin la fila nadie lo sabia."""
    src = _leer(_ROOT / md)
    assert "`PATCH /api/profile`" in src, f"{md}: falta la fila de `PATCH /api/profile`"
    fila = [l for l in src.splitlines() if "`PATCH /api/profile`" in l][0]
    assert "llm_usage_events" in fila, (
        f"{md}: la fila no dice donde va el gasto. Que NO vaya a `api_usage` es justo lo "
        "que hace que un cambio de idioma no queme credito de planes"
    )


def test_las_dos_copias_de_claudemd_siguen_siendo_la_misma():
    a = _leer(_ROOT / "CLAUDE.md")
    b = _leer(_ROOT / "backend" / "CLAUDE.md")
    assert a == b, (
        "las dos copias de CLAUDE.md divergieron. Anadir una fila a una sola es como "
        "empieza el drift"
    )


# ───────────── 6b. P3-I18N-HORA-DEL-COACH-SIGUE-EN-12H (la mitad del servidor) ─────────────

def test_el_bloque_temporal_del_prompt_da_la_hora_en_24h():
    """El cierre del cliente (arriba) dejaba al SERVIDOR diciéndole al modelo «02:30 PM»;
    el modelo copia la forma que ve. Se mide la CONDUCTA de la función, no el fichero."""
    import re
    from prompts.chat_agent import build_temporal_context
    out = build_temporal_context(local_date="2026-07-26", tz_offset=240)
    hora = re.search(r"La hora local es (\d{2}:\d{2})", out)
    assert hora, f"el bloque temporal ya no dice la hora como HH:MM: {out!r}"
    assert not re.search(r"\b[AP]M\b", out), f"la hora del prompt sigue en 12 h: {out!r}"
    assert 0 <= int(hora.group(1)[:2]) <= 23
