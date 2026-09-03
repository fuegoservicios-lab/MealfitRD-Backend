"""[P2-I18N-LANG-MIENTE-SI-FALLA-CATALOGO + P2-I18N-MANIFEST-HREF-CONGELADO +
P3-I18N-ENTRADAS-DUPLICADAS + P2-I18N-GUARD-DISPLAY-CIEGO-AL-OPTIONAL-CHAINING · 2026-08-22]
Cuatro cosas que el motor de idiomas dejaba a medias.

  1. `<html lang>` MENTÍA CUANDO EL CATÁLOGO NO BAJABA. `initLocale` aplica el atributo con
     el locale guardado ANTES de pedir su catálogo —tiene que hacerlo, si no la app arranca
     declarando español y parpadea— pero el `catch` de `loadLocale` no lo revertía. Así que
     un chunk que no sube deja el texto en español y el atributo diciendo `fr-FR`: un lector
     de pantalla lee castellano con voz francesa, y `hreflang` y los correctores del
     navegador toman la decisión equivocada. Se reaplica `_locale` —el idioma REALMENTE
     vigente— y no `DEFAULT_LOCALE`: si el usuario venía de un idioma cargado y falla el
     SIGUIENTE, lo que se sigue pintando es el anterior.

  2. EL MANIFIESTO SE QUEDABA EN EL IDIOMA DEL ARRANQUE. `index.html` reescribe el `href`
     con el locale que encuentra guardado y ahí moría: cambiar de idioma con el selector
     dejaba el manifiesto anterior. El manifiesto es lo ÚNICO que el sistema operativo
     recuerda de la app, así que quien instalaba la PWA tras cambiar de idioma se llevaba al
     escritorio el nombre viejo — y desde la web ya no hay forma de corregirlo. Va en
     `_applyLang`, el único punto por el que pasan las tres vías de cambio.

  3. `ENTRADAS` ESTABA DUPLICADA, y las dos copias fallan DISTINTO. Si le falta una entrada
     a `huerfanos.mjs`, el script grita una lista de falsos huérfanos. Si le falta a
     `i18n-alcance.mjs`, el alcance COLAPSA EN SILENCIO: los ficheros que colgaban de esa
     entrada dejan de contarse y el trinquete BAJA. Un número que mejora solo es lo último
     que alguien investiga — y desde `P2-I18N-ESCANER-RECALL` ese trinquete está en cero.

  4. EL GUARD DE `_display` ANCLABA UNA GRAFÍA. Prohibía literalmente `._display[` y no veía
     `?._display?.[`, que es la forma idiomática de este repo. Auditado: hoy NO hay ninguna
     violación de esa forma — lo roto era la defensa, no el código. Y al anclarlo por
     propiedad, el guard empezó a saltar por la PROSA de un comentario que cita
     `_display[locale]`: comentario-vence-guard en su dirección menos obvia. La salida no es
     recortar prosa ajena, es que la regla mire código.

tooltip-anchor: P2-I18N-LANG-MIENTE-SI-FALLA-CATALOGO
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FRONT = _BACKEND.parent / "frontend"

_MARKER = "P2-I18N-LANG-MIENTE-SI-FALLA-CATALOGO"


def _fuente(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip(f"no existe {p} (¿repo hermano sin clonar?)")
    return io.open(p, encoding="utf-8").read()


def test_el_atributo_lang_se_corrige_cuando_el_catalogo_falla() -> None:
    src = _fuente("src/i18n/index.js")
    m = re.search(r"\} catch \(err\) \{(.*?)\n\}", src, re.S)
    assert m, f"no encontré el `catch` de `loadLocale` [{_MARKER}]"
    cuerpo = m.group(1)
    assert "_applyLang(_locale)" in cuerpo, (
        f"el `catch` de `loadLocale` dejó de corregir `<html lang>`. El arranque lo aplica "
        f"con el locale GUARDADO antes de pedir el catálogo, así que sin esta línea un "
        f"chunk que no sube deja el texto en español y el atributo declarando el idioma que "
        f"falló. [{_MARKER}]"
    )
    assert "_applyLang(DEFAULT_LOCALE)" not in cuerpo, (
        f"se corrige a `DEFAULT_LOCALE` en vez de a `_locale`. No es lo mismo: si el usuario "
        f"venía de un idioma YA cargado y falla el siguiente, lo que se sigue pintando es el "
        f"anterior, no el español. [{_MARKER}]"
    )


def test_el_manifiesto_sigue_al_idioma_vivo() -> None:
    src = _fuente("src/i18n/index.js")
    assert "_aplicarManifiesto" in src, (
        f"desapareció la reescritura del manifiesto. `index.html` sólo lo hace en el boot: "
        f"sin esto, cambiar de idioma deja el manifiesto anterior, y es lo único que el "
        f"sistema operativo recuerda de la app. [{_MARKER}]"
    )
    m = re.search(r"function _applyLang\(code\) \{(.*?)\n\}", src, re.S)
    assert m and "_aplicarManifiesto(code)" in m.group(1), (
        f"la reescritura del manifiesto salió de `_applyLang`. Ahí va porque es el ÚNICO "
        f"punto por el que pasan las tres vías de cambio (arranque, selector y "
        f"`syncLocaleFromProfile`) — el mismo argumento que ya justifica la telemetría de "
        f"idioma en esa función. [{_MARKER}]"
    )
    # Y la regla es la del generador: base → manifest.json, resto → manifest.<locale>.json.
    gen = _fuente("scripts/build-manifests-i18n.mjs")
    generados = set(re.findall(r"^\s+'([a-z]{2}-[A-Z]{2})': \{", gen, re.M))
    assert generados, f"no pude leer los locales del generador de manifiestos [{_MARKER}]"
    ssot = _fuente("src/i18n/locales.js")
    soportados = set(re.findall(r"'([a-z]{2}-[A-Z]{2})'", ssot))
    faltan = (soportados - generados) - {"es-DO"}
    assert not faltan, (
        f"hay locales soportados sin manifiesto generado: {sorted(faltan)}. `_applyLang` "
        f"apuntaría el `href` a un fichero que no existe. [{_MARKER}]"
    )


def test_las_entradas_del_grafo_tienen_un_solo_dueno() -> None:
    ssot = _fuente("scripts/entradas.mjs")
    assert "export const ENTRADAS" in ssot, (
        f"desapareció el SSOT de las entradas del grafo. [{_MARKER}]"
    )
    for script in ("scripts/huerfanos.mjs", "scripts/i18n-alcance.mjs"):
        src = _fuente(script)
        assert re.search(r"import \{ ENTRADAS \} from '\./entradas\.mjs'", src), (
            f"{script} dejó de importar el SSOT. Con dos copias, la de `i18n-alcance.mjs` "
            f"falla EN SILENCIO: el alcance colapsa, las cadenas de esa rama dejan de "
            f"contarse y el trinquete BAJA — un número que mejora solo es lo último que "
            f"alguien investiga. [{_MARKER}]"
        )
        assert "const ENTRADAS = [" not in src, (
            f"{script} volvió a declarar su propia copia de `ENTRADAS`. [{_MARKER}]"
        )


def test_el_guard_de_display_ancla_la_propiedad_y_no_la_grafia() -> None:
    src = _fuente("src/__tests__/displayMeal.test.js")
    assert "ACCESO_INDEXADO_A_DISPLAY" in src, (
        f"el guard volvió a anclar la grafía literal `._display[`, que no ve "
        f"`?._display?.[` — la forma idiomática de este repo. [{_MARKER}]"
    )
    m = re.search(r"const ACCESO_INDEXADO_A_DISPLAY = (/.+/);", src)
    assert m, f"no encontré el patrón del guard [{_MARKER}]"
    patron = m.group(1)
    for forma in (r"\?\\\.", r"\\s"):
        assert re.search(forma, patron), (
            f"el patrón `{patron}` dejó de tolerar la forma opcional o el espacio: vuelve a "
            f"ser una grafía. [{_MARKER}]"
        )
    # Y mira CÓDIGO, no prosa: un comentario que cite `_display[locale]` no puede disparar.
    assert "function _readCode" in src, (
        f"el guard volvió a leer el fichero entero, comentarios incluidos. Un `// "
        f"_display[locale]` en la prosa que explica el arreglo lo dispara — "
        f"comentario-vence-guard en su dirección menos obvia. [{_MARKER}]"
    )


# ── El lote de doc: cifras y razones que la doc afirmaba y no eran ─────────────

_DOC = _BACKEND / "docs" / "i18n_dashboard.md"


def _doc() -> str:
    if not _DOC.exists():
        pytest.skip(f"no existe {_DOC}")
    return io.open(_DOC, encoding="utf-8").read()


def test_la_cifra_del_catalogo_es_la_de_la_base() -> None:
    """347 filas, no «206 + 60». Y está en la fila que define la frontera dura."""
    src = _doc()
    assert "206 alimentos + 60 platos criollos" not in src, (
        f"volvió la cifra vieja del catálogo. Medido contra Neon el 2026-08-22: "
        f"`master_ingredients` tiene **347** filas, todas con `name_en`. Está en la fila que "
        f"define la frontera dura del motor —los nombres que `pantry_names_match`, el guard "
        f"de coherencia y el backstop de alergias resuelven por igualdad— que es el peor "
        f"sitio posible para una cifra falsa. [{_MARKER}]"
    )
    assert "**347 filas**" in src, f"la fila del catálogo perdió su cifra [{_MARKER}]"


def test_la_razon_de_excluir_el_landing_es_la_verdadera() -> None:
    src = _doc()
    # La regla es «no se AFIRMA», no «no aparece». La fila cita la frase vieja entre
    # comillas angulares para explicar qué estaba mal —el repo anota en vez de borrar, y esa
    # convención vale justo aquí: sin la cita, nadie entiende contra qué se corrigió—. Un
    # `not in` pelado se acusaba a sí mismo: comentario-vence-guard, ahora dentro de la doc.
    afirmada = [
        i for i in range(len(src))
        if src.startswith("14 páginas estáticas fuera del build de React", i)
        and (i == 0 or src[i - 1] != "«")
    ]
    assert not afirmada, (
        f"volvió la razón falsa. El landing son 19 rutas de `PAPER_SURFACE_ROUTES`, "
        f"componentes React con `lazy()` DENTRO del mismo build de Vite. La exclusión es "
        f"correcta; la razón no lo era — y una razón falsa invita a «corregirla» metiendo el "
        f"landing donde ya está. [{_MARKER}]"
    )
    assert "hreflang" in src, (
        f"la fila del landing perdió la razón REAL (URLs por idioma y `hreflang`: un cambio "
        f"de arquitectura de rutas y SEO, no de copy). [{_MARKER}]"
    )


def test_la_seccion_del_gate_declara_las_cuatro_palancas() -> None:
    """Faltaba la que puede desactivarlo, en la sección que lo llama «la única defensa»."""
    src = _doc()
    seccion = src.split("## 5.", 1)[1].split("## 6.", 1)[0]
    import json as _json
    pkg = _json.loads(io.open(_FRONT / "package.json", encoding="utf-8").read())
    palancas = [k for k in pkg.get("scripts", {}) if k.startswith("i18n:")]
    faltan = [k for k in palancas if k not in seccion]
    assert not faltan, (
        f"§5 no declara {faltan}. `i18n:baseline` es la palanca que RESETEA el trinquete: "
        f"omitirla en la sección que llama al gate «la única defensa» deja fuera justo lo que "
        f"puede desactivarla. [{_MARKER}]"
    )


def test_la_asimetria_de_la_escotilla_esta_escrita() -> None:
    """No es un gap: es un diseño de tres superficies con tres tratos."""
    src = _doc()
    assert "MEALFIT_CI_I18N_STRICT" in src, (
        f"§5 dejó de declarar la escotilla. Existe en el gate local, el deploy la RECHAZA con "
        f"un `throw` y en Actions no existe — tres tratos distintos a propósito, y sin "
        f"escribirlo cualquiera de los tres parece una omisión. [{_MARKER}]"
    )
    # Y el deploy sigue rechazándola: si eso cae, la doc pasa a mentir.
    deploy = _BACKEND.parent / "deploy-mealfit.ps1"
    if deploy.exists():
        assert "MEALFIT_CI_I18N_STRICT" in io.open(deploy, encoding="utf-8").read(), (
            f"el deploy dejó de comprobar la escotilla, y la doc afirma que la rechaza. "
            f"[{_MARKER}]"
        )
