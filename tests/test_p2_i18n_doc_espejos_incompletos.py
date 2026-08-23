"""[P2-I18N-DOC-ESPEJOS-INCOMPLETOS + P2-I18N-DOC-LISTA-BILINGUE-FALSA +
P2-I18N-DOC-SUPERFICIES-SIN-DECLARAR + P2-I18N-CLAUDEMD-ALCANCE-MIENTE · 2026-08-22]
La doc canónica de i18n afirmaba cuatro cosas que no eran verdad.

Una doc canónica equivocada no confunde sólo a las personas: esta misma doc ya provocó que
la primera pasada de la auditoría de idiomas dejara fuera la superficie i18n más cara del
producto, y está anotado en su propio §1. Por eso estas cuatro se cierran con un guard y no
sólo con una corrección.

LAS CUATRO:

  1. §6 decía «**cada fila tiene su propio test**» sobre una tabla de doce espejos. No es
     exacto: `test_p2_i18n_espejos_sin_ancla.py` cubre diez con nueve funciones (una
     parametrizada), y las filas 5 y 6 —las dos copias del `CHECK` de la migración— las
     ancla `test_p1_i18n_dashboard.py`. La sección se cerraba además con «pone rojos los
     NUEVE» dos párrafos después de decir DOCE.

  2. La fila de la lista de compras prometía «el gloss **en el idioma del usuario**». Es
     siempre INGLÉS: `glossShoppingItemName` compone `name_en` + el nombre español para
     cualquier locale que no sea `es-DO`, así que un francés lee «Black beans (Habichuelas
     rojas)». `name_en` es un campo estático del catálogo, no una traducción por idioma.

  3. La tabla de alcance no tenía fila para cinco superficies que hoy SÍ siguen el idioma:
     los dos PDF, las notificaciones push, el help bot, los insights y la autodetección del
     primer arranque. Una superficie que nadie declara es una superficie que nadie revisa —
     el mismo argumento que la propia doc usa para declarar el correo OTP.

  4. `CLAUDE.md` (las DOS copias) seguía diciendo que plan/recetas/coach no se traducen.
     Es la misma mentira que `P2-I18N-DOC-ALCANCE-MIENTE` corrigió en agosto en uno de los
     dos documentos — y CLAUDE.md se carga en CADA turno, así que es el que más caro sale.

QUÉ VIGILA ESTE GUARD: que las CIFRAS de la doc sigan siendo las de la realidad. Una doc que
afirma un número se queda atrás sin que nada lo note; es exactamente lo que pasó aquí.

tooltip-anchor: P2-I18N-DOC-ESPEJOS-INCOMPLETOS
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_DOC = _BACKEND / "docs" / "i18n_dashboard.md"
_ESPEJOS = _BACKEND / "tests" / "test_p2_i18n_espejos_sin_ancla.py"

_MARKER = "P2-I18N-DOC-ESPEJOS-INCOMPLETOS"


def _doc() -> str:
    if not _DOC.exists():
        pytest.skip(f"no existe {_DOC}")
    return io.open(_DOC, encoding="utf-8").read()


def _filas_de_espejos(src: str) -> list[str]:
    """Las filas numeradas de la tabla de §6."""
    seccion = src.split("## 6. Añadir un sexto idioma", 1)
    assert len(seccion) == 2, f"desapareció §6 de {_DOC.name} [{_MARKER}]"
    cuerpo = seccion[1].split("## 7.", 1)[0]
    return re.findall(r"^\|\s*(\d+)\s*\|", cuerpo, re.M)


def test_la_cifra_de_espejos_de_la_doc_es_la_de_la_tabla() -> None:
    src = _doc()
    filas = _filas_de_espejos(src)
    assert filas, f"no encontré la tabla de espejos en §6 [{_MARKER}]"
    n = len(filas)
    assert [int(f) for f in filas] == list(range(1, n + 1)), (
        f"la tabla de espejos está mal numerada: {filas}. [{_MARKER}]"
    )
    # La prosa de §6 declara la cifra en letra («**Son doce**»). Se compara contra las filas
    # REALES: un `\bdoce\b|\b{n}\b` habría seguido pasando con una fila nueva, porque la
    # palabra vieja sigue ahí — que es justo la forma en que esta cifra se quedó atrás.
    palabras = {
        "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
        "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciséis": 16,
        "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
    }
    m = re.search(r"\*\*Son (\w+)\*\*", src)
    assert m, (
        f"§6 dejó de declarar en prosa cuántos espejos son (`**Son doce**`). Esa frase es el "
        f"ancla de la cifra. [{_MARKER}]"
    )
    declarada = palabras.get(m.group(1).lower())
    assert declarada == n, (
        f"§6 dice «Son {m.group(1)}» y la tabla tiene {n} filas. [{_MARKER}]"
    )


def test_la_doc_no_vuelve_a_prometer_un_test_por_fila() -> None:
    src = _doc()
    assert "cada fila tiene **su propio test**" not in src.lower().replace("*", "*"), (
        f"§6 vuelve a prometer un test por fila. Las filas 5 y 6 (las dos copias del "
        f"`CHECK`) las ancla `test_p1_i18n_dashboard.py`, no el fichero de espejos. "
        f"[{_MARKER}]"
    )
    reales = len(re.findall(r"^def test_", io.open(_ESPEJOS, encoding="utf-8").read(), re.M))
    m = re.search(r"cubre\s+(\d+)\s+espejos\s+con\s+(\d+)\s+funciones", src)
    assert m, (
        f"§6 dejó de declarar cuántos espejos cubre `{_ESPEJOS.name}` y con cuántas "
        f"funciones. Esa frase es el ancla: sin ella la cifra vuelve a poder mentir. "
        f"[{_MARKER}]"
    )
    assert int(m.group(2)) == reales, (
        f"§6 dice {m.group(2)} funciones en `{_ESPEJOS.name}` y hay {reales}. [{_MARKER}]"
    )


def test_la_fila_de_la_lista_no_promete_un_gloss_por_idioma() -> None:
    """El gloss es SIEMPRE inglés, y sólo existe en el PDF."""
    src = _doc()
    fila = next((l for l in src.split("\n") if l.startswith("| Lista de compras")), "")
    assert fila, f"desapareció la fila de la lista de compras [{_MARKER}]"
    assert "el gloss en el idioma del usuario Y el nombre" not in fila, (
        f"la fila vuelve a prometer «el gloss en el idioma del usuario». Es siempre INGLÉS: "
        f"`glossShoppingItemName` compone `name_en` + el nombre español para cualquier "
        f"locale que no sea es-DO. [{_MARKER}]"
    )
    assert "INGLÉS" in fila or "inglés" in fila, (
        f"la fila dejó de decir que el gloss es inglés. [{_MARKER}]"
    )


def test_la_tabla_de_alcance_declara_las_superficies_que_siguen_el_idioma() -> None:
    src = _doc()
    alcance = src.split("## 1.", 1)[1].split("## 2.", 1)[0]
    faltan = [
        etiqueta
        for etiqueta, patron in (
            ("PDF", r"^\| PDF"),
            ("push", r"^\| Notificaciones push"),
            ("help bot / insights", r"^\| Help bot"),
            ("autodetección", r"^\| Autodetección"),
            # [P2-I18N-ALCANCE-SIN-SPLASH-NI-MANIFIESTO · 2026-08-23] las dos más persistentes.
            ("splash", r"^\| Splash"),
            ("manifiesto / nombre de la PWA", r"^\| Nombre y atajos de la PWA"),
        )
        if not re.search(patron, alcance, re.M)
    ]
    assert not faltan, (
        f"la tabla de alcance dejó de declarar {faltan}. Son superficies que HOY siguen el "
        f"idioma del usuario, y una superficie que nadie declara es una superficie que nadie "
        f"revisa — el mismo argumento con el que esta doc declara el correo OTP. [{_MARKER}]"
    )


def test_las_dos_copias_de_claude_md_dicen_la_verdad_sobre_el_alcance() -> None:
    """Y son ESPEJO: corregir una sola es cómo se llegó hasta aquí."""
    copias = [_ROOT / "CLAUDE.md", _BACKEND / "CLAUDE.md"]
    presentes = [p for p in copias if p.exists()]
    assert presentes, f"no encontré ninguna copia de CLAUDE.md [{_MARKER}]"

    for p in presentes:
        src = io.open(p, encoding="utf-8").read()
        assert "NO el contenido (plan/recetas/coach los escribe el LLM en español)" not in src, (
            f"{p} vuelve a afirmar que plan/recetas/coach no se traducen. Es falso desde el "
            f"2026-08-19, y este fichero se carga en CADA turno. [{_MARKER}]"
        )
        assert "IDENTIFICADOR no se toca" in src, (
            f"{p} perdió la regla que sustituye a la vieja: no es «lo que escribe el LLM no "
            f"se toca», es «lo que el motor usa como IDENTIFICADOR no se toca». [{_MARKER}]"
        )


# ── P2-I18N-DOC-DISPLAY-CONGELADA ─────────────────────────────────────────────

_DISPLAY_PY = _BACKEND / "plan_display_i18n.py"
_DISPLAY_MD = _BACKEND / "docs" / "plan_display_i18n.md"

# Marcadores que el módulo CITA pero no le pertenecen: son contratos de otros sistemas a
# los que se adhiere. Documentarlos aquí sería duplicar su doc, no completar la propia.
_REFERENCIAS_AJENAS = {
    "P1-COACH-LANGUAGE-NATIVE",   # el idioma del coach, no de la capa de display
    "P1-DIET-CANON-SSOT",         # canonicalización de dieta (constants.py)
    "P3-PREVIEW-MODEL-KNOB",      # convención del repo: modelo por knob, no hardcodeado
}


def test_la_doc_del_display_no_se_queda_congelada() -> None:
    """Citaba 5 marcadores cuando el módulo llevaba 22, e «insights» no salía ni una vez."""
    if not _DISPLAY_PY.exists() or not _DISPLAY_MD.exists():
        pytest.skip("falta el módulo o su doc")
    modulo = set(
        re.findall(r"P[0-9]-[A-Z0-9]+(?:-[A-Z0-9]+)*", io.open(_DISPLAY_PY, encoding="utf-8").read())
    )
    doc = io.open(_DISPLAY_MD, encoding="utf-8").read()
    faltan = sorted(m for m in modulo - _REFERENCIAS_AJENAS if m not in doc)
    assert not faltan, (
        f"{len(faltan)} marcador(es) que `plan_display_i18n.py` declara y su doc SSOT no "
        f"menciona. Así se congeló: cinco citados contra veintidós reales, y la palabra "
        f"«insights» sin aparecer pese a existir `_INSIGHTS_ADDENDUM`. Si el marcador es una "
        f"referencia a OTRO sistema, añádelo a `_REFERENCIAS_AJENAS` con su razón.\n"
        + "\n".join(f"  · {m}" for m in faltan)
        + f"\n[{_MARKER}]"
    )


def test_la_doc_del_display_declara_que_no_hay_evidencia_de_produccion() -> None:
    """Lo único que ningún test verde puede demostrar."""
    if not _DISPLAY_MD.exists():
        pytest.skip("falta la doc")
    doc = io.open(_DISPLAY_MD, encoding="utf-8").read()
    assert "P1-I18N-SIN-EVIDENCIA-PRODUCCION" in doc, (
        f"la doc dejó de declarar que esta capa no ha traducido un plato en producción "
        f"(5 ejecuciones de por vida, 1 plan de 44, 0 comidas, 0 filas de telemetría, 0 de "
        f"19 usuarios con locale ≠ es-DO). Es lo ÚNICO que ningún test verde puede cerrar: "
        f"los tests miden el archivo, no el mundo. [{_MARKER}]"
    )


def test_la_doc_declara_que_el_idioma_no_ofrecido_cae_al_espanol() -> None:
    """[P3-I18N-IDIOMA-NO-OFRECIDO-CAE-A-ESPANOL · 2026-08-23] La conducta existía desde el
    primer día y nadie la había escrito como decisión. El párrafo tiene que seguir en la doc
    Y describir lo que el código hace: `detectBrowserLocale` devuelve `DEFAULT_LOCALE`
    cuando nada casa."""
    src = _doc()
    assert "### El idioma que no ofrecemos cae al español" in src, (
        f"desapareció la decisión sobre el idioma no ofrecido [{_MARKER}]")
    locales_js = _ROOT / "frontend" / "src" / "i18n" / "locales.js"
    if not locales_js.exists():
        pytest.skip("frontend no está en este checkout")
    fn = locales_js.read_text(encoding="utf-8")
    i = fn.find("export function detectBrowserLocale")
    assert i > 0
    cuerpo = fn[i:]
    cierre = cuerpo[: cuerpo.find("\n}\n") + 3]
    assert cierre.rstrip().endswith("return DEFAULT_LOCALE;\n}"), (
        "detectBrowserLocale ya no cae a DEFAULT_LOCALE: si es deliberado, reescribe la "
        f"decisión en §2 de la doc antes [{_MARKER}]")



def test_el_overview_del_sistema_menciona_el_idioma() -> None:
    """[P3-I18N-DOC-OVERVIEW-SIN-IDIOMA · 2026-08-23] `system_overview.md` no mencionaba el
    idioma ni una vez con la app en cinco. La sección tiene que seguir, enlazar a las dos docs
    SSOT y decir la frontera."""
    p = _BACKEND / "docs" / "system_overview.md"
    src = p.read_text(encoding="utf-8")
    assert "## 8. Idioma" in src, f"desapareció la sección de idioma del overview [{_MARKER}]"
    for enlace in ("i18n_dashboard.md", "plan_display_i18n.md"):
        assert enlace in src, f"el overview dejó de enlazar {enlace}"
    assert "IDENTIFICADOR" in src, "la frontera (lo que el motor usa como identificador no se toca) dejó de estar"


def test_la_tabla_api_documenta_i18nkey_y_los_formateadores() -> None:
    """[P3-I18N-DOC-API-SIN-I18NKEY · 2026-08-23] `i18nKey()` existía desde el 21-ago y no
    estaba en la tabla de API; los formateadores nuevos del 23 tampoco."""
    src = _doc()
    api = src.split("### API", 1)[1].split("###", 1)[0]
    for simbolo in ("`i18nKey(es)`", "`formatPercent(", "`formatTemperature(", "`currencySymbol(", "`compareText("):
        assert simbolo in api, f"la tabla de API no documenta {simbolo} [{_MARKER}]"


def test_los_knobs_del_idioma_estan_en_la_doc_del_idioma() -> None:
    """[P3-I18N-DOC-KNOBS-DISPERSOS · 2026-08-23] Un operador que busca «idioma» tiene que
    encontrar los knobs aquí, no sólo en la doc del _display ni en .env.example."""
    src = _doc()
    assert "## 4b. Knobs del sistema de idiomas" in src, f"desapareció la tabla de knobs [{_MARKER}]"
    seccion = src.split("## 4b. Knobs del sistema de idiomas", 1)[1].split("## 5.", 1)[0]
    for knob in ("VITE_AUTO_LOCALE", "MEALFIT_PLAN_DISPLAY_I18N", "MEALFIT_CHAT_TITLE_MODEL"):
        assert knob in seccion, f"la tabla de knobs dejó de listar {knob}"
