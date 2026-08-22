"""[P2-PDF-WORDMARK-EN-CADENA + P2-PDF-MARCA-MEALFITRD + P3-I18N-PDF-NOMBRE-ARCHIVO ·
2026-08-22] La marca en los dos PDF, y el nombre del fichero que se descarga.

LAS TRES:

  1. `<Wordmark />` ESCRITO DENTRO DE UN TEMPLATE LITERAL. El PDF de receta se construye
     como CADENA --`html2pdf` recibe HTML, no JSX-- y ahí dentro una etiqueta `<Wordmark />`
     no instancia nada: el navegador la lee como un elemento desconocido y VACÍO. La cabecera
     del documento salía sin marca, y el `import` que la traía no lo usaba nadie. Es el modo
     de fallo que ningún test de render puede ver, porque no hay render: hay concatenación.

  2. EL PIE SEGUÍA DICIENDO «MEALFITRD IA». El rebrand a Bioboros es de julio; esta copia se
     quedó atrás y --peor-- la pasada de i18n la propagó a los cuatro catálogos, así que la
     marca vieja quedó traducida cuatro veces. `Wordmark.jsx` existe precisamente porque el
     rebrand se dejó UNA de doce copias y el usuario vio «Mealfit» en la app ya renombrada:
     la respuesta no es corregir el literal, es que no haya literal.

  3. EL NOMBRE DEL FICHERO mezclaba prefijo español fijo con partes traducidas
     (`Lista_de_compras_7_days_…pdf`). En la carpeta de Descargas ese nombre es lo único que
     distingue un documento de otro.

LO QUE ESTE GUARD ANCLA, y por qué cada cosa:

  · Que la marca NO esté escrita a mano en ningún PDF. Un literal «Bioboros» en una cadena
    es la copia número trece.
  · Que el nombre del fichero pase por el saneador. Un copy traducido puede traer `:` o `,`,
    que un nombre de fichero no admite.

tooltip-anchor: P2-PDF-WORDMARK-EN-CADENA
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_FRONT = _ROOT / "frontend" / "src"
_WORDMARK = _FRONT / "components" / "common" / "Wordmark.jsx"
_RECIPES = _FRONT / "pages" / "Recipes.jsx"
_DASHBOARD = _FRONT / "pages" / "Dashboard.jsx"
_NOMBRE = _FRONT / "utils" / "pdfFileName.js"

_MARKER = "P2-PDF-WORDMARK-EN-CADENA"

# Los ficheros que construyen un PDF concatenando HTML.
_PDFS = (_RECIPES, _DASHBOARD)


def _fuente(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"no existe {p} (¿repo hermano sin clonar?)")
    return io.open(p, encoding="utf-8").read()


def test_ningun_pdf_escribe_la_marca_a_mano() -> None:
    """La copia número trece. `Wordmark.jsx` existe para que no la haya."""
    texto_marca = re.search(r"WORDMARK_TEXT = '([^']+)'", _fuente(_WORDMARK))
    assert texto_marca, f"desapareció `WORDMARK_TEXT` del SSOT [{_MARKER}]"
    marca = texto_marca.group(1)

    culpables = []
    for p in _PDFS:
        src = _fuente(p)
        for n, linea in enumerate(src.split("\n"), 1):
            # El import y la interpolación del SSOT son justo lo que se quiere.
            if marca not in linea:
                continue
            if "WORDMARK_TEXT" in linea or "wordmarkHtml" in linea or "Wordmark'" in linea:
                continue
            if linea.lstrip().startswith(("//", "*", "/*")):
                continue
            culpables.append(f"{p.name}:{n}: {linea.strip()[:90]}")

    assert not culpables, (
        f"la marca vuelve a estar escrita a mano en un PDF. Al próximo rebrand esta copia se "
        f"queda atrás como se quedó «MEALFITRD IA» — y si además va dentro de un `t()`, se "
        f"propaga traducida a los cuatro catálogos.\n"
        + "\n".join(f"  · {c}" for c in culpables)
        + f"\n[{_MARKER}]"
    )


def test_la_marca_vieja_no_vive_en_ningun_catalogo() -> None:
    """La pasada de i18n tradujo «MEALFITRD IA» a los cuatro idiomas."""
    import json

    locales = _ROOT / "frontend" / "src" / "i18n" / "locales"
    if not locales.exists():
        pytest.skip("sin catálogos")

    sucios = []
    for p in sorted(locales.glob("*.json")):
        datos = json.loads(io.open(p, encoding="utf-8").read())
        for clave, valor in datos.items():
            if "MEALFITRD" in clave.upper() or "MEALFITRD" in str(valor).upper():
                sucios.append(f"{p.name}: {clave[:60]!r}")

    assert not sucios, (
        f"la marca anterior al rebrand sigue en los catálogos. Traducida cuatro veces, "
        f"además.\n" + "\n".join(f"  · {c}" for c in sucios) + f"\n[{_MARKER}]"
    )


def test_el_componente_no_se_usa_dentro_de_una_cadena() -> None:
    """`<Wordmark />` en un template literal es una etiqueta desconocida y vacía."""
    for p in _PDFS:
        src = _fuente(p)
        assert "<Wordmark" not in src, (
            f"{p.name} vuelve a escribir `<Wordmark …>` en un fichero que construye su PDF "
            f"como CADENA. Un template literal no instancia nada: el documento sale SIN "
            f"marca y ningún test de render lo ve, porque no hay render. Usa "
            f"`${{wordmarkHtml()}}`. [{_MARKER}]"
        )
    assert "wordmarkHtml()" in _fuente(_RECIPES), (
        f"la cabecera del PDF de receta dejó de llevar marca. [{_MARKER}]"
    )


def test_los_dos_pdf_sanean_el_nombre_del_fichero() -> None:
    assert _NOMBRE.exists(), f"desapareció `pdfFileName.js` [{_MARKER}]"
    for p in _PDFS:
        src = _fuente(p)
        m = re.search(r"filename:\s*(.+?),\n", src)
        assert m, f"no encontré el `filename` del PDF en {p.name} [{_MARKER}]"
        assert "pdfFileName(" in m.group(1), (
            f"{p.name} vuelve a componer el nombre del fichero a mano: `{m.group(1)[:70]}`. "
            f"Mezcla un prefijo español fijo con partes traducidas, y además un copy "
            f"traducido puede traer `:` o `,` — que un nombre de fichero no admite. "
            f"[{_MARKER}]"
        )


def test_el_saneador_quita_lo_que_un_nombre_de_fichero_no_admite() -> None:
    """Anclaje del contrato; la conducta se prueba en vitest."""
    src = _fuente(_NOMBRE)
    for prohibido in ("A-Za-z0-9", "u0300-\\u036f"):
        assert prohibido in src, (
            f"`pdfFileName` dejó de sanear (`{prohibido}` desapareció): un copy traducido con "
            f"`:` o con acentos vuelve a llegar entero al nombre del fichero. [{_MARKER}]"
        )
