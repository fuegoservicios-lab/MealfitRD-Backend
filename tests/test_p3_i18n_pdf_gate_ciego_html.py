"""[P3-I18N-PDF-GATE-CIEGO-HTML + 1 · 2026-08-23] Los dos últimos gaps accionables.

LO QUE ESTOS TRES ENSEÑAN

1. EL GATE ERA CIEGO, Y SE COMPROBÓ INYECTANDO. El gap decía «las ~500 líneas de copy del
   PDF no las vigila nadie». Medido: ese copy **ya está envuelto** —lo estaba desde
   `P2-I18N-PDF-NOTA-CLINICA` y sus vecinos—, así que la cifra del gap describía un riesgo,
   no una deuda. Lo que sí era cierto es lo que importa: un `<p>Revisa las cantidades…</p>`
   nuevo metido en el generador del PDF **pasaba el gate en VERDE**, verificado por
   inyección antes de escribir la regla.

   Los PDFs se construyen concatenando HTML (`html2pdf` recibe una cadena, no JSX), así que
   su copy vive en template literals — y el escáner miraba atributos, props, tablas, tuplas,
   toasts y expresiones JSX, ninguna de las cuales cubre «texto entre dos etiquetas dentro
   de un backtick».

   La regla costó tres versiones, y las dos primeras fallaron igual: casaban la prosa de un
   comentario de CSS dentro del bloque `<style>`. La forma del copy real es `<tag>texto</`,
   abierto por una etiqueta y CERRADO por una; exigir sólo el `>` de apertura no basta.

2. UN GLOSS INERTE PARA TODOS LOS PLANES VIVOS. El gap estimaba «9 planes». Medido contra
   Neon: de los **49 planes con lista, CERO** traen `item_ref.display_name_en`. O sea que
   hoy, en los cuatro idiomas no-base, la lista de la compra del PDF sale ÍNTEGRA en español
   para cualquier usuario. El código del gloss existía y estaba inerte en producción —
   exactamente como le pasaba a la capa `_display`.

   El respaldo sale del catálogo (`name_en`, 347/347, ya en caché de 24 h en el cliente), así
   que no cuesta ni una petición. Gana siempre el campo embebido si está: el catálogo es
   respaldo, no sustituto.

3. UNA DECISIÓN DE PRODUCTO NO ES MÍA. `P2-UNIDADES-CUERPO-VS-PAIS` queda ABIERTO, y a
   propósito.

   El wizard arranca en pies y libras para los cinco idiomas, y el daño es real: `weightUnit`
   es obligatorio y validado (`P0-FORM-4`), pero un italiano que teclea «70» pensando en
   kilos sobre un campo puesto en `lb` manda 70 lb = **31,7 kg**, justo por encima del mínimo
   de 30, y el plan sale con el BMR de un adulto de 31 kilos.

   Llegué a implementar el default por idioma. Estaba MAL: `P3-DEFAULT-IMPERIAL · 2026-05-20`
   es una decisión escrita con fichero de test propio, y su texto dice que el código anterior
   **ramificaba por `navigator.language`** —en-US → lb, resto → kg— y que eso se quitó a
   propósito porque «usuarios DO/LATAM veían inputs en kg/cm aunque el mercado objetivo
   prefiere imperial». Mi arreglo reintroducía exactamente lo eliminado.

   Yo había leído el ancla equivocada (`P0-FORM-4`, que trata de que la unidad sea
   OBLIGATORIA) y no la que gobierna. Revertido. La doctrina del repo es explícita: un gap
   técnico se cierra implementando; una decisión de producto se cierra con consenso, y leer
   la memoria correspondiente ANTES de invertir esfuerzo.

   Lo que queda para el dueño, dicho una vez: el riesgo de los 31,7 kg existe, `P0-FORM-4`
   ya dejó una heurística que lo registra (peso ≥150 con post-conversión ≤35 kg), y decidir
   si el default debe seguir al idioma para pt-BR/fr-FR/it-IT es suyo, no de un arreglo de
   traducciones.

   ⚠️ Y si algún día se decide que sí: NO se resuelve por PAÍS aunque el gap lo pida así.
   `QCountry` se pregunta DESPUÉS de las medidas (paso 420 contra 343), así que ahí el país
   todavía es el valor sembrado.

tooltip-anchor: P3-I18N-PDF-GATE-CIEGO-HTML
"""
from __future__ import annotations

import io
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_FRONT = _BACKEND.parent / "frontend"
_MARKER = "P3-I18N-PDF-GATE-CIEGO-HTML"


def _leer(p: Path) -> str:
    return io.open(p, encoding="utf-8").read()


def _sin_comentarios(src: str) -> str:
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("//"))


# ───────────────────── 1. el gate ve el copy dentro del HTML ────────────────────────

def test_el_escaner_mira_las_cadenas_de_html():
    src = _leer(_FRONT / "scripts" / "i18n-sin-envolver.mjs")
    assert "'html-en-plantilla'" in src, (
        "el escáner volvió a ser ciego al copy dentro de las cadenas de HTML. Los PDFs se "
        "construyen concatenando HTML, así que su copy vive en template literals y ninguna "
        f"de las otras posiciones lo cubre [{_MARKER}]"
    )
    assert "TemplateLiteral" in src


def test_la_regla_exige_una_etiqueta_de_cierre():
    """Las dos primeras versiones casaban la prosa de un comentario de CSS dentro del bloque
    `<style>`. La forma del copy real es `<tag>texto</`."""
    src = _leer(_FRONT / "scripts" / "i18n-sin-envolver.mjs")
    assert "([^<>{}]{6,})<\\//g" in src or "([^<>{}]{6,})<\\/" in src, (
        "la regla dejó de exigir la etiqueta de CIERRE. Sin ella, un comentario de CSS con "
        f"un `>` y un `<` sueltos entra como copy y el gate se pone rojo por prosa [{_MARKER}]"
    )


# ───────────────────── 2. el gloss de la lista, con respaldo ────────────────────────

def test_el_gloss_tiene_respaldo_del_catalogo():
    src = _sin_comentarios(_leer(_FRONT / "src" / "utils" / "shoppingHelpers.js"))
    assert "export const buildGlossIndex" in src, (
        "desapareció el índice de respaldo. CERO de los 49 planes vivos traen "
        "`display_name_en`, así que sin él la lista del PDF sale íntegra en español en los "
        f"cuatro idiomas no-base [{_MARKER}]"
    )
    assert "glossIndex.get(_sinAcentos(spanishName))" in src


def test_el_campo_embebido_sigue_ganando():
    """El catálogo es respaldo, no sustituto: un plan que traiga su propio gloss puede tener
    un nombre que el catálogo ya no conozca."""
    src = _sin_comentarios(_leer(_FRONT / "src" / "utils" / "shoppingHelpers.js"))
    # Se ancla la EXPRESIÓN que codifica la precedencia, no el orden de aparición: comparar
    # posiciones de `displayNameEn` y `glossIndex.get` lo satisfacía la FIRMA de la función,
    # que nombra el parámetro antes que cualquier lógica. Invertir la precedencia dejaba el
    # guard verde.
    assert "if (!_fuente && glossIndex" in src, (
        "el índice del catálogo dejó de estar condicionado a que el campo embebido esté "
        "vacío: se consulta siempre, o antes. El embebido tiene que ganar — puede llevar un "
        f"nombre que el catálogo ya no conozca [{_MARKER}]"
    )


def test_el_pdf_construye_el_indice_y_lo_pasa():
    dash = _sin_comentarios(_leer(_FRONT / "src" / "pages" / "Dashboard.jsx"))
    # [P1-I18N-GLOSS-CACHE-FRIA · 2026-08-23] El índice se construye del catálogo en memoria
    # o se lee de la caché persistida (`getCachedGlossIndex`) cuando la memoria está fría:
    # cualquiera de las dos formas es «construir el índice».
    assert "buildGlossIndex(" in dash and "getCachedGlossIndex()" in dash, (
        f"el PDF dejó de construir el índice de gloss [{_MARKER}]"
    )
    assert "_dashLocale, _glossIdx)" in dash, (
        "el índice se construye pero no llega al gloss: existe y no sirve"
    )


# ─────────── [P3-I18N-GATE-HTML-CIEGO-A-LA-PROSA-PEGADA-A-INTERPOLACION · 2026-08-23] ───────────
# La regla miraba cada `quasi` por separado y la forma de casi todas las líneas del PDF es
# `<td>${qty} unidades de ${name}</td>`: « unidades de » no tiene etiqueta a ningún lado y
# las etiquetas viven en otros quasis. Ahora se reconstruye el esqueleto del template.
# Medido al cerrarlo: 0 hallazgos en el árbol (el copy del PDF ya pasa por `t()` dentro de
# las interpolaciones) — era un riesgo, y se fija por INYECCIÓN, no por lectura.

import json
import shutil
import subprocess


def _detectar(fuente: str) -> list:
    if shutil.which("node") is None:
        pytest.skip("node no está en PATH")
    tmp = _FRONT / "scripts" / "_t_p3_gate_html_interp.mjs"
    tmp.write_text(
        "import { detectarEnFuente } from './i18n-sin-envolver.mjs';\n"
        f"console.log(JSON.stringify(detectarEnFuente({json.dumps(fuente)})));\n",
        encoding="utf-8",
    )
    try:
        r = subprocess.run(["node", str(tmp)], cwd=str(_FRONT), capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
    finally:
        tmp.unlink(missing_ok=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_la_prosa_pegada_a_una_interpolacion_se_ve():
    fuente = (
        "export function fila(qty, name, total) {\n"
        "  return `<tr><td>${qty} unidades de ${name}</td><td>Total estimado: ${total}</td></tr>`;\n"
        "}\n"
    )
    textos = sorted(h["texto"] for h in _detectar(fuente) if h["posicion"] == "html-en-plantilla")
    assert textos == ["Total estimado:", "unidades de"], (
        f"la prosa pegada a `${{…}}` sigue siendo invisible: {textos} [{_MARKER}]")


def test_el_copy_que_ya_pasa_por_t_no_se_reporta():
    """CONTROL: la forma REAL del PDF de hoy."""
    fuente = (
        "export function fila(qty, name, t) {\n"
        "  return `<tr><td>${qty} ${t('unidades de')} ${name}</td><td>${t('Total estimado:')} ${total}</td></tr>`;\n"
        "}\n"
    )
    assert [h for h in _detectar(fuente) if h["posicion"] == "html-en-plantilla"] == []
