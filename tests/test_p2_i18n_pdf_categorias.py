"""[P2-I18N-PDF-CATEGORIAS + P2-I18N-PDF-LEYENDA-UD + P3-I18N-PDF-GLOSS-TAUTOLOGICO ·
2026-08-22] Lo que le quedaba en español a la lista de compras del PDF.

El documento que el usuario se lleva al supermercado tenía el nombre del alimento resuelto
--el contrato bilingüe de agosto-- y en español todo lo demás: los rótulos de sección, la
abreviatura de unidad y la leyenda que dice cómo leerlo.

LAS TRES, y cómo se midieron (43 planes vivos, 3.558 ítems):

  1. CATEGORÍAS. `display_category` llega del backend y se imprimía cruda: PROTEÍNAS,
     VEGETALES, DESPENSA, justo debajo de banners que sí se traducen. 8 valores distintos en
     producción.

     La trampa está en QUÉ MÁS es ese valor: `cat` es la clave con la que se agrupan los
     ítems en perecederos/estables, la que ordena las secciones, y la que consultan dos
     comparaciones literales --incluido el heurístico de subcadena `PERISHABLE_PREFIXES`, que
     decide si un alimento va a la sección de 7 días o a la de despensa. Traducirla en el
     dato habría mandado la carne a la sección equivocada, y sólo para quien NO habla
     español. Se traduce AL IMPRIMIR, en el único sitio donde el valor se pinta.

     (`is_perishable` viene como bool en los 3.558 ítems, así que ese heurístico hoy es
     inalcanzable. Da igual: sigue en el código, y una defensa que depende de que un dato
     nunca falte no es una defensa.)

  2. ABREVIATURA DE UNIDAD. `Ud.`/`Uds.` es la TERCERA forma más frecuente de la flota --524
     de 3.558 ítems-- y se quedaba en español en los cuatro idiomas: el barrido de envases
     captura sólo letras, y la abreviatura lleva punto. Es un fallo de MI PROPIO cierre de
     `P1-I18N-CANTIDAD-LISTA`, que dio el vocabulario por completo sin contar la flota. Con
     él salieron otros cuatro huecos del mismo espejo: `diente(s)`, `taza(s)` --que están en
     el `PLURALS` del backend-- y `malla`/`bandeja`, que llegan de `supermarket_products`.

  3. LEYENDA. Decodificaba una abreviatura que el documento no imprime: pt-BR prometía
     «Un.» y fr-FR «U.» donde la línea ponía «Ud.», y el ejemplo decía «2 Cabezas» donde la
     línea ya ponía «2 Heads». Escritos a mano, el rótulo y su leyenda divergen sin que nada
     lo note. Ahora los dos ejemplos se INTERPOLAN desde las mismas llamadas que traducen las
     líneas, así que no pueden discrepar.

  4. (P3) GLOSS TAUTOLÓGICO. «Cilantro (Cilantro)». 23 de 347 filas de `master_ingredients`
     tienen `name_en` igual al nombre español, y **17 de esas 23 sólo se diferencian en una
     tilde** (Salmón/Salmon, Melón/Melon, Kétchup/Ketchup) --justo lo que un `===` no ve.

tooltip-anchor: P2-I18N-PDF-CATEGORIAS
"""
from __future__ import annotations

import ast
import io
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_SHOPPING = _BACKEND / "shopping_calculator.py"
_HELPERS = _ROOT / "frontend" / "src" / "utils" / "shoppingHelpers.js"
_DASHBOARD = _ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"

_MARKER = "P2-I18N-PDF-CATEGORIAS"


def _fuente(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"no existe {p} (¿repo hermano sin clonar?)")
    return io.open(p, encoding="utf-8").read()


def _sin_acentos(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFD", str(s or "").strip().lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _categorias_del_backend() -> set[str]:
    """Todo valor que `_get_display_category` puede devolver.

    Son los valores de `DISPLAY_CATEGORY_MAP` MÁS los `return "…"` literales de la cascada
    NLP, que es por donde salen los ingredientes sin categoría en DB. Mirar sólo el dict
    dejaría fuera «OTROS», que es el default.
    """
    src = _fuente(_SHOPPING)
    arbol = ast.parse(src)

    valores: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Assign):
            nombres = {t.id for t in nodo.targets if isinstance(t, ast.Name)}
            if "DISPLAY_CATEGORY_MAP" in nombres and isinstance(nodo.value, ast.Dict):
                for v in nodo.value.values:
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        valores.add(v.value)
        if isinstance(nodo, ast.FunctionDef) and nodo.name == "_get_display_category":
            for hijo in ast.walk(nodo):
                if (
                    isinstance(hijo, ast.Return)
                    and isinstance(hijo.value, ast.Constant)
                    and isinstance(hijo.value.value, str)
                ):
                    valores.add(hijo.value.value)
    return valores


def _vocabulario_del_glosador() -> set[str]:
    """Las claves normalizadas que `glossShoppingCategory` sabe resolver."""
    src = _fuente(_HELPERS)
    m = re.search(r"const _CATEGORIAS_DE_LISTA = \{(.*?)\n\};", src, re.S)
    assert m, f"no encontré `_CATEGORIAS_DE_LISTA` en {_HELPERS.name} [{_MARKER}]"
    return set(re.findall(r"i18nKey\('([^']+)'\)", m.group(1)))


def test_toda_categoria_que_el_backend_produce_la_sabe_traducir_el_pdf() -> None:
    backend = _categorias_del_backend()
    assert backend, f"no extraje ni una categoría de {_SHOPPING.name} [{_MARKER}]"

    conocidas = {_sin_acentos(c) for c in _vocabulario_del_glosador()}
    huerfanas = sorted(c for c in backend if _sin_acentos(c) not in conocidas)
    assert not huerfanas, (
        f"{len(huerfanas)} categoría(s) que el backend imprime en la lista y que "
        f"`glossShoppingCategory` no sabe traducir. Esa sección del PDF sale en español "
        f"debajo de un banner traducido, y el gate de i18n no lo ve porque el rótulo no es "
        f"una clave en el código: viene del dato.\n"
        + "\n".join(f"  · {c!r}" for c in huerfanas)
        + f"\n[{_MARKER}]"
    )


def test_la_categoria_se_traduce_al_IMPRIMIR_y_no_en_el_dato() -> None:
    """`cat` no es sólo un rótulo: es clave de agrupación y de dos comparaciones."""
    src = _fuente(_DASHBOARD)

    assert "glossShoppingCategory(cat, t)" in src, (
        f"el PDF dejó de glosar la categoría al imprimirla. [{_MARKER}]"
    )
    # …y el valor que se guarda en `consData` sigue siendo el canónico del backend.
    assert re.search(r"let cat = i18nKey\(", src), (
        f"la categoría volvió a traducirse EN EL DATO (`let cat = t(…)`). Ese valor agrupa "
        f"los ítems en perecederos/estables y lo consulta el heurístico de subcadena "
        f"`PERISHABLE_PREFIXES`: traducido, la carne acaba en la sección de despensa del "
        f"documento con el que alguien hace la compra — y sólo para quien no habla español. "
        f"[{_MARKER}]"
    )
    assert re.search(r"cat = item\.display_category \|\| item\.category \|\| i18nKey\(", src), (
        f"el fallback de la categoría volvió a `t(…)`: mismo problema. [{_MARKER}]"
    )
    # El heurístico legacy sigue razonando en español, que es la razón de todo lo anterior.
    assert "'proteína', 'lácteo', 'vegetal', 'fruta', 'urgente'" in src, (
        f"cambió `PERISHABLE_PREFIXES` en el frontend. Si ya no razona sobre el español "
        f"canónico, revisa si sigue teniendo sentido mantener `cat` sin traducir — pero "
        f"NO relajes este test sin mirar el otro lado. [{_MARKER}]"
    )


def test_el_vocabulario_de_cantidades_no_se_deja_lo_que_el_backend_pluraliza() -> None:
    """El espejo de `PLURALS`, que ya se dejó fuera cinco entradas una vez."""
    src_backend = _fuente(_SHOPPING)
    m = re.search(r"PLURALS = \{(.*?)\n    \}", src_backend, re.S)
    assert m, f"no encontré `PLURALS` en {_SHOPPING.name} [{_MARKER}]"
    # Sólo el lado izquierdo: la forma singular es la que el barrido tiene que reconocer.
    singulares = set(re.findall(r"'([^']+)':\s*'[^']+'", m.group(1)))

    src_front = _fuente(_HELPERS)
    # Las claves del vocabulario de envases y las de la abreviatura, juntas.
    # TODAS las entradas de cada línea, no sólo la primera: el vocabulario declara varias
    # por línea (`'cartón': t('cartón'), carton: t('cartón'), cartones: …`) y un `^\s{4}`
    # sólo veía la de la izquierda — el guard habría acusado a `carton` de faltar.
    conocidas = set(re.findall(r"'?([A-Za-zÁÉÍÓÚÑáéíóúñ.]+)'?:\s*t\(", src_front))
    conocidas |= set(re.findall(r"_UNIDADES_NO_TRADUCIBLES = new Set\(\[([^\]]*)\]", src_front)[0]
                     .replace("'", "").replace(" ", "").split(",")) if re.search(
        r"_UNIDADES_NO_TRADUCIBLES = new Set", src_front) else conocidas

    faltan = sorted(s for s in singulares if s not in conocidas)
    assert not faltan, (
        f"{len(faltan)} unidad(es) que el backend pluraliza y que el barrido del PDF no "
        f"reconoce: se imprimen en español en los cuatro idiomas. Es exactamente cómo se "
        f"escapó `Ud.`, que resultó ser la tercera forma más frecuente de la flota (524 de "
        f"3.558 ítems).\n"
        + "\n".join(f"  · {s!r}" for s in faltan)
        + f"\n[{_MARKER}]"
    )


def test_la_leyenda_no_puede_divergir_de_lo_que_el_documento_imprime() -> None:
    src = _fuente(_DASHBOARD)

    assert "const _leyendaUd = t('Ud.');" in src, (
        f"la leyenda volvió a escribir la abreviatura a mano en vez de leerla de la misma "
        f"llamada que traduce las líneas. Así llegó a prometer «Un.» en pt-BR y «U.» en "
        f"fr-FR para algo que el documento imprimía «Ud.» en los cuatro. [{_MARKER}]"
    )
    assert "const _leyendaCabezasRaw = t('cabezas');" in src, (
        f"el ejemplo de la leyenda volvió a llevar «Cabezas» escrito a mano, donde la línea "
        f"de la lista ya imprime la traducción. [{_MARKER}]"
    )
    assert "<strong>{ud}</strong> = unidad" in src and "<em>2 {cabezas}" in src, (
        f"la clave de la leyenda perdió sus placeholders: vuelve a ser texto fijo que puede "
        f"contradecir al documento. [{_MARKER}]"
    )


def test_el_gloss_bilingue_no_se_repite_a_si_mismo() -> None:
    """«Cilantro (Cilantro)» — y sobre todo «Salmon (Salmón)», que un `===` no ve."""
    src = _fuente(_HELPERS)
    assert re.search(r"_sinAcentos\(englishGloss\) === _sinAcentos\(spanishName\)", src), (
        f"el gloss volvió a poder repetirse. La comparación DEBE ir sin diacríticos: 17 de "
        f"las 23 filas tautológicas de `master_ingredients` sólo se diferencian en una tilde "
        f"(Salmón/Salmon, Melón/Melon, Kétchup/Ketchup), y un `===` las deja pasar todas. "
        f"[{_MARKER}]"
    )
