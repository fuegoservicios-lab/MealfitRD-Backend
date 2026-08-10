"""[P1-STAPLE-SEARCH-RANK · 2026-08-09] El buscador de "Mis básicos" ordenaba
por alfabeto y cortaba a 8 ANTES de ordenar.

REPORTE DEL OWNER (con captura, paso 15 en móvil): escribió «hu» y le salieron
«Clara de huevo», tres habichuelas, y «Huevo» en QUINTO lugar. Preguntó si no
sobraba tener clara de huevo y huevo separados.

DOS COSAS DISTINTAS, Y SOLO UNA ES DEFECTO:

1. Fusionar los alimentos del catálogo sería un ERROR. `master_ingredients`
   alimenta la lista de compras, el descuento de la Nevera y el cálculo de
   macros; clara (~52 kcal, 0 g grasa/100 g) y huevo entero (~155 kcal, ~11 g)
   no son el mismo alimento. Es la misma clase que ya documenta
   P1-PANTRY-NAME-RESOLUTION: `GLOBAL_REVERSE_MAP` colapsa pechuga→pollo a
   propósito, y por eso comerse una pechuga descontaría del muslo.

2. Pero para el GATE de variedad sí colapsan: la tabla de alias mapea tanto
   «Huevo» como «Clara de huevo» al rótulo `huevo`. Elegir ambos gasta dos de
   los ocho cupos para un solo efecto, y el usuario no tenía forma de saberlo.

Este test ancla el contrato del backend: el catálogo anota cada alimento con su
rótulo del gate, calculado desde el SSOT, para que el cliente pueda avisar del
colapso SIN mantener una segunda copia de la tabla de alias (P1-DIET-CANON-SSOT:
eran 3 tablas a mano, driftaron, y la del filtro servía pollo a vegetarianas).

Tooltip-anchor: P1-STAPLE-SEARCH-RANK
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ROUTER = _REPO_ROOT / "backend" / "routers" / "user_data.py"
_QSTAPLE = (_REPO_ROOT / "frontend" / "src" / "components" / "assessment"
            / "questions" / "QStapleFoods.jsx")


def test_el_rotulo_sale_del_ssot_no_de_una_copia():
    """El endpoint debe anotar usando el matcher del gate. Si alguien reimplementa
    la tabla de alias aquí, el catálogo y el motor divergen en silencio."""
    src = _ROUTER.read_text(encoding="utf-8")
    assert "_protein_gate_labels_in_text" in src, (
        "P1-STAPLE-SEARCH-RANK: el catálogo dejó de anotar con el SSOT del gate. "
        "El rótulo DEBE salir de `_protein_gate_labels_in_text` — una tabla de "
        "alias propia en este router driftaría contra el motor."
    )
    assert "staple_gate_label" in src, (
        "P1-STAPLE-SEARCH-RANK: el campo `staple_gate_label` desapareció del "
        "catálogo. Sin él, el cliente no puede avisar de básicos que colapsan."
    )
    # La tabla de alias NO debe vivir aquí.
    assert "_MAIN_PROTEIN_ALIASES = " not in src, (
        "P1-STAPLE-SEARCH-RANK: la tabla de alias se copió al router. Debe haber "
        "UNA sola (graph_orchestrator) — ver P1-DIET-CANON-SSOT."
    )


def test_la_anotacion_es_fail_safe():
    """El paso es OPCIONAL. Un fallo anotando no puede tumbar el catálogo entero:
    sin catálogo, el usuario no puede ni buscar."""
    src = _ROUTER.read_text(encoding="utf-8")
    m = re.search(r"from graph_orchestrator import _protein_gate_labels_in_text(.*?)return \{\"items\": items\}",
                  src, re.DOTALL)
    assert m, "P1-STAPLE-SEARCH-RANK: no encuentro el bloque de anotación del catálogo"
    bloque = m.group(1)
    assert "except Exception" in bloque, (
        "P1-STAPLE-SEARCH-RANK: la anotación no es fail-safe. Si lanza, el usuario "
        "se queda sin catálogo y el paso queda inservible por un campo accesorio."
    )


def test_huevo_y_clara_colapsan_al_mismo_rotulo():
    """El caso exacto del reporte, contra el SSOT vivo. Si alguien separa los
    alias, el aviso del frontend dejaría de dispararse justo en el caso que lo
    motivó."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT / "backend"))
    from graph_orchestrator import _protein_gate_labels_in_text

    assert _protein_gate_labels_in_text("Huevo") == {"huevo"}
    assert _protein_gate_labels_in_text("Clara de huevo") == {"huevo"}, (
        "P1-STAPLE-SEARCH-RANK: «Clara de huevo» dejó de mapear al rótulo `huevo`. "
        "O el alias cambió, o el matcher dejó de ver la palabra dentro del nombre."
    )


def test_lo_que_no_participa_del_gate_no_se_agrupa():
    """Las legumbres y vegetales YA pueden repetirse: el gate ni los mira. Si
    devolvieran rótulo, el aviso saltaría donde no hay nada que avisar."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT / "backend"))
    from graph_orchestrator import _protein_gate_labels_in_text

    for nombre in ("Habichuelas rojas", "Lechuga romana", "Arroz blanco"):
        assert _protein_gate_labels_in_text(nombre) == set(), (
            f"P1-STAPLE-SEARCH-RANK: «{nombre}» devolvió un rótulo del gate. "
            "Agruparía básicos que en realidad no compiten entre sí."
        )


def test_el_orden_va_antes_del_corte():
    """El defecto de la captura. Cortar a 8 y ORDENAR DESPUÉS deja fuera lo que
    el usuario escribió literalmente cuando hay muchas coincidencias."""
    src = _QSTAPLE.read_text(encoding="utf-8")
    assert "rankOf" in src, (
        "P1-STAPLE-SEARCH-RANK: el buscador perdió el orden por relevancia; vuelve "
        "a listar por alfabeto y «Huevo» cae debajo de las habichuelas."
    )
    i_sort = src.index(".sort(")
    i_slice = src.index(".slice(")
    assert i_sort < i_slice, (
        "P1-STAPLE-SEARCH-RANK: el `.slice` va ANTES del `.sort`. Así se corta a 8 "
        "por alfabeto y luego se ordena la sobra — la coincidencia exacta puede no "
        "estar entre esos 8."
    )


def test_el_cliente_no_reimplementa_la_tabla_de_alias():
    """La razón por la que el rótulo viaja en el catálogo. Un mapa de alias en JS
    sería la segunda copia, y las segundas copias driftean."""
    src = _QSTAPLE.read_text(encoding="utf-8")
    assert "staple_gate_label" in src, (
        "P1-STAPLE-SEARCH-RANK: el cliente dejó de leer el rótulo del catálogo."
    )
    for sospechoso in ("'claras'", '"claras"', "'pechuga de pollo'", '"pechuga de pollo"'):
        assert sospechoso not in src, (
            f"P1-STAPLE-SEARCH-RANK: aparece {sospechoso} en el cliente — señal de "
            "que se está reimplementando la tabla de alias del gate en JS. El "
            "rótulo debe venir del backend."
        )
