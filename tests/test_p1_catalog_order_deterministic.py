"""[P1-CATALOG-ORDER-DETERMINISTIC · 2026-08-19] La resolución del catálogo deja de
depender del orden físico del heap.

Incidente: el fill masivo del gloss del catálogo (347 UPDATEs) reescribió el heap de
`master_ingredients` y — con `SELECT *` sin ORDER BY + sort estable por longitud +
first-hit en CONTAINS — flipeó 4 resoluciones REALES del corpus DO («Pollo horneado al
limón con arroz amarillo» pasó de Pechuga de pollo a Arroz blanco; «Repollo morado»
dejó de resolver a su propia fila). El guard C3 de F2 lo cazó en el gate del deploy.

Tres piezas, las tres ancladas aquí:
  1. `ORDER BY name` en el SELECT del catálogo (orden estable de entrada).
  2. Sort del índice por `(-len, alias)` (empates de longitud ya no heredan orden de filas).
  3. `_best_contains_match`: best-match por (longitud desc, POSICIÓN en el string,
     alfabético) en vez de first-hit — en empates de longitud la identidad del plato
     encabeza («Pollo horneado ... con arroz» es pollo), y 'pernil' sigue ganándole a
     'cerdo' en «cerdo para pernil» (longitud primero: el retarget documentado de F2
     se preserva).

El contrato de RESULTADOS vive en el baseline C3 regenerado
(scripts/data/do_corpus_retarget_baseline_2026_08_18.json, 2026-08-19) — este archivo
ancla el MECANISMO.
"""
import os
import re

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


def test_select_del_catalogo_lleva_order_by():
    src = _read("shopping_calculator.py")
    assert "SELECT * FROM master_ingredients ORDER BY name" in src, (
        "el SELECT del catálogo debe llevar ORDER BY name — sin él, el ganador de cada "
        "colisión del índice es el orden físico del heap, que cualquier UPDATE masivo re-baraja"
    )


def test_sort_del_indice_desempata_por_alias():
    src = _read("shopping_calculator.py")
    assert "all_aliases.sort(key=lambda x: (-len(x[0]), x[0]))" in src, (
        "el sort del índice debe desempatar por alias (alfabético) tras la longitud — el sort "
        "estable por longitud sola hereda el orden de filas en los empates"
    )


def test_best_contains_match_existe_con_clave_len_posicion_alias():
    src = _read("shopping_calculator.py")
    assert "tooltip-anchor: P1-CATALOG-ORDER-DETERMINISTIC" in src
    m = re.search(r"key = \(-len\(alias_stripped\), m\.start\(\), alias_stripped, master_name\)", src)
    assert m, (
        "la clave del best-match debe ser (longitud desc, posición, alias) — invertir "
        "longitud/posición rompe el retarget documentado de pernil (F2); quitar la posición "
        "devuelve los empates al azar"
    )


try:
    import sys
    sys.path.insert(0, BACKEND)
    from shopping_calculator import _best_contains_match
    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover
    _best_contains_match = None
    _IMPORT_ERR = _e


@pytest.mark.skipif(_best_contains_match is None, reason="shopping_calculator no importable")
def test_best_match_semantica_unit():
    """Unit puro (sin DB): longitud primero, posición después, alfabético al final."""
    def pats(*pairs):
        return [(re.compile(r"\b" + re.escape(a) + r"\b"), m, a) for a, m in pairs]

    # longitud gana aunque llegue después en el string (pernil-class)
    assert _best_contains_match(
        "cerdo para pernil", pats(("cerdo", "Cerdo"), ("pernil", "Pernil"))
    ) == "Pernil"
    # empate de longitud ⇒ posición (identidad del plato encabeza)
    assert _best_contains_match(
        "pollo horneado con arroz", pats(("arroz", "Arroz blanco"), ("pollo", "Pechuga de pollo"))
    ) == "Pechuga de pollo"
    # empate de longitud ⇒ posición: 'zeta' encabeza aunque 'beta' sea alfabéticamente menor
    assert _best_contains_match(
        "zeta beta", pats(("zeta", "FilaZ"), ("beta", "FilaB"))
    ) == "FilaZ"
    # alias DUPLICADO en dos filas (empate total hasta el alias) ⇒ master alfabético,
    # jamás el orden de entrada de la lista
    assert _best_contains_match(
        "mariscos frescos", pats(("mariscos", "Pulpo"), ("mariscos", "Calamar"))
    ) == "Calamar"
    assert _best_contains_match(
        "mariscos frescos", pats(("mariscos", "Calamar"), ("mariscos", "Pulpo"))
    ) == "Calamar"
    # sin match ⇒ None
    assert _best_contains_match("nada aqui", pats(("xyz", "X"))) is None
