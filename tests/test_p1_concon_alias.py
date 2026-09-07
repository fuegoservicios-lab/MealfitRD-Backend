# -*- coding: utf-8 -*-
"""[P1-CONCON-ALIAS · 2026-09-07] El concón entra al catálogo como lo que es: arroz.

El concón —la costra tostada del fondo del caldero, el plato dominicano por excelencia— no existía
en el catálogo de ninguna forma. Medido antes de tocar nada:

    normalize_name('arroz blanco cocido') = 'Arroz blanco'   <- «cocido» está en _NORMALIZE_STOPS
    normalize_name('concón')              = 'Concón'          <- se devuelve a sí mismo: NO resuelve

Un nombre que no resuelve es un nombre invisible: no casa en la Nevera (`pantry_names_match`), el
guard de coherencia no lo ve, y el backstop de alergias no puede razonar sobre él.

## Por qué ALIAS y no fila propia — la decisión que este test ancla

Es la regla que dio el dueño: **«si tienes arroz, prácticamente tienes concón»**, la misma forma
que ya rige `Clara de huevo` (`price_source='derived_huevo'`: sin SKU propio, su precio sale del
cartón porque para tener una clara compras el huevo). El concón no se compra: sale del arroz que
ya está en la lista.

Una fila propia exigiría macros de «cocido y tostado» que nadie ha medido. Inventarlas sería peor
que no tener el alimento: un número fabricado con formato de dato no se distingue de uno medido.

Y el alias no introduce ninguna inconsistencia NUEVA — le da exactamente el trato que «arroz
blanco cocido» ya recibe, porque ambos colapsan a la misma fila.

## Lo que se midió y se decidió NO implementar

`Arroz blanco` son 358,6 kcal/100 g: arroz CRUDO. Con `cocido` como stopword, una línea en gramos
de arroz ya cocido contaría de más. Parecía un defecto grande. Medido sobre 95 planes vivos y
11.148 líneas de ingrediente: **55 líneas (0,5 %)** dan gramos de un alimento que absorbe agua
marcado como cocido, **45 de ellas son soya texturizada**, y sus cantidades («25 g», «55 g») son
peso SECO — el modelo escribe «cocido» como preparación mientras la cantidad ya es la del grano
seco, así que colapsar a la fila cruda es CORRECTO ahí. `arroz` aparece una vez, y escrito
«arroz blanco crudo».

> Deducir un defecto del código no es encontrarlo. El tercer «defecto real que no importa» de la
> misma jornada, junto al queso cottage (0 de 25) y la vía síncrona (0 de 72).
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

ALIAS_CONCON = ("concón", "concon", "arroz tostado", "raspa de arroz")
FILA = "Arroz blanco"


def _sa(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _catalogo():
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    from shopping_calculator import get_master_ingredients
    return get_master_ingredients() or []


def test_el_concon_resuelve_a_arroz_blanco():
    """Lo que faltaba: que el nombre exista para el motor."""
    filas = _catalogo()
    fila = next((r for r in filas if r.get("name") == FILA), None)
    assert fila, f"{FILA!r} desapareció del catálogo"
    presentes = {_sa(a) for a in (fila.get("aliases") or [])}
    for a in ALIAS_CONCON:
        assert _sa(a) in presentes, f"falta el alias {a!r} en {FILA!r}"


def test_ningun_alias_del_concon_es_palabra_suelta():
    """`raspa` a secas casaría con «raspar» en un paso de receta.

    La colisión por subcadena lleva 17 apariciones documentadas en este proyecto (sal⊂salsa,
    res⊂fresco, pollo⊂repollo, «Clara de huevo»⊂huevo esta misma jornada). Las frases completas
    son la defensa, no una preferencia de estilo.
    """
    for a in ALIAS_CONCON:
        if " " not in a:
            assert _sa(a).startswith("concon"), (
                f"{a!r} es una palabra suelta que no es «concón»: exige frase completa")


def test_los_alias_no_secuestran_otra_fila():
    """Barrido real contra el catálogo vivo: ningún alias es subcadena de otro alimento."""
    filas = _catalogo()
    choques = []
    for a in ALIAS_CONCON:
        na = _sa(a)
        for r in filas:
            if r.get("name") == FILA:
                continue
            for cand in [r.get("name") or ""] + list(r.get("aliases") or []):
                nc = _sa(cand)
                if nc and (na in nc or nc in na):
                    choques.append((a, r.get("name"), cand))
    assert not choques, f"colisiones: {choques}"


def test_no_se_creo_una_fila_propia_para_el_concon():
    """La decisión, anclada: sin macros medidas, no hay fila.

    Si algún día alguien mide el concón de verdad (cocido y tostado tiene menos agua que el arroz
    hervido y más que el grano seco), esta prueba es la que hay que cambiar A PROPÓSITO — con el
    dato delante, no para hacer sitio a una estimación.
    """
    filas = _catalogo()
    propias = [r.get("name") for r in filas if "concon" in _sa(r.get("name") or "")]
    assert not propias, (
        f"apareció una fila propia para el concón ({propias}): si trae macros medidas, cambia este "
        f"test y su docstring; si son estimadas, bórrala")


def test_el_script_de_siembra_comprueba_colisiones_antes_de_escribir():
    """El guard vive en el script, no solo aquí: se corre ANTES del INSERT, no después."""
    src = (_BACKEND / "scripts" / "seed_concon_2026_09_07.py").read_text(encoding="utf-8")
    assert "colisión — NO se escribe nada" in src
    assert "return 3" in src, "el script debe salir con código propio si detecta una colisión"
