"""[P1-FRUIT-PAD-FILTERED · 2026-07-30] (audit solver+seeder v5 · P1-6)

El seeder filtra el catálogo de frutas por alergias/dislikes/dieta al principio
(`_get_fast_filtered_catalogs`)… y después lo deshace en el padding.

Cuando el pool tiene menos de 4 frutas, el relleno recorre `_DEFAULT_DR_FRUITS`
—una tupla FIJA de 10— con un único chequeo:

    for _df in _DEFAULT_DR_FRUITS:
        if _df.lower() not in existing:      # ← "¿ya está?" y NADA más
            unique_fruits.append(_df)

Nunca consulta `filtered_fruits`. Y no es un caso raro: `num_fruits_to_pick = min(2, ...)`,
así que el while SIEMPRE añade ≥2 defaults sin filtrar, en CADA plan con frutas.

Peor aún, la fruta rechazada es la MÁS probable de entrar: el filtro la sacó del pool, así
que no está en `existing`, así que el padding la elige. Un usuario con dislike de mango recibe
mango; con `temporary_dislike` ("no quiero mango esta semana") igual. Con una ALERGIA el
allergen-guard downstream lo bloquea, pero al costo de rechazo → retry → tokens.

`P1-FRUIT-SEEDER-GATE-CONTRACT` (2026-07-26) ya curó esta lista por el lado del GATE (sacó
Limón y Naranja porque el gate las excluye) — cero mención de los filtros clínicos. Media
curación: la lista se alineó con lo que el gate acepta, no con lo que el USUARIO acepta.
"""
from __future__ import annotations

import ai_helpers as ah


_ASSIGN = "Frutas asignadas al día"


def _fruit_block(prompt: str) -> str:
    """SOLO el segmento donde el seeder ASIGNA frutas ("Frutas asignadas al día: X y Y").

    Aserción acotada a propósito. El prompt trae ejemplos NEGATIVOS hardcodeados que nombran
    frutas ("Una merienda de solo fruta … (mango con casabe …)"): filtrar por la palabra
    'fruta' arrastra esas líneas y el test daría rojo por una frase que le dice al LLM lo que
    NO debe hacer — un falso positivo, no el bug. Segundo caso hoy de la misma clase (el
    hermano de la nevera cayó igual): sobre un prompt largo, `not in` necesita anclarse al
    fragmento que produce la decisión, no al vocabulario."""
    out = []
    for ln in prompt.splitlines():
        i = ln.find(_ASSIGN)
        if i >= 0:
            out.append(ln[i:])
    return "\n".join(out)


def test_el_vehiculo_del_test_ve_las_asignaciones():
    """Sanity del filtro: si el formato del prompt cambia y `_fruit_block` deja de encontrar
    las asignaciones, TODOS los tests de abajo pasarían en vacío."""
    blk = _fruit_block(_prompt())
    assert blk.count(_ASSIGN) >= 3, f"esperadas ≥3 asignaciones (una por opción), got:\n{blk}"


def _prompt(**extra) -> str:
    fd = {"allergies": [], "dislikes": []}
    fd.update(extra)
    return ah.get_deterministic_variety_prompt("", fd, user_id=None)


def test_dislike_no_vuelve_por_el_padding():
    """Mango es el 2º default: casi siempre entra por el relleno, y justamente porque el filtro
    lo sacó del pool (no está en `existing`)."""
    blk = _fruit_block(_prompt(dislikes=["mango"])).lower()
    assert "mango" not in blk, f"el padding re-sembró la fruta rechazada:\n{blk}"


def test_alergia_no_vuelve_por_el_padding():
    blk = _fruit_block(_prompt(allergies=["fresa"])).lower()
    assert "fresa" not in blk, f"el padding re-sembró un alérgeno:\n{blk}"


def test_varios_dislikes_a_la_vez():
    """Cuatro de los 10 defaults rechazados: el relleno debe saltárselos todos y seguir
    llenando con los permitidos, sin repetir ni quedarse corto."""
    dis = ["mango", "lechosa", "piña", "guineo"]
    blk = _fruit_block(_prompt(dislikes=dis)).lower()
    for d in dis:
        assert d not in blk, f"'{d}' rechazado y presente en la asignación:\n{blk}"


def test_sigue_habiendo_frutas_asignadas():
    """Regresión inversa: el filtro no puede dejar el bloque de frutas VACÍO — eso cambiaría
    el bug por otro (día sin fruta). Con dislikes normales el seeder sigue asignando."""
    blk = _fruit_block(_prompt(dislikes=["mango"]))
    assert blk.strip(), "el bloque de frutas quedó vacío"
    assert len(blk) > 40, blk


def test_sin_restricciones_el_padding_sigue_igual():
    """Sin dislikes/alergias el comportamiento no cambia (el fix es un filtro, no un rediseño)."""
    blk = _fruit_block(_prompt())
    assert blk.strip()


def test_el_padding_consulta_el_pool_filtrado():
    """Estructural: el while del relleno debe consultar `filtered_fruits`. Anclado al bloque
    real (`_DEFAULT_DR_FRUITS` → fin del while), no a una ventana de bytes fija."""
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "ai_helpers.py"), encoding="utf-8") as f:
        src = f.read()
    i = src.index("_DEFAULT_DR_FRUITS = (")
    # [P2-SEEDER-DAYS-COUNT · 2026-08-03] El corte era `unique_proteins[:3]`; el 3 literal pasó a
    # `_dc` (días del chunk). Solo cambia el ancla que DELIMITA el bloque — lo que este test
    # protege (que el padding consulte `filtered_fruits`) se comprueba igual.
    blk = src[i:src.index("chosen_proteins = unique_proteins[:_dc]", i)]
    assert "filtered_fruits" in blk, (
        "el padding de frutas no consulta el pool filtrado por alergias/dislikes/dieta")
