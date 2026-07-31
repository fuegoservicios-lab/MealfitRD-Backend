"""[P2-CATALOG-FILTER-SSOT · 2026-07-31] (audit solver+seeder v6 · P2 / F13+F29+F14+F31+F15)
Cuatro defectos del mismo filtro, `constants._get_fast_filtered_catalogs`.

1. PLURALES (F13/F29). Los catch-alls de categoría expanden a términos SINGULARES
   ('almendra', 'nuez', 'pistacho', 'huevo') pero el regex maestro es `\\b(term)\\b` sin sufijo,
   y el catálogo dominicano está en PLURAL: 'Almendras fileteadas', 'Pistachos',
   'Nueces/Almendras', 'Huevos'. El chip 'Frutos Secos' no excluía ninguno.
   El escáner hermano (`_scan_allergen_violations`) SÍ tiene `(?:s|es)?` desde hace tiempo —
   otra vez el fix que aterrizó en una superficie y no en su hermana.

2. GLUTEN DRIFTEADO (F14). El catch-all lista 5 términos; `_ALLERGEN_SYNONYMS['gluten']` tiene
   ~20. Bulgur, Cebada, Galletas de soda y Tortilla integral viven en el catálogo y sobrevivían
   al chip 'Gluten' de un celíaco.

3. STRING VACÍO (F31). `'|'.join(...)` con una restricción vacía produce una ALTERNATIVA VACÍA
   en el patrón maestro (`\\b(mani||...)\\b`), que matchea en cualquier posición ⇒ los CUATRO
   catálogos salen vacíos, en silencio, y el seeder devuelve "".

4. DISLIKES SIN SINÓNIMOS (F15). El chip se compara contra el nombre CRUDO del catálogo:
   'Hongos' nunca excluye 'Champiñones', aunque `VEGGIE_FAT_SYNONYMS` ya sabe que son lo mismo.

La defensa contra la re-aparición de (2) no es copiar la lista buena: es un test de PARIDAD que
usa el escáner canónico como oráculo. Si mañana alguien añade un derivado a `_ALLERGEN_SYNONYMS`
y no al filtro, esto falla.

Anchor de producción: P2-CATALOG-FILTER-SSOT.
"""
import pytest


def _pools(allergies=(), dislikes=(), diet=""):
    from constants import _get_fast_filtered_catalogs
    return _get_fast_filtered_catalogs(tuple(allergies), tuple(dislikes), diet)


def _todos(pools):
    return [x for pool in pools for x in pool]


def _presentes(pools, *nombres):
    blob = [str(x).lower() for x in _todos(pools)]
    return [n for n in nombres if any(n.lower() in b for b in blob)]


# --------------------------------------------------------------- F13 / F29 · plurales

def test_chip_frutos_secos_excluye_los_plurales_del_catalogo():
    """'almendra' debe alcanzar a 'Almendras fileteadas'; 'nuez' a 'Nueces/Almendras'."""
    sobreviven = _presentes(_pools(allergies=["Frutos Secos"]),
                            "Almendras fileteadas", "Pistachos", "Nueces/Almendras",
                            "Mantequilla de almendras")
    assert not sobreviven, f"frutos secos que sobreviven al chip: {sobreviven}"


def test_chip_huevo_excluye_huevos():
    sobreviven = _presentes(_pools(allergies=["Huevo"]), "Huevos")
    assert not sobreviven, f"'Huevos' sobrevive al chip 'Huevo' (singular vs plural): {sobreviven}"


# --------------------------------------------------------------- F14 · gluten drifteado

def test_chip_gluten_excluye_los_cereales_del_catalogo():
    sobreviven = _presentes(_pools(allergies=["Gluten"]),
                            "Bulgur", "Cebada", "Galletas de soda", "Tortilla integral")
    assert not sobreviven, f"gluten que sobrevive en el pool de un celíaco: {sobreviven}"


# ------------------------------------------------- paridad con el escáner canónico (blanket)

@pytest.mark.parametrize("chip", [
    "Frutos Secos", "Gluten", "Lacteos", "Huevo", "Mani", "Soya", "Mariscos", "Pescado",
])
def test_paridad_filtro_vs_escaner_canonico(chip):
    """Oráculo: lo que `_scan_allergen_violations` marcaría como violación NO puede sobrevivir.

    Cierra la clase: el filtro y el escáner son dos copias del mismo conocimiento y ya drifearon
    una vez. Si vuelven a separarse, falla aquí en vez de en el plato de un alérgico.
    """
    from graph_orchestrator import _scan_allergen_violations

    supervivientes = _todos(_pools(allergies=[chip]))
    viola = _scan_allergen_violations(
        {"days": [{"meals": [{"name": "x", "ingredients": [str(i) for i in supervivientes]}]}]},
        [chip],
    )
    assert not viola, (
        f"con el chip {chip!r} el filtro deja pasar {len(viola)} alimentos que el escáner "
        f"canónico marca como violación: {[v[1] for v in viola][:6]}"
    )


# --------------------------------------------------------------- F31 · string vacío

def test_restriccion_vacia_no_vacia_los_catalogos():
    """`'|'.join` con un string vacío crea una alternativa vacía que matchea SIEMPRE."""
    p, c, v, f = _pools(allergies=["maní", ""], dislikes=[""])
    assert p and c and v and f, (
        f"un chip vacío vació los catálogos: P={len(p)} C={len(c)} V={len(v)} F={len(f)}"
    )
    assert not _presentes((p, c, v, f), "Maní"), "la restricción real sigue aplicándose"


def test_solo_restricciones_vacias_equivale_a_sin_restricciones():
    completo = [len(x) for x in _pools()]
    con_vacios = [len(x) for x in _pools(allergies=["", "  "], dislikes=[""])]
    assert con_vacios == completo, f"restricciones vacías filtraron algo: {con_vacios} vs {completo}"


# --------------------------------------------------------------- F15 · dislikes por sinónimo

def test_dislike_por_sinonimo_excluye_el_nombre_del_catalogo():
    """'Hongos' y 'Champiñones' son el mismo alimento y `VEGGIE_FAT_SYNONYMS` ya lo sabe."""
    sobreviven = _presentes(_pools(dislikes=["Hongos"]), "Champiñones")
    assert not sobreviven, f"'Champiñones' sobrevive al dislike 'Hongos': {sobreviven}"


def test_dislike_por_sinonimo_tambien_en_alergias():
    """El mismo vocabulario aplica al canal de alergias (mismo `restrictions`)."""
    sobreviven = _presentes(_pools(allergies=["setas"]), "Champiñones")
    assert not sobreviven, f"'Champiñones' sobrevive a la alergia 'setas': {sobreviven}"


# --------------------------------------------------------------- controles negativos

def test_no_sobrefiltra_por_el_sufijo_plural():
    """El sufijo `(?:s|es)?` no puede reintroducir el falso positivo de subcadena."""
    # 'leche' NO puede llevarse 'Lechosa' (papaya) — el caso word-boundary documentado del repo.
    p, c, v, f = _pools(allergies=["Lacteos"])
    assert _presentes((p, c, v, f), "Lechosa"), "'Lechosa' (papaya) fue excluida por 'leche'"


def test_los_alias_no_se_cruzan_con_el_vocabulario_de_categoria():
    """Los alias resuelven el nombre que el usuario ESCRIBIÓ, nunca la expansión de un catch-all.

    Medido al implementar F15: el alias de 'Harina de maíz precocida' incluye "harina pan" (la
    marca P.A.N.) y el de 'Queso Blanco' incluye "queso de pasta". Cruzarlos con los términos
    derivados del chip 'Gluten' (`pan`, `pasta`) excluía dos alimentos SIN gluten del pool de un
    celíaco. Un sesgo a sobre-filtrar es aceptable en seguridad; inventar restricciones que el
    usuario no declaró, no.
    """
    faltan = [n for n in ("Harina de maíz precocida", "Queso Blanco")
              if not _presentes(_pools(allergies=["Gluten"]), n)]
    assert not faltan, (
        f"el chip 'Gluten' excluyó alimentos sin gluten por colisión de alias: {faltan}"
    )


def test_sin_restricciones_devuelve_el_catalogo_completo():
    from constants import (DOMINICAN_PROTEINS, DOMINICAN_CARBS,
                           DOMINICAN_VEGGIES_FATS, DOMINICAN_FRUITS)
    p, c, v, f = _pools()
    assert (len(p), len(c), len(v), len(f)) == (
        len(DOMINICAN_PROTEINS), len(DOMINICAN_CARBS),
        len(DOMINICAN_VEGGIES_FATS), len(DOMINICAN_FRUITS))


def test_un_dislike_normal_sigue_funcionando():
    """Regresión: el camino común (nombre exacto del catálogo) no puede romperse."""
    assert not _presentes(_pools(dislikes=["Berenjena"]), "Berenjena")
