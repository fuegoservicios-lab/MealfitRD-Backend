"""[P1-DIET-CANON-SSOT · 2026-07-31] (audit solver+seeder v6 · P1 / F11) El femenino legacy
pasaba el boundary del router pero no matcheaba NINGUNA rama del filtro de catálogo.

`routers/plans.py` acepta 10 valores de `dietType` (`_DIET_TYPE_ENUM`). El filtro
`constants._get_fast_filtered_catalogs` listaba a mano los masculinos:

    if diet in ["vegano", "vegan"]: ...
    elif diet in ["vegetariano", "vegetarian"]: ...
    elif diet in ["pescetariano", "pescatarian"]: ...

`'vegetariana'` y `'vegana'` — ACEPTADAS explícitamente por el boundary como data legacy real de
`health_profile.dietType` — caían al else implícito ⇒ pool = catálogo COMPLETO ⇒ el seeder ofrecía
Pollo y Res como bases obligatorias del esqueleto a una vegetariana. En el path LLM el
`DIET_HARD_GUARD` lo rechazaba (retry-storm); en el path degradado se entregaba (ese era el P0
hermano, [P0-DEGRADED-SAFETY-SCAN]).

El comentario del propio router anota "(catalog reconoce)" al lado de los masculinos — o sea que
quien lo escribió sabía exactamente cuáles reconocía el filtro, y no notó que los femeninos no.

La cura NO es añadir dos strings a esas listas: eso es crecimiento por incidente, y el repo ya
tiene el registro de a dónde lleva. Había TRES tablas de canonicalización de dieta y ya habían
drifteado entre sí (`ovo-lacto-vegetariano` estaba en una y no en la otra). Este fix crea el SSOT
en `constants.py` (el módulo más bajo, sin ciclo de imports) y hace que las tres lo consulten.

Anchor de producción: P1-DIET-CANON-SSOT.
"""
import pytest

# Vocabulario del boundary (routers/plans.py). Se importa de allí a propósito: si alguien añade
# un valor legacy nuevo al enum sin cablearlo al filtro, estos tests fallan.
from routers.plans import _DIET_TYPE_ENUM, _DIET_TYPE_CANONICAL, _DIET_TYPE_LEGACY_ACCEPTED

_CARNE = ("Pollo", "Res", "Cerdo", "Pechuga de pollo")


def _pools(diet):
    from constants import _get_fast_filtered_catalogs
    return _get_fast_filtered_catalogs((), (), diet)


def _tiene_carne(pool):
    from constants import strip_accents
    out = []
    for item in pool:
        n = strip_accents(str(item).lower())
        if any(strip_accents(c.lower()) in n for c in _CARNE):
            out.append(item)
    return out


# --------------------------------------------------------------- el bug concreto

@pytest.mark.parametrize("diet", ["vegetariana", "vegana"])
def test_femenino_legacy_excluye_la_carne(diet):
    """El bug tal cual: forma aceptada por el boundary, invisible para el filtro."""
    proteinas, _c, _v, _f = _pools(diet)
    carne = _tiene_carne(proteinas)
    assert not carne, (
        f"dietType={diet!r} (aceptado por el boundary como legacy) deja carne en el pool "
        f"proteico: {carne[:6]}"
    )


@pytest.mark.parametrize("femenino,masculino", [("vegetariana", "vegetariano"), ("vegana", "vegano")])
def test_femenino_y_masculino_dan_el_mismo_pool(femenino, masculino):
    """Invariante fuerte: no "más seguro" ni "menos", IGUAL que la forma que ya funcionaba."""
    fem = [set(p) for p in _pools(femenino)]
    mas = [set(p) for p in _pools(masculino)]
    for i, nombre in enumerate(("proteínas", "carbos", "vegetales", "frutas")):
        assert fem[i] == mas[i], (
            f"pool de {nombre} difiere entre {femenino!r} y {masculino!r}. "
            f"Solo femenino: {sorted(fem[i] - mas[i])[:5]} · solo masculino: {sorted(mas[i] - fem[i])[:5]}"
        )


# ------------------------------------------------- blanket sobre el enum del boundary

def test_todo_valor_del_enum_restrictivo_filtra_de_verdad():
    """Blanket: si el boundary lo acepta y canonicaliza a una dieta restrictiva, DEBE filtrar.

    Cierra la clase entera: añadir un valor legacy nuevo al router sin cablear el filtro
    (exactamente lo que pasó con los femeninos) falla aquí.
    """
    from constants import canonicalize_diet_type

    completo = set(_pools("")[0])
    fallos = []
    for valor in sorted(_DIET_TYPE_ENUM):
        canon = canonicalize_diet_type(valor)
        if canon == "balanced":
            continue
        proteinas = set(_pools(valor)[0])
        if proteinas == completo:
            fallos.append(f"{valor!r} (canon={canon}) devuelve el catálogo COMPLETO")
        elif _tiene_carne(proteinas):
            fallos.append(f"{valor!r} (canon={canon}) deja carne: {_tiene_carne(proteinas)[:3]}")
    assert not fallos, "valores del enum que no filtran: " + " · ".join(fallos)


def test_omnivora_y_balanced_no_se_sobrefiltran():
    """Control negativo: el fix no puede volverse un sobre-filtro. `omnivora` es balanced."""
    completo = set(_pools("")[0])
    for valor in ("balanced", "omnivora", "Omnívora"):
        assert set(_pools(valor)[0]) == completo, (
            f"dietType={valor!r} es omnívora: no debe filtrar nada del pool proteico"
        )
    assert _tiene_carne(completo), "sanity: el catálogo completo SÍ tiene carne (si no, el test no mide nada)"


# --------------------------------------------------------------- el SSOT

def test_las_tres_tablas_de_dieta_coinciden():
    """Había tres canonicalizadores y ya habían drifteado. Deben dar el MISMO veredicto."""
    from constants import canonicalize_diet_type
    from condition_rules import _canon_diet
    from graph_orchestrator import _canonicalize_diet_type

    vocabulario = sorted(_DIET_TYPE_ENUM) + [
        "ovolactovegetariano", "ovo-lacto-vegetariano", "pescetariana", "pescatariana",
        "pescatariano", "pescetarian", "Vegetariana", "  VEGANA  ", "", None, "keto",
    ]
    discrepancias = []
    for v in vocabulario:
        veredictos = {
            "constants": canonicalize_diet_type(v),
            "condition_rules": _canon_diet(v),
            "graph_orchestrator": _canonicalize_diet_type(v),
        }
        if len(set(veredictos.values())) > 1:
            discrepancias.append(f"{v!r} → {veredictos}")
    assert not discrepancias, "las tablas de dieta drifearon: " + " · ".join(discrepancias)


def test_canonicalize_cubre_el_vocabulario_conocido():
    """Los mapeos que el resto del sistema da por hechos."""
    from constants import canonicalize_diet_type as c

    assert c("vegana") == c("vegano") == c("vegan") == "vegan"
    assert c("vegetariana") == c("vegetariano") == c("vegetarian") == "vegetarian"
    assert c("ovolactovegetariano") == c("ovo-lacto-vegetariano") == "vegetarian"
    assert c("pescetariana") == c("pescetariano") == c("pescatarian") == "pescatarian"
    assert c("omnivora") == c("Omnívora") == c("balanced") == c("") == c(None) == "balanced"
    assert c("  VeGeTaRiAnA  ") == "vegetarian", "debe tolerar espacios y mayúsculas"
    assert c(["vegana"]) == "balanced", "un tipo no-string no puede crashear: degrada a balanced"


def test_el_filtro_usa_el_canonicalizador_y_no_una_lista_a_mano():
    """tooltip-anchor de producción: P1-DIET-CANON-SSOT"""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "constants.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    m = re.search(r"def _get_fast_filtered_catalogs\(.*?\n(?=\ndef |\Z)", src, re.S)
    assert m, "no se encontró `_get_fast_filtered_catalogs`"
    cuerpo = m.group(0)
    assert "canonicalize_diet_type" in cuerpo, (
        "el filtro debe canonicalizar la dieta, no comparar contra listas literales de variantes"
    )
    # El `not in` va contra las líneas que DECIDEN, no contra la prosa: los comentarios citan el
    # código viejo a propósito, y anclar al vocabulario haría fallar al test por su propia
    # documentación (pasó en la primera corrida de este mismo test).
    codigo = "\n".join(l for l in cuerpo.splitlines() if not l.lstrip().startswith("#"))
    assert 'diet in ["vegano", "vegan"]' not in codigo, (
        "quedó la lista literal de variantes en código vivo: es la que se olvidó de los femeninos"
    )
