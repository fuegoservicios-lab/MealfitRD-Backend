# -*- coding: utf-8 -*-
"""[ARQ27-P0-01 + ARQ27-P1-01 · 2026-09-06] El selector perdía la dieta y las legumbres.

Dos defectos del mismo sitio — `dish_registry.template_candidates`, el embudo que decide qué platos
del registro curado ve el prompt.

**P0-01: la dieta no llegaba.** Filtraba franja, familia de proteína y alérgenos, y jamás miraba el
tipo de dieta. Medido sobre los seis snapshots el 06-sep: **1.109 de 1.646 candidatos ofrecidos
(67,4 %) eran incompatibles** con la dieta pedida en 2 dietas × 6 bibliotecas × 4 franjas — a un
vegano se le ofrecía paella de pollo y conejo; a una vegetariana, yogur con jamón.

Las guardas finales seguían ahí y el modelo podía proponer otra variante, así que esto **no era una
tasa de planes peligrosos**: era capacidad desperdiciada y una instrucción contradictoria. El prompt
pedía un plato que la propia dieta del perfil prohíbe, y el arreglo llegaba más tarde y más caro.

**P1-01: `legumbre` es una etiqueta de CLASE.** De las diez etiquetas `protein` que existen en los
snapshots, nueve nombran el alimento y ésta nombra una familia: dice «esta receta se apoya en una
legumbre», no CUÁL. El allocator programa la familia por su nombre («Lentejas»), así que
`family_matches('Lentejas', 'legumbre')` era False —igual que Garbanzos, Habichuelas y Guandules— y
las 64 plantillas de legumbre quedaban inalcanzables para el único perfil que las necesita.

El puente se abre SOLO para las etiquetas genéricas, y eso es load-bearing: resolver siempre por
constituyentes haría que «caldo de pollo» dentro de un guiso de res colara la receta como familia
`pollo`. Los tests 3 y 4 de abajo son esa mitad.

Y apareció una 14ª de la clase «dos ortografías del mismo alimento»: el catálogo escribe `Gandules`
(fila canónica, «guandules» es su ALIAS) y la tabla de familias solo tenía la forma con u, así que la
familia `Guandules` —que el pool VEGANO programa— no alcanzaba ni una plantilla.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import dish_registry as DR  # noqa: E402
import horizon as H  # noqa: E402

PAISES = ["DO", "PR", "MX", "CO", "ES", "US"]
FRANJAS = ["desayuno", "almuerzo", "merienda", "cena"]


def _constituyentes(country, tid):
    for t in (DR.load_registry(country) or {}).get("templates") or []:
        if t.get("template_id") == tid:
            return [c.get("canonical") or c.get("name") for c in (t.get("constituents") or [])]
    return []


def _guard():
    from graph_orchestrator import _diet_pool_item_banned
    return _diet_pool_item_banned


# ── P0-01: cero candidatos incompatibles en la batería 6 × 2 × 4 ──────────────────────────────
@pytest.mark.parametrize("dieta", ["vegan", "vegetarian"])
def test_cero_candidatos_incompatibles_en_las_seis_bibliotecas(dieta):
    """El criterio de cierre del gap, tal cual: cero candidatos incompatibles en las 6 bibliotecas ×
    2 dietas × 4 franjas. `k=99` para mirar TODO el pool, no solo el corte que ve el prompt."""
    banned = _guard()
    malos = []
    for c in PAISES:
        for s in FRANJAS:
            for cd in DR.template_candidates(c, s, None, k=99, diet=dieta):
                cons = _constituyentes(c, cd["template_id"])
                if any(banned(x, dieta) for x in cons):
                    malos.append((c, s, cd["template_id"], cd["name"]))
    assert not malos, f"{len(malos)} candidatos incompatibles con {dieta}: {malos[:5]}"


def test_sin_el_filtro_el_defecto_seguia_ahi():
    """Verificar contra el código roto. Sin `diet=`, el selector conserva su conducta previa y sigue
    ofreciendo incompatibles — si este test empezara a pasar en cero, el de arriba no probaría nada."""
    banned = _guard()
    incompatibles = 0
    for c in PAISES:
        for s in FRANJAS:
            for cd in DR.template_candidates(c, s, None, k=99):
                if any(banned(x, "vegan") for x in _constituyentes(c, cd["template_id"])):
                    incompatibles += 1
    assert incompatibles > 100, ("sin dieta el selector ya no ofrece incompatibles: o el catálogo "
                                 "cambió de raíz, o el filtro se coló donde no debía")


def test_una_dieta_sin_restriccion_no_recorta():
    """`balanced` (y una dieta desconocida) no filtran nada: el gap era ofrecer de más, no de menos."""
    for c in ("DO", "ES"):
        base = DR.template_candidates(c, "almuerzo", None, k=99)
        for d in (None, "", "balanced", "omnivora", "loquesea"):
            assert len(DR.template_candidates(c, "almuerzo", None, k=99, diet=d)) == len(base), \
                f"la dieta {d!r} recortó candidatos en {c}"


def test_la_dieta_recorta_de_verdad():
    """Contrapeso del anterior: vegano SÍ recorta. Un filtro que no quita nada no filtra."""
    base = len(DR.template_candidates("DO", "almuerzo", None, k=99))
    veg = len(DR.template_candidates("DO", "almuerzo", None, k=99, diet="vegan"))
    assert 0 < veg < base, f"vegano ofreció {veg} de {base} en DO/almuerzo"


def test_las_formas_femeninas_tambien_filtran():
    """`vegana`/`vegetariana` son data legacy real de `health_profile` (P1-DIET-CANON-SSOT: la tabla
    que las olvidó servía pollo a vegetarianas). Aquí llegan por `canonicalize_diet_type`."""
    ref = len(DR.template_candidates("DO", "almuerzo", None, k=99, diet="vegan"))
    for alias in ("vegana", "vegano", "VEGAN"):
        assert len(DR.template_candidates("DO", "almuerzo", None, k=99, diet=alias)) == ref, \
            f"{alias} no canonizó a vegan"


def test_sin_guard_no_se_ofrece_nada(monkeypatch):
    """La dirección segura cuando el guard SSOT no carga: quedarse sin bloque de registro (la conducta
    previa a F6), NUNCA ofrecer carne a un vegano. Fail-open aquí sería fail-open sobre una dieta."""
    monkeypatch.setattr(DR, "_diet_scope", lambda d: DR._DIET_UNAVAILABLE)
    assert DR.template_candidates("DO", "almuerzo", None, k=99, diet="vegan") == []


# ── P1-01: el puente de la etiqueta de clase ──────────────────────────────────────────────────
@pytest.mark.parametrize("familia,esperado_min", [("Lentejas", 15), ("Garbanzos", 10),
                                                  ("Habichuelas", 20), ("Guandules", 1)])
def test_las_familias_de_legumbre_alcanzan_plantillas(familia, esperado_min):
    """Antes: 0 en las cuatro. La cifra exacta puede subir si se curan más recetas; el suelo protege
    contra volver al cero."""
    n = 0
    for c in PAISES:
        for t in (DR.load_registry(c) or {}).get("templates") or []:
            prot = str(t.get("protein") or "none").lower()
            if prot in ("none", "mixta"):
                continue
            cons = [x.get("canonical") or x.get("name") for x in (t.get("constituents") or [])]
            if H.family_matches_template(familia, prot, cons):
                n += 1
    assert n >= esperado_min, f"la familia {familia} solo alcanza {n} plantillas"


def test_garbanzos_no_satisface_un_ancla_de_lentejas():
    """El criterio de cierre lo dice literal. `legumbre` las etiqueta a las dos, así que resolver por
    la etiqueta las confundiría; resolver por constituyentes las distingue."""
    falsos = []
    for c in PAISES:
        for t in (DR.load_registry(c) or {}).get("templates") or []:
            prot = str(t.get("protein") or "none").lower()
            if prot in ("none", "mixta"):
                continue
            cons = [x.get("canonical") or x.get("name") for x in (t.get("constituents") or [])]
            if H.family_matches_template("Lentejas", prot, cons) and \
                    not any("lenteja" in str(x).lower() for x in cons):
                falsos.append((c, t["template_id"], t["name"]))
    assert not falsos, f"plantillas sin lentejas que pasan el ancla 'Lentejas': {falsos[:5]}"


def test_una_etiqueta_especifica_no_se_abre_por_constituyentes():
    """El puente vale SOLO para las genéricas. Si se abriera para todas, un guiso de res con caldo de
    pollo pasaría por familia `pollo` — y ese falso positivo es peor que el gap que cerramos."""
    for c in PAISES:
        for t in (DR.load_registry(c) or {}).get("templates") or []:
            prot = str(t.get("protein") or "none").lower()
            if prot in ("none", "mixta") or prot in H._GENERIC_PROTEIN_TAGS:
                continue
            cons = [x.get("canonical") or x.get("name") for x in (t.get("constituents") or [])]
            for fam in ("Pollo", "Res", "Pescado", "Huevo", "Queso"):
                assert H.family_matches_template(fam, prot, cons) == H.family_matches(fam, prot), \
                    f"{c}/{t['template_id']}: la etiqueta específica {prot!r} se abrió para {fam}"


def test_legumbre_es_la_unica_etiqueta_generica():
    """Ancla del diseño: si mañana el compilador emite otra etiqueta de CLASE (`cereal`, `fruto seco`)
    su familia quedará inalcanzable en silencio, igual que le pasó a `legumbre`. Este test lo acusa."""
    etiquetas = set()
    for c in PAISES:
        for t in (DR.load_registry(c) or {}).get("templates") or []:
            etiquetas.add(str(t.get("protein") or "none").lower())
    # `tofu` la trajo ARQ27-P1-03 y NOMBRA UN ALIMENTO, así que resuelve por etiqueta y no necesita
    # puente. Este test es el que obligó a clasificarla en vez de dejarla entrar sin decidir nada.
    conocidas = {"none", "mixta", "pollo", "huevo", "pescado", "queso", "res", "atun", "cerdo",
                 "camarones", "pavo", "tofu"} | set(H._GENERIC_PROTEIN_TAGS)
    assert etiquetas <= conocidas, (
        f"etiquetas `protein` nuevas sin clasificar: {sorted(etiquetas - conocidas)}. Decide si "
        f"nombran un alimento (nada que hacer) o una CLASE (van a _GENERIC_PROTEIN_TAGS).")


def test_gandules_sin_u_es_la_fila_del_catalogo():
    """La 14ª de la clase «dos ortografías del mismo alimento». El representante de familia se inyecta
    como ingrediente y tiene que ser el nombre de la FILA, no su alias."""
    assert H.family_matches("Guandules", "Gandules"), "la forma sin u no casa con la familia"
    assert H.family_matches("Guandules", "Guandules"), "se perdió la forma con u"
    assert H.family_representative("guandules") == "Gandules"


def _familia_alcanza_alguna_plantilla(fam) -> bool:
    return any(
        H.family_matches_template(fam, str(t.get("protein") or "none").lower(),
                                  [x.get("canonical") or x.get("name") for x in (t.get("constituents") or [])])
        for c in PAISES for t in ((DR.load_registry(c) or {}).get("templates") or [])
        if str(t.get("protein") or "none").lower() not in ("none", "mixta"))


# `Tofu` estuvo aquí como `xfail(strict=True)` mientras F1 estaba sola: no faltaba el puente, faltaban
# las RECETAS —tofu firme tenía CERO usos como constituyente en las 690 plantillas—, así que un quinto
# del pool vegano programaba una familia que el registro no podía servir. ARQ27-P1-03 (F2) curó 88
# platos veganos y el caso pasó a XPASS, que es justo lo que el `strict` obligaba a atender. La lista
# de abajo se lee ahora entera y sin excepciones.
@pytest.mark.parametrize("fam", ["Lentejas", "Garbanzos", "Habichuelas", "Guandules",
                                 "Tofu", "Soya texturizada", "Edamame"])
def test_el_pool_vegano_programa_familias_alcanzables(fam):
    """Cierra el bucle: cada familia que el pool vegano programa tiene que llegar a alguna plantilla en
    alguna biblioteca. Programar una familia inalcanzable es gastar un día del horizonte en nada."""
    assert _familia_alcanza_alguna_plantilla(fam), f"la familia {fam} no alcanza ninguna plantilla"


def test_el_pool_vegano_y_la_parametrizacion_no_derivan():
    """Si alguien añade una familia al pool y no la añade arriba, quedaría sin comprobar — y una
    familia inalcanzable no se nota: el día simplemente se genera peor."""
    marcados = {"Lentejas", "Garbanzos", "Habichuelas", "Guandules", "Tofu", "Soya texturizada", "Edamame"}
    assert set(H._FAMILIES_BY_DIET["vegan"]) == marcados
