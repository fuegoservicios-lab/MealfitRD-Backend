# -*- coding: utf-8 -*-
"""[ARQ27-P1-03 · 2026-09-06] Las proteínas vegetales que el catálogo ya tenía, muertas.

**Tofu firme, Soya texturizada y Leche de soya existían como filas del catálogo público con CERO usos
como constituyente** en las 690 plantillas de las seis bibliotecas; Edamame aparecía una vez, en US.
El gap no era de catálogo: era de RECETAS. Nada las alcanzaba, así que la directiva histórica «sin
tofu: no se vende» convivía con una familia `Tofu` en el pool vegano y una fila `Tofu firme` en el
catálogo — un quinto del pool programaba una familia que el registro no podía servir, y ese día del
horizonte se gastaba en nada.

Medido con el embudo (`scripts/coverage_funnel.py`), mínimo de candidatos veganos por franja:

| Biblioteca | antes | después |
|---|---|---|
| DO | 4 | 7 |
| PR | 4 | 7 |
| MX | 3 | 7 |
| CO | 3 | 7 |
| ES | **1** | 7 |
| US | 3 | 7 |

Y el cuello que sólo apareció al medir la segunda vez: **`Avena` lleva la clase `gluten`**, y los
desayunos veganos se apoyaban en pan y avena. Un vegano celíaco tenía **un** desayuno en PR, ES y US.
La segunda tanda se construyó sobre quinoa, maíz, víveres y fruta y los dejó en 7.

El criterio de cierre del gap no es «añadimos N platos» sino **que el candidato nuevo llegue a la
decisión**: entre `7b6df93` y `8d83abb` se añadieron 23 plantillas y los IDs ofrecidos por el
blueprint fueron exactamente los mismos antes y después. Por eso estos tests miran el selector y el
pool, no el fichero de plantillas.
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
SLOTS = ["desayuno", "almuerzo", "merienda", "cena"]
# Piso de variedad: 7 ocasiones sin repetir en una franja ⇒ 7 recetas distintas (CATALOGO-VEGETAL §4).
PISO = 7


def _cons_de(country):
    for t in (DR.load_registry(country) or {}).get("templates") or []:
        for c in (t.get("constituents") or []):
            yield t, (c.get("canonical") or c.get("name") or "")


@pytest.mark.parametrize("alimento", ["Tofu firme", "Soya texturizada", "Leche de soya", "Edamame"])
def test_las_proteinas_vegetales_del_catalogo_se_usan(alimento):
    """Antes: 0, 0, 0 y 1. La fila existía en el catálogo público y ninguna receta la nombraba."""
    usos = sum(1 for c in PAISES for _t, n in _cons_de(c) if alimento.lower() in n.lower())
    assert usos >= 4, f"{alimento} solo aparece en {usos} constituyentes"


@pytest.mark.parametrize("alimento", ["Tofu firme", "Soya texturizada", "Leche de soya", "Edamame"])
def test_cada_una_llega_a_mas_de_una_cocina(alimento):
    """Concentrarlas en una biblioteca dejaría a las otras cinco igual que estaban."""
    paises = {c for c in PAISES for _t, n in _cons_de(c) if alimento.lower() in n.lower()}
    assert len(paises) >= 4, f"{alimento} solo llega a {sorted(paises)}"


@pytest.mark.parametrize("cc", PAISES)
def test_el_vegano_tiene_piso_de_variedad_en_las_cuatro_franjas(cc):
    """La medición que importa: lo que el SELECTOR ofrece, no lo que el fichero contiene. Añadir 23
    plantillas sin tocar los IDs ofrecidos ya pasó una vez."""
    flojas = {s: len(DR.template_candidates(cc, s, None, k=999, diet="vegan")) for s in SLOTS}
    bajo = {s: n for s, n in flojas.items() if n < PISO}
    assert not bajo, f"{cc}: franjas veganas por debajo de {PISO}: {bajo} (todas: {flojas})"


@pytest.mark.parametrize("cc", PAISES)
def test_el_vegano_sin_gluten_conserva_desayuno(cc):
    """`Avena` lleva `gluten` y los desayunos veganos se apoyaban en pan y avena: PR, ES y US se
    quedaban con UNO. El piso aquí es más bajo que 7 a propósito — es un cruce, no el caso base— pero
    un cruce con un solo candidato no es una oferta, es una repetición forzada."""
    n = len(DR.template_candidates(cc, "desayuno", None, k=999, diet="vegan",
                                   exclude_allergens=("gluten",)))
    assert n >= 5, f"{cc}: solo {n} desayunos veganos sin gluten"


def test_las_altas_no_dependen_todas_de_la_soja():
    """«Ninguna ola queda cubierta exclusivamente con soja» (CATALOGO-VEGETAL §6). Si al quitar la
    soja el desayuno vegano se cae, la variedad era aparente."""
    for cc in PAISES:
        n = len(DR.template_candidates(cc, "desayuno", None, k=999, diet="vegan",
                                       exclude_allergens=("soya",)))
        assert n >= 5, f"{cc}: sin soja quedan {n} desayunos veganos"


def test_todas_las_altas_compilan_integras():
    """Una plantilla nueva con un ingrediente sin resolver nacería `partial` y el selector no la
    ofrecería nunca: el alta sería trabajo perdido y silencioso (ARQ27-P0-02)."""
    for cc in PAISES:
        snap = DR.load_registry(cc) or {}
        parciales = [t["name"] for t in snap.get("templates") or [] if t.get("status") != "ok"]
        # DO conserva sus 4 históricas (zapote, chillo, salami de pavo, menta); nada más.
        assert len(parciales) <= (4 if cc == "DO" else 0), f"{cc}: {parciales}"


def test_las_altas_veganas_no_llevan_nada_animal():
    """El guard SSOT sobre los constituyentes de cada plantilla que declare una franja vegana. No se
    fía del nombre del plato: «Chuleta de soya» y «Chili vegano» son exactamente los que un matcher
    por título clasificaría mal."""
    from graph_orchestrator import _diet_pool_item_banned
    fallos = []
    for cc in PAISES:
        for s in SLOTS:
            for cd in DR.template_candidates(cc, s, None, k=999, diet="vegan"):
                t = next(x for x in DR.load_registry(cc)["templates"]
                         if x["template_id"] == cd["template_id"])
                for c in (t.get("constituents") or []):
                    n = c.get("canonical") or c.get("name")
                    if _diet_pool_item_banned(n, "vegan"):
                        fallos.append((cc, t["name"], n))
    assert not fallos, fallos[:5]


def test_una_biblioteca_solo_ofrece_lo_que_su_mercado_vende():
    """Las altas usan nombres del catálogo global; si alguna metiera un ingrediente que su propio
    mercado no lleva, el filtro de ARQ27-P1-07 la borraría del pool y el alta sería inútil."""
    from catalog_capability import is_available
    fallos = []
    for cc in PAISES:
        for t in (DR.load_registry(cc) or {}).get("templates") or []:
            if t.get("status") != "ok":
                continue
            for c in (t.get("constituents") or []):
                n = c.get("canonical") or c.get("name")
                if is_available(n, cc) is False:
                    fallos.append((cc, t["name"], n))
    assert not fallos, f"plantillas con ingrediente fuera de su propio mercado: {fallos[:5]}"


def test_el_pool_vegano_incluye_las_dos_familias_nuevas():
    """Se nombran con el nombre de FILA del catálogo a propósito: se resuelven por el puente de
    etiqueta genérica (`legumbre`) mirando los constituyentes. Escribir «Soya» a secas habría hecho
    que «Salsa de soya» —un condimento de 8 g de proteína— pasara por proteína del día."""
    pool = H._FAMILIES_BY_DIET["vegan"]
    assert "Soya texturizada" in pool and "Edamame" in pool
    assert "Soya" not in pool, "«Soya» a secas alcanzaría «Salsa de soya»"


def test_ninguna_alta_repite_nombre_dentro_de_su_biblioteca():
    """Dos platos con el mismo nombre no son variedad: son el mismo plato contado dos veces, y el
    embudo los contaría como dos candidatos."""
    for cc in PAISES:
        nombres = [t["name"] for t in (DR.load_registry(cc) or {}).get("templates") or []]
        dup = {n for n in nombres if nombres.count(n) > 1}
        assert not dup, f"{cc}: nombres repetidos {dup}"
