# -*- coding: utf-8 -*-
"""[P1-DO-SHARED-FOODS · 2026-09-07] Siete básicos criollos estaban archivados como comida de otro país.

La expansión de países dio de alta 141 filas SIN precio a propósito (los beta no tienen mercado RD
que cotizar) y las repartió por país. La partición asumía que cada alimento pertenece a UN país.
Siete no lo cumplen — son de ese país **y también** de República Dominicana:

    coditos, tocineta, salchichas -> archivados como US
    pernil                        -> PR
    fideos                        -> ES
    chicharron                    -> MX
    gallina criolla               -> CO

Y sin precio no existían para el generador dominicano. El gate del catálogo verificado
(`_vc_comprable` en `graph_orchestrator`) es:

    if (price_per_lb or 0) > 0 or (price_per_unit or 0) > 0: return True
    return bool(_iccui and _iccui(name, country=_vc_country))   # _iccui es None cuando el país es DO

Con `country == "DO"` no hay segunda rama. Así que a nadie en RD le salía un espagueti con
salchichas ni un sancocho con gallina criolla, aunque las siete filas existieran desde agosto con
su nutrición y su `fdc_id` correctos.

> Un alimento que existe en el catálogo pero no tiene precio no está incompleto: está INVISIBLE.
> Y una taxonomía por país es una PARTICIÓN — la forma equivocada para lo que se comparte.

Lo encontró una pregunta del dueño («¿por qué a un usuario no le puede salir coditos o fideos?»),
no un linter: ningún test podía verlo porque todos afirmaban justo lo contrario — que esas altas
DEBÍAN estar a cero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

PROMOVIDOS = ("Coditos", "Fideos", "Tocineta", "Salchichas", "Chicharrón",
              "Gallina criolla", "Pernil")

# Siguen sin precio A PROPÓSITO: se derivan de un padre ya precificado (los vegetales del sofrito,
# el pan) en vez de comprarse — el mismo patrón que `Clara de huevo` (`price_source='derived_huevo'`,
# sale del cartón). Están aquí para que una pasada futura no los arrastre sin pensarlo.
NO_PROMOVIDOS = ("Sofrito", "Pan rallado")


@pytest.fixture(scope="module")
def sc():
    import shopping_calculator
    return shopping_calculator


# ----------------------------------------------------- el token salió de la partición


def test_los_siete_ya_no_se_reconocen_como_alta_sin_precio(sc):
    """Con precio, el rescate por token SOBRA — y dejarlo puesto es un bug real.

    `_vc_comprable` los admite por la primera rama para TODOS los países, no solo el suyo. Un
    token superviviente los marcaría a la vez como priced y como «sin precio», que es lo que
    `test_i2_registry_collision_sweep_extendido_a_aliases` detecta sobre
    `canonicalize_shopping_food_name`.
    """
    for nombre in PROMOVIDOS:
        assert not sc.is_country_catalog_unpriced_item(nombre), (
            f"{nombre!r} sigue en `_COUNTRY_CATALOG_UNPRICED_BY_COUNTRY`")


def test_los_vecinos_que_no_se_promovieron_siguen_dentro(sc):
    """El recorte fue quirúrgico: quitar de más habría dejado comida fuera de la lista."""
    for nombre in NO_PROMOVIDOS:
        assert sc.is_country_catalog_unpriced_item(nombre), (
            f"{nombre!r} perdió su token sin haber ganado un precio: se cae de la lista")
    # y los vecinos LÉXICOS de `salchichas`, que son otro alimento
    assert sc.is_country_catalog_unpriced_item("Salchicha italiana")
    assert sc.is_country_catalog_unpriced_item("Salsa de salchicha")


def test_ningun_pais_perdio_mas_tokens_de_los_suyos(sc):
    """Cuántos salió de cada país. Fija el recorte para que un `replace` amplio se note."""
    d = sc._COUNTRY_CATALOG_UNPRICED_BY_COUNTRY
    assert {k: len(v) for k, v in d.items()} == {
        "ES": 31, "MX": 27, "CO": 17, "PR": 18, "US": 40, "DO": 1}


def test_la_vista_plana_se_deriva_y_no_se_edita_a_mano(sc):
    """Las dos vistas no pueden driftear: la plana SALE de la partición."""
    plana = set(sc._COUNTRY_CATALOG_UNPRICED_TOKENS)
    union = {t for ts in sc._COUNTRY_CATALOG_UNPRICED_BY_COUNTRY.values() for t in ts}
    assert plana == union


# ----------------------------------------------------- el alias que el dueño corrigió


def test_pernil_responde_a_pierna_de_cerdo(sc):
    """«Pernil se le dice pierna de cerdo» — corrección del dueño, 2026-09-07.

    La fila ya tenía `pierna de cerdo para hornear`, pero no el llano, que es como lo dice la
    gente y como lo lista Supermercado Nacional. Sin él el nombre no resuelve a la fila.
    """
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    from db_core import execute_sql_query

    r = (execute_sql_query("SELECT aliases FROM master_ingredients WHERE name='Pernil'",
                           fetch_all=True) or [])
    assert r, "la fila `Pernil` desapareció del catálogo"
    assert "pierna de cerdo" in (r[0]["aliases"] or []), r[0]["aliases"]


# ----------------------------------------------------- el efecto, medido donde importa


def test_los_siete_son_comprables_para_un_dominicano():
    """La prueba que da sentido a todo lo demás: ¿los ve el catálogo verificado de RD?

    Se reproduce la condición EXACTA de `_vc_comprable` con `country == "DO"` (sin rescate por
    token) en vez de llamar al helper del orquestador: ese helper filtra además por alergias y
    dieta del formulario, y aquí lo que se ancla es la puerta del precio.
    """
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    from shopping_calculator import get_master_ingredients

    por_nombre = {r.get("name"): r for r in (get_master_ingredients() or [])}
    for nombre in PROMOVIDOS:
        r = por_nombre.get(nombre)
        assert r, f"{nombre!r} desapareció del catálogo vivo"
        assert (r.get("price_per_lb") or 0) > 0 or (r.get("price_per_unit") or 0) > 0, (
            f"{nombre!r} volvió a quedarse sin precio: invisible para un dominicano")


def test_el_script_de_siembra_documenta_de_donde_sale_cada_precio():
    """Un precio sin procedencia es una invención con formato de dato.

    El script es el único registro de que estas siete cifras salieron de etiquetas reales y no de
    una estimación — igual que `nutrition_source_ref` lo es para la nutrición.
    """
    src = (_BACKEND / "scripts" / "seed_do_shared_foods_2026_09_07.py").read_text(encoding="utf-8")
    assert "Supermercado Nacional" in src
    for nombre in PROMOVIDOS:
        assert f'"{nombre}"' in src, f"{nombre} no aparece en el script de siembra"
    # `Pernil` es el único con confianza baja y el script debe decir por qué
    assert '"low"' in src and "NO DISPONIBLE" in src
