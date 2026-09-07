# -*- coding: utf-8 -*-
"""[P1-GLOSS-MAPUEY-DO · 2026-09-07] El gloss que va en dirección contraria a los otros 21.

`p1_country_gloss_es_2026_08_23.sql` pobló 22 glosses con UNA dirección: explicarle un
dominicanismo a un extranjero. «Auyama (calabaza)», «Chinola (maracuyá)», «Guineo (banana)» — el
nombre canónico es el que un dominicano reconoce y el paréntesis es para los demás. Por eso la
regla de render (`glossShoppingItemName`) excluye a DO: a un dominicano no le aclara nada.

El mapuey es el caso inverso, reportado por el dueño: en RD la gente dice «ñame» y «mapuey» suena
a otra cosa. Medido sobre 95 planes vivos: **83 menciones de «mapuey» en 12 planes** (y 77 de
«ñame»), así que la confusión es frecuente, no teórica. Aquí el nombre canónico es el RARO y el
paréntesis es el que aclara.

## Las tres decisiones que este test ancla

1. **«ñame indio», NO «ñame» a secas.** `Ñame` es OTRA fila viva del catálogo, con su propio
   precio (RD$76/lb contra RD$99/lb) y su propio `fdc_id` (170071 contra 169238): son *Dioscorea
   alata* y *Dioscorea trifida*. Glosar el mapuey como «ñame» mandaría a comprar el tubérculo
   equivocado, RD$23/lb más barato. Y `Ñame` NO gana gloss: hacerlo reintroduce la confusión al
   revés.

2. **Y no «ñame morado».** Ésa fue mi primera propuesta y la corrigió la fuente que trajo el
   dueño antes de que se escribiera nada: la pulpa del mapuey es CLARA. Sus otros nombres reales
   en RD son «yampí», «ñame indio» y «ñame blanco».

3. **`gloss_es`, jamás un alias.** El gloss es display-only por contrato — no toca `name`,
   aliases ni ninguna clave de matching. Como ALIAS sería peligroso: «ñame» es subcadena de
   «ñame indio» y colisionaría con la otra fila. Es la clase de bug que este proyecto lleva 17
   veces documentada (sal⊂salsa, res⊂fresco, pollo⊂repollo).

## Y la excepción es por ALIMENTO, no una puerta abierta

El dueño pidió primero abrirlo para los 22. Se le puso delante la lista de lo que leería —
«Auyama (calabaza)», «Guineo (banana)», «Chinola (maracuyá)» en cada línea — y eligió sólo el
mapuey. El guard del frontend comprueba que los otros 21 siguen limpios en DO.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_ROOT = _BACKEND.parent
_MIGRACION = "p1_gloss_mapuey_do_2026_09_07.sql"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _sql() -> str:
    return (_BACKEND / "migrations" / _MIGRACION).read_text(encoding="utf-8")


# ------------------------------------------------------------------ la migración


def test_la_migracion_vive_en_los_dos_ssot_y_es_identica():
    """[P3-MIGRATIONS-SSOT] Los dos repos son hermanos: cada uno necesita su copia física."""
    a, b = _BACKEND / "migrations" / _MIGRACION, _ROOT / "migrations" / _MIGRACION
    assert a.exists() and b.exists()
    assert a.read_bytes() == b.read_bytes()


def test_la_migracion_no_toca_la_identidad_y_es_idempotente():
    sql = _sql()
    assert "SET name =" not in sql, "`name` es identidad canónica: no se toca"
    assert "DROP" not in sql.upper()
    assert "IS DISTINCT FROM" in sql, "sin esto re-aplicarla escribiría de nuevo"
    assert "DO $$" in sql and "RAISE EXCEPTION" in sql


def test_la_migracion_glosa_el_mapuey_y_deja_el_name_intacto():
    sql = _sql()
    assert "'ñame indio'" in sql
    assert "WHERE name = 'Mapuey'" in sql
    # el sanity que impide la confusión simétrica
    assert "Ñame ganó un gloss" in sql


def test_la_migracion_explica_de_donde_sale_el_texto():
    """Un gloss sin procedencia es una elección de estilo; con ella es una decisión revisable."""
    sql = _sql()
    assert "Dioscorea" in sql and "170071" in sql and "169238" in sql
    assert "pulpa del mapuey es CLARA" in sql or "pulpa es CLARA" in sql or "CLARA" in sql


# ------------------------------------------------------------------ el dato vivo


def test_el_catalogo_vivo_tiene_el_gloss_y_name_no():
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    from db_core import execute_sql_query

    filas = execute_sql_query(
        "SELECT name, gloss_es, price_per_lb FROM master_ingredients WHERE name IN ('Mapuey','Ñame')",
        fetch_all=True) or []
    por = {r["name"]: r for r in filas}
    assert "Mapuey" in por and "Ñame" in por, f"faltan filas: {sorted(por)}"
    assert por["Mapuey"]["gloss_es"] == "ñame indio"
    assert por["Ñame"]["gloss_es"] is None, (
        "`Ñame` ganó un gloss: son dos filas distintas y glosar las dos reintroduce la confusión")
    assert float(por["Mapuey"]["price_per_lb"] or 0) != float(por["Ñame"]["price_per_lb"] or 0), (
        "si los precios se igualaran, la distinción dejaría de tener consecuencia en la compra")


def test_name_indio_no_es_alias_de_nadie():
    """Display-only. Como alias, «ñame» ⊂ «ñame indio» secuestraría la otra fila."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    from db_core import execute_sql_query

    filas = execute_sql_query(
        "SELECT name, aliases FROM master_ingredients WHERE aliases IS NOT NULL", fetch_all=True) or []
    culpables = [r["name"] for r in filas
                 if any("indio" in str(a).lower() for a in (r["aliases"] or []))]
    assert not culpables, f"«ñame indio» entró como alias en {culpables}"


# ------------------------------------------------------------------ la regla de render


def test_el_frontend_abre_DO_solo_para_los_glosses_inversos():
    src = (_ROOT / "frontend" / "src" / "utils" / "shoppingHelpers.js").read_text(encoding="utf-8")
    assert "_GLOSS_INVERSO_DO" in src
    assert "new Set(['mapuey'])" in src, "la excepción debe seguir siendo por alimento"
    # los cuatro mercados beta no cambian
    assert "_SPANISH_GLOSS_COUNTRIES = new Set(['ES', 'MX', 'CO', 'PR'])" in src, (
        "DO NO entra en el conjunto por país: entrar ahí abriría los 22 glosses de golpe")


def test_la_lista_inversa_sigue_siendo_una_excepcion():
    """Si crece, deja de ser excepción y pasa a ser propiedad del alimento — o sea, una columna.

    Un set en el cliente que enumera alimentos driftea del catálogo en silencio. Con uno o dos
    entradas el coste de esa deuda es cero; con quince, es el bug de mañana.
    """
    src = (_ROOT / "frontend" / "src" / "utils" / "shoppingHelpers.js").read_text(encoding="utf-8")
    bloque = src[src.index("_GLOSS_INVERSO_DO = new Set("):]
    entradas = bloque[:bloque.index(")")].count("'") // 2
    assert entradas <= 3, (
        f"{entradas} alimentos en `_GLOSS_INVERSO_DO`: pásalo a una columna de "
        f"`master_ingredients` en vez de seguir enumerando en el cliente")
