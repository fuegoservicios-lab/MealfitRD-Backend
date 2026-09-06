# -*- coding: utf-8 -*-
"""[P1-COVERAGE-FUNNEL · 2026-09-05] ¿Cuántos platos sobreviven de verdad a los filtros de un usuario?

Las seis bibliotecas suman 667 entradas y ese número no dice nada útil: nadie había medido cuántas quedan
después de dieta, alergias, franja y conservación. Este guion responde eso con un EMBUDO — cuántos candidatos
entran y cuántos caen en cada filtro, con el motivo— para los cruces que ya sabemos duros.

No es una función del producto: es una medición que se vuelve a correr cuando cambian catálogo o reglas.

    python scripts/coverage_funnel.py                 # tabla completa
    python scripts/coverage_funnel.py --json          # para diffear entre versiones
    python scripts/coverage_funnel.py --perfil dominican_criolla

Lo que NO mide: si el solver puede dimensionar el plato con esos gramos, ni si el precio cabe en el presupuesto.
Un candidato que sobrevive aquí es elegible, no necesariamente servible — por eso la última columna se llama
«elegibles» y no «viables».
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# La consola de Windows llega en cp1252 y esta tabla lleva flechas y acentos: sin esto el guion muere al
# imprimir, no al medir, que es la peor forma de perder una medición.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import dish_registry as dr          # noqa: E402
import pantry_durability as pdur    # noqa: E402

SLOTS = ("desayuno", "almuerzo", "merienda", "cena")

# [ARQ27-P1-02 · 2026-09-06] Aquí vivía `_DIET_EXCLUDES`: una tabla PROPIA de familias de `protein`
# que cada dieta excluye. Medía otra cosa que el selector — la etiqueta `protein` dice qué protagoniza
# el plato, jamás qué contiene, así que una plantilla `none` con lácteos o una `mixta` con jamón
# pasaban el filtro del embudo y el vegano las veía como elegibles. Sobre ES/almuerzo vegano el
# recuento caía de 26 a 5 al ejecutar la guarda real sobre los constituyentes.
#
# Un embudo que no usa el filtro del producto no mide el producto: mide su propia tabla. Ahora la
# etapa de dieta llama a `_diet_pool_item_banned`, el mismo guard SSOT que `template_candidates`, y
# la etapa de mercado al mismo `catalog_capability` que la política. La prueba de que no han vuelto a
# divergir es `test_p0_arq27_f1_embudo.py`, que compara los dos recuentos.
# Alérgenos tal como los ESCRIBE el registry, en español y con las dos formas que convive cada uno
# («lacteos»/«lactosa», «huevo»/«huevos»). Medido sobre los seis snapshots, no supuesto: escribir «dairy» aquí
# dejaba el filtro inerte y las columnas salían idénticas a las de «sin alergias» sin que nada avisara.
_ALLERGY_SETS = {
    "sin alergias": (),
    "lácteos": ("lacteos", "lactosa"),
    "lácteos + huevo": ("lacteos", "lactosa", "huevo", "huevos"),
    "mariscos + pescado": ("mariscos", "pescado"),
    "gluten": ("gluten",),
    "soya": ("soya",),
    "soya + gluten": ("soya", "gluten"),
}

# Los cruces que ya sabemos duros, más el caso base de cada biblioteca.
ESCENARIOS = [
    # (etiqueta, perfil cultural, mercado, dieta, alergias, día del ciclo, modo de congelador)
    ("base",                       None, None, "omnivora",     "sin alergias",        None, "limited"),
    ("vegetariano",                None, None, "vegetariana",  "sin alergias",        None, "limited"),
    ("vegano",                     None, None, "vegana",       "sin alergias",        None, "limited"),
    ("veg + sin lácteos",          None, None, "vegetariana",  "lácteos",             None, "limited"),
    ("veg + sin lácteos ni huevo", None, None, "vegetariana",  "lácteos + huevo",     None, "limited"),
    ("sin mariscos ni pescado",    None, None, "omnivora",     "mariscos + pescado",  None, "limited"),
    ("sin gluten",                 None, None, "omnivora",     "gluten",              None, "limited"),
    ("vegano sin gluten",          None, None, "vegana",       "gluten",              None, "limited"),
    ("vegano sin soja",            None, None, "vegana",       "soya",                None, "limited"),
    ("vegano sin soja ni gluten",  None, None, "vegana",       "soya + gluten",       None, "limited"),
    ("día 25 sin congelador",      None, None, "omnivora",     "sin alergias",          25, "none"),
    ("veg · día 25 sin congelador", None, None, "vegetariana", "sin alergias",          25, "none"),
    ("día 25 congelador limitado", None, None, "omnivora",     "sin alergias",          25, "limited"),
]


def _templates(country: str) -> list:
    snap = dr.load_registry(country) or {}
    return list(snap.get("templates") or [])


def _cons(t) -> list:
    return [c.get("canonical") or c.get("name") for c in (t.get("constituents") or [])]


def _mercado(country):
    """Nombres comprables del país, o `None` si el catálogo no se pudo leer. `None` no es cero: sin
    catálogo la etapa se declara omitida en vez de tirar todos los platos."""
    try:
        from catalog_capability import known_ingredient_names
        return known_ingredient_names(country)
    except Exception:
        return None


def embudo(country: str, slot: str, dieta: str, alergias: tuple, need_days, freezer_mode: str,
           *, requiere_nutrientes: tuple = ()) -> dict:
    """Cuenta cuántos candidatos quedan tras cada filtro, en el MISMO orden que aplica el selector, y
    conserva QUÉ cayó en cada etapa. Un contador dice que se perdieron veinte platos; la lista dice
    cuáles, que es lo único con lo que se puede actuar."""
    from graph_orchestrator import _diet_pool_item_banned

    etapas, caidas, restantes = [], {}, _templates(country)
    etapas.append(("catálogo", len(restantes)))

    def _filtrar(nombre, predicado):
        nonlocal restantes
        pasan, caen = [], []
        for t in restantes:
            (pasan if predicado(t) else caen).append(t)
        restantes = pasan
        if caen:
            caidas[nombre] = [{"id": t.get("template_id"), "name": t.get("name")} for t in caen]
        etapas.append((nombre, len(restantes)))

    _filtrar("compilado ok", lambda t: t.get("status") == "ok")
    _filtrar("franja", lambda t: slot in (t.get("slots") or []))

    ex = {a.lower() for a in alergias}
    _filtrar("alergias", lambda t: not ex.intersection(
        {a.lower() for a in (t.get("intrinsic_risk_attributes") or {}).get("allergens", [])}) if ex else True)

    # La guarda REAL, sobre todos los constituyentes — no la etiqueta `protein`.
    _filtrar("dieta", lambda t: not any(_diet_pool_item_banned(x, dieta) for x in _cons(t)))

    conocidos = _mercado(country)
    if conocidos is None:
        etapas.append(("mercado (omitido)", len(restantes)))
    else:
        from plan_policy import _matches
        _set = list(conocidos)
        _filtrar("mercado", lambda t: all(any(_matches(x, k) for k in _set) for x in _cons(t)))

    if requiere_nutrientes:
        req = set(requiere_nutrientes)
        _filtrar("datos exigidos", lambda t: not req.intersection(t.get("nutrition_unknown") or {}))

    if need_days:
        ventana = pdur.freeze_window_days(freezer_mode, 30)
        permite_congelar = int(need_days) < ventana
        _filtrar("conservación", lambda t: pdur.template_fits(
            (t.get("logistics") or {}).get("days_fresh_min"),
            (t.get("logistics") or {}).get("days_with_freezer_min"),
            int(need_days) + 1, permite_congelar))
    else:
        etapas.append(("conservación", len(restantes)))

    return {"etapas": etapas, "elegibles": len(restantes), "caidas": caidas,
            "ejemplos": [t["name"] for t in restantes[:3]]}


def correr(perfiles=None) -> dict:
    import cultural_profiles as cp
    salida = {"perfiles": {}}
    for pid, perfil in cp.PROFILES.items():
        if perfiles and pid not in perfiles:
            continue
        # El perfil es un dict y su mercado se llama `market_default`; `country` no existe y devolvía None,
        # con lo que las seis bibliotecas cargaban el snapshot por defecto y la tabla salía SEIS VECES IGUAL.
        country = (perfil or {}).get("market_default") or (perfil or {}).get("library")
        salida["perfiles"][pid] = {"country": country, "escenarios": {}}
        for etiqueta, _p, _m, dieta, alergia, need_days, freezer in ESCENARIOS:
            por_slot = {}
            for slot in SLOTS:
                por_slot[slot] = embudo(country, slot, dieta, _ALLERGY_SETS[alergia], need_days, freezer)
            salida["perfiles"][pid]["escenarios"][etiqueta] = por_slot
    return salida


def render(res: dict) -> str:
    filas = ["", "EMBUDO DE COBERTURA — cuántos platos sobreviven a los filtros de un usuario", ""]
    for pid, p in res["perfiles"].items():
        filas.append(f"── {pid}  (mercado {p['country']}) " + "─" * max(0, 46 - len(pid)))
        filas.append(f"   {'escenario':30s} {'desayuno':>9s} {'almuerzo':>9s} {'merienda':>9s} {'cena':>9s}   {'mínimo':>7s}")
        for etiqueta, por_slot in p["escenarios"].items():
            n = [por_slot[s]["elegibles"] for s in SLOTS]
            marca = "  ⚠" if min(n) < 3 else ("  ×" if min(n) == 0 else "")
            filas.append(f"   {etiqueta:30s} " + " ".join(f"{x:9d}" for x in n) + f"   {min(n):7d}{marca}")
        filas.append("")
    filas.append("⚠ = alguna franja baja de 3 candidatos · × = alguna franja se queda sin ninguno")
    filas.append("«elegibles» = pasa franja, alergias, dieta (guarda real sobre constituyentes), mercado")
    filas.append("y conservación. NO incluye solver, precio ni cuotas: elegible no es servible.")
    return "\n".join(filas)


def desglose(res: dict, pid: str, etiqueta: str) -> str:
    p = res["perfiles"][pid]["escenarios"][etiqueta]
    out = [f"", f"desglose · {pid} · {etiqueta}", ""]
    for slot in SLOTS:
        et = p[slot]["etapas"]
        cadena = "  →  ".join(f"{nombre} {n}" for nombre, n in et)
        out.append(f"   {slot:9s} {cadena}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Embudo de cobertura del catálogo de platos")
    ap.add_argument("--json", action="store_true", help="salida en JSON para diffear entre versiones")
    ap.add_argument("--perfil", action="append", help="limitar a uno o varios perfiles")
    ap.add_argument("--desglose", metavar="ESCENARIO", help="etapa a etapa para ese escenario")
    a = ap.parse_args()
    # [ARQ27-P1-07] La etapa de mercado lee `master_ingredients`, y fuera de FastAPI el pool no está
    # abierto: sin esto el catálogo sale vacío, la etapa se declara omitida y el embudo mide de menos
    # sin decir por qué. Fail-open: si no hay DB, la etapa se salta y la tabla lo dice.
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_BACKEND, ".env"))
        import db_core
        db_core.connection_pool.open()
    except Exception as e:
        print(f"(sin catálogo vivo: {type(e).__name__} — la etapa de mercado se omite)", file=sys.stderr)
    res = correr(a.perfil)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))     # [P2-LOGGER-EXEMPT: guion de línea de órdenes]
        return 0
    print(render(res))                                            # [P2-LOGGER-EXEMPT: guion de línea de órdenes]
    if a.desglose:
        for pid in res["perfiles"]:
            print(desglose(res, pid, a.desglose))                 # [P2-LOGGER-EXEMPT: guion de línea de órdenes]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
