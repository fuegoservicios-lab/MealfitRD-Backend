"""[P1-COUNTRY-DIARY-DISHES-60-DE-60-DO · 2026-08-23] Genera el catálogo de platos del DIARIO
para los países beta, con la misma forma y el mismo método que `build_dominican_dishes.py`.

EL DEFECTO QUE CIERRA (medido): `load_dishes()` devolvía 60 claves y las 60 eran dominicanas.
Búsqueda por término: paella 0 · gazpacho 0 · cocido 0 · tacos 0 · pozole 0 · arepa 0 · ajiaco 0.
El modo seguimiento (P1-PLAN-MODE) se vende como producto independiente: un usuario beta abre
«Registrar comida», busca lo que acaba de comer y no encuentra NADA de su cocina — tiene que
componer una paella ingrediente a ingrediente, que es justo el trabajo que el componedor existe
para evitarle. O la registra mal, que es peor: contamina sus macros del día.

DE DÓNDE SALEN LOS PLATOS, Y POR QUÉ NO ME LOS INVENTO. No hay recetas nuevas aquí: las
plantillas por país (`data/dish_templates_{es,mx,co,pr}.json`, 203 en total) YA traen
`constituents` curados con gramos, porque el generador de planes las usa para componer menús.
Este script las lee y calcula los macros resolviendo cada constituyente contra
`master_ingredients` — exactamente el mismo camino que los 60 dominicanos. Los números no son
una estimación mía: son la suma de filas del catálogo dividida entre el peso final.

SE AÑADE, NUNCA SE QUITA. `P2-DIARY-CATALOG-COUNTRY` decidió no filtrar el catálogo por país, y
esa decisión es correcta: un dominicano en España sigue comiendo mangú, y un español en Santo
Domingo sigue comiendo tortilla. Este script produce ficheros APARTE que `load_dishes()` une a
los dominicanos; ninguno de los 60 originales se toca.

USO (el script escribe el fichero él mismo; NO redirijas stdout, ver la nota del final):
    python backend/scripts/build_country_dishes.py es    → data/spanish_dishes.json
    python backend/scripts/build_country_dishes.py mx    → data/mexican_dishes.json
    python backend/scripts/build_country_dishes.py co    → data/colombian_dishes.json
    python backend/scripts/build_country_dishes.py pr    → data/puertorican_dishes.json

Requiere el pool abierto (fuera de FastAPI hay que abrirlo o `master_ingredients` sale vacío y
se mide el vacío, no el sistema — lección de `runbook_sql_forensic_sop`).
"""
# [P2-LOGGER-EXEMPT: script CLI de generación; su salida ES el fichero y el informe va a stderr]
import io
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#: Peso final por defecto cuando la plantilla no lo declara. Las plantillas describen UNA ración
#: ya emplatada, así que el peso final es la suma de crudos menos la merma de cocción. 0.88 es el
#: factor que el set dominicano promedia entre hervido, plancha y horno; no vale para frituras
#: profundas, pero ninguna plantilla lo es.
_MERMA = 0.88

#: Nombre del fichero por país. Los dominicanos ya viven en `dominican_dishes.json`.
_FICHERO = {"es": "spanish", "mx": "mexican", "co": "colombian", "pr": "puertorican"}


def _clave(nombre: str) -> str:
    """`Paella de mariscos con gambas` → `paella_de_mariscos_con_gambas`. Sin acentos porque la
    clave viaja en la URL del buscador y en el JSON, y el catálogo dominicano usa el mismo estilo."""
    s = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def main():
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ("es", "mx", "co", "pr"):
        print("uso: build_country_dishes.py <es|mx|co|pr>", file=sys.stderr)
        return 2
    pais = sys.argv[1].lower()

    from nutrition_db import IngredientNutritionDB
    import db_core
    if getattr(db_core, "connection_pool", None):
        db_core.connection_pool.open()
    db = IngredientNutritionDB()

    with open(os.path.join(_DATA, f"dish_templates_{pais}.json"), encoding="utf-8") as f:
        plantillas = (json.load(f) or {}).get("templates") or []

    out, sin_resolver, faltantes = {}, 0, set()
    print(f"{'plato':40} {'kcal/100g':>9} {'P':>5} {'C':>6} {'F':>5}  {'cob':>4}", file=sys.stderr)
    for tpl in plantillas:
        ings = tpl.get("constituents") or []
        if not ings:
            continue
        tot = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
        constituents, resueltos, crudo_g = [], 0, 0.0
        for ing in ings:
            name = ing.get("name")
            g = float(ing.get("grams") or ing.get("g") or 0)
            crudo_g += g
            m = db.macros_from_ingredient_string(f"{g}g {name}") or {}
            ok = bool(m)
            if ok:
                resueltos += 1
                for k in tot:
                    tot[k] += (m.get(k) or 0.0)
            constituents.append({"name": name, "g": g, "resolved": ok,
                                 "macros": {k: round(m.get(k) or 0.0, 1) for k in tot} if ok else None})
        if not crudo_g:
            continue
        fg = round(crudo_g * _MERMA)
        per100 = {k: round(tot[k] / fg * 100, 1) for k in tot}
        cov = round(resueltos / len(ings), 2)
        if cov < 1.0:
            # [P1-COUNTRY-DIARY-DISHES · 2026-08-23] Un plato al que le falta un ingrediente NO
            # se publica. Sus macros no son «aproximados»: son BAJOS por la cantidad exacta que
            # falta, y el diario los suma al total del día del usuario — una comida registrada
            # de menos es peor que una comida no registrada, porque parece registrada.
            # Medido: los 18 platos mexicanos que fallaban lo hacían por UN ingrediente ausente
            # del catálogo, «Tortilla de maíz», o sea justo su base: publicarlos habría sido
            # registrar tacos sin tortilla. Mejor 31 platos ciertos que 49 con 18 mintiendo.
            # El informe de stderr los nombra para que el hueco del catálogo se pueda cerrar.
            sin_resolver += 1
            faltantes.update(c["name"] for c in constituents if not c["resolved"])
            print(f"  OMITIDO {tpl['name'][:44]:44} cob={cov} falta: "
                  + ", ".join(c["name"] for c in constituents if not c["resolved"]),
                  file=sys.stderr)
            continue
        out[_clave(tpl["name"])] = {
            "label": tpl["name"],
            "method": tpl.get("technique") or "",
            "finished_g": fg,
            "per_100g": per100,
            "resolution_coverage": cov,
            "constituents": constituents,
            "fndds_cross_check": None,
            "provenance": f"computed from catalog constituents of dish_templates_{pais}.json (CC0)",
            "country_origin": pais.upper(),
        }
        print(f"{tpl['name'][:40]:40} {per100['kcal']:9} {per100['protein']:5} "
              f"{per100['carbs']:6} {per100['fats']:5}  {cov:4}", file=sys.stderr)

    print(f"\n{len(out)} platos publicados · {sin_resolver} OMITIDOS por cobertura <1.0",
          file=sys.stderr)
    if faltantes:
        print("  ingredientes que el catálogo no tiene: " + ", ".join(sorted(faltantes)),
              file=sys.stderr)
    # [P1-COUNTRY-DIARY-DISHES · 2026-08-23] Escribe el fichero ÉL MISMO, en UTF-8 explícito,
    # en vez de dejar que el shell redirija stdout: en Windows la redirección usa cp1252 y
    # convierte cada acento en un byte inválido — los cuatro catálogos salieron ilegibles la
    # primera vez, y el JSON no se podía ni cargar.
    destino = os.path.join(_DATA, f"{_FICHERO[pais]}_dishes.json")
    payload = json.dumps({
        "_note": (f"[P1-COUNTRY-DIARY-DISHES-60-DE-60-DO] Platos de {pais.upper()} para el buscador "
                  "del diario, derivados de los constituyentes ya curados de "
                  f"dish_templates_{pais}.json y resueltos contra master_ingredients. Se AÑADEN a "
                  "los 60 dominicanos, no los sustituyen. Generado por scripts/build_country_dishes.py."),
        "dishes": out,
    }, ensure_ascii=False, indent=1)
    with io.open(destino, "w", encoding="utf-8", newline="\n") as f:
        f.write(payload + "\n")
    print(f"escrito {destino}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
