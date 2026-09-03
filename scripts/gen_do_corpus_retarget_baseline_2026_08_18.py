#!/usr/bin/env python
"""[P1-COUNTRY-SYSTEM-F2 · ola final (review de fase) · 2026-08-18 · C3 Durable Guard]
Genera `scripts/data/do_corpus_retarget_baseline_2026_08_18.json`, el mapping COMMITTED que
ancla `tests/test_p1_country_system_f2.py::test_c3_durable_guard_do_corpus_retarget_baseline`.

CORPUS (unión, exactamente como lo describe el review de fase):
  1. `data/dish_templates.json` (DO): cada `name`/`protein`/`base` de cada template.
  2. Los 4 pools `DOMINICAN_*` (constants.py): cada string.
  3. `GLOBAL_REVERSE_MAP` (constants.py): cada KEY (variante) y cada VALUE (base).
  4. Cada fila PRE-FASE de `master_ingredients` (nombre canónico + cada alias) -- "pre-fase" se
     define como toda fila cuyo `name` NO aparece en ninguno de los 4
     `scripts/data/new_foods_*_2026_08_17.json` (144 altas de Fase 2). No hay columna
     `created_at` en `master_ingredients` (verificado contra `information_schema.columns`), así
     que el criterio es el explícito permitido por el contrato: "an explicit frozen list" --
     aquí, la lista de nombres de los 4 JSON de altas, que SÍ está committeada en git.

Para cada string del corpus se resuelve HOY (`shopping_calculator.normalize_name`, catálogo
vivo) -- eso es lo que el mapping fija. 4 entradas difieren de lo que hubieran resuelto
PRE-FASE (reconstruido quitando las 144 filas nuevas del catálogo en memoria) -- las 4 son
mejoras aceptadas documentadas en `accepted_deltas` con su razón; NINGUNA es uno de los 6 bugs
que esta misma ola cerró (esos ya vuelven a coincidir con pre-fase, ver `C3` en
`shopping_calculator.py`/`retarget_alias_fix_2026_08_18.py`).

Re-correr esto NO debería cambiar el archivo salvo que:
  (a) el catálogo vivo cambie de forma que altere una resolución YA fijada (regresión real -- el
      test que lee este JSON lo atrapa primero, no hace falta re-generar a ciegas), o
      NUNCA re-generar para silenciar un fallo del test sin revisar el diff primero.
  (b) se añade contenido nuevo al corpus (nuevo template DO, nuevo pool, nueva fila) y el string
      nuevo necesita su entrada -- entonces SÍ re-correr y revisar el diff línea a línea antes
      de commitear (mismo principio que el `country_gaps/*.json` --commit).

USO:
    cd backend
    python scripts/gen_do_corpus_retarget_baseline_2026_08_18.py

[P2-LOGGER-EXEMPT: script CLI one-shot, la salida a stdout ES el producto]
"""
import copy
import datetime
import json
import os
import sys

_AQUI = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_AQUI)
sys.path.insert(0, _BACKEND)
os.chdir(_BACKEND)

os.environ.setdefault("MEALFIT_DB_BACKEND", "neon")
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_BACKEND, ".env"))
except Exception:
    pass

import db_core  # noqa: E402
db_core.connection_pool.open()

import shopping_calculator as sc  # noqa: E402
import constants as const  # noqa: E402

_NEW_FOOD_FILES = [
    "scripts/data/new_foods_es_2026_08_17.json",
    "scripts/data/new_foods_mx_co_2026_08_17.json",
    "scripts/data/new_foods_pr_us_2026_08_17.json",
    "scripts/data/new_foods_rd_topup_2026_08_17.json",
]


def build_do_corpus():
    """(corpus:set[str], new_row_names:set[str]) -- ver docstring del módulo para la receta."""
    strings = set()

    with open("data/dish_templates.json", encoding="utf-8") as f:
        dt = json.load(f)
    for t in dt["templates"]:
        for field in ("name", "protein", "base"):
            if t.get(field):
                strings.add(t[field])

    for pool_name in ("DOMINICAN_PROTEINS", "DOMINICAN_CARBS", "DOMINICAN_VEGGIES_FATS", "DOMINICAN_FRUITS"):
        strings.update(getattr(const, pool_name))

    for k, v in const.GLOBAL_REVERSE_MAP.items():
        strings.add(k)
        strings.add(v)

    new_row_names = set()
    for fn in _NEW_FOOD_FILES:
        with open(fn, encoding="utf-8") as f:
            for r in json.load(f):
                new_row_names.add(r["name"])

    master_list = sc.get_master_ingredients()
    for r in master_list:
        if r["name"] in new_row_names:
            continue
        strings.add(r["name"])
        strings.update(r.get("aliases") or [])

    strings = {s for s in strings if isinstance(s, str) and s.strip()}
    return strings, new_row_names


# [C3 Durable Guard] Los únicos deltas conocidos entre pre-fase (fila nunca existió) y hoy
# (post-ola-final). Cada uno es una mejora ACEPTADA, no un bug -- razón inline.
ACCEPTED_DELTAS = {
    "alubias blancas": (
        "Pre-fase: sin resolver (passthrough). 'Judías blancas' (alta ES T5) trae 'alubias "
        "blancas' como alias propio -- relleno de un hueco, no un retargeteo desde OTRA fila "
        "establecida (verificado: ninguna fila pre-fase, incluida 'Habichuelas blancas', tenía "
        "'alubias blancas' como alias)."
    ),
    "fideos integrales": (
        "Pre-fase: sin resolver (passthrough). 'Fideos' (alta ES T5) matchea via su alias bare "
        "'fideos' (CONTAINS). Candidato alternativo 'Pasta integral' (fila pre-fase) NO tiene "
        "'fideos integrales' como alias -- no hay retargeteo desde una fila establecida. Macro "
        "similar (kcal 371 vs 370.5, protein 13g vs 13.5g, carbs 74.7g vs 73.1g) salvo fibra "
        "(3.2g vs 10.1g) -- divergencia menor, no clínicamente peligrosa como el caso chicharrón."
    ),
    "chicharron de cerdo": (
        "Pre-fase: 'Cerdo' (genérico, via substring). Ahora 'Chicharrón' (alta CO T6) -- MEJORA "
        "ACEPTADA a propósito, documentada en new_foods_mx_co_2026_08_17.json._provenance: "
        "chicharrón real es cuero+grasa RENDIDA (kcal 544/fat 31.3g) vs 'Cerdo' genérico (kcal "
        "169.6/fat 9.47g) -- >200% de diferencia, clínicamente significativa en la dirección "
        "MÁS precisa."
    ),
    "chicharrón de cerdo": (
        "Mismo caso que 'chicharron de cerdo' (variante con tilde) -- ver esa entrada."
    ),
}


def resolve_all(strings):
    return {s: sc.normalize_name(s) for s in sorted(strings)}


def main():
    strings, new_row_names = build_do_corpus()
    mapping = resolve_all(strings)

    # sanity: los 6 bugs que esta ola cerró deben resolver a su target correcto (no re-generar
    # un baseline que congele un bug).
    _sanity = {
        "chicharrón de pollo": "Pechuga de pollo",
        "chicharron de pollo": "Pechuga de pollo",
        "ricotta": "Queso ricotta",
        "pinto beans": "Frijoles pintos",
    }
    for s, expected in _sanity.items():
        assert mapping.get(s) == expected, (
            f"SANITY FALLIDA: {s!r} resuelve a {mapping.get(s)!r}, esperaba {expected!r} -- "
            f"¿corriste esto ANTES de aplicar retarget_alias_fix_2026_08_18.py --commit y el "
            f"guard C3.1 de shopping_calculator.py?"
        )
    for s in ("lambi", "lambí"):
        if s in mapping:
            assert mapping[s] != "Cordero", (
                f"SANITY FALLIDA: {s!r} sigue resolviendo a 'Cordero' -- el alias 'lamb' no se "
                f"removió o el fuzzy match sigue activo"
            )

    payload = {
        "_doc": (
            "Committed baseline del C3 Durable Guard (P1-COUNTRY-SYSTEM-F2 ola final, "
            "2026-08-18). Generado por scripts/gen_do_corpus_retarget_baseline_2026_08_18.py. "
            "Cambios intencionales van en 'accepted_deltas' con su razón -- editar el mapping a "
            "mano sin pasar por el generador rompe la trazabilidad."
        ),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "corpus_size": len(strings),
        "new_fase2_rows_excluded_count": len(new_row_names),
        "mapping": mapping,
        "accepted_deltas": ACCEPTED_DELTAS,
    }

    out_path = os.path.join(_AQUI, "data", "do_corpus_retarget_baseline_2026_08_18.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {out_path}: corpus_size={len(strings)}, accepted_deltas={len(ACCEPTED_DELTAS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
