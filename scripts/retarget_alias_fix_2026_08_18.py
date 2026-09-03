#!/usr/bin/env python
"""[P1-COUNTRY-SYSTEM-F2 · ola final (review de fase) · 2026-08-18 · C3] Quita 4 alias BARE de
filas nuevas de Fase 2 que colisionaban con vocabulario DO-reachable pre-existente
(`master_ingredients`, Neon):

    Fila (Fase 2)        Alias removido    Retargeteaba (bug)         Dueño pre-fase / correcto
    -------------------- ----------------- --------------------------- --------------------------
    Chicharrón (CO, T6)  'chicharron'      bare 'chicharron' -> fila   redundante: el NOMBRE
                                            (ya cubierto por el         canónico de la fila ya
                                            propio nombre)              provee el mismo match
    Cordero (ES, T5)     'lamb'            'lambí'/'lambi' (fuzzy      NINGUNO -- pre-fase quedaba
                                            0.889 >= umbral 0.87)      sin resolver (drop), 'lambí'
                                            -> 'Cordero'               es un molusco (lambí/carrucho),
                                                                       no cordero
    Requesón (ES, T5)    'ricotta'         'ricotta' -> 'Requesón'    'Queso ricotta' (fila PRICED
                                                                       pre-fase que lo tenía primero)
    Judías pintas (ES,T5) 'pinto beans'    'pinto beans' -> 'Judías   'Frijoles pintos' (fila PRICED
                                            pintas'                   pre-fase que lo tenía primero)

Los 4 son alias BARE genuinamente redundantes o colisionantes -- las filas conservan sus alias
propios (requeson/requesón, judias pintas/judías pintas/alubias pintas, cordero/carne de
cordero/pierna de cordero, chicharrones/cuero de cerdo frito/torreznos). El caso 'Chicharrón de
pollo' (retargeteo separado, related pero NO resuelto por este script porque el nombre canónico de
la fila sigue matcheando via CONTAINS incluso sin el alias) se cierra con un guard temprano en
`shopping_calculator.normalize_name` (ver comentario `C3.1` ahí), no aquí.

IDEMPOTENTE: si el alias ya no está presente, reporta "sin cambios" y salta -- re-correr no rompe
nada. Actualiza TANTO la DB (UPDATE aliases) COMO los JSON SSOT de origen
(`scripts/data/new_foods_es_2026_08_17.json`, `scripts/data/new_foods_mx_co_2026_08_17.json`) para
que un futuro re-seed desde JSON no reintroduzca el alias retirado.

USO:
    cd backend
    python scripts/retarget_alias_fix_2026_08_18.py              # DRY-RUN
    python scripts/retarget_alias_fix_2026_08_18.py --commit      # aplica de verdad

[P2-LOGGER-EXEMPT: script CLI one-shot, la salida a stdout ES el producto]
"""
import json
import os
import sys

try:
    from dotenv import load_dotenv
    for _p in (os.path.join(os.path.dirname(__file__), "..", ".env"),
               os.path.join(os.getcwd(), ".env"), "/opt/mealfit/backend/.env"):
        if os.path.exists(_p):
            load_dotenv(_p)
            break
except Exception:
    pass

import psycopg

_AQUI = os.path.dirname(os.path.abspath(__file__))
_NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv

# (nombre de fila, alias bare a remover, JSON SSOT que la originó)
_REMOVALS = [
    ("Chicharrón", "chicharron", "new_foods_mx_co_2026_08_17.json"),
    ("Cordero", "lamb", "new_foods_es_2026_08_17.json"),
    ("Requesón", "ricotta", "new_foods_es_2026_08_17.json"),
    ("Judías pintas", "pinto beans", "new_foods_es_2026_08_17.json"),
]


def _fix_json_ssot(json_filename: str, name: str, bad_alias: str) -> bool:
    """Retira `bad_alias` del registro `name` en el JSON SSOT. Retorna True si cambió algo."""
    path = os.path.join(_AQUI, "data", json_filename)
    if not os.path.exists(path):
        print(f"  ! JSON SSOT no encontrado: {path} (salteando sync de archivo)")
        return False
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    changed = False
    for rec in data:
        if rec.get("name") == name and bad_alias in (rec.get("aliases") or []):
            rec["aliases"] = [a for a in rec["aliases"] if a != bad_alias]
            changed = True
    if changed and COMMIT:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return changed


def main():
    if not _NEON:
        print("FATAL: NEON url ausente", file=sys.stderr)
        return 1

    with psycopg.connect(_NEON) as conn:
        cambios = sin_cambios = 0
        for name, bad_alias, json_file in _REMOVALS:
            row = conn.execute(
                "SELECT aliases FROM public.master_ingredients WHERE name = %s", [name]
            ).fetchone()
            if row is None:
                print(f"  ! FILA NO ENCONTRADA: {name!r} -- saltando")
                continue
            aliases = list(row[0] or [])
            if bad_alias not in aliases:
                print(f"  ~ SIN CAMBIOS: {name!r} ya no tiene el alias {bad_alias!r} ({aliases})")
                sin_cambios += 1
                continue
            nuevos = [a for a in aliases if a != bad_alias]
            print(f"  {'DIFF' if not COMMIT else 'APLICANDO'}: {name!r} aliases "
                  f"{aliases!r} -> {nuevos!r} (removido: {bad_alias!r})")
            if COMMIT:
                conn.execute(
                    "UPDATE public.master_ingredients SET aliases = %s WHERE name = %s",
                    [nuevos, name],
                )
            json_changed = _fix_json_ssot(json_file, name, bad_alias)
            print(f"      JSON SSOT ({json_file}): "
                  f"{'actualizado' if json_changed and COMMIT else ('DIFF pendiente' if json_changed else 'sin diff')}")
            cambios += 1
        if COMMIT:
            conn.commit()
            print(f"\nCOMMITTED. cambios={cambios}, sin-cambios={sin_cambios}")
        else:
            print(f"\nDRY-RUN. cambios-a-aplicar={cambios}, sin-cambios={sin_cambios}. Re-corre con --commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
