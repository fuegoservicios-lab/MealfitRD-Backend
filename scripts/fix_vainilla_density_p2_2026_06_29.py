# [P2-SEASONING-RESTOCK-CLEAR · 2026-06-29] Data patch (data-quality, complementa el fix de CÓDIGO).
#
# Contexto: tras un restock, "Vainilla" reaparecía sola en la lista de compras. La causa REAL la cierra el
# fix de código en `shopping_calculator.aggregate_and_deduct_shopping_list` (SEASONING-CATALOG-KEEP ahora
# salta el condimento si tu Nevera ya lo cubre). Este script es el complemento de DATA: Vainilla tenía
# `density_g_per_unit = NULL` (aunque `container_weight_g=148` + `density_g_per_cup=208` ya estaban). Sin
# `density_g_per_unit`, una cantidad en unidad "botella" no se convierte a peso → la deducción/clamp por peso
# no la concilia en escenarios con peso. Lo fijamos a 148 (1 botella de 148 ml ≈ 148 g), consistente con
# `container_weight_g`. Idempotente (solo toca filas con `density_g_per_unit IS NULL`), dry-run por default.
#
# Uso (en el VPS, env mealfit):
#   python scripts/fix_vainilla_density_p2_2026_06_29.py            # dry-run (solo muestra)
#   python scripts/fix_vainilla_density_p2_2026_06_29.py --commit   # aplica el UPDATE
import os
import sys

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
import psycopg  # noqa: E402

_TARGET = "Vainilla"
_DENSITY_G_PER_UNIT = 148.0  # 1 botella (148 ml) ≈ 148 g (== container_weight_g)
COMMIT = "--commit" in sys.argv


def main() -> None:
    url = os.environ["NEON_DATABASE_URL"].strip().strip("'").strip('"')
    with psycopg.connect(url) as conn:
        cur = conn.execute(
            "SELECT name, default_unit, density_g_per_unit, container_weight_g "
            "FROM master_ingredients WHERE name = %s",
            (_TARGET,),
        )
        row = cur.fetchone()
        if not row:
            print(f"[fix-vainilla-density] '{_TARGET}' no encontrado en master_ingredients — no-op.")
            return
        name, unit, dgpu, cwg = row
        print(f"[fix-vainilla-density] ANTES: name={name} default_unit={unit} "
              f"density_g_per_unit={dgpu} container_weight_g={cwg}")
        if dgpu is not None:
            print("[fix-vainilla-density] density_g_per_unit ya está poblado → no-op (idempotente).")
            return
        if not COMMIT:
            print(f"[fix-vainilla-density] DRY-RUN: setearía density_g_per_unit={_DENSITY_G_PER_UNIT}. "
                  f"Re-corre con --commit para aplicar.")
            return
        conn.execute(
            "UPDATE master_ingredients SET density_g_per_unit = %s "
            "WHERE name = %s AND density_g_per_unit IS NULL",
            (_DENSITY_G_PER_UNIT, _TARGET),
        )
        conn.commit()
        after = conn.execute(
            "SELECT density_g_per_unit FROM master_ingredients WHERE name = %s", (_TARGET,)
        ).fetchone()
        print(f"[fix-vainilla-density] DESPUÉS: density_g_per_unit={after[0]} — UPDATE aplicado.")


if __name__ == "__main__":
    main()
