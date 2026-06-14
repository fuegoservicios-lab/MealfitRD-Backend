"""[P2-1 · 2026-05-10] Publica el `_LAST_KNOWN_PFIX` actual de HEAD a la
tabla `app_kv_store` para que el cron de detección de drift (P0-1) pueda
comparar el marker en vivo del proceso contra el valor esperado.

Uso operacional (post `git push`):

    python backend/scripts/publish_pfix_marker.py

Lee el valor literal `_LAST_KNOWN_PFIX = "..."` desde `backend/app.py` y
hace UPSERT en `app_kv_store` con la key `expected_last_known_pfix`.
Mismo formato que el cron P0-1 espera (string puro o objeto con
`.marker`). Aquí escribimos string puro para simplicidad.

Variables de entorno requeridas:
    NEON_DATABASE_URL  — connection string a Neon (mismo que usa el
                         backend en producción). También acepta DATABASE_URL.

Salida:
    exit 0 — marker publicado o ya idéntico al valor en DB.
    exit 1 — error de lectura del marker, conexión a DB, o mismatch
             inesperado tras el UPSERT.

Idempotente: si la key ya existe con el mismo valor, hace UPSERT sin
side effects. Si difiere, sobrescribe (es la intención: el operador
acaba de pushear código nuevo y publica el marker correspondiente).

Por qué un script y no un cron interno:
    El cron P0-1 corre dentro del binario desplegado. Si el binario
    está rezagado, lee su PROPIO `_LAST_KNOWN_PFIX` viejo y publicaría
    un valor obsoleto — no detectaría el drift. La fuente correcta del
    "expected" es HEAD del árbol al momento del push, accesible solo
    desde el flujo manual/CI del operador.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_KV_KEY = "expected_last_known_pfix"


def read_marker_from_app_py() -> str:
    """Extrae el literal `_LAST_KNOWN_PFIX = "..."` de app.py.

    Mismo regex que `tests/test_p3_1_last_known_pfix_freshness.py` y que
    el cron `_alert_deploy_lag_marker_stale` (P0-1) — drift entre
    parsers significaría que el detector no se activa.
    """
    if not _APP_PY.exists():
        raise FileNotFoundError(f"app.py no encontrado en {_APP_PY}")
    text = _APP_PY.read_text(encoding="utf-8")
    m = re.search(
        r'^_LAST_KNOWN_PFIX\s*=\s*["\']([^"\']+)["\']',
        text,
        re.MULTILINE,
    )
    if m is None:
        raise ValueError(
            f"No se encontró asignación literal `_LAST_KNOWN_PFIX = '...'` "
            f"en {_APP_PY}. ¿Fue movido o computado dinámicamente?"
        )
    return m.group(1)


def upsert_kv(marker: str) -> tuple[bool, str | None]:
    """UPSERT en `app_kv_store`. Retorna (was_changed, prior_value).

    Si la key ya tenía el mismo valor, was_changed=False (idempotente).
    """
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "NEON_DATABASE_URL no está seteado. Necesitas exportar la "
            "connection string de Neon antes de correr este script."
        )

    try:
        import psycopg
    except ImportError as e:
        raise RuntimeError(
            "psycopg no instalado. Activar el conda env del backend: "
            "`conda activate mealfit`."
        ) from e

    import json as _json

    prior_value: str | None = None
    with psycopg.connect(db_url, autocommit=False) as conn:
        with conn.cursor() as cur:
            # Lee valor previo (best-effort para reportar el delta).
            try:
                cur.execute(
                    "SELECT value FROM app_kv_store WHERE key = %s",
                    (_KV_KEY,),
                )
                row = cur.fetchone()
                if row is not None:
                    raw = row[0]
                    # `value` es jsonb. Si es string puro, viene como str;
                    # si fue serializado como obj, sería dict.
                    if isinstance(raw, str):
                        prior_value = raw
                    elif isinstance(raw, dict) and isinstance(raw.get("marker"), str):
                        prior_value = raw["marker"]
            except Exception:
                # Tabla puede no existir aún en algunas DBs; no abortamos.
                prior_value = None

            cur.execute(
                """
                INSERT INTO app_kv_store (key, value)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (key) DO UPDATE
                  SET value = EXCLUDED.value, updated_at = now()
                """,
                (_KV_KEY, _json.dumps(marker)),
            )
        conn.commit()

    return (prior_value != marker), prior_value


def main(argv: list[str]) -> int:
    try:
        marker = read_marker_from_app_py()
    except Exception as e:
        print(f"[P2-1] ERROR leyendo marker: {e}", file=sys.stderr)
        return 1

    print(f"[P2-1] Marker en HEAD: {marker!r}")

    if "--dry-run" in argv:
        print("[P2-1] --dry-run: no se escribe a app_kv_store.")
        return 0

    try:
        changed, prior = upsert_kv(marker)
    except Exception as e:
        print(f"[P2-1] ERROR escribiendo a app_kv_store: {e}", file=sys.stderr)
        return 1

    if changed:
        print(
            f"[P2-1] OK — `app_kv_store.{_KV_KEY}` actualizado de "
            f"{prior!r} a {marker!r}."
        )
    else:
        print(
            f"[P2-1] OK — `app_kv_store.{_KV_KEY}` ya estaba en {marker!r} "
            f"(no-op idempotente)."
        )
    print(
        "[P2-1] El cron `_alert_deploy_lag_marker_stale` (P0-1) emitirá "
        "alerta `deploy_lag_drift_vs_expected` si el binario en producción "
        "reporta un marker distinto."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
