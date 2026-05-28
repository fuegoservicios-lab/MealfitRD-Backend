"""[P2-PROD-AUDIT-BUNDLE · 2026-05-27] Detecta deploy-lag por *edad* del KV
`app_kv_store.expected_last_known_pfix` respecto al `_LAST_KNOWN_PFIX` del
código local.

Contexto del modo de fallo que motiva este script (audit prod-readiness 2026-05-27):

    El cron `_alert_deploy_lag_marker_stale` (cron_tasks.py) compara `live_marker`
    (binario corriendo en prod) vs `expected_marker` (KV). Detecta drift binario,
    pero NO da señal cuando el operador trabaja local-only ([[user_workflow_local_only]])
    y la deuda de deploy se acumula.

    Snapshot 2026-05-27 23:55 UTC:
      live_marker (prod): "P1-RLS-INITPLAN · 2026-05-20"
      _LAST_KNOWN_PFIX (HEAD local): "P2-CRONS-HEALTH-AGGREGATE · 2026-05-27"
      delta: 7 días → cierre P0-DEAD-LETTER-USER-NOTIFY en local sigue sin propagar.

    El cron del backend NO puede emitir este warning (vive dentro del binario
    deployado; no ve HEAD local). El operador debe correr este check on-demand
    o como parte del SOP local-only periódico.

Uso operacional:

    # Reporte humano (lee SOLO el código local, sin Supabase):
    py backend/scripts/check_deploy_lag_age.py --local-only

    # Comparar contra KV de producción (requiere SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY
    # en `.env`):
    py backend/scripts/check_deploy_lag_age.py

    # Threshold custom:
    py backend/scripts/check_deploy_lag_age.py --max-age-days 7

Argumentos:
    --max-age-days N    Falla si `delta = local_date - kv_date > N` (default 14).
                        Knob env equivalente `MEALFIT_DEPLOY_LAG_MAX_AGE_DAYS`
                        (clamp [1, 365]).
    --local-only        No consulta Supabase. Solo reporta el marker local.
    --kv-marker STR     Override del marker remoto (testing/dry-run).
    --quiet             Suprime reporte OK.

Exit code:
    0 — delta dentro del umbral, o `--local-only` sin marker remoto.
    1 — delta excede umbral.
    2 — error (marker local mal formado, Supabase no alcanzable sin --local-only).

Diseño:
    NO depende del runtime backend. Solo Python 3.10+ + `httpx` (opcional, para el
    fetch del KV). Pensado para correr en local del operador o en un git hook.

Tooltip-anchor: P2-PROD-AUDIT-BUNDLE.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    """Backend root (`backend/`) calculado relativo al archivo del script."""
    return Path(__file__).resolve().parent.parent


def _parse_marker(marker: str) -> Optional[tuple[str, date]]:
    """Parsea `Pn-X · YYYY-MM-DD` o `Pn-X-... · YYYY-MM-DD`. Devuelve (prefix, date)
    o None si no matchea."""
    pattern = re.compile(
        r"^(?P<prefix>P\d+(?:-[A-Z0-9]+)+)\s+·\s+(?P<date>\d{4}-\d{2}-\d{2})$"
    )
    m = pattern.match(marker.strip())
    if not m:
        return None
    try:
        d = datetime.strptime(m.group("date"), "%Y-%m-%d").date()
    except ValueError:
        return None
    return (m.group("prefix"), d)


def _read_local_marker() -> Optional[str]:
    """Extrae `_LAST_KNOWN_PFIX = "..."` de `backend/app.py` vía regex sin
    importar el módulo (mismo patrón que test_p3_1_last_known_pfix_freshness)."""
    app_py = _repo_root() / "app.py"
    if not app_py.exists():
        return None
    text = app_py.read_text(encoding="utf-8")
    m = re.search(
        r'^_LAST_KNOWN_PFIX\s*=\s*["\'](?P<val>[^"\']+)["\']',
        text,
        re.MULTILINE,
    )
    return m.group("val") if m else None


def _fetch_remote_marker(timeout_s: float = 10.0) -> Optional[str]:
    """Lee `app_kv_store.expected_last_known_pfix` vía Supabase REST.

    Requiere env vars `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` (o
    `SUPABASE_KEY`). Devuelve None si falta config o el fetch falla — el
    caller decide si eso es exit 2 o degrade silencioso.
    """
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or ""
    )
    if not url or not key:
        return None

    try:
        import httpx  # type: ignore
    except ImportError:
        print(
            "[P2-PROD-AUDIT-BUNDLE] WARN: `httpx` no instalado. "
            "Para fetch del KV remoto: `pip install httpx`. "
            "Usa --local-only para skip el fetch.",
            file=sys.stderr,
        )
        return None

    try:
        resp = httpx.get(
            f"{url}/rest/v1/app_kv_store",
            params={
                "select": "value",
                "key": "eq.expected_last_known_pfix",
                "limit": 1,
            },
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            },
            timeout=timeout_s,
        )
    except Exception as exc:
        print(
            f"[P2-PROD-AUDIT-BUNDLE] WARN: fetch del KV falló: {exc}",
            file=sys.stderr,
        )
        return None

    if resp.status_code != 200:
        print(
            f"[P2-PROD-AUDIT-BUNDLE] WARN: KV REST devolvió {resp.status_code} — "
            f"verifica SUPABASE_SERVICE_ROLE_KEY.",
            file=sys.stderr,
        )
        return None

    rows = resp.json()
    if not rows:
        return None
    val = rows[0].get("value")
    # `value` es JSONB; cuando el contenido es string, Supabase lo devuelve como string
    # directo (no envuelto en lista). Tolera ambos casos.
    if isinstance(val, list) and val:
        val = val[0]
    return str(val) if val else None


def _clamp_max_age_days(raw: int) -> int:
    if raw < 1:
        return 1
    if raw > 365:
        return 365
    return raw


def main(argv: list[str]) -> int:
    env_default_raw = os.environ.get("MEALFIT_DEPLOY_LAG_MAX_AGE_DAYS", "14")
    try:
        env_default = _clamp_max_age_days(int(env_default_raw))
    except (TypeError, ValueError):
        env_default = 14

    parser = argparse.ArgumentParser(
        description=(
            "Falla con exit 1 si `_LAST_KNOWN_PFIX` (local) está N días "
            "más adelante que `app_kv_store.expected_last_known_pfix` (KV "
            "remoto). Pensado para el operador local-only que acumula "
            "deuda de deploy."
        )
    )
    parser.add_argument(
        "--max-age-days", type=int, default=env_default,
        help=(
            f"Umbral de delta en días entre local y KV. Default "
            f"{env_default} (env MEALFIT_DEPLOY_LAG_MAX_AGE_DAYS, "
            f"clamp [1, 365])."
        ),
    )
    parser.add_argument(
        "--local-only", action="store_true",
        help="No consulta Supabase. Solo reporta marker local.",
    )
    parser.add_argument(
        "--kv-marker", type=str, default=None,
        help="Override del marker remoto (testing/dry-run).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suprime reporte OK.",
    )
    args = parser.parse_args(argv)

    threshold = _clamp_max_age_days(int(args.max_age_days))

    # 1. Marker local desde backend/app.py
    local_marker_raw = _read_local_marker()
    if not local_marker_raw:
        print(
            "[P2-PROD-AUDIT-BUNDLE] ERROR: `_LAST_KNOWN_PFIX` no encontrado "
            "en `backend/app.py`. ¿Lo renombraron?",
            file=sys.stderr,
        )
        return 2
    local_parsed = _parse_marker(local_marker_raw)
    if not local_parsed:
        print(
            f"[P2-PROD-AUDIT-BUNDLE] ERROR: marker local mal formado: "
            f"`{local_marker_raw}`. Esperado `Pn-X · YYYY-MM-DD`.",
            file=sys.stderr,
        )
        return 2
    _, local_date = local_parsed

    # 2. Marker remoto
    if args.local_only:
        if not args.quiet:
            print(
                f"[P2-PROD-AUDIT-BUNDLE] LOCAL-ONLY: "
                f"local={local_marker_raw} (date={local_date}). "
                f"Skip fetch KV."
            )
        return 0

    remote_marker_raw = args.kv_marker if args.kv_marker else _fetch_remote_marker()
    if not remote_marker_raw:
        # Sin KV no podemos calcular delta. En modo no-quiet reportar; exit 2 para
        # que el caller decida (un git hook puede ignorar 2 si es local-only setup).
        print(
            "[P2-PROD-AUDIT-BUNDLE] ERROR: no se pudo leer "
            "`app_kv_store.expected_last_known_pfix` (¿faltan env vars "
            "SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY?). Usa --local-only "
            "o --kv-marker para dry-run.",
            file=sys.stderr,
        )
        return 2

    remote_parsed = _parse_marker(remote_marker_raw)
    if not remote_parsed:
        print(
            f"[P2-PROD-AUDIT-BUNDLE] ERROR: marker remoto mal formado: "
            f"`{remote_marker_raw}`.",
            file=sys.stderr,
        )
        return 2
    _, remote_date = remote_parsed

    # 3. Delta + decision
    delta_days = (local_date - remote_date).days
    over_threshold = delta_days > threshold

    sigil = "[WARN]" if over_threshold else "[info]"
    print(
        f"[P2-PROD-AUDIT-BUNDLE] local={local_marker_raw} ({local_date})  "
        f"remote={remote_marker_raw} ({remote_date})  "
        f"delta={delta_days}d  threshold={threshold}d  {sigil}"
    )

    if over_threshold:
        print(
            f"[P2-PROD-AUDIT-BUNDLE] FAIL: delta {delta_days}d > {threshold}d. "
            f"Deploy lag acumulado — considera redeploy + bump del KV "
            f"`expected_last_known_pfix` (script "
            f"`backend/scripts/publish_pfix_marker.py`).",
            file=sys.stderr,
        )
        return 1

    if not args.quiet:
        print(
            f"[P2-PROD-AUDIT-BUNDLE] OK (delta {delta_days}d within threshold "
            f"{threshold}d)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
