"""[P3-DEPLOY-LAG-UPSERT-MESSAGE · 2026-05-20] Test anti-regresión del
UPSERT del alert `deploy_lag_drift_vs_expected` — debe actualizar `message`
en el ON CONFLICT.

Bug observado:
    Cuando el cron `_alert_deploy_lag_marker_stale` re-disparaba el alert
    (porque la condición seguía existiendo), el ON CONFLICT actualizaba
    solo `triggered_at`, `metadata` y `resolved_at` — NO el `message`.
    Resultado: el campo `message` preservaba el texto del PRIMER trigger
    (con live/expected stale), confundiendo al SRE cuando debuggea.

    Observado 2026-05-20: el alert reabierto a las 04:01 mostraba
    "Producción reporta P3-RUNBOOK-CONSOLIDATION..." aunque el binary
    real ya tenía marker P1-CHAT-PROD-AUDIT — texto desactualizado.

Fix:
    Añadir `message = EXCLUDED.message` al UPDATE del ON CONFLICT.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_TASKS_PY = _BACKEND_ROOT / "cron_tasks.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_both_deploy_lag_alerts_update_message():
    """[P3-DEPLOY-LAG-UPSERT-MESSAGE] AMBOS INSERTs en cron_tasks.py
    (`deploy_lag_marker_stale` señal A + `deploy_drift` señal B) deben
    incluir `message = EXCLUDED.message` en el ON CONFLICT.

    Sin esto, el alert preserva texto stale del primer trigger cuando
    re-dispara, confundiendo al SRE durante debugging."""
    src = _read(_CRON_TASKS_PY)
    # Capturar el SET clause que viene DESPUÉS del alert_type específico de
    # deploy_lag. Match VALUES (..., 'deploy_xxx', ...) + ... + ON CONFLICT
    # ... DO UPDATE SET <captured>.
    deploy_clauses = re.findall(
        r"VALUES\s*\([^)]*'(?:deploy_lag_marker_stale|deploy_drift)'[^)]*\).*?"
        r"ON CONFLICT \(alert_key\) DO UPDATE\s+SET([^\"]+?)\"\"\"",
        src,
        re.DOTALL,
    )
    assert len(deploy_clauses) == 2, (
        f"Esperaba exactamente 2 SET clauses para deploy_lag (marker_stale + "
        f"drift); encontradas {len(deploy_clauses)}. Refactor inesperado de "
        f"`_alert_deploy_lag_marker_stale`. Ver P3-DEPLOY-LAG-UPSERT-MESSAGE."
    )
    for i, set_clause in enumerate(deploy_clauses):
        assert re.search(r"message\s*=\s*EXCLUDED\.message", set_clause), (
            f"SET clause deploy_lag #{i+1} ausente de "
            f"`message = EXCLUDED.message`. Sin esto, message queda stale "
            f"tras re-trigger. Ver P3-DEPLOY-LAG-UPSERT-MESSAGE · 2026-05-20.\n"
            f"SET clause actual: {set_clause.strip()[:200]}"
        )
        for field in ["triggered_at", "metadata", "resolved_at"]:
            assert field in set_clause, (
                f"SET clause deploy_lag #{i+1}: field `{field}` perdido — "
                f"refactor incompleto."
            )
