"""[P3-RECOVERY-BACKEND-DOWN-EXIT · 2026-05-16] Cuando el backend muere
durante el recovery polling, el frontend debe DETECTAR consecutive failures
y salir del loading screen con toast de error, NO quedarse colgado indefinidamente.

Síntoma reportado por usuario:
> "cuando apague el backend manualmente siguio en la pantalla de carga
> y debio sacarme"

Pre-fix: `fetchPendingStatus()` retorna null en network error / 401.
checkOnce hacía `if (!status) return;` (sale temprano sin acción) →
polling continuaba indefinidamente cada 10s sin progreso. User stuck.

Post-fix: counter `consecutiveFailuresRef` incrementa por cada null.
Al alcanzar `_FAIL_THRESHOLD` (6 polls = 60s), asumimos backend down,
clearPendingFlag + toast error + navigate a /assessment si estaba en /plan.
Reset del counter cuando una poll succeed (backend volvió).
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_RECOVERY = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "components" / "PendingPipelineRecovery.jsx"
).read_text(encoding="utf-8")


def test_marker_present():
    assert "P3-RECOVERY-BACKEND-DOWN-EXIT" in _RECOVERY, (
        "Marker P3-RECOVERY-BACKEND-DOWN-EXIT ausente — un refactor podría "
        "borrar el exit y volver al loop infinito de polling."
    )


def test_fail_threshold_constant_declared():
    """`_FAIL_THRESHOLD` debe estar declarado en rango razonable [3, 30].
    <3 = falso positivo en blip de red; >30 = ~5min de espera, demasiado."""
    m = re.search(r"_FAIL_THRESHOLD\s*=\s*(\d+)", _RECOVERY)
    assert m, "Constante `_FAIL_THRESHOLD` no declarada."
    n = int(m.group(1))
    assert 3 <= n <= 30, (
        f"_FAIL_THRESHOLD={n} fuera de rango [3, 30]. "
        "3 polls × 10s = 30s mínimo (cubre blips); 30 polls × 10s = 5min máximo."
    )


def test_failure_counter_ref_declared():
    """`consecutiveFailuresRef` debe ser un `useRef(0)` para persistir
    el counter entre re-renders sin disparar re-render."""
    assert "consecutiveFailuresRef = useRef(0)" in _RECOVERY, (
        "Ref `consecutiveFailuresRef` no declarado con `useRef(0)`. "
        "Sin él, el counter se resetea en cada re-render → nunca alcanza "
        "threshold → loop infinito."
    )


def test_counter_increments_on_failure():
    """En el branch `if (!status)`, el counter DEBE incrementarse antes
    de chequear el threshold."""
    # Slice del checkOnce
    idx = _RECOVERY.find("async function checkOnce")
    assert idx > 0
    end = _RECOVERY.find("\n        }\n", idx + 100)  # cierre de la función
    body = _RECOVERY[idx:end if end > 0 else idx + 3000]

    assert "consecutiveFailuresRef.current += 1" in body, (
        "Counter no se incrementa en el branch `if (!status)`. Sin esto, "
        "nunca alcanza threshold → loop indefinido."
    )


def test_counter_resets_on_success():
    """Cuando una poll succeed, el counter debe resetearse. Sin esto,
    un blip transient + recovery durarían sumando hacia el threshold."""
    idx = _RECOVERY.find("async function checkOnce")
    body = _RECOVERY[idx:idx + 3000]
    assert "consecutiveFailuresRef.current = 0" in body, (
        "Counter no se resetea tras una poll exitosa. Sin reset, un blip "
        "transient eventualmente activa el exit aunque backend esté vivo."
    )


def test_threshold_triggers_exit():
    """Al alcanzar threshold, debe: (a) marcar handledRef, (b) limpiar
    flag local, (c) mostrar toast error, (d) navegar si en /plan."""
    idx = _RECOVERY.find("async function checkOnce")
    body = _RECOVERY[idx:idx + 4000]

    # Buscar el bloque del threshold check
    pat = re.compile(
        r"consecutiveFailuresRef\.current\s*>=\s*_FAIL_THRESHOLD",
    )
    assert pat.search(body), (
        "Check `consecutiveFailuresRef.current >= _FAIL_THRESHOLD` no encontrado."
    )

    # Dentro del bloque debe haber:
    # - handledRef.current = true (detener polling)
    # - clearPendingFlag() (no recovery espurio en próximo mount)
    # - toast.error (notificar al user)
    # - navigate (sacar del loading screen)
    threshold_idx = body.find("_FAIL_THRESHOLD")
    threshold_block = body[threshold_idx:threshold_idx + 2000]

    assert "handledRef.current = true" in threshold_block, (
        "Tras threshold, `handledRef.current=true` ausente — polling sigue corriendo."
    )
    assert "clearPendingFlag()" in threshold_block, (
        "Tras threshold, `clearPendingFlag()` ausente — flag stale dispararía "
        "recovery espurio en el próximo mount."
    )
    assert "toast.error" in threshold_block, (
        "Tras threshold, no se muestra toast.error — user no sabe qué pasó."
    )
    assert "navigate(" in threshold_block, (
        "Tras threshold, no hay navigate — user queda en /plan sin loading "
        "(handledRef detiene polling pero no lo saca del LoadingScreen)."
    )


def test_diagnostic_logs_removed():
    """Los `console.error` con `[P3-RECOVERY-DEBUG]` fueron temporales para
    diagnóstico. Deben estar removidos en producción."""
    assert "P3-RECOVERY-DEBUG" not in _RECOVERY, (
        "Diagnostic logs `[P3-RECOVERY-DEBUG]` siguen presentes. Esos eran "
        "temporales solo para identificar el bug; ahora que está resuelto, "
        "deben ser removidos (no spam de logs en prod)."
    )
