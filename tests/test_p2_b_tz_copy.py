"""[P2-B-TZ-COPY · 2026-05-10] El push de
`_escalate_unrecoverable_chunk` DEBE tener copy específico para
`escalation_reason="unrecoverable_tz_unresolved"`.

Bug original (audit 2026-05-10):
    `unrecoverable_tz_unresolved` está en `ESCALATION_REASONS`
    (constants.py:2277) — razón canónica registrada y usada en
    cron_tasks.py:7880 cuando un chunk dead-lettera porque el plan
    tenía `tzOffset=NULL` y la live TZ no fue resoluble tras N
    attempts del recovery cron.

    Sin embargo, la cadena `if/elif` que mapea `escalation_reason →
    {title, body, cta, url}` (cron_tasks.py:8749-8789) solo cubría 3
    razones explícitas + el `else` default. Para `unrecoverable_tz_unresolved`
    el push mostraba "Tu plan necesita atención" (copy genérico de
    `recovery_exhausted`) y el deeplink iba a `/dashboard?recovery_exhausted=1`,
    sin contexto de que la causa era zona horaria.

    Resultado UX: el usuario no sabía si arreglar la TZ del dispositivo,
    actualizar la app, o regenerar — el push genérico no ofrecía pistas.

Fix (P2-B · 2026-05-10):
    Añadido `elif escalation_reason == "unrecoverable_tz_unresolved"` en
    cron_tasks.py con:
      - title: "Tu plan necesita regenerarse"
      - body: explica que no se pudo confirmar TZ y que el plan se
              pausó para no generar días desfasados
      - cta: "Regenerar plan"
      - url: `/dashboard?action_required=tz_unresolved` (deeplink dedicado
             para que el frontend pueda mostrar el banner correcto)

Cobertura de este test (parser-based, no DB):
    1. La cadena if/elif en `_escalate_unrecoverable_chunk` cubre
       `unrecoverable_tz_unresolved` explícitamente.
    2. El copy contiene el deeplink dedicado `action_required=tz_unresolved`.
    3. Cada razón en `ESCALATION_REASONS` (excepto el default
       `recovery_exhausted` que cae al `else`) tiene un branch dedicado
       — meta-test que detecta razones futuras añadidas a la whitelist
       sin actualizar el copy mapping.

Out of scope:
    - Test E2E del push real (Firebase): el copy se construye en backend
      y se envía via push_log; este test asegura que el branch existe.
    - Cobertura del banner en frontend (Dashboard.jsx) que lea
      `_user_action_required.url` y matche `tz_unresolved`: queda como
      mejora UX si se observa confusión del usuario en producción.
"""
from __future__ import annotations

import inspect
import re

import pytest


def _get_escalation_source() -> str:
    """Source de `_escalate_unrecoverable_chunk` vía `inspect.getsource`
    — más robusto que regex contra docstrings que mencionan SQL como prosa."""
    from cron_tasks import _escalate_unrecoverable_chunk
    return inspect.getsource(_escalate_unrecoverable_chunk)


def _get_escalation_reasons() -> tuple[str, ...]:
    """Importa el tuple canónico desde constants — single source of truth."""
    from constants import ESCALATION_REASONS
    return tuple(ESCALATION_REASONS)


# ---------------------------------------------------------------------------
# 1. Branch explícito para unrecoverable_tz_unresolved.
# ---------------------------------------------------------------------------
def test_escalation_has_explicit_branch_for_tz_unresolved():
    """`unrecoverable_tz_unresolved` debe tener su propio `elif`.
    Si cae al default, el push muestra copy genérico sin contexto de TZ."""
    src = _get_escalation_source()

    # Patrón: `elif escalation_reason == "unrecoverable_tz_unresolved":`.
    has_branch = bool(re.search(
        r'elif\s+escalation_reason\s*==\s*[\'"]unrecoverable_tz_unresolved[\'"]\s*:',
        src,
    ))
    assert has_branch, (
        "El bloque if/elif que mapea escalation_reason → copy DEBE tener "
        "un branch explícito para `unrecoverable_tz_unresolved`. Sin él, "
        "el push cae al default (`recovery_exhausted`) y el usuario no "
        "sabe que la causa fue zona horaria — peor UX que genérica."
    )


# ---------------------------------------------------------------------------
# 2. Deeplink dedicado.
# ---------------------------------------------------------------------------
def test_tz_branch_uses_dedicated_deeplink():
    """El copy debe incluir el deeplink `action_required=tz_unresolved`
    para que el frontend pueda diferenciar el banner de los demás
    `action_required=*`."""
    src = _get_escalation_source()

    assert "action_required=tz_unresolved" in src, (
        "El branch de `unrecoverable_tz_unresolved` debe usar el deeplink "
        "`action_required=tz_unresolved` (no `recovery_exhausted=1` ni "
        "ningún otro). Esto permite que el frontend diferencie el banner "
        "de los demás motivos de dead-letter."
    )


# ---------------------------------------------------------------------------
# 3. Meta-test: cada ESCALATION_REASON tiene branch o cae al default
#    documentado (recovery_exhausted).
# ---------------------------------------------------------------------------
def test_every_escalation_reason_has_copy_branch_or_falls_to_default():
    """Drift detection: si alguien añade una razón a `ESCALATION_REASONS`
    sin actualizar el if/elif del copy, esta razón cae al default
    genérico — UX degradada silenciosamente. Este test detecta el drift.

    `recovery_exhausted` es la única razón cuyo copy vive en el `else`
    del if/elif (default explícito documentado). Cualquier otra razón
    DEBE tener su propio branch.
    """
    reasons = _get_escalation_reasons()
    assert reasons, "ESCALATION_REASONS parece vacío."

    src = _get_escalation_source()

    # `recovery_exhausted` cae al else default — exento del branch
    # explícito por convención documentada.
    DEFAULT_REASON = "recovery_exhausted"
    missing_branches = []
    for reason in reasons:
        if reason == DEFAULT_REASON:
            continue
        # Aceptamos `if` o `elif`: el primer branch del chain usa `if`,
        # los siguientes `elif`. Cualquiera satisface el contrato
        # "branch dedicado".
        pattern = (
            rf'(?:^|\s)(?:if|elif)\s+escalation_reason\s*==\s*'
            rf'[\'"]{re.escape(reason)}[\'"]\s*:'
        )
        if not re.search(pattern, src, re.MULTILINE):
            missing_branches.append(reason)

    assert not missing_branches, (
        f"Razones en ESCALATION_REASONS sin branch de copy en "
        f"`_escalate_unrecoverable_chunk`: {missing_branches}. "
        f"Cualquier razón distinta a `{DEFAULT_REASON}` cae al default "
        f"genérico ('Tu plan necesita atención') si no tiene su propio "
        f"`elif`. Añadir un elif con title/body/cta/url específicos para "
        f"que el push tenga contexto del problema real."
    )


# ---------------------------------------------------------------------------
# 4. Sanity del whitelist (cross-check con P2-NEW-3).
# ---------------------------------------------------------------------------
def test_tz_unresolved_in_canonical_whitelist():
    """`unrecoverable_tz_unresolved` debe estar en `ESCALATION_REASONS`
    (P2-NEW-3 valida at-entry en `_escalate_unrecoverable_chunk`). Si
    desaparece del whitelist, el copy nuevo sería inalcanzable."""
    reasons = _get_escalation_reasons()
    assert "unrecoverable_tz_unresolved" in reasons, (
        "`unrecoverable_tz_unresolved` debe estar en ESCALATION_REASONS. "
        "Si lo removiste del whitelist, también remueve el branch del "
        "copy en `_escalate_unrecoverable_chunk` (sino queda código muerto)."
    )
