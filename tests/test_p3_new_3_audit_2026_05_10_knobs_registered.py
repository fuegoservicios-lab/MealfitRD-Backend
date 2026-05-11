"""[P3-NEW-3 · 2026-05-10] Lock-the-contract test: los knobs introducidos
por el audit del 2026-05-10 DEBEN aparecer en `_KNOBS_REGISTRY` tras
importar los módulos correspondientes.

Bug observado en el audit 2026-05-10:
    Dos knobs nuevos (`MEALFIT_SCHEDULER_JITTER_S` introducido por
    P0-NEW-2, `MEALFIT_INVENTORY_RPC_STRICT` introducido por P1-NEW-1)
    se leían inicialmente solo dentro de paths patológicos:
      - JITTER_S al module-level de cron_tasks.py (OK).
      - INVENTORY_RPC_STRICT solo dentro del except RPC fallback (gap
        cerrado por P2-NEW-2 con lectura top-level adicional).

    Sin un test que enumere ambos en el registry, una regresión que mueva
    la lectura JITTER_S de vuelta a lazy (función) o que elimine la
    pre-lectura RPC_STRICT pasaría sin disparar el test inventario P2-1
    (cuyo scope era los 5 módulos pre-2026-05-10).

Este test es complementario, NO duplicado: enumera específicamente los
2 knobs nuevos del audit 2026-05-10. Si en el futuro el audit cierra
otro batch de knobs, añadirlos aquí (snapshot incremental por audit).

Estrategia:
    1. Importar cron_tasks (registra JITTER_S al module-init).
    2. Importar db_inventory (registra INVENTORY_RPC_STRICT vía
       pre-lectura top-level de P2-NEW-2).
    3. Verificar ambos en `get_knobs_registry_snapshot()` con type correcto.
    4. Drift: si un knob nuevo se añade al audit 2026-05-10 (P2-NEW-2
       o futuros), añadir al dict `_AUDIT_2026_05_10_KNOBS`.
"""
from __future__ import annotations

import pytest


# Knobs introducidos por el audit 2026-05-10. Lista cerrada y revisada.
# Si añades un knob nuevo a backend que sea parte de la cohorte de fixes
# del audit, agrégalo aquí con su type esperado.
_AUDIT_2026_05_10_KNOBS = {
    # P0-NEW-2: jitter SSOT en cron_tasks.py al module-init de
    # `register_plan_chunk_scheduler` block (top-level vars cron_tasks.py).
    "MEALFIT_SCHEDULER_JITTER_S": "int",
    # P1-NEW-1 + P2-NEW-2: RPC strict mode + pre-lectura top-level en
    # db_inventory.py para que aparezca al import-time (no solo cuando
    # el except path ejecuta).
    "MEALFIT_INVENTORY_RPC_STRICT": "bool",
}


def _trigger_audit_2026_05_10_registrations() -> None:
    """Importa los módulos donde viven los knobs del audit.
    Tras esto el registry debe estar completo para esta cohorte.
    """
    try:
        import cron_tasks  # noqa: F401 — module-init registra JITTER_S
    except Exception as e:
        pytest.skip(
            f"cron_tasks no se puede importar en este entorno "
            f"(probable falta de env Supabase): {e}"
        )
    try:
        import db_inventory  # noqa: F401 — pre-lectura top-level registra RPC_STRICT
    except Exception as e:
        pytest.skip(
            f"db_inventory no se puede importar en este entorno: {e}"
        )


def test_audit_2026_05_10_knobs_all_in_registry() -> None:
    """Los 2 knobs introducidos por P0-NEW-2 y P1-NEW-1/P2-NEW-2 deben
    estar en `_KNOBS_REGISTRY` tras importar sus módulos.
    """
    _trigger_audit_2026_05_10_registrations()

    try:
        from knobs import get_knobs_registry_snapshot
    except Exception as e:
        pytest.skip(f"knobs.get_knobs_registry_snapshot no importable: {e}")

    snap = get_knobs_registry_snapshot()
    assert isinstance(snap, dict), (
        f"P3-NEW-3: snap no es dict ({type(snap).__name__})."
    )

    missing = []
    wrong_type = []
    for name, expected_type in _AUDIT_2026_05_10_KNOBS.items():
        if name not in snap:
            missing.append(name)
            continue
        actual_type = snap[name].get("type")
        if actual_type != expected_type:
            wrong_type.append(
                f"{name} (esperado={expected_type}, actual={actual_type!r})"
            )

    if missing or wrong_type:
        msg = ["P3-NEW-3 regresión: contrato del audit 2026-05-10 roto."]
        if missing:
            msg.append(
                "\nKnobs NO encontrados en `_KNOBS_REGISTRY`:\n  - "
                + "\n  - ".join(missing)
                + "\n\nProbable causa: la lectura top-level se movió a una "
                "función lazy o se eliminó. Sin top-level read, el knob "
                "no aparece en `/admin/knobs` ni `/health/version` hasta "
                "que el path patológico ejecute — exactamente lo que "
                "P0-NEW-2/P2-NEW-2 cerraron."
            )
        if wrong_type:
            msg.append(
                "\nKnobs con tipo incorrecto:\n  - "
                + "\n  - ".join(wrong_type)
                + "\n\nProbable causa: alguien cambió `_env_int` ↔ `_env_bool` "
                "o `_env_str` sin actualizar este test. Cambiar el type "
                "rompe el contrato (ej: bool → int significa que el knob "
                "ya no acepta `true`/`false` strings)."
            )
        pytest.fail("".join(msg))


def test_audit_2026_05_10_knob_defaults_documented_in_memories() -> None:
    """Drift detection: cada knob del audit debe tener su default
    documentado en alguna memoria del proyecto.

    Si añades un knob al audit, asegúrate que tu memoria menciona el
    default (CLAUDE.md o un memoria entry); sin doc, un operador no
    sabe qué valor pasar para apagar/activar.
    """
    import os
    from pathlib import Path

    memory_dir = Path.home() / ".claude" / "projects" / (
        "c--Users-angel-OneDrive-Escritorio-MealfitRD-IA"
    ) / "memory"
    if not memory_dir.exists():
        pytest.skip(f"Memory dir no existe: {memory_dir}")

    # Concatenar todo el contenido de las memorias del audit 2026-05-10
    # (filename patrón `project_p{0,1,2,3}_new_*_2026_05_10*.md`).
    relevant = []
    for f in memory_dir.glob("project_p*_new_*_2026_05_10*.md"):
        relevant.append(f.read_text(encoding="utf-8"))
    combined = "\n".join(relevant)

    undocumented = []
    for name in _AUDIT_2026_05_10_KNOBS:
        if name not in combined:
            undocumented.append(name)

    assert not undocumented, (
        f"P3-NEW-3: knobs sin doc en memorias del audit 2026-05-10: "
        f"{undocumented}. Cada knob debe aparecer en al menos una "
        f"memoria explicando su default + cuándo flipearlo. Sin doc, un "
        f"operador no puede usar el knob como kill switch."
    )
