"""[P1-PANTRY-RECONCILIATION · 2026-08-07] La Nevera pregunta en vez de adivinar.

La regla del producto es "la Nevera solo baja por lo que el usuario registra",
y es la correcta. Su consecuencia inevitable: lo que come sin registrar NUNCA
sale, y a las 2-3 semanas la Nevera sobre-reporta y la lista de compras
sub-compra.

El arreglo NO es descontar automatico — eso rompe la regla y devuelve al
problema que P1-PANTRY-NAME-RESOLUTION cerro (mover la Nevera por algo que el
usuario no puede auditar). El arreglo es PREGUNTAR.

Lo que este archivo protege:

  1. La señal de "quieto" NO es solo `updated_at`. Esa columna no se mantiene
     en el camino principal de descuento, asi que sola preguntaria por comida
     que SI se uso.
  2. El lote esta capeado. Una lista de 40 preguntas no se contesta, se ignora.
  3. `spoiled` y `used` no se colapsan: el desperdicio es informacion de
     COMPRA, no de consumo.
  4. Nada se mueve sin respuesta humana.
"""
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_MIGRATION = _BACKEND / "migrations" / "p1_pantry_reconciliation_2026_08_07.sql"
_INVENTORY = _BACKEND / "db_inventory.py"
_PLANS = _BACKEND / "routers" / "plans.py"

_UID = "11111111-1111-4111-8111-111111111111"


def _ago(days):
    return datetime.now(timezone.utc) - timedelta(days=days)


# ---------------------------------------------------------------------------
# 1. Migración
# ---------------------------------------------------------------------------

def test_migration_extends_the_existing_ledger():
    """Una tabla paralela obligaria a unir dos fuentes para contestar la MISMA
    pregunta ("¿que movio mi Nevera?") — el patron de duplicacion que en este
    repo termina drifteando."""
    sql = _MIGRATION.read_text(encoding="utf-8")
    assert "ALTER TABLE public.inventory_consumption_events" in sql
    assert "CREATE TABLE" not in sql, "no debe crear una tabla paralela al ledger"
    assert "'spoiled'" in sql and "'reconciliation'" in sql


def _sql_body(path: Path) -> str:
    """SQL sin líneas de comentario.

    Contar sobre el fuente crudo mide también la cabecera explicativa (que cita
    las mismas sentencias para justificarlas) y da falsos positivos.
    """
    return "\n".join(
        l for l in path.read_text(encoding="utf-8").splitlines()
        if not l.lstrip().startswith("--")
    )


def test_migration_is_idempotent():
    """Un CHECK no admite IF NOT EXISTS: se recrea con DROP ... IF EXISTS."""
    sql = _sql_body(_MIGRATION)
    assert sql.count("DROP CONSTRAINT IF EXISTS") == 2
    assert sql.count("ADD CONSTRAINT") == 2
    sql = _MIGRATION.read_text(encoding="utf-8")
    assert "RAISE EXCEPTION" in sql
    assert "BEGIN;" in sql and sql.rstrip().endswith("COMMIT;")


def test_migration_keeps_the_old_outcomes():
    """Añadir `spoiled` no puede tirar los existentes: filas historicas del
    ledger violarian el CHECK nuevo y el ALTER fallaria."""
    sql = _MIGRATION.read_text(encoding="utf-8")
    for viejo in ("deducted", "inferred", "not_in_pantry", "failed"):
        assert f"'{viejo}'" in sql
    for viejo in ("chat", "photo", "plan_meal", "chunk_reconcile", "agent_tool", "unknown"):
        assert f"'{viejo}'" in sql


# ---------------------------------------------------------------------------
# 2. La señal: no solo updated_at
# ---------------------------------------------------------------------------

def test_signal_combines_three_independent_sources():
    """`updated_at` sola preguntaria por comida que SI se uso: el RPC
    `apply_inventory_delta` no la toca y no hay trigger BEFORE UPDATE."""
    src = _INVENTORY.read_text(encoding="utf-8")
    body = src[src.index("def get_reconciliation_candidates"):]
    body = body[:body.index("\n_RECONCILE_ACTIONS")]
    assert "GREATEST(" in body
    assert "ui.created_at" in body
    assert "ui.updated_at" in body
    assert "inventory_consumption_events" in body, (
        "sin el ledger la señal es ciega al consumo — que es justo lo que la "
        "reconciliacion quiere detectar."
    )


def test_anchor_rationale_is_documented_in_source():
    """Anclaje del POR QUÉ: si alguien 'simplifica' a `updated_at` sola, el
    comentario que lo desaconseja debe seguir ahi para que lo lea antes."""
    src = _INVENTORY.read_text(encoding="utf-8")
    assert "P1-PANTRY-RECONCILIATION-ENGINE" in src
    assert "apply_inventory_delta" in src[src.index("P1-PANTRY-RECONCILIATION"):]


def _fake_rows(*pairs):
    return [{"id": i + 1, "ingredient_name": n, "quantity": 1.0, "unit": "unidad",
             "last_signal": sig} for i, (n, sig) in enumerate(pairs)]


def test_only_quiet_items_are_asked_about():
    import db_inventory
    rows = _fake_rows(("Huevo", _ago(30)), ("Leche", _ago(1)), ("Cerdo", _ago(20)))
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "execute_sql_query", return_value=rows):
        out = db_inventory.get_reconciliation_candidates(_UID)
    nombres = [i["ingredient_name"] for i in out]
    assert "Huevo" in nombres and "Cerdo" in nombres
    assert "Leche" not in nombres, "1 día no es 'quieto'"


def test_unparseable_signal_is_never_asked_about():
    """Inventar que algo esta quieto seria la misma clase de mentira que este
    workstream viene cerrando."""
    import db_inventory
    rows = _fake_rows(("Misterio", None), ("Otro", "no-es-fecha"))
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "execute_sql_query", return_value=rows):
        assert db_inventory.get_reconciliation_candidates(_UID) == []


def test_batch_is_capped():
    """La primera corrida sobre una Nevera vieja califica casi todo. 40
    preguntas no se contestan, se ignoran — y entonces la feature no existe."""
    import db_inventory
    rows = _fake_rows(*[(f"Item{i}", _ago(60)) for i in range(40)])
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "execute_sql_query", return_value=rows), \
         patch.object(db_inventory, "_reconciliation_knobs", return_value=(14, 8)):
        out = db_inventory.get_reconciliation_candidates(_UID)
    assert len(out) == 8


def test_days_quiet_is_reported():
    import db_inventory
    rows = _fake_rows(("Huevo", _ago(30)))
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "execute_sql_query", return_value=rows):
        out = db_inventory.get_reconciliation_candidates(_UID)
    assert out[0]["days_quiet"] >= 29


def test_knobs_are_clamped():
    import db_inventory, os
    for var, absurdo in (("MEALFIT_PANTRY_RECONCILE_STALE_DAYS", "0"),
                         ("MEALFIT_PANTRY_RECONCILE_BATCH", "9999")):
        os.environ[var] = absurdo
        try:
            dias, lote = db_inventory._reconciliation_knobs()
            assert 3 <= dias <= 180 and 1 <= lote <= 50
        finally:
            os.environ.pop(var, None)


# ---------------------------------------------------------------------------
# 3. Las tres respuestas
# ---------------------------------------------------------------------------

_ROW = {"ingredient_name": "Cerdo", "quantity": 0.5, "unit": "lb"}


def _resolve(action, row=_ROW):
    import db_inventory
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "execute_sql_query", return_value=row), \
         patch.object(db_inventory, "execute_sql_write") as write, \
         patch.object(db_inventory, "_persist_consumption_events") as ledger:
        out = db_inventory.resolve_reconciliation_item(_UID, 7, action)
    return out, write, ledger


def test_keep_only_resets_the_clock():
    out, write, ledger = _resolve("keep")
    assert out["ok"] and out["action"] == "keep"
    sql = write.call_args[0][0]
    assert "SET updated_at = NOW()" in sql
    assert "quantity" not in sql, "`keep` no debe tocar la cantidad"
    assert "AND user_id = %s" in sql
    ledger.assert_not_called(), "no movió la Nevera: no hay evento que registrar"


@pytest.mark.parametrize("action,outcome", [("used", "deducted"), ("spoiled", "spoiled")])
def test_used_and_spoiled_remove_and_are_recorded_distinctly(action, outcome):
    out, write, ledger = _resolve(action)
    assert out["ok"] and out["action"] == action
    assert "DELETE FROM user_inventory" in write.call_args[0][0]
    assert "AND user_id = %s" in write.call_args[0][0]
    uid, meal_id, source, events = ledger.call_args[0]
    assert (uid, source) == (_UID, "reconciliation")
    assert events[0]["outcome"] == outcome
    assert events[0]["name"] == "Cerdo" and events[0]["qty"] == 0.5
    # Sin meal: no hay registro de diario que deshacer, así que queda
    # naturalmente fuera del revert sin caso especial.
    assert meal_id is None


def test_spoiled_is_not_collapsed_into_deducted():
    """El desperdicio es informacion de COMPRA (comprar menos perecedero, o en
    envase mas chico). Colapsarlo con consumo lo hace inmedible."""
    _, _, l_used = _resolve("used")
    _, _, l_spoil = _resolve("spoiled")
    assert l_used.call_args[0][3][0]["outcome"] != l_spoil.call_args[0][3][0]["outcome"]


def test_reconciliation_events_stay_out_of_the_undo_path():
    """`revert_consumption_events` busca POR `consumed_meal_id`; los eventos de
    reconciliacion nacen con NULL. Devolver comida que el usuario declaro
    dañada la resucitaria."""
    import db_inventory
    src = _INVENTORY.read_text(encoding="utf-8")
    body = src[src.index("def revert_consumption_events"):]
    body = body[:body.index("\ndef ")]
    assert "consumed_meal_id = %s" in body
    assert "spoiled" not in db_inventory._REVERSIBLE_OUTCOMES


@pytest.mark.parametrize("bad", ["borrar", "", "USED ", None])
def test_invalid_actions_are_rejected(bad):
    import db_inventory
    out = db_inventory.resolve_reconciliation_item(_UID, 7, bad)
    assert out["ok"] is False and out["reason"] == "invalid_action"


def test_foreign_item_is_not_found():
    """El SELECT filtra `AND user_id = %s`, asi que un row_id ajeno no existe
    para este usuario — y nada se borra."""
    out, write, ledger = _resolve("used", row=None)
    assert out["ok"] is False and out["reason"] == "not_found"
    write.assert_not_called()
    ledger.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Endpoints
# ---------------------------------------------------------------------------

def test_endpoints_are_quota_exempt():
    """Doctrina P1-NEVERA-QUOTA-EXEMPT: al cap el usuario no podria corregir su
    propia Nevera Y cada respuesta quemaria credito de PLANES."""
    src = _PLANS.read_text(encoding="utf-8")
    for fn in ("def api_get_reconciliation_candidates", "def api_resolve_reconciliation_item"):
        body = src[src.index(fn):]
        body = body[:body.index("\n@router.")]
        assert "verify_api_quota" not in body
        assert "_RECONCILE_LIMITER" in body


def test_endpoint_rejects_unknown_action_before_touching_the_pantry():
    from fastapi import HTTPException
    import routers.plans as plans
    with pytest.raises(HTTPException) as exc:
        plans.api_resolve_reconciliation_item(
            {"item_id": 7, "action": "borrar"}, verified_user_id=_UID)
    assert exc.value.status_code == 422


def test_endpoint_requires_auth():
    from fastapi import HTTPException
    import routers.plans as plans
    with pytest.raises(HTTPException) as exc:
        plans.api_get_reconciliation_candidates(verified_user_id=None)
    assert exc.value.status_code == 403


def test_endpoint_404s_a_foreign_item_without_leaking_existence():
    from fastapi import HTTPException
    import routers.plans as plans
    import db_inventory
    with patch.object(db_inventory, "resolve_reconciliation_item",
                      return_value={"ok": False, "reason": "not_found"}):
        with pytest.raises(HTTPException) as exc:
            plans.api_resolve_reconciliation_item(
                {"item_id": 7, "action": "used"}, verified_user_id=_UID)
    assert exc.value.status_code == 404
    assert "tu Nevera" in str(exc.value.detail)


def test_tooltip_anchors_alive():
    assert "P1-PANTRY-RECONCILIATION-ENGINE" in _INVENTORY.read_text(encoding="utf-8")
    assert "P1-PANTRY-RECONCILIATION-ENDPOINT" in _PLANS.read_text(encoding="utf-8")
