"""[P1-CONSUMPTION-LEDGER · 2026-08-07] "Deshacer registro" devuelve la comida.

Asimetria que cierra:

    registra "2 huevos"  -> diario +1 fila, Nevera 3 -> 1
    deshace el registro  -> diario -1 fila, Nevera SIGUE EN 1

Y no se podia arreglar sin la tabla: para devolver hay que saber QUE se
descontó, y eso se perdia al aplicar el delta. El string original ("2 huevos")
no basta — P1-PANTRY-NAME-RESOLUTION pudo mapearlo a la fila "Huevo" y
P1-PANTRY-INFER pudo inventar la cantidad. Re-parsear al revertir repetiria
ambas decisiones y podria devolver una cantidad DISTINTA de la que se quito.

Los tres invariantes:

  1. Solo se devuelve lo que de verdad bajo la Nevera. `not_in_pantry` y
     `failed` no movieron nada; devolverlos CREARIA comida inexistente.
  2. El revert es idempotente. Un segundo DELETE no vuelve a sumar.
  3. El revert filtra por user_id. Un meal_id ajeno no toca ninguna Nevera.
"""
import re
from pathlib import Path
from unittest.mock import patch, call

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_MIGRATION = _BACKEND / "migrations" / "p1_consumption_ledger_2026_08_07.sql"
_DIARY = _BACKEND / "routers" / "diary.py"
_INVENTORY = _BACKEND / "db_inventory.py"

_UID = "11111111-1111-4111-8111-111111111111"
_MEAL = "22222222-2222-4222-8222-222222222222"


# ---------------------------------------------------------------------------
# 1. La migración
# ---------------------------------------------------------------------------

def test_migration_exists_and_is_idempotent():
    sql = _MIGRATION.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS public.inventory_consumption_events" in sql
    assert sql.count("CREATE INDEX IF NOT EXISTS") >= 2
    assert "DROP POLICY IF EXISTS" in sql
    assert "RAISE EXCEPTION" in sql, "falta el sanity check post-migration"
    assert sql.strip().startswith("--") and "BEGIN;" in sql and sql.rstrip().endswith("COMMIT;")


def test_migration_has_no_fk_to_consumed_meals():
    """A propósito: la fila de `consumed_meals` es borrable por el usuario.
    Un FK CASCADE borraria el registro de una devolucion que SI ocurrio, y un
    RESTRICT impediria el propio DELETE que este ledger existe para soportar."""
    sql = _MIGRATION.read_text(encoding="utf-8")
    assert not re.search(r"consumed_meal_id[^,\n]*REFERENCES", sql), (
        "apareció un FK en `consumed_meal_id` — rompe el DELETE que este ledger "
        "existe para soportar, o borra el rastro de la devolución."
    )


def test_migration_constrains_outcome_and_positive_quantity():
    sql = _MIGRATION.read_text(encoding="utf-8")
    # El signo lo pone la operación, no el evento: un evento negativo al
    # revertirse restaria mas en vez de devolver.
    assert "CHECK (quantity > 0)" in sql
    for outcome in ("deducted", "inferred", "not_in_pantry", "failed"):
        assert outcome in sql


def test_migration_does_not_let_the_client_write_the_ledger():
    """Un ledger que el cliente puede escribir o borrar no prueba nada.

    [corregido 2026-08-07] Este test EXIGÍA `ENABLE ROW LEVEL SECURITY` + una
    policy con `auth.uid()`. Pasaba en verde y la migración era imposible de
    aplicar: `auth` es un schema de Supabase, eliminado en P1-NEON-DB-MIGRATION.
    Al correrla contra Neon: `InvalidSchemaName: schema "auth" does not exist`.

    Un test parser-based que exige sintaxis muerta es peor que no tenerlo:
    da confianza sin verificar nada. Lo que de verdad impide que el cliente
    escriba el ledger en esta arquitectura NO es RLS — es que el cliente no abre
    conexión a la base (PostgREST prohibido; el frontend va por endpoints). Así
    que el contrato pasa a ser: la migración no le concede NADA a un rol
    client-facing, ni invoca identidad que no existe.
    """
    sql = _MIGRATION.read_text(encoding="utf-8")
    # Sin comentarios: la migración CITA el error que la rompía para que nadie
    # lo reintroduzca, y ese texto no debe disparar el propio assert.
    sql = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    # `auth.` a secas, no solo `auth.uid()`: la primera corrección arregló la
    # policy y dejó viva `REFERENCES auth.users(id)`, que falla idéntico.
    for muerto in ("auth.", "TO authenticated", "TO anon"):
        assert muerto not in sql, (
            f"`{muerto}` es de Supabase; en Neon no existe y hace que la "
            f"migración falle al aplicarse (schema \"auth\" does not exist)."
        )
    for verbo in ("FOR INSERT", "FOR UPDATE", "FOR DELETE"):
        assert verbo not in sql, (
            f"apareció una policy {verbo} — solo el backend muta el ledger."
        )
    assert "GRANT" not in sql, (
        "un GRANT a un rol client-facing abriría el ledger a escritura directa."
    )


def test_no_post_neon_migration_resurrects_the_supabase_auth_schema():
    """El bug de arriba es de CLASE, no de este archivo.

    Cualquier migración nueva escrita copiando una vieja arrastra `auth.uid()` y
    revienta igual — y como el fallo aparece recién al APLICARLA en producción,
    nadie se entera hasta ese momento. Alcance: solo migraciones fechadas
    2026-06-12 o después (el corte de P1-NEON-DB-MIGRATION). Las anteriores son
    historia de la era Supabase y se dejan como están.
    """
    corte = (2026, 6, 12)
    culpables = []
    for f in sorted(_MIGRATION.parent.glob("*.sql")):
        m = re.search(r"_(\d{4})_(\d{2})_(\d{2})\.sql$", f.name)
        if not m:
            continue  # sin fecha en el nombre => pre-convención, pre-Neon
        if tuple(int(g) for g in m.groups()) < corte:
            continue
        cuerpo = f.read_text(encoding="utf-8")
        # Ignorar comentarios: esta misma migración EXPLICA el error citándolo.
        codigo = "\n".join(l for l in cuerpo.splitlines()
                           if not l.strip().startswith("--"))
        # `auth.` cubre las DOS formas que rompieron esta migración: la policy
        # (`auth.uid()`) y el foreign key (`REFERENCES auth.users(id)`).
        if "auth." in codigo or "TO authenticated" in codigo:
            culpables.append(f.name)

    assert not culpables, (
        "migraciones post-Neon que invocan el schema `auth` de Supabase "
        f"(no existe en Neon; fallan al aplicarse): {culpables}"
    )


# ---------------------------------------------------------------------------
# 2. Qué es reversible y qué no
# ---------------------------------------------------------------------------

def test_only_real_deductions_are_reversible():
    import db_inventory
    assert set(db_inventory._REVERSIBLE_OUTCOMES) == {"deducted", "inferred"}, (
        "`not_in_pantry` y `failed` no movieron la Nevera: devolverlos crearía "
        "comida que el usuario nunca tuvo."
    )


def test_revert_query_filters_by_user_and_pending_and_outcome():
    """Anclaje del SQL del revert: los tres filtros son invariantes distintos
    (I2 / idempotencia / no-crear-comida) y perder cualquiera es un bug."""
    src = _INVENTORY.read_text(encoding="utf-8")
    body = src[src.index("def revert_consumption_events"):]
    body = body[:body.index("\ndef ")]
    assert "AND user_id = %s" in body, "invariante I2: sin esto, un meal_id ajeno mueve otra Nevera"
    assert "reverted_at IS NULL" in body, "sin esto el revert deja de ser idempotente"
    assert "outcome = ANY(%s)" in body, "sin esto se devolverían items que nunca bajaron"
    assert "SET reverted_at = NOW()" in body


# ---------------------------------------------------------------------------
# 3. Funcional: el revert devuelve lo que se quitó
# ---------------------------------------------------------------------------

def _rows(*triples):
    return [{"ingredient_name": n, "quantity": q, "unit": u} for n, q, u in triples]


def test_revert_adds_back_the_recorded_amounts():
    import db_inventory
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "execute_sql_write",
                      return_value=_rows(("Huevo", 2.0, "unidad"), ("Queso blanco", 0.25, "lb"))), \
         patch.object(db_inventory, "add_or_update_inventory_item", return_value=True) as add_mock:
        out = db_inventory.revert_consumption_events(_UID, _MEAL)

    assert len(out["reverted"]) == 2
    # Suma con la MISMA (name, unit) que se restó — no re-parsea el string.
    assert add_mock.call_args_list == [
        call(_UID, "Huevo", 2.0, "unidad", mutation_type="consumption_revert"),
        call(_UID, "Queso blanco", 0.25, "lb", mutation_type="consumption_revert"),
    ]
    # Positivo: devuelve, no vuelve a restar.
    for c in add_mock.call_args_list:
        assert c[0][2] > 0


def test_revert_is_idempotent_when_nothing_pending():
    """Segundo DELETE del mismo meal: el UPDATE no reclama filas (ya tienen
    `reverted_at`), asi que no hay nada que sumar."""
    import db_inventory
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "execute_sql_write", return_value=[]), \
         patch.object(db_inventory, "add_or_update_inventory_item") as add_mock:
        out = db_inventory.revert_consumption_events(_UID, _MEAL)
    assert out["reverted"] == []
    add_mock.assert_not_called()


def test_revert_claims_before_adding():
    """El UPDATE que marca `reverted_at` corre ANTES de sumar. Si el proceso
    muere a mitad, el modo de fallo es "no devolví todo" (visible: la Nevera
    queda baja) en vez de "devolví dos veces" (invisible: queda alta y el plan
    compra de menos)."""
    import db_inventory
    orden = []
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "execute_sql_write",
                      side_effect=lambda *a, **k: (orden.append("claim"), _rows(("Huevo", 2.0, "unidad")))[1]), \
         patch.object(db_inventory, "add_or_update_inventory_item",
                      side_effect=lambda *a, **k: (orden.append("add"), True)[1]):
        db_inventory.revert_consumption_events(_UID, _MEAL)
    assert orden == ["claim", "add"]


def test_revert_survives_a_row_that_cannot_be_applied():
    """Una fila que ya no existe / unidad incompatible no debe abortar el resto
    de la devolución."""
    import db_inventory
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "execute_sql_write",
                      return_value=_rows(("Huevo", 2.0, "unidad"), ("Fantasma", 1.0, "lb"))), \
         patch.object(db_inventory, "add_or_update_inventory_item",
                      side_effect=[True, False]):
        out = db_inventory.revert_consumption_events(_UID, _MEAL)
    assert len(out["reverted"]) == 1 and out["skipped"] == 1


def test_revert_noop_without_ids():
    import db_inventory
    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "execute_sql_write") as w:
        assert db_inventory.revert_consumption_events(_UID, "")["reverted"] == []
        assert db_inventory.revert_consumption_events("", _MEAL)["reverted"] == []
    w.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Los productores atan el evento a su comida
# ---------------------------------------------------------------------------

def test_deduct_records_one_event_per_outcome():
    import db_inventory
    rows = [{"id": 1, "ingredient_name": "Huevo", "quantity": 3.0, "unit": "unidad",
             "reserved_quantity": 0.0, "reservation_details": None}]

    def _resolve(_u, name, **_kw):
        return (rows, "canonical") if "huevo" in name.lower() else ([], "none")

    with patch.object(db_inventory, "_db_available", return_value=True), \
         patch.object(db_inventory, "execute_sql_query", return_value=rows), \
         patch.object(db_inventory, "find_pantry_rows_for_name", side_effect=_resolve), \
         patch.object(db_inventory, "_consume_reserved_inventory", return_value=True), \
         patch.object(db_inventory, "add_or_update_inventory_item", return_value=True), \
         patch.object(db_inventory, "_persist_failed_inventory_deductions"), \
         patch.object(db_inventory, "_persist_consumption_events") as ledger:
        db_inventory.deduct_consumed_meal_from_inventory(
            _UID, ["2 huevos", "1 pan integral"],
            consumed_meal_id=_MEAL, source="photo")

    uid, meal_id, source, events = ledger.call_args[0]
    assert (uid, meal_id, source) == (_UID, _MEAL, "photo")
    por_outcome = {e["outcome"] for e in events}
    assert por_outcome == {"deducted", "not_in_pantry"}
    # El evento lleva el nombre RESUELTO, no el string crudo del usuario.
    ded = next(e for e in events if e["outcome"] == "deducted")
    assert ded["qty"] == 2 and "huevo" in ded["name"].lower()


@pytest.mark.parametrize("path,fuente", [
    # [P1-MANUAL-FOOD-LOG - 2026-08-11] `photo` salio de esta lista: /consumed ya no
    # llama a deduct directamente. El camino comun `_persist_consumed_meal` (que
    # comparten la foto y el componedor manual) es quien ata el evento, y lo cubre
    # `test_shared_persist_ties_and_parametrizes_source` abajo con photo Y manual.
    ("routers/diary.py", "plan_meal"),
    ("tools.py", "chat"),
    ("db_inventory.py", "chunk_reconcile"),
])
def test_every_producer_ties_the_event_to_its_meal(path, fuente):
    """Sin `consumed_meal_id` el evento existe pero es huérfano: el revert no
    puede encontrarlo y "Deshacer registro" vuelve a no devolver nada."""
    src = (_BACKEND / path).read_text(encoding="utf-8")
    m = re.search(
        r"deduct_consumed_meal_from_inventory\([\s\S]{0,600}?source=\"" + fuente + r"\"",
        src)
    assert m, f"no se encontró el productor `{fuente}` en {path}"
    assert "consumed_meal_id=" in m.group(0), (
        f"el productor `{fuente}` no ata el evento a su fila de consumed_meals"
    )


def test_shared_persist_ties_and_parametrizes_source():
    """[P1-MANUAL-FOOD-LOG - 2026-08-11] El tie al ledger vive UNA vez, en
    `_persist_consumed_meal`, con `source` parametrizado — y cada caller declara el
    suyo. Si alguien des-extrae el camino comun o le quita el consumed_meal_id, el
    revert de "Deshacer registro" vuelve a quedarse sin encontrar el evento."""
    src = (_BACKEND / "routers/diary.py").read_text(encoding="utf-8")
    m = re.search(
        r"deduct_consumed_meal_from_inventory\([\s\S]{0,600}?source=source", src)
    assert m, "el camino comun no parametriza `source` en la deduccion"
    assert "consumed_meal_id=" in m.group(0), (
        "el camino comun no ata el evento a su fila de consumed_meals"
    )
    foto = src[src.index("def api_log_consumed_meal("):src.index("def api_log_manual_meal(")]
    assert 'source="photo"' in foto, "/consumed dejo de declarar su source"
    manual = src[src.index("def api_log_manual_meal("):src.index("def api_frequent_foods(")]
    assert 'source="manual"' in manual, "el componedor dejo de declarar su source"


# ---------------------------------------------------------------------------
# 5. El endpoint devuelve antes de borrar
# ---------------------------------------------------------------------------

def test_delete_endpoint_reverts_before_deleting():
    src = _DIARY.read_text(encoding="utf-8")
    body = src[src.index("def api_delete_consumed_meal"):]
    body = body[:body.index("\n@router.")]
    i_rev = body.index("revert_consumption_events(verified_user_id")
    # La LLAMADA, no el nombre de la función que la contiene: buscar
    # `delete_consumed_meal(` a secas matchea `def api_delete_consumed_meal(`
    # en la primera línea y da un índice de 8.
    i_del = body.index("delete_consumed_meal(verified_user_id")
    assert i_rev < i_del, (
        "el revert debe correr ANTES del DELETE: si se borra primero y el "
        "revert falla, la comida se pierde sin rastro visible."
    )
    assert "returned_to_pantry" in body, (
        "lo devuelto debe viajar al cliente: si la Nevera sube y nadie lo "
        "explica, parece un bug."
    )


def test_delete_endpoint_returns_what_went_back():
    import routers.diary as diary
    import db_inventory
    with patch.object(db_inventory, "revert_consumption_events",
                      return_value={"reverted": ["2.0 unidad de Huevo"], "skipped": 0}), \
         patch.object(diary, "delete_consumed_meal", return_value=True):
        out = diary.api_delete_consumed_meal(_MEAL, verified_user_id=_UID)
    assert out["success"] is True
    assert out["returned_to_pantry"] == ["2.0 unidad de Huevo"]


def test_delete_of_a_foreign_meal_still_404s():
    """El revert filtra por user_id, así que no devuelve nada; el DELETE
    tampoco encuentra fila y el 404 se mantiene."""
    from fastapi import HTTPException
    import routers.diary as diary
    import db_inventory
    with patch.object(db_inventory, "revert_consumption_events",
                      return_value={"reverted": [], "skipped": 0}), \
         patch.object(diary, "delete_consumed_meal", return_value=False):
        with pytest.raises(HTTPException) as exc:
            diary.api_delete_consumed_meal(_MEAL, verified_user_id=_UID)
    assert exc.value.status_code == 404


def test_tooltip_anchor_alive():
    assert "P1-CONSUMPTION-LEDGER-ENGINE" in _INVENTORY.read_text(encoding="utf-8")
