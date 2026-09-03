"""[P1-PANTRY-ROUNDTRIP-E2E · 2026-08-07] La Nevera baja y vuelve a subir, de verdad.

Casi toda la red de seguridad de este repo es parser-based: lee el fuente y
verifica que cierta línea siga ahí. Es barata y atrapa renombres, pero no puede
contestar la única pregunta que importa aquí — *¿la comida vuelve?*. Y ya se vio
lo que cuesta esa ceguera: `test_p1_consumption_ledger` estaba VERDE exigiendo
`CREATE POLICY ... auth.uid()`, sintaxis de Supabase que hacía la migración
imposible de aplicar contra Neon. Un test que solo lee texto no puede detectar
eso; uno que corre SQL, sí — a la primera.

Esto ejercita las funciones y los endpoints REALES contra un Postgres de verdad,
con las migraciones aplicadas verbatim.

    createdb mealfit_e2e
    MEALFIT_E2E_DATABASE_URL=postgresql://user:pass@127.0.0.1:5432/mealfit_e2e \
        pytest backend/tests/test_p1_pantry_roundtrip_e2e.py -v

Sin esa variable, TODO se salta. Es deliberado: los runners de CI no tienen
Postgres, y un test que revienta por falta de infraestructura entrena a la gente
a ignorar el rojo.

⚠️ Límite conocido: las tablas base (`user_inventory`, `consumed_meals`,
`user_profiles`, `master_ingredients`, `meal_plans`) son anteriores a la
convención `migrations/` y NO están versionadas en ninguna parte. El bootstrap
de abajo las RECONSTRUYE desde lo que el código lee y escribe. Si producción
tiene columnas que aquí faltan, este harness puede pasar mientras producción
falla — pasó tres veces mientras se escribía. Lo que sí es fiel al 100% son las
migraciones reales, que se aplican sin editar.
"""
import json
import os
import uuid

import pytest

_URL = os.environ.get("MEALFIT_E2E_DATABASE_URL")

# `e2e` NO es decorativo y NO es solo un filtro de CI: es la LLAVE de la guarda
# `db_core._guard_test_write_to_prod` (P0-TEST-DB-ISOLATION), que bloquea con
# RuntimeError todo INSERT/UPDATE/DELETE hecho bajo pytest desde un test sin el
# marker. Este archivo escribe en cada caso, asi que sin el marker los 12 mueren
# en el primer INSERT.
#
# Que no enganie el hecho de que pase sin marker: la guarda tiene una segunda
# puerta, `_db_target_is_nonprod`, que deja pasar la escritura si AMBAS URLs
# contienen "test"/"staging"/"localhost"/"127.0.0.1". El ejemplo del docstring
# usa `127.0.0.1`, o sea que pasaba por esa coincidencia de substring, no por
# declarar su intencion. Apunta `MEALFIT_E2E_DATABASE_URL` a un Postgres
# desechable cuyo host no lleve ninguna de esas palabras -- el servicio
# `postgres:16` de un runner, un branch de Neon-- y los 12 vuelven a morir.
# Medido: con host `pgbox` (alias de 127.0.0.1), 12 failed en 0,32 s.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _URL,
        reason="requiere MEALFIT_E2E_DATABASE_URL apuntando a un Postgres desechable",
    ),
]

# A nivel de MÓDULO, no dentro del fixture: `db_core` construye sus pools al
# IMPORTARSE, y para cuando pytest llega al fixture el conftest ya lo importó —
# con lo que `connection_pool` sería None y todo esto fallaría con un
# `'NoneType' object has no attribute 'open'` que no dice nada del problema.
if _URL:
    os.environ["MEALFIT_DB_BACKEND"] = "neon"
    os.environ["NEON_DATABASE_URL"] = _URL
    os.environ["NEON_DATABASE_URL_POOLED"] = _URL
    os.environ.setdefault("ENVIRONMENT", "development")

_MIGRACIONES = (
    "p0_4_apply_inventory_delta_rpc.sql",
    "p1_consumption_ledger_2026_08_07.sql",
    "p1_pantry_reconciliation_2026_08_07.sql",
)

# Reconstrucción de las tablas base (ver la advertencia del docstring).
_BOOTSTRAP = """
DO $$ BEGIN CREATE ROLE anon NOLOGIN;          EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE ROLE authenticated NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE ROLE service_role NOLOGIN;  EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS public.user_profiles (
    id UUID PRIMARY KEY, email TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());

CREATE TABLE IF NOT EXISTS public.user_inventory (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    ingredient_name TEXT NOT NULL,
    quantity NUMERIC NOT NULL DEFAULT 0,
    unit TEXT, source TEXT, category TEXT,
    master_ingredient_id UUID, last_mutation_type TEXT,
    reserved_quantity NUMERIC NOT NULL DEFAULT 0,
    reservation_details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW());

-- Espejo del trigger que existe VIVO en producción y que no está en ninguna
-- migration (verificado 2026-08-07). Sin él, el harness mediría un
-- comportamiento distinto al real.
CREATE OR REPLACE FUNCTION public.set_updated_at() RETURNS trigger
LANGUAGE plpgsql AS $$ BEGIN NEW.updated_at := NOW(); RETURN NEW; END $$;
DROP TRIGGER IF EXISTS set_updated_at ON public.user_inventory;
CREATE TRIGGER set_updated_at BEFORE UPDATE ON public.user_inventory
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

CREATE TABLE IF NOT EXISTS public.consumed_meals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    meal_name TEXT, meal_type TEXT, ingredients JSONB,
    calories NUMERIC DEFAULT 0, protein NUMERIC DEFAULT 0,
    carbs NUMERIC DEFAULT 0, healthy_fats NUMERIC DEFAULT 0,
    inventory_synced_at TIMESTAMPTZ,
    consumed_at TIMESTAMPTZ NOT NULL DEFAULT NOW());

CREATE TABLE IF NOT EXISTS public.failed_inventory_deductions (
    id BIGSERIAL PRIMARY KEY, user_id UUID NOT NULL,
    ingredients JSONB, attempts INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());

CREATE TABLE IF NOT EXISTS public.master_ingredients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL, slug TEXT, category TEXT,
    density_g_per_unit NUMERIC, density_g_per_cup NUMERIC, shelf_life_days INTEGER);

CREATE TABLE IF NOT EXISTS public.meal_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    name TEXT, plan_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW());

INSERT INTO public.master_ingredients (name, slug, category, density_g_per_unit, shelf_life_days)
SELECT * FROM (VALUES
    ('Huevo','huevo','Proteína',50::numeric,21),
    ('Pan','pan','Carbohidrato',28::numeric,7),
    ('Queso','queso','Proteína',25::numeric,30)) v
WHERE NOT EXISTS (SELECT 1 FROM public.master_ingredients);
"""

# Claves EXACTAS del esquema del plato (`schemas.py`): el momento del día es
# `meal` (no `slot`) y las kcal del plato son `cals` (`calories` es el total del
# plan). Si alguien las renombra, el diario registra 0 kcal en silencio.
_PLAN = {"days": [{"day": 1, "date": "2026-08-07", "meals": [{
    "name": "Sandwich de huevo", "meal": "Desayuno",
    "cals": 420, "protein": 22, "carbs": 38, "fats": 18,
    "ingredients": ["2 huevos", "2 rebanadas de pan"]}]}]}


@pytest.fixture(scope="module")
def db():
    """Postgres desechable con el esquema base + las migraciones REALES."""
    import psycopg
    raiz = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "migrations")
    with psycopg.connect(_URL, autocommit=True) as conn:
        conn.execute(_BOOTSTRAP)
        for m in _MIGRACIONES:
            ruta = os.path.join(raiz, m)
            if not os.path.isfile(ruta):
                pytest.skip(f"falta la migración {m}")
            conn.execute(open(ruta, encoding="utf-8").read())

    import db_core
    if db_core.connection_pool is None:
        # Alguien importó db_core antes de que existieran las env vars (el
        # conftest, otro test del mismo run). Recargarlo lo reconstruye.
        import importlib
        db_core = importlib.reload(db_core)
    assert db_core.connection_pool is not None, (
        "db_core no construyó el pool ni tras recargar — revisa MEALFIT_E2E_DATABASE_URL")
    db_core.connection_pool.open()

    # Recargar db_core NO basta. Una docena de módulos hacen
    # `from db_core import connection_pool` al importarse, lo que copia el VALOR
    # de entonces — `None` — a su propio namespace. Después de recargar,
    # `db_core.connection_pool` es un pool vivo mientras `shopping_calculator`
    # sigue viendo None y responde "No connection_pool available to fetch
    # master_ingredients": el descuento se queda sin catálogo y el endpoint
    # devuelve 500. Se ve solo al correr el archivo completo (aislado pasa), que
    # es el peor sabor de test intermitente.
    #
    # Se parchean SOLO los módulos cuyo `connection_pool` es None ahora mismo:
    # los que ya tienen uno bueno no se tocan.
    import sys
    for mod in list(sys.modules.values()):
        if getattr(mod, "connection_pool", "no-tiene") is None:
            mod.connection_pool = db_core.connection_pool
    return db_core


@pytest.fixture()
def usuario(db):
    def _crear(despensa=()):
        uid = str(uuid.uuid4())
        db.execute_sql_write("INSERT INTO user_profiles (id,email) VALUES (%s,%s)",
                             (uid, f"e2e-{uid[:8]}@prueba.local"))
        for nombre, qty, unidad in despensa:
            db.execute_sql_write(
                "INSERT INTO user_inventory (user_id,ingredient_name,quantity,unit,source) "
                "VALUES (%s,%s,%s,%s,'restock')", (uid, nombre, qty, unidad))
        return uid
    return _crear


def _nevera(db, uid):
    filas = db.execute_sql_query(
        "SELECT ingredient_name, quantity::float8 AS q FROM user_inventory "
        "WHERE user_id=%s ORDER BY 1", (uid,), fetch_all=True)
    return {r["ingredient_name"]: r["q"] for r in filas}


def _eventos(db, meal_id):
    return db.execute_sql_query(
        "SELECT ingredient_name, quantity::float8 q, unit, outcome, reverted_at "
        "FROM inventory_consumption_events WHERE consumed_meal_id=%s ORDER BY 1",
        (meal_id,), fetch_all=True)


# ---------------------------------------------------------------------------
# 1. El round-trip
# ---------------------------------------------------------------------------
def test_comer_y_deshacer_devuelve_la_comida_exacta(db, usuario):
    """El caso que originó todo: 'un sándwich con dos huevos'.

    `"2 huevos"` en plural contra la fila `Huevo` en singular es el bug de
    P1-PANTRY-NAME-RESOLUTION: antes devolvía éxito SIN descontar nada.
    """
    from db import deduct_consumed_meal_from_inventory, revert_consumption_events

    uid = usuario([("Huevo", 12, "unidad"), ("Pan", 10, "rebanada")])
    meal = str(uuid.uuid4())
    inicial = _nevera(db, uid)

    r = deduct_consumed_meal_from_inventory(
        uid, ["2 huevos", "2 rebanadas de pan"], consumed_meal_id=meal, source="plan_meal")
    assert r["failed_to_deduct"] == [], r
    assert _nevera(db, uid) == {"Huevo": 10.0, "Pan": 8.0}

    ev = _eventos(db, meal)
    assert len(ev) == 2 and all(e["outcome"] == "deducted" for e in ev), ev

    revert_consumption_events(uid, meal)
    assert _nevera(db, uid) == inicial, "la Nevera no volvió a su estado original"
    assert all(e["reverted_at"] is not None for e in _eventos(db, meal))


def test_deshacer_dos_veces_no_duplica_la_comida(db, usuario):
    """Doble tap, reintento de red, o el usuario impaciente."""
    from db import deduct_consumed_meal_from_inventory, revert_consumption_events

    uid = usuario([("Huevo", 12, "unidad")])
    meal = str(uuid.uuid4())
    inicial = _nevera(db, uid)
    deduct_consumed_meal_from_inventory(uid, ["2 huevos"], consumed_meal_id=meal, source="chat")
    revert_consumption_events(uid, meal)
    revert_consumption_events(uid, meal)
    assert _nevera(db, uid) == inicial


def test_un_ajeno_no_puede_deshacer_tu_registro(db, usuario):
    from db import deduct_consumed_meal_from_inventory, revert_consumption_events

    uid = usuario([("Huevo", 12, "unidad")])
    meal = str(uuid.uuid4())
    deduct_consumed_meal_from_inventory(uid, ["2 huevos"], consumed_meal_id=meal, source="chat")
    esperado = _nevera(db, uid)

    r = revert_consumption_events(usuario(), meal)
    assert not r.get("reverted"), r
    assert _nevera(db, uid) == esperado


def test_la_conversion_de_unidades_es_simetrica(db, usuario):
    """Nevera en `lb`, plato en `lonjas`.

    Una conversión que no cuadre por unos gramos no rompe nada visible hoy: la
    Nevera deriva callada, comida a comida, hasta que la lista de compras
    empieza a mentir. Es el modo de fallo más difícil de notar del sistema.
    """
    from db import deduct_consumed_meal_from_inventory, revert_consumption_events

    uid = usuario([("Queso", 2, "lb")])
    meal = str(uuid.uuid4())
    inicial = _nevera(db, uid)

    deduct_consumed_meal_from_inventory(
        uid, ["2 lonjas de queso"], consumed_meal_id=meal, source="plan_meal")
    bajo_g = (inicial["Queso"] - _nevera(db, uid)["Queso"]) * 453.592
    assert 10 <= bajo_g <= 120, f"2 lonjas de queso no pueden ser {bajo_g:.1f} g"

    revert_consumption_events(uid, meal)
    assert _nevera(db, uid) == inicial, "la ida y la vuelta no son simétricas"


def test_deshacer_no_crea_comida_que_nunca_estuvo(db, usuario):
    """Comerse algo que no estaba en la Nevera es normal (lo compró en la calle).

    Lo que no puede pasar es que "Deshacer" lo CREE: quedaría comida que nunca
    existió y el inventario empezaría a mentir hacia arriba.
    """
    from db import deduct_consumed_meal_from_inventory, revert_consumption_events

    uid = usuario([("Huevo", 6, "unidad")])
    meal = str(uuid.uuid4())
    inicial = _nevera(db, uid)

    r = deduct_consumed_meal_from_inventory(
        uid, ["1 aguacate"], consumed_meal_id=meal, source="chat")
    assert "1 aguacate" in r["not_in_pantry"], r
    assert any(e["outcome"] == "not_in_pantry" for e in _eventos(db, meal))

    revert_consumption_events(uid, meal)
    assert _nevera(db, uid) == inicial
    assert "aguacate" not in {k.lower() for k in _nevera(db, uid)}


# ---------------------------------------------------------------------------
# 2. La reconciliación PREGUNTA, no descuenta
# ---------------------------------------------------------------------------
def test_listar_candidatos_no_mueve_la_nevera(db, usuario):
    """El invariante central de la fase 5.

    La regla del producto es "la Nevera solo baja por lo que el usuario
    registra". Si listar descontara, el banner sería descuento automático con
    otro nombre.
    """
    from db import get_reconciliation_candidates

    uid = usuario([("Lechuga", 1, "unidad"), ("Leche", 1, "galon")])
    db.execute_sql_write("ALTER TABLE user_inventory DISABLE TRIGGER set_updated_at")
    db.execute_sql_write(
        "UPDATE user_inventory SET created_at = NOW() - INTERVAL '40 days', "
        "updated_at = NOW() - INTERVAL '40 days' "
        "WHERE user_id=%s AND ingredient_name <> 'Leche'", (uid,))
    db.execute_sql_write("ALTER TABLE user_inventory ENABLE TRIGGER set_updated_at")

    inicial = _nevera(db, uid)
    cands = get_reconciliation_candidates(uid)
    assert _nevera(db, uid) == inicial, "LISTAR movió la Nevera"
    nombres = {c["ingredient_name"] for c in cands}
    assert "Lechuga" in nombres and "Leche" not in nombres, nombres


def test_se_dano_y_lo_use_quedan_distinguibles(db, usuario):
    """La comida que se daña es información de COMPRA, no de consumo.

    Colapsar ambas en `deducted` haría imposible medir desperdicio, que es medio
    motivo por el que alguien lleva una nevera digital.
    """
    from db import get_reconciliation_candidates, resolve_reconciliation_item

    uid = usuario([("Lechuga", 1, "unidad"), ("Yogur", 4, "unidad")])
    db.execute_sql_write("ALTER TABLE user_inventory DISABLE TRIGGER set_updated_at")
    db.execute_sql_write(
        "UPDATE user_inventory SET created_at = NOW() - INTERVAL '40 days', "
        "updated_at = NOW() - INTERVAL '40 days' WHERE user_id=%s", (uid,))
    db.execute_sql_write("ALTER TABLE user_inventory ENABLE TRIGGER set_updated_at")

    por_nombre = {c["ingredient_name"]: c for c in get_reconciliation_candidates(uid)}
    resolve_reconciliation_item(uid, por_nombre["Lechuga"]["id"], "used")
    resolve_reconciliation_item(uid, por_nombre["Yogur"]["id"], "spoiled")

    ev = {e["ingredient_name"]: e["outcome"] for e in db.execute_sql_query(
        "SELECT ingredient_name, outcome FROM inventory_consumption_events "
        "WHERE user_id=%s AND source='reconciliation'", (uid,), fetch_all=True)}
    assert ev.get("Yogur") == "spoiled", ev
    assert ev.get("Lechuga") != "spoiled", ev
    assert _nevera(db, uid) == {}


def test_sigue_ahi_no_toca_nada_y_reinicia_el_reloj(db, usuario):
    from db import get_reconciliation_candidates, resolve_reconciliation_item

    uid = usuario([("Arroz", 5, "lb")])
    db.execute_sql_write("ALTER TABLE user_inventory DISABLE TRIGGER set_updated_at")
    db.execute_sql_write(
        "UPDATE user_inventory SET created_at = NOW() - INTERVAL '40 days', "
        "updated_at = NOW() - INTERVAL '40 days' WHERE user_id=%s", (uid,))
    db.execute_sql_write("ALTER TABLE user_inventory ENABLE TRIGGER set_updated_at")

    fila = get_reconciliation_candidates(uid)[0]
    resolve_reconciliation_item(uid, fila["id"], "keep")
    assert _nevera(db, uid) == {"Arroz": 5.0}
    assert "Arroz" not in {c["ingredient_name"] for c in get_reconciliation_candidates(uid)}


def test_un_ajeno_no_puede_resolver_tu_item(db, usuario):
    from db import get_reconciliation_candidates, resolve_reconciliation_item

    uid = usuario([("Arroz", 5, "lb")])
    db.execute_sql_write("ALTER TABLE user_inventory DISABLE TRIGGER set_updated_at")
    db.execute_sql_write(
        "UPDATE user_inventory SET created_at = NOW() - INTERVAL '40 days', "
        "updated_at = NOW() - INTERVAL '40 days' WHERE user_id=%s", (uid,))
    db.execute_sql_write("ALTER TABLE user_inventory ENABLE TRIGGER set_updated_at")

    fila = get_reconciliation_candidates(uid)[0]
    r = resolve_reconciliation_item(usuario(), fila["id"], "used")
    assert not r.get("ok"), r
    assert _nevera(db, uid) == {"Arroz": 5.0}


# ---------------------------------------------------------------------------
# 3. Los endpoints, no solo las funciones
# ---------------------------------------------------------------------------
def _cliente(uid):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth import get_verified_user_id, verify_api_quota
    from routers.diary import router as diary
    from routers.plans import router as plans

    app = FastAPI()
    app.include_router(diary)
    app.include_router(plans)
    app.dependency_overrides[get_verified_user_id] = lambda: uid
    app.dependency_overrides[verify_api_quota] = lambda: uid
    return TestClient(app)


def test_endpoint_me_lo_comi_y_deshacer(db, usuario):
    uid = usuario([("Huevo", 12, "unidad"), ("Pan", 10, "rebanada")])
    plan_id = str(uuid.uuid4())
    db.execute_sql_write(
        "INSERT INTO meal_plans (id,user_id,name,plan_data) VALUES (%s,%s,%s,%s::jsonb)",
        (plan_id, uid, "Plan e2e", json.dumps(_PLAN)))
    c = _cliente(uid)
    inicial = _nevera(db, uid)

    r = c.post("/api/diary/consumed-from-plan",
               json={"plan_id": plan_id, "day_index": 0, "meal_index": 0})
    assert r.status_code == 200, r.text
    assert _nevera(db, uid) == {"Huevo": 10.0, "Pan": 8.0}

    fila = db.execute_sql_query(
        "SELECT id, calories, protein, meal_type FROM consumed_meals WHERE user_id=%s",
        (uid,), fetch_one=True)
    # Sin esto el diario registra el plato vacío: el usuario cree que anotó su
    # desayuno de 420 kcal y guardó 0. Un rename de `cals`/`meal` en el esquema
    # del plato lo provocaría en silencio.
    assert int(fila["calories"]) == 420, fila
    assert int(fila["protein"]) == 22, fila
    assert (fila["meal_type"] or "").lower() == "desayuno", fila

    r = c.delete(f"/api/diary/consumed/{fila['id']}")
    assert r.status_code == 200, r.text
    assert r.json().get("returned_to_pantry"), r.json()
    assert _nevera(db, uid) == inicial


def test_endpoint_no_sirve_el_plan_de_otro(db, usuario):
    """Invariante I2 por HTTP: el `AND user_id = %s` del SELECT."""
    dueno = usuario([("Huevo", 12, "unidad")])
    plan_id = str(uuid.uuid4())
    db.execute_sql_write(
        "INSERT INTO meal_plans (id,user_id,name,plan_data) VALUES (%s,%s,%s,%s::jsonb)",
        (plan_id, dueno, "Plan ajeno", json.dumps(_PLAN)))
    esperado = _nevera(db, dueno)

    r = _cliente(usuario()).post("/api/diary/consumed-from-plan",
                                 json={"plan_id": plan_id, "day_index": 0, "meal_index": 0})
    assert r.status_code != 200, r.text
    assert _nevera(db, dueno) == esperado


def test_endpoint_reconcile_rechaza_acciones_inventadas(db, usuario):
    uid = usuario([("Arroz", 5, "lb")])
    fila = db.execute_sql_query(
        "SELECT id FROM user_inventory WHERE user_id=%s", (uid,), fetch_one=True)
    c = _cliente(uid)

    assert c.get("/api/plans/inventory/reconcile").status_code == 200
    assert _nevera(db, uid) == {"Arroz": 5.0}, "el GET movió la Nevera"

    r = c.post("/api/plans/inventory/reconcile",
               json={"item_id": fila["id"], "action": "borrar_todo"})
    assert r.status_code != 200 or not r.json().get("success"), r.text
    assert _nevera(db, uid) == {"Arroz": 5.0}
