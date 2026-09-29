# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-814 · 2026-09-29] El medidor del Dish Registry y su alerta `registry_dishes_unused`.

## Lo que estaba roto (medido con SELECT sobre producción el 28-sep)

1. **El medidor sólo miraba el nombre.** `dish_provenance` resolvía la plantilla por nombre EXACTO y nunca por
   `_template_id`/`_recipe_template_id`: 55 platos montados desde el catálogo (días deterministas del 14-15 sep)
   puntuaban 24 por nombre y 0 aplicables. Y «aplicables» dependía de un regex que lee «½ cebolla» como
   «12 cebolla» y «3 huevos» como otro alimento que «Huevo».
2. **La alerta contaba filas, no entregas.** Se emite una fila de fidelidad por INTENTO de revisión, así que los
   reintentos triplicaban una entrega. La alerta que sigue abierta desde el 26-sep 11:28 UTC («1/76 platos en 72 h,
   n_runs 5») eran 3 entregas: 2 de una cuenta admin y UNA de dbrito contada tres veces (rebanada `150515fd`).
   Deduplicada no habría saltado.
3. **Sin muestra, la alerta quedaba abierta con un dato viejo y sin decir que era viejo.** Y el arreglo obvio
   —resolverla por antigüedad— la deja muda: deduplicada y a 72 h sólo habría sido evaluable 66 horas en 20 días.
   «No concluyente» no colapsa a ningún lado: ni se abre ni se cierra; se marca `stale`.

## Lo que NO hace este lote (a propósito)

- **No deshace renombrados** («X y Huevo» no se vuelve «X»): contradice la doctrina de coincidencia exacta de
  `recipe_for_dish_name`. Sólo el id —la procedencia DECLARADA— rescata un plato renombrado.
- **No toca `_foods_de_comida`**: la comparte `apply_library_recipe`. El parser tolerante vive en una copia propia
  del medidor; la costura sigue exigiendo lo que exigía.
- **No cablea el catálogo al generador de días.**

Knob `MEALFIT_REGISTRY_PROVENANCE_V2` (default True): `0` vuelve al medidor y al cron de antes, byte a byte.
"""
from __future__ import annotations

import json

import pytest

import recipe_library as rl


_ADMIN = "4da5c079-f0ba-47af-87d1-3ec732d187d9"
_DBRITO = "c7b90ca3-fa9b-469e-bcc5-6c4124347692"

# Las 5 filas REALES de `pipeline_metrics` (node=plan_policy_fidelity) que abrieron la alerta el 26-sep 11:28 UTC
# con «1/76 en 72 h, n_runs 5». Copiadas con SELECT el 28-sep; sólo las columnas que lee el cron.
_FILAS_26_SEP = [
    {"id": 304426, "created_at": "2026-09-23T12:48:25.669018+00:00", "user_id": _ADMIN,
     "plan_id": "6594aae1-9e1d-48aa-8731-9ccfac531fc9",
     "slice_hash": "2fc9f81785692660d5c60701d50bdf4831b630e67c78a0be572866de66b21cce",
     "total": 12, "matched": 11, "applicable": 1},
    {"id": 308716, "created_at": "2026-09-25T04:32:24.273418+00:00", "user_id": _DBRITO,
     "plan_id": "92328ff7-fd6d-43d9-a08c-b1314de27d7b",
     "slice_hash": "150515fdcd8e1c801159906082d09933657f79215a44e3169ff42881bf974241",
     "total": 16, "matched": 0, "applicable": 0},
    {"id": 308734, "created_at": "2026-09-25T04:34:27.168665+00:00", "user_id": _DBRITO,
     "plan_id": "92328ff7-fd6d-43d9-a08c-b1314de27d7b",
     "slice_hash": "150515fdcd8e1c801159906082d09933657f79215a44e3169ff42881bf974241",
     "total": 16, "matched": 0, "applicable": 0},
    {"id": 308748, "created_at": "2026-09-25T04:36:07.482271+00:00", "user_id": _DBRITO,
     "plan_id": "92328ff7-fd6d-43d9-a08c-b1314de27d7b",
     "slice_hash": "150515fdcd8e1c801159906082d09933657f79215a44e3169ff42881bf974241",
     "total": 16, "matched": 0, "applicable": 0},
    {"id": 311201, "created_at": "2026-09-26T04:32:11.160927+00:00", "user_id": _ADMIN,
     "plan_id": "6594aae1-9e1d-48aa-8731-9ccfac531fc9",
     "slice_hash": "f41db9abdfb08b5eff09b23f3f7eceeb6aba807ccc4efea156e85daea0cb92f2",
     "total": 16, "matched": 0, "applicable": 0},
]


@pytest.fixture(autouse=True)
def _limpia(monkeypatch):
    monkeypatch.delenv("MEALFIT_RECIPE_LIBRARY_SELECT", raising=False)
    monkeypatch.delenv("MEALFIT_REGISTRY_PROVENANCE_V2", raising=False)
    monkeypatch.delenv("MEALFIT_REGDISH_RATE_LOOKBACK_H", raising=False)
    monkeypatch.delenv("MEALFIT_REGDISH_RATE_MIN_SAMPLES", raising=False)
    monkeypatch.delenv("MEALFIT_REGDISH_RATE_FLOOR", raising=False)
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", "")
    for f in (rl._library, rl._name_index, rl._registry_name_index):
        f.cache_clear()
    yield
    for f in (rl._library, rl._name_index, rl._registry_name_index):
        f.cache_clear()


def _plantilla(i=0):
    """Una plantilla REAL del registry DO con receta y ≥3 alimentos no triviales."""
    import dish_registry as dr
    lib = rl._library("DO")
    vistas = 0
    for t in ((dr.load_registry("DO") or {}).get("templates") or []):
        cs = [c for c in (t.get("constituents") or []) if c.get("name")]
        if t.get("template_id") in lib and len(cs) >= 3:
            if vistas == i:
                nom = ((t.get("editorial") or {}).get("display_name") or {}).get("es") or t.get("name")
                return t["template_id"], nom, cs
            vistas += 1
    pytest.skip("el registry DO no está en el árbol")


def _linea(c):
    return f"{int(c.get('grams') or 100)} g de {c.get('name')}"


# ------------------------------------------------------------------------------------------------ el medidor


def test_resuelve_por_template_id_aunque_el_cerrador_renombre():
    """El plato salió del catálogo (lo DECLARA su `_template_id`) y el cerrador de proteína le añadió «y Huevo».
    Antes: 0 del catálogo. Ahora: 1 — por la procedencia declarada, no por deshacer el nombre."""
    tid, nom, cs = _plantilla()
    plato = {"name": f"{nom} y Huevo", "_template_id": tid, "_protein_closed": True,
             "ingredients": [_linea(c) for c in cs]}
    p = rl.dish_provenance([{"meals": [plato]}])
    assert p["total"] == 1
    assert p["del_registry"] == 1, f"el `_template_id` declarado no resolvió la plantilla: {p}"
    assert p["con_receta"] == 1
    assert p["aplicables"] == 1, f"mismos alimentos que la plantilla y no cuenta como aplicable: {p}"


def test_resuelve_por_recipe_template_id():
    tid, nom, cs = _plantilla(1)
    plato = {"name": "Nombre que el modelo reescribió entero", "_recipe_template_id": tid,
             "ingredients": [_linea(c) for c in cs]}
    assert rl.dish_provenance([{"meals": [plato]}])["del_registry"] == 1


def test_un_id_que_no_existe_cae_al_nombre_exacto():
    tid, nom, cs = _plantilla()
    por_nombre = {"name": nom, "_template_id": "tpl_que_no_existe", "ingredients": [_linea(c) for c in cs]}
    ajeno = {"name": "Sopa de piedras del río Ozama", "_template_id": "tpl_que_no_existe"}
    p = rl.dish_provenance([{"meals": [por_nombre, ajeno]}])
    assert p["total"] == 2 and p["del_registry"] == 1, p


def test_sin_heuristica_de_deshacer_renombrados():
    """Sin id, «<plantilla> y Huevo» NO es del catálogo. Adivinar el nombre original es la coincidencia
    aproximada que la doctrina de `recipe_for_dish_name` prohíbe (un parecido serviría la receta de otro plato)."""
    tid, nom, cs = _plantilla()
    plato = {"name": f"{nom} y Huevo", "_protein_closed": True, "ingredients": [_linea(c) for c in cs]}
    p = rl.dish_provenance([{"meals": [plato]}])
    assert p["del_registry"] == 0, f"el medidor deshizo un renombrado sin id: {p}"


def test_parser_tolerante_del_medidor():
    """«½ cebolla», «3 huevos», «Sal al gusto», «2½ papas medianas», «(≈199 g)»: la misma comida que la plantilla."""
    tid, nom, cs = _plantilla()
    import recipe_library as m
    lineas = []
    for c in cs:
        n = str(c.get("name"))
        if n.lower() == "cebolla":
            lineas.append("½ cebolla")
        elif n.lower() == "huevo":
            lineas.append("3 huevos")
        elif n.lower() == "sal":
            lineas.append("Sal al gusto")
        else:
            lineas.append(f"1 {n} (≈{int(c.get('grams') or 100)} g)")
    plato = {"name": nom, "ingredients": lineas}
    assert m._foods_medidor_comida(plato) == m._foods_medidor_plantilla(cs), (
        f"{lineas} no se leyó como {[c.get('name') for c in cs]}")
    assert rl.dish_provenance([{"meals": [plato]}])["aplicables"] == 1


@pytest.mark.parametrize("linea,esperado", [
    ("½ cebolla", "cebolla"),
    ("3 huevos", "huevo"),
    ("Sal al gusto", None),
    ("2½ papas medianas", "papa"),
    ("1 pechuga de pollo (≈199 g)", "pechuga de pollo"),
    ("¼ cdta de Aceite de oliva", "aceite de oliva"),
    ("2 tortas pequeño de casabe", "casabe"),
    ("½ pedazo mediano de yuca (≈200 g)", "yuca"),
    ("1.2 clara de huevo", "clara de huevo"),
    ("Queso de hoja", "queso de hoja"),
])
def test_lineas_reales_del_informe(linea, esperado):
    """Formas copiadas de platos reales (replay del 28-sep). `None` = condimento trivial, no decide el plato."""
    got = rl._alimento_medidor(linea)
    want = rl._alimento_medidor(esperado) if esperado else None
    assert got == want, f"{linea!r} → {got!r}, esperaba {want!r}"


def test_la_costura_no_cambia():
    """`_foods_de_comida` la comparte `apply_library_recipe`: el parser tolerante NO se le cuela."""
    assert rl._foods_de_comida({"ingredients": ["½ cebolla"]}) == frozenset({"12 cebolla"})
    assert rl._foods_de_comida({"ingredients": ["3 huevos"]}) == frozenset({"3 huevos"})
    import inspect
    src = inspect.getsource(rl.apply_library_recipe)
    assert "_foods_de_comida(meal)" in src and "_medidor" not in src


def test_v2_nunca_cuenta_menos_aplicables_que_la_costura(monkeypatch):
    """Lo que la costura sustituiría, el medidor v2 lo cuenta (v2 ⊇ costura); lo contrario no hace falta."""
    comidas = []
    for i in range(6):
        try:
            tid, nom, cs = _plantilla(i)
        except pytest.skip.Exception:
            break
        comidas.append({"name": nom, "ingredients": [_linea(c) for c in cs]})
    comidas.append({"name": comidas[0]["name"], "ingredients": ["½ cebolla", "3 huevos"]})
    medido = rl.dish_provenance([{"meals": comidas}])["aplicables"]
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    sustituidas = sum(1 for m in comidas if rl.apply_library_recipe(dict(m)))
    assert medido >= sustituidas and sustituidas == len(comidas) - 1


def test_knob_apagado_es_el_medidor_de_antes(monkeypatch):
    tid, nom, cs = _plantilla()
    plato = {"name": f"{nom} y Huevo", "_template_id": tid, "ingredients": [_linea(c) for c in cs]}
    monkeypatch.setenv("MEALFIT_REGISTRY_PROVENANCE_V2", "0")
    assert rl.provenance_v2_enabled() is False
    assert rl.dish_provenance([{"meals": [plato]}])["del_registry"] == 0


# ------------------------------------------------------------------------------------------------ la alerta


class _Db:
    """Doble de las dos funciones SQL: responde las filas crudas y registra cada escritura."""

    def __init__(self, filas):
        self.filas = filas
        self.consultas = []
        self.escrituras = []

    def query(self, sql, params=None, **kw):
        self.consultas.append((str(sql), params))
        return [dict(f) for f in self.filas]

    def write(self, sql, params=None, **kw):
        self.escrituras.append((" ".join(str(sql).split()), params))

    def a_alertas(self):
        return [(s, p) for s, p in self.escrituras if "system_alerts" in s]

    def tick(self):
        t = [p for s, p in self.escrituras if "_registry_dish_rate_alert_job_tick" in s]
        assert t, "el tick tiene que ser observable siempre"
        return json.loads(t[-1][1])


def _correr(monkeypatch, filas, **env):
    import cron_tasks
    db = _Db(filas)
    monkeypatch.setattr(cron_tasks, "execute_sql_query", db.query, raising=False)
    monkeypatch.setattr(cron_tasks, "execute_sql_write", db.write, raising=False)
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cron_tasks._registry_dish_rate_alert_job()
    return db


def test_la_fila_triplicada_de_dbrito_es_una_entrega(monkeypatch):
    """Las 5 filas reales de la alerta del 26-sep son 3 entregas. Con 3 < 5 no se alerta."""
    db = _correr(monkeypatch, _FILAS_26_SEP)
    t = db.tick()
    assert t["n_deliveries"] == 3 and t["n_rows"] == 5, t
    assert not any("INSERT" in s for s, _ in db.a_alertas()), "3 entregas no bastan para alertar"
    assert not any("resolved_at = NOW()" in s for s, _ in db.a_alertas()), "sin muestra no se resuelve"
    assert "insufficient_samples" in (t["skip_reason"] or "")


def test_las_cuentas_admin_no_cuentan(monkeypatch):
    db = _correr(monkeypatch, _FILAS_26_SEP, MEALFIT_ADMIN_USER_IDS=f"61a13831-x,{_ADMIN.upper()}")
    t = db.tick()
    assert t["n_deliveries"] == 1 and t["n_rows_admin_excluded"] == 2, t
    assert not any("INSERT" in s for s, _ in db.a_alertas())


def test_v1_con_las_mismas_filas_si_alertaba(monkeypatch):
    """El control: el cron de antes, sobre las mismas 5 filas, abre la alerta (1/76 < 50 %)."""
    import cron_tasks
    escr = []
    monkeypatch.setattr(cron_tasks, "execute_sql_query", lambda *a, **k: [
        {"n": len(_FILAS_26_SEP), "platos": sum(f["total"] for f in _FILAS_26_SEP),
         "del_catalogo": sum(f["applicable"] for f in _FILAS_26_SEP)}], raising=False)
    monkeypatch.setattr(cron_tasks, "execute_sql_write", lambda s, p=None, **k: escr.append(str(s)), raising=False)
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    monkeypatch.setenv("MEALFIT_REGISTRY_PROVENANCE_V2", "0")
    cron_tasks._registry_dish_rate_alert_job()
    assert any("INSERT INTO system_alerts" in s for s in escr)


def _entregas(n, total=12, matched=0, applicable=0, user=_DBRITO):
    return [{"id": 1000 + i, "created_at": f"2026-09-2{i % 8}T10:00:00+00:00", "user_id": user,
             "plan_id": f"plan-{i}", "slice_hash": f"rebanada-{i}", "total": total, "matched": matched,
             "applicable": applicable} for i in range(n)]


def test_muestra_insuficiente_no_resuelve_y_marca_stale(monkeypatch):
    """«No concluyente» no colapsa a ningún lado: la alerta abierta NO se cierra; se marca `stale` sin tocar
    `triggered_at` ni `resolved_at`."""
    db = _correr(monkeypatch, _entregas(2, matched=12))
    alertas = db.a_alertas()
    assert not any("INSERT" in s for s, _ in alertas)
    assert not any("resolved_at = NOW()" in s for s, _ in alertas), "resolvió sin muestra"
    stale = [(s, p) for s, p in alertas if s.startswith("UPDATE system_alerts") and "metadata" in s]
    assert stale, f"la alerta abierta no quedó marcada stale: {alertas}"
    s, p = stale[0]
    assert "resolved_at IS NULL" in s and "triggered_at" not in s
    assert "verdict_counting" in s and "'v1_filas'" in s, "el veredicto abierto no dice qué conteo lo emitió"
    meta = json.loads(p[0])
    assert meta["stale"] is True and "stale_checked_at" in meta and "last_evaluable_at" not in meta


def test_alerta_sobre_la_procedencia_no_sobre_aplicables(monkeypatch):
    """10/12 platos del catálogo en 6 entregas y 0 aplicables: el catálogo SÍ se usa. v1 alertaba."""
    db = _correr(monkeypatch, _entregas(6, matched=10, applicable=0))
    t = db.tick()
    assert t["registry_dish_rate"] == round(60 / 72, 3) and t["n_applicable"] == 0
    assert any("resolved_at = NOW()" in s for s, _ in db.a_alertas()), "sobre el piso se resuelve"
    assert not any("INSERT" in s for s, _ in db.a_alertas())


def test_alerta_con_muestra_y_procedencia_baja(monkeypatch):
    db = _correr(monkeypatch, _entregas(6, matched=1, applicable=1))
    ins = [(s, p) for s, p in db.a_alertas() if "INSERT INTO system_alerts" in s]
    assert ins, db.escrituras
    meta = json.loads(ins[0][1][3])
    assert meta["counting"] == "v2_entrega_plan_rebanada"
    assert meta["n_deliveries"] == 6 and meta["n_from_registry"] == 6 and meta["n_dishes"] == 72
    assert meta["n_applicable"] == 6 and meta["stale"] is False and meta["last_evaluable_at"]


def test_se_queda_la_ultima_fila_de_cada_entrega(monkeypatch):
    """Los reintentos de la misma (plan, rebanada): cuenta el último, que es el entregado."""
    filas = _entregas(5, matched=12)
    filas.append(dict(filas[0], id=9999, created_at="2026-09-29T10:00:00+00:00", matched=0))
    db = _correr(monkeypatch, filas)
    t = db.tick()
    assert t["n_deliveries"] == 5 and t["n_rows"] == 6
    assert t["n_from_registry"] == 48


def test_ventana_de_168_h_por_defecto(monkeypatch):
    db = _correr(monkeypatch, [])
    assert db.consultas and "168" in [str(x) for x in (db.consultas[0][1] or ())], db.consultas
    assert db.tick()["lookback_h"] == 168


def test_documentado_en_la_tabla_de_alertas():
    import pathlib
    import cron_tasks
    doc = (pathlib.Path(cron_tasks.__file__).resolve().parent / "docs" / "system_alerts_resolution_table.md")
    fila = [l for l in doc.read_text(encoding="utf-8").splitlines() if l.startswith("| `registry_dishes_unused`")]
    assert fila and "P1-PLAN-LOTE-814" in fila[0] and "stale" in fila[0]


def test_tooltip_anchor_en_el_cron():
    import inspect
    import cron_tasks
    src = inspect.getsource(cron_tasks._registry_dish_rate_alert_job)
    assert "P1-PLAN-LOTE-814" in src and "registry_dish_alert" in src
