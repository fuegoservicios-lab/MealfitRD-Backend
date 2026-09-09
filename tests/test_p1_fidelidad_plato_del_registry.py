# -*- coding: utf-8 -*-
"""[P1-FIDELIDAD-PLATO-DEL-REGISTRY · 2026-09-09] El informe de fidelidad ya cuenta los platos.

## Lo que se midió antes de escribir esto

Un plan REAL, generado en producción el 09-sep a las 15:17 UTC contra el registry de 179
plantillas, con `MEALFIT_PLAN_POLICY_MODE=enforce`:

    enforced: true    registry_in_prompt: true    score: 1.0    issues: []
    platos del catálogo: 0 de 12

Las dos afirmaciones eran ciertas a la vez. `registry_in_prompt` dice que los candidatos VIAJARON
en el prompt; nadie publicaba si el modelo los USÓ. La fidelidad puntúa anclas, repetición y el
contrato de la rebanada — la identidad del plato no entra en su cuenta, así que su 1.0 no era un
error: era una respuesta a otra pregunta.

Es la tercera vez que este repo se encuentra la misma forma: *«un veredicto que no puede fallar no
informa»*, *«el gate cuenta VEREDICTOS no DESTINOS»*. Aquí se cierra contando.

## Las tres cifras, y por qué son tres

La costura (`apply_library_recipe`) exige DOS condiciones: que el nombre resuelva a una plantilla y
que el conjunto de alimentos servido COINCIDA con el de esa plantilla. Publicar sólo la primera
prometería un rendimiento que la segunda no entrega.

    del_registry  el nombre resuelve a una plantilla
    con_receta    esa plantilla además tiene pasos escritos
    aplicables    nombre Y alimentos coinciden ⇒ la costura sustituiría de verdad

`tasa` es `aplicables / total`, la pesimista. Prometer la optimista es exactamente cómo un informe
acaba afirmando algo que no ocurrió — el defecto que `P1-REVIEW-KIND-HONEST` cerró.

## Las dos trampas que estos tests fijan

1. **El medidor NO cuelga del knob que informa.** Un instrumento gateado por
   `MEALFIT_RECIPE_LIBRARY_SELECT` sólo sabría medir después de encenderlo, que es cuando ya no
   hace falta. Es la trampa de `P1-I18N-DEAD-VEREDICTO`: la defensa reprodujo dentro de sí el
   defecto que venía a cerrar.

2. **El medidor y la costura normalizan IGUAL, por construcción.** `dish_registry._norm` conserva
   la puntuación y `recipe_library._norm` la retira. Medir con una y sustituir con la otra daría
   un número que no predice nada. `test_el_medidor_predice_a_la_costura` lo ejercita contra las
   dos funciones reales en vez de confiar en que sigan pareciéndose.
"""
import json

import pytest

import recipe_library as rl


@pytest.fixture(autouse=True)
def _limpia(monkeypatch):
    monkeypatch.delenv("MEALFIT_RECIPE_LIBRARY_SELECT", raising=False)
    for f in (rl._library, rl._name_index, rl._registry_name_index):
        f.cache_clear()
    yield
    for f in (rl._library, rl._name_index, rl._registry_name_index):
        f.cache_clear()


def _plantillas_reales(n=4):
    """Platos copiados del registry vivo: nombre exacto + sus alimentos. El control POSITIVO."""
    import dish_registry as dr
    snap = dr.load_registry("DO") or {}
    fuera = []
    for t in (snap.get("templates") or []):
        if not t.get("constituents"):
            continue
        nom = ((t.get("editorial") or {}).get("display_name") or {}).get("es") or t.get("name")
        if not nom:
            continue
        fuera.append({
            "name": nom,
            "ingredients": [f"{int(c.get('grams') or 100)} g de {c.get('name')}"
                            for c in t["constituents"]],
        })
        if len(fuera) >= n:
            break
    return fuera


# --------------------------------------------------------------------------- el medidor


def test_mide_con_el_knob_APAGADO():
    """La cifra que decide si encender el knob no puede necesitar el knob encendido."""
    assert rl.library_select_enabled() is False
    platos = _plantillas_reales(3)
    if not platos:
        pytest.skip("el registry no está en el árbol")
    p = rl.dish_provenance([{"meals": platos}])
    assert p["del_registry"] == len(platos), (
        f"con el knob apagado el medidor devolvió {p}; un instrumento que sólo sabe medir "
        f"después de decidir no informa la decisión")


def test_control_positivo_llega_a_uno():
    """Una sonda que falla contra TODO habla de la sonda, no del producto."""
    platos = _plantillas_reales(4)
    if not platos:
        pytest.skip("el registry no está en el árbol")
    p = rl.dish_provenance([{"meals": platos}])
    assert p["total"] == len(platos)
    assert p["tasa"] == 1.0, f"el control positivo no llega a 1.0: {p}"
    assert p["del_registry"] == p["con_receta"] == p["aplicables"] == len(platos)


def test_control_negativo_y_el_vacio_no_son_lo_mismo():
    """`tasa: 0.0` («medí y no había») y `tasa: None` («no hay nada que medir») son veredictos
    distintos. Colapsarlos es lo que `P2-GATE-INCONCLUSO` prohíbe: «no concluyente» no puede
    caer hacia ningún lado."""
    inventado = rl.dish_provenance([{"meals": [{"name": "Sopa de piedras del río Ozama",
                                               "ingredients": ["3 piedras"]}]}])
    assert inventado["total"] == 1 and inventado["del_registry"] == 0 and inventado["tasa"] == 0.0

    for vacio in ([], None, [{"meals": []}], [{"meals": [{"name": "  "}]}]):
        p = rl.dish_provenance(vacio)
        assert p["total"] == 0 and p["tasa"] is None, f"{vacio!r} debería no medir nada: {p}"


def test_el_nombre_casa_pero_la_comida_es_otra():
    """La razón de que `aplicables` exista aparte de `del_registry`.

    Medido el 08-sep: «Huevos revueltos con cebolla y casabe» casaba de nombre y traía otros
    alimentos. Servirle la receta de la plantilla sería pedir comida ausente."""
    platos = _plantillas_reales(1)
    if not platos:
        pytest.skip("el registry no está en el árbol")
    impostor = {"name": platos[0]["name"], "ingredients": ["200 g de Piedras", "1 g de Humo"]}
    p = rl.dish_provenance([{"meals": [impostor]}])
    assert p["del_registry"] == 1, "el nombre SÍ es del catálogo"
    assert p["aplicables"] == 0, "pero la costura no podría sustituir: los alimentos son otros"
    assert p["tasa"] == 0.0, "la tasa publica el número pesimista, no el halagüeño"


def test_el_medidor_predice_a_la_costura():
    """El contrato entero: `aplicables` == cuántas veces `apply_library_recipe` diría que sí.

    Si alguien cambia una de las dos normalizaciones, esta prueba cae antes de que producción
    empiece a publicar un número que no predice nada.
    """
    platos = _plantillas_reales(5)
    if not platos:
        pytest.skip("el registry no está en el árbol")
    impostor = {"name": platos[0]["name"], "ingredients": ["200 g de Piedras"]}
    ajeno = {"name": "Plato que no existe en ningún catálogo", "ingredients": []}
    comidas = [dict(m) for m in platos] + [impostor, ajeno]

    medido = rl.dish_provenance([{"meals": comidas}])["aplicables"]

    import os
    os.environ["MEALFIT_RECIPE_LIBRARY_SELECT"] = "1"
    try:
        sustituidas = sum(1 for m in comidas if rl.apply_library_recipe(dict(m)))
    finally:
        os.environ.pop("MEALFIT_RECIPE_LIBRARY_SELECT", None)

    assert medido == sustituidas, (
        f"el medidor dice {medido} y la costura sustituye {sustituidas}: el informe estaría "
        f"prometiendo un rendimiento que el knob no entrega")


def test_pais_sin_registry_no_revienta():
    p = rl.dish_provenance([{"meals": [{"name": "Cualquier cosa"}]}], country="XX")
    assert p == {"total": 0, "del_registry": 0, "con_receta": 0, "aplicables": 0, "tasa": None}


def test_dias_malformados_fail_open():
    """Fail-open: el medidor jamás puede tumbar una generación."""
    for basura in (["no soy un día"], [{"meals": "tampoco"}], [{"meals": [None, 7, "x"]}], [None]):
        p = rl.dish_provenance(basura)
        assert isinstance(p, dict) and p["del_registry"] == 0


# --------------------------------------------------------------------------- el informe


def test_el_informe_publica_la_procedencia_y_no_toca_el_veredicto():
    """`registry_dishes` mide; `score` e `issues` siguen siendo lo que eran.

    Meter la procedencia en el score cambiaría, sin que nadie lo decida, qué planes se rechazan
    en `MEALFIT_FIDELITY_GATE=block`."""
    import horizon

    dias = [{"meals": _plantillas_reales(2)}]
    con = horizon.fidelity_report(dias, None, None, surface="test")
    sin = horizon.fidelity_report([{"meals": [{"name": "Sopa de piedras"}]}], None, None, surface="test")

    for r in (con, sin):
        assert "registry_dishes" in r, "el informe dejó de publicar la procedencia"
        assert set(r["registry_dishes"]) == {"total", "del_registry", "con_receta", "aplicables", "tasa"}
    assert con["score"] == sin["score"], "la procedencia se coló en el score"
    assert con["issues"] == sin["issues"] == [], "la procedencia se coló en los issues"
    assert con["codes"] == sin["codes"], "la procedencia se coló en los codes"


def test_la_metrica_aplana_las_cuatro_claves():
    """El cron agrega por claves jsonb planas; un sub-objeto anidado lo dejaría contando ceros."""
    import inspect

    import horizon

    src = inspect.getsource(horizon.emit_fidelity_metric)
    for k in ("registry_dishes_total", "registry_dishes_matched",
              "registry_dishes_applicable", "registry_dish_rate"):
        assert f'"{k}"' in src, f"la métrica dejó de emitir {k}, que es lo que el cron agrega"


def test_el_pais_de_la_procedencia_es_el_de_la_COCINA():
    """I16: la cocina no es la tienda. Resolver por mercado serviría el catálogo equivocado."""
    import inspect

    import horizon

    src = inspect.getsource(horizon._registry_dishes_for_effective)
    assert "_culture_country" in src, "la procedencia dejó de resolver el país por cultura"
    assert "country_for_form_data" not in src, "la procedencia resolvió por MERCADO (I16 roto)"


# --------------------------------------------------------------------------- la alerta


def test_el_cron_no_evalua_con_la_biblioteca_apagada(monkeypatch):
    """Con el knob apagado, una tasa de 0 es el comportamiento PEDIDO. Alertar sobre el estado
    normal enseña a ignorar la alerta, y entonces no avisa el día que importa."""
    import cron_tasks

    escrituras = []
    monkeypatch.setattr(cron_tasks, "execute_sql_write",
                        lambda *a, **k: escrituras.append(a[0] if a else ""), raising=False)
    monkeypatch.setattr(cron_tasks, "execute_sql_query",
                        lambda *a, **k: pytest.fail("no debería consultar con el knob apagado"),
                        raising=False)
    monkeypatch.delenv("MEALFIT_RECIPE_LIBRARY_SELECT", raising=False)

    cron_tasks._registry_dish_rate_alert_job()

    assert escrituras, "el tick tiene que ser observable SIEMPRE, también cuando no aplica"
    assert not any("system_alerts" in str(s) for s in escrituras), "no debe alertar con el knob apagado"
    assert any("_registry_dish_rate_alert_job_tick" in str(s) for s in escrituras)


def test_el_cron_alerta_cuando_se_fuerza_el_catalogo_y_no_prende(monkeypatch):
    import cron_tasks

    escrituras = []

    def _write(sql, params=None):
        escrituras.append((str(sql), params))

    monkeypatch.setattr(cron_tasks, "execute_sql_write", _write, raising=False)
    monkeypatch.setattr(cron_tasks, "execute_sql_query",
                        lambda *a, **k: [{"n": 9, "platos": 108, "del_catalogo": 4}], raising=False)
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    rl._library.cache_clear()

    cron_tasks._registry_dish_rate_alert_job()

    alertas = [s for s, _ in escrituras if "system_alerts" in s and "INSERT" in s]
    assert alertas, f"4/108 con la biblioteca encendida tenía que alertar; escrituras={escrituras}"
    tick = [p for s, p in escrituras if "_registry_dish_rate_alert_job_tick" in s]
    assert tick and json.loads(tick[0][1])["registry_dish_rate"] == 0.037


def test_el_cron_resuelve_la_alerta_cuando_el_catalogo_prende(monkeypatch):
    import cron_tasks

    escrituras = []
    monkeypatch.setattr(cron_tasks, "execute_sql_write",
                        lambda sql, params=None: escrituras.append(str(sql)), raising=False)
    monkeypatch.setattr(cron_tasks, "execute_sql_query",
                        lambda *a, **k: [{"n": 9, "platos": 108, "del_catalogo": 100}], raising=False)
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")

    cron_tasks._registry_dish_rate_alert_job()

    assert any("UPDATE system_alerts" in s and "resolved_at" in s for s in escrituras), (
        "sobre el piso, la alerta se auto-resuelve (Auto-implicit)")
    assert not any("INSERT INTO system_alerts" in s for s in escrituras)


def test_muestra_insuficiente_no_alerta(monkeypatch):
    """El hermano `review_failed_delivered_rate_high` está hoy en su muestra mínima (3/5) y se
    mueve entero con un caso. Este pide lo mismo antes de hablar."""
    import cron_tasks

    escrituras = []
    monkeypatch.setattr(cron_tasks, "execute_sql_write",
                        lambda sql, params=None: escrituras.append((str(sql), params)), raising=False)
    monkeypatch.setattr(cron_tasks, "execute_sql_query",
                        lambda *a, **k: [{"n": 2, "platos": 24, "del_catalogo": 0}], raising=False)
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")

    cron_tasks._registry_dish_rate_alert_job()

    assert not any("system_alerts" in s for s, _ in escrituras)
    tick = [p for s, p in escrituras if "_registry_dish_rate_alert_job_tick" in s]
    assert tick and "insufficient_samples" in json.loads(tick[0][1])["skip_reason"]


def test_el_cron_esta_registrado_y_el_alert_key_documentado():
    import inspect
    import pathlib

    import cron_tasks

    src = inspect.getsource(cron_tasks.register_plan_chunk_scheduler)
    assert "_registry_dish_rate_alert_job" in src, "el cron no está registrado en el SSOT"
    assert 'id="registry_dish_rate_alert"' in src

    doc = pathlib.Path(cron_tasks.__file__).resolve().parent / "docs" / "system_alerts_resolution_table.md"
    assert "registry_dishes_unused" in doc.read_text(encoding="utf-8"), (
        "todo alert_key vive en la tabla canónica (P2-AUDIT-4)")
