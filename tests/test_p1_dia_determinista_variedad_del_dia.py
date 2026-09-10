# -*- coding: utf-8 -*-
"""[P1-DIA-DETERMINISTA-VARIEDAD-DEL-DIA · 2026-09-10] Las puertas de variedad del día no miraban a este camino.

## De dónde sale

El dueño preguntó si el «queso de papa» podía ser cualquier queso. Al pasarlo a gouda salieron a la
vista dos platos que eran el mismo —«Arepa de maíz con queso gouda» y «Arepitas de maíz con queso
gouda (horneadas)»—: 11 de 120 comidas, y un día con los dos.

El camino del modelo tiene dos puertas para eso y el día determinista no pasa por ellas —la cuarta
vez hoy con la misma forma, tras el retinol, el sodio y el path degradado—:

- `_SAME_DAY_PROTEIN_GATE_LABELS`: carnes, pescados y HUEVO (lo pidió el dueño al ver huevo en
  desayuno y cena el mismo día); exime queso, legumbres y yogur, que en RD se repiten por cultura.
- `_LIGHT_BASE_TOKENS`: la misma base (avena, casabe, arepa…) en desayuno Y merienda.

Medido sobre sus 30 días, con esas definiciones y no con las mías: **12 días** repetían proteína y
**4** base ligera. Su plan está en `balanced`, que según `horizon` NO permite repetir proteína.

## Diseño

Se PREFIERE, no se descarta — el patrón del sodio: el candidato que choca queda de reserva y sólo se
sirve si no hay otro. Entre dos reservas, primero la de variedad (blanda) y después la de sodio (el
día sigue pasando por el juicio final de sodio).
"""
import inspect

import pytest

import deterministic_day as dd


# ── las reglas se LEEN, no se copian ─────────────────────────────────────────
def test_las_puertas_salen_del_ssot():
    src = inspect.getsource(dd._variedad_ssot)
    assert "_SAME_DAY_PROTEIN_GATE_LABELS" in src and "bases_ligeras" in src
    labels, tokens = dd._variedad_ssot()
    from graph_orchestrator import _SAME_DAY_PROTEIN_GATE_LABELS, _LIGHT_BASE_TOKENS
    assert labels == frozenset(_SAME_DAY_PROTEIN_GATE_LABELS)
    assert tokens == tuple(_LIGHT_BASE_TOKENS), "el modelo y el día determinista leen bases distintas"


def test_la_arepa_y_las_arepitas_son_la_misma_base():
    """Con «arepita» como etiqueta aparte, arepa (desayuno) y arepitas (merienda) no chocaban."""
    _, tokens = dd._variedad_ssot()
    a = dd._bases_ligeras_de({"name": "Arepa de maíz con queso gouda"}, tokens)
    b = dd._bases_ligeras_de({"name": "Arepitas de maíz con queso gouda (horneadas)"}, tokens)
    assert a & b == {"arepa"}


def test_el_detector_del_modelo_agrupa_por_familia():
    """La misma ceguera estaba en la autocrítica del modelo: se arregla en el SSOT, para los dos."""
    import graph_orchestrator as go
    dias = [{"meals": [
        {"meal": "Desayuno", "name": "Arepa de maíz con queso gouda", "ingredients": []},
        {"meal": "Merienda", "name": "Arepitas de maíz con queso gouda (horneadas)", "ingredients": []}]}]
    assert go._detect_light_base_repeats(dias), "la autocrítica del modelo sigue sin ver arepa + arepitas"


def test_la_puerta_nace_APAGADA_hasta_que_haya_memoria_entre_dias(monkeypatch):
    """A/B con el mismo instrumento sobre los 30 días del dueño: encendida, las semanas que rompen el
    tope de repetición de `balanced` (2 por 7 días) pasan de 9 a 28. No se enciende por defecto hasta
    que el día determinista recuerde lo que sirvió los días anteriores."""
    monkeypatch.delenv("MEALFIT_DETERMINISTIC_DAY_SAME_DAY_VARIETY", raising=False)
    assert dd._variedad_del_dia_on() is False


def test_la_puerta_tiene_su_knob_de_marcha_atras(monkeypatch):
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_SAME_DAY_VARIETY", "false")
    assert dd._variedad_del_dia_on() is False
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_SAME_DAY_VARIETY", "true")
    assert dd._variedad_del_dia_on() is True


def test_el_queso_y_las_legumbres_no_cuentan():
    """La regla es del dueño y es cultural: el arroz con habichuela se repite y no fatiga."""
    labels, _ = dd._variedad_ssot()
    assert "huevo" in labels and "pollo" in labels and "pescado" in labels
    for exento in ("queso", "legumbre", "yogur", "yogurt"):
        assert exento not in labels, f"«{exento}» pasó a contar como proteína que fatiga"


@pytest.mark.parametrize("modo,esperado", [("routine", True), ("balanced", False), ("explore", False)])
def test_el_permiso_de_repetir_sale_de_horizon(modo, esperado):
    fd = {"_plan_policy_effective": {"recurrence": {"global_mode": modo}}}
    assert dd._repetir_proteina_ok(fd) is esperado


def test_sin_politica_manda_el_defecto_de_horizon():
    assert dd._repetir_proteina_ok({}) is False
    assert dd._repetir_proteina_ok(None) is False


# ── la base ligera, con el criterio del detector del modelo ──────────────────
@pytest.mark.parametrize("nombre,base", [
    ("Arepitas de maíz con queso gouda (horneadas)", "arepa"),
    ("Arepa de maíz con queso gouda", "arepa"),
    ("Casabe con queso gouda y manzana", "casabe"),
    ("Avena cocida con claras de huevo y maní", "avena"),
])
def test_reconoce_la_base_como_el_detector_del_modelo(nombre, base):
    _, tokens = dd._variedad_ssot()
    assert base in dd._bases_ligeras_de({"name": nombre, "ingredients": []}, tokens)


def test_una_comida_sin_base_ligera_devuelve_vacio():
    _, tokens = dd._variedad_ssot()
    assert dd._bases_ligeras_de({"name": "Pechuga a la plancha con yuca", "ingredients": []}, tokens) == set()


@pytest.mark.parametrize("basura", [None, {}, {"name": None}, "no soy un plato"])
def test_no_revienta_con_basura(basura):
    _, tokens = dd._variedad_ssot()
    assert dd._bases_ligeras_de(basura, tokens) == set()


# ── el cableado ──────────────────────────────────────────────────────────────
def test_el_choque_de_variedad_queda_de_RESERVA_no_descartado():
    src = inspect.getsource(dd.build_day_for_skeleton)
    assert "_reserva_var = (_c, _t, _na)" in src, "volvió a descartar en vez de reservar"
    assert "_reserva_var or _reserva" in src, (
        "la reserva de variedad (blanda) tiene que servirse ANTES que la de sodio")


def test_la_proteina_y_la_base_se_anotan_solo_al_servir():
    """Anotarlas al descartar bloquearía el día con platos que nadie come: se anotan en UN sitio,
    después de decidir entre el candidato limpio y las reservas, justo antes de servir."""
    src = inspect.getsource(dd.build_day_for_skeleton)
    assert src.count("_proteinas_hoy.add(") == 1 and src.count("_bases_hoy |=") == 1
    antes_de_servir = src.split("meals.append(comida)")[0].rsplit("if comida is None:", 1)[1]
    assert "_proteinas_hoy.add(" in antes_de_servir and "_bases_hoy |=" in antes_de_servir


def test_la_regla_compartida_ya_ve_las_arepitas():
    """«arepitas» no empieza por «arepa» (a-r-e-p-I): la puerta del modelo tampoco las veía."""
    _, tokens = dd._variedad_ssot()
    assert "arepita" in tokens
