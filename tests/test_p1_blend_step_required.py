"""[P1-BLEND-STEP-REQUIRED + P1-STEP-UNIT-PLURAL + P1-FRUIT-COUNT-CAPS-EXT + P1-COUNT-UNIT-NOUN
· 2026-07-25] Los defectos que el owner vio en las capturas del plan `ea79db0e`.

Los cuatro verificados contra `plan_data` antes de tocar código:

1. **Batido que no dice licuar** — D3 "Batido Cremoso de Lechosa y Piña": 2 pasos, mise ("Mide la
   leche…") y montaje ("Vierte el batido en un vaso"). El paso que CONVIERTE los ingredientes en
   batido no existe. El chequeo de COMPLETION lo daba por bueno porque sí hay paso de servido.
2. **Pasos que piden más unidades de las que hay** — D1 Tostadas: ingrediente `1 huevo`, pasos
   "rompe CADA huevo" y "Repite con EL OTRO huevo".
3. **`5½ guayabas`** en un desayuno y `4½` en un batido: 9.6 guayabas en 3 días para una persona.
   La guayaba no tenía rama de cap.
4. **`6½ láminas de casabe`** — el cap `casabe: 2.0` que yo mismo añadí NO disparaba nunca: el
   regex de conteo mira el primer sustantivo ("lamina"), no el alimento.
"""
import pytest

import graph_orchestrator as go


# ───────────── 1. el batido tiene que licuar ─────────────

def _batido_sin_licuar():
    return {"days": [{"day": 1, "meals": [{
        "name": "Batido Cremoso de Lechosa y Piña",
        "ingredients": ["270 g de lechosa madura", "2 tazas de leche descremada"],
        "recipe": ["Mise en place: pela la lechosa y córtala en trozos. Mide la leche.",
                   "Montaje: vierte el batido en un vaso grande y sirve."],
    }]}]}


def test_inserta_el_paso_de_licuar():
    res = _batido_sin_licuar()
    go._run_assembly_validations(res, {}, set())
    pasos = res["days"][0]["meals"][0]["recipe"]
    assert any("licú" in str(s).lower() or "licu" in str(s).lower() for s in pasos), pasos


def test_el_licuado_va_ANTES_del_montaje():
    """Licuar después de servir no es una receta, es una broma."""
    res = _batido_sin_licuar()
    go._run_assembly_validations(res, {}, set())
    pasos = [str(s).lower() for s in res["days"][0]["meals"][0]["recipe"]]
    i_licua = next(i for i, s in enumerate(pasos) if "licu" in s)
    i_montaje = next(i for i, s in enumerate(pasos) if s.startswith("montaje"))
    assert i_licua < i_montaje


def test_no_duplica_si_ya_licua():
    res = {"days": [{"day": 1, "meals": [{
        "name": "Batido de Fresa", "ingredients": ["100 g de fresas"],
        "recipe": ["Mise en place: lava las fresas.",
                   "El Toque de Fuego: licúa todo por 1 minuto.",
                   "Montaje: sirve frío."]}]}]}
    go._run_assembly_validations(res, {}, set())
    pasos = res["days"][0]["meals"][0]["recipe"]
    assert sum(1 for s in pasos if "licú" in str(s).lower() or "licu" in str(s).lower()) == 1
    assert len(pasos) == 3, "idempotente"


def test_no_toca_platos_que_no_son_batidos():
    res = {"days": [{"day": 1, "meals": [{
        "name": "Pollo Guisado", "ingredients": ["150 g de pollo"],
        "recipe": ["Mise en place: corta el pollo.", "Montaje: sirve caliente."]}]}]}
    go._run_assembly_validations(res, {}, set())
    assert not any("licu" in str(s).lower() for s in res["days"][0]["meals"][0]["recipe"])


# ───────────── 2. los pasos no piden más de lo que hay ─────────────

def test_un_huevo_listado_no_permite_pasos_en_plural():
    res = {"days": [{"day": 1, "meals": [{
        "name": "Tostadas Integrales con Huevo",
        "ingredients": ["1 huevo", "1 rebanada de pan integral"],
        "recipe": ["Mise en place: rompe cada huevo en un tazón pequeño.",
                   "El Toque de Fuego: cocina 3-4 minutos. Repite con el otro huevo.",
                   "Montaje: sirve."]}]}]}
    go._run_assembly_validations(res, {}, set())
    blob = " ".join(res["days"][0]["meals"][0]["recipe"]).lower()
    assert "cada huevo" not in blob and "el otro huevo" not in blob, blob
    assert "el huevo" in blob, "queda en singular, no desaparece la instrucción"


def test_con_dos_huevos_listados_el_plural_es_correcto():
    """El plural no es el defecto: el defecto es que no cuadre con lo listado."""
    res = {"days": [{"day": 1, "meals": [{
        "name": "Revoltillo", "ingredients": ["2 huevos"],
        "recipe": ["Mise en place: rompe cada huevo en un tazón.", "Montaje: sirve."]}]}]}
    go._run_assembly_validations(res, {}, set())
    assert "cada huevo" in " ".join(res["days"][0]["meals"][0]["recipe"]).lower()


def test_no_cambia_cantidades_ni_macros():
    """Se corrige el TEXTO, no el ingrediente: los macros y la lista se calculan desde
    `ingredients`, así que subir la cantidad cambiaría el plan entregado."""
    res = {"days": [{"day": 1, "meals": [{
        "name": "Tostadas con Huevo", "ingredients": ["1 huevo"], "cals": 200,
        "recipe": ["Mise: rompe cada huevo.", "Montaje: sirve."]}]}]}
    go._run_assembly_validations(res, {}, set())
    m = res["days"][0]["meals"][0]
    assert m["ingredients"] == ["1 huevo"] and m["cals"] == 200


# ───────────── 3. caps de conteo ─────────────

def test_cap_de_guayaba():
    days = [{"day": 1, "meals": [{"name": "Tostadas", "ingredients": ["5½ guayabas"]}]}]
    go._cap_unrealistic_portions(days)
    assert days[0]["meals"][0]["ingredients"][0] == "3 guayabas"


def test_presentacion_sin_gramos_usa_el_cap_del_alimento():
    """'6½ láminas de casabe' → el regex miraba 'lamina' (sin cap) en vez de 'casabe'."""
    days = [{"day": 1, "meals": [{"name": "Casabe", "ingredients": ["6½ láminas de casabe"]}]}]
    go._cap_unrealistic_portions(days)
    assert days[0]["meals"][0]["ingredients"][0].startswith("2 lámina")


def test_presentacion_CON_gramos_no_se_toca():
    """'6½ láminas de casabe (95 g)' tiene conteo absurdo pero MASA razonable. Capear ahí
    recortaría 95 g → 29 g: quitar comida real por un problema de etiqueta, y la banda lo paga.
    Los caps por gramos ya gobiernan ese caso."""
    days = [{"day": 1, "meals": [{"name": "Casabe",
                                  "ingredients": ["6½ láminas de casabe (95 g)"]}]}]
    go._cap_unrealistic_portions(days)
    assert days[0]["meals"][0]["ingredients"][0] == "6½ láminas de casabe (95 g)"


def test_presentacion_de_alimento_sin_cap_no_inventa_techo():
    days = [{"day": 1, "meals": [{"name": "Tilapia", "ingredients": ["3 filetes de tilapia"]}]}]
    go._cap_unrealistic_portions(days)
    assert days[0]["meals"][0]["ingredients"][0] == "3 filetes de tilapia"


# ───────────── 4. knobs ─────────────

@pytest.mark.parametrize("linea", [
    'BLEND_STEP_REQUIRED = _env_bool("MEALFIT_BLEND_STEP_REQUIRED", True)',
    'STEP_UNIT_PLURAL_FIX = _env_bool("MEALFIT_STEP_UNIT_PLURAL_FIX", True)',
])
def test_knobs_de_rollback(linea):
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert linea in src
