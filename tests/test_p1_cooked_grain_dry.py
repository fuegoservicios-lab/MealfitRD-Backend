"""[P1-COOKED-GRAIN-DRY · 2026-07-24] Gramos COCIDOS resueltos contra la fila SECA del catálogo.

Auditoría del plan vivo `732588f8`: `"65g de arroz blanco cocido"` → 233 kcal donde hay ~85
(**2.76×**). TODAS las filas de granos/legumbres del catálogo están en seco (verificado en Neon:
arroz blanco 358.6, habichuelas rojas 344.7, lentejas 361.5, pasta integral 370.5 kcal/100g), y el
normalizador de nombres pela "cocido"/"cocida" (constants.py:2274) antes de resolver.

El motor se contradice a sí mismo: el kcal-floor de gain_muscle calcula su delta con
`_GM_RICE_KCAL_G = 1.3` kcal/g (arroz COCIDO) y emite el string `"{g}g de arroz blanco cocido"`,
que después resuelve a 3.586 kcal/g. Dos números para el mismo arroz en el mismo pipeline. Y el
prompt del generador (L4941) instruye a la LLM en kcal COCIDAS — el desajuste es sistemático.

El fix convierte la línea a los gramos SECOS equivalentes: las unidades que el catálogo habla. Así
quedan bien de una vez macros, lista de compras, precio y PDF, sin tocar el resolver.

⚠️ ORDEN: este pase es requisito de #4 de la auditoría (densidades de limón/fresas). Abrir el
abort-gate de `_truth_up_meal_macros_from_strings` con estas líneas rotas presentes salta el día 1
a 2570 kcal (+22%).
"""
import pytest

import graph_orchestrator as go


FAKE_KCAL = {
    "arroz blanco": 358.6, "arroz integral": 365.6, "arroz": 358.6,
    "habichuelas rojas": 344.7, "lentejas": 361.5, "pasta integral": 370.5,
    "pollo": 165.0, "pechuga de pollo": 165.0,
}


@pytest.fixture(autouse=True)
def _inject_kcal(monkeypatch):
    monkeypatch.setattr(go, "_COOKED_CATALOG_KCAL_CACHE", dict(FAKE_KCAL), raising=False)
    yield
    monkeypatch.setattr(go, "_COOKED_CATALOG_KCAL_CACHE", None, raising=False)


def _days(*lines):
    return [{"day": 1, "meals": [{"name": "Bowl", "ingredients": list(lines),
                                  "ingredients_raw": list(lines)}]}]


# ───────────── 1. la conversión ─────────────

def test_arroz_cocido_del_plan_vivo():
    days = _days("65g de arroz blanco cocido")
    out = go._normalize_cooked_grain_lines(days)
    assert len(out) == 2, "una reescritura por lista (display + raw)"
    line = days[0]["meals"][0]["ingredients"][0]
    # 65 g cocidos × (130 / 358.6) ≈ 23.6 g secos
    assert line == "24 g de arroz blanco crudo", line
    # El error que cerramos: 65 g contra la fila seca daban 233 kcal.
    assert abs(24 * 3.586 - 65 * 1.30) < 15, "los gramos secos deben reproducir las kcal cocidas"


def test_kcal_antes_vs_despues():
    """La cifra que motivó el fix, explícita para que nadie la 'optimice' de vuelta."""
    antes = 65 / 100 * 358.6          # línea original resuelta contra la fila seca
    despues = 24 / 100 * 358.6        # línea reescrita
    assert round(antes) == 233 and round(despues) == 86
    assert antes / despues > 2.5


@pytest.mark.parametrize("line,expected_prefix", [
    ("120 g de habichuelas rojas cocidas", "44 g de habichuelas rojas crud"),
    ("90 g de lentejas hervidas", "32 g de lentejas crud"),
    ("100 g de pasta integral cocida", "43 g de pasta integral crud"),
])
def test_otras_clases(line, expected_prefix):
    days = _days(line)
    go._normalize_cooked_grain_lines(days)
    assert days[0]["meals"][0]["ingredients"][0].startswith(expected_prefix)


def test_concordancia_de_genero_y_numero():
    """El usuario lee esta línea en la app y en el PDF: 'habichuelas rojas crudo' canta."""
    days = _days("120 g de habichuelas rojas cocidas", "65 g de arroz blanco cocido")
    go._normalize_cooked_grain_lines(days)
    ings = days[0]["meals"][0]["ingredients"]
    assert ings[0].endswith("crudas"), ings[0]
    assert ings[1].endswith("crudo"), ings[1]


# ───────────── 2. lo que NO debe tocar ─────────────

def test_no_toca_proteina_cocida():
    """La fila de pollo del catálogo ya está en cocido/crudo comparable (ratio < 1.5):
    el factor se auto-desactiva en vez de doble-corregir."""
    days = _days("150 g de pechuga de pollo cocida")
    assert go._normalize_cooked_grain_lines(days) == []
    assert days[0]["meals"][0]["ingredients"][0] == "150 g de pechuga de pollo cocida"


def test_no_toca_lineas_sin_estado():
    days = _days("65 g de arroz blanco", "270 g de mero")
    assert go._normalize_cooked_grain_lines(days) == []


def test_no_toca_volumenes():
    """'1 taza de arroz cocido' es volumen: `density_g_per_cup` del catálogo es de arroz SECO,
    así que convertir gramos ahí exigiría otra cadena. Limitación consciente y acotada."""
    days = _days("1 taza de arroz blanco cocido")
    assert go._normalize_cooked_grain_lines(days) == []


def test_idempotente():
    days = _days("65g de arroz blanco cocido")
    assert len(go._normalize_cooked_grain_lines(days)) == 2
    assert go._normalize_cooked_grain_lines(days) == [], "la línea ya dice 'crudo': no re-matchea"


def test_reescribe_raw_aunque_las_listas_esten_desalineadas():
    """[P1-PHANTOM-RAW-PARITY · 2026-07-24] El shopping calculator lee `ingredients_raw`
    PRIMERO. En el plan vivo 732588f8 las listas NO están alineadas por índice (Casabe: 4 vs 5),
    así que reescribir sólo cuando coincidían los largos dejaba la lista comprando 2.76× de
    arroz justo en las comidas desalineadas. Cada lista se procesa por separado."""
    days = [{"day": 1, "meals": [{
        "name": "Casabe Tropical",
        "ingredients": ["2 tortas pequeño de casabe", "65 g de queso cottage"],
        "ingredients_raw": ["65g de arroz blanco cocido", "42.17g de queso cottage",
                            "2 tortas pequeño de casabe"],
    }]}]
    go._normalize_cooked_grain_lines(days)
    assert days[0]["meals"][0]["ingredients_raw"][0] == "24 g de arroz blanco crudo", (
        "la línea que compra la lista tiene que quedar en gramos secos"
    )



def test_sin_parentesis_numerico_en_la_salida():
    """Varios parsers leen `alimento (N g)` como cantidad autoritativa: escribir
    '(rinde ~65 g cocido)' reintroduciría el número que acabamos de corregir."""
    days = _days("65g de arroz blanco cocido")
    go._normalize_cooked_grain_lines(days)
    assert "(" not in days[0]["meals"][0]["ingredients"][0]


# ───────────── 3. cableado ─────────────

def test_corre_antes_de_la_lista_y_despues_del_kcal_floor():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i_floor = src.index("[P1-GAINMUSCLE-KCAL-FLOOR] en assemble falló")
    i_rewrite = src.index("_normalize_cooked_grain_lines(result.get(\"days\")")
    i_list = src.index("# Calcular shopping lists")
    assert i_floor < i_rewrite < i_list, (
        "debe correr DESPUÉS del kcal-floor (uno de los escritores de la línea cocida) "
        "y ANTES de la lista de compras (que tiene que comprar gramos secos)"
    )


def test_knob_de_rollback():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'COOKED_GRAIN_DRY_REWRITE = _env_bool("MEALFIT_COOKED_GRAIN_DRY_REWRITE", True)' in src
