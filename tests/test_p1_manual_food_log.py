"""[P1-MANUAL-FOOD-LOG · 2026-08-11] El registro manual de comida — Fase 1 del modo
seguimiento, útil desde el día uno también para quien SÍ tiene plan (comió algo en la
calle y hoy solo puede anotarlo con una foto).

Criterio de la fase: anotar una comida sin foto, sin chat y sin gastar un crédito.

Lo que estos casos anclan, por sección:
  1. `resolve_line` — la aritmética corre server-side desde REFERENCIAS, nunca desde
     macros del cliente (doctrina de `consumed-from-plan`).
  2. LA REGLA DEL YIELD — la serialización a Nevera va en gramos CRUDOS con el nombre
     EXACTO del catálogo, sin adjetivos de cocción. `_calculate_yield_multiplier`
     convertiría «180 g de habichuelas cocidas» en 63 g: la Nevera bajaría un tercio
     de lo debido, en silencio.
  3. Fail-atómico: una ref irresoluble tumba la comida ENTERA (422), nunca a medias.
  4. El endpoint reusa `_persist_consumed_meal` — el mismo camino que la foto, con
     `source='manual'` en el ledger (migración p1_manual_food_log_ledger_source).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import food_search as fs

_BACKEND = Path(__file__).resolve().parents[1]
_DIARY = _BACKEND / "routers" / "diary.py"

CATALOGO = [
    {"id": "11", "name": "Arroz blanco", "kcal_per_100g": 358.6, "protein_g_per_100g": 7.04,
     "carbs_g_per_100g": 80.3, "fats_g_per_100g": 1.03, "density_g_per_cup": 185.0,
     "density_g_per_unit": None},
    {"id": "22", "name": "Plátano verde", "kcal_per_100g": 152.4, "protein_g_per_100g": 1.25,
     "carbs_g_per_100g": 36.7, "fats_g_per_100g": 0.07, "density_g_per_cup": None,
     "density_g_per_unit": 280.0},
    {"id": "33", "name": "Pechuga de pollo", "kcal_per_100g": 107.4, "protein_g_per_100g": 22.5,
     "carbs_g_per_100g": 0.0, "fats_g_per_100g": 1.93, "density_g_per_cup": None,
     "density_g_per_unit": 170.0},
]


# ─────────────────────────── resolve_line ───────────────────────────

def test_alimento_en_gramos():
    r = fs.resolve_line({"ref": "food:11", "qty": 200, "unit": "g"}, CATALOGO)
    assert r["name"] == "Arroz blanco"
    assert r["grams"] == 200
    assert round(r["macros"]["kcal"]) == round(358.6 * 2)
    assert r["pantry_lines"] == ["200 g de Arroz blanco"]


def test_alimento_en_tazas_usa_la_densidad_del_catalogo():
    # 1 taza de arroz = 185 g SEGÚN LA TABLA, no según un 240 genérico de cocina
    # gringa. Si alguien mete un fallback «razonable», este número deja de cuadrar.
    r = fs.resolve_line({"ref": "food:11", "qty": 1, "unit": "taza"}, CATALOGO)
    assert r["grams"] == 185.0
    assert r["pantry_lines"] == ["185 g de Arroz blanco"]


def test_alimento_en_unidades():
    r = fs.resolve_line({"ref": "food:22", "qty": 2, "unit": "unidad"}, CATALOGO)
    assert r["grams"] == 560.0
    assert round(r["macros"]["kcal"]) == round(152.4 * 5.6)


def test_unidad_sin_densidad_NO_se_adivina():
    # El arroz no tiene peso por unidad. Adivinarlo sería drift contra el catálogo.
    with pytest.raises(fs.LineaIrresoluble):
        fs.resolve_line({"ref": "food:11", "qty": 1, "unit": "unidad"}, CATALOGO)


def test_ref_inexistente_lanza_no_devuelve_a_medias():
    with pytest.raises(fs.LineaIrresoluble):
        fs.resolve_line({"ref": "food:999", "qty": 100, "unit": "g"}, CATALOGO)
    with pytest.raises(fs.LineaIrresoluble):
        fs.resolve_line({"ref": "dish:no-existe", "qty": 1, "unit": "racion"}, CATALOGO)


def test_cantidades_absurdas_se_rechazan():
    with pytest.raises(fs.LineaIrresoluble):
        fs.resolve_line({"ref": "food:11", "qty": 0, "unit": "g"}, CATALOGO)
    with pytest.raises(fs.LineaIrresoluble):
        fs.resolve_line({"ref": "food:11", "qty": 99999, "unit": "g"}, CATALOGO)
    with pytest.raises(fs.LineaIrresoluble):
        fs.resolve_line({"ref": "food:22", "qty": 51, "unit": "unidad"}, CATALOGO)


def test_custom_declara_y_no_toca_nevera():
    # La misma autoridad que ya tiene el escáner: el usuario declara. Pero lo que el
    # catálogo no conoce NO baja la Nevera — no hay fila que restar.
    r = fs.resolve_line({"ref": "custom", "qty": 1, "unit": "g", "name": "Empanada de la esquina",
                         "macros": {"kcal": 380, "protein": 9, "carbs": 40, "fats": 20}}, CATALOGO)
    assert r["name"] == "Empanada de la esquina"
    assert r["macros"]["kcal"] == 380
    assert r["pantry_lines"] == []


def test_custom_respeta_los_mismos_topes_que_la_via_de_fotos():
    # `ConsumedMealRequest` rechaza kcal>10000 en la frontera HTTP; la vía manual no
    # puede ser la puerta de atrás para contaminar agregados con NaN o millones.
    r = fs.resolve_line({"ref": "custom", "qty": 1, "unit": "g", "name": "X",
                         "macros": {"kcal": 99999, "protein": float("nan"), "carbs": -5, "fats": 3}}, CATALOGO)
    assert r["macros"]["kcal"] == 10000.0
    assert r["macros"]["protein"] == 0.0
    assert r["macros"]["carbs"] == 0.0


# ─────────────────────────── los platos curados ───────────────────────────

def test_plato_una_racion_es_finished_g():
    dishes = fs.load_dishes()
    # [reapuntado 2026-08-23, P1-COUNTRY-DIARY-DISHES-60-DE-60-DO] El catálogo dejó de ser
    # sólo criollo: se le unieron los platos de los cuatro países beta (245 en total). El
    # número exacto es un dato que crece; lo que este test mide es OTRA cosa —que una ración
    # sea `finished_g`—, así que se ancla el suelo y se deja que el catálogo crezca.
    assert len(dishes) >= 60, "el catálogo de platos perdió los 60 criollos"
    slug = "moro"
    d = dishes[slug]
    r = fs.resolve_line({"ref": f"dish:{slug}", "qty": 1, "unit": "racion"}, CATALOGO)
    assert r["grams"] == d["finished_g"]
    esperado = d["per_100g"]["kcal"] * d["finished_g"] / 100.0
    assert abs(r["macros"]["kcal"] - esperado) < 0.5


def test_plato_serializa_constituyentes_CRUDOS_con_nombre_de_catalogo():
    """LA REGLA DEL YIELD, con los datos reales. Cada línea de Nevera de un plato debe
    ser el constituyente crudo tal cual lo nombra el catálogo. Si alguien "mejora" la
    serialización añadiendo el método de cocción («guisadas», «hervido»),
    `_calculate_yield_multiplier` re-convertirá cocido→crudo un gramaje QUE YA ES
    crudo, y la Nevera bajará un tercio de lo debido sin que nadie lo vea."""
    r = fs.resolve_line({"ref": "dish:moro", "qty": 1, "unit": "racion"}, CATALOGO)
    assert r["pantry_lines"], "el moro no expandió constituyentes"
    patron = re.compile(r"^\d+ g de .+$")
    ADJETIVOS = ("cocid", "hervid", "guisad", "frit", "asad", "horneada")
    for linea in r["pantry_lines"]:
        assert patron.match(linea), f"línea fuera de formato: {linea!r}"
        assert not any(a in linea.lower() for a in ADJETIVOS), (
            f"la línea {linea!r} lleva adjetivo de cocción: el yield la recortaría"
        )
    juntos = " · ".join(r["pantry_lines"])
    assert "Arroz blanco" in juntos and "Habichuelas rojas" in juntos


def test_media_racion_escala_constituyentes():
    entera = fs.resolve_line({"ref": "dish:moro", "qty": 1, "unit": "racion"}, CATALOGO)
    media = fs.resolve_line({"ref": "dish:moro", "qty": 0.5, "unit": "racion"}, CATALOGO)
    g_entera = int(entera["pantry_lines"][0].split(" ")[0])
    g_media = int(media["pantry_lines"][0].split(" ")[0])
    assert abs(g_media * 2 - g_entera) <= 1
    assert abs(media["macros"]["kcal"] * 2 - entera["macros"]["kcal"]) < 1


def test_la_vista_del_cliente_NO_lleva_constituyentes():
    # La deducción es server-side; mandar la receta entera al cliente es peso muerto y
    # una invitación a que alguien la use para deducir client-side (I6 en espíritu).
    vista = fs.dishes_for_client()
    # Mismo reapuntado: la vista del cliente sigue a la de servidor, y ambas crecen.
    assert len(vista) == len(fs.load_dishes())
    assert all("constituents" not in p for p in vista)
    assert all(p.get("finished_g") for p in vista)


# ─────────────────────────── utilidades ───────────────────────────

def test_merge_suma_duplicados():
    # El moro y el locrio comparten arroz: DOS líneas deben llegar como UNA resta.
    out = fs.merge_pantry_lines(["60 g de Arroz blanco", "45 g de Habichuelas rojas",
                                 "80 g de Arroz blanco"])
    assert out == ["140 g de Arroz blanco", "45 g de Habichuelas rojas"]


def test_derive_meal_name():
    assert fs.derive_meal_name(["Moro"]) == "Moro"
    assert fs.derive_meal_name(["Moro", "Pollo guisado"]) == "Moro y Pollo guisado"
    assert fs.derive_meal_name(["A", "B", "C"]) == "A, B y C"
    assert len(fs.derive_meal_name(["X" * 300])) <= 200


# ─────────────────────────── anclas del endpoint ───────────────────────────

def test_el_camino_de_escritura_se_extrajo_no_se_copio():
    """`_persist_consumed_meal` debe existir y los DOS endpoints (foto y manual) deben
    llamarlo. Ahí viven el fail-loud, el sentinel "deduped", el atado al ledger y la
    forma de la respuesta; una copia garantiza que en tres meses uno tenga el fix y el
    otro no — el modo de fallo exacto que este repo ya pagó con los 4 prompts del chat."""
    src = _DIARY.read_text(encoding="utf-8")
    assert "def _persist_consumed_meal(" in src, "no existe el camino de escritura común"
    cuerpo_foto = src[src.index("def api_log_consumed_meal("):src.index("def api_log_consumed_meal_from_plan(")]
    assert "_persist_consumed_meal(" in cuerpo_foto, "/consumed dejó de usar el camino común"
    assert "def api_log_manual_meal(" in src, "no existe el endpoint manual"
    cuerpo_manual = src[src.index("def api_log_manual_meal("):]
    cuerpo_manual = cuerpo_manual[:cuerpo_manual.index("\n@router") if "\n@router" in cuerpo_manual else len(cuerpo_manual)]
    assert "_persist_consumed_meal(" in cuerpo_manual, "el manual no usa el camino común"


def test_source_manual_en_el_ledger():
    """El descuento del registro manual viaja con `source='manual'`. Sin la fila en el
    CHECK del ledger el INSERT falla, db_inventory se lo traga con un warning, y
    «Deshacer registro» deja de devolver comida a la Nevera — en silencio."""
    src = _DIARY.read_text(encoding="utf-8")
    i = src.index("def api_log_manual_meal(")
    assert 'source="manual"' in src[i:i + 6000] or "source='manual'" in src[i:i + 6000]
    # Y la migración existe en LOS DOS directorios (P3-MIGRATIONS-SSOT).
    for base in (_BACKEND / "migrations", _BACKEND.parent / "migrations"):
        f = base / "p1_manual_food_log_ledger_source_2026_08_11.sql"
        assert f.exists(), f"falta la migración en {base.name}/"
        assert "'manual'" in f.read_text(encoding="utf-8")


def test_los_endpoints_manuales_son_quota_exempt():
    """Cero costo LLM ⇒ RateLimiter, no paywall: al cap, el usuario tiene que poder
    seguir anotando lo que come (doctrina Historial-quota-exemption, misma que
    /consumed-from-plan). `verify_api_quota` aquí quemaría crédito de PLANES por comer."""
    src = _DIARY.read_text(encoding="utf-8")
    for fn in ("def api_log_manual_meal(", "def api_frequent_foods(", "def api_repeat_consumed_meal("):
        assert fn in src, f"falta {fn}"
        cuerpo = src[src.index(fn):src.index(fn) + 900]
        assert "verify_api_quota" not in cuerpo, f"{fn} quedó detrás del paywall"
        assert "LIMITER" in cuerpo, f"{fn} sin RateLimiter: spam sin freno"
