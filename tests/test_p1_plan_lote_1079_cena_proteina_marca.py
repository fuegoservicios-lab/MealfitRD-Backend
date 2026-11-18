"""[P1-PLAN-LOTE-1079 · 2026-11-18] Tres ajustes tras el día de la cuenta de prueba (17-nov, ganar músculo, 3350 kcal):
1. el aviso de la cena con ~1246 kcal por cubrir propuso ~500 kcal → la cena a la medida de lo que queda;
2. con la cena llegó a 162 de 157 g de proteína y nadie lo dijo → el registro que cruza la meta se celebra una vez;
3. «yogurt sissys» se estimó genérico (5 g de proteína) → la marca parecida se reconoce (Siggi's, 15 g)."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import aviso_del_dia as ad
import cena_a_la_medida as cm
import marca_parecida as mp
from cierre_del_dia import nota_proteina_cumplida

RAIZ = Path(__file__).resolve().parents[1]


# ── 1. la cena a la medida ──────────────────────────────────────────────────────────────────────────────
def test_caso_ganar_musculo_con_1246_por_cubrir():
    r = cm.regla("Cena", 1246, "gain_muscle")
    assert "P1-PLAN-LOTE-1079" in r and "~1246 kcal" in r and "~1100 kcal" in r
    assert "NO una cena ligera" in r and "batido o un vaso de leche" in r


@pytest.mark.parametrize("comida,quedan,objetivo,esperado", [
    ("Cena", 700, "lose_fat", "~700 kcal"),
    ("Cena", 1400, "lose_fat", "~750 kcal"),
    ("Cena", 900, "maintenance", "~900 kcal"),
])
def test_el_plato_tiene_techo_por_objetivo_y_no_empuja_a_recuperar(comida, quedan, objetivo, esperado):
    r = cm.regla(comida, quedan, objetivo)
    assert f"Propón un plato de {esperado}" in r and "Sin decirle que «recupere» el día" in r


@pytest.mark.parametrize("comida,quedan", [("Cena", 599), ("Almuerzo", 1500), ("Merienda", 1200), ("Cena", None), (None, 900)])
def test_solo_la_cena_y_con_margen_grande(comida, quedan):
    assert cm.regla(comida, quedan, "gain_muscle") == ""


def test_engancha_en_el_bloque_del_aviso(monkeypatch):
    monkeypatch.setattr(ad, "_metas", lambda h: (3350, 157))
    consumido = [{"meal_type": "desayuno", "meal_name": "Pan con huevos", "calories": 850, "protein": 57},
                 {"meal_type": "extra", "meal_name": "Mass gainer", "calories": 1254, "protein": 57}]
    b = ad.bloque_del_dia(consumido, {"mainGoal": "gain_muscle"}, "Cena")
    assert "CENA A LA MEDIDA" in b and "~1246 kcal" in b
    assert "CENA A LA MEDIDA" not in ad.bloque_del_dia(consumido, {"mainGoal": "gain_muscle"}, "Almuerzo")


# ── 2. proteína cumplida ────────────────────────────────────────────────────────────────────────────────
T0 = datetime(2026, 11, 18, 2, 7, tzinfo=timezone.utc)
METAS = {"kcal": 3350, "protein_g": 157}


def _f(prot, minutos_antes):
    return {"protein": prot, "created_at": T0 - timedelta(minutes=minutos_antes)}


def test_el_registro_que_cruza_la_meta_se_celebra():
    filas = [_f(57, 800), _f(15, 400), _f(42, 300), _f(48, 0)]   # 114 → 162 con la cena
    n = nota_proteina_cumplida(METAS, filas)
    assert "PROTEÍNA CUMPLIDA" in n and "~162 de ~157 g" in n


def test_una_sola_vez_los_siguientes_ya_no_cruzan():
    filas = [_f(57, 800), _f(15, 400), _f(42, 300), _f(48, 60), _f(1, 0)]   # la piña: 162 → 163
    assert nota_proteina_cumplida(METAS, filas) == ""


def test_sin_llegar_sin_meta_o_sin_filas_nada():
    assert nota_proteina_cumplida(METAS, [_f(57, 10), _f(15, 0)]) == ""
    assert nota_proteina_cumplida({"protein_g": None}, [_f(200, 0)]) == ""
    assert nota_proteina_cumplida(METAS, []) == ""
    assert nota_proteina_cumplida(None, None) == ""


def test_la_fila_sin_created_at_cuenta_como_la_ultima():
    assert "PROTEÍNA CUMPLIDA" in nota_proteina_cumplida(METAS, [_f(120, 30), {"protein": 40}])


def test_engancha_en_el_total_del_dia():
    src = (RAIZ / "tools.py").read_text(encoding="utf-8")
    assert '__import__("cierre_del_dia").nota_proteina_cumplida(_metas, _rows)' in src


# ── 3. la marca parecida ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("texto,palabra,marca", [
    ("yogurt sissys", "sissys", "Siggi's"),
    ("yogurt siggy", "siggy", "Siggi's"),
    ("barra quets de chocolate", "quets", "Quest"),
    ("yogurt yoplay", "yoplay", "Yoplait"),
])
def test_reconoce_la_marca_mal_escrita(texto, palabra, marca):
    assert mp.detectar(texto) == (palabra, marca)
    b = mp.bloque_para_chat(texto)
    assert f"«{palabra}» casi seguro es «{marca}»" in b and "no como un producto genérico" in b


@pytest.mark.parametrize("texto", [
    "yogurt siggi's", "Yogur Oikos Pro", "yogur griego natural", "me comí un yogurt de fresa", "batido primer de vainilla",
    "leche con queso", "me comí arroz con habichuelas", "sissys", "Merendé esto\n1 vaso · Yogur Activa", "", None,
])
def test_no_inventa_marcas(texto):
    assert mp.detectar(texto) is None and mp.bloque_para_chat(texto) == ""


def test_las_claves_estan_normalizadas():
    assert all(k == mp._norm(k) for k in mp.MARCAS)


def test_marca_y_lo_mismo_enganchan_en_el_prompt_del_chat():
    src = (RAIZ / "agent.py").read_text(encoding="utf-8")
    i = src.index('__import__("lo_mismo_de_antes").bloque_para_chat(user_id, prompt, tz_offset)')
    assert '__import__("marca_parecida").bloque_para_chat(prompt)' in src[i:i + 300]
