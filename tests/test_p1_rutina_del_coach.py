"""Regresión: [P1-COACH-RUTINA-GUSTOS] Aprendizaje continuo de rutinas y gustos reales por franja."""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
import chat_prompt_opt as cpo
import habitos_de_la_foto as hf
import prompt_desglose as pd
import rutina_del_coach as rc


_HISTORIAL_REAL = [
    {
        "id": "m1",
        "meal_type": "Desayuno",
        "meal_name": "2 sándwiches de pan integral con queso cottage, mantequilla de maní y 4 huevos hervidos",
        "calories": 820,
        "protein": 52,
        "ingredients": ["pan integral", "queso cottage", "mantequilla de maní", "huevos"],
        "consumed_at": "2026-11-12T13:00:00Z",
    },
    {
        "id": "m2",
        "meal_type": "Desayuno",
        "meal_name": "Sándwiches con queso cottage, mantequilla de maní y huevos hervidos",
        "calories": 780,
        "protein": 48,
        "ingredients": ["pan", "queso cottage", "mantequilla de maní", "huevos hervidos"],
        "consumed_at": "2026-11-14T13:10:00Z",
    },
    {
        "id": "m3",
        "meal_type": "Desayuno",
        "meal_name": "Bowl de lechoza, sandía y piña",
        "calories": 86,
        "protein": 1,
        "ingredients": ["lechoza", "sandía", "piña"],
        "consumed_at": "2026-11-15T12:00:00Z",
    },
    {
        "id": "m4",
        "meal_type": "Almuerzo",
        "meal_name": "Arroz blanco con cerdo guisado y vegetales",
        "calories": 630,
        "protein": 33,
        "ingredients": ["arroz blanco", "cerdo guisado", "vegetales"],
        "consumed_at": "2026-11-15T18:00:00Z",
    },
    {
        "id": "m5",
        "meal_type": "Cena",
        "meal_name": "4 sándwiches de jamón y queso",
        "calories": 950,
        "protein": 50,
        "nutrition": {
            "base_meal": {"meal_name": "4 sándwiches de jamón y queso", "calories": 950, "protein": 50},
            "addons": [{"meal_name": "1 jugo de mango", "calories": 290, "protein": 1}],
        },
        "consumed_at": "2026-11-15T23:50:00Z",
    },
    {
        "id": "m6",
        "meal_type": "complemento",
        "meal_name": "Té verde sin azúcar",
        "calories": 2,
        "protein": 0,
        "consumed_at": "2026-11-15T14:00:00Z",
    },
]


def test_extraer_rutina_agrupa_habituales_y_excluye_complementos():
    rutina = rc.extraer_rutina_desde_registros(_HISTORIAL_REAL, facts=[{"fact": "Le gusta desayunar sándwiches con queso cottage"}])
    desayunos = rutina["por_franja"]["desayuno"]
    assert desayunos, "debe extraer desayunos reales"
    assert desayunos[0]["count"] == 2
    assert desayunos[0]["habitual"] is True
    assert "cottage" in desayunos[0]["nombre"].lower()

    # El complemento <= 50 kcal nunca debe entrar en merienda/desayuno
    todas_las_franjas = [p["nombre"].lower() for lista in rutina["por_franja"].values() for p in lista]
    assert all("té verde" not in n for n in todas_las_franjas)

    # Los sándwiches con base_meal limpian el addon circunstancial del nombre de la rutina
    cenas = rutina["por_franja"]["cena"]
    assert cenas and "jugo" not in cenas[0]["nombre"].lower()
    assert len(rutina["habitos"]["rellenos_pan"]) >= 1


def test_para_chat_inyecta_reglas_sin_balanza_ni_resta_manana():
    rc.invalidate_rutina_cache("u-rutina")
    with patch("db_facts.get_consumed_meals_since", return_value=_HISTORIAL_REAL), \
         patch("db_facts.get_all_user_facts", return_value=[]):
        bloque = rc.para_chat("u-rutina", consumed_today=[])
    assert "🔄 RUTINA Y GUSTOS APRENDIDOS DEL USUARIO" in bloque
    assert "PRIORIZA SU RUTINA REAL" in bloque
    assert "PORCIONES CASERAS, NO BALANZA" in bloque
    assert "PROHIBIDO RESTAR DEL PRESUPUESTO DE MAÑANA" in bloque
    assert "2x habitual" in bloque


def test_invalidar_habitos_invalida_rutina_tambien():
    with patch("db_facts.get_consumed_meals_since", return_value=_HISTORIAL_REAL) as mock_m, \
         patch("db_facts.get_all_user_facts", return_value=[]):
        rc.invalidate_rutina_cache("u-cache")
        rc.para_chat("u-cache")
        rc.para_chat("u-cache")
        assert mock_m.call_count == 1
        hf.invalidate_habitos_cache("u-cache")
        rc.para_chat("u-cache")
        assert mock_m.call_count == 2


def test_cache_order_y_prompt_desglose_reconocen_rutina_gustos(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_CACHE_ORDER", "1")
    prefijo = "PREFIJO ESTATICO\n" + ("X" * 80)
    cut = len(prefijo)
    sp = (
        prefijo
        + "\n<!--MEALFIT_VOLATILE_SYS_SPLIT-->\nEl user_id actual es: u1"
        + "\n\n🕒 CONTEXTO TEMPORAL ACTUAL: 01:30"
        + "\n\n🛑 PERFIL CLÍNICO DEL USUARIO (PROTEGER CONTRA ALERGIAS): ok"
        + "\n\nDIARIO DE HOY: 2050 kcal"
        + "\n\n🧊 NEVERA: huevos"
        + "\n\n🔄 RUTINA Y GUSTOS APRENDIDOS DEL USUARIO (aprendizaje continuo de su diario real):\n- Desayuno: pan"
        + "\n\n🍽️ DIARIO REAL DE DÍAS ANTERIORES:\n- Ayer: 2000 kcal"
    )
    rebuilt = cpo.reordenar_bloques_cache(sp, prefijo_estatico_chars=cut)
    assert len(rebuilt) == len(sp)
    assert any(name == "rutina_gustos" for name, _ in pd._VOLATILE_BLOCK_SPECS)


def test_plan_context_en_pausa_prohibe_plan_para_manana_sin_decir_activo(monkeypatch):
    monkeypatch.setattr(agent, "_plan_mode_for_chat", lambda _uid: "tracking")
    bloque = agent._plan_context_for_chat("u1", {"days": [{"day": 1, "meals": []}]})
    assert "mañana" in bloque.lower()
    assert "pausa" in bloque.lower()
    assert "activo" not in bloque.lower()
