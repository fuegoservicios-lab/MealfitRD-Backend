"""[P1-PLAN-LOTE-166 · 2026-09-22] Lo que el usuario ESCRIBE a mano en el formulario llega a quien lo cuida.

El formulario guarda los chips en `allergies` / `medicalConditions` / `medications` / `dislikes` y lo tecleado en
«Otra…» aparte (`otherAllergies`, `otherConditions`, `otherMedications`, `otherDislikes`). El generador de planes los
une al empezar (`_merge_other_text_fields`), pero el perfil GUARDADO no los une nunca, y cada lector que va al perfil
guardado veía solo los chips:

  · el bloque clínico del coach (`build_clinical_guard_context`) — «Maní» tecleado no llegaba al chat;
  · `suggest_foods_for_nutrient` y `proponer_comida` — las dos herramientas con las que el coach recomienda comida
    en el modo contador, que es justo lo que usan los testers de la beta;
  · el camino degradado del generador (`cron_tasks`, Smart Shuffle/Edge Recipes), que arma días SIN LLM.

Medido en producción (solo lectura, 22-sep): 1 de 9 perfiles con texto libre fuera del array
(`allergies=["Mariscos"]`, `otherAllergies="Shrimp"`). El arreglo NO inventa otra regla de unión: todos pasan por
`graph_orchestrator.profile_with_free_text`, que es el MISMO `_merge_other_text_fields` del generador (centinela
«Ninguna» incluido). Además, el cron de arranque en frío (`get_similar_user_patterns`) leía `dietTypes`, que el
formulario no escribe (0 de 9 perfiles): la segmentación por dieta estaba muerta.

Tooltip-anchor: P1-PLAN-LOTE-166
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _leer(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ─────────────────────────── el SSOT de la unión ───────────────────────────

def test_la_union_es_la_del_generador_y_no_toca_el_original():
    from graph_orchestrator import profile_with_free_text
    perfil = {"allergies": ["Mariscos"], "otherAllergies": "Shrimp, maní",
              "medicalConditions": [], "otherConditions": "Gota",
              "dislikes": ["Cebolla"], "otherDislikes": "apio"}
    out = profile_with_free_text(perfil)
    assert out["allergies"] == ["Mariscos", "Shrimp", "maní"]
    assert out["medicalConditions"] == ["Gota"]
    assert out["dislikes"] == ["Cebolla", "apio"]
    # el perfil del llamador queda intacto (es el que se guarda y el que pinta el formulario)
    assert perfil["allergies"] == ["Mariscos"] and perfil["otherAllergies"] == "Shrimp, maní"


def test_el_centinela_ninguna_manda_como_en_el_generador():
    from graph_orchestrator import profile_with_free_text
    out = profile_with_free_text({"allergies": ["Ninguna"], "otherAllergies": "Maní"})
    assert out["allergies"] == ["Ninguna"]


def test_perfil_vacio_o_raro_no_revienta():
    from graph_orchestrator import profile_with_free_text
    assert profile_with_free_text(None) == {}
    assert profile_with_free_text({}) == {}
    # un CSV legacy sale como lista (lo normaliza el mismo pliegue del generador), nunca se pierde
    assert profile_with_free_text({"allergies": "Lacteos", "otherAllergies": ""})["allergies"] == ["Lacteos"]


# ─────────────────────────── el coach ───────────────────────────

def test_el_bloque_clinico_del_coach_ve_lo_tecleado():
    from prompts.chat_agent import build_clinical_guard_context as bloque
    out = bloque({
        "allergies": ["Mariscos"], "otherAllergies": "Maní, kiwi",
        "medicalConditions": [], "otherConditions": "Gota",
        "medications": [], "otherMedications": "Alopurinol",
    })
    for dato in ("Mariscos", "Maní", "kiwi", "Gota", "Alopurinol"):
        assert dato in out, f"«{dato}» se declaró en el formulario y el coach no lo recibe"


def test_solo_texto_libre_ya_no_es_un_perfil_sin_alergias():
    """Antes devolvía "" — para el coach, «no ha declarado nada» — con una alergia escrita en el formulario."""
    from prompts.chat_agent import build_clinical_guard_context as bloque
    out = bloque({"allergies": [], "otherAllergies": "Maní"})
    assert "Maní" in out and "NUNCA" in out
    # y los centinelas siguen sin ser datos clínicos
    assert bloque({"allergies": ["Ninguna"], "medicalConditions": ["ninguno"]}) == ""
    assert bloque({"medications": [], "otherMedications": "  "}) == ""


def _load_tools():
    try:
        import tools as tools_mod  # noqa
        import shopping_calculator  # noqa
        return tools_mod, shopping_calculator
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tools/shopping_calculator no importable: {e}")


def test_sugerir_alimentos_filtra_lo_tecleado(monkeypatch):
    tools_mod, shopping_calculator = _load_tools()
    catalogo = [
        {"name": "Sardinas en lata", "calcium_mg_per_100g": 382.0},
        {"name": "Brocoli", "calcium_mg_per_100g": 47.0},
        {"name": "Espinaca", "calcium_mg_per_100g": 99.0},
    ]
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: catalogo)
    monkeypatch.setattr(tools_mod, "get_user_profile", lambda uid: {"health_profile": {
        "allergies": [], "otherAllergies": "sardinas", "dietType": "balanced"}})
    out = tools_mod.suggest_foods_for_nutrient.func(user_id="u1", nutrient="calcio", top_n=5)
    assert "Sardinas en lata" not in out, "el coach recomienda justo lo que el usuario escribió como alergia"
    assert "Brocoli" in out


def test_proponer_comida_lee_las_restricciones_con_texto_libre():
    import coach_day_context as cdc
    alergias, dieta, excluidos = cdc.restricciones_del_perfil({
        "health_profile": {"allergies": ["Mariscos"], "otherAllergies": "Maní",
                           "dislikes": [], "otherDislikes": "apio", "dietType": "vegana"},
    })
    assert "Maní" in alergias and "Mariscos" in alergias
    assert "apio" in excluidos
    assert dieta == "vegana"


def test_las_demas_lecturas_del_perfil_pasan_por_la_union():
    """Parser: los lectores del perfil GUARDADO que deciden qué comida sale usan la unión, no el chip a secas."""
    tools_src = _leer("tools.py")
    i = tools_src.index("def suggest_foods_for_nutrient(")
    assert "profile_with_free_text(" in tools_src[i:i + 4000]
    j = tools_src.index("    if UPDATE_CLINICAL_GUARD:")
    tramo = tools_src[j:j + 600]
    assert "profile_with_free_text" in tramo and '_clin_allergies = _hp.get("allergies")' not in tramo
    k = tools_src.index('"allergies": [str(a).strip() for a in (')
    assert tools_src[k:k + 90].startswith('"allergies": [str(a).strip() for a in (_pwft_cm(_hp)')
    cron_src = _leer("cron_tasks.py")
    k = cron_src.index("# [GAP C FIX: Filtrar prior_days contra alergias y rechazos actuales]")
    tramo = cron_src[k:k + 2600]
    assert "profile_with_free_text as _pwft_deg" in tramo and "_hp_union = _pwft_deg(health_profile)" in tramo
    assert 'current_allergies = health_profile.get("allergies", [])' not in tramo


# ─────────────────────────── cron de arranque en frío ───────────────────────────

def test_arranque_en_frio_segmenta_por_la_dieta_que_el_formulario_escribe(monkeypatch):
    try:
        import cron_tasks
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cron_tasks no importable: {e}")
    vistas = []

    def _sql(query, params=None, **kw):
        vistas.append((query, params, kw))
        return []

    monkeypatch.setattr(cron_tasks, "execute_sql_query", _sql)
    monkeypatch.setattr(cron_tasks, "get_active_rejections", lambda uid: [])
    cron_tasks.get_similar_user_patterns("u1", {"mainGoal": "lose_fat", "activityLevel": "moderate",
                                               "dietType": "vegana"})
    assert vistas, "no llegó a consultar"
    query, params, kw = vistas[0]
    assert "dietType" in query and "dietTypes" not in query
    assert kw.get("fetch_all") is True
    variantes = [p for p in params if isinstance(p, list)]
    assert variantes and "vegan" in variantes[0] and "vegana" in variantes[0]


def test_una_dieta_sin_restriccion_no_filtra(monkeypatch):
    try:
        import cron_tasks
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cron_tasks no importable: {e}")
    vistas = []
    monkeypatch.setattr(cron_tasks, "execute_sql_query",
                        lambda q, p=None, **kw: vistas.append((q, p)) or [])
    monkeypatch.setattr(cron_tasks, "get_active_rejections", lambda uid: [])
    cron_tasks.get_similar_user_patterns("u1", {"mainGoal": "lose_fat", "activityLevel": "moderate",
                                               "dietType": "balanced"})
    assert vistas and "dietType" not in vistas[0][0]


def test_el_marcador_esta_al_dia():
    src = _leer("app.py")
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', src, re.M)
    assert m and int(m.group(1)) >= 166
