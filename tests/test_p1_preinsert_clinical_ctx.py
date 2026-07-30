"""[P1-PREINSERT-CLINICAL-CTX · 2026-07-30] El shield pre-INSERT corría sin contexto clínico.

Cazado EN PRODUCCIÓN por el warning que `P1-UPDATE-RECAP-ALL-SURFACES` había subido de `debug`
a `warning` una hora antes, durante una renovación real del owner:

    ⚠️ [P1-UPDATE-CLINICAL-RECAP] surface=form_gen_final_closer sin form_data
       — re-cap clínico omitido en 1 día(s) que el motor acaba de re-dimensionar.

Aquel fix cableó `form_data` a las 4 superficies que llamaban al motor sin él, y para la del
shield pre-INSERT usó `(_pd.get("form_data") or data.get("form_data") or {})` — la MISMA
expresión que el recompute de micros de 10 líneas más abajo ya usaba, asumiendo que si estaba
ahí es que traía algo.

No traía nada. Medido en Neon: **31 planes en 7 días, 0 con `form_data` dentro de `plan_data`**;
y `data` solo lleva `plan_data`/`user_id`. O sea el fix quedó INERTE en esa superficie, y el
recompute de micros pre-INSERT llevaba desde su día corriendo sin condiciones (los techos por
condición —sodio en HTA, potasio en ERC— no se aplicaban en esa pasada).

La corrección no es otra expresión sobre las mismas dos fuentes vacías: es ir al perfil por
`user_id`, que sí está. Y como esta hidratación ya estaba copiada a mano en 3 sitios
(swap-persist, swap-persist-day, chat-modify), se extrae a UN helper — la cuarta copia era la
señal de que faltaba el SSOT.
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel: str) -> str:
    with open(os.path.join(_BACKEND, *rel.split("/")), encoding="utf-8") as f:
        return f.read()


_DBP = _read("db_plans.py")


# ═════════════ el helper SSOT ═════════════

def test_helper_hidrata_desde_el_perfil(monkeypatch):
    import db_profiles
    monkeypatch.setattr(db_profiles, "get_user_profile", lambda uid: {"health_profile": {
        "gender": "female", "age": 41,
        "medicalConditions": ["Diabetes Tipo 2"],
        "otherConditions": "hipotiroidismo",
        "medications": ["metformina"],
        "allergies": [" maní ", ""],
        "dietType": "balanceada",
        "dislikes": ["cilantro"],
    }})
    out = db_profiles.build_clinical_form_from_profile("u1")
    assert "Diabetes Tipo 2" in out["medicalConditions"]
    assert "hipotiroidismo" in out["medicalConditions"], (
        "el free-text clínico debe plegarse en medicalConditions (mismo merge que las otras "
        "superficies): un ERC declarado a mano tiene que activar el renal-skip")
    assert out["allergies"] == ["maní"], "alergias normalizadas y sin vacíos"
    assert out["gender"] == "female" and out["age"] == 41
    # [P2-MICRO-SEED-CLINICAL-CTX] las 3 claves del scan de alérgenos van SIEMPRE presentes:
    # "ausente" no es lo mismo que "sin alergias" — el scan se gatea por presencia de key.
    for k in ("allergies", "dietType", "dislikes"):
        assert k in out


def test_helper_es_fail_open(monkeypatch):
    """Nunca puede tumbar un INSERT: sin user_id, sin perfil o con la query rota → {}."""
    import db_profiles

    def _boom(_uid):
        raise RuntimeError("DB down")

    monkeypatch.setattr(db_profiles, "get_user_profile", _boom)
    assert db_profiles.build_clinical_form_from_profile("u1") == {}
    monkeypatch.setattr(db_profiles, "get_user_profile", lambda uid: None)
    assert db_profiles.build_clinical_form_from_profile("u1") == {}
    assert db_profiles.build_clinical_form_from_profile(None) == {}
    assert db_profiles.build_clinical_form_from_profile("") == {}


def test_helper_sin_condiciones_devuelve_dict_utilizable(monkeypatch):
    """Un perfil sano no debe producir `{}` (que el motor lee como 'no sé'): produce un dict
    con las claves presentes y vacías, que es la afirmación honesta 'miré y no hay nada'."""
    import db_profiles
    monkeypatch.setattr(db_profiles, "get_user_profile",
                        lambda uid: {"health_profile": {"gender": "male", "age": 30}})
    out = db_profiles.build_clinical_form_from_profile("u1")
    assert out and out["medicalConditions"] == [] and out["allergies"] == []


# ═════════════ el shield lo usa (para el motor Y para los micros) ═════════════

def _shield_block() -> str:
    i = _DBP.index("P1-COHERENCE-FINALIZE] Aplicando coherencia") if \
        "P1-COHERENCE-FINALIZE] Aplicando coherencia" in _DBP else _DBP.index("def _finalize_plan_data_for_insert(")
    return _DBP[i:_DBP.index("\ndef ", i + 10)]


def _clin_helper_alias() -> str:
    """Alias con el que `db_plans` importa el helper. El repo importa con alias por convención,
    así que buscar el nombre pelado dentro del cuerpo da 0 matches y el test pasaría en vacío —
    el mismo tropiezo que ya costó rojos en esta sesión, tres veces."""
    key = "build_clinical_form_from_profile as "
    i = _DBP.index(key)
    return _DBP[i + len(key):_DBP.index(")", i)].strip().rstrip(",")


def test_el_shield_deriva_el_contexto_clinico_del_perfil():
    blk = _shield_block()
    assert f"{_clin_helper_alias()}(" in blk, (
        "el shield sigue confiando en dos fuentes que NUNCA traen nada "
        "(`plan_data['form_data']` y `data['form_data']`): 0 de 31 planes las tienen")


def test_el_recompute_de_micros_usa_el_mismo_contexto():
    """El recompute de micros pre-INSERT sufría el MISMO `{}`: sin condiciones, los techos por
    condición (sodio/HTA, potasio/ERC) no se aplican en esa pasada."""
    blk = _shield_block()
    i_micro = blk.index("recompute_micronutrient_report_for_plan as _rmr")
    linea = blk[i_micro:blk.index("\n", blk.index("_rmr(", i_micro))]
    assert "_clin_ctx" in linea or "build_clinical_form_from_profile" in linea, (
        f"el recompute de micros sigue recibiendo el dict vacío: {linea.strip()!r}")


def test_el_contexto_se_resuelve_una_sola_vez():
    """La hidratación consulta el perfil: resolverla por cada consumidor duplicaría la query
    dentro del path más caliente del INSERT."""
    blk = _shield_block()
    assert blk.count(f"{_clin_helper_alias()}(") == 1, (
        "el contexto clínico debe resolverse UNA vez y reutilizarse")
    assert blk.count("_clin_ctx") >= 3, "y debe consumirse desde la variable, no re-derivarse"
