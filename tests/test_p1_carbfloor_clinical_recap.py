"""[P1-CARBFLOOR-CLINICAL-RECAP · 2026-07-30] (audit solver+seeder v5 · P1-2)

El re-cap clínico de v4 vive DENTRO de `apply_update_macro_engine` y está gateado a `_hit`
— es decir, solo corre sobre días que el motor RE-DIMENSIONÓ, y el motor solo actúa sobre
días FUERA de banda [0.90, 1.12].

El carb-floor (`_close_carb_gap_for_day`) escala HACIA el target: cuando cierra el hueco,
el día queda EN banda **por construcción**. Consecuencia exacta en swap-persist y chat-modify:

    carb-floor infla la batata ×2.5  →  día queda en banda  →  el motor de macros no-op
    →  `_hit` es False  →  el re-cap clínico NUNCA corre  →  se persiste 300-375 g de
    almidón alto-IG para un DM2.

Y el carb-floor no tiene guard clínico propio: `_close_carb_gap_for_day` no recibe
`form_data`, su único skip es "arroz en la cena", y elige el carbo-dominante MÁS RICO del
día — que en un plan DM2 es justo el que la capa clínica acababa de recortar a 150 g.

En form-gen el orden es seguro (carb-floor en assemble, caps DESPUÉS). En las dos
superficies de update el orden está invertido y nadie re-capea: el `apply_update_macro_engine`
posterior sí recibe form_data pero es no-op por la banda, y `apply_update_condition_ceilings`
mide sodio/grasa saturada/K/Mg/fibra — no la porción de almidón.

Fix: helper SSOT `reapply_clinical_portion_caps` invocado en las 2 superficies TRAS el bloque
closer+carb-floor+requantize. Los dos caps ya son idempotentes (`if grams <= cap: continue`) y
no-op sin la condición, así que es seguro llamarlos incondicionalmente.
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel: str) -> str:
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")
_PLANS = _read(os.path.join("routers", "plans.py"))
_TOOLS = _read("tools.py")


# ═════════════════ 1 · el helper SSOT ═════════════════

def test_helper_exists_and_runs_both_caps(monkeypatch):
    import graph_orchestrator as go
    calls = {"dm2": 0, "bar": 0}
    monkeypatch.setattr(go, "cap_dm2_high_gi_portions",
                        lambda days, fd, db=None, **kw: calls.__setitem__("dm2", calls["dm2"] + 1) or 2)
    monkeypatch.setattr(go, "cap_bariatric_portions",
                        lambda days, fd, db=None, **kw: calls.__setitem__("bar", calls["bar"] + 1) or 1)
    plan = {"days": [{"meals": [{"ingredients": ["300 g de batata"]}]}]}
    n = go.reapply_clinical_portion_caps(plan, {"medicalConditions": ["Diabetes Tipo 2"]},
                                         surface="swap_persist")
    assert n == 3
    assert calls == {"dm2": 1, "bar": 1}


def test_helper_noop_without_form_data(monkeypatch):
    """Sin condición conocida no se puede capear — pero tampoco debe explotar ni tocar el plan."""
    import graph_orchestrator as go
    calls = {"n": 0}
    monkeypatch.setattr(go, "cap_dm2_high_gi_portions",
                        lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1) or 0)
    monkeypatch.setattr(go, "cap_bariatric_portions", lambda *a, **kw: 0)
    assert go.reapply_clinical_portion_caps({"days": []}, None, surface="s") == 0
    assert go.reapply_clinical_portion_caps({"days": []}, {}, surface="s") == 0
    assert calls["n"] == 0


def test_helper_is_fail_safe(monkeypatch):
    """Nunca puede tumbar el mutador: un cap que lanza degrada a 0, no propaga."""
    import graph_orchestrator as go

    def _boom(*a, **kw):
        raise RuntimeError("cap roto")

    monkeypatch.setattr(go, "cap_dm2_high_gi_portions", _boom)
    monkeypatch.setattr(go, "cap_bariatric_portions", lambda *a, **kw: 0)
    assert go.reapply_clinical_portion_caps({"days": [{"meals": []}]},
                                            {"medicalConditions": ["DM2"]}, surface="s") == 0


def test_helper_caps_a_day_that_is_IN_band(monkeypatch):
    """El corazón del bug: el día está EN banda (el motor no lo tocaría) y aun así hay una
    porción sobre el cap clínico porque el carb-floor acaba de inflarla. El helper NO consulta
    la banda — ese era precisamente el gate que dejaba el hueco abierto."""
    import graph_orchestrator as go
    seen = {}
    monkeypatch.setattr(go, "cap_dm2_high_gi_portions",
                        lambda days, fd, db=None, **kw: seen.setdefault("days", days) and 0 or 1)
    monkeypatch.setattr(go, "cap_bariatric_portions", lambda *a, **kw: 0)
    # plan perfectamente en banda: entregado == target
    plan = {"macros": {"protein": "100 g", "carbs": "200 g", "fats": "60 g"},
            "days": [{"meals": [{"protein": 100, "carbs": 200, "fats": 60,
                                 "ingredients": ["375 g de batata"]}]}]}
    assert go.reapply_clinical_portion_caps(plan, {"medicalConditions": ["Diabetes Tipo 2"]},
                                            surface="swap_persist") == 1
    assert seen["days"] is plan["days"]


# ═════════════════ 2 · las 2 superficies lo invocan tras el carb-floor ═════════════════

def _idx_all(src: str, needle: str) -> list[int]:
    out, i = [], 0
    while True:
        j = src.find(needle, i)
        if j < 0:
            return out
        out.append(j)
        i = j + 1


def test_swap_persist_recaps_after_carb_floor():
    """Orden: carb-floor → (requantize/retrim) → re-cap clínico. Si el re-cap corriera ANTES,
    el carb-floor volvería a inflar la porción y el fix sería un no-op."""
    i_floor = _PLANS.index("_ccg_sw(")
    i_recap = _recap_callsites(_PLANS)[0]
    i_engine = _PLANS.index('surface="swap_persist"', i_floor)
    assert i_floor < i_recap
    # y ANTES del motor final, para que el motor mida el estado ya capado.
    assert i_recap < i_engine, "el re-cap va antes del motor: el motor debe ver porciones capadas"
    seg = _PLANS[i_recap:_PLANS.index("\n", _PLANS.index(")", i_recap))]
    assert "_micro_form" in seg, "debe usar el form_data SERVER-SIDE, no el body del cliente"


def _recap_callsites(src: str) -> list[int]:
    """Posiciones de las INVOCACIONES del helper. El repo importa con alias
    (`from graph_orchestrator import reapply_clinical_portion_caps as _rcpc_x`) para evitar
    ciclos de import, así que hay que resolver el alias — buscar el nombre pelado con paréntesis
    daría 0 matches y el test pasaría en vacío por el lado equivocado."""
    out = []
    for i in _idx_all(src, "import reapply_clinical_portion_caps as "):
        alias = src[i + len("import reapply_clinical_portion_caps as "):src.index("\n", i)].strip()
        assert alias, "import con alias vacío"
        out += _idx_all(src, alias + "(")
    return sorted(out)


def test_chat_modify_recaps_in_both_passes():
    """chat-modify corre el patrón closer-order en DOS pasadas (pre-listas + callback fresh).
    Si el re-cap aterriza solo en una, las listas y el plan persistido divergen."""
    hits = _recap_callsites(_TOOLS)
    assert len(hits) >= 2, "pre-listas y callback fresh"
    for h in hits:
        seg = _TOOLS[h:_TOOLS.index("\n", _TOOLS.index(")", h))]
        assert "_micro_form_cm" in seg, seg
    # cada invocación va después de su carb-floor
    floors = _idx_all(_TOOLS, "_carb_floor_updates_cm(")
    floors = [f for f in floors if not _TOOLS[max(0, f - 5):f].strip().endswith("def")]
    assert len(floors) >= 2
    for f, h in zip(sorted(floors), hits):
        assert f < h, "el re-cap clínico corre DESPUÉS del carb-floor que infla"


def test_carb_floor_hole_is_documented_at_the_callsite():
    """El comentario debe decir POR QUÉ no basta el motor posterior (que sí recibe form_data):
    el carb-floor deja el día EN banda ⇒ `_hit` False ⇒ el re-cap interno no corre. Sin esa
    frase, el próximo lector borra este bloque por 'redundante con el motor'."""
    for src in (_PLANS, _TOOLS):
        i = src.index("reapply_clinical_portion_caps")
        ctx = src[max(0, i - 1400):i]
        assert "P1-CARBFLOOR-CLINICAL-RECAP" in ctx
        assert "en banda" in ctx.lower()
