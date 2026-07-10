"""[P2-BAND-GATE-KCAL-SEMANTICS · 2026-07-10] Forensic corr=d57ffe04 (2026-07-10): `_maybe_mark_low_band_
degraded` excluía kcal INCONDICIONALMENTE del trigger per-macro del banner ("P/C/F la acoplan
aritméticamente 4/4/9 — si las 3 están en banda, kcal lo está") — cierto para la banda MACRO amplia
[0.90,1.12] pero NO para la banda kcal, más estrecha [0.95,1.05]: un plan puede tener P/C/F en banda
ancha y kcal fuera de SU banda propia. En la corrida real, kcal=0.333 (igual que carbs) nunca pudo
disparar el banner por sí solo. El retry-gate YA tenía un backstop de kcal independiente
(BAND_GATE_KCAL_BACKSTOP/THRESHOLD, P2-KCAL-GATE-BACKSTOP) — este batch extiende el MISMO backstop
(mismos knobs, mismo rollback) al banner, vía un helper SSOT compartido por las 3 funciones que
computaban `_pm_low` independientemente (mark/parity/clear — el clear debe ser el complemento EXACTO
del mark, así que las 3 DEBEN compartir criterio).
"""
import graph_orchestrator as go


def _gate_on(monkeypatch):
    monkeypatch.setattr(go, "BAND_SCORE_GATE_ENABLED", True)
    monkeypatch.setattr(go, "BAND_SCORE_GATE_THRESHOLD", 0.5)
    monkeypatch.setattr(go, "BAND_GATE_PER_MACRO", True)
    monkeypatch.setattr(go, "BAND_GATE_PER_MACRO_THRESHOLD", 0.34)
    monkeypatch.setattr(go, "BAND_GATE_KCAL_BACKSTOP", True)
    monkeypatch.setattr(go, "BAND_GATE_KCAL_THRESHOLD", 0.5)


def test_helper_marker_present():
    assert "P2-BAND-GATE-KCAL-SEMANTICS" in open(go.__file__, encoding="utf-8").read()


def test_shared_helper_exists():
    assert hasattr(go, "_band_gate_per_macro_low")


def test_kcal_now_triggers_per_macro_via_backstop(monkeypatch):
    """Antes: kcal_never_triggers. Ahora: kcal SÍ dispara el banner vía el backstop ya establecido
    en el retry-gate — mismos knobs, misma semántica, cero invención de umbral nuevo."""
    _gate_on(monkeypatch)
    plan: dict = {}
    payload = {"per_macro": {"protein": 0.9, "carbs": 0.8, "fats": 0.7, "kcal": 0.0}}
    marked = go._maybe_mark_low_band_degraded(plan, 0.7, False, attempt=1, band_payload=payload)
    assert marked is True
    assert plan["_quality_degraded_reason"] == "low_band_macro:kcal"


def test_kcal_backstop_off_restores_old_behavior(monkeypatch):
    """Rollback sin redeploy: MEALFIT_BAND_GATE_KCAL_BACKSTOP=false vuelve al comportamiento previo
    (kcal jamás dispara el trigger per-macro del banner)."""
    _gate_on(monkeypatch)
    monkeypatch.setattr(go, "BAND_GATE_KCAL_BACKSTOP", False)
    plan: dict = {}
    payload = {"per_macro": {"protein": 0.9, "carbs": 0.8, "fats": 0.7, "kcal": 0.0}}
    assert go._maybe_mark_low_band_degraded(plan, 0.7, False, attempt=1, band_payload=payload) is False


def test_kcal_uses_its_own_threshold_not_per_macro_threshold(monkeypatch):
    """kcal 0.4 con BAND_GATE_KCAL_THRESHOLD=0.3 (más laxo) NO dispara, aunque
    BAND_GATE_PER_MACRO_THRESHOLD=0.5 (más estricto) sí lo haría si se usara por error."""
    _gate_on(monkeypatch)
    monkeypatch.setattr(go, "BAND_GATE_KCAL_THRESHOLD", 0.3)
    plan: dict = {}
    payload = {"per_macro": {"protein": 0.9, "carbs": 0.8, "fats": 0.7, "kcal": 0.4}}
    assert go._maybe_mark_low_band_degraded(plan, 0.7, False, attempt=1, band_payload=payload) is False


def test_pc_f_unaffected_by_kcal_backstop_knob(monkeypatch):
    """Apagar el backstop de kcal NO afecta el trigger normal de protein/carbs/fats."""
    _gate_on(monkeypatch)
    monkeypatch.setattr(go, "BAND_GATE_KCAL_BACKSTOP", False)
    plan: dict = {}
    payload = {"per_macro": {"protein": 1.0, "carbs": 0.0, "fats": 1.0, "kcal": 1.0}}
    marked = go._maybe_mark_low_band_degraded(plan, 0.75, False, attempt=3, band_payload=payload)
    assert marked is True
    assert plan["_quality_degraded_reason"] == "low_band_macro:carbs"


def test_clear_stale_is_exact_complement_of_mark_for_kcal(monkeypatch):
    """El clear-only debe usar el MISMO criterio kcal que el mark — si kcal sigue bajo umbral tras
    finalize, NO limpia el banner (sería un falso 'ya está bien')."""
    _gate_on(monkeypatch)
    monkeypatch.setattr(go, "BAND_DEGRADED_STALE_CLEAR_ENABLED", True)
    plan = {"_quality_degraded": True, "_quality_degraded_reason": "low_band_macro:kcal",
           "macros": {"protein": "150g", "carbs": "200g", "fats": "60g"}, "calories": 2000,
           # P/C/F perfectos pero `cals` explícito 1200 (fuera de la banda kcal [0.95,1.05] de 2000)
           # → per_macro['kcal']=0.0 (0/1 días en banda) < BAND_GATE_KCAL_THRESHOLD=0.5 → sigue fallando.
           "days": [{"day": 1, "meals": [{"protein": 150, "carbs": 200, "fats": 60, "cals": 1200}]}]}
    cleared = go.clear_stale_low_band_degraded(plan)
    assert cleared is False, "kcal sigue fuera de banda tras finalize → el clear NO debe limpiar el banner"
    assert plan["_quality_degraded"] is True
