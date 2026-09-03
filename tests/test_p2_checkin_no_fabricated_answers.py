"""[P2-CHECKIN-NO-FABRICATED-ANSWERS · 2026-09-03] El check-in de renovación guarda SOLO lo respondido.

La adherencia venía precargada al 80 % y viajaba siempre. El backend la usa como COMPUERTA
(`nutrition_calculator`: por debajo de `CHECKIN_ADHERENCE_FLOOR` no ajusta las calorías por el
cambio de peso). Con el 80 % de fábrica, quien renovaba sin pensar pasaba la compuerta y el
sistema atribuía su peso a un plan que quizá no siguió. Ahora arranca «sin responder» (viaja
`null`, que el endpoint ya acepta), el peso del perfil no se guarda sin editarlo o confirmarlo, y
un solo botón genera (sin nada respondido, no se escribe ningún check-in).
"""
from __future__ import annotations

from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def modal_src(frontend_repo_path):
    src = (frontend_repo_path / "src" / "components" / "plan" / "RenewalCheckinModal.jsx").read_text(encoding="utf-8")
    # solo CÓDIGO: la cabecera cita los botones viejos para explicar el cambio (comentario-vence-guard)
    return chr(10).join(l for l in src.splitlines() if not l.strip().startswith("//"))


def test_backend_accepts_missing_signals_and_keeps_weight_mandatory():
    src = (BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    i = src.index("def api_renewal_checkin") if "def api_renewal_checkin" in src else src.index('"/renewal-checkin"')
    body = src[i:i + 4000]
    assert '_adherence = _clamp_int(data.get("adherence_pct"), 0, 100)' in body
    assert "return None" in body                       # _clamp_int local: sin valor ⇒ None, no 0
    assert 'raise HTTPException(status_code=400, detail="Peso inválido.")' in body


def test_gate_treats_unknown_adherence_as_no_evidence():
    src = (BACKEND / "nutrition_calculator.py").read_text(encoding="utf-8")
    assert "isinstance(_adh, (int, float)) and _adh < CHECKIN_ADHERENCE_FLOOR" in src


def test_modal_has_no_fabricated_defaults_and_one_button(modal_src):
    assert "useState(80)" not in modal_src
    assert "const [adherence, setAdherence] = useState(null);" in modal_src
    assert "const [weightConfirmed, setWeightConfirmed] = useState(false);" in modal_src
    assert "if (!anythingAnswered) {" in modal_src and "onDone(null);" in modal_src
    assert "t('Generar mi plan')" in modal_src
    assert "Generar sin guardar" not in modal_src and "Guardar y generar mi plan" not in modal_src
    assert "t('Solo guardamos lo que respondas.')" in modal_src


def test_marker_present():
    assert "P2-CHECKIN-NO-FABRICATED-ANSWERS" in (BACKEND / "app.py").read_text(encoding="utf-8")
