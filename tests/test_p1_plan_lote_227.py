# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-227 · 2026-09-25] El revisor no quema intentos con no-defectos, y el desayuno respeta la alergia.

Batería real de 16 perfiles con el código de los lotes 220-221:
  · «no me gusta el pescado ni la berenjena»: el revisor LLM rechazó 4 veces un plan sin defectos —«Severidad: none»
    y textos que se niegan solos («no es violación», «El plan es seguro», «ninguno aparece en el plan»)— y el plan se
    entregó degradado tras 3 intentos.
  · alergia a «Mariscos»: rechazo CRÍTICO porque el plan traía pescado de aleta y «mariscos puede incluir pescado en el
    uso coloquial. Se debe aclarar con el paciente antes de aprobar» — una aclaración, no un defecto (lote 182: «confirme
    si…» ya lo era). Arrastró el plan a 3 intentos y a una entrega degradada.
  · alergia a gluten y huevo: el esqueleto le asignó «Avena/Cereales» de desayuno → «30 g de avena» → ALÉRGENO DETECTADO.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


def test_rechazo_sin_severidad_es_aprobado():
    ok, issues, sev, avisos = go._downgrade_reviewer_non_issues(
        False, ["Los rechazos declarados son Pescado y Berenjena, y ninguno aparece en el plan. El plan es seguro."], "none")
    assert (ok, issues, sev) == (True, [], "low") and len(avisos) == 1


def test_observaciones_que_se_niegan_solas_pasan_a_aviso():
    ok, issues, sev, avisos = go._downgrade_reviewer_non_issues(False, [
        "Día 1 | Cena: contiene queso mozzarella (lácteo) — sin alergia a lácteos declarada, no es violación.",
        "Día 1 | Merienda: mantequilla de maní; sin alergia declarada, no constituye violación.",
    ], "minor")
    assert (ok, issues, sev) == (True, [], "low") and len(avisos) == 2


def test_lo_que_sigue_tras_sin_embargo_se_queda():
    real = "El plan no contiene pescado; sin embargo, se detecta una inconsistencia grave en el Día 3 Merienda."
    ok, issues, sev, avisos = go._downgrade_reviewer_non_issues(False, [real, "Día 3: 9 huevos en una comida."], "minor")
    assert ok is False and issues == [real, "Día 3: 9 huevos en una comida."] and avisos == []


def test_el_knob_lo_apaga(monkeypatch):
    monkeypatch.setattr(go, "REVIEWER_NON_ISSUES_ADVISORY", False)
    ok, issues, sev, _ = go._downgrade_reviewer_non_issues(False, ["El plan es seguro."], "none")
    assert ok is False and issues == ["El plan es seguro."]


def test_aclarar_con_el_paciente_es_aclaracion():
    txt = ("El plan incluye pescado (tilapia/mero y atún), pero la alergia declarada a «mariscos» puede incluir pescado "
           "en el uso coloquial. Se debe aclarar con el paciente si también es alérgico al pescado antes de aprobar.")
    ok, issues, sev, avisos = go._downgrade_reviewer_verification_demands(False, [txt], "critical")
    assert ok is True and issues == [] and avisos == [txt]


def test_cableado_en_el_revisor():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("                _downgrade_reviewer_verification_demands(approved, issues, severity)")  # la llamada
    j = src.index("_downgrade_reviewer_non_issues(approved, issues, severity)", i)
    assert j - i < 600
    assert "_verif_advisories = list(_verif_advisories or []) + list(_non_issue_advisories or [])" in src[j:j + 400]
    assert "tooltip-anchor: P1-PLAN-LOTE-227-NO-DEFECTOS" in src
    assert "tooltip-anchor: P1-PLAN-LOTE-227-DESAYUNO-POR-ALERGIA" in src


def test_la_categoria_de_desayuno_por_alergia(monkeypatch):
    """Réplica del bloque del esqueleto sobre días sintéticos (el bloque vive dentro de `plan_skeleton_node`)."""
    import desayuno_por_alergia as dpa
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '__import__("desayuno_por_alergia").reasignar(skel_days, form_data)' in src
    skel_days = [{"day": 1, "breakfast_category": "Avena/Cereales"},
                 {"day": 2, "breakfast_category": "Revoltillo/Tortilla"},
                 {"day": 3, "breakfast_category": "Mangú/Tubérculos"}]
    assert dpa.reasignar(skel_days, {"allergies": ["Gluten", "Huevo"]}) == 2
    cats = [d["breakfast_category"] for d in skel_days]
    assert cats[0] == "Batido/Bowl" and cats[2] == "Mangú/Tubérculos", cats
    assert not {"Avena/Cereales", "Pan/Tostadas", "Revoltillo/Tortilla"} & set(cats), cats
    # sin alergias no se toca nada
    skel_days2 = [{"day": 1, "breakfast_category": "Avena/Cereales"}]
    assert dpa.reasignar(skel_days2, {"allergies": ["Ninguna"]}) == 0
    assert skel_days2[0]["breakfast_category"] == "Avena/Cereales"


def test_el_queso_con_apellido_lo_conserva_y_el_cottage_no_se_corta():
    import humanize_ingredients as hi
    assert hi.humanize_ingredient("119 g de queso cottage") == "119 g de queso cottage"
    assert "lonja" not in hi.humanize_ingredient("50 g de queso ricotta")   # la ricotta tiene su cda propia
    assert hi.humanize_ingredient("31 g de queso mozzarella") == "1¼ lonjas/pedazos de queso mozzarella"
    assert hi.humanize_ingredient("38 g de queso pasteurizado") == "1½ lonjas/pedazos de queso pasteurizado"
    # sin apellido, lo de siempre; y el blanco conserva su propia medida
    assert hi.humanize_ingredient("119 g de queso") == "4¾ lonjas/pedazos de queso"
    assert hi.humanize_ingredient("69 g de queso blanco") == "2¾ lonjas de queso blanco"


def test_el_revisor_recibe_la_aclaracion_de_mariscos():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'Alergias declaradas: {json.dumps(allergies) if allergies else "Ninguna"}{_mariscos_note}' in src
    assert '__import__("etiquetas_clinicas").notas_para_el_revisor(form_data)' in src
    import etiquetas_clinicas as _ec
    assert "pescado está PERMITIDO" in _ec.notas_para_el_revisor({"allergies": ["Mariscos"]})[0]
    assert "tooltip-anchor: P1-PLAN-LOTE-227-MARISCOS-AL-REVISOR" in src
    import etiquetas_clinicas as ec
    assert ec._solo_mariscos({"allergies": ["Lacteos", "Mariscos"]}) is True
    assert ec._solo_mariscos({"allergies": ["Mariscos", "Pescado"]}) is False
    txt = ("El plan incluye atún, mero y pescado blanco pese a que el reporte clínico recomienda tratar el pescado como "
           "potencialmente contraindicado hasta aclarar el alcance de la alergia a mariscos.")
    ok, issues, _sev, _av = go._downgrade_reviewer_verification_demands(False, [txt], "critical")
    assert ok is True and issues == []


def test_el_revisor_sabe_que_el_marisco_no_es_pescado():
    import etiquetas_clinicas as ec
    assert ec._pescado_sin_mariscos({"dislikes": ["Pescado", "Berenjena"]}) == "rechazo"
    assert ec._pescado_sin_mariscos({"allergies": ["Pescado"]}) == "alergia"
    assert ec._pescado_sin_mariscos({"allergies": ["Pescado"], "dislikes": ["Mariscos"]}) == ""
    assert ec._pescado_sin_mariscos({"dislikes": ["Berenjena"]}) == ""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'Alimentos que no le gustan: {json.dumps(dislikes) if dislikes else "Ninguno"}{_pescado_note}' in src
    assert "tooltip-anchor: P1-PLAN-LOTE-227-PESCADO-NO-ES-MARISCO" in src


def test_la_conclusion_que_se_niega_manda_aunque_diga_sin_embargo():
    txt = ("El paciente declaró que NO le gusta el pescado, pero el plan no contiene pescado; sin embargo, se detecta que "
           "el plan incluye alimentos rechazados: no se encontró pescado ni berenjena en las comidas, por lo que este "
           "punto se cumple.")
    real = "Día 1 | Desayuno: se repite 'pan integral' dos veces en la misma comida."
    ok, issues, sev, avisos = go._downgrade_reviewer_non_issues(False, [txt, real], "minor")
    assert ok is False and issues == [real] and avisos == [txt]
    # una observación real con un «el resto sin problema» al final NO se degrada
    otra = "Día 3: 9 huevos en una comida; el resto sin problema."
    ok, issues, _s, _a = go._downgrade_reviewer_non_issues(False, [otra], "minor")
    assert ok is False and issues == [otra]


def test_la_nota_de_embarazo_describe_el_plato_final():
    """El tope de pescado cambió el pescado por pollo DESPUÉS de anotar: la nota vieja hablaba de pescado."""
    import etiquetas_clinicas as ec
    vieja = ("🤰 Seguridad alimentaria (embarazo/lactancia): lava y desinfecta las frutas, verduras y hierbas frescas "
             "antes de usarlas (aunque se vayan a cocinar); cocina el pescado y los mariscos POR COMPLETO (opacos y "
             "firmes; nada crudo ni a medio cocer).")
    plan = {"days": [{"day": 2, "meals": [{
        "meal": "Almuerzo", "name": "Pollo a la parrilla con plátano verde y ensalada",
        "ingredients": ["180 g de pechuga de pollo", "½ plátano verde", "½ taza de repollo"],
        "recipe": ["Mise en place: corta todo.", "El Toque de Fuego: asa el pollo.", "Montaje: sirve.", vieja]}]}]}
    ec.etiquetar(plan, {"medicalConditions": ["Embarazo"], "gender": "female"})
    nota = [s for s in plan["days"][0]["meals"][0]["recipe"] if s.startswith("🤰")]
    assert len(nota) == 1, nota
    assert "cocina las carnes y el pollo POR COMPLETO (74 °C" in nota[0]
    assert "pescado" not in nota[0]


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 227
