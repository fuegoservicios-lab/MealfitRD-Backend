"""[P2-REVIEW-ISSUES-CLARO · 2026-09-02] Las observaciones entregadas al usuario son cortas y
dicen QUÉ plato mirar.

Medido: en 30 días la ÚNICA familia entregada fue «COMIDA FUERA DE HORARIO» y no estaba en el
mapa de P2-REVIEW-ISSUES-HUMANIZE, así que el toast mostraba el párrafo técnico entero (captura
del dueño). Ahora: «Día 2, almuerzo: el plato es más de otro momento del día. Si no te convence,
cámbialo con «Cambiar Plato».». El raw sigue en `_review_issues_raw`.

Tooltip-anchor: P2-REVIEW-ISSUES-CLARO | _REVIEW_ISSUE_DAY_SLOT_RE
"""
import graph_orchestrator as go

RAW = ("COMIDA FUERA DE HORARIO (rechazo de coherencia cultural es-DO): Día 2, almuerzo: «Wrap crujiente de "
       "queso de hoja, granola y rábano» es comida de desayuno como plato principal del almuerzo "
       "(cereal/panqueque/waffle/avena), que no corresponde al almuerzo dominicano. Cámbialo por un plato "
       "propio del horario. El almuerzo es el plato fuerte: arroz+habichuela+proteína+ensalada, locrio, moro, "
       "asopao, pasta criolla, o pescado/carne con tubérculo y vegetal.")


def test_fuera_de_horario_short_with_day_and_slot():
    out = go._humanize_review_issue(RAW)
    assert out.startswith("Día 2, almuerzo: ")
    assert "Cambiar Plato" in out
    assert len(out) < 140, out
    assert "coherencia cultural" not in out and "locrio" not in out


def test_family_without_day_slot_gets_capitalized_copy():
    out = go._humanize_review_issue("SODIO EXCESIVO: 3 días superan 2300 mg")
    assert out.startswith("Algún día pasa el sodio recomendado")


def test_repeated_dish_between_days_family_now_covered():
    out = go._humanize_review_issue("MISMO PLATO REPETIDO ENTRE DÍAS (día 1 y día 3): Locrio de pollo")
    assert out.startswith("Un mismo plato se repite en varios días")


def test_unknown_family_still_preserved():
    out = go._humanize_review_issue("ALGO NUEVO DEL REVISOR: detalle. action=reject_minor.")
    assert out == "ALGO NUEVO DEL REVISOR: detalle."


def test_day_slot_regex_is_accent_and_case_insensitive():
    assert go._REVIEW_ISSUE_DAY_SLOT_RE.search("... Dia 3, Cena: ...").group(2).lower() == "cena"
