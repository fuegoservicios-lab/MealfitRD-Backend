"""[P1-MEDICAL-SCOPE-GATE · 2026-08-09] El formulario preguntaba MENOS de lo que
el motor sabe, y lo que caía fuera no producía ni regla ni aviso.

HALLAZGO (auditado contra los registries reales):

  · `condition_rules.CONDITION_RULES` tiene 12 reglas clínicas completas; el
    paso ofrecía 7 chips. **Enfermedad renal, anemia ferropénica, gota e hígado
    graso tenían regla escrita y eran INDECLARABLES desde el wizard.**
  · `medication_rules.MEDICATION_RULES` tiene 13; los 14 chips cubrían 12.
    El que faltaba era **`maoi`** — tiramina + IMAO es crisis hipertensiva. El
    motor sabía protegerte y el formulario no dejaba decirlo.
  · No existía detección de condición desconocida: lo que no matcheaba ninguna
    regla generaba plan como si el usuario no hubiera declarado nada.
  · El enunciado decía «escribe otras» y el input de texto libre se eliminó el
    2026-08-01 — invitaba a hacer algo imposible.

EL GATE ES POR VALOR EXACTO, NO POR SUBCADENA. `detect_active_rules` matchea
por contención y este repo lleva 16 incidentes de esa clase; un blocklist sobre
prosa sería el 17º. Y aquí las dos direcciones del error son graves: un falso
positivo DENIEGA servicio, un falso negativo ENTREGA un plan inseguro.

Tooltip-anchor: P1-MEDICAL-SCOPE-GATE
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_BACKEND = _HERE.parent.parent
_REPO_ROOT = _BACKEND.parent
_QMEDICAL = _REPO_ROOT / "frontend" / "src" / "components" / "assessment" / "questions" / "QMedical.jsx"
_FLOW = _REPO_ROOT / "frontend" / "src" / "components" / "assessment" / "InteractiveAssessmentFlow.jsx"
_PLANS = _BACKEND / "routers" / "plans.py"

if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _chip_labels(first_label: str) -> list[str]:
    """Literales del array de chips que se renderiza (`{['A', 'B', …].map(`).

    Ancla en `{['<primer label>` y NO en el label suelto: los mismos literales
    aparecen antes en los mapas de iconos (`CONDITION_ICONS`/`MED_ICONS`), así
    que buscar el label a secas encontraba el bloque equivocado.
    """
    src = _QMEDICAL.read_text(encoding="utf-8")
    marker = "{['" + first_label + "'"
    i = src.index(marker)
    arr = re.search(r"\{\[(.*?)\]\.map\(", src[i:], re.DOTALL)
    assert arr, f"P1-MEDICAL-SCOPE-GATE: no encuentro el array de chips en {marker!r}"
    return re.findall(r"'([^']+)'", arr.group(1))


# ── 1. Cada chip dispara EXACTAMENTE una regla ───────────────────────────────

def test_every_condition_chip_fires_exactly_one_rule():
    """Un chip cuyo label no matchea ningún `term` es decoración: el usuario
    cree que declaró algo y el motor no se entera (clase «feature INERTE»).
    Y uno que matchea DOS es peor: aplica reglas que el usuario no declaró."""
    from condition_rules import CONDITION_RULES
    from constants import strip_accents

    labels = _chip_labels("Diabetes T2")
    assert len(labels) >= 11, (
        f"P1-MEDICAL-SCOPE-GATE: se esperaban >=11 chips de condición, hay {len(labels)}. "
        "Los 4 añadidos (renal/anemia/gota/hígado graso) tienen regla en el backend."
    )
    for lbl in labels:
        norm = strip_accents(lbl.strip().lower())
        hits = [r.id for r in CONDITION_RULES if any(t in norm for t in r.terms)]
        assert len(hits) == 1, (
            f"P1-MEDICAL-SCOPE-GATE: el chip {lbl!r} dispara {hits or 'NINGUNA'} regla(s). "
            "Debe disparar exactamente una. El matcher es por SUBCADENA: al tocar esta "
            "lista hay que reverificar que ningún label contenga el `term` de otra regla."
        )


def test_the_four_rules_that_had_no_chip_now_have_one():
    """Ancla el hallazgo concreto, no solo la propiedad genérica de arriba."""
    from condition_rules import CONDITION_RULES
    from constants import strip_accents

    labels = [strip_accents(l.strip().lower()) for l in _chip_labels("Diabetes T2")]
    by_id = {r.id: r for r in CONDITION_RULES}
    for rid in ("renal", "anemia", "gout", "nafld"):
        rule = by_id[rid]
        alcanzable = any(any(t in lbl for t in rule.terms) for lbl in labels)
        assert alcanzable, (
            f"P1-MEDICAL-SCOPE-GATE: la regla `{rid}` ({rule.label}) volvió a quedar sin "
            "chip. Tiene reglas clínicas completas escritas y el usuario no puede "
            "declararla: recibiría un plan sin ninguna de ellas y sin aviso."
        )


def test_the_maoi_rule_is_reachable_from_the_form():
    """El más urgente de los cinco: tiramina + IMAO es crisis hipertensiva."""
    from medication_rules import MEDICATION_RULES
    from constants import strip_accents

    labels = [strip_accents(l.strip().lower()) for l in _chip_labels("Metformina")]
    maoi = next(r for r in MEDICATION_RULES if r.id == "maoi")
    assert any(any(t in lbl for t in maoi.terms) for lbl in labels), (
        "P1-MEDICAL-SCOPE-GATE: la regla `maoi` volvió a quedar sin chip. El motor "
        "modela la interacción tiramina↔IMAO y el formulario no deja declararla."
    )


# ── 2. El gate de alcance ────────────────────────────────────────────────────

def test_the_out_of_scope_literals_match_across_the_stack():
    """Si frontend y backend drifean, el gate del servidor deja de reconocer lo
    que el formulario emite y el bloqueo se evapora EN SILENCIO — el modo de
    fallo más peligroso posible para un gate de seguridad."""
    jsx = _QMEDICAL.read_text(encoding="utf-8")
    py = _PLANS.read_text(encoding="utf-8")
    for js_name, py_name in (
        ("OUT_OF_SCOPE_CONDITION", "_OUT_OF_SCOPE_CONDITION"),
        ("OUT_OF_SCOPE_MEDICATION", "_OUT_OF_SCOPE_MEDICATION"),
    ):
        js_val = re.search(rf"export const {js_name} = '([^']+)'", jsx)
        py_val = re.search(rf'{py_name} = "([^"]+)"', py)
        assert js_val, f"P1-MEDICAL-SCOPE-GATE: falta `{js_name}` en QMedical.jsx"
        assert py_val, f"P1-MEDICAL-SCOPE-GATE: falta `{py_name}` en routers/plans.py"
        assert js_val.group(1) == py_val.group(1), (
            f"P1-MEDICAL-SCOPE-GATE: DRIFT — {js_name}={js_val.group(1)!r} vs "
            f"{py_name}={py_val.group(1)!r}. El backend dejaría de reconocer la señal."
        )


def test_the_gate_matches_by_exact_value_never_by_substring():
    """El punto entero del chip es no depender de interpretar prosa."""
    py = _PLANS.read_text(encoding="utf-8")
    fn = re.search(r"def _has_out_of_scope_clinical_declaration.*?\n\n\n", py, re.DOTALL)
    assert fn, "P1-MEDICAL-SCOPE-GATE: no encuentro `_has_out_of_scope_clinical_declaration`"
    body = fn.group(0)
    assert "==" in body, "P1-MEDICAL-SCOPE-GATE: el gate debe comparar por igualdad."
    assert " in c" not in body and ".startswith(" not in body, (
        "P1-MEDICAL-SCOPE-GATE: el gate usa contención. Debe ser igualdad sobre el "
        "valor canónico — un blocklist por subcadena sería la 17ª de su clase, y aquí "
        "un falso positivo deniega servicio y un falso negativo entrega un plan inseguro."
    )


def test_both_generation_endpoints_are_gated():
    """Dos puertas de generación (`/analyze` y `/analyze/stream`); una guarda en
    una sola es un agujero — el cliente elige por cuál entra."""
    py = _PLANS.read_text(encoding="utf-8")
    n = len(re.findall(r"_has_out_of_scope_clinical_declaration\(data\)", py))
    assert n >= 2, (
        f"P1-MEDICAL-SCOPE-GATE: el gate se invoca {n} vez/veces. Deben ser las DOS "
        "puertas de generación, igual que el cap de condiciones."
    )
    assert py.count('"clinical_scope_exceeded"') >= 2, (
        "P1-MEDICAL-SCOPE-GATE: falta el código de error en alguno de los dos endpoints."
    )


def test_the_form_blocks_before_making_the_user_fill_five_more_steps():
    jsx = _QMEDICAL.read_text(encoding="utf-8")
    assert "outOfScopeSelected" in jsx, "P1-MEDICAL-SCOPE-GATE: falta `outOfScopeSelected`"
    nb = re.search(r"<NextButton(.*?)/>", jsx, re.DOTALL)
    assert nb and "outOfScopeSelected" in nb.group(1), (
        "P1-MEDICAL-SCOPE-GATE: el botón Siguiente no considera `outOfScopeSelected`. "
        "Bloquear solo al generar haría rellenar cinco pasos más para nada."
    )


# ── 3. El copy que prometía un campo inexistente ─────────────────────────────

def test_the_subtitle_no_longer_promises_a_field_that_does_not_exist():
    """El input de texto libre se eliminó el 2026-08-01 y el enunciado siguió
    diciendo «escribe otras» — invitaba a hacer algo imposible."""
    flow = _FLOW.read_text(encoding="utf-8")
    # Sin regex sobre el literal: el subtítulo lleva comillas ESCAPADAS dentro
    # (`\"Ninguna\"`), y cualquier patrón `[^"]*` corta ahí y captura media
    # frase. Se toma la línea entera, que es inequívoca y no tiene ese problema.
    linea = next(
        (ln for ln in flow.splitlines()
         if "subtitle:" in ln and "Marca todas las que apliquen" in ln),
        None,
    )
    assert linea, "P1-MEDICAL-SCOPE-GATE: no encuentro el subtítulo del paso médico"
    sub = linea
    assert "escribe otras" not in sub, (
        "P1-MEDICAL-SCOPE-GATE: el subtítulo volvió a prometer texto libre. Ese input "
        "no existe; la vía para lo no listado es el chip «Otra condición»."
    )
    assert "Otra condición" in sub, (
        "P1-MEDICAL-SCOPE-GATE: el subtítulo debe señalar la vía real para lo no listado."
    )
