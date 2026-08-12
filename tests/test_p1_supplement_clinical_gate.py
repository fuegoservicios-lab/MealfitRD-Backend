# -*- coding: utf-8 -*-
"""[P1-SUPPLEMENT-CLINICAL-GATE · 2026-08-12] Los suplementos eran la ÚNICA pieza
del plan sin gate clínico: el backstop determinista opera sobre comidas, el
Revisor Médico no los mencionaba, y el prompt con selección explícita ORDENA
incluirlos «ni más, ni menos» — un hipertenso que marcaba Pre-Entreno lo
recibía por orden directa.

Cuatro capas ancladas aquí:
  1. Tabla SSOT (constants.SUPPLEMENT_CONTRAINDICATIONS) — integridad referencial
     contra los registries reales (nada de strings del wizard).
  2. Detector (condition_rules.contraindicated_supplements) — comportamiento.
  3. Gate del prompt (build_supplements_context) — filtra selección + prohíbe.
  4. Barredora post-gen (graph_orchestrator) + Revisor Médico + espejo UI —
     anclas parser (el enforcement inline vive en assemble; el espejo es UX).
"""
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
FRONTEND_SRC = BACKEND.parent / "frontend" / "src"

from constants import SUPPLEMENT_CONTRAINDICATIONS, SUPPLEMENT_MATCH_KEYWORDS, SUPPLEMENT_NAMES
from condition_rules import CONDITION_RULES, contraindicated_supplements
from medication_rules import MEDICATION_RULES


# ---------------------------------------------------------------------------
# 1. Integridad de la tabla
# ---------------------------------------------------------------------------

def test_tabla_referencia_solo_ids_reales():
    cond_ids = {r.id for r in CONDITION_RULES}
    med_ids = {r.id for r in MEDICATION_RULES}
    for supp, spec in SUPPLEMENT_CONTRAINDICATIONS.items():
        assert supp in SUPPLEMENT_NAMES, f"{supp} no es un suplemento del SSOT"
        _malas_c = set(spec["conditions"]) - cond_ids
        _malas_m = set(spec["medications"]) - med_ids
        assert not _malas_c, f"{supp}: condition ids inexistentes {_malas_c}"
        assert not _malas_m, f"{supp}: medication ids inexistentes {_malas_m}"
        assert spec["reason"].strip(), f"{supp}: sin razón"


def test_todo_vetado_tiene_keywords_para_la_barredora():
    faltan = set(SUPPLEMENT_CONTRAINDICATIONS) - set(SUPPLEMENT_MATCH_KEYWORDS)
    assert not faltan, (
        f"Suplementos vetables sin keywords de match: {faltan} — la barredora "
        f"post-gen no podría reconocerlos en el nombre libre del LLM."
    )
    for supp, kws in SUPPLEMENT_MATCH_KEYWORDS.items():
        for kw in kws:
            assert len(kw) >= 4, (
                f"{supp}: keyword {kw!r} <4 chars — la clase «sal⊆salami» "
                f"tiene 15 cicatrices en este repo."
            )


# ---------------------------------------------------------------------------
# 2. Detector — perfiles con los CHIPS exactos del wizard
# ---------------------------------------------------------------------------

def test_hipertenso_veta_estimulantes():
    v = contraindicated_supplements({"medicalConditions": ["Hipertensión"]})
    assert "pre_workout" in v and "fat_burner" in v
    assert "creatine" not in v and "omega3" not in v


def test_embarazo_y_lactancia_vetan_estimulantes():
    for chip in ("Embarazo", "Lactancia"):
        v = contraindicated_supplements({"medicalConditions": [chip]})
        assert "pre_workout" in v and "fat_burner" in v, chip


def test_renal_veta_creatina_y_proteinas():
    v = contraindicated_supplements({"medicalConditions": ["Enfermedad Renal"]})
    for k in ("creatine", "whey_protein", "vegan_protein", "bcaa"):
        assert k in v, k
    assert "omega3" not in v


def test_warfarina_veta_omega3():
    v = contraindicated_supplements({"medications": ["Warfarina"]})
    assert v.keys() == {"omega3"}


def test_imao_veta_estimulantes():
    v = contraindicated_supplements({"medications": ["Antidepresivo IMAO"]})
    assert "pre_workout" in v and "fat_burner" in v


def test_perfil_limpio_no_veta_nada():
    assert contraindicated_supplements({"medicalConditions": ["Ninguna"], "medications": ["Ninguno"]}) == {}
    assert contraindicated_supplements({}) == {}


def test_texto_libre_tambien_activa_el_veto():
    """El espejo UI solo ve chips; el backend DEBE ver texto libre (el gate real)."""
    v = contraindicated_supplements({"medicalConditions": ["tengo la presion alta"]})
    assert "pre_workout" in v


# ---------------------------------------------------------------------------
# 3. Gate del prompt
# ---------------------------------------------------------------------------

def _ctx(form):
    from prompts.plan_generator import build_supplements_context
    return build_supplements_context(form)


def test_seleccion_vetada_sale_del_debes_y_entra_al_prohibido():
    ctx = _ctx({
        "includeSupplements": True,
        "selectedSupplements": ["pre_workout", "creatine"],
        "medicalConditions": ["Hipertensión"],
    })
    assert "PROHIBIDOS POR SEGURIDAD CLÍNICA" in ctx
    # La línea DEBES exacta: el vetado NO está; el permitido sí. («Pre-Entreno»
    # SÍ aparece más abajo — en ❌ NO INCLUIR y en el bloque prohibido, que es
    # justo el comportamiento correcto.)
    _linea_debes = next(l for l in ctx.splitlines() if "LISTA EXACTA" in l)
    assert "Creatina" in _linea_debes
    assert "Pre-Entreno" not in _linea_debes, "el vetado sigue en la lista DEBES"
    assert "TOTAL: 1 suplemento(s)" in ctx


def test_seleccion_totalmente_vetada_cae_a_rama_libre_con_prohibicion():
    ctx = _ctx({
        "includeSupplements": True,
        "selectedSupplements": ["creatine"],
        "medicalConditions": ["Enfermedad Renal"],
    })
    assert "SUPLEMENTOS PERSONALIZADOS" in ctx  # rama libre
    assert "PROHIBIDOS POR SEGURIDAD CLÍNICA" in ctx
    assert "Creatina" in ctx


def test_rama_libre_con_warfarina_prohibe_omega3():
    ctx = _ctx({
        "includeSupplements": True,
        "selectedSupplements": [],
        "medications": ["Warfarina"],
    })
    assert "PROHIBIDOS POR SEGURIDAD CLÍNICA" in ctx
    assert "Omega-3" in ctx


def test_perfil_limpio_sin_bloque_prohibido():
    ctx = _ctx({"includeSupplements": True, "selectedSupplements": ["creatine"]})
    assert "PROHIBIDOS POR SEGURIDAD CLÍNICA" not in ctx
    assert "Creatina" in ctx


def test_toggle_apagado_intacto():
    ctx = _ctx({"includeSupplements": False, "medicalConditions": ["Hipertensión"]})
    assert "NO INCLUIR (OBLIGATORIO)" in ctx


# ---------------------------------------------------------------------------
# 4. Barredora + Revisor + espejo UI (anclas parser)
# ---------------------------------------------------------------------------

def test_barredora_postgen_anclada_en_orquestador():
    src = (BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("[P1-SUPPLEMENT-CLINICAL-GATE")  # ancla del bloque else
    bloque = src[i:i + 2500]
    assert "contraindicated_supplements" in bloque
    assert "SUPPLEMENT_MATCH_KEYWORDS" in bloque
    # fail-safe explícito: si el veto no se puede calcular, se loggea y NO revienta
    assert "barredora inactiva" in bloque


def test_revisor_medico_menciona_suplementos():
    src = (BACKEND / "prompts" / "medical_reviewer.py").read_text(encoding="utf-8")
    assert "5. SUPLEMENTOS" in src
    for familia in ("hipertensión", "anticoagulantes", "enfermedad renal"):
        assert familia in src.lower(), familia


def test_espejo_ui_cubre_las_mismas_claves():
    """Paridad de CLAVES backend↔frontend. Los chips del espejo se validan
    contra los literales del wizard (QMedical) — igualdad, no substring."""
    fv = (FRONTEND_SRC / "config" / "formValidation.js").read_text(encoding="utf-8")
    m = re.search(r"export const SUPPLEMENT_BLOCKERS = \{(.*?)\n\};", fv, re.DOTALL)
    assert m, "SUPPLEMENT_BLOCKERS no encontrado en formValidation.js"
    front_keys = set(re.findall(r"^\s{4}(\w+): \{", m.group(1), re.MULTILINE))
    assert front_keys == set(SUPPLEMENT_CONTRAINDICATIONS), (
        f"drift de claves espejo UI vs backend: solo-front {front_keys - set(SUPPLEMENT_CONTRAINDICATIONS)}, "
        f"solo-back {set(SUPPLEMENT_CONTRAINDICATIONS) - front_keys}"
    )
    qm = (FRONTEND_SRC / "components" / "assessment" / "questions" / "QMedical.jsx").read_text(encoding="utf-8")
    # Embarazo/Lactancia viven en PREGNANCY_CHIP_LABELS (_shared.jsx), no como
    # literales de QMedical — el universo de chips son AMBOS archivos.
    # Extracción CONFINADA a literales de array `['A', 'B', ...]`: el pareo
    # secuencial de comillas sobre el JSX entero se desalinea con cualquier
    # apóstrofe suelto de la prosa (el instrumental ya mintió así una vez).
    shared = (FRONTEND_SRC / "components" / "assessment" / "questions" / "_shared.jsx").read_text(encoding="utf-8")
    chips = set()
    for arr in re.findall(r"\[(\s*'[^\]]+?)\]", qm + "\n" + shared):
        chips |= set(re.findall(r"'([^']+)'", arr))
    # SOLO los labels dentro de los arrays conditions/medications — los hints
    # son prosa y también arrancan en mayúscula (el primer intento los barría).
    for arr in re.findall(r"(?:conditions|medications): \[([^\]]*)\]", m.group(1)):
        for label in re.findall(r"'([^']+)'", arr):
            assert label in chips, (
                f"El espejo UI referencia el chip {label!r} que no existe en "
                f"QMedical.jsx — chip renombrado sin actualizar SUPPLEMENT_BLOCKERS."
            )
