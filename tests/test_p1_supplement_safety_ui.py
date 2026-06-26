"""[P1-SUPPLEMENT-SAFETY-UI · 2026-06-26] (auditoría gap #4) La suplementación debe ser SEGURA de mostrar.

Antes: el backend exponía dosis numéricas (ej. magnesio "200–400 mg/día", que EXCEDE su propio UL 350)
SIN renderizar precaución, disclaimer ni clamp en ninguna de las 2 superficies de UI, y no era
condition-aware (sugeriría magnesio a un perfil ERC → hipermagnesemia). Este test ancla 4 cosas:

  1. Ninguna plantilla con UL declarado sugiere una dosis máxima que lo exceda (catch del magnesio 400>350).
  2. build_supplement_recommendations es condition-aware: suprime magnesio en ERC (fail-secure), lo conserva
     en condiciones no-renales.
  3. Cada item lleva `precaucion` y el builder retorna `disclaimer` (los datos que la UI ahora SÍ renderiza).
  4. Las 2 superficies de UI (MicronutrientMeter, NotificationCenter) referencian precaucion + disclaimer
     (parser-based: si alguien borra el render, este test falla antes que producción).
"""
from __future__ import annotations

import re
from pathlib import Path

import micronutrients as mn

_BACKEND = Path(__file__).resolve().parent.parent
_FRONTEND_DASH = _BACKEND.parent / "frontend" / "src" / "components" / "dashboard"


def _report_with_gap(key: str, nutriente: str, status: str = "bajo") -> dict:
    return {"gaps": [{"key": key, "nutriente": nutriente, "valor": 1.0,
                      "piso": 10.0, "unidad": "mg", "status": status}]}


def test_no_template_dose_exceeds_its_declared_ul():
    """Cualquier plantilla con campo `ul` debe sugerir una dosis máxima <= UL (catch magnesio 400>350)."""
    checked = 0
    for key, tpl in mn._SUPPLEMENT_TEMPLATES.items():
        ul = tpl.get("ul")
        if ul is None:
            continue
        checked += 1
        dose = " ".join(str(tpl.get(k, "")) for k in ("dosis", "dosis_f", "dosis_m", "dosis_f_preg"))
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", dose)]
        assert nums, f"{key}: no se pudo parsear ninguna dosis numérica para validar contra el UL"
        assert max(nums) <= float(ul), (
            f"{key}: dosis máxima {max(nums)} excede su propio UL declarado {ul} — la tarjeta se "
            f"auto-contradice (el magnesio decía 400 con UL 350). Alinea la dosis al UL."
        )
    assert checked >= 1, "esperaba al menos una plantilla con UL declarado (magnesio)"


def test_magnesium_suppressed_in_renal_kept_otherwise():
    rep = _report_with_gap("magnesium_mg", "Magnesio")
    # Sin condición declarada → la tarjeta de magnesio aparece.
    base = mn.build_supplement_recommendations(rep, sex="F")
    assert any(i["key"] == "magnesium_mg" for i in base["items"]), "magnesio debería aparecer sin condición"
    # ERC → suprimido (hipermagnesemia: el riñón insuficiente no excreta el exceso).
    renal = mn.build_supplement_recommendations(rep, sex="F", conditions=["enfermedad renal crónica"])
    assert not any(i["key"] == "magnesium_mg" for i in renal["items"]), (
        "magnesio NO debe sugerirse en ERC (riesgo de hipermagnesemia)"
    )
    # Condición no-renal (HTA) → se conserva (no es el riesgo).
    hta = mn.build_supplement_recommendations(rep, sex="F", conditions=["hipertensión arterial"])
    assert any(i["key"] == "magnesium_mg" for i in hta["items"]), "magnesio sí aplica en HTA"


def test_items_carry_precaucion_and_builder_returns_disclaimer():
    out = mn.build_supplement_recommendations(_report_with_gap("iron_mg", "Hierro"), sex="F")
    assert out["items"], "esperaba al menos un item de suplementación"
    for it in out["items"]:
        assert (it.get("precaucion") or "").strip(), f"item {it.get('key')} sin precaucion"
        assert (it.get("dosis_sugerida") or "").strip(), f"item {it.get('key')} sin dosis_sugerida"
    assert (out.get("disclaimer") or "").strip(), "el builder debe retornar un disclaimer no vacío"


def test_ui_surfaces_render_precaucion_and_disclaimer():
    """Parser-based: las 2 superficies que muestran dosis deben referenciar precaucion + disclaimer."""
    meter = (_FRONTEND_DASH / "MicronutrientMeter.jsx").read_text(encoding="utf-8")
    notif = (_FRONTEND_DASH / "NotificationCenter.jsx").read_text(encoding="utf-8")
    assert "precaucion" in meter and "disclaimer" in meter, (
        "MicronutrientMeter.jsx debe renderizar precaucion + disclaimer (dosis sin caveat = inseguro)"
    )
    assert "precaucion" in notif and "disclaimer" in notif, (
        "NotificationCenter.jsx (MicrosDetail) debe renderizar precaucion + disclaimer"
    )
