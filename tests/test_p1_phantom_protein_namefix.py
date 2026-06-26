"""[P1-PHANTOM-PROTEIN-NAMEFIX · 2026-06-26] Honestidad del nombre: proteína-fantasma del título → real.

HALLAZGO (audit de apetibilidad del owner, planes reales e0bf6c46/d21fe7de): el LLM titulaba platos con
una proteína que NO incluía — 'Pollo a la Plancha …' con 0g de pollo (solo ñame/brócoli; el closer
rellenó con camarón → '… y Camarones'), 'Cerdo a la Parrilla con Salsa de Yogurt' sin cerdo. El
`_reflect_added_protein_in_name` (P2-DISH-COHERENCE) reflejaba la proteína AÑADIDA pero no quitaba la
FANTASMA del título.

FIX: `_fix_phantom_protein_in_name` reemplaza la proteína-fantasma LÍDER por la proteína cárnica REAL del
plato (la del closer) y borra el sufijo redundante → 'Camarones a la Plancha …'. CONSERVADOR: si no hay
reemplazo cárnico (proteína = lácteo/yogurt, plato genuinamente incoherente), NO toca el nombre. Cero
falsos positivos sobre platos donde el título SÍ tiene la proteína. Knob MEALFIT_PHANTOM_PROTEIN_NAMEFIX
(default True). Anchor: P1-PHANTOM-PROTEIN-NAMEFIX.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _go_or_skip():
    try:
        import graph_orchestrator as go
        return go
    except Exception as e:  # pragma: no cover - venv sin langchain_openai
        pytest.skip(f"graph_orchestrator no importable en este venv: {type(e).__name__}: {e}")


def _sa(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def _fix(go, name, ings):
    meal = {"name": name, "ingredients_raw": ings}
    changed = go._fix_phantom_protein_in_name(meal, _sa)
    return meal["name"], changed


def test_knob_default_on():
    go = _go_or_skip()
    assert go.PHANTOM_PROTEIN_NAMEFIX_ENABLED is True


def test_knob_registrado():
    from knobs import _env_bool, get_knobs_registry_snapshot
    _env_bool("MEALFIT_PHANTOM_PROTEIN_NAMEFIX", True)
    assert "MEALFIT_PHANTOM_PROTEIN_NAMEFIX" in get_knobs_registry_snapshot()


def test_pollo_fantasma_con_camaron_real_se_corrige():
    """Caso real e0bf6c46: 'Pollo a la Plancha … y Camarones' sin pollo → 'Camarones a la Plancha …'."""
    go = _go_or_skip()
    new, ch = _fix(go,
        "Pollo a la Plancha con Puré de Brócoli y Ñame Asado y Camarones",
        ["0.5 ñame mediano", "4.75 tazas de brócoli", "Jugo de 1 limón", "65g de camarones cocido (65g)"])
    assert ch is True
    assert "Camarones" in new
    assert "Pollo" not in new
    assert "y Camarones" not in new  # sufijo redundante eliminado


def test_cerdo_sin_reemplazo_carnico_no_se_toca():
    """'Cerdo a la Parrilla con Salsa de Yogurt' sin cerdo y proteína = yogurt (lácteo) → fail-safe,
    NO tocar (es una incoherencia de generación, no de naming)."""
    go = _go_or_skip()
    new, ch = _fix(go,
        "Cerdo a la Parrilla con Ñame Asado y Salsa de Yogurt",
        ["0.5 ñame mediano", "1.25 taza de yogurt griego natural (301 g)", "1 cucharada de jugo de limón"])
    assert ch is False
    assert new == "Cerdo a la Parrilla con Ñame Asado y Salsa de Yogurt"


def test_salami_fantasma_con_yogurt_no_se_toca():
    go = _go_or_skip()
    new, ch = _fix(go,
        "Mangú de Ñame con Salami Salteado y Cebolla Caramelizada y Yogurt",
        ["1 ñame mediano", "1.5 cebolla mediana", "1 taza de yogurt cocido (259g)"])
    assert ch is False  # yogurt no es reemplazo cárnico


def test_pechuga_de_pollo_real_no_se_toca():
    """Cero falso positivo: el título tiene 'Pollo' Y los ingredientes traen pechuga de pollo real."""
    go = _go_or_skip()
    new, ch = _fix(go,
        "Pechuga de Pollo Rellena de Queso de Freír con Puré de Batata",
        ["150g de pechuga de pollo", "queso de freir", "batata"])
    assert ch is False


def test_huevos_con_camaron_no_es_fantasma():
    """'Huevos Revueltos con Auyama y Camarones' — la proteína líder (huevos) no está en el mapa cárnico;
    camarones SÍ está presente. No hay fantasma → no toca."""
    go = _go_or_skip()
    new, ch = _fix(go,
        "Huevos Revueltos con Auyama Salteada y Camarones",
        ["2 huevos enteros", "auyama", "camarones cocido"])
    assert ch is False


def test_ceviche_de_pescado_con_filete_real_no_se_toca():
    go = _go_or_skip()
    new, ch = _fix(go,
        "Ceviche de Pescado Blanco con Tostones",
        ["120g de filete de pescado blanco", "0.5 plátano verde"])
    assert ch is False  # 'filete' satisface 'pescado'


def test_meal_sin_nombre_no_revienta():
    go = _go_or_skip()
    assert go._fix_phantom_protein_in_name({"name": "", "ingredients_raw": []}, _sa) is False
    assert go._fix_phantom_protein_in_name({}, _sa) is False


def test_knob_off_no_toca(monkeypatch):
    go = _go_or_skip()
    monkeypatch.setattr(go, "PHANTOM_PROTEIN_NAMEFIX_ENABLED", False)
    new, ch = _fix(go,
        "Pollo a la Plancha con Brócoli y Camarones",
        ["brócoli", "65g de camarones cocido"])
    assert ch is False


def test_pass_cableado_en_assemble():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "graph_orchestrator.py"), encoding="utf-8").read()
    assert "_fix_phantom_protein_in_name(_pm, _sa_phantom)" in src, "el pass no está cableado en assemble"
    assert "P1-PHANTOM-PROTEIN-NAMEFIX" in src
