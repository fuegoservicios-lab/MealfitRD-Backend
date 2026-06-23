"""[P3-TOFU-REMOVE · 2026-06-22] El tofu (y la soya como producto) se eliminó del pool de
proteínas OFRECIBLES porque La Sirena no lo vende. Pero las refs de tofu en allergy/condition
handling SE CONSERVAN (defensa: usuario alérgico a soya que menciona tofu en texto libre).

Este test ancla la asimetría:
- OFERTA: "Soya/Tofu" fuera de DOMINICAN_PROTEINS; "soya/tofu" fuera del variety map; el mensaje
  de violación de dieta no sugiere tofu.
- HANDLING: condition_rules conserva el swap tofu→pollo (soy allergy) y el token map de soya.
"""
from __future__ import annotations

from pathlib import Path

from constants import DOMINICAN_PROTEINS

_BACKEND = Path(__file__).resolve().parent.parent


def test_tofu_not_offered_in_protein_pool():
    joined = " | ".join(p.lower() for p in DOMINICAN_PROTEINS)
    assert "tofu" not in joined, f"tofu no debe estar en DOMINICAN_PROTEINS: {DOMINICAN_PROTEINS}"
    assert "soya" not in joined, f"soya no debe estar en DOMINICAN_PROTEINS: {DOMINICAN_PROTEINS}"
    # Sanidad: las leguminosas (proteína vegana de reemplazo) siguen presentes.
    assert any("lenteja" in p.lower() or "garbanzo" in p.lower() or "habichuela" in p.lower()
               for p in DOMINICAN_PROTEINS), "deben quedar leguminosas como proteína vegana"


def test_tofu_marker_present_and_variety_clean():
    src = (_BACKEND / "constants.py").read_text(encoding="utf-8")
    assert "P3-TOFU-REMOVE" in src
    # El variety map de legumbres ya no incluye soya/tofu.
    # (buscamos la línea de legumbres y verificamos que no tenga soya/tofu)
    for line in src.splitlines():
        if '"legumbres"' in line and "habichuelas" in line:
            code_part = line.split("#")[0]  # ignorar comentario (que sí menciona soya/tofu)
            assert "soya/tofu" not in code_part, f"variety legumbres aún ofrece soya/tofu: {line.strip()}"
            break
    else:
        raise AssertionError("no se encontró la línea variety 'legumbres'")


def test_diet_violation_message_does_not_suggest_tofu():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    # El mensaje de DIETA INCOMPATIBLE no debe sugerir tofu como alternativa.
    assert "leguminosas, tofu" not in src, "el mensaje de violación de dieta aún sugiere tofu"


def test_soy_allergy_handling_preserved():
    """CRÍTICO: las refs de tofu en allergy handling SE CONSERVAN (no romper seguridad alérgica)."""
    cr = (_BACKEND / "condition_rules.py").read_text(encoding="utf-8")
    assert "tofu" in cr.lower(), "condition_rules debe conservar el handling de tofu para alérgicos a soya"
    # El token de alérgeno 'soy' debe seguir cubriendo tofu.
    assert '"soy"' in cr and "tofu" in cr.lower()
