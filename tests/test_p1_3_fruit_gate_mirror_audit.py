"""[P1-FRUIT-GATE-MIRROR-AUDIT · 2026-07-10] Audit del roadmap post-forensic corr=d57ffe04: "8 rechazos/
72h pese a FRUIT-DEDUP". Root cause encontrada: `dedup_featured_fruits_in_plan` corre UNA sola vez en
todo el pipeline (assemble, línea ~18749) — mientras `self_critique_node` corrige días vía LLM (línea
~9462) y verifica el residual de PROTEÍNA post-corrección (P1-CRITIQUE-SAMEDAY-PROTEIN-PARITY, comentario
literal "Espeja el cierre de FRUTA") pero NUNCA re-ejecuta el dedup de fruta sobre los días recién
corregidos. Si la corrección LLM de self-critique reintroduce (o deja) una fruta repetida, nada la
detecta hasta que el revisor la rechaza — quemando un intento completo idéntico al patrón ya resuelto
para proteína. Fix: re-fire `dedup_featured_fruits_in_plan` en el MISMO seam que la verificación de
proteína, cerrando la asimetría que el comentario ya afirmaba (falsamente) que estaba cerrada.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO_SRC = f.read()


def test_fruit_dedup_re_fired_in_self_critique_protein_parity_seam():
    """El seam que verifica residual de proteína post self-critique (P1-CRITIQUE-SAMEDAY-PROTEIN-PARITY)
    debe re-disparar TAMBIÉN el dedup de fruta sobre `partial` — mismo plan mutado en el mismo punto."""
    # anchor único al seam real (`partial["days"] = days` dentro de self_critique_node — el marker
    # por sí solo también aparece en el bloque de knobs al tope del archivo y en la docstring del
    # detector espejo de proteína, ambos falsos positivos para `.index()`).
    anchor = 'if corrected_any:\n                partial["days"] = days'
    i = _GO_SRC.index(anchor)
    # ventana amplia: cubre el fix de fruta + el check de proteína residual + la rama else
    window = _GO_SRC[i:i + 2400]
    assert "dedup_featured_fruits_in_plan(partial)" in window, \
        "self_critique debe re-ejecutar el dedup de fruta sobre el plan corregido, no solo verificar proteína"
    # orden: el dedup de fruta corre ANTES del check de proteína residual, ambos dentro del mismo
    # `if corrected_any:` seam — no después de la rama del else visual-ok.
    assert window.index("dedup_featured_fruits_in_plan(partial)") < window.index(
        "P1-CRITIQUE-SAMEDAY-PROTEIN-PARITY"), \
        "el re-fire de fruta debe vivir ANTES del check de proteína, en la misma rama de corrección"


def test_marker_present_for_the_audit_fix():
    assert "P1-FRUIT-GATE-MIRROR-AUDIT" in _GO_SRC
