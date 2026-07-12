"""[P2-VISION-GUINEO-PLATANO · 2026-07-12] gemma distingue guineo de plátano (RD).

Vivo (owner): foto de UN GUINEO al chat → gemma lo nombró 'plátano' (traducción
genérica banana→plátano) → el agente agregó 'Plátano verde' a la Nevera. El
owner buscó "gui", no vio nada nuevo y concluyó que la tool no corrió — la
escritura SÍ ocurrió (user_inventory 12:30:00 exacto), pero sobre el alimento
EQUIVOCADO. En RD guineo (banana dulce, cruda) y plátano (grande, se cocina)
son alimentos distintos con macros distintas.

Regla añadida a AMBOS prompts de visión (meal-scan de vision_agent y escáner
de Nevera de user_data — mismo modelo, misma confusión).
tooltip-anchor: P2-VISION-GUINEO-PLATANO
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

from vision_agent import _MEAL_VISION_PROMPT  # noqa: E402


def test_meal_scan_prompt_teaches_guineo():
    assert "GUINEO" in _MEAL_VISION_PROMPT
    assert "NO lo llames platano" in _MEAL_VISION_PROMPT
    assert "se come cruda" in _MEAL_VISION_PROMPT
    assert _MEAL_VISION_PROMPT.isascii(), "el prompt sigue 100% ASCII (transporte Ollama)"


def test_pantry_scan_prompt_teaches_guineo():
    with open(os.path.join(_BACKEND, "routers", "user_data.py"), encoding="utf-8") as f:
        ud = f.read()
    i = ud.find("_VISION_PROMPT = (")
    assert i != -1, "el prompt del escáner de Nevera desapareció"
    win = ud[i:ud.find("\n)", i)]
    assert "GUINEO" in win and "NO lo llames platano" in win, \
        "misma lección en el escáner de Nevera (mismo modelo, misma confusión)"
