"""
[P1-4] Test golden: garantiza que el auto-tuner de fatigue_decay no tiene
literales mágicos en el código y respeta los overrides de env vars.

Antes el auto-tuner (cron_tasks._inject_advanced_learning_signals) tenía 5
literales hardcoded:
  - L3514: `tuning_metrics.get("fatigue_decay", 0.9)` → fallback ignoraba env override
  - L3528: `mean_fp > 0.6 and fatigue_decay > 0.70` → FP threshold + lower clamp
  - L3529: `fatigue_decay - 0.03` → step down
  - L3531: `mean_fp < 0.2 and fatigue_decay < 0.98` → FP threshold + upper clamp
  - L3532: `fatigue_decay + 0.02` → step up

Si alguien sube `INGREDIENT_FATIGUE_DECAY_FACTOR=0.85` vía env, los usuarios sin
`tuning_metrics.fatigue_decay` persistido seguían arrancando desde 0.9 — silent
drift entre el default global y el comportamiento real.

Ahora todos vienen de constants.py. Este test es estructural (golden) sobre el
fuente para detectar que los literales no han vuelto.
"""
import os
import re
import sys
from unittest.mock import MagicMock

sys.modules.setdefault('langgraph', MagicMock())
sys.modules.setdefault('langgraph.graph', MagicMock())
sys.modules.setdefault('langgraph.graph.message', MagicMock())


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CRON_TASKS = os.path.join(os.path.dirname(_THIS_DIR), "cron_tasks.py")
_CONSTANTS = os.path.join(os.path.dirname(_THIS_DIR), "constants.py")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_a_constants_module_exposes_all_autotune_params():
    """[P1-4] constants.py debe declarar las 6 constantes del auto-tuner."""
    src = _read(_CONSTANTS)
    expected = [
        "INGREDIENT_FATIGUE_DECAY_FACTOR",
        "INGREDIENT_FATIGUE_DECAY_LOWER_CLAMP",
        "INGREDIENT_FATIGUE_DECAY_UPPER_CLAMP",
        "INGREDIENT_FATIGUE_FP_HIGH_THRESHOLD",
        "INGREDIENT_FATIGUE_FP_LOW_THRESHOLD",
        "INGREDIENT_FATIGUE_DECAY_STEP_DOWN",
        "INGREDIENT_FATIGUE_DECAY_STEP_UP",
    ]
    for name in expected:
        # Línea de la forma: NAME = float(os.environ.get("NAME", "..."))
        assert re.search(rf"^{name}\s*=\s*float\(os\.environ\.get\(", src, re.M), \
            f"constants.py debe declarar {name} con env override"


def test_b_autotuner_imports_constants_not_literals():
    """[P1-4] cron_tasks.py auto-tuner debe importar las 6 constantes y NO usar literales."""
    src = _read(_CRON_TASKS)

    # Bloque del auto-tuner: localizar la docstring que lo identifica.
    block_start = src.find("MEJORA 4: Auto-Tuning del fatigue_decay")
    assert block_start > 0, "El comentario marcador del auto-tuner debe existir"
    # Buscar el final aproximado del bloque (la siguiente "MEJORA" o función nueva).
    block_end = src.find("MEJORA 2: Feedback Loop", block_start)
    assert block_end > block_start, "Bloque del auto-tuner debe estar bien delimitado"
    block = src[block_start:block_end]

    # Las 6 constantes deben aparecer (importadas con alias o por nombre).
    must_appear = [
        "INGREDIENT_FATIGUE_DECAY_FACTOR",
        "INGREDIENT_FATIGUE_DECAY_LOWER_CLAMP",
        "INGREDIENT_FATIGUE_DECAY_UPPER_CLAMP",
        "INGREDIENT_FATIGUE_FP_HIGH_THRESHOLD",
        "INGREDIENT_FATIGUE_FP_LOW_THRESHOLD",
        "INGREDIENT_FATIGUE_DECAY_STEP_DOWN",
        "INGREDIENT_FATIGUE_DECAY_STEP_UP",
    ]
    for name in must_appear:
        assert name in block, \
            f"Constante {name} debe estar referenciada en el bloque auto-tuner"

    # Literales prohibidos: el formato ", 0.9)" en `.get("fatigue_decay", 0.9)`
    assert 'tuning_metrics.get("fatigue_decay", 0.9)' not in block, \
        "REGRESIÓN P1-4: el fallback hardcoded 0.9 volvió. Debe usar INGREDIENT_FATIGUE_DECAY_FACTOR."

    # Literales `> 0.70`, `> 0.6`, `< 0.2`, `< 0.98`, `- 0.03`, `+ 0.02`
    # en líneas que NO sean comentarios.
    forbidden_patterns = [
        (r"mean_fp\s*>\s*0\.6\b", "FP_HIGH_THRESHOLD literal"),
        (r"mean_fp\s*<\s*0\.2\b", "FP_LOW_THRESHOLD literal"),
        (r"fatigue_decay\s*>\s*0\.70\b", "DECAY_LOWER_CLAMP literal"),
        (r"fatigue_decay\s*<\s*0\.98\b", "DECAY_UPPER_CLAMP literal"),
        (r"fatigue_decay\s*-\s*0\.03\b", "DECAY_STEP_DOWN literal"),
        (r"fatigue_decay\s*\+\s*0\.02\b", "DECAY_STEP_UP literal"),
    ]
    for pattern, label in forbidden_patterns:
        # Filtrar líneas comentadas
        for line in block.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not re.search(pattern, line), \
                f"REGRESIÓN P1-4: literal hardcoded ({label}) en línea no-comentada: {line.strip()[:120]}"


def test_c_env_var_override_changes_constant_at_import():
    """[P1-4] Cambiar la env var DEBE reflejarse en constants.* al importarse.

    [P0-5] Crítico: cualquier test que hace `del sys.modules["constants"]` y
    luego `import constants` crea un MÓDULO DISTINTO (nueva identidad) que
    reemplaza el original en `sys.modules`. Los tests cargados antes (como
    `test_pantry_validation_runs_in_llm_path`) ya habían hecho `import constants`
    a nivel de módulo y retienen una referencia a la v1; tras este test, el
    nuevo `sys.modules['constants']` es la v2. Cuando `cron_tasks`'s lazy
    `from constants import X` resuelve, lee la v2; mientras que el test
    `patch.object(constants, "X", mock)` patchea la v1 que tiene como referencia
    local. Resultado: el `_vip` del worker es el real, no el mock; falla
    `test_phantom_ingredient_triggers_retry::vip_mock.call_count = 0`.
    Solución: SALVAR el módulo original antes de borrarlo y RESTAURARLO en el
    finally. Así `sys.modules['constants']` tiene la misma identidad que tenía
    al inicio del test, y todos los demás tests siguen viendo el mismo objeto.
    """
    _orig_constants = sys.modules.get("constants")
    if "constants" in sys.modules:
        del sys.modules["constants"]
    os.environ["INGREDIENT_FATIGUE_DECAY_FACTOR"] = "0.75"
    try:
        import constants  # type: ignore
        assert abs(constants.INGREDIENT_FATIGUE_DECAY_FACTOR - 0.75) < 1e-9, \
            f"env override debe reflejarse, got {constants.INGREDIENT_FATIGUE_DECAY_FACTOR}"
    finally:
        del os.environ["INGREDIENT_FATIGUE_DECAY_FACTOR"]
        # [P0-5] Restaurar la instancia ORIGINAL para preservar identity de módulo
        # entre tests. Si no había una instancia previa (collection ya borró
        # `constants`), eliminamos la v2 — no hay nada que preservar.
        if _orig_constants is not None:
            sys.modules["constants"] = _orig_constants
        elif "constants" in sys.modules:
            del sys.modules["constants"]


def test_d_default_values_are_sane():
    """[P1-4] Los defaults deben respetar invariantes lógicos del auto-tuner.

    [P0-5] Quitamos el `del sys.modules["constants"]` previo al import —
    redundante y, peor, alteraba la identity del módulo cargado, contaminando
    tests posteriores que retenían referencias a la versión anterior. Tomamos
    la versión cacheada en `sys.modules` (test_c ya garantiza que está poblada
    con env defaults).
    """
    import constants  # type: ignore

    # Decay debe estar dentro de los clamps
    assert (
        constants.INGREDIENT_FATIGUE_DECAY_LOWER_CLAMP
        <= constants.INGREDIENT_FATIGUE_DECAY_FACTOR
        <= constants.INGREDIENT_FATIGUE_DECAY_UPPER_CLAMP
    ), "Default decay debe estar dentro del rango [LOWER_CLAMP, UPPER_CLAMP]"

    # FP thresholds: low < high, ambos en (0,1)
    assert 0 < constants.INGREDIENT_FATIGUE_FP_LOW_THRESHOLD < constants.INGREDIENT_FATIGUE_FP_HIGH_THRESHOLD < 1
    # Steps positivos
    assert constants.INGREDIENT_FATIGUE_DECAY_STEP_DOWN > 0
    assert constants.INGREDIENT_FATIGUE_DECAY_STEP_UP > 0
    # Clamps en (0,1)
    assert 0 < constants.INGREDIENT_FATIGUE_DECAY_LOWER_CLAMP < constants.INGREDIENT_FATIGUE_DECAY_UPPER_CLAMP < 1
