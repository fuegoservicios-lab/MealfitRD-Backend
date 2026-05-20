"""[P1-AGENT-WELCOME-NO-TIME · 2026-05-20] Refinamiento del welcome screen
del AgentPage tras feedback del user.

Cambios cerrados:
  1. Removida la hora literal del welcome ("Son las 04:29 a. m.." → no más).
     Razón UX: como el welcome se regenera cada 30min (no en cada nav), la
     hora mostrada podía desfasarse ±30min de la hora real → se ve raro.
     El `timeGreeting` (madrugadas/días/tardes/noches) sigue dando contexto
     temporal grueso sin precisión innecesaria.

  2. `_setWelcomeIfAbsent` ahora regenera si el welcome existente es >30min.
     Pre-fix (P1-AGENT-WELCOME-STABLE inicial) lo mantenía PARA SIEMPRE
     → si el user dejaba el tab abierto al amanecer, seguía diciendo
     "Buenas madrugadas" cuando debería decir "Buenos días". 30min es el
     sweet spot: refresca el saludo sin causar flash perceptible.

  3. Todos los callsites que crean welcome objects (`{isWelcome: true}`)
     ahora incluyen `welcomeAt: Date.now()` timestamp para que el helper
     pueda detectar si está fresco/stale.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_welcome_return_has_no_literal_time():
    """[P1-AGENT-WELCOME-NO-TIME] El return de `generateIntelligentWelcome`
    NO debe concatenar `Son las X:Y a.m.` ni `timeStr`. El saludo grueso
    `timeGreeting` (Buenas madrugadas/días/etc.) es suficiente."""
    src = _read(_AGENT_PAGE_JSX)
    fn = re.search(r"const generateIntelligentWelcome\s*=.*?\};", src, re.DOTALL)
    assert fn, "generateIntelligentWelcome no encontrada"
    body = fn.group(0)
    # Extraer solo el return template literal final.
    ret = re.search(r"return\s+`[^`]+`", body)
    assert ret, "return template literal no encontrado"
    ret_str = ret.group(0)
    assert "Son las" not in ret_str, (
        f"Return aún contiene 'Son las' literal — el bug del refresh visible "
        f"de la hora vuelve. Ver P1-AGENT-WELCOME-NO-TIME · 2026-05-20.\n"
        f"Return: {ret_str}"
    )
    assert "timeStr" not in ret_str, (
        f"Return aún concatena `timeStr` — misma incidencia.\n"
        f"Return: {ret_str}"
    )


def test_set_welcome_helper_refreshes_after_30min():
    """[P1-AGENT-WELCOME-NO-TIME] El helper `_setWelcomeIfAbsent` debe
    regenerar el welcome si el actual es >=30min, NO mantenerlo eternamente.
    Sin esto, el saludo "Buenas madrugadas" sigue al user durante todo
    el día."""
    src = _read(_AGENT_PAGE_JSX)
    helper = re.search(
        r"_setWelcomeIfAbsent\s*=\s*useCallback\(\s*\(\s*\)\s*=>\s*\{(.+?)\}\s*,",
        src,
        re.DOTALL,
    )
    assert helper, "_setWelcomeIfAbsent no encontrado"
    body = helper.group(1)
    assert "welcomeAt" in body, (
        "Helper no consulta `welcomeAt` — no puede detectar staleness. Ver "
        "P1-AGENT-WELCOME-NO-TIME · 2026-05-20."
    )
    # Debe haber una constante o cálculo de 30min en el módulo.
    has_30min = bool(re.search(r"30\s*\*\s*60\s*\*\s*1000|_WELCOME_REFRESH_MS", src))
    assert has_30min, (
        "Constante de refresh 30min (30*60*1000 o _WELCOME_REFRESH_MS) ausente. "
        "Ver P1-AGENT-WELCOME-NO-TIME."
    )
    # El helper debe comparar age vs el refresh threshold.
    assert re.search(r"ageMs|Date\.now\(\)\s*-", body), (
        "Helper no compara age del welcome — no detecta cuándo regenerar."
    )


def test_all_welcome_objects_have_welcomeAt():
    """[P1-AGENT-WELCOME-NO-TIME] TODOS los callsites que crean
    `{isWelcome: true}` deben incluir `welcomeAt: Date.now()` cerca
    (dentro de 150 chars). Sin esto, el helper ve `welcomeAt: undefined`
    y siempre regenera (bug del refresh constante vuelve)."""
    src = _read(_AGENT_PAGE_JSX)
    positions = [m.start() for m in re.finditer(r"isWelcome\s*:\s*true", src)]
    missing = []
    for pos in positions:
        window = src[max(0, pos - 150):pos + 150]
        if "welcomeAt" not in window:
            snippet = src[max(0, pos - 30):pos + 30]
            missing.append(snippet)
    assert not missing, (
        f"{len(missing)} callsite(s) de `isWelcome: true` SIN `welcomeAt` "
        f"cercano. El helper no podrá detectar staleness en esos casos. "
        f"Ver P1-AGENT-WELCOME-NO-TIME · 2026-05-20.\n"
        f"Snippets: {missing}"
    )
    # Sanity: al menos 4 callsites totales (initializer, logout, hydration,
    # handleNewChat, retry-failed-msg, helper).
    assert len(positions) >= 4, (
        f"Solo {len(positions)} callsites de `isWelcome: true` — esperaba >=4. "
        f"¿Algún refactor quitó el welcome de algún lugar legítimo?"
    )
