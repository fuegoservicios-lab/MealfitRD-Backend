"""[P3-CLAUDEMD-CAP · 2026-05-14] Test guard: CLAUDE.md size cap.

CLAUDE.md se auto-carga en cada turn de la conversación con Claude
Code. Chars extras se traducen directo a costo de tokens + latencia
per-turn. Históricamente: ~25k inicial → 90.5k pre-trim → 50.4k
post-trim 2026-05-14 (limpieza de 6 fases, -46% del original).

Este test bloquea la regresión: si CLAUDE.md vuelve a crecer
descontroladamente (más P-fixes, más anti-patrones, más SOPs sin
podar el contenido viejo), CI falla y forza al autor a aplicar uno
de los patrones de limpieza establecidos.

Cuando este test falla:

  1. Revisa el diff. ¿Qué se añadió?

  2. ¿Es contrato load-bearing irreducible? Si sí, dos opciones:
     a) Bumpear el cap (commit visible en review). Bumps al cap
        señalan deuda acumulada — si suben >10% en una sesión,
        considerar limpieza estructural (como la de 2026-05-14).
     b) Cortar contenido equivalente viejo para mantener cap.

  3. ¿Es narrativa / ejemplo pedagógico / SOP? Mueve a memoria o
     docs/ siguiendo los patrones establecidos 2026-05-14:

       - Tabla canónica con test parser-based  → `docs/<nombre>.md`
         (test parsea el doc, CLAUDE.md tiene 1-line + link).
       - Diagrama ASCII / narrativa larga      → memoria runbook
         + 1-line stub en CLAUDE.md.
       - Bloque `# ❌ NUNCA` pedagógico         → memoria
         (ej. `runbook_security_antipatterns.md`).
       - SOPs paso-a-paso                       → memoria runbook
         (ej. `runbook_system_alerts_sops_*.md`).
       - Bullet de Convenciones >300 chars      → memoria + link
         inline.

Override del cap:
  `MEALFIT_CLAUDE_MD_MAX_CHARS=N python -m pytest ...`

Default conservador: 52000 (margen ~1.6k sobre el estado post-trim
2026-05-14). Clamp [10000, 200000] para defensa contra typos.

Tooltip-anchor: P3-CLAUDEMD-CAP-START | size guard 2026-05-14
"""
from __future__ import annotations

import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"

_DEFAULT_CAP = 56500  # [P1-NEON-DB-MIGRATION · 2026-06-12] +3.5k para los 3 bullets de las migraciones del día (DeepSeek + Cohere + Neon), ya adelgazados doc-first (detalle en docs/llm_tier_routing.md, docs/embeddings_cohere.md, docs/neon_db_migration.md). Bump 6.6%, bajo el threshold 10% que dispara limpieza estructural. Próxima limpieza candidata: mover tablas "Advisors aceptados" a docs/ (verificar antes test_p2_whitelist_advisors_anchors_alive).
_CAP_FLOOR = 10000     # típico minimum útil (esqueleto de invariantes)
_CAP_CEILING = 200000  # ~5x el threshold del UI; arriba de eso es absurdo


def _get_cap() -> int:
    """Lee cap del env var con default conservador. Clampa al rango
    [_CAP_FLOOR, _CAP_CEILING] para defensa contra typos (un 0
    accidental dejaría el test siempre rojo; un 9999999 lo dejaría
    siempre verde)."""
    raw = os.environ.get("MEALFIT_CLAUDE_MD_MAX_CHARS", str(_DEFAULT_CAP))
    try:
        cap = int(raw)
    except (TypeError, ValueError):
        cap = _DEFAULT_CAP
    return max(_CAP_FLOOR, min(cap, _CAP_CEILING))


def test_claude_md_exists():
    """Sanity: CLAUDE.md está en la raíz del repo. Si moviste la raíz,
    actualizar `_REPO_ROOT` en este test."""
    assert _CLAUDE_MD.exists(), (
        f"CLAUDE.md no encontrado en {_CLAUDE_MD}. ¿Refactor de la raíz "
        f"del repo? Actualizar `_REPO_ROOT` en {__file__}."
    )


def test_cap_knob_clamp_lower():
    """Knob inválido (negativo / 0 / typo) → clampa al floor. Defensa
    contra `MEALFIT_CLAUDE_MD_MAX_CHARS=0` accidental que dejaría el
    test siempre rojo."""
    os.environ["MEALFIT_CLAUDE_MD_MAX_CHARS"] = "0"
    try:
        assert _get_cap() == _CAP_FLOOR, (
            "Knob clamp inferior roto: 0 debería clampar a _CAP_FLOOR."
        )
        os.environ["MEALFIT_CLAUDE_MD_MAX_CHARS"] = "no-es-int"
        assert _get_cap() == _DEFAULT_CAP, (
            "Knob malformado debería caer al default, no crashear."
        )
    finally:
        os.environ.pop("MEALFIT_CLAUDE_MD_MAX_CHARS", None)


def test_cap_knob_clamp_upper():
    """Knob absurdamente alto → clampa al ceiling. Defensa contra
    `MEALFIT_CLAUDE_MD_MAX_CHARS=9999999` que dejaría el guard inútil."""
    os.environ["MEALFIT_CLAUDE_MD_MAX_CHARS"] = "999999999"
    try:
        assert _get_cap() == _CAP_CEILING, (
            "Knob clamp superior roto: valores absurdos deberían clampar."
        )
    finally:
        os.environ.pop("MEALFIT_CLAUDE_MD_MAX_CHARS", None)


def test_claude_md_size_under_cap():
    """**Test principal**: CLAUDE.md debe estar bajo el cap configurado.

    CLAUDE.md se auto-carga en cada turn de la conversación; chars en
    exceso se traducen directo a costo de tokens + latencia per-turn.

    SOP cuando falla (de menor a mayor invasividad):

      1. **Revisar el diff** (`git diff CLAUDE.md`). Identifica qué bloque
         creció. ¿Una sección de anti-patrones? ¿Un bullet de convenciones?

      2. **¿Contiene un ejemplo de código `# ❌ NUNCA` o un bloque
         SQL pedagógico?** Mueve a memoria
         (`runbook_security_antipatterns.md` para anti-patrones; nuevo
         runbook para otros). CLAUDE.md mantiene header + 1-line +
         link.

      3. **¿Una tabla larga con test parser-based?** Mueve a
         `docs/<nombre>.md`. Actualiza la constante `_CLAUDE_MD` en el
         test para apuntar al doc. CLAUDE.md mantiene 1-line + link.
         (Pattern probado 2026-05-14 con system_alerts +
         coherence_surfaces).

      4. **¿Un diagrama ASCII o narrativa "qué pasó antes"?**
         Mueve a memoria runbook. CLAUDE.md mantiene 1-line + link.

      5. **¿Contenido es contrato load-bearing irreducible?** Bumpea
         el cap (visible en code review). Considera: si el cap sube
         >10% en una sesión, planifica una limpieza estructural (~3-6
         horas, ahorra hasta 50% del tamaño según pattern 2026-05-14).
    """
    size = _CLAUDE_MD.stat().st_size
    cap = _get_cap()
    assert size <= cap, (
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"CLAUDE.md = {size:,} chars > cap {cap:,} chars (diff +{size - cap:,})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"CLAUDE.md se auto-carga en CADA turn de la conversación con\n"
        f"Claude Code. Chars extras = tokens extras = latencia extra POR TURN.\n"
        f"\n"
        f"SOP para resolver (ver docstring de este test para detalle):\n"
        f"  1. `git diff CLAUDE.md`         → identifica qué creció\n"
        f"  2. ejemplo `# ❌ NUNCA` / SQL    → mover a memoria runbook\n"
        f"  3. tabla con test parser-based  → mover a `docs/`\n"
        f"  4. diagrama ASCII / narrativa    → mover a memoria runbook\n"
        f"  5. contrato load-bearing irreducible → bumpear el cap\n"
        f"\n"
        f"Patrones establecidos en limpieza 2026-05-14 (-46% del original):\n"
        f"  - `backend/docs/system_alerts_resolution_table.md` (tabla canónica)\n"
        f"  - `backend/docs/coherence_surfaces_table.md`\n"
        f"  - `runbook_security_antipatterns.md` (auth/billing/webhook/agent)\n"
        f"  - `runbook_plan_id_lifecycle.md` (diagrama ASCII)\n"
        f"  - `runbook_coherence_guard_flow.md` (diagrama + trade-offs)\n"
        f"  - `runbook_advisors_operational_subsections.md`\n"
        f"\n"
        f"Override del cap (último recurso):\n"
        f"  MEALFIT_CLAUDE_MD_MAX_CHARS={cap + 5000} python -m pytest ...\n"
        f"  Mejor: bumpea `_DEFAULT_CAP` en este test (visible en review).\n"
    )
