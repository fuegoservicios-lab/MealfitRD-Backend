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

_DEFAULT_CAP = 69800  # [P2-SUPERMARKET-TOKEN-SPLIT · 2026-08-14] 68700→69800 (+1,60%, muy bajo el threshold 10%): la fila del Supermercado pasa a declarar que su gate ya NO es el `CRON_SECRET`, y con ella el precio que se aceptó a cambio — DOS secretos que rotar. Ese precio es justo lo que un futuro lector querría saber antes de "simplificar" volviendo a un solo token: sin la fila, unificarlos otra vez parece limpieza y es reabrir el radio de daño (el editor vive en una página PÚBLICA y el maestro abre `purge-data` sobre 33 tablas). Próxima limpieza candidata (sigue vigente): mover tablas "Advisors aceptados" a docs/ (verificar antes test_p2_whitelist_advisors_anchors_alive). # [P2-LANDING-PROD · 2026-08-14] 66900→68700 (+2,69%, bajo el threshold 10%): entran las cuatro reglas anti-refactor de la tanda P2, y las cuatro son del tipo «la corrección obvia es la equivocada» — el material por el que existe esta sección. Sin la de COPY-TRUTH alguien deriva el texto del plan anual del objeto `PRICING`, donde `ultra.annual` sigue vivo, y resucita un plan que cobra 449.99 CADA MES; sin la de SUPERMARKET-CACHE alguien mueve la invalidación a después del `await` creyendo que así cierra la carrera (deja la MISMA ventana desplazada); sin la de SITEMAP alguien edita a mano un fichero generado y su cambio desaparece en el build; sin la de OLA1-DIET alguien devuelve lucide a `manualChunks` sin saber que un vendor NOMBRADO recibe modulepreload eager. Los 25 gaps y sus mediciones viven en docs/, no aquí. Próxima limpieza candidata (sigue vigente): mover tablas "Advisors aceptados" a docs/ (verificar antes test_p2_whitelist_advisors_anchors_alive). # [P1-LANDING-PROD · 2026-08-14] 64800→66900 (+3,24%, bajo el threshold 10%): entran los tres P1 del plan de producción del landing, y las tres entradas son advertencias anti-refactor con su causa medida — exactamente lo que esta sección existe para conservar. Sin la de SW-DEFER alguien devuelve `immediate: true` creyendo que acelera el arranque (acelera el del SW, a costa del hero); sin la de OBS-PAPER alguien "resuelve" el peso de la observabilidad bajando `VITE_SENTRY_REPLAYS_SESSION_RATE` a 0, que NO ahorra un byte porque el chunk se descarga igual; sin la de HEAD-PRELOAD alguien sustituye el bloque gateado por un `<link>` fijo y le devuelve a app.* los 226 kB de landing que P3-APP-SUBDOMAIN-BUILD-SEP le había quitado. El detalle largo (25 gaps) vive en docs/, no aquí. Próxima limpieza candidata (sigue vigente): mover tablas "Advisors aceptados" a docs/ (verificar antes test_p2_whitelist_advisors_anchors_alive). # [P1-PLAN-MODE · 2026-08-11] 63426→64800 (+2,17%, bajo el threshold 10%): el modo seguimiento entra con sus dos piezas canónicas — la fila de quota-exemption (GET/PUT /profile/plan-mode + GET /nutrition/targets: el PUT es la puerta de REANUDAR, un 402 ahí deja al usuario atrapado en pausa) y la nota de producto (pausa de DOS capas: gate SQL en el pickup + cancelación de los 5 estados resucitables INCLUIDO pending_user_action, orden flag-first). Sin la nota, el próximo auditor ve un flag en user_profiles que "nadie lee" en plan_data y lo simplifica a una columna del jsonb — que es exactamente el modo de fallo _frozen_at que motivó el diseño. # [P1-MANUAL-FOOD-LOG · 2026-08-11] 62100→63426 (+2.14%, bajo el threshold 10%): una fila NUEVA en la tabla Historial-quota-exemption — los 3 endpoints del componedor del diario (manual/frequent/repeat) + el limitador que a /consumed le faltaba desde siempre. Es exactamente el contenido que esa tabla existe para conservar: sin la fila, el proximo auditor ve endpoints de escritura sin verify_api_quota y los "arregla" poniendo el paywall — que es el bug historico (congelar la funcion Y quemar credito de planes) que la tabla documenta. # [P1-HERO-DEDUP-ACCENT · 2026-08-09] 60800→62100 (+2,14%, muy bajo el threshold 10%): el margen previo era de 95 bytes, o sea que CUALQUIER entrada lo rompía — el bump no lo causa esta entrada sola, lo causa que el cap anterior ya estaba agotado. La entrada nueva es una decisión de producto que SUPERSEDE PARCIALMENTE a P2-PAPER-NO-INK ocho días después de ella: nace la primera tinta de color del sistema papel, con su semántica ("marca la cifra de un SSOT", condición necesaria no suficiente) y sus 2 call sites. Sin ella, el próximo que lea «blanco y negro estricto, cero color» en esta misma sección borra el acento creyendo que es drift — que es exactamente el modo de fallo que la sección "Decisiones de producto" existe para prevenir. Próxima limpieza candidata (sigue vigente): mover tablas "Advisors aceptados" a docs/ (verificar antes test_p2_whitelist_advisors_anchors_alive). [P1-CHUNK-REBASE-PAUSED · 2026-08-08] 60400→60800 (+0,66%): la secuela del rebase — los chunks pausados eran invisibles para la cadena y w4 aterrizó encima de la w2 pausada el MISMO día siguiente al fix del ancla; la entrada ya nació comprimida dos veces (1.090→410 bytes) y el margen previo era ~100 bytes, o sea que el bump es el crecimiento real de la regla, no grasa. Próxima limpieza candidata (sigue vigente): mover tablas "Advisors aceptados" a docs/ (verificar antes test_p2_whitelist_advisors_anchors_alive). [P1-CHUNK-OFFSET-REBASE · 2026-08-07] 59700→60400 (+1,17%): la regla "si mueves el ancla del plan, mueve los offsets de la cola" es anti-refactor pura — la ventana rolling ya ha mordido seis veces y esta vez dejó a 3 de 3 planes vivos sin relleno a tiempo. [P1-PANTRY-RECONCILIATION · 2026-08-07] 58400→59700 (+2,23%, muy bajo el threshold 10%): dos olas de trabajo entraron sin bumpear el cap y lo dejaron en rojo por 71 bytes ANTES de esta sesión — P1-LANDING-BENCH-1 (benchmark del landing) y las dos entradas de la rama de consumo (fila de quota-exemption de `POST /api/diary/consumed-from-plan` + el párrafo anti-refactor P1-PANTRY-NAME-RESOLUTION, "no reimplementes la identidad de la Nevera sobre GLOBAL_REVERSE_MAP porque colapsa pechuga→pollo"). Las tres son exactamente lo que CLAUDE.md existe para conservar: una fila de tabla canónica y una advertencia anti-refactor con su causa. Nota de proceso: el cap se mide sobre la CLAUDE.md RAÍZ (parents[2]), y la copia `backend/CLAUDE.md` es la que las ramas editan — si solo tocas una, este test no lo ve hasta que alguien sincroniza. Próxima limpieza candidata (sigue vigente): mover tablas "Advisors aceptados" a docs/ (verificar antes test_p2_whitelist_advisors_anchors_alive). [P1-DIET-CANON-SSOT · 2026-07-31] 58000→58400 (+0.69%, muy bajo el threshold 10%): el audit solver+seeder v6 añadió la sección "El path degradado necesita su propio backstop" (P0-DEGRADED-SAFETY-SCAN + P1-DIET-CANON-SSOT). Ambas entradas son advertencias anti-refactor —"no borres el tamiz creyendo que el filtro ya cubre eso", "no escribas una 4ª tabla de dieta"— o sea justo lo que CLAUDE.md existe para conservar; comprimirlas más las volvía crípticas. Se llegó a +1 char sobre el cap recortando dos veces: el margen real era CERO, no el bump. Próxima limpieza candidata (sigue vigente): mover tablas "Advisors aceptados" a docs/ (verificar antes test_p2_whitelist_advisors_anchors_alive). [P2-TRIAGE-REALBUGS · 2026-06-16] 56500→58000 (+2.65%, bajo el threshold 10%): la CLAUDE.md root creció a 56733 por bullets P-fix acumulados (UVICORN-RELOAD/SQL-forensic/soft-fail/shift-plan-exempt) y rompía el cap previo sin margen de respiración (test_p1_prod_final_1 exige ≥800). Próxima limpieza candidata: mover tablas "Advisors aceptados" a docs/ (verificar antes test_p2_whitelist_advisors_anchors_alive). [P1-NEON-DB-MIGRATION · 2026-06-12] el bump previo a 56500 absorbió DeepSeek+Cohere+Neon, ya adelgazados doc-first.
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
    # [2026-07-31] BYTES, no caracteres — ver la nota en `test_p1_prod_final_1.py`.
    # En español la brecha es de ~1000 (acentos y emojis a 2+ bytes), así que quien
    # depure con `len(read_text())` verá otro número. La medida se queda en bytes
    # (conservadora); la etiqueta es la que estaba mal.
    size = _CLAUDE_MD.stat().st_size
    cap = _get_cap()
    assert size <= cap, (
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"CLAUDE.md = {size:,} bytes > cap {cap:,} bytes (diff +{size - cap:,})\n"
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
