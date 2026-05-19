"""[P3-HIST-CHUNK-SCHEDULED · 2026-05-18] Backend split de chunk_in_flight_count
en `chunk_scheduled_count` (dormido) vs `chunk_running_now_count` (corriendo)
+ copy preciso del banner del modal.

Síntoma reportado por el usuario:
> "El texto que dice que mealfit está generando los 4 días restantes en
>  segundo plano ahora es real que lo está haciendo? ya que no debería
>  generarlos ahora ya que no ha llegado su tiempo, si no es el tiempo
>  que le corresponde debe estar encolado para cuando llegue el momento."

Causa raíz: el endpoint /history-list devolvía `chunk_in_flight_count`
contando chunks con status IN ('pending', 'processing', 'stale') —
mezclando 2 dimensiones independientes:

  1. ¿Es elegible AHORA según `execute_after`? El worker filtra `WHERE
     execute_after <= NOW()` (cron_tasks.py:20376) antes del pickup.
  2. ¿Está siendo ejecutado AHORA? Solo `status='processing'`.

Un chunk `pending` con `execute_after = grocery_start_date + 3 días`
(plan de 7d, hoy es día 1) está DORMIDO esperando llegar al jueves —
el worker NO lo toca aún. El frontend mostraba "Mealfit los está
generando AHORA en segundo plano" para estos chunks → mentira.

Fix:
  - Backend: SQL split en 2 contadores nuevos
    - `chunk_scheduled_count`: pending/stale con execute_after > NOW()
    - `chunk_running_now_count`: processing + pending/stale con
      execute_after <= NOW() (NULL counts here, defensive)
  - Frontend: ramas separadas en missingDaysReason del modal
    - _runningNow > 0  → "Mealfit los está generando ahora..."
    - _scheduled > 0   → "Se generarán automáticamente cuando llegue..."
    - mixto            → "Algunos ahora; el resto cuando llegue su momento"
    - fallback _inFlight > 0 (backend legacy) → copy neutro sin mentir
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_HISTORY_JSX = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "History.jsx"
).read_text(encoding="utf-8")
_PLANS_ROUTER = (_BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")


def test_marker_present_in_source():
    """Marker P3-HIST-CHUNK-SCHEDULED debe permanecer en JSX + plans.py
    como anchor de regresión."""
    assert "P3-HIST-CHUNK-SCHEDULED" in _HISTORY_JSX, (
        "Marker P3-HIST-CHUNK-SCHEDULED ausente en History.jsx — el "
        "split runningNow/scheduled puede haberse degradado."
    )
    assert "P3-HIST-CHUNK-SCHEDULED" in _PLANS_ROUTER, (
        "Marker P3-HIST-CHUNK-SCHEDULED ausente en routers/plans.py — "
        "el SQL split puede haberse perdido."
    )


def test_backend_split_sql_scheduled_count():
    """El SQL del LATERAL qstats DEBE definir `scheduled_count` con el
    filtro `status IN ('pending','stale') AND execute_after > NOW()`.
    Sin este split, el frontend cae al fallback genérico ("generará
    automáticamente"), perdiendo la precisión de distinguir dormidos
    vs corriendo."""
    # Filtro canónico: el orden de elementos en el SET puede variar
    # entre 'pending' primero o 'stale' primero — regex tolerante.
    assert "scheduled_count" in _PLANS_ROUTER, (
        "Columna `scheduled_count` ausente del SELECT del LATERAL qstats."
    )
    # Reglas del filtro: ambos elementos presentes + execute_after >
    # NOW() comparison.
    pat = re.compile(
        r"COUNT\(\*\)\s+FILTER\s*\(\s*WHERE\s+status\s+IN\s*\(\s*'pending'\s*,\s*'stale'\s*\)[^)]*execute_after\s*>\s*NOW\(\)[^)]*\)\s+AS\s+scheduled_count",
        re.DOTALL,
    )
    assert pat.search(_PLANS_ROUTER), (
        "Filtro de `scheduled_count` no matchea el patrón canónico "
        "(WHERE status IN ('pending','stale') AND execute_after > NOW()). "
        "Si lo cambiaste, asegúrate de que el contador SIGUE significando "
        "'chunks dormidos esperando su execute_after'."
    )


def test_backend_split_sql_running_now_count():
    """`running_now_count` DEBE cubrir processing + (pending/stale con
    execute_after vencido O NULL). El NULL es defensivo para chunks
    legacy sin execute_after seteado."""
    assert "running_now_count" in _PLANS_ROUTER, (
        "Columna `running_now_count` ausente del SELECT del LATERAL qstats."
    )
    # Filtro debe incluir 'processing' en el set Y el predicado
    # `execute_after IS NULL OR execute_after <= NOW()`.
    pat = re.compile(
        r"COUNT\(\*\)\s+FILTER\s*\(\s*WHERE\s+status\s+IN\s*\([^)]*'processing'[^)]*\)[^)]*"
        r"\(\s*execute_after\s+IS\s+NULL\s+OR\s+execute_after\s*<=\s*NOW\(\)\s*\)"
        r"[^)]*\)\s+AS\s+running_now_count",
        re.DOTALL,
    )
    assert pat.search(_PLANS_ROUTER), (
        "Filtro de `running_now_count` no matchea el patrón canónico. "
        "Debe incluir 'processing' en el status set + cubrir "
        "`execute_after IS NULL OR execute_after <= NOW()`. "
        "Sin el OR NULL, chunks legacy sin execute_after se invisibilizan."
    )


def test_backend_response_dict_exposes_split_keys():
    """El dict de respuesta de `/history-list` DEBE exponer
    `chunk_scheduled_count` Y `chunk_running_now_count` para que el
    frontend pueda leerlos."""
    assert '"chunk_scheduled_count":' in _PLANS_ROUTER, (
        "Key `chunk_scheduled_count` ausente del dict de respuesta. "
        "El SELECT trae el dato pero no se devuelve al cliente."
    )
    assert '"chunk_running_now_count":' in _PLANS_ROUTER, (
        "Key `chunk_running_now_count` ausente del dict de respuesta."
    )


def test_backend_response_dict_preserves_in_flight_count():
    """Por retro-compat con consumidores externos (admin tools,
    monitoring), `chunk_in_flight_count` se preserva (es la SUMA de los
    dos nuevos contadores)."""
    assert '"chunk_in_flight_count":' in _PLANS_ROUTER, (
        "`chunk_in_flight_count` removido del response — backends "
        "legacy y dashboards de monitoring que dependen de él rompen."
    )


def test_frontend_reads_split_counters():
    """`History.jsx` DEBE leer ambos contadores del summary embebido —
    sin esto, las nuevas ramas del `if/else if` quedan en 0 y caen al
    fallback genérico."""
    assert "_plan.chunk_scheduled_count" in _HISTORY_JSX, (
        "Frontend NO lee `chunk_scheduled_count` del summary del plan. "
        "El branch de copy 'se generará cuando llegue su momento' "
        "nunca dispara — el usuario sigue viendo 'generando ahora'."
    )
    assert "_plan.chunk_running_now_count" in _HISTORY_JSX, (
        "Frontend NO lee `chunk_running_now_count` del summary."
    )


def test_frontend_branches_in_priority_order():
    """En el `if/else if` de missingDaysReason, el orden DEBE ser:
      1. _exhaustedCount > 0 (failed irrecoverable) — peor caso.
      2. _puac > 0 (pending_user_action) — necesita user action.
      3. _failedC > 0 (failed transitorio).
      4. _runningNow > 0 ("ahora corriendo" — info).
      5. _scheduled > 0 ("dormido esperando turno" — info).
      6. _inFlight > 0 (fallback legacy del backend pre-fix).
      7. else: nunca generados (info neutro).

    Cambiar el orden alteraría qué copy gana cuando un plan tiene
    múltiples flags simultáneos. El test ancla el orden actual."""
    # Localizamos el if/else if encadenado por sus literales únicos.
    anchors_in_order = [
        "if (_exhaustedCount > 0) {",
        "} else if (_puac > 0) {",
        "} else if (_failedC > 0) {",
        "} else if (_runningNow > 0) {",
        "} else if (_scheduled > 0) {",
        "} else if (_inFlight > 0) {",
    ]
    last_idx = -1
    for anchor in anchors_in_order:
        idx = _HISTORY_JSX.find(anchor, last_idx + 1)
        assert idx > last_idx, (
            f"Anchor `{anchor}` no encontrado en orden esperado. "
            "El if/else if de missingDaysReason cambió de estructura. "
            "Si reordenaste a propósito, actualiza este test."
        )
        last_idx = idx


def test_frontend_legacy_inflight_fallback_does_not_lie():
    """La rama fallback `_inFlight > 0` (cuando el backend NO devuelve
    el split nuevo — deploy lag) DEBE usar copy neutro que NO afirme
    'generando ahora'. Si afirma, perdimos el beneficio del fix.

    Verificamos el `_reason = '...'` literal específicamente — no el
    bloque completo (los comentarios explicativos pueden citar la
    frase prohibida para documentar QUÉ se está evitando)."""
    # Localizamos el bloque del fallback inFlight.
    idx = _HISTORY_JSX.find("} else if (_inFlight > 0) {")
    assert idx > 0
    block = _HISTORY_JSX[idx:idx + 800]
    # Extraer solo la línea del `_reason = '...';` (no comentarios).
    reason_match = re.search(r"_reason\s*=\s*['\"`]([^'\"`]+)['\"`]\s*;", block)
    assert reason_match, (
        "No encontré la asignación `_reason = '...'` en el fallback. "
        "Si reformateaste, anclalo aquí."
    )
    reason_str = reason_match.group(1)
    # Antifrase: "ahora" + "generando" juntos = el mensaje engañoso.
    assert "generando ahora" not in reason_str, (
        "El fallback `_inFlight > 0` (backend legacy) SIGUE afirmando "
        "'generando ahora' en el copy mostrado al usuario — exactamente "
        "el bug que el fix corrige. Cambia a copy neutro como "
        "'cuando llegue su momento'."
    )
    # Frase confirmatoria del nuevo copy.
    assert "cuando llegue su momento" in reason_str, (
        "El fallback no usa el copy neutro esperado en el `_reason` "
        "literal. Debe incluir 'cuando llegue su momento' para "
        "comunicar honestamente."
    )


def test_frontend_running_now_with_mixed_scheduled_mentions_both():
    """Cuando hay running_now > 0 Y scheduled > 0, el copy debe
    mencionar AMBOS estados (no solo "ahora") — algunos corren ya y
    otros esperan turno."""
    idx = _HISTORY_JSX.find("} else if (_runningNow > 0) {")
    assert idx > 0
    block = _HISTORY_JSX[idx:idx + 1200]
    # Debe haber un sub-branch que considere `_scheduled > 0`.
    assert "_scheduled > 0" in block, (
        "La rama running_now NO sub-divide para el caso mixto "
        "(running_now + scheduled simultáneos). El usuario con plan de "
        "10 días donde solo los primeros 3 chunks corren y los demás "
        "duermen vería 'generando ahora' sin mencionar los dormidos."
    )
