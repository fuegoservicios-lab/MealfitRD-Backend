"""[P0-CHUNK-DAYS-CAP-SPLIT-PARITY · 2026-07-10] El cap de seguridad P0-A2
(`_MAX_DAYS_TO_GENERATE`) debe cubrir el tamaño MÁXIMO legítimo que produce el splitter
de chunks (`split_with_absorb`) — dos features correctas por separado que jamás
funcionaron juntas:

  1. P1-A (constants.split_with_absorb): planes largos usan chunks de base+1=4 días
     (y hasta 2×base=6 con leftover absorbido: 21d → [3,4,4,4,6]).
  2. P0-A2 (graph_orchestrator._enforce_days_to_generate_cap): capaba `_days_to_generate`
     a PLAN_CHUNK_SIZE=3 como defensa contra inyección hostil (9999).

Resultado en vivo (2026-07-10, plan 72c8b965 de un usuario real): el worker pasa
`_days_to_generate=4` (server-side, legítimo), el cap lo recorta a 3
(`🛡️ P0-A2: fuera de rango [1, 3]. Capeando a 3` ×4 en logs), el pipeline genera
3 días, la validación GAP3 espera 4 (`Esperado [19,20,21,22], recibido [19,20,21]`)
→ retry → mismo cap → LOOP hasta dead-letter a los 5 intentos. TODO chunk de 4+ días
de TODO plan quincenal/mensual moría igual (semanas 7-8 del usuario estancadas +
5 pipelines completos quemados por chunk).

tooltip-anchor: P0-CHUNK-DAYS-CAP-SPLIT-PARITY
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))


def test_cap_covers_max_split_chunk_size():
    """Invariante de PARIDAD: ningún chunk que el splitter pueda producir excede el cap.
    Si mañana el splitter crece (absorbe más), este test revienta ANTES de producción."""
    from constants import split_with_absorb, PLAN_CHUNK_SIZE
    from graph_orchestrator import _MAX_DAYS_TO_GENERATE
    worst = 0
    for total in range(PLAN_CHUNK_SIZE, 61):  # 3..60 días (cubre weekly/biweekly/monthly ×2)
        for count in split_with_absorb(total, PLAN_CHUNK_SIZE):
            worst = max(worst, count)
            assert count <= _MAX_DAYS_TO_GENERATE, (
                f"P0-CHUNK-DAYS-CAP-SPLIT-PARITY: split_with_absorb({total}) produce un chunk "
                f"de {count} días > _MAX_DAYS_TO_GENERATE={_MAX_DAYS_TO_GENERATE} — ese chunk "
                f"morirá en loop GAP3 (el cap P0-A2 recorta el _days_to_generate legítimo y la "
                f"validación de numeración espera {count} días). Sube el cap o ajusta el splitter."
            )
    # sanity: el caso documentado 21d → [3,4,4,4,6] existe (worst 6) — el cap no es holgura muerta
    assert worst >= PLAN_CHUNK_SIZE + 1, "el splitter ya no produce chunks >base — revisar P1-A"


def test_enforce_cap_allows_legit_chunk_sizes_and_blocks_hostile():
    from graph_orchestrator import _enforce_days_to_generate_cap, _MAX_DAYS_TO_GENERATE
    # el caso vivo del bug: 4 días legítimos NO deben modificarse
    fd = {"_days_to_generate": 4}
    assert _enforce_days_to_generate_cap(fd) is False, (
        "P0-CHUNK-DAYS-CAP-SPLIT-PARITY regresión: _days_to_generate=4 (chunk legítimo del "
        "splitter P1-A) vuelve a capearse → loop GAP3 en todos los planes quincenales/mensuales."
    )
    assert fd["_days_to_generate"] == 4
    # el caso 6 días (21d → leftover absorbido) también legítimo
    fd6 = {"_days_to_generate": 6}
    assert _enforce_days_to_generate_cap(fd6) is False and fd6["_days_to_generate"] == 6
    # la defensa P0-A2 contra inyección hostil SIGUE viva
    fd_hostile = {"_days_to_generate": 9999}
    assert _enforce_days_to_generate_cap(fd_hostile) is True
    assert fd_hostile["_days_to_generate"] == _MAX_DAYS_TO_GENERATE
    fd_neg = {"_days_to_generate": -1}
    assert _enforce_days_to_generate_cap(fd_neg) is True


def test_cap_derived_from_chunk_size_not_hardcoded():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "_MAX_DAYS_TO_GENERATE = PLAN_CHUNK_SIZE * 2" in src, (
        "el cap debe DERIVARSE de PLAN_CHUNK_SIZE (2×base = máx del splitter con absorb) — "
        "un literal suelto repite la clase de bug (constante hermana que nadie actualiza)"
    )
