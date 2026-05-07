"""[P1-A] Tests para el nuevo algoritmo de `split_with_absorb`.

Cubre:
  1. Casos especiales preservados:
       - 7d → [3, 4]
       - total_days <= base+1 → [total_days]
  2. Planes cortos (n_full < 5) usan la lógica original (sin cambio).
  3. Planes largos con `total_days % 3 == 0` y `n_full >= 5` aplican P1-A:
       - 15d → [3, 4, 4, 4]
       - 18d → [3, 4, 4, 4, 3]
       - 21d → [3, 4, 4, 4, 6]   (leftover absorbido en último chunk target)
       - 24d → [3, 4, 4, 4, 4, 5]
       - 27d → [3, 4, 4, 4, 4, 4, 4]
       - 30d → [3, 4, 4, 4, 4, 4, 4, 3]
  4. Planes largos con `rem != 0` siguen lógica original (no aplica P1-A).
  5. Invariantes globales: sum == total, todos los chunks >= base.

Ejecutar:
    cd backend && python -m pytest tests/test_p1_a_split_with_absorb.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from constants import split_with_absorb, PLAN_CHUNK_SIZE


# ---------------------------------------------------------------------------
# 1. Casos especiales preservados
# ---------------------------------------------------------------------------
def test_7_days_returns_three_then_four():
    """Caso especial preexistente: 7d kickstart 3 + 4 días con aprendizaje."""
    assert split_with_absorb(7, 3) == [3, 4]


@pytest.mark.parametrize("total", [1, 2, 3, 4])
def test_short_total_returns_single_chunk(total):
    """total_days <= base+1 → un solo chunk del total."""
    assert split_with_absorb(total, 3) == [total]


# ---------------------------------------------------------------------------
# 2. Planes cortos (no aplica P1-A)
# ---------------------------------------------------------------------------
def test_8_days_keeps_legacy_distribution():
    """8d: rem=2, n_full=2 → lógica original con rem absorbido como [4,4]."""
    result = split_with_absorb(8, 3)
    assert sum(result) == 8
    assert all(c >= 3 for c in result)
    # Comportamiento legacy: n_base=0, [4]*2.
    assert result == [4, 4]


def test_9_days_keeps_legacy_three_threes():
    """9d: n_full=3 < umbral P1-A → [3, 3, 3] preservado."""
    assert split_with_absorb(9, 3) == [3, 3, 3]


def test_10_days_keeps_legacy_distribution():
    """10d: rem=1, n_full=3 → [3, 3, 4] (lógica original)."""
    assert split_with_absorb(10, 3) == [3, 3, 4]


def test_12_days_keeps_legacy_four_threes():
    """12d: n_full=4 < umbral P1-A → [3, 3, 3, 3] preservado."""
    assert split_with_absorb(12, 3) == [3, 3, 3, 3]


def test_14_days_keeps_legacy_distribution():
    """14d: rem=2, n_full=4 → [3, 3, 4, 4]."""
    assert split_with_absorb(14, 3) == [3, 3, 4, 4]


# ---------------------------------------------------------------------------
# 3. Planes largos donde P1-A aplica (rem == 0 y n_full >= 5)
# ---------------------------------------------------------------------------
def test_15_days_applies_p1a():
    """15d: antes [3,3,3,3,3] (5 chunks); ahora [3,4,4,4] (4 chunks)."""
    assert split_with_absorb(15, 3) == [3, 4, 4, 4]


def test_18_days_applies_p1a_with_trailing_three():
    """18d: rest=15, n_target=3, leftover=3 (>=base) → [3, 4, 4, 4, 3]."""
    assert split_with_absorb(18, 3) == [3, 4, 4, 4, 3]


def test_21_days_absorbs_small_leftover():
    """21d: rest=18, n_target=4, leftover=2 (<base) → absorber en último target.
    Resultado: [3, 4, 4, 4, 6] (último chunk = 4+2).
    """
    assert split_with_absorb(21, 3) == [3, 4, 4, 4, 6]


def test_24_days_absorbs_leftover_one():
    """24d: rest=21, n_target=5, leftover=1 → [3, 4, 4, 4, 4, 5]."""
    assert split_with_absorb(24, 3) == [3, 4, 4, 4, 4, 5]


def test_27_days_clean_split():
    """27d: rest=24, n_target=6, leftover=0 → [3, 4, 4, 4, 4, 4, 4]."""
    assert split_with_absorb(27, 3) == [3, 4, 4, 4, 4, 4, 4]


def test_30_days_applies_p1a():
    """30d: antes [3]*10 (10 chunks); ahora [3, 4, 4, 4, 4, 4, 4, 3] (8 chunks)."""
    assert split_with_absorb(30, 3) == [3, 4, 4, 4, 4, 4, 4, 3]


def test_60_days_applies_p1a():
    """60d: rest=57, n_target=14, leftover=1 → [3, 4*13, 5]."""
    result = split_with_absorb(60, 3)
    assert sum(result) == 60
    assert all(c >= 3 for c in result)
    assert result[0] == 3
    # 14 chunks de 4 + último de 5: total 1 + 13 + 1 = 15 elementos.
    assert len(result) == 15


# ---------------------------------------------------------------------------
# 4. Planes largos con rem != 0 (NO aplica P1-A)
# ---------------------------------------------------------------------------
def test_16_days_uses_legacy_logic_because_rem_nonzero():
    """16d: rem=1, n_full=5. Aunque n_full>=5, rem!=0 → lógica original.
    n_base=4, [3]*4 + [4] = [3, 3, 3, 3, 4]."""
    assert split_with_absorb(16, 3) == [3, 3, 3, 3, 4]


def test_17_days_uses_legacy_logic():
    """17d: rem=2, n_full=5 → [3, 3, 3, 4, 4]."""
    assert split_with_absorb(17, 3) == [3, 3, 3, 4, 4]


# ---------------------------------------------------------------------------
# 5. Invariantes globales
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "total_days",
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
     24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 60, 90],
)
def test_invariant_sum_equals_total(total_days):
    chunks = split_with_absorb(total_days, PLAN_CHUNK_SIZE)
    assert sum(chunks) == total_days


@pytest.mark.parametrize(
    "total_days",
    [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
     24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 60, 90],
)
def test_invariant_all_chunks_at_least_base(total_days):
    """Para total >= base+1, todos los chunks deben ser >= base."""
    chunks = split_with_absorb(total_days, PLAN_CHUNK_SIZE)
    assert all(c >= PLAN_CHUNK_SIZE for c in chunks), (
        f"split_with_absorb({total_days}) violó base mínimo: {chunks}"
    )


def test_first_chunk_is_always_three_for_long_plans():
    """[P1-A] El primer chunk SIEMPRE es 3 días en planes que aplican P1-A —
    es el invariante UX clave: arranque rápido y aprendizaje del primer bloque.
    """
    for total in [15, 18, 21, 24, 27, 30, 60, 90]:
        chunks = split_with_absorb(total, 3)
        assert chunks[0] == 3, f"Primer chunk de {total}d debió ser 3, got {chunks[0]}"


def test_p1a_reduces_chunk_count_vs_legacy():
    """[P1-A] Verificar reducción de número de chunks vs algoritmo viejo
    (calculado a mano). Esto es la motivación del fix: menos llamadas LLM.
    """
    # 15d viejo: 5 chunks. Nuevo: 4 chunks.
    assert len(split_with_absorb(15, 3)) == 4
    # 30d viejo: 10 chunks. Nuevo: 8 chunks.
    assert len(split_with_absorb(30, 3)) == 8
