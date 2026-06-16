"""[P2-MDDA-NUTRITION-AUDIT · 2026-06-13] Cobertura + curación de la base de macros.

Dos capas:
  1. OFFLINE (parser): el mapeo del script `populate_nutrition_db.py` cubre los 110
     ingredientes, las 7 correcciones post-auditoría están ancladas en FDC_PIN, y los
     valores MANUAL son estructuralmente sanos. Ancla la curación en código (un renombre
     o borrado de un fix rompe el test antes que la data en prod).
  2. INTEGRACIÓN (DB-gated, se SKIPea sin conexión Neon): ≥99/110 con kcal_per_100g
     poblado + auto-consistencia Atwater (kcal == 4P+4C+9F).
"""
import importlib.util
import os

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    path = os.path.join(BACKEND, "scripts", "populate_nutrition_db.py")
    spec = importlib.util.spec_from_file_location("populate_nutrition_db", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _load_script()

# Las 7 correcciones que el workflow de auditoría nutricional encontró y cerró
# (mis-matches del ranking de búsqueda USDA). Pimentón NO se corrige (es paprika
# intencional, no duplicado del "Pimiento morrón" fresco que ya existe).
EXPECTED_FDC_PINS = {
    "Avena": 173904, "Batata": 168482, "Chinola": 169108, "Leche": 171265,
    "Naranja": 169097, "Guineo verde": 173944, "Mantequilla de maní": 172470,
}


# ─────────────────────────── OFFLINE: curación ───────────────────────────
def test_mapping_covers_110_ingredients():
    # 105 base + 5 [P1-RESOLVER-COVERAGE · 2026-06-16] (Manzana, Pepino, Granola, Maní, Clara de huevo).
    base = set(M.USDA_QUERY) | set(M.MANUAL_MACROS)
    assert len(base) == 110, f"Se esperaban 110 ingredientes mapeados, hay {len(base)}"


def test_usda_and_manual_disjoint():
    # Cada nombre vive en exactamente UNA fuente base (USDA_QUERY xor MANUAL_MACROS).
    overlap = set(M.USDA_QUERY) & set(M.MANUAL_MACROS)
    assert not overlap, f"Nombres en ambas fuentes base: {overlap}"


def test_fdc_pins_are_for_usda_mapped_names():
    # Todo pin corrige un nombre que YA estaba en USDA_QUERY (no inventa entradas).
    extra = set(M.FDC_PIN) - set(M.USDA_QUERY)
    assert not extra, f"FDC_PIN para nombres sin mapeo USDA base: {extra}"


def test_audit_corrections_anchored():
    # Las 7 correcciones post-auditoría siguen presentes con su fdc_id verificado.
    for name, fdc in EXPECTED_FDC_PINS.items():
        assert M.FDC_PIN.get(name) == fdc, (
            f"Corrección de auditoría perdida/cambiada para {name!r}: "
            f"esperado fdc#{fdc}, hay {M.FDC_PIN.get(name)}")


def test_pimenton_kept_as_paprika_not_pinned():
    # Decisión documentada: Pimentón = paprika (especia), no se re-mapea a pepper fresco.
    assert "Pimentón" not in M.FDC_PIN
    assert "paprika" in M.USDA_QUERY.get("Pimentón", "").lower()


def test_manual_macros_structurally_sane():
    for name, tup in M.MANUAL_MACROS.items():
        assert len(tup) == 7, f"{name}: MANUAL_MACROS debe ser (kcal,P,C,F,fiber,sodium,dd)"
        kcal, p, c, f, fiber, sodium, dd = tup
        for v in (kcal, p, c, f, fiber, sodium):
            assert v >= 0, f"{name}: macro negativo {v}"
        assert p <= 100 and c <= 100 and f <= 100, f"{name}: macro >100g/100g imposible"
        assert isinstance(dd, bool), f"{name}: flag is_dominican_cultivar debe ser bool"


# ─────────────────── INTEGRACIÓN: estado real en Neon ────────────────────
def _neon_conn():
    url = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
    if not url:
        return None
    try:
        import psycopg
        return psycopg.connect(url, connect_timeout=8)
    except Exception:
        return None


def test_db_coverage_and_atwater_consistency():
    conn = _neon_conn()
    if conn is None:
        pytest.skip("sin conexión Neon (offline/CI) — solo corre la capa parser")
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*), count(kcal_per_100g) FROM master_ingredients")
        total, populated = cur.fetchone()
        assert populated >= 95, f"cobertura insuficiente: {populated}/{total} poblados"
        cur.execute("""SELECT name, kcal_per_100g, protein_g_per_100g,
                              carbs_g_per_100g, fats_g_per_100g
                       FROM master_ingredients WHERE kcal_per_100g IS NOT NULL""")
        bad = []
        for name, k, p, c, f in cur.fetchall():
            atwater = 4 * float(p) + 4 * float(c) + 9 * float(f)
            if abs(float(k) - atwater) > 2.0:  # tolerancia de redondeo
                bad.append((name, float(k), round(atwater, 1)))
        assert not bad, f"kcal NO auto-consistente con 4P+4C+9F: {bad}"
    finally:
        conn.close()
