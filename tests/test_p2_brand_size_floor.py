"""[P2-BRAND-SIZE-FLOOR · 2026-07-06] Piso de tamaño en el picker de duraderos.

Feedback del owner viendo Maní (lista usa Funda 800 gr): "si el plan necesita
800 g, ¿por qué aparecen fundas de 55 g? Deben ser cantidades iguales o mayores
— o como mucho un poco menos, tipo 12 Oz vs 16 Oz". Regla: variantes ≥ 70% del
envase que la lista usa (12/16 Oz = 75% → pasa; Cashitas 55 gr y potes 300 gr
para 800 g → fuera, quedan tras el link '+N de otros tamaños en el catálogo').
La variante YA elegida siempre visible; si el piso vaciara la lista → se enseña
todo (fail-open).
"""
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
BRANDS_JSX = (BACKEND.parent / "frontend" / "src" / "components" / "dashboard"
              / "SupermarketBrands.jsx").read_text(encoding="utf-8")


def test_floor_ratio_is_70_pct():
    assert "P2-BRAND-SIZE-FLOOR" in BRANDS_JSX
    m = re.search(r"const MIN_STABLE_SIZE_RATIO = ([\d.]+)", BRANDS_JSX)
    assert m and abs(float(m.group(1)) - 0.7) < 1e-9, (
        "piso = 70% del envase de la lista (el ejemplo del owner 12/16 Oz = 75% debe pasar)"
    )


def test_floor_applied_in_stable_pool():
    i = BRANDS_JSX.index("const stableSortedVariants")
    body = BRANDS_JSX[i:i + 1400]
    assert "targetG * MIN_STABLE_SIZE_RATIO" in body, "el filtro compara contra el envase de la lista"
    assert "v.id === chosenId" in body, "la variante YA elegida siempre se muestra (para quitarla)"
    assert "if (floored.length) pool = floored;" in body, (
        "fail-open: si el piso vacía la lista, se enseña todo (jamás un picker vacío)"
    )


def test_unknown_size_variants_survive():
    i = BRANDS_JSX.index("const stableSortedVariants")
    body = BRANDS_JSX[i:i + 1400]
    assert "typeof v.size_g !== 'number'" in body, (
        "variante sin tamaño parseable no se descarta (no hay evidencia de que sea chica)"
    )
