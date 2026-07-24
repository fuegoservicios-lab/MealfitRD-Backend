"""[P2-RICE-WATER-RATIO · 2026-07-24] El agua del arroz tiene que cocinar arroz, no hacer sopa.

Defecto en vivo (plan a060108b, revisión de recetas del owner):

    "Locrio de Pescado Blanco con Guisantes Secos"
      ingredientes: ¼ taza de arroz blanco
      paso: "Agrega el arroz, los guisantes, la hoja de laurel, el cilantro, y **2 tazas de
             agua caliente** … tapa y cocina a fuego bajo por 20 minutos."

    8:1. Con esa proporción el locrio no sale. El bowl de pollo del mismo plan pedía 1 taza
    para ¼ taza de arroz integral (4:1). Ambos verificados en la DB.

Decisión: corrección determinista, no gate del revisor. Mandar esto a rechazo cuesta una
regeneración completa —los rechazos son el driver de costo medido del pipeline— para arreglar
una división.

Guardas (lo que el test protege):
  - Solo por encima de 3:1. El asopao y los estilos caldosos son platos legítimos.
  - Si el agua está en la LISTA de ingredientes, no se toca: corregir solo el paso dejaría
    lista y receta contradiciéndose, que es justo la clase de bug que perseguimos.
  - Sin cantidad de arroz reconocible, no inventa nada.
"""
from __future__ import annotations

import graph_orchestrator as g


def _meal(ings, steps, name="Locrio de Pescado Blanco"):
    return [{"meals": [{"name": name, "ingredients": list(ings), "recipe": list(steps)}]}]


def _steps(days):
    return days[0]["meals"][0]["recipe"]


# ---------------------------------------------------------------------------
# 1. El caso reportado
# ---------------------------------------------------------------------------
def test_corrige_el_locrio_8_a_1():
    days = _meal(
        ["½ filete de pescado blanco (101g)", "¼ taza de arroz blanco"],
        ["El Toque de Fuego: Agrega el arroz, los guisantes y 2 tazas de agua caliente; "
         "tapa y cocina a fuego bajo por 20 minutos."],
    )
    n = g._rice_water_ratio_fix(days)
    assert n == 1
    out = _steps(days)[0]
    assert "2 tazas de agua" not in out
    assert "½ taza de agua" in out, out   # ¼ × 2 = ½


def test_corrige_el_bowl_integral_4_a_1():
    days = _meal(
        ["½ pechuga de pollo", "¼ taza de arroz integral"],
        ["Mise en place: enjuaga ¼ taza de arroz y hiérvelo en 1 taza de agua con sal."],
        name="Bowl de Pollo con Arroz Integral",
    )
    assert g._rice_water_ratio_fix(days) == 1
    out = _steps(days)[0]
    assert "1 taza de agua" not in out
    # integral = 2.5:1 → ¼ × 2.5 = 0.625, que el conversor lleva a la grilla de cuartos.
    # Se afirma la BANDA (2:1–3:1), no un glifo: el redondeo es del conversor compartido y
    # fijarlo aquí haría el test frágil ante un cambio legítimo de grilla.
    assert ("½ taza de agua" in out or "¾ taza de agua" in out), out


# ---------------------------------------------------------------------------
# 2. Lo que NO se toca
# ---------------------------------------------------------------------------
def test_no_toca_proporciones_razonables():
    for agua in ("½ taza de agua", "2 tazas de agua"):
        days = _meal(["1 taza de arroz blanco"], [f"Agrega el arroz y {agua}."])
        assert g._rice_water_ratio_fix(days) == 0, agua


def test_no_toca_si_el_agua_es_ingrediente():
    """Si la lista declara el agua, cambiar solo el paso los desincroniza."""
    days = _meal(
        ["¼ taza de arroz blanco", "0.9 taza de agua (para el arroz)"],
        ["Agrega el arroz y 2 tazas de agua caliente."],
    )
    assert g._rice_water_ratio_fix(days) == 0
    assert "2 tazas de agua" in _steps(days)[0]


def test_no_toca_platos_sin_arroz():
    days = _meal(["½ taza de Guisantes secos"],
                 ["Cocina los guisantes en 3 tazas de agua hirviendo por 20 minutos."])
    assert g._rice_water_ratio_fix(days) == 0


def test_no_inventa_sin_cantidad_de_arroz():
    days = _meal(["arroz blanco al gusto"], ["Agrega el arroz y 2 tazas de agua."])
    assert g._rice_water_ratio_fix(days) == 0


# ---------------------------------------------------------------------------
# 3. Contrato
# ---------------------------------------------------------------------------
def test_idempotente():
    days = _meal(["¼ taza de arroz blanco"], ["Agrega el arroz y 2 tazas de agua caliente."])
    assert g._rice_water_ratio_fix(days) == 1
    assert g._rice_water_ratio_fix(days) == 0


def test_knob_permite_rollback(monkeypatch):
    monkeypatch.setattr(g, "RICE_WATER_RATIO_ENABLED", False)
    days = _meal(["¼ taza de arroz blanco"], ["Agrega el arroz y 2 tazas de agua caliente."])
    assert g._rice_water_ratio_fix(days) == 0


def test_tolera_entradas_raras():
    for bad in (None, [], [{}], [{"meals": None}], [{"meals": [{"recipe": "no lista"}]}]):
        assert g._rice_water_ratio_fix(bad) == 0


def test_cableado_al_finalize():
    import pathlib
    src = pathlib.Path(g.__file__).with_suffix(".py").read_text(encoding="utf-8", errors="replace")
    assert "_rice_water_ratio_fix(days)" in src
    assert 'parts.append(f"rice_water=' in src
    assert "[P2-RICE-WATER-RATIO · 2026-07-24]" in src
