"""[P2-INVENTORY-CONTAINER-MERGE · 2026-07-12] '2 unidades' se fusiona con la fila en 'lata'.

Vivo (owner): "agrega 2 leche evaporada" al chat → la Nevera ya tenía
'Leche evaporada' en unit='lata' (restock, marca Wala) → convert_amount
('unidad'→'lata') = None (los envases discretos no viven en ningún dominio
masa/volumen) → add_or_update insertó una fila DUPLICADA ('unidad', Genérico)
en vez de sumar. Misma clase de fragmentación que 'Plátano'/'Plátano verde'
(esa es por NOMBRE y sigue pendiente).

Regla de dos familias (riesgos distintos):
  - lata/botella/pote/tarro ≈ unidad → 1:1 SIEMPRE (envase de una pieza).
  - paquete/funda/cartón/caja ↔ unidad → SOLO con datos del master
    (container_weight_g + density_g_per_unit); sin datos → None. 1:1 aquí
    sería catastrófico: 1 cartón de huevos = 30 unidades.
tooltip-anchor: P2-INVENTORY-CONTAINER-MERGE
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

from db_inventory import _container_count_equivalence  # noqa: E402


def test_single_item_containers_are_one_to_one():
    assert _container_count_equivalence(2, "unidad", "lata", {}) == 2
    assert _container_count_equivalence(3, "lata", "unidad", {}) == 3
    assert _container_count_equivalence(1, "unidades", "botella", {}) == 1
    assert _container_count_equivalence(2, "pote", "tarro", {}) == 2


def test_multipack_requires_master_data():
    # Sin datos: None → fila separada (fragmentar es más seguro que corromper).
    assert _container_count_equivalence(6, "unidad", "carton", {}) is None
    assert _container_count_equivalence(1, "paquete", "unidad", {}) is None
    # Con datos: ratio real — 6 huevos (50g c/u) sobre cartón de 1500g = 0.2.
    master = {"container_weight_g": 1500, "density_g_per_unit": 50}
    assert _container_count_equivalence(6, "unidad", "carton", master) == 0.2
    assert _container_count_equivalence(1, "carton", "unidad", master) == 30


def test_non_container_units_untouched():
    # masa/volumen siguen siendo territorio de convert_amount, no de este fallback.
    assert _container_count_equivalence(100, "g", "lata", {}) is None
    assert _container_count_equivalence(1, "taza", "unidad", {}) is None


def test_fallback_wired_into_add_or_update():
    """OJO: convert_amount tiene 3 callsites (2 son del sistema de RESERVAS de
    chunks, semántica skip-on-incompatible intencional) — anclar DENTRO de
    add_or_update_inventory_item, no en la primera ocurrencia global."""
    with open(os.path.join(_BACKEND, "db_inventory.py"), encoding="utf-8") as f:
        src = f.read()
    j = src.find("def add_or_update_inventory_item(")
    assert j != -1
    body = src[j:j + 6000]
    i = body.find("converted_qty = convert_amount(quantity, unit, current_unit, master_item)")
    assert i != -1
    win = body[i:i + 800]
    assert "_container_count_equivalence(quantity, unit, current_unit, master_item)" in win, \
        "el fallback debe correr EXACTAMENTE cuando convert_amount devuelve None"
    assert "P2-INVENTORY-CONTAINER-MERGE" in win
