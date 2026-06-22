"""[P1-POTE-PRICING · 2026-06-22] Items que se venden en ENVASE (default_unit ∈
pote/lata/paquete/...) con container_weight_g pero SIN market_container poblado se
cobraban A GRANEL (per-lb) en vez de por envase. Ej: Mantequilla de maní ½ lb ≈ RD$29
en vez de 1 pote = RD$117. Fix: apply_smart_market_units cae a default_unit cuando
market_container es NULL (solo unidades de envase, no pesos)."""
import shopping_calculator as sc


def _mi(**kw):
    base = dict(
        name='X', default_unit='pote', container_weight_g=454, market_container=None,
        available_sizes_g=None, price_per_lb=0, price_per_unit=0,
        density_g_per_cup=None, density_g_per_unit=None, category='Despensa',
    )
    base.update(kw)
    return base


def test_pote_null_market_container_se_cobra_por_pote():
    # Mantequilla de maní: pote 454g, price_per_unit 117, price_per_lb 116.89, market_container NULL.
    mi = _mi(name='Mantequilla de maní', default_unit='pote', container_weight_g=454,
             price_per_lb=116.89, price_per_unit=117, density_g_per_cup=258)
    m = sc.apply_smart_market_units('Mantequilla de maní', 0.5, 'lb', 0.5, mi)
    assert (m.get('market_unit') or '').lower() == 'pote', m
    assert float(m.get('market_qty')) == 1.0, m
    # 1 pote × price_per_unit (117), NO ½ lb × price_per_lb (≈58/29).
    assert sc._cost_from_market(m, mi, 116.89, 117) == 117.0


def test_paquete_null_market_container_se_unitariza():
    mi = _mi(name='Harina de trigo', default_unit='paquete', container_weight_g=907,
             price_per_lb=24.5, price_per_unit=49)
    m = sc.apply_smart_market_units('Harina de trigo', 1.0, 'lb', 1.0, mi)
    assert (m.get('market_unit') or '').lower() == 'paquete', m
    assert sc._cost_from_market(m, mi, 24.5, 49) == 49.0


def test_lata_null_market_container_se_unitariza():
    mi = _mi(name='Leche evaporada', default_unit='lata', container_weight_g=377,
             price_per_lb=0, price_per_unit=64)
    m = sc.apply_smart_market_units('Leche evaporada', 0.4, 'lb', 0.4, mi)
    assert (m.get('market_unit') or '').lower() == 'lata', m
    assert sc._cost_from_market(m, mi, 0, 64) == 64.0


def test_item_a_granel_real_NO_se_envasa():
    # default_unit='lb' (carne a granel) NO debe envasarse aunque market_container sea NULL.
    mi = _mi(name='Pechuga de pollo', default_unit='lb', container_weight_g=None,
             price_per_lb=175, price_per_unit=0, category='Carnes')
    m = sc.apply_smart_market_units('Pechuga de pollo', 1.5, 'lb', 1.5, mi)
    assert (m.get('market_unit') or '').lower() in ('lb', 'lbs'), m
