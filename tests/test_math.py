from shopping_calculator import apply_smart_market_units

master_mock = {'density_g_per_cup': 0, 'density_g_per_unit': 0}

print('Testing Pechuga de Pavo')
base_q = 1.92857 # for 1.92857 * 7 = 13.5
for mult in [1.0, 3.0, 4.0, 5.0]:
    eff_mult = mult * 2.33333333
    q = base_q * eff_mult
    res = apply_smart_market_units('Pechuga de pavo', 0, 'lbs', q, master_mock)
    print(f'{mult} personas -> eff_mult={eff_mult:.2f} -> Base_q={q:.2f} -> {res["display_string"]}')

print('\nTesting Yuca')
base_y = 1.75 # for 1.75 * 7 = 12.25
for mult in [1.0, 3.0, 4.0, 5.0]:
    eff_mult = mult * 2.33333333
    q = base_y * eff_mult
    res = apply_smart_market_units('Yuca', 0, 'lbs', q, master_mock)
    print(f'{mult} personas -> eff_mult={eff_mult:.2f} -> Base_q={q:.2f} -> {res["display_string"]}')
