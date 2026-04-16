import math

def get_plural_unit(qty, unit):
    if unit.lower() in ['lb', 'lbs', 'libra', 'libras']:
        return "lb" if qty <= 1 else "lbs"
    return unit + "s" if qty > 1 and not unit.endswith("s") else unit

def print_result(weight_in_lbs):
    was_unitarized = False
    
    if weight_in_lbs < 0.23:
        display_qty = "¼ lb"
    else:
        whole = math.floor(weight_in_lbs)
        frac = weight_in_lbs - whole
        fraction_str = ""
        
        if frac < 0.15: fraction_str = ""
        elif frac <= 0.35: fraction_str = "1/4"
        elif frac <= 0.65: fraction_str = "1/2"
        elif frac <= 0.85: fraction_str = "3/4"
        else:
            fraction_str = ""
            whole += 1
            
        if whole > 0 and fraction_str:
            display_qty = f"{whole} {fraction_str} lbs"
        elif whole > 0:
            display_qty = f"{whole} {'lb' if whole == 1 else 'lbs'}"
        else:
            display_qty = f"{fraction_str} lb"
            
    print(f"Weight: {weight_in_lbs} -> Display: {display_qty}")

def run():
    print("Testing meat fraction logic:")
    # Assuming Pechuga evaluates the following weights:
    for base_w in [1.5, 2.0, 2.5]: # Original 1-person weight
        print(f"\n--- Base: {base_w} lbs ---")
        for p in [1, 2, 3, 4, 5, 6]: # Households
            eff_mult = (7.0/3.0) * p # weekly 
            scaled_w = base_w * eff_mult
            print(f"[{p} persons] ", end="")
            print_result(scaled_w)

if __name__ == "__main__":
    run()
