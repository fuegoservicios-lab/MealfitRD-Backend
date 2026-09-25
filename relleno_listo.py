# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-238/239 · 2026-09-25 · fuera del grafo en P1-PLAN-LOTE-240] Lo que un relleno determinista añade o
cambia respeta el país del mercado y las restricciones del formulario.

238 — El carbohidrato listo del relleno de ganar músculo con «Nada» de tiempo depende del MERCADO (I16): casabe en
RD/PR, tortilla de maíz en México, pan integral en el resto (España, EE. UU., Colombia) — el casabe no se vende en un
súper de Madrid. Y el relleno (este y el de arroz/batata) no miraba alergias, rechazos ni dieta: «60 g de casabe» a un
alérgico a la yuca, batata a quien la rechazó. Se elige el primero de la lista del país que no viole nada; si ninguno,
la comida no se rellena. tooltip-anchor: P1-PLAN-LOTE-238-RELLENO-POR-PAIS

239 — El arroz de la cena se cambia por un tubérculo (batata/yuca/casabe/ñame/auyama) que no miraba nada, y como
también RENOMBRA el plato («Pollo con yuca») la última palabra del escudo (lote 233) no puede quitarlo: ya es la
identidad del plato. Con «Nada» de tiempo solo cabe el casabe (listo). Si ninguno sirve, no se cambia (el arroz de noche
es una advertencia de horario, no una restricción del usuario). tooltip-anchor: P1-PLAN-LOTE-239-ARROZ-DE-NOCHE
"""
from __future__ import annotations

_GM_READY_FOODS = {
    "casabe": ("casabe", 3.47, 0.853, 0.013, "casabe",
               "🫓 Acompaña con el casabe de tus ingredientes: está listo para comer, sin cocción."),
    "tortilla de maíz": ("tortilla de maíz", 2.18, 0.446, 0.057, "tortilla de ma",
                         "🫓 Calienta la tortilla de maíz 1 min en el comal o la sartén y sírvela de acompañante."),
    "pan integral": ("pan integral", 2.52, 0.427, 0.125, "pan integral",
                     "🍞 Acompaña con el pan integral de tus ingredientes: está listo, sin cocción."),
}
_GM_READY_ORDER = {"DO": ("casabe", "pan integral", "tortilla de maíz"),
                   "PR": ("casabe", "pan integral", "tortilla de maíz"),
                   "MX": ("tortilla de maíz", "pan integral", "casabe")}
_GM_READY_ORDER_DEFAULT = ("pan integral", "tortilla de maíz", "casabe")


def _gm_line_violates(line: str, form_data) -> bool:
    """¿Viola la línea alguna alergia (con texto libre), rechazo o dieta declarados?"""
    try:
        import graph_orchestrator as _go
        fd = _go.profile_with_free_text(form_data) if isinstance(form_data, dict) else {}
        mini = {"days": [{"meals": [{"name": "", "ingredients": [line]}]}]}
        _alg = [a for a in (fd.get("allergies") or []) if str(a).strip().lower() not in _go._SENTINEL_NONE_VALUES]
        if _alg and _go._scan_allergen_violations(mini, _alg):
            return True
        if _go._scan_dislike_violations(mini, fd):
            return True
        return bool(_go.DIET_HARD_GUARD and _go._scan_diet_violations(mini, fd.get("dietType")))
    except Exception:
        return False


def _gm_ready_carb_for(form_data):
    """(food, kcal/g, carb/g, prot/g, clave, nota) del carbohidrato listo del país, o None."""
    try:
        from constants import country_for_form_data as _cff
        _pais = str(_cff(form_data or {}) or "DO").upper()
    except Exception:
        _pais = "DO"
    for _k in _GM_READY_ORDER.get(_pais, _GM_READY_ORDER_DEFAULT):
        _t = _GM_READY_FOODS[_k]
        if not _gm_line_violates(f"50 g de {_t[0]}", form_data):
            return _t
    return None


def _night_rice_sub_for(di: int, form_data) -> "str | None":
    """Tubérculo que sustituye el arroz de la cena, sin violar el formulario (o None: no se cambia)."""
    import graph_orchestrator as _go
    rot = list(_go._NIGHT_RICE_SUB_ROTATION)
    try:
        base = __import__("nevera_exigida").preferir(rot, di)
    except Exception:
        base = rot[int(di) % len(rot)]
    i0 = rot.index(base) if base in rot else 0
    cands = [rot[(i0 + k) % len(rot)] for k in range(len(rot))]
    if isinstance(form_data, dict) and str(form_data.get("cookingTime") or "").strip().lower() == "none":
        cands = [c for c in cands if c == "Casabe"]
    for c in cands:
        if not isinstance(form_data, dict) or not _gm_line_violates(f"150 g de {c.lower()}", form_data):
            return c
    return None
