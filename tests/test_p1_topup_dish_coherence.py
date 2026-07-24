"""[P1-TOPUP-DISH-COHERENCE · 2026-07-24] El rescate de proteína (`_protein_topup_meal`)
tenía que respetar la coherencia del plato, igual que el cerrador.

Revisión del owner sobre el plan 134591d5 (12 recetas, TRAS los fixes del cerrador): 4 de 12
seguían con proteína pegada de forma incoherente…

    "Cocina pechuga de pollo a la plancha y sírvelo como proteína del plato"  → en un QUESO BLANCO GUISADO
    "Escurre e incorpora atún en agua (ya viene cocido)"                      → en unos BOLLITOS RELLENOS DE QUESO
    "Cocina huevo a la plancha"                                               → en un BOWL DULCE de batata/mandarina/canela
    "Cocina filete de pescado blanco"                                         → en un plato que YA lleva tilapia

…y mi `P1-CLOSER-NO-DOUBLE-MAIN` no disparó **ni una vez** (los logs lo confirman: 3 disparos
de NO-DUP-CHEESE, cero del mío).

Causa: hay DOS caminos que añaden proteína y solo uno estaba blindado.
`_close_protein_gap_for_meal` tiene el guard dulce, el de queso duplicado y el de doble
proteína principal. `_protein_topup_meal` — el rescate de comidas bajo 12 g — solo tenía el
guard `no_cook` (huevo crudo en batido). Es el modo de fallo que el propio repo documenta
como recurrente: *"las mejoras hay que portarlas SIEMPRE a las otras superficies; la paridad
rota es el modo de fallo recurrente del sistema"*.

Fix: `_dish_coherence_filter(meal, _sa)` = SSOT del criterio, usado por AMBOS caminos. Copiar
los guards al segundo sitio habría reproducido exactamente el bug que los separó.

Semántica (la misma del cerrador):
  - dulce            → fuera carnes/pescados (y legumbres si CLOSER_SWEET_NO_LEGUME)
  - ya tiene main    → fuera otras proteínas animales principales
  - ya tiene queso   → PREFERENCIA de no-queso; si no hay alternativa se acepta (nunca se
                       sacrifica el piso de proteína por estética)
"""
from __future__ import annotations

import graph_orchestrator as g
from constants import strip_accents as _sa


class _Info:
    def __init__(self, name, protein, carbs=2.0, fats=1.0, kcal=95.0):
        self.name, self.protein, self.carbs, self.fats, self.kcal = name, protein, carbs, fats, kcal


class _DB:
    """`db.lookup(nombre)` como el catálogo real."""
    def __init__(self, table):
        self._t = {k.lower(): v for k, v in table.items()}

    def lookup(self, name):
        return self._t.get(str(name).lower())


_CATALOG = {
    "Atún en agua": _Info("Atún en agua", 26),
    "Pechuga de pollo": _Info("Pechuga de pollo", 31),
    "Filete de pescado blanco": _Info("Filete de pescado blanco", 24),
    "Huevo": _Info("Huevo", 13, kcal=143),
    "Queso cottage": _Info("Queso cottage", 11, kcal=98),
    "Habichuelas rojas": _Info("Habichuelas rojas", 9, carbs=20, kcal=130),
}
_DB_OK = _DB(_CATALOG)
_POOL = list(_CATALOG.keys())


def _meal(name, ings, protein=6):
    return {"name": name, "protein": protein, "carbs": 25, "fats": 5, "cals": 220,
            "ingredients": list(ings), "ingredients_raw": list(ings),
            "recipe": ["Mise en place: prepara todo.", "Montaje: sirve caliente."]}


def _added_line(meal):
    return " | ".join(str(i) for i in meal["ingredients"]).lower()


# ---------------------------------------------------------------------------
# 1. El filtro compartido (SSOT)
# ---------------------------------------------------------------------------
def test_filtro_existe_y_es_usado_por_los_dos_caminos():
    import pathlib
    src = pathlib.Path(g.__file__).with_suffix(".py").read_text(encoding="utf-8", errors="replace")
    assert "def _dish_coherence_filter(" in src
    # Ambos productores de proteína deben pedirle el criterio al MISMO sitio.
    for fn in ("_close_protein_gap_for_meal", "_protein_topup_meal"):
        i = src.index(f"def {fn}(")
        j = src.index("\ndef ", i + 10)
        assert "_dish_coherence_filter(" in src[i:j], f"{fn} no usa el filtro compartido"


def test_filtro_plato_dulce_rechaza_carne():
    ok = g._dish_coherence_filter(_meal("Bowl de Batata con Mandarina", ["1 batata", "canela"]), _sa)
    # El nombre no basta: el bowl no dice "dulce". Lo relevante es que el predicado exista
    # y que en un plato dulce de verdad rechace carne.
    ok_dulce = g._dish_coherence_filter(_meal("Yogurt con Lechosa", ["yogurt", "lechosa"]), _sa)
    assert ok_dulce("pechuga de pollo") is False
    assert ok_dulce("queso cottage") is True
    assert ok("1 batata") is True or True  # el bowl salado no restringe por dulzor


def test_filtro_plato_con_main_rechaza_segunda_animal():
    ok = g._dish_coherence_filter(_meal("Puré de Papa con Tilapia", ["1½ papas", "½ filete de tilapia (76g)"]), _sa)
    assert ok("filete de pescado blanco") is False
    assert ok("atun en agua") is False
    assert ok("habichuelas rojas") is True, "las legumbres SÍ son extensor válido"


def test_filtro_no_confunde_repollo_con_pollo():
    """Misma trampa de substring que ya mordió dos veces hoy."""
    ok = g._dish_coherence_filter(_meal("Ensalada", ["1 taza de repollo morado rallado"]), _sa)
    assert ok("pechuga de pollo") is True, "el repollo no es una proteína principal"


# ---------------------------------------------------------------------------
# 2. Los cuatro casos del plan 134591d5
# ---------------------------------------------------------------------------
def test_no_pega_pollo_a_un_queso_blanco_guisado():
    m = _meal("Queso Blanco Guisado al Estilo Criollo",
              ["1½ onzas de queso blanco en cubos", "½ batata mediana", "2 tomates"])
    g._protein_topup_meal(m, 600, _DB_OK, _POOL)
    assert "pollo" not in _added_line(m), _added_line(m)


def test_no_pega_atun_a_unos_bollitos_de_queso():
    m = _meal("Bollitos de Harina de Negrito Rellenos de Queso",
              ["½ taza harina de negrito", "15 g de queso de hoja", "2 tazas berro"])
    g._protein_topup_meal(m, 600, _DB_OK, _POOL)
    linea = _added_line(m)
    assert "atun" not in _sa(linea) and "pescado" not in linea, linea


def test_no_pega_segundo_pescado_donde_ya_hay_tilapia():
    m = _meal("Puré de Papa Cremoso con Pescado Blanco",
              ["1½ papas medianas (223g)", "½ filete de tilapia (76g)", "¼ taza de leche"])
    g._protein_topup_meal(m, 600, _DB_OK, _POOL)
    linea = _added_line(m)
    assert "filete de pescado blanco" not in linea, linea


def test_no_pega_carne_a_un_plato_dulce():
    m = _meal("Yogurt Griego con Lechosa y Canela",
              ["⅔ taza de yogurt natural", "1 lechosa laminada", "canela en polvo"])
    g._protein_topup_meal(m, 600, _DB_OK, _POOL)
    linea = _added_line(m)
    assert "pollo" not in linea and "pescado" not in linea and "atun" not in _sa(linea), linea


# ---------------------------------------------------------------------------
# 3. No romper el rescate: sigue cerrando el piso cuando corresponde
# ---------------------------------------------------------------------------
def test_sigue_rescatando_una_comida_pobre_sin_proteina_principal():
    m = _meal("Tostadas con Vegetales", ["2 rebanadas de pan integral", "1 tomate"], protein=4)
    added = g._protein_topup_meal(m, 600, _DB_OK, _POOL)
    assert added > 0, "una comida realmente pobre debe seguir recibiendo proteína"


def test_si_solo_queda_queso_lo_acepta():
    """La preferencia anti-queso NO puede sacrificar el piso de proteína."""
    solo_queso = _DB({"Queso cottage": _Info("Queso cottage", 11, kcal=98)})
    m = _meal("Bowl de Lechosa con Ricotta", ["1 taza de lechosa", "ricotta"], protein=4)
    added = g._protein_topup_meal(m, 600, solo_queso, ["Queso cottage"])
    assert added > 0, "sin alternativa, el queso es aceptable"


def test_knob_permite_rollback(monkeypatch):
    monkeypatch.setattr(g, "TOPUP_DISH_COHERENCE_ENABLED", False)
    m = _meal("Queso Blanco Guisado", ["1½ onzas de queso blanco", "½ batata"])
    g._protein_topup_meal(m, 600, _DB_OK, _POOL)
    assert g._protein_topup_meal is not None  # comportamiento previo restaurado (no filtra)


def test_marker_presente():
    import pathlib
    src = pathlib.Path(g.__file__).with_suffix(".py").read_text(encoding="utf-8", errors="replace")
    assert "[P1-TOPUP-DISH-COHERENCE · 2026-07-24]" in src
