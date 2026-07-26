"""[P1-NIGHT-RICE-MIN-G · 2026-07-26] El "arroz de noche" se evadía por gramaje.

## Lo medido (60 planes vivos, 184 comidas)

De 41 ingredientes que contienen "arroz", **38 son de almuerzo** — correcto, el arroz es EL
staple dominicano del plato fuerte. Los otros tres:

    [Cena]     "Wrap Fresco de Atún con Lechuga, Tomate y Pepino"  → 35 g de arroz blanco crudo
    [Cena]     "Tortilla de Ñame Rallado con Ensalada…"            → 25 g de arroz blanco cocido
    [Merienda] "Pera Asada con Mantequilla de Maní y Canela…"      → 20 g de arroz blanco crudo

Las dos cenas evadían el gate **por partida doble**: el detector de slot es name-only por
diseño (anti-falso-positivo: "Panqueques de harina de arroz" no debe flagear) y ninguno de los
dos nombres dice "arroz"; y el pase ingredient-driven que existe justo para eso las dejaba
pasar porque caían bajo `NIGHT_RICE_INGREDIENT_MIN_G=60`.

35 g de arroz crudo son ≈105 g cocidos — dos tercios de taza. Eso no es la "guarnición
tangencial" que el umbral pretendía proteger; es una porción. Lo tangencial ya lo cubren los
excludes SSOT (`harina/leche/vinagre de arroz`, `toque de arroz`, `crocante de arroz`), que es
la defensa correcta para ese caso. El umbral, en la práctica, solo dejaba huecos.

## Lo que NO se tocó

El arroz del ALMUERZO (38 de los 41) no entra aquí: el pase es solo de cena. Y una cena cuyo
NOMBRE sí dice arroz —"Sopa de pollo con arroz"— la sigue cazando el detector name-based con
su degradación soft, así que bajar el umbral no cambia nada para ella.

## Lo que este fix NO resuelve

La merienda. No hay regla de producto que prohíba el arroz en merienda, y no puede haberla a
lo bruto: **"Arroz con leche" es una merienda dominicana legítima**, protegida explícitamente
en `SLOT_INAPPROPRIATE_FOODS`. El caso de la pera asada se ataca por el prompt (regla nueva
"no dupliques el carbohidrato"), que es un lever blando. 1 caso de 46 meriendas no justifica
inventar un blocklist que rompería el arroz con leche.

tooltip-anchor: P1-NIGHT-RICE-MIN-G
"""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(g.__file__).resolve().parent


class _DB:
    def grams_from_ingredient_string(self, s):
        m = re.search(r"(\d+)\s*g", s)
        return float(m.group(1)) if m else 150.0


def _wire(monkeypatch):
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)
    monkeypatch.setattr(g, "NIGHT_RICE_AUTOFIX_ENABLED", True)
    monkeypatch.setattr(g, "NIGHT_RICE_INGREDIENT_PASS", True)


def _cena(name, ings, day=0):
    return [{"day": day, "meals": [{"meal": "Cena", "name": name, "ingredients": ings,
                                    "recipe": ["Cocina el arroz 15 min."]}]}]


def _sin_arroz(meal):
    return not any(re.search(r"\barroz\b", str(i).lower()) for i in meal["ingredients"])


# ───────────── 1. el umbral ─────────────

def test_el_default_bajo_a_20():
    assert g.NIGHT_RICE_INGREDIENT_MIN_G == 20


def test_el_clamp_sigue_vivo():
    """La constante se lee con clamp [20, 200]; el suelo impide desactivarlo poniendo 0."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "max(20, min(200, _env_int(\"MEALFIT_NIGHT_RICE_INGREDIENT_MIN_G\"" in src


# ───────────── 2. los DOS casos vivos, ahora sí ─────────────

def test_el_wrap_de_atun_con_35g(monkeypatch):
    """Caso real. 2 tortillas integrales YA eran el carbo del plato; el arroz iba encima."""
    _wire(monkeypatch)
    days = _cena("Wrap Fresco de Atún con Lechuga, Tomate y Pepino",
                 ["115 g de atún en agua escurrido", "2 tortillas integrales",
                  "35 g de arroz blanco crudo"])
    assert g._night_rice_autofix(days, db=_DB()) == 1
    assert _sin_arroz(days[0]["meals"][0])


def test_la_tortilla_de_name_con_25g(monkeypatch):
    """Caso real. El ñame YA era el carbo."""
    _wire(monkeypatch)
    days = _cena("Tortilla de Ñame Rallado con Ensalada de Lechuga y Tomate",
                 ["¾ taza de ñame rallado", "1 huevo", "25g de arroz blanco cocido"])
    assert g._night_rice_autofix(days, db=_DB()) == 1
    assert _sin_arroz(days[0]["meals"][0])


def test_el_nombre_no_se_toca(monkeypatch):
    """Contrato heredado del pase: si el nombre no delata arroz, no se renombra."""
    _wire(monkeypatch)
    days = _cena("Wrap Fresco de Atún", ["35 g de arroz blanco crudo"])
    g._night_rice_autofix(days, db=_DB())
    assert days[0]["meals"][0]["name"] == "Wrap Fresco de Atún"


def test_idempotente(monkeypatch):
    _wire(monkeypatch)
    days = _cena("Bowl de pollo", ["120 g de Pollo", "35 g de arroz blanco"])
    assert g._night_rice_autofix(days, db=_DB()) == 1
    assert g._night_rice_autofix(days, db=_DB()) == 0


# ───────────── 3. lo que sigue intacto ─────────────

def test_bajo_el_suelo_sigue_sin_tocarse(monkeypatch):
    """Con el suelo en 20, una traza de 15 g sigue exenta — el umbral no desaparece."""
    _wire(monkeypatch)
    days = _cena("Ensalada de pollo", ["120 g de Pollo", "15 g de arroz blanco"])
    assert g._night_rice_autofix(days, db=_DB()) == 0


def test_los_excludes_ssot_mandan_sobre_el_gramaje(monkeypatch):
    """La defensa contra lo tangencial son los excludes, no el umbral. 200 g de harina de
    arroz en una cena NO son arroz de noche."""
    _wire(monkeypatch)
    days = _cena("Panqueques de cena", ["200 g de harina de arroz"])
    assert g._night_rice_autofix(days, db=_DB()) == 0


def test_el_almuerzo_no_entra(monkeypatch):
    """38 de los 41 usos de arroz del corpus son de almuerzo y son CORRECTOS. Si este pase
    tocara el almuerzo, le quitaría al plan su plato dominicano por excelencia."""
    _wire(monkeypatch)
    days = [{"day": 0, "meals": [{"meal": "Almuerzo", "name": "Pollo guisado",
                                  "ingredients": ["1 taza de arroz blanco (150g)"],
                                  "recipe": ["Sirve con arroz."]}]}]
    assert g._night_rice_autofix(days, db=_DB()) == 0


def test_el_arroz_con_leche_sigue_siendo_merienda_legitima():
    """Ancla la razón por la que NO se añadió una regla de arroz en merienda."""
    from constants import slot_violations_for_meal_name
    assert slot_violations_for_meal_name("Arroz con Leche", "merienda") == []


# ───────────── 4. el lever upstream ─────────────

def test_el_prompt_prohibe_duplicar_el_carbohidrato():
    """El gate solo cubre la CENA. El caso de la merienda (pera asada + arroz) solo se puede
    atacar upstream, y sin el prompt el modelo lo sigue proponiendo."""
    import prompts.day_generator as dg
    src = Path(dg.__file__).resolve().read_text(encoding="utf-8")
    assert "NO DUPLIQUES EL CARBOHIDRATO" in src
    i = src.index("NO DUPLIQUES EL CARBOHIDRATO")
    bloque = src[i:i + 900]
    # Los tres casos REALES citados en la regla, para que nadie la reduzca a una frase vaga.
    for pista in ("tortilla", "ñame", "Pera Asada", "arroz", "P1-NIGHT-RICE-MIN-G"):
        assert pista in bloque, pista
    assert "AGRANDA la base" in bloque, "debe decir qué hacer, no solo qué no hacer"
