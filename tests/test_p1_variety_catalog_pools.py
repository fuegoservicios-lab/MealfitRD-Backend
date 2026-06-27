"""[P1-VARIETY-CATALOG-POOLS · 2026-06-27] Las pools de variedad del planner (DOMINICAN_*) se expandieron con
los alimentos verificados del catálogo (202) para que cada renovación explote toda la variedad. CRÍTICO de
SEGURIDAD: el filtro de dieta debe EXCLUIR las proteínas animales nuevas para vegano/vegetariano/pescetariano
(el regex \\bpescado\\b/\\bcarne\\b no matchea 'salmon'/'conejo'/'pulpo' por nombre).
"""
from __future__ import annotations

from constants import (
    DOMINICAN_PROTEINS, DOMINICAN_CARBS, DOMINICAN_VEGGIES_FATS, DOMINICAN_FRUITS,
    _get_fast_filtered_catalogs, strip_accents,
)

NEW_ANIMAL = ["Mero", "Tilapia", "Salmón", "Bacalao", "Sardinas en lata", "Arenque", "Conejo", "Chivo",
              "Pulpo", "Calamar", "Mejillones", "Cangrejo", "Pavo molido", "Jamón de pavo", "Hígado de res",
              "Costilla de cerdo", "Muslo de pollo"]
NEW_FISH_SEAFOOD = ["Mero", "Tilapia", "Salmón", "Bacalao", "Sardinas en lata", "Arenque", "Pulpo",
                    "Calamar", "Mejillones", "Cangrejo"]
NEW_LAND_MEAT = ["Conejo", "Chivo", "Pavo molido", "Jamón de pavo", "Hígado de res", "Costilla de cerdo"]


def _nset(items):
    return {strip_accents(str(x).lower()) for x in items}


def test_pools_were_expanded():
    assert len(DOMINICAN_PROTEINS) >= 45, len(DOMINICAN_PROTEINS)
    for x in ("Mero", "Conejo", "Pulpo", "Salmón", "Cangrejo"):
        assert x in DOMINICAN_PROTEINS, x
    for x in ("Quinoa", "Bulgur", "Cebada", "Mapuey"):
        assert x in DOMINICAN_CARBS, x
    for x in ("Kale", "Espinacas", "Champiñones", "Pistachos", "Espárragos"):
        assert x in DOMINICAN_VEGGIES_FATS, x
    for x in ("Manzana", "Kiwi", "Guayaba", "Uva", "Pera"):
        assert x in DOMINICAN_FRUITS, x


def test_treats_excluded_from_fruit_rotation():
    """Treats/secos y coco NO deben estar en la rotación de fruta fresca."""
    for bad in ("Cereza maraschino", "Durazno en almíbar", "Dátiles", "Pasas", "Ciruela pasa", "Coco", "Tamarindo"):
        assert bad not in DOMINICAN_FRUITS, bad


def test_vegan_excludes_all_animal_proteins():
    prot, _, _, _ = _get_fast_filtered_catalogs((), (), "vegano")
    pset = _nset(prot)
    for animal in NEW_ANIMAL + ["Pollo", "Cerdo", "Res", "Pescado", "Atún", "Camarones", "Huevos"]:
        assert strip_accents(animal.lower()) not in pset, f"vegano dejó pasar {animal}: {prot}"
    # tampoco quesos/yogur (lácteos) en vegano
    assert not any("queso" in p or "yogur" in p for p in pset), prot
    # pero SÍ quedan proteínas vegetales (pool no vacío)
    assert any(x in prot for x in ("Lentejas", "Garbanzos", "Edamame", "Soya texturizada", "Gandules")), prot


def test_vegetarian_excludes_meat_and_seafood_keeps_egg_dairy():
    prot, _, _, _ = _get_fast_filtered_catalogs((), (), "vegetariano")
    pset = _nset(prot)
    for animal in NEW_FISH_SEAFOOD + NEW_LAND_MEAT + ["Pollo", "Cerdo", "Res", "Pescado", "Atún", "Camarones"]:
        assert strip_accents(animal.lower()) not in pset, f"vegetariano dejó pasar {animal}"
    # vegetariano SÍ permite huevo + lácteos
    assert "huevos" in pset
    assert any("queso" in p for p in pset)


def test_pescatarian_excludes_land_meat_keeps_fish_seafood():
    prot, _, _, _ = _get_fast_filtered_catalogs((), (), "pescetariano")
    pset = _nset(prot)
    for meat in NEW_LAND_MEAT + ["Pollo", "Cerdo", "Res"]:
        assert strip_accents(meat.lower()) not in pset, f"pescetariano dejó pasar carne {meat}"
    # pescado + mariscos PERMITIDOS para pescetariano
    assert any(strip_accents(f.lower()) in pset for f in NEW_FISH_SEAFOOD), f"pescetariano debería permitir pescado/marisco: {prot}"


def test_omnivore_keeps_everything():
    prot, carb, veg, fruit = _get_fast_filtered_catalogs((), (), "")
    assert "Mero" in prot and "Conejo" in prot and "Pulpo" in prot
    assert "Quinoa" in carb and "Kale" in veg and "Manzana" in fruit


def test_marker_anchor():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "constants.py").read_text(encoding="utf-8")
    assert "P1-VARIETY-CATALOG-POOLS" in src
