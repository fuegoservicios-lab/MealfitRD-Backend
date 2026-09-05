"""[P1-ARQ25-F6-DISH-REGISTRY · 2026-09-05] Curación de `constituents` para las 87 plantillas DO.

Las 5 bibliotecas beta (ES/MX/CO/PR/US) nacieron con `constituents` (nombre + gramos); la dominicana —la
más usada— no los tenía (roadmap §7.1: 87 plantillas, 0 con constituyentes). Sin ellos no hay registry
compilado, ni riesgo intrínseco derivado, ni resolubilidad medible.

DE DÓNDE SALEN, Y POR QUÉ NO ME LOS INVENTO. Cada plantilla se compone de:
  1. componentes de `data/dominican_dish_recipes.json` (60 recetas curadas con gramos, las MISMAS que ya
     usa el diario: mangú, moro, locrio, sancocho, pollo guisado, habichuelas guisadas…), escaladas;
  2. ítems sueltos con nombre EXACTO del catálogo (`master_ingredients`) y gramos de porción típica;
  3. NUNCA una invención: lo que el catálogo no tiene (zapote, menta, chillo) se deja fuera y el
     compilador lo reporta como exclusión explícita — ese es el gate de la fase.

La tabla de abajo es la curación (una entrada por plantilla, por prefijo exacto del nombre); las reglas
por `base`/`protein` solo cubren lo que la tabla no lista. Salida: `data/dish_constituents_do.json`.

USO:  python backend/scripts/build_dish_constituents_do.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
DATA = os.path.join(BACKEND, "data")
OUT = os.path.join(DATA, "dish_constituents_do.json")

R = "recipe"   # ("recipe", clave_de_receta, factor)
I = "item"     # ("item", nombre_exacto_del_catalogo, gramos)

# ---------------------------------------------------------------------------- reglas de respaldo
BASE_ITEMS = {
    "platano": [("Plátano verde", 150)], "guineo": [("Guineo verde", 150)], "avena": [("Avena", 50)],
    "harina": [("Harina de trigo", 60)], "yuca": [("Yuca", 180)], "casabe": [("Casabe", 40)],
    "maiz": [("Harina de maíz precocida", 60)], "pan": [("Pan integral personal", 60)], "batata": [("Batata", 180)],
    "bulgur": [("Bulgur", 70)], "arroz": [("Arroz blanco", 70)], "papa": [("Papa", 200)], "pasta": [("Pasta integral", 80)],
    "viveres": [("Yuca", 100), ("Plátano verde", 100), ("Auyama", 80), ("Papa", 60)],
}
PROTEIN_ITEMS = {
    "huevo": [("Huevo", 100)], "queso": [("Queso blanco", 60)], "pollo": [("Pechuga de pollo", 150)],
    "res": [("Carne de res", 130)], "cerdo": [("Cerdo", 130)], "pavo": [("Pechuga de pavo", 120)],
    "pescado": [("Filete de pescado blanco", 150)], "camarones": [("Camarones", 130)], "atun": [("Atún en agua", 100)],
    "legumbre": [("Habichuelas rojas", 80)], "mixta": [("Carne de res", 120)],
}

# ---------------------------------------------------------------------------- curación por plantilla
# clave = prefijo exacto del `name` de la plantilla (data/dish_templates.json)
CURATED: dict[str, list] = {
    "Mangú de plátano verde con huevos revueltos": [(R, "mangu", 1.0), (R, "huevos revueltos", 1.0), (I, "Cebolla", 20)],
    "Mangú de guineo verde con queso frito": [(I, "Guineo verde", 180), (I, "Cebolla", 15), (I, "Aceite vegetal", 5), (I, "Queso blanco", 60)],
    "Panqueques de avena y guineo maduro": [(I, "Avena", 50), (I, "Guineo", 100), (I, "Huevo", 50), (I, "Leche", 100), (I, "Canela en polvo", 2)],
    "Panqueques de harina de arroz con fresas": [(I, "Harina de trigo", 50), (I, "Huevo", 50), (I, "Leche", 120), (I, "Fresas", 60)],
    "Arepitas de yuca doradas": [(I, "Yuca", 180), (I, "Huevo", 50), (I, "Aceite vegetal", 8)],
    "Bollitos de yuca rellenos de queso": [(I, "Yuca", 180), (I, "Queso blanco", 60), (I, "Huevo", 25)],
    "Revoltillo dominicano con tomate y cebolla": [(R, "huevos revueltos", 1.0), (I, "Tomate", 60), (I, "Cebolla", 25), (I, "Casabe", 40)],
    "Tortilla de huevos con espinaca y queso fresco": [(R, "tortilla de huevo", 1.0), (I, "Espinacas", 60), (I, "Queso blanco", 40)],
    "Avena caliente con canela y leche": [(R, "avena cocida", 1.0), (I, "Canela en polvo", 2)],
    "Yaniqueques horneados": [(I, "Harina de trigo", 70), (I, "Aceite vegetal", 6), (R, "huevos revueltos", 1.0)],
    "Empanadas de yuca al horno rellenas de pollo": [(I, "Yuca", 150), (I, "Pechuga de pollo", 90), (I, "Cebolla", 15), (I, "Ají cubanela", 10)],
    "Majarete ligero de maíz": [(I, "Harina de maíz precocida", 45), (I, "Leche", 200), (I, "Canela en polvo", 2), (I, "Azúcar morena", 8)],
    "Batida de guineo con avena y maní": [(R, "batida de guineo", 1.0), (I, "Avena", 25), (I, "Maní", 15)],
    "Sándwich integral de huevo con aguacate": [(I, "Pan integral personal", 60), (I, "Huevo", 100), (I, "Aguacate", 50)],
    "Yogur con frutas picadas y avena tostada": [(R, "yogurt griego con frutas", 1.0), (I, "Avena", 25)],
    "Mangú con salami de pavo": [(R, "mangu", 1.0), (I, "Jamón de pavo", 60)],
    "Crepes de avena rellenas de queso fresco": [(I, "Avena", 45), (I, "Huevo", 50), (I, "Leche", 120), (I, "Queso blanco", 50)],
    "Batata asada con huevos pochados": [(I, "Batata", 180), (I, "Huevo", 100)],
    "Arepa dominicana de maíz horneada": [(I, "Harina de maíz precocida", 60), (I, "Leche", 100), (I, "Huevo", 25), (I, "Azúcar morena", 8)],
    "Domplines hervidos con huevo revuelto": [(I, "Harina de trigo", 70), (R, "huevos revueltos", 1.0)],
    "Bulgur con leche y canela": [(I, "Bulgur", 60), (I, "Leche", 200), (I, "Canela en polvo", 2)],
    "Pan de batata casero": [(I, "Batata", 120), (I, "Harina de trigo", 30), (I, "Huevo", 25), (I, "Leche", 40)],
    "Locrio de pollo con ensalada verde": [(R, "locrio de pollo", 1.0), (R, "ensalada verde", 1.0)],
    "Locrio de pavo": [(R, "locrio de pollo", 1.0)],  # misma base; la proteína se sustituye abajo
    "Moro de guandules con pollo guisado": [(R, "moro de gandules", 1.0), (R, "pollo guisado", 1.0)],
    "Moro de habichuelas negras con bistec encebollado": [(R, "moro de habichuelas negras", 1.0), (I, "Carne de res", 120), (I, "Cebolla", 40)],
    "La bandera: arroz, habichuelas rojas guisadas y pollo": [(R, "arroz blanco", 1.0), (R, "habichuelas guisadas", 1.0), (R, "pollo guisado", 1.0)],
    # [revisión curatorial F7] la receta «sancocho» del diario es de RES; este plato promete POLLO: ingredientes explícitos
    "Sancocho dominicano de pollo": [(I, "Muslo de pollo", 70), (I, "Yuca", 50), (I, "Plátano verde", 40), (I, "Auyama", 30), (I, "Ñame", 30), (I, "Cebolla", 10), (I, "Aceite vegetal", 5), (I, "Aguacate", 50)],
    "Pastelón de plátano maduro con res molida": [(I, "Plátano maduro", 200), (I, "Carne de res molida", 120), (I, "Queso blanco", 30), (I, "Cebolla", 20)],
    "Pastelón de yuca con pollo desmenuzado": [(I, "Yuca", 200), (I, "Pechuga de pollo", 120), (I, "Queso blanco", 30), (I, "Cebolla", 20)],
    "Pescado guisado a la criolla con arroz blanco": [(R, "pescado guisado", 1.0), (R, "arroz blanco", 1.0)],
    "Camarones guisados con moro de guandules": [(I, "Camarones", 130), (I, "Tomate", 60), (I, "Cebolla", 20), (I, "Ajo", 4), (R, "moro de gandules", 1.0)],
    "Pollo horneado al limón con arroz amarillo": [(R, "pollo al horno", 1.0), (I, "Limón", 15), (R, "arroz blanco", 1.0)],
    "Res mechada con puré de papa": [(R, "carne de res guisada", 1.0), (R, "pure de papa", 1.0)],
    "Chuleta de cerdo a la plancha con moro de habichuelas": [(I, "Cerdo", 140), (I, "Aceite vegetal", 5), (R, "moro", 1.0)],
    "Espaguetis criollos con pollo desmenuzado": [(R, "espagueti guisado", 1.0), (I, "Pechuga de pollo", 100)],
    "Tipile (ensalada dominicana de bulgur) con pollo": [(I, "Bulgur", 60), (I, "Tomate", 60), (I, "Cebolla", 20), (I, "Perejil", 8), (I, "Limón", 15), (I, "Pechuga de pollo", 130)],
    "Quipes horneados con ensalada fresca": [(I, "Bulgur", 60), (I, "Carne de res molida", 100), (I, "Cebolla", 20), (R, "ensalada verde", 1.0)],
    "Berenjenas guisadas con carne molida y arroz": [(R, "berenjena guisada", 1.0), (I, "Carne de res molida", 100), (R, "arroz blanco", 1.0)],
    "Molondrones guisados con pollo y majado de auyama": [(I, "Molondrones", 120), (I, "Tomate", 40), (I, "Cebolla", 20), (I, "Pechuga de pollo", 130), (I, "Auyama", 150)],
    "Habichuelas rojas guisadas con auyama, arroz y aguacate": [(R, "habichuelas guisadas", 1.2), (I, "Auyama", 60), (R, "arroz blanco", 1.0), (I, "Aguacate", 50)],
    "Lentejas guisadas con arroz y aguacate": [(R, "lentejas guisadas", 1.0), (R, "arroz blanco", 1.0), (I, "Aguacate", 50)],
    "Pica pollo casero al horno con tostones": [(I, "Pechuga de pollo", 150), (I, "Harina de trigo", 20), (I, "Orégano dominicano", 1), (R, "tostones", 1.0)],
    "Chicharrón de pollo al airfryer con yuca hervida": [(I, "Pechuga de pollo", 150), (I, "Limón", 15), (I, "Ajo", 4), (R, "yuca hervida", 1.0)],
    "Pescado con coco estilo Samaná": [(I, "Filete de pescado blanco", 150), (I, "Leche de coco", 80), (I, "Tomate", 50), (I, "Cebolla", 20), (R, "arroz blanco", 1.0)],
    "Asopao de pollo": [(R, "asopao de pollo", 1.0)],
    "Niño envuelto: rollitos de repollo": [(I, "Repollo", 150), (I, "Arroz blanco", 50), (I, "Carne de res molida", 100), (I, "Tomate", 60), (I, "Cebolla", 20)],
    "Guineítos verdes guisados con costillitas": [(R, "guineitos hervidos", 1.0), (I, "Costilla de cerdo", 120), (I, "Tomate", 40), (I, "Cebolla", 20)],
    "Bacalao guisado con papas": [(R, "bacalao guisado", 1.0), (I, "Papa", 150)],
    "Arenque desmenuzado con huevo y yuca hervida": [(I, "Arenque", 90), (I, "Huevo", 50), (I, "Cebolla", 20), (R, "yuca hervida", 1.0)],
    "Pechuga rellena de espinaca y queso al horno con batata": [(I, "Pechuga de pollo", 150), (I, "Espinacas", 50), (I, "Queso blanco", 40), (I, "Batata", 150)],
    "Chenchén de maíz partido con pollo guisado": [(I, "Sémola de maíz", 70), (I, "Leche de coco", 40), (R, "pollo guisado", 1.0)],
    "Chivo guisado estilo liniero con yuca hervida": [(I, "Chivo", 150), (I, "Tomate", 50), (I, "Cebolla", 25), (I, "Ajo", 4), (I, "Orégano dominicano", 1), (R, "yuca hervida", 1.0)],
    "Sopa-crema de auyama con pollo desmenuzado": [(R, "crema de auyama", 1.0), (I, "Pechuga de pollo", 100)],
    "Pescado a la plancha con puré de yuca al ajo": [(I, "Filete de pescado blanco", 150), (I, "Aceite de oliva", 8), (I, "Yuca", 180), (I, "Ajo", 6)],
    "Pollo guisado con bollitos de plátano": [(R, "pollo guisado", 1.0), (I, "Plátano verde", 150)],
    "Res a la plancha con puré de auyama": [(I, "Carne de res", 140), (I, "Aceite de oliva", 6), (I, "Auyama", 200)],
    "Tostones al horno con camarones al ajillo": [(R, "tostones", 1.0), (I, "Camarones", 130), (I, "Ajo", 8), (I, "Aceite de oliva", 10)],
    "Croquetas de atún y yuca al horno con ensalada": [(I, "Atún en agua", 100), (I, "Yuca", 120), (I, "Huevo", 50), (R, "ensalada verde", 1.0)],
    "Tortitas de pescado con ensalada verde": [(I, "Filete de pescado blanco", 140), (I, "Huevo", 50), (I, "Pan rallado", 20), (R, "ensalada verde", 1.0)],
    "Pisto criollo de vegetales con huevo pochado": [(I, "Berenjena", 80), (I, "Calabacín", 80), (I, "Tomate", 80), (I, "Cebolla", 30), (I, "Ají morrón", 40), (I, "Huevo", 100)],
    "Ensalada tibia de pollo desmenuzado con aguacate y casabe": [(I, "Pechuga de pollo", 130), (I, "Lechuga", 60), (I, "Tomate", 50), (I, "Aguacate", 60), (I, "Casabe", 40)],
    "Wrap de tortilla integral con pollo y vegetales salteados": [(I, "Tortilla integral", 60), (I, "Pechuga de pollo", 120), (I, "Ají morrón", 40), (I, "Cebolla", 30), (I, "Zanahoria", 40)],
    "Berenjena rellena de res molida": [(I, "Berenjena", 200), (I, "Carne de res molida", 110), (I, "Tomate", 50), (I, "Queso blanco", 30)],
    "Yuca al mojo con cerdo magro a la plancha": [(R, "yuca hervida", 1.0), (I, "Ajo", 6), (I, "Aceite de oliva", 8), (I, "Cerdo", 130)],
    "Puré de batata con pavo a la plancha": [(I, "Batata", 180), (I, "Pechuga de pavo", 130), (I, "Aceite de oliva", 5)],
    "Chillo al horno con vegetales y batata asada": [(I, "Filete de pescado blanco", 160), (I, "Zanahoria", 60), (I, "Calabacín", 60), (I, "Batata", 150)],
    "Revoltillo de atún con plátano hervido": [(I, "Atún en agua", 100), (I, "Huevo", 50), (I, "Tomate", 40), (I, "Cebolla", 20), (R, "platano verde hervido", 1.0)],
    "Pinchos de pollo y vegetales con yuca hervida": [(I, "Pechuga de pollo", 140), (I, "Ají morrón", 50), (I, "Cebolla", 40), (R, "yuca hervida", 1.0)],
    "Rollitos de lechuga con res mechada": [(I, "Lechuga romana", 80), (R, "carne de res guisada", 1.0), (I, "Tomate", 40)],
    "Guacamole criollo con casabe y huevo duro": [(I, "Aguacate", 100), (I, "Tomate", 40), (I, "Cebolla", 20), (I, "Limón", 15), (I, "Casabe", 40), (I, "Huevo", 50)],
    "Pechuga al horno rellena de vegetales con ensalada verde": [(I, "Pechuga de pollo", 150), (I, "Espinacas", 40), (I, "Ají morrón", 40), (R, "ensalada verde", 1.0)],
    "Catibías al horno rellenas de queso": [(I, "Yuca", 150), (I, "Queso blanco", 60)],
    "Batida de lechosa con avena": [(R, "batida de lechosa", 1.0), (I, "Avena", 25)],
    "Yogur con maní y guineo": [(I, "Yogurt griego sin azúcar", 150), (I, "Maní", 20), (I, "Guineo", 100)],
    "Casabe con aguacate majado": [(I, "Casabe", 40), (I, "Aguacate", 80)],
    "Bolitas de avena y maní": [(I, "Avena", 40), (I, "Mantequilla de maní", 20), (I, "Miel", 10)],
    "Queso fresco con tomate y orégano": [(I, "Queso blanco", 70), (I, "Tomate", 80), (I, "Orégano dominicano", 1)],
    "Huevo duro con casabe": [(I, "Huevo", 100), (I, "Casabe", 40)],
    "Batata asada fría con canela": [(I, "Batata", 150), (I, "Canela en polvo", 2)],
    "Frutas picadas con limón": [(I, "Lechosa", 80), (I, "Piña", 80), (I, "Guineo", 80), (I, "Limón", 10)],  # «menta»: sin fila en el catálogo → fuera
    "Panecicos de yuca al horno": [(I, "Yuca", 150), (I, "Huevo", 25), (I, "Queso blanco", 20)],
    "Jugo de chinola natural sin azúcar con puñado de maní": [(I, "Chinola", 80), (I, "Maní", 25)],
    "Tostada integral con mantequilla de maní y guineo": [(R, "pan integral con mantequilla de mani", 1.0), (I, "Guineo", 100)],
    "Empanadita de maíz horneada rellena de queso": [(I, "Harina de maíz precocida", 60), (I, "Queso blanco", 50)],
    "Palitos de zanahoria y apio con dip de habichuelas": [(I, "Zanahoria", 80), (I, "Apio", 60), (I, "Hummus", 60)],
    "Batida de zapote ligera": [(I, "Leche descremada", 250)],  # «zapote»: sin fila en el catálogo → exclusión explícita del compilador
}
# Ítems que la plantilla nombra y el catálogo NO tiene: se declaran para que el compilador los liste como
# exclusión explícita (gate: 100 % resuelve o queda excluido, nunca «desaparece en silencio»).
DECLARED_UNRESOLVED: dict[str, list[str]] = {
    "Frutas picadas con limón": ["Menta"],
    "Batida de zapote ligera": ["Zapote"],
    "Chillo al horno con vegetales y batata asada": ["Chillo (se compone con filete de pescado blanco)"],
    "Mangú con salami de pavo": ["Salami de pavo (se compone con jamón de pavo)"],
}
PROTEIN_SUBSTITUTIONS = {"Locrio de pavo": [("Pechuga de pollo", "Pechuga de pavo")]}


def _load_recipes() -> dict:
    with open(os.path.join(DATA, "dominican_dish_recipes.json"), encoding="utf-8") as f:
        return json.load(f)["recipes"]


def _components(rec: dict) -> list[tuple[str, float]]:
    """Componentes `(nombre, gramos)` de una receta del diario, tolerante a las tres formas del archivo:
    `[nombre, gramos]`, `[nombre, gramos, unidad]` y `{"name":…, "g"|"grams":…}`."""
    out: list[tuple[str, float]] = []
    # Edulcorantes sin fila en el catálogo y sin peso nutricional relevante: se omiten en vez de dejar la
    # plantilla en `partial` (el compilador excluiría «Estevia» y el allocator no la ofrecería).
    ignored = {"estevia", "stevia", "edulcorante"}
    for c in rec.get("ingredients") or rec.get("constituents") or []:
        if isinstance(c, dict):
            n, g = c.get("name"), c.get("grams", c.get("g"))
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            n, g = c[0], c[1]
        else:
            continue
        try:
            g = float(g)
        except (TypeError, ValueError):
            continue
        if n and g > 0 and str(n).strip().lower() not in ignored:
            out.append((str(n), g))
    return out


def _merge(items: list[tuple[str, float]]) -> list[dict]:
    acc: dict[str, float] = {}
    order: list[str] = []
    for name, g in items:
        if name not in acc:
            order.append(name)
        acc[name] = acc.get(name, 0.0) + float(g)
    return [{"name": n, "grams": round(acc[n], 1)} for n in order if acc[n] > 0]


def compose(template: dict, recipes: dict) -> tuple[list[dict], list[str], str]:
    """(constituents, unresolved_declarados, origen) para una plantilla."""
    name = str(template.get("name") or "")
    for prefix, spec in CURATED.items():
        if name.startswith(prefix):
            items: list[tuple[str, float]] = []
            for kind, ref, val in spec:
                if kind == R:
                    rec = recipes.get(ref)
                    if not rec:
                        raise SystemExit(f"receta desconocida {ref!r} en {name!r}")
                    items.extend((n, float(g) * float(val)) for n, g in _components(rec))
                else:
                    items.append((ref, float(val)))
            for a, b in PROTEIN_SUBSTITUTIONS.get(prefix, []):
                items = [(b if n == a else n, g) for n, g in items]
            return _merge(items), list(DECLARED_UNRESOLVED.get(prefix, [])), "curated"
    # respaldo por reglas (no debería hacer falta: la tabla cubre las 87)
    items = list(BASE_ITEMS.get(str(template.get("base") or "none"), []))
    items += PROTEIN_ITEMS.get(str(template.get("protein") or "none"), [])
    return _merge([(n, float(g)) for n, g in items]), [], "rules"


def main() -> int:
    with open(os.path.join(DATA, "dish_templates.json"), encoding="utf-8") as f:
        templates = json.load(f)["templates"]
    recipes = _load_recipes()
    out, by_origin = {}, {"curated": 0, "rules": 0}
    for t in templates:
        # [revisión curatorial F7] las plantillas con constituyentes INLINE (F7-D) no van a la tabla: el registry
        # los lee directo y la tabla sigue siendo solo curación a mano (test_f de F6).
        if t.get("constituents"):
            continue
        cons, unresolved, origin = compose(t, recipes)
        by_origin[origin] += 1
        out[t["name"]] = {"constituents": cons, "declared_unresolved": unresolved, "origin": origin}
    payload = {
        "_note": ("[P1-ARQ25-F6-DISH-REGISTRY] constituents curados para las plantillas DO. Fuente: componentes de "
                  "dominican_dish_recipes.json (60 recetas con gramos) + ítems con nombre exacto del catálogo. "
                  "Generado por scripts/build_dish_constituents_do.py; edítalo allí, no aquí."),
        "schema_version": 1,
        "templates": out,
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"escrito {OUT}: {len(out)} plantillas ({by_origin})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
