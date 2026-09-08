# -*- coding: utf-8 -*-
"""[P1-REGISTRY-COLOR-PROMETE · 2026-09-08] Un color en el título exige quien lo aporte.

`Pollo horneado al limón con arroz AMARILLO` traía limón, naranja, ajo y `Arroz blanco`. **El limón
no pone amarillo el arroz**: hace falta bija, achiote, cúrcuma o azafrán, y la plantilla no lleva
ninguno. Lo encontró el dueño leyendo la biblioteca, igual que encontró
`Panqueques de harina de ARROZ` con harina de trigo el día anterior.

## Por qué hace falta OTRO detector si ya existe `P1-REGISTRY-TITLE-TRUTH`

Aquél caza la forma «<núcleo> **de** <calificativo>». Ésta es un **adjetivo**, sin «de» por medio,
y es estructuralmente invisible para el otro: no hay par que extraer. Misma clase de defecto —el
título promete lo que la plantilla no da— y distinta gramática. *Un detector cubre una forma, no
una idea.*

## La forma que MEDÍ y descarté

Probé también «al X / a la X / con X» (que «al limón» tenga limón). Sobre las **779** plantillas de
los seis países dio **140 hallazgos y casi todos falsos**: «con arepa» contra `Harina de maíz
precocida`, «con puré» contra `Auyama`, «con ensalada» contra `Lechuga`+`Tomate` — la arepa y el
puré son la FORMA PREPARADA de un ingrediente que sí está. Un detector que dispara sobre el 18 % de
los platos correctos se silencia solo. **No lo construyas otra vez**: la medición está aquí.

El de color, en cambio, dispara sobre **4** títulos en 779 y **despeja 3**: `Merluza en salsa verde`
(lleva perejil), `Arroz rojo con chile poblano` y `Arroz rojo con atún a la mexicana` (llevan
tomate). Queda **1**, el real. Tiene superficie de disparo y paso discriminante — puede fallar, que
es lo que hace informativo su veredicto.

## La trampa que lleva dentro

El color tiene que ir pegado a un **núcleo de plato** (arroz, moro, sopa, crema, salsa, puré…). Sin
esa condición, `Pimienta negra` y `Habichuelas rojas` disparan en cada plantilla que las lleve: el
color ahí es parte del NOMBRE del ingrediente, no una promesa sobre el plato.
"""
import json
import re
import unicodedata
from pathlib import Path

import pytest

_REGISTRY = Path(__file__).resolve().parents[1] / "data" / "registry"

# color -> lo que puede aportarlo. Corta a propósito: cada entrada es una afirmación culinaria.
_COLOR = {
    "amarillo": ("bija", "achiote", "curcuma", "azafran", "onoto", "color", "sazon"),
    "amarilla": ("bija", "achiote", "curcuma", "azafran", "onoto", "color", "sazon"),
    "verde": ("cilantro", "perejil", "espinaca", "albahaca", "aguacate", "guandul", "gandul",
              "brocoli", "guisante", "arveja", "pesto", "chile", "aji", "tomatillo", "platano"),
    "rojo": ("tomate", "achiote", "bija", "pimenton", "remolacha", "salsa", "chile", "aji"),
    "roja": ("tomate", "achiote", "bija", "pimenton", "remolacha", "salsa", "chile", "aji"),
    "negro": ("frijol", "habichuela", "tinta", "cacao", "chocolate", "mole"),
    "negra": ("frijol", "habichuela", "tinta", "cacao", "chocolate", "mole", "pimienta"),
}
_NUCLEOS = ("arroz", "moro", "locrico", "asopao", "sopa", "crema", "salsa", "pure", "guiso")


def _sa(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def color_sin_fuente(nombre: str, constituyentes: list) -> list:
    """Colores que el título pega a un núcleo de plato y que ningún constituyente puede dar."""
    blob = " ".join(_sa(c) for c in constituyentes)
    n = _sa(nombre)
    fuera = []
    for col, fuentes in _COLOR.items():
        if not re.search(rf"\b({'|'.join(_NUCLEOS)})\w*\s+{col}\b", n):
            continue
        if any(f in blob for f in fuentes):
            continue
        fuera.append(col)
    return fuera


def _snapshots():
    return sorted(_REGISTRY.glob("dish_registry_*_v1.json"))


def _plantillas():
    for p in _snapshots():
        d = json.loads(p.read_text(encoding="utf-8"))
        for t in (d.get("templates") or []):
            yield p.stem, t


def test_hay_snapshots_que_auditar():
    """Sin este guard, borrar los snapshots deja el test en verde sin haber mirado nada."""
    assert len(_snapshots()) >= 4, "faltan snapshots del registry: el test no estaría midiendo"
    assert sum(1 for _ in _plantillas()) >= 500


def test_ningun_titulo_promete_un_color_que_nadie_aporta():
    fuera = []
    for snap, t in _plantillas():
        cons = [c.get("name") or c.get("canonical") or "" for c in (t.get("constituents") or [])]
        for col in color_sin_fuente(t.get("name") or "", cons):
            fuera.append(f"{snap}: {t.get('name')} «{col}» — constituyentes {cons}")
    assert not fuera, "títulos que prometen un color sin quien lo dé:\n  " + "\n  ".join(fuera)


def test_el_caso_que_lo_origino_seria_detectado():
    """Un detector que no puede volver a encontrar lo que ya encontró no mide nada. Aquí se fija
    la plantilla EXACTA que lo originó, con sus constituyentes reales de aquel día."""
    assert color_sin_fuente(
        "Pollo horneado al limón con arroz amarillo",
        ["Pechuga de pollo", "Aceite de oliva", "Ajo", "Naranja", "Orégano dominicano",
         "Cebolla", "Sal", "Pimienta negra", "Limón", "Arroz blanco", "Aceite vegetal"],
    ) == ["amarillo"]


@pytest.mark.parametrize("nombre,cons", [
    # los tres que el detector despeja bien: el color SÍ tiene quien lo aporte
    ("Merluza en salsa verde con almejas", ["Merluza", "Perejil", "Ajo", "Almejas"]),
    ("Arroz rojo con chile poblano", ["Arroz blanco", "Tomate", "Chile poblano"]),
    # y los dos que dispararían sin la condición del núcleo: el color es del INGREDIENTE
    ("Pollo guisado con pimienta negra", ["Pechuga de pollo", "Pimienta negra"]),
    ("Ensalada con habichuelas rojas", ["Lechuga", "Habichuelas rojas"]),
])
def test_lo_que_NO_debe_disparar(nombre, cons):
    assert color_sin_fuente(nombre, cons) == []
