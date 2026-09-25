# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-197 · 2026-09-24] Los consejos de micronutrientes no le recomiendan a nadie lo que no puede comer.

rd21 (alergia a lácteos y mariscos): el plan salió limpio, pero su panel decía «Calcio bajo — Refuerza con lácteos
(yogur/queso)…» y el consejo de suplemento «primero_alimentos: yogur/queso, sardina con espina…». Y la DIRECTIVA de
micronutrientes que va al prompt del generador le pedía al modelo «Calcio → lácteos (yogur, queso)», «Zinc → mariscos»,
«Vitamina E → nueces/almendras», «Magnesio → maní» — a un usuario cuya alergia va, en el mismo prompt, como prohibición
dura. Dos instrucciones que se contradicen: de ahí salen yogures que el modelo mete solo.

Las tres superficies (nota del panel, alimentos del consejo de suplemento y directiva del prompt) pasan por aquí: cada
opción de alimento se comprueba con el escáner de alérgenos SSOT (`_allergen_pool_item_banned`, con alergias Y
rechazos) y se quita si choca; para el calcio sin lácteos se ofrecen las alternativas de verdad (bebida vegetal
fortificada, tofu con calcio). Sin alergias ni rechazos, el texto sale EXACTAMENTE igual que antes.
Knob `MEALFIT_MICROS_SIN_ALERGENOS` (True). tooltip-anchor: P1-PLAN-LOTE-197-MICROS-SIN-ALERGENOS
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Nota del panel: (prefijo, opciones, sufijo). Sólo los nutrientes cuyas notas nombran alimentos que son alérgenos.
_NOTAS = {
    "calcium_mg": ("Refuerza con ", ["lácteos (yogur/queso)", "vegetales de hoja verde", "sésamo"], "."),
    "vit_d_mcg": ("Una dieta de alimentos enteros rara vez alcanza la vit D: añade ",
                  ["pescado graso (salmón/sardina 1-2x/sem)", "lácteo fortificado"],
                  ", o considera un suplemento de 600-800 UI."),
    "b12_mcg": ("Asegura fuentes animales (", ["huevo", "lácteos", "carne", "pescado"],
                "); si eres vegano, suplemento de B12."),
    "zinc_mg": ("Refuerza con ", ["carnes (res/cerdo)", "mariscos", "huevo", "legumbres",
                                  "nueces/semillas (calabaza/ajonjolí)"], "."),
    "selenium_mcg": ("Refuerza con ", ["pescado/mariscos", "huevo", "carnes", "nuez de Brasil (1-2 al día bastan)"], "."),
    "omega3_g": ("Aumenta ", ["pescado graso (sardina/salmón)", "linaza/chía", "nueces", "aceite de canola"], "."),
    "vit_e_mg": ("Refuerza con ", ["nueces/semillas (almendra, girasol)", "aceites vegetales", "aguacate", "hoja verde"], "."),
    "magnesium_mg": ("Aumenta ", ["vegetales de hoja verde", "legumbres", "nueces/semillas", "granos integrales"],
                     " (clave del patrón DASH)."),
    "vit_a_mcg": ("Aumenta ", ["vegetales naranja/verde oscuro (zanahoria, auyama, batata, espinaca)", "huevo", "lácteos"], "."),
}
# Alternativas que se AÑADEN cuando la fuente principal se cae (y no chocan a su vez).
_EXTRAS = {
    "calcium_mg": ["bebidas vegetales fortificadas con calcio", "tofu cuajado con calcio"],
    "vit_d_mcg": ["bebidas vegetales fortificadas", "yema de huevo"],
}
# Directiva del prompt: nombre del nutriente en la viñeta → clave (para las alternativas).
# [P1-PLAN-LOTE-224 · 2026-09-24] Con nombre para que el frontend la traduzca (`microsCopy.js`) y el test la ate.
SIN_FUENTE_SEGURA = "Pide a tu nutricionista fuentes compatibles con tu alergia; aquí no hay una segura que sugerirte."
_VINETA_CLAVE = {"calcio": "calcium_mg", "zinc": "zinc_mg", "vitamina e": "vit_e_mg", "omega-3": "omega3_g",
                 "magnesio": "magnesium_mg", "hierro": "iron_mg"}


def enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_MICROS_SIN_ALERGENOS", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _vetado(opcion: str, vetos) -> bool:
    try:
        import graph_orchestrator as go
        return bool(go._allergen_pool_item_banned(opcion, list(vetos)))
    except Exception:                                                          # noqa: BLE001
        return False


def _unir(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def _partes(texto: str) -> list:
    """Divide una lista «a, b (c, d), e y f» en sus elementos, sin romper los paréntesis."""
    out, buf, prof = [], "", 0
    for ch in texto:
        if ch == "(":
            prof += 1
        elif ch == ")":
            prof = max(0, prof - 1)
        if ch == "," and prof == 0:
            out.append(buf.strip())
            buf = ""
            continue
        buf += ch
    if buf.strip():
        out.append(buf.strip())
    return [p for p in out if p]


def _filtrar(items: list, vetos, clave=None) -> tuple:
    """(elementos que quedan + alternativas, ¿cambió algo?)."""
    quedan = [i for i in items if not _vetado(i, vetos)]
    if len(quedan) == len(items):
        return items, False
    primeras = {i.split()[0].lower() for i in quedan if i.split()}
    for extra in _EXTRAS.get(clave, ()):
        if extra.split()[0].lower() not in primeras and not _vetado(extra, vetos):   # «tofu» ya está: no «tofu cuajado…»
            quedan.append(extra)
    return quedan, True


def nota_panel(clave: str, original: str, vetos) -> str:
    """La nota del panel para `clave`, sin alimentos que el usuario no puede comer."""
    try:
        if not (enabled() and vetos and clave in _NOTAS):
            return original
        prefijo, opciones, sufijo = _NOTAS[clave]
        quedan, cambio = _filtrar(opciones, vetos, clave)
        if not cambio:
            return original
        if not quedan:
            return SIN_FUENTE_SEGURA
        return prefijo + _unir(quedan) + sufijo
    except Exception:                                                          # noqa: BLE001
        return original


def alimentos_seguros(clave: str, original: str, vetos) -> str:
    """«primero_alimentos» del consejo de suplemento (lista separada por comas), sin lo que el usuario no puede comer."""
    try:
        if not (enabled() and vetos and original):
            return original
        quedan, cambio = _filtrar(_partes(original), vetos, clave)
        return ", ".join(quedan) if cambio else original
    except Exception:                                                          # noqa: BLE001
        return original


def directiva_segura(texto: str, vetos) -> str:
    """La directiva de micronutrientes del prompt, con cada viñeta «• X ≥N → a, b, c.» filtrada y la lista de la línea
    PRIORIDAD «(pollo, pescado, …)» también. Una viñeta sin nada seguro que sugerir desaparece."""
    try:
        if not (enabled() and vetos and texto):
            return texto
        lineas = []
        for linea in texto.split("\n"):
            if linea.startswith("•") and "→" in linea:
                cabeza, resto = linea.split("→", 1)
                if ";" in resto:
                    lista, cola = resto.split(";", 1)
                    cola = ";" + cola
                else:
                    lista, cola = (resto[:-1], ".") if resto.rstrip().endswith(".") else (resto, "")
                    lista = lista.rstrip().rstrip(".")
                nombre = re.sub(r"\s*≥.*$", "", cabeza.lstrip("• ").strip()).lower()
                quedan, cambio = _filtrar(_partes(lista.strip()), vetos, _VINETA_CLAVE.get(nombre))
                if cambio:
                    if not quedan:
                        continue
                    linea = f"{cabeza.rstrip()} → {', '.join(quedan)}{cola}"
            elif linea.startswith("PRIORIDAD") and "(" in linea:
                m = re.search(r"\(([^()]*)\)", linea)
                if m:
                    quedan, cambio = _filtrar(_partes(m.group(1)), vetos)
                    if cambio and quedan:
                        linea = linea[:m.start(1)] + ", ".join(quedan) + linea[m.end(1):]
            lineas.append(linea)
        return "\n".join(lineas)
    except Exception:                                                          # noqa: BLE001
        return texto
