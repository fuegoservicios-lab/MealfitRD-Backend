# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1079 · 2026-11-18] Una marca mal escrita se reconoce: «yogurt sissys» es Siggi's, no «un yogur».

Caso (cuenta de prueba, 17-nov, 15:55): «yogurt sissys» se anotó como un yogur genérico (150 kcal, 5 g de proteína);
el usuario tuvo que pedir «elimina eso» y escribir «yogurt siggi's» (140 kcal, 15 g). En yogures, barras y batidos la
marca cambia la proteína al triple: estimarlo genérico es el error caro.

Determinista y estrecho: solo con una palabra de PRODUCTO en el mensaje (yogur, barra, batido, proteína…), y solo si
una palabra está a ≤2 ediciones (≤1 si es corta) de una marca conocida, con la misma inicial, sin serlo exactamente.
Normaliza antes «y»→«i», acentos y apóstrofos («sissys» → «sissis», a 2 de «siggis»). El prompt recibe la marca con la
orden de anotarla así y decirlo en una frase, para que corrija si no era.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

# normalizada → como se escribe
MARCAS = {
    "siggis": "Siggi's", "chobani": "Chobani", "oikos": "Oikos", "ioplait": "Yoplait", "activia": "Activia",
    "danone": "Danone", "dannon": "Dannon", "fage": "Fage", "alpina": "Alpina", "iopro": "YoPro",
    "fairlife": "Fairlife", "quest": "Quest", "premier": "Premier Protein", "isopure": "Isopure",
    "dimatize": "Dymatize", "muscletech": "MuscleTech",
}
_PRODUCTO = re.compile(r"\b(?:iogur\w*|ioghurt\w*|barr(?:a|ita)s?|batidos?|shakes?|proteinas?|protein|whey|skir|leche)\b")
# palabras corrientes que caen cerca de alguna marca y nunca lo son
_COMUNES = {"queso", "quest", "fresa", "fresas", "griego", "natural", "premio", "primer", "primera", "alpino", "danos",
            "dame", "fase", "facil", "oigo", "otros", "activa", "activo", "darle", "dannos", "pequena", "pequeno"}


def _norm(t) -> str:
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"['’`´]", "", t)
    return t.replace("y", "i")


def _distancia(a: str, b: str, tope: int) -> int:
    if abs(len(a) - len(b)) > tope:
        return tope + 1
    previa = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        fila = [i]
        for j, cb in enumerate(b, 1):
            fila.append(min(previa[j] + 1, fila[j - 1] + 1, previa[j - 1] + (ca != cb)))
        if min(fila) > tope:
            return tope + 1
        previa = fila
    return previa[-1]


def detectar(texto: str) -> Optional[tuple]:
    """`(palabra tal cual la escribió, marca)` o None."""
    crudo = str(texto or "")
    if not _PRODUCTO.search(_norm(crudo)):
        return None
    pares = [(o, _norm(o)) for o in re.findall(r"[^\W\d_]+(?:['’`´][^\W\d_]+)?", crudo)]
    if any(n in MARCAS for _o, n in pares):
        return None                          # ya la nombró bien: nada que corregir
    for original, w in pares:
        if len(w) < 4 or w in _COMUNES or _PRODUCTO.fullmatch(w):
            continue
        tope = 2 if len(w) >= 5 else 1
        for clave, marca in MARCAS.items():
            if clave[0] == w[0] and abs(len(clave) - len(w)) <= tope and _distancia(w, clave, tope) <= tope:
                return original, marca
    return None


def bloque_para_chat(texto: str) -> str:
    """Texto «Para el asistente» o "". Nunca lanza."""
    try:
        r = detectar(texto)
        if not r:
            return ""
        palabra, marca = r
        return (f"\n\n🏷️ MARCA [P1-PLAN-LOTE-1079]: «{palabra}» casi seguro es «{marca}» escrito de otra forma. "
                f"Regístralo como «{marca}» con las cifras de ESA marca (no como un producto genérico) y dilo en tu "
                f"frase para que corrija si no era («lo anoté como {marca}; si era otra marca, dime»).\n")
    except Exception:  # noqa: BLE001
        return ""
