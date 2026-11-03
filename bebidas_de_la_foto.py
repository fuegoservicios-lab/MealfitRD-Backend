# backend/bebidas_de_la_foto.py
# [P1-PHOTO-DRINK-IDENTITY · 2026-11-03] SSOT de detección y dudas de bebidas en foto.
#
# POR QUÉ EXISTE:
# Un usuario fotografió una taza con líquido amarillento claro y escribió «Sin azúcar». El modelo de visión
# lo rotuló como «Té herbal» sin levantar dudas (porque todas las opciones rondan ~2 kcal y la regla previa
# solo pedía dudas si «cambian mucho las calorías»). El coach dio por cerrado el desayuno («Anoté tu desayuno...
# Cuando almuerces, cuéntame qué comiste»).
#
# REGLAS FUNDAMENTALES:
# 1. Ninguna foto de líquido en taza/vaso/termo permite saber a simple vista los ingredientes hervidos o mezclados.
#    Debe levantarse SIEMPRE una duda de IDENTIDAD (¿De qué es el té?) y una de ADITIVOS (¿Le añadiste algo?).
# 2. Las opciones de aditivos contemplan miel (+60 kcal), leche (+40 kcal), limón (+2 kcal) y azúcar (+32 kcal).
# 3. Si la aclaración del usuario ya excluye algo («Sin azúcar»), esa opción no se ofrece ni se selecciona.
# 4. Máximo 2 dudas en total para cumplir el contrato del modal y del chat.
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional


def _normalizar_texto(texto: str) -> str:
    """Normaliza quitando acentos y caracteres especiales para matching léxico."""
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    return " ".join(t.split())


_PATRON_BEBIDA = re.compile(
    r"\b(te|infusion|tisana|cafe|jugo|batido|licuado|caldo|consome|chocolate caliente)\b"
    r"|\b(taza de|vaso de|termo de|copa de)\b",
    re.IGNORECASE,
)

_PATRON_ALIMENTOS_SOLIDOS = re.compile(
    r"\b(arroz|mangu|platano|guineo|pollo|carne|filete|res|cerdo|pescado|tostada|pan|"
    r"arepa|huevo|huevos|ensalada|habichuela|habichuelas|pasta|spaghetti|salami|queso frito)\b",
    re.IGNORECASE,
)

_NOMBRES_GENERICOS_BEBIDA = {
    "te", "te herbal", "te de hierbas", "infusion", "infusion herbal",
    "infusion de hierbas", "tisana", "bebida", "bebida caliente", "taza de te",
    "taza de infusion", "jugo", "jugo natural", "vaso de jugo", "batido",
    "licuado", "caldo", "consome",
}


def es_bebida(meal_name: str, description: str = "", items: Optional[List[Dict[str, Any]]] = None) -> bool:
    """Determina si el plato analizado es primordialmente una bebida líquida aislada.

    Guarda contra falsos positivos:
      - 'filete de res' (contiene 'te' como subcadena pero no palabra completa).
      - 'chocolate' de barra (no es chocolate caliente).
      - platos con comida sólida fuerte (ej. 'Mangú con huevo y café' no debe secuestrar las dudas del plato).
    """
    n_norm = _normalizar_texto(meal_name)
    d_norm = _normalizar_texto(description)
    texto_combinado = f"{n_norm} {d_norm}"

    # Verificación de palabra de bebida
    if not _PATRON_BEBIDA.search(texto_combinado):
        return False

    # Si hay items estructurados, verificar si hay comida sólida sustancial
    if items and isinstance(items, list):
        kcal_solidos = 0.0
        for it in items:
            if not isinstance(it, dict):
                continue
            nombre_it = _normalizar_texto(it.get("name", ""))
            # Si el item es un alimento sólido conocido y no es bebida
            if _PATRON_ALIMENTOS_SOLIDOS.search(nombre_it) and not _PATRON_BEBIDA.search(nombre_it):
                kcal_solidos += float(it.get("calories") or 0.0)
        # Si los sólidos suman 50+ kcal, el plato tiene comida real; la bebida es secundaria
        if kcal_solidos >= 50.0:
            return False

    # Verificar si el nombre del plato es predominantemente comida sólida
    if _PATRON_ALIMENTOS_SOLIDOS.search(n_norm) and not any(
        n_norm.startswith(b) for b in ("te ", "cafe ", "jugo ", "batido ", "caldo ", "infusion ")
    ):
        return False

    return True


def tipo_de_bebida(meal_name: str, description: str = "") -> str:
    """Devuelve la categoría de la bebida ('te', 'cafe', 'jugo', 'caldo')."""
    texto = _normalizar_texto(f"{meal_name} {description}")
    if re.search(r"\b(cafe)\b", texto):
        return "cafe"
    if re.search(r"\b(jugo|batido|licuado)\b", texto):
        return "jugo"
    if re.search(r"\b(caldo|consome)\b", texto):
        return "caldo"
    return "te"


def nombre_generico_bebida(meal_name: str) -> bool:
    """¿El nombre del plato carece de especificidad de ingredientes/hierbas?"""
    n_norm = _normalizar_texto(meal_name)
    return n_norm in _NOMBRES_GENERICOS_BEBIDA or n_norm.startswith("taza de") or n_norm.startswith("vaso de")


def crear_duda_identidad(tipo: str, meal_name: str = "") -> Dict[str, Any]:
    """Crea la duda obligatoria sobre la identidad de la bebida."""
    if tipo == "cafe":
        return {
            "sobre": "café",
            "pregunta": "¿Cómo preparaste el café?",
            "opciones": [
                {"texto": "Café negro solo", "supuesta": True, "nombre_plato": "Café negro", "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
                {"texto": "Con leche", "supuesta": False, "nombre_plato": "Café con leche", "ajuste": {"calories": 40, "protein": 2, "carbs": 3, "healthy_fats": 2}},
                {"texto": "Con leche y azúcar", "supuesta": False, "nombre_plato": "Café con leche y azúcar", "ajuste": {"calories": 75, "protein": 2, "carbs": 12, "healthy_fats": 2}},
            ],
        }
    if tipo == "jugo":
        return {
            "sobre": "jugo",
            "pregunta": "¿Cómo preparaste el jugo o batido?",
            "opciones": [
                {"texto": "Natural sin azúcar", "supuesta": True, "nombre_plato": "Jugo natural sin azúcar", "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
                {"texto": "Con azúcar", "supuesta": False, "nombre_plato": "Jugo con azúcar", "ajuste": {"calories": 60, "protein": 0, "carbs": 15, "healthy_fats": 0}},
                {"texto": "Con leche", "supuesta": False, "nombre_plato": "Batido con leche", "ajuste": {"calories": 80, "protein": 4, "carbs": 10, "healthy_fats": 3}},
            ],
        }
    if tipo == "caldo":
        return {
            "sobre": "caldo",
            "pregunta": "¿De qué es el caldo o consomé?",
            "opciones": [
                {"texto": "Caldo de verduras", "supuesta": True, "nombre_plato": "Caldo de verduras", "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
                {"texto": "Caldo de pollo", "supuesta": False, "nombre_plato": "Caldo de pollo", "ajuste": {"calories": 25, "protein": 3, "carbs": 1, "healthy_fats": 1}},
                {"texto": "Caldo de res", "supuesta": False, "nombre_plato": "Caldo de res", "ajuste": {"calories": 35, "protein": 4, "carbs": 1, "healthy_fats": 2}},
            ],
        }
    # Por defecto: té o infusión
    return {
        "sobre": "té",
        "pregunta": "¿De qué es el té?",
        "opciones": [
            {"texto": "Jengibre", "supuesta": True, "nombre_plato": "Té de jengibre", "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
            {"texto": "Manzanilla", "supuesta": False, "nombre_plato": "Té de manzanilla", "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
            {"texto": "Canela y anís", "supuesta": False, "nombre_plato": "Té de canela y anís", "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
            {"texto": "Té verde", "supuesta": False, "nombre_plato": "Té verde", "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
            {"texto": "Limoncillo", "supuesta": False, "nombre_plato": "Té de limoncillo", "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
        ],
    }


def crear_duda_aditivos(tipo: str, aclaracion: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Crea la segunda duda sobre ingredientes agregados (miel, leche, limón, azúcar)."""
    if tipo != "te":
        return None

    acl_norm = _normalizar_texto(aclaracion or "")
    sin_azucar = "sin azucar" in acl_norm or "sin dulce" in acl_norm

    opciones = [
        {"texto": "Solo", "supuesta": True, "nombre_plato": "", "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
        {"texto": "Con limón", "supuesta": False, "nombre_plato": "", "ajuste": {"calories": 2, "protein": 0, "carbs": 1, "healthy_fats": 0}},
        {"texto": "Con miel", "supuesta": False, "nombre_plato": "", "ajuste": {"calories": 60, "protein": 0, "carbs": 16, "healthy_fats": 0}},
        {"texto": "Con leche", "supuesta": False, "nombre_plato": "", "ajuste": {"calories": 40, "protein": 2, "carbs": 3, "healthy_fats": 2}},
    ]
    if not sin_azucar:
        opciones.append(
            {"texto": "Con azúcar", "supuesta": False, "nombre_plato": "", "ajuste": {"calories": 32, "protein": 0, "carbs": 8, "healthy_fats": 0}}
        )

    return {
        "sobre": "té",
        "pregunta": "¿Le añadiste algo al té?",
        "opciones": opciones,
    }


def tiene_duda_identidad(dudas: List[Dict[str, Any]]) -> bool:
    """Verifica si en las dudas existentes ya se preguntó de qué es la bebida."""
    for d in dudas or []:
        preg = _normalizar_texto(d.get("pregunta", ""))
        if any(frase in preg for frase in ("de que es", "como preparaste", "que tipo de", "que te", "que jugo", "que cafe")):
            return True
    return False


def inyectar_dudas_bebida(result: Dict[str, Any], dudas_crudas: List[Dict[str, Any]], aclaracion: Optional[str] = None) -> List[Dict[str, Any]]:
    """Inyecta deterministamente dudas de identidad y aditivos en bebidas si no existen."""
    # Si vino como 'items' pero es una sola bebida servida en taza/vaso, colapsar a 'plato'
    if result.get("photo_kind") == "items":
        desc = result.get("description", "")
        if es_bebida(desc, desc, result.get("items")):
            items_list = result.get("items") or []
            if len(items_list) <= 1:
                result["photo_kind"] = "plato"

    if result.get("photo_kind") != "plato":
        return list(dudas_crudas or [])

    meal_name = str(result.get("meal_name") or "").strip()
    desc = str(result.get("description") or "").strip()
    items = result.get("items") or []

    if not es_bebida(meal_name, desc, items):
        return list(dudas_crudas or [])

    # Si ya traía duda sobre la identidad de la bebida, respetarla
    if tiene_duda_identidad(dudas_crudas):
        return list(dudas_crudas or [])

    tipo = tipo_de_bebida(meal_name, desc)
    duda_id = crear_duda_identidad(tipo, meal_name)
    duda_ad = crear_duda_aditivos(tipo, aclaracion)

    # Si el nombre del plato era genérico, asignamos como base el nombre de la opción supuesta
    if nombre_generico_bebida(meal_name):
        sup_nombre = duda_id["opciones"][0].get("nombre_plato")
        if sup_nombre:
            result["meal_name"] = sup_nombre

    # Ensamblar respetando el tope de 2 dudas
    nuevas_dudas = [duda_id]
    if duda_ad is not None:
        nuevas_dudas.append(duda_ad)

    # Si había alguna otra duda no relacionada con identidad, conservarla como segunda si cabe
    if len(nuevas_dudas) < 2 and dudas_crudas:
        nuevas_dudas.append(dudas_crudas[0])

    return nuevas_dudas[:2]
