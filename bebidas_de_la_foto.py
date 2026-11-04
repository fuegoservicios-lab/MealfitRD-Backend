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


# Palabras de bebida "fuertes": identifican una bebida en el nombre o en la descripción.
_PATRON_BEBIDA = re.compile(
    r"\b(infusion|tisana|cafe|jugo|batido|licuado|caldo|consome|chocolate caliente)\b"
    r"|\bte\b",
    re.IGNORECASE,
)
# En la DESCRIPCIÓN (texto libre del modelo) el pronombre «te» es ruido: solo cuenta «te» como bebida si va
# en una frase de bebida («taza de te», «te verde», «te de jengibre»...).
_PATRON_BEBIDA_DESC = re.compile(
    r"\b(infusion|tisana|cafe|jugo|batido|licuado|caldo|consome|chocolate caliente)\b"
    r"|\b(?:taza|vaso|termo|jarra)\s+de\s+te\b|\bte\s+(?:verde|negro|herbal|de|con|helado|frio|caliente)\b",
    re.IGNORECASE,
)

# Cualquier alimento sólido (en nombre o descripción) descarta la foto como «solo bebida»: un plato completo con un
# vaso de jugo o una taza de café al lado se analiza con sus dudas normales, no con la duda de identidad de la bebida.
_SOLIDOS = (
    "arroz", "mangu", "platano", "guineo", "pollo", "carne", "filete", "res", "cerdo", "pescado", "tostada", "pan",
    "arepa", "huevo", "ensalada", "habichuela", "pasta", "spaghetti", "salami", "queso", "jamon", "granola",
    "avena", "yogurt", "yogur", "galleta", "sandwich", "sopa", "fideo", "vegetal", "fruta", "cereal", "tostone",
    "yuca", "papa", "atun", "pavo", "aguacate", "tortilla", "panqueque", "pancake", "wrap", "hamburguesa", "pizza",
    "bandera", "mondongo", "sancocho", "locrio", "moro", "chofan", "pastelon", "empanada",
)
_PATRON_ALIMENTOS_SOLIDOS = re.compile(r"\b(?:" + "|".join(_SOLIDOS) + r")(?:s|es)?\b", re.IGNORECASE)

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

    # La bebida debe nombrarse en el rótulo, o (rótulo vacío/genérico) en la descripción con una frase de bebida.
    if not (_PATRON_BEBIDA.search(n_norm) or _PATRON_BEBIDA_DESC.search(d_norm)):
        return False

    # Cualquier alimento sólido en el rótulo o en la descripción → es un plato con bebida al lado, no «solo bebida».
    if _PATRON_ALIMENTOS_SOLIDOS.search(n_norm) or _PATRON_ALIMENTOS_SOLIDOS.search(d_norm):
        return False

    # Con componentes estructurados: cualquier componente sólido con calorías reales descarta.
    if items and isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            nombre_it = _normalizar_texto(it.get("name", ""))
            if _PATRON_ALIMENTOS_SOLIDOS.search(nombre_it) and not _PATRON_BEBIDA.search(nombre_it):
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


_FRASES_IDENTIDAD = (
    "de que es", "como preparaste", "que tipo de", "que te", "que jugo", "que cafe", "que caldo",
    "que bebida", "cual bebida", "que tienes", "que hay en la taza", "que hay en el vaso",
    "que tomas", "que estas tomando", "que liquido", "es cafe o", "es te o", "identifica",
    "que contiene la taza", "que contiene el vaso", "que es la bebida", "de que es la bebida",
    "que bebida es", "que bebida tienes",
)

_FAMILIAS_BEBIDA = (
    "cafe", "te", "infusion", "chocolate", "jugo", "caldo", "agua", "batida", "batido", "tisana"
)


def es_duda_multibebida(duda: Dict[str, Any]) -> bool:
    """[BB-030] Detecta si una duda ofrece opciones entre categorías distintas de bebidas (ej. café vs té vs chocolate)."""
    if not isinstance(duda, dict):
        return False
    opciones = duda.get("opciones") or []
    familias_encontradas = set()
    for op in opciones:
        if not isinstance(op, dict):
            continue
        t_norm = _normalizar_texto(op.get("texto", "") + " " + op.get("nombre_plato", ""))
        for fam in _FAMILIAS_BEBIDA:
            if re.search(r"\b" + re.escape(fam) + r"\b", t_norm):
                familias_encontradas.add(fam)
    return len(familias_encontradas) >= 2


def es_duda_identidad(duda: Dict[str, Any]) -> bool:
    """[BB-030] Verifica si una duda indaga sobre la identidad fundamental de la bebida."""
    if not isinstance(duda, dict):
        return False
    if es_duda_multibebida(duda):
        return True
    preg = _normalizar_texto(duda.get("pregunta", ""))
    return any(frase in preg for frase in _FRASES_IDENTIDAD)


def tiene_duda_identidad(dudas: List[Dict[str, Any]]) -> bool:
    """Verifica si en las dudas existentes ya se preguntó de qué es la bebida."""
    return any(es_duda_identidad(d) for d in (dudas or []))


def _asume_bebida_especifica(duda: Dict[str, Any]) -> Optional[str]:
    """[BB-030] Detecta si la pregunta asume una bebida específica (ej. 'el café', 'al té') en lugar de ser neutra."""
    preg = _normalizar_texto(duda.get("pregunta", ""))
    for b in ("cafe", "te", "infusion", "chocolate", "jugo", "caldo"):
        if re.search(r"\b" + re.escape(b) + r"\b", preg):
            return b
    return None


def sanitizar_dudas_bebida(dudas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """[BB-030 / P1-BEVERAGE-DOUBTS-NEUTRAL-HIERARCHY · 2026-11-04] Garantiza jerarquía y consistencia lógica en dudas de bebidas:
    1. Si hay una duda multi-bebida (ej. '¿Qué bebida tienes en la taza?'), va SIEMPRE de primera.
    2. NUNCA permite una segunda duda que asuma una bebida específica (ej. '¿Cómo preparaste el café?')
       junto a una duda multi-bebida, pues crea contradicciones si el usuario toma té o chocolate.
    3. Acota el resultado a máximo 2 dudas limpias.
    """
    if not dudas:
        return []

    # Buscar si hay alguna duda multi-bebida
    idx_multi = None
    for i, d in enumerate(dudas):
        if es_duda_multibebida(d):
            idx_multi = i
            break

    if idx_multi is not None:
        duda_multi = dudas[idx_multi]
        resultado = [duda_multi]
        # Filtrar el resto de dudas: descartar las que asuman una bebida específica o sean idénticas
        for i, d in enumerate(dudas):
            if i == idx_multi:
                continue
            if _normalizar_texto(d.get("pregunta", "")) == _normalizar_texto(duda_multi.get("pregunta", "")):
                continue
            if _asume_bebida_especifica(d) is not None:
                # Contradicción directa: asume café/té cuando la duda principal pregunta qué bebida es
                continue
            # Solo permitir una duda secundaria verdaderamente neutra (ej. aditivos generales)
            resultado.append(d)
            if len(resultado) == 2:
                break
        return resultado

    # Si no hay multi-bebida, asegurar que si hay una de identidad vaya de primera
    idx_id = None
    for i, d in enumerate(dudas):
        if es_duda_identidad(d):
            idx_id = i
            break

    if idx_id is not None and idx_id > 0:
        dudas_reordenadas = [dudas[idx_id]] + [d for j, d in enumerate(dudas) if j != idx_id]
        return dudas_reordenadas[:2]

    return list(dudas)[:2]


def inyectar_dudas_bebida(result: Dict[str, Any], dudas_crudas: List[Dict[str, Any]], aclaracion: Optional[str] = None) -> List[Dict[str, Any]]:
    """Inyecta deterministamente dudas de identidad y aditivos en bebidas si no existen."""
    # Si vino como 'items' pero es una sola bebida servida en taza/vaso, colapsar a 'plato'
    if result.get("photo_kind") == "items":
        desc = result.get("description", "")
        if es_bebida("", desc, result.get("items")):
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

    # Si ya traía duda sobre la identidad de la bebida, sanitizarla y respetarla (BB-030)
    if tiene_duda_identidad(dudas_crudas):
        return sanitizar_dudas_bebida(dudas_crudas)

    tipo = tipo_de_bebida(meal_name, desc)
    duda_id = crear_duda_identidad(tipo, meal_name)
    duda_ad = crear_duda_aditivos(tipo, aclaracion)

    # Si el nombre del plato era genérico, asignamos como base el nombre de la opción supuesta
    if nombre_generico_bebida(meal_name):
        sup_nombre = duda_id["opciones"][0].get("nombre_plato")
        if sup_nombre:
            result["meal_name"] = sup_nombre

    # Ensamblar respetando el tope de 2 dudas y la jerarquía
    nuevas_dudas = [duda_id]
    if duda_ad is not None:
        nuevas_dudas.append(duda_ad)

    # Si había alguna otra duda no relacionada con identidad, conservarla como segunda si cabe
    if len(nuevas_dudas) < 2 and dudas_crudas:
        nuevas_dudas.append(dudas_crudas[0])

    return sanitizar_dudas_bebida(nuevas_dudas)
