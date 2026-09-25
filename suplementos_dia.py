# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-263 · 2026-09-25] Los suplementos que el usuario eligió llegan a CADA día del plan: ni más, ni menos.

Batería final (whey + creatina + omega-3, con estatina): el plan no traía NINGÚN suplemento. El prompt los pide por día
(`build_supplements_context`) y el schema del día los admite, pero la autocrítica reemplaza el día con lo que devuelve
el corrector, cuyo modelo (`SingleDayCorrectionModel`) no tiene el campo: todo día corregido los perdía (y el corrector
Pro ni siquiera recibe el día original). Además, «ni más, ni menos» solo era una instrucción del prompt.

Dos capas: `conservar` devuelve al día corregido los suplementos del original (conserva el texto personalizado del
modelo) y `completar`, al ensamblar, deja en cada día exactamente los elegidos que no veta el perfil clínico. Para cada
uno usa la entrada que el modelo escribió en ese día; si falta, la de otro día del plan; y si no hay ninguna, una
genérica que remite a la etiqueta del producto. Sin selección («recomiéndame»), solo copia a los días vacíos lo que el
modelo recomendó en otro día: no inventa recomendaciones. tooltip-anchor: P1-PLAN-LOTE-263-SUPLEMENTOS
"""
from __future__ import annotations

import copy
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Orden = prioridad al reconocer un nombre libre: «Electrolitos (Sodio + Potasio + Magnesio)» es electrolitos, no magnesio,
# y «Proteína vegana de guisante» no es whey.
_CLAVES = (
    ("electrolytes", r"electrolit|electrolyte|sales de rehidratacion|suero oral"),
    ("multivitamin", r"multi-?vitamin|multivit"),
    ("vegan_protein", r"proteina vegan|proteina vegetal|vegan protein|guisante|pea protein"),
    ("whey_protein", r"whey|suero de leche|proteina de suero|suero de proteina"),
    ("collagen", r"colageno|collagen"),
    ("creatine", r"creatin"),
    ("bcaa", r"bcaa|\beaa\b|aminoacidos? (?:ramificad|esencial)"),
    ("pre_workout", r"pre-?\s?entreno|pre-?\s?workout|beta-?\s?alanina"),
    ("fat_burner", r"quemador|termogenic|fat burner"),
    ("omega3", r"omega-?\s?3|aceite de pescado|fish oil|\bepa\b|\bdha\b"),
    ("probiotics", r"probiotic"),
    ("magnesium", r"magnesi"),
)
_CLAVES_RX = tuple((k, re.compile(p)) for k, p in _CLAVES)

_POR_DEFECTO = {
    "whey_protein": ("Proteína Whey", "1 scoop (≈30 g), la porción de tu etiqueta", "Después de entrenar o en la merienda",
                     "Te ayuda a completar la proteína del día."),
    "vegan_protein": ("Proteína Vegana", "1 scoop (≈30 g), la porción de tu etiqueta",
                      "Después de entrenar o en la merienda", "Proteína vegetal para completar la del día."),
    "creatine": ("Creatina Monohidrato", "3-5 g al día", "A cualquier hora, con una comida",
                 "Apoya la fuerza y la masa muscular; no aporta calorías."),
    "bcaa": ("Aminoácidos BCAA / EAA", "1 porción según la etiqueta", "Durante o después de entrenar",
             "Lo elegiste en tu formulario."),
    "pre_workout": ("Pre-Entreno", "1 porción según la etiqueta", "30 minutos antes de entrenar (nunca de noche)",
                    "Lo elegiste en tu formulario."),
    "fat_burner": ("Quemador de Grasa", "1 porción según la etiqueta", "Con el desayuno",
                   "Lo elegiste en tu formulario."),
    "collagen": ("Colágeno Hidrolizado", "1 porción según la etiqueta", "Con el desayuno",
                 "No cuenta como proteína completa para tu meta."),
    "multivitamin": ("Multivitamínico Completo", "1 al día", "Con el desayuno",
                     "Cubre huecos de vitaminas y minerales."),
    "omega3": ("Omega-3 (Aceite de Pescado)", "1 porción según la etiqueta", "Con una comida principal",
               "Grasas omega-3 para tu salud cardiovascular."),
    "magnesium": ("Magnesio (Citrato o Glicinato)", "1 porción según la etiqueta", "Por la noche",
                  "Apoya el descanso y la función muscular."),
    "probiotics": ("Probióticos", "1 al día", "Con el desayuno", "Apoya tu flora intestinal."),
    "electrolytes": ("Electrolitos", "1 porción según la etiqueta, disuelta en agua",
                     "Durante o después de entrenar, o si sudas mucho", "Repone el sodio, el potasio y el magnesio."),
}


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s).strip()


def clave_de(nombre):
    """La clave de `SUPPLEMENT_NAMES` que nombra este texto libre, o None."""
    t = _norm(nombre)
    for k, rx in _CLAVES_RX:
        if rx.search(t):
            return k
    return None


def _nombre(s) -> str:
    return str(s.get("name", "") if isinstance(s, dict) else s or "")


def elegidos(form_data) -> list:
    """Los suplementos elegidos (enum válido, sin repetir, en su orden) menos los que veta el perfil clínico."""
    if not isinstance(form_data, dict):
        return []
    try:
        from constants import SUPPLEMENT_NAMES
    except Exception:
        return []
    out = []
    from suplementos import normalizar_suplementos   # [P1-PLAN-LOTE-292] formulario nuevo o viejo, un solo lector
    for s in normalizar_suplementos(form_data)["toma"]:
        if s in SUPPLEMENT_NAMES and s not in out:
            out.append(s)
    if out:
        try:
            from condition_rules import contraindicated_supplements
            vetados = contraindicated_supplements(form_data) or {}
        except Exception as e:  # fail-secure: sin veto calculable no se AÑADE nada (el prompt sigue pidiéndolos)
            logger.warning(f"[P1-PLAN-LOTE-263] veto de suplementos no calculable ({type(e).__name__}): no se completan")
            return []
        out = [s for s in out if s not in vetados]
    return out


def conservar(corregido, original):
    """El corrector de la autocrítica no tiene campo `supplements`: el día corregido recupera los del original."""
    try:
        if (isinstance(corregido, dict) and isinstance(original, dict) and not corregido.get("supplements")
                and original.get("supplements")):
            corregido["supplements"] = copy.deepcopy(original["supplements"])
    except Exception:
        pass
    return corregido


def completar(result, form_data) -> int:
    """Deja en cada día los suplementos elegidos, ni más ni menos. Devuelve cuántas entradas añadió."""
    try:
        from suplementos import normalizar_suplementos, RECOMENDABLES
        _n = normalizar_suplementos(form_data)   # [P1-PLAN-LOTE-292]
        if not isinstance(result, dict) or not isinstance(form_data, dict) or not (_n["toma"] or _n["recomendar"]):
            return 0
        dias = [d for d in (result.get("days") or []) if isinstance(d, dict)]
        if not dias:
            return 0
        pedidos = elegidos(form_data)
        anadidas = quitadas = 0
        if not pedidos:
            if _n["toma"]:
                return 0          # todo lo elegido está vetado: la barredora clínica ya hizo su trabajo
            modelo = next((d["supplements"] for d in dias if d.get("supplements")), None)
            for d in dias:
                if modelo and not d.get("supplements"):
                    d["supplements"] = copy.deepcopy(modelo)
                    anadidas += len(modelo)
        else:
            del_modelo = {}
            for d in dias:
                for s in d.get("supplements") or []:
                    k = clave_de(_nombre(s))
                    if k in pedidos and k not in del_modelo:
                        del_modelo[k] = s
            for d in dias:
                antes = [s for s in (d.get("supplements") or [])]
                nuevos, vistos = [], set()
                for s in antes:
                    k = clave_de(_nombre(s))
                    if k in pedidos and k not in vistos:
                        nuevos.append(s)
                        vistos.add(k)
                    elif _n["recomendar"] and k in RECOMENDABLES and k not in vistos:
                        # [P1-PLAN-LOTE-292] «¿Te recomendamos?» = sí: una recomendación con respaldo se queda
                        # (el veto clínico ya lo pasó la barredora del orquestador); un quemador, nunca.
                        nuevos.append(s)
                        vistos.add(k)
                    else:
                        quitadas += 1
                for k in pedidos:
                    if k in vistos:
                        continue
                    if k in del_modelo:
                        nuevos.append(copy.deepcopy(del_modelo[k]))
                    else:
                        n, dosis, cuando, por_que = _POR_DEFECTO[k]
                        nuevos.append({"name": n, "dose": dosis, "timing": cuando, "reason": por_que})
                    anadidas += 1
                d["supplements"] = nuevos
        if anadidas or quitadas:
            logger.info(f"💊 [P1-PLAN-LOTE-263] suplementos por día: +{anadidas} restituido(s), -{quitadas} no "
                        f"elegido(s) o repetido(s) (elegidos: {pedidos or 'recomendación libre'}).")
        return anadidas
    except Exception as e:  # nunca rompe el ensamblado
        logger.warning(f"[P1-PLAN-LOTE-263] completar suplementos falló ({type(e).__name__}: {e})")
        return 0
