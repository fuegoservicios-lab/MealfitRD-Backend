# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-240 · 2026-09-25] Cláusulas de las notas clínicas: embarazo/lactancia y por condición. Movido VERBATIM desde `graph_orchestrator.py`
(aire en el god-file); el grafo lo re-exporta con el mismo nombre, así que `go.<nombre>` sigue siendo el mismo objeto.
Solo datos: ningún import, ninguna lógica."""

_PREGNANCY_SAFETY_CLAUSES = (
    ("huevo", ("huevo", "huevos", "clara", "claras", "yema", "yemas"),
     ("yema y clara firmes",),
     "cocina el huevo POR COMPLETO (yema y clara firmes, sin puntos líquidos)"),
    # [P1-PLAN-LOTE-183] Carnes y aves (rd12: «pollo guisado sin 74 °C» ⇒ EMERGENCIA). tooltip-anchor: P1-PLAN-LOTE-183-CARNES-EMBARAZO
    ("carnes", ("pollo", "pechuga", "pechugas", "pavo", "cerdo", "res", "carne", "chivo", "cordero",
                "muslo", "muslos", "higado", "costilla", "costillas", "chuleta", "chuletas", "molida"),
     ("74 °c", "74°c", "sin partes rosadas"),
     "cocina las carnes y el pollo POR COMPLETO (74 °C por dentro, sin partes rosadas)"),
    ("deli", ("jamon", "salami", "mortadela", "fiambre", "embutido", "deli",
              "salchicha", "salchichon", "pepperoni", "tocineta", "tocino",
              # [P0-PREG-CURED-BETA · 2026-08-23] Curados/embutidos de los cinco
              # catálogos beta. El catálogo dinámico de abajo amplía por fila exacta;
              # estos tokens son el fallback fail-safe si la lectura no está disponible.
              "chorizo", "sobrasada", "cecina", "embuchado", "morcilla", "butifarra",
              "chistorra", "panceta", "longaniza", "chuleta ahumada"),
     ("74 °c", "74°c", "hasta que humee"),
     "calienta los embutidos/carnes tipo deli hasta que humeen (74 °C) y sírvelos al momento — "
     "fríos hay riesgo de listeria"),
    # [P1-REVIEWER-SEES-SAFETY-NOTES · 2026-08-09] Generalizada tras el residual medido
    # (corr=9909fb32: «lavado… no explicitadas para TODOS los platos» — la versión hojas-only
    # dejaba frutas/hierbas fuera y el reviewer generalizaba el rechazo). El lavado es guía
    # válida aunque el producto se cocine después.
    ("hojas", ("espinaca", "espinacas", "rucula", "arugula", "lechuga", "repollo",
               "berro", "berros", "acelga", "acelgas", "kale", "col rizada", "bok choy",
               "cilantro", "perejil", "albahaca", "tomate", "pepino", "zanahoria", "apio",
               "remolacha", "cebolla", "aji", "pimiento", "mango", "lechosa", "papaya", "pina", "fresa", "fresas",  # [P1-PLAN-LOTE-188] +cebolla/ají
               "guineo", "banana", "melon", "sandia", "uva", "uvas", "manzana", "pera",
               "chinola", "maracuya", "limon", "naranja", "toronja", "aguacate", "kiwi",
               "granada", "guayaba"),
     ("desinfecta",),
     "lava y desinfecta las frutas, verduras y hierbas frescas antes de usarlas (aunque "
     "se vayan a cocinar)"),
    # [P1-REVIEWER-SEES-SAFETY-NOTES · 2026-08-09] Canela: residual medido corr=9909fb32
    # («1 cucharadita de canela… debe reducirse a una pizca o sustituirse por canela de
    # Ceilán» — cumarina de la Cassia). La absolución «ceilan» aplica también si el
    # ingrediente ya la nombra (covered escanea receta + nombre + ingredientes).
    ("canela", ("canela",),
     ("ceilan",),
     "usa canela de Ceilán (no Cassia) y limítala a una pizca durante el embarazo"),
    ("papaya", ("lechosa", "papaya"),
     ("completamente madura",),
     "usa la lechosa/papaya COMPLETAMENTE madura (verde o pintona está contraindicada)"),
    ("hongos", ("champinon", "champinones", "hongo", "hongos", "setas", "portobello"),
     ("champinones por completo", "hongos por completo"),
     "cocina los champiñones/hongos por completo (nunca crudos)"),
    ("mariscos", ("pescado", "mejillon", "mejillones", "camaron", "camarones", "pulpo",
                  "calamar", "cangrejo", "langosta", "tilapia", "salmon", "mero",
                  "arenque", "bacalao", "sardina", "sardinas"),
     ("mariscos por completo",),
     "cocina el pescado y los mariscos POR COMPLETO (opacos y firmes; nada crudo ni a "
     "medio cocer)"),
)


_CONDITION_SAFETY_CLAUSES = {
    "dyslipidemia": (
        ("lacteo_magro", ("yogur", "yogurt", "queso", "leche"),
         ("descremad", "desnatad", "0%", "bajo en grasa", "light"),
         "usa los lácteos DESCREMADOS o 0-2% de grasa (yogur/queso/leche)"),
        ("yemas", ("huevo", "huevos", "yema", "yemas"),
         ("claras", "solo clara"),
         "limita las yemas a 3-4 por semana; las claras puedes usarlas libremente"),
    ),
    "hta": (
        ("sodio", ("queso", "jamon", "salami", "embutido", "tocineta", "enlatado",
                   "pan integral", "pan de agua", "tortilla integral", "atun en agua"),
         ("bajo en sodio", "baja en sodio", "sin sal"),
         "elige las versiones BAJAS EN SODIO (queso/pan/enlatados) y no añadas sal en la mesa"),
    ),
    "hypothyroid": (
        # [P1-SWAP-MACRO-REPAIR-BATCH · 2026-08-09] +toronja (rechazo medido run 31311796944:
        # «la toronja puede reducir la absorción de levotiroxina»). covered pasa de
        # («levotiroxina») a («4 horas»): mencionar el fármaco SIN la práctica correcta no
        # absuelve — el mismo run mostró al reviewer leyendo una separación de «1-12 minutos»
        # (el DESAYUNO con lácteos minutos después de la dosis en ayunas); la separación
        # horaria del slot es clase aparte (composición del desayuno), documentada en #14.
        ("levotiroxina", ("leche", "yogur", "yogurt", "queso", "soya", "soja", "tofu",
                          "linaza", "espinaca", "espinacas", "cafe", "toronja", "pomelo"),
         ("4 horas",),
         "toma la levotiroxina en ayunas y separa estos alimentos (lácteos/soya/linaza/"
         "espinacas/café/toronja) al menos 4 horas de la dosis — interfieren su absorción"),
    ),
    "pcos": (
        ("fruta_ig", ("mango", "lechosa", "papaya", "pina", "guineo", "banana", "uva",
                      "uvas", "sandia", "melon", "batido"),
         ("porcion pequena", "media taza", "acompanada de proteina"),
         "sirve la fruta dulce en porción PEQUEÑA (~½ taza) y acompáñala de proteína o "
         "grasa (yogur, queso, maní) para suavizar el pico glucémico"),
    ),
    "gastritis": (
        ("irritantes", ("limon", "naranja", "toronja", "pina", "vinagre", "picante",
                        "aji picante", "cafe", "salsa de tomate"),
         ("version suave", "sin picante"),
         "prepara la versión SUAVE: poco cítrico/vinagre, nada de picante, y prefiere "
         "cocción hervida, guisada u horneada sobre frituras"),
    ),
}
