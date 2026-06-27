"""[P3-CONDITION-ENGINE · 2026-06-14] Motor de constraints clínicos DECLARATIVO.

Generaliza el patrón ad-hoc por-condición (que vivía disperso en ~7 callsites de graph_orchestrator
+ plan_generator) a un REGISTRO de datos: cada condición del set Pareto cardiometabólico DR es una
fila `ConditionRule` con (detección SSOT, bloque de prompt citable, sustituciones deterministas de
ingredientes, precedencia para comorbilidad, clasificación safety/derivación). Los consumidores
(prompt builder, guard de sustituciones, gate FS9) ITERAN el registro → añadir HTA/dislipidemia/
anemia es declarar una fila, no editar 5 sitios. Cierra el drift detector↔prompt (un solo SSOT) y la
explosión combinatoria de comorbilidades.

Estado del enforcement por condición (honesto):
- ERC: cap de proteína 0.8 g/kg + gate nefrólogo → enforced en graph (no migrado aquí aún) + referral.
- DM2: fibra ADA (advisory) + sustitución azúcar→stevia (enforced, via este motor).
- HTA: sustitución de sodio (embutidos/cubitos/bacalao→fresco) ENFORCED via este motor (NUEVO).
- Dislipidemia: sustitución de grasa saturada (mantequilla→aceite, lácteos enteros→bajos en grasa,
  tocino/chicharrón→lean) ENFORCED via este motor + techo satfat<7% kcal medido en el panel
  (columna saturated_fat_g poblada desde USDA, P4-UNIFIED-RESOLVER). El swap baja el LDL preservando
  proteína; el techo del panel marca el residual. [P4-DYSLIPIDEMIA-ENFORCED · 2026-06-14]
- Anemia: prompt + densidad de hierro (el panel ya computa iron_mg).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from constants import (
    RENAL_CONDITION_TERMS, DIABETES_CONDITION_TERMS, HTA_CONDITION_TERMS,
    DYSLIPIDEMIA_CONDITION_TERMS, ANEMIA_CONDITION_TERMS,
    PREGNANCY_CONDITION_TERMS, HYPOTHYROID_CONDITION_TERMS, GOUT_CONDITION_TERMS,
    NAFLD_CONDITION_TERMS, PCOS_CONDITION_TERMS, GASTRITIS_CONDITION_TERMS,
    BARIATRIC_CONDITION_TERMS,
)

SAFETY_HARD = "safety_hard"            # reglas que el motor REESCRIBE determinísticamente
CLINICAL_REFERRAL = "clinical_referral"  # se derivan al profesional (no se auto-prescriben)

_MEDICAL_NONE_SENTINELS = frozenset({
    "", "ninguna", "ninguno", "ningunas", "ningunos", "none", "n/a", "na",
    "no", "nada", "sin alergias", "sin condiciones", "ninguna alergia",
})


@dataclass(frozen=True)
class ConditionRule:
    """Una condición del set Pareto como DATO. `substitutions` = ((tokens...), reemplazo, etiqueta,
    preserve_qty?); `preserve_qty` (4º elem, opcional, default False) marca staples que aportan peso
    (embutidos/bacalao) para los que el guard preserva el prefijo de cantidad y recalcula macros.
    `sub_negatives` = frases que vetan la sustitución (ej. 'baja en sodio'). `precedence` menor =
    mayor prioridad clínica de seguridad (ERC manda)."""
    id: str
    label: str
    terms: tuple
    prompt_block: str
    substitutions: tuple = ()
    sub_negatives: tuple = ()
    precedence: int = 50
    classification: str = SAFETY_HARD


# ── Tablas de sustitución determinista (patrón generalizado del sugar-guard) ──
# Cada fila: (tokens, reemplazo, etiqueta, preserve_qty). `preserve_qty=True` SOLO para staples que
# aportan peso/macros (embutidos, bacalao salado) → el guard preserva el prefijo de cantidad
# ("100g de longaniza" → "100g de Pechuga de pollo") y recalcula los macros del plato por delta.
# `False` para condimentos/azúcares donde la UNIDAD misma es lo contraindicado ("1 cubito de…",
# "1 cda de miel") o son "al gusto" — preservar el prefijo dejaría la palabra ofensora en el string.
_DM2_SUGAR_SUBS = (
    (("miel", "honey"), "Stevia al gusto", "miel", False),
    (("sirope", "jarabe", "syrup"), "Stevia al gusto", "sirope/jarabe", False),
    (("panela", "melaza", "molasses"), "Stevia al gusto", "panela/melaza", False),
    (("leche condensada", "dulce de leche", "condensed milk"), "Leche evaporada sin azúcar", "leche condensada", False),
    (("refresco", "gaseosa", "soda", "malta", "jugo de caja", "jugo concentrado",
      "jugo embotellado", "jugo de cajita"), "Agua", "bebida azucarada", False),
    (("azucar", "azúcar", "sugar"), "Stevia al gusto", "azúcar", False),
)
_DM2_SUGAR_NEGATIVES = ("sin azucar", "no azucar", "0 azucar", "cero azucar", "libre de azucar",
                        "0% azucar", "sin azucares", "bajo en azucar")

# [P1-DM2-GLYCEMIC-GUARD · 2026-06-27] Sustituciones DETERMINISTAS calorie-neutral para DM2 que el revisor
# médico estaba rechazando: (1) TORONJA/POMELO → fruta baja en IG: la toronja inhibe CYP3A4 y potencia
# sulfonilureas/repaglinida/saxagliptina → riesgo de hipoglucemia severa. Decisión del owner: evitarla SIEMPRE
# en DM2 (sin depender de que el usuario declare el fármaco). (2) REFINADOS de alto índice glucémico → su
# versión INTEGRAL (mismo alimento, mismas calorías, IG mucho menor) → aplana el pico postprandial. preserve_qty
# (mismo gramaje, el delta de macros lo ajusta el guard). Naturalmente idempotente (lo integral no re-matchea
# el token "blanco"). Reusa el motor de sustitución → aplica en S1 (Guard 3) y en las superficies de UPDATE
# (condition_substitution_backstop_for_meal). Los reemplazos resuelven al catálogo verificado.
_DM2_GLYCEMIC_SUBS = (
    (("toronja", "pomelo", "grapefruit"), "Fresa", "toronja/pomelo (interacción CYP3A4 con antidiabéticos → hipoglucemia)", True),
    (("arroz blanco", "arroz pulido"), "Arroz integral", "arroz blanco refinado (IG alto)", True),
    (("pan blanco", "pan rallado", "pan de molde blanco", "pan de agua"), "Pan integral", "pan blanco/refinado (IG alto)", True),
    (("tortilla de trigo", "tortilla de harina"), "Pan integral", "tortilla de trigo refinada (IG alto)", True),
    (("harina de trigo refinada", "harina blanca de trigo"), "Avena", "harina refinada (IG alto)", True),
)

_HTA_SODIUM_SUBS = (
    (("embutido", "salami", "longaniza", "salchicha", "mortadela", "tocineta",
      "bacon", "chorizo", "jamon", "jamón"), "Pechuga de pollo", "embutidos", True),
    # OJO: tokens estrechos para no colisionar con proteína legítima — 'cubito de' (NO 'cubito', que
    # matchearía 'cubitos de pollo' = pollo en cubos); 'salsa de soya' (NO 'soya' desnudo, que
    # borraría tofu/leche-de-soya/carne-de-soya, proteína vegetal). Bug encontrado por review adversaria.
    (("cubito de", "sazon en polvo", "sazón en polvo", "sazon completa", "caldo en cubo",
      "consome", "consomé", "maggi", "sopita", "sazonador"), "Especias naturales (ajo, cebolla, orégano, comino)", "cubitos/sazón en polvo", False),
    # [P2-SUBS-RESOLVE · 2026-06-15] "Filete de pescado blanco" (nombre del catálogo) — NO "Pescado
    # fresco" (no resolvía → fantasma en la lista + delta de macros perdía el ingrediente). Audit P2.
    (("bacalao salado", "bacalao seco", "arenque salado"), "Filete de pescado blanco", "pescado salado", True),
    (("salsa de soya", "salsa de soja", "teriyaki"), "Limón con especias", "salsa de soya", False),
    (("tajin", "tajín", "sal de ajo", "sal de cebolla"), "Especias sin sal añadida", "sazonadores salados", False),
)
_HTA_SODIUM_NEGATIVES = ("baja en sodio", "bajo en sodio", "sin sal", "sin sodio", "reducido en sodio")

# [P4-DYSLIPIDEMIA-ENFORCED · 2026-06-14] Dislipidemia → grasa saturada: swaps fuente-grasa-saturada →
# versión magra/insaturada de la MISMA categoría (preserva proteína, baja el LDL). preserve_qty=True: el
# reemplazo conserva la cantidad/volumen (misma comida, versión leaner) y el delta de macros lo ajusta el
# guard. Tokens ESTRECHOS (lección del bug 'soya'): nada de 'manteca' desnudo (matchea 'mantecado'),
# 'nata' (matchea 'natural'/'natilla') ni 'crema' (matchea 'crema de maní') — solo frases inequívocas.
_DYSLIPIDEMIA_SATFAT_SUBS = (
    (("mantequilla", "manteca de cerdo", "manteca vegetal", "margarina"), "Aceite de oliva", "mantequilla/manteca/margarina", True),
    # [P2-2 · 2026-06-16] REACTIVADO: ya existe la fila distinta `Leche descremada` (migración p2_2), así que
    # el swap reduce grasa real (entera 3.25g → descremada ~0.1g). Token ESTRECHO 'leche entera' (NO 'leche'
    # desnudo, que borraría descremada/evaporada). El negative 'descremad' (abajo) lo hace idempotente.
    (("leche entera", "leche completa"), "Leche descremada", "leche entera", True),
    (("crema de leche", "crema espesa"), "Leche evaporada", "crema de leche", True),
    (("queso amarillo", "queso cheddar", "queso crema"), "Queso cottage", "queso alto en grasa", True),
    (("yogur griego entero", "yogurt griego entero", "yogur entero", "yogurt entero", "yogur natural entero"),
     "Yogurt griego sin azúcar", "yogur entero", True),
    (("tocino", "tocineta", "chicharron", "chicharrón"), "Pechuga de pollo", "tocino/chicharrón", True),
)
# [P2-PREGNANCY-MERCURY-GUARD · 2026-06-22] (audit fresco P2-6) Swap determinista de pescados ALTOS EN
# MERCURIO (metilmercurio = teratógeno) → pescado blanco bajo en mercurio, SOLO para embarazo/lactancia.
# Espejo de los subs de condición (mismo mecanismo token→reemplazo resoluble). Pre-fix la ÚNICA defensa era
# el prompt_block + el gate FS9 (obstetra). Incluye SOLO especies inequívocamente altas (FDA "Choices to
# Avoid"): tiburón, pez espada, marlin, blanquillo (tilefish del Golfo), king mackerel (caballa gigante /
# macarela rey). EXCLUYE 'atún' A PROPÓSITO: el atún light/enlatado es FDA "Best/Good Choice" en moderación
# durante el embarazo → un swap ciego de un staple barato sería over-restrictivo (degradaría calidad/costo);
# el prompt_block ya advierte sobre 'atún grande'. preserve_qty=True (misma porción de pescado blanco).
# 'Filete de pescado blanco' resuelve al catálogo (lo usa también _HTA_SODIUM_SUBS).
_PREGNANCY_MERCURY_SUBS = (
    (("tiburon", "tiburón", "pez espada", "pez-espada", "marlin", "blanquillo",
      "caballa gigante", "macarela rey", "king mackerel"),
     "Filete de pescado blanco", "pescado alto en mercurio", True),
)
_PREGNANCY_MERCURY_NEGATIVES = ("bajo en mercurio", "blanco", "tilapia")

_DYSLIPIDEMIA_NEGATIVES = ("descremad", "baja en grasa", "bajo en grasa", "light", "desnatad",
                           "sin grasa", "0% grasa", "0 grasa",
                           # nueces/semillas = grasa INSATURADA (saludable): 'mantequilla de maní/almendra'
                           # NO debe caer en el swap de 'mantequilla'. (Veto análogo al 'baja en sodio' de HTA.)
                           "mani", "maní", "cacahuate", "almendra", "marañon", "maranon", "semilla",
                           # [review adversaria] 'chicharron'/'tocino' desnudos colisionan con
                           # 'chicharrón DE POLLO' (pollo frito criollo = pollo), 'tocino DE PAVO' (ya magro)
                           # y 'tocino DE CIELO' (postre). Ningún swap de dislipidemia debe disparar sobre
                           # un '…de pollo/pavo'. 'coco' (crema de coco) NO es lácteo → no mislabelear.
                           "de pollo", "de pavo", "de cielo", "coco")


# ── El REGISTRO declarativo (SSOT del comportamiento por condición) ──
CONDITION_RULES: tuple = (
    ConditionRule(
        # [P1-RENAL-SODIUM-SUBS · 2026-06-19] (audit fresco P1-4) La fila renal ENFORZA ahora la
        # restricción de sodio de forma determinista, no solo en el prompt_block. Reusa la tabla de HTA
        # (`_HTA_SODIUM_SUBS`: embutidos/cubitos/bacalao→fresco) — la restricción de sodio en ERC es
        # estándar-de-cuidado (sobrecarga de volumen/edema/proteinuria) y los ofensores son los mismos. El
        # swap embutido→Pechuga de pollo sube proteína/100g, pero el cap renal KDIGO la re-trima POST-subs
        # (Guard 3.6 `_trim_day_protein_to_ceiling`) y la re-verifica POST-truthup (Guard 8z.1, P1-3) → no
        # rompe el techo de proteína. Cierra la asimetría con HTA (sodio enforced) que tenía solo ERC.
        # DEPENDENCIA DE KNOB (operador): el re-trim del cap (Guard 1/3.6/4d/8z.1) está gateado por
        # RENAL_CAP_ENABLED, mientras estos subs corren bajo CONDITION_RULES_ENABLED → si se apaga el cap pero
        # NO las condiciones, el swap subiría proteína sin re-trim (default RENAL_CAP_ENABLED=True lo evita;
        # asimetría pre-existente compartida con los subs dislipidemia/DM2-en-renal).
        # TRADE-OFF renal+VEGANO (follow-up del owner): para dietas veg* el redirect diet-aware reemplaza el
        # ofensor por 'Lentejas' (alto K/fósforo, lo que KDIGO pide moderar) en vez de pollo → es un trade
        # lateral (quita sodio, añade K/P), acotado en cantidad y cubierto por el gate nefrólogo. Anclado por
        # test (renal+vegano → Lentejas); cambiarlo a un veto K/P es decisión clínica del owner (ver P2 audit).
        # [P2-RENAL-POTASSIUM-DETERMINISTIC · 2026-06-22] (audit fresco P2-10) DECISIÓN documentada (el audit
        # permite "implementar O documentar la asimetría"): NO se añade un cap/veto DETERMINISTA de potasio/
        # fósforo para ERC (a diferencia de proteína —cap KDIGO— y sodio —subs— que SÍ son deterministas).
        # Razón: un cap de K/P seguro requiere (a) data de potasio/fósforo por-ingrediente VALIDADA (el catálogo
        # no la tiene con confianza clínica) y (b) que el umbral por estadio (G3a→G5) lo defina un nefrólogo —
        # un veto ciego "guineo/aguacate/leguminosas fuera" degradaría planes y chocaría con HTA (que pide K
        # ALTO) y anemia bajo comorbilidad. La defensa actual es: prompt_block ("modera K/P si aparecen en
        # exceso") + classification=CLINICAL_REFERRAL + gate FS9 (el plan se deriva al nefrólogo). Revisitar si
        # el owner consigue data de K/P validada + criterio clínico por estadio.
        id="renal", label="Enfermedad renal crónica", terms=RENAL_CONDITION_TERMS,
        precedence=10, classification=CLINICAL_REFERRAL,
        substitutions=_HTA_SODIUM_SUBS, sub_negatives=_HTA_SODIUM_NEGATIVES,
        prompt_block=(
            "🫘 REGLA CLÍNICA — ENFERMEDAD RENAL (KDIGO 2024) — PRECAUCIÓN, REQUIERE NEFRÓLOGO:\n"
            "   • Proteína MODERADA, NO alta: porciones modestas de proteína de alta calidad (huevo, "
            "pescado, pollo) — NO maximices la proteína (lo contrario a un plan de hipertrofia).\n"
            "   • SODIO BAJO: sal medida y mínima; evita embutidos, cubitos/sazón, bacalao salado, ultra-procesados.\n"
            "   • Modera alimentos MUY altos en potasio/fósforo si aparecen en exceso (vísceras, lácteos en gran "
            "cantidad, exceso de guineo/aguacate); el balance fino lo define el nefrólogo.\n"
            "   • Este plan es ORIENTATIVO y NO sustituye la indicación de un profesional de salud renal."),
    ),
    ConditionRule(
        id="dm2", label="Diabetes T2 / prediabetes", terms=DIABETES_CONDITION_TERMS,
        precedence=30, substitutions=_DM2_SUGAR_SUBS + _DM2_GLYCEMIC_SUBS, sub_negatives=_DM2_SUGAR_NEGATIVES,
        prompt_block=(
            "🩸 REGLA CLÍNICA — DIABETES T2 / PREDIABETES (ADA 2025/2026, CALIDAD DEL CARBOHIDRATO):\n"
            "   • NO se trata de 'bajar los carbohidratos' ni de un % fijo: prioriza la CALIDAD del carbohidrato.\n"
            "   • FIBRA ALTA (objetivo ≥14 g por cada 1000 kcal): incluye leguminosas (habichuelas, lentejas, "
            "gandules), avena, vegetales abundantes y fruta entera con cáscara.\n"
            "   • GRANOS INTEGRALES INTACTOS: arroz integral, avena y víveres con fibra (batata, yuca, plátano "
            "verde) sobre harinas refinadas, pan blanco y arroz blanco pelado.\n"
            "   • PORCIÓN DE ALMIDÓN — MÁXIMO ~150 g por comida de víver/almidón de ALTO índice glucémico "
            "(batata, yuca, papa, plátano maduro, arroz, pan, casabe). NO sirvas porciones grandes (300-400g) "
            "de un solo almidón: dispara la glucosa postprandial. Reparte el resto del plato en proteína magra, "
            "grasa saludable y vegetales/fibra para aplanar el pico.\n"
            "   • PROHIBIDA la TORONJA/POMELO: interactúa con medicamentos antidiabéticos (CYP3A4) y puede causar "
            "hipoglucemia severa. Usa otras frutas bajas en índice glucémico (fresa, manzana, cítricos pequeños).\n"
            "   • PROHIBIDAS las bebidas azucaradas y los azúcares añadidos (miel, sirope, dulces); endulza con "
            "fruta o estevia.\n"
            "   • Combina SIEMPRE el carbohidrato con proteína + grasa saludable + fibra en la misma comida."),
    ),
    ConditionRule(
        id="hta", label="Hipertensión arterial", terms=HTA_CONDITION_TERMS,
        precedence=40, substitutions=_HTA_SODIUM_SUBS, sub_negatives=_HTA_SODIUM_NEGATIVES,
        prompt_block=(
            "🧂 REGLA CLÍNICA — HIPERTENSIÓN (patrón DASH, NHLBI/AHA-ACC 2025):\n"
            "   • SODIO BAJO (meta ≤1500 mg/día): NADA de embutidos (salami, longaniza, jamón), cubitos/sazón "
            "en polvo, bacalao salado, ni 'sal al gusto' genérica. Especifica sal medida mínima (≤1 g/día).\n"
            "   • POTASIO/MAGNESIO/CALCIO ALTOS: prioriza vegetales, frutas (guineo, aguacate con moderación), "
            "leguminosas, lácteos bajos en grasa, vegetales de hoja verde.\n"
            "   • Sabor sin sal: ajo, cebolla, orégano, comino, cilantro, limón, vinagre.\n"
            "   • Evita ultra-procesados y enlatados altos en sodio."),
    ),
    ConditionRule(
        id="dyslipidemia", label="Dislipidemia / colesterol alto", terms=DYSLIPIDEMIA_CONDITION_TERMS,
        precedence=45, substitutions=_DYSLIPIDEMIA_SATFAT_SUBS, sub_negatives=_DYSLIPIDEMIA_NEGATIVES,
        prompt_block=(
            "🫀 REGLA CLÍNICA — DISLIPIDEMIA (AHA 2021/ACC 2025):\n"
            "   • GRASA SATURADA BAJA (<7% de las calorías): evita frituras, piel de pollo, grasa visible de "
            "carnes, embutidos, mantequilla y lácteos enteros. Usa cocción al horno/plancha/hervido.\n"
            "   • FIBRA SOLUBLE alta (baja el LDL): avena, habichuelas/lentejas, berenjena, manzana, cítricos.\n"
            "   • GRASAS SALUDABLES: aguacate, aceite de oliva, pescado graso (sardina/salmón), nueces — con moderación.\n"
            "   • Sin grasas trans (margarina dura, productos de repostería industrial)."),
    ),
    # [P2-ANEMIA-TARGET · 2026-06-15] DECISIÓN: la fila de anemia NO lleva `substitutions` a propósito.
    # Anemia es un DÉFICIT a llenar, no un exceso a remover → no hay token-ofensor natural (un swap como
    # 'pollo'→res destruiría variedad/proteína y chocaría con gota/ERC por comorbilidad), y el catálogo
    # es-DO NO tiene hígado (un reemplazo a hígado violaría test_p2_subs_resolubility_contract). La
    # cobertura va por: prompt_block + piso de hierro RDA-por-sexo (gap) + build_supplement_recommendations
    # + el condition_target advisory de anemia (micronutrients.py, P2-ANEMIA-TARGET). NO añadir swaps sin
    # re-evaluar P2-3.
    ConditionRule(
        id="anemia", label="Anemia ferropénica", terms=ANEMIA_CONDITION_TERMS,
        precedence=60,
        prompt_block=(
            "🩸 REGLA CLÍNICA — ANEMIA FERROPÉNICA (densidad de hierro):\n"
            "   • HIERRO HEMO: incluye carnes rojas magras, hígado (1x/sem), pollo, pescado.\n"
            "   • POTENCIA LA ABSORCIÓN: acompaña el hierro con vitamina C (naranja, limón, tomate, pimiento).\n"
            "   • FOLATO + B12: leguminosas, vegetales de hoja verde, huevo.\n"
            "   • SEPARA del café/té/lácteos en la misma comida (inhiben la absorción de hierro)."),
    ),
    # [P1-CONDITION-COVERAGE · 2026-06-14] Condiciones comunes que faltaban. ADVISORY (prompt_block +
    # gate de derivación FS9, classification CLINICAL_REFERRAL): la guía es estándar-de-cuidado general;
    # la regla fina (qué limitar/sustituir por estadio/severidad) la valida el profesional, NO el motor
    # determinista (evita enforcement clínico sin revisión humana — cautela del audit P1). Embarazo es
    # además SEGURIDAD: el déficit calórico ya lo bloquea el gate en nutrition_calculator.
    ConditionRule(
        id="pregnancy", label="Embarazo / lactancia", terms=PREGNANCY_CONDITION_TERMS,
        precedence=15, classification=CLINICAL_REFERRAL,
        # [P2-PREGNANCY-MERCURY-GUARD · 2026-06-22] Swap determinista de pescado alto en mercurio (ver
        # `_PREGNANCY_MERCURY_SUBS`). Antes embarazo era advisory-puro (solo prompt + FS9).
        substitutions=_PREGNANCY_MERCURY_SUBS, sub_negatives=_PREGNANCY_MERCURY_NEGATIVES,
        prompt_block=(
            "🤰 REGLA CLÍNICA — EMBARAZO / LACTANCIA (SEGURIDAD — REQUIERE OBSTETRA/NUTRICIONISTA):\n"
            "   • NUNCA un déficit calórico: usa AL MENOS mantenimiento (el requerimiento sube en 2º/3º "
            "trimestre y lactancia). NO es momento de perder peso.\n"
            "   • FOLATO + HIERRO altos: vegetales de hoja verde, leguminosas, carnes magras, huevo; "
            "acompaña el hierro con vitamina C.\n"
            "   • CALCIO + PROTEÍNA suficientes: lácteos PASTEURIZADOS, pescado bajo en mercurio.\n"
            "   • EVITA por listeria/seguridad: embutidos y quesos/lácteos NO pasteurizados, pescado "
            "crudo, carne/huevo poco cocidos, pescados altos en mercurio (tiburón, pez espada, atún "
            "grande), alcohol y exceso de cafeína.\n"
            "   • Este plan es ORIENTATIVO y NO sustituye el control prenatal."),
    ),
    ConditionRule(
        id="hypothyroid", label="Hipotiroidismo", terms=HYPOTHYROID_CONDITION_TERMS,
        precedence=55, classification=CLINICAL_REFERRAL,
        prompt_block=(
            "🦋 REGLA CLÍNICA — HIPOTIROIDISMO (orientativa, requiere endocrinólogo):\n"
            "   • YODO ADECUADO (sin exceso): sal yodada con moderación, pescado/mariscos, huevo, lácteos.\n"
            "   • SELENIO + ZINC: nuez (1-2/día si no hay alergia), huevo, pollo, mariscos.\n"
            "   • Modera los GOITRÓGENOS CRUDOS en gran cantidad (repollo, brócoli, coliflor, yuca cruda); "
            "cocidos pierden el efecto — no es necesario eliminarlos.\n"
            "   • Separa los suplementos de hierro/calcio de la levotiroxina (interfieren su absorción)."),
    ),
    ConditionRule(
        id="gout", label="Gota / ácido úrico alto", terms=GOUT_CONDITION_TERMS,
        precedence=55, classification=CLINICAL_REFERRAL,
        prompt_block=(
            "🦶 REGLA CLÍNICA — GOTA / HIPERURICEMIA (orientativa, requiere médico):\n"
            "   • PURINAS BAJAS: limita vísceras (hígado, riñón, mollejas), mariscos, sardina/anchoa y "
            "el exceso de carnes rojas.\n"
            "   • CERO alcohol (especialmente cerveza) y bebidas/alimentos altos en fructosa (refrescos, "
            "jugos azucarados).\n"
            "   • HIDRATACIÓN abundante (agua) + lácteos BAJOS en grasa (protectores).\n"
            "   • Vegetales, fruta entera, granos integrales y proteína magra con moderación."),
    ),
    ConditionRule(
        id="nafld", label="Hígado graso (NAFLD/MAFLD)", terms=NAFLD_CONDITION_TERMS,
        precedence=50, classification=CLINICAL_REFERRAL,
        prompt_block=(
            "🫥 REGLA CLÍNICA — HÍGADO GRASO (NAFLD/MAFLD) (orientativa, requiere médico):\n"
            "   • REDUCE azúcar añadida y FRUCTOSA (refrescos, jugos, dulces, sirope) y carbohidratos "
            "refinados (pan blanco, harinas) — son el principal motor de la grasa hepática.\n"
            "   • PATRÓN MEDITERRÁNEO: aceite de oliva, pescado, vegetales, leguminosas, granos integrales.\n"
            "   • CERO alcohol.\n"
            "   • Si hay sobrepeso, una pérdida gradual de peso (7-10%) mejora la esteatosis."),
    ),
    ConditionRule(
        id="pcos", label="SOP (ovario poliquístico)", terms=PCOS_CONDITION_TERMS,
        precedence=50, classification=CLINICAL_REFERRAL,
        prompt_block=(
            "🌸 REGLA CLÍNICA — SOP / OVARIO POLIQUÍSTICO (orientativa, requiere ginecólogo/endocrino):\n"
            "   • CALIDAD DEL CARBOHIDRATO (resistencia a la insulina): prioriza fibra alta, granos "
            "integrales intactos y leguminosas; evita azúcares añadidos y harinas refinadas.\n"
            "   • PROTEÍNA + GRASA SALUDABLE en cada comida para estabilizar la glucosa.\n"
            "   • Patrón antiinflamatorio (pescado graso, aceite de oliva, vegetales).\n"
            "   • Combina con actividad física; el manejo del peso mejora la sensibilidad a la insulina."),
    ),
    # [P1-GASTRITIS-RULE · 2026-06-26] (auditoría gap #8) El chip 'Gastritis' del form no tenía ConditionRule
    # → solo disparaba el FS9 genérico advisory. Ahora tiene guía citable (estándar-de-cuidado: evitar
    # irritantes). ADVISORY (prompt_block + CLINICAL_REFERRAL) como hipotiroidismo/gota/NAFLD: la severidad
    # (qué tolera cada paciente) la define el gastroenterólogo, no el motor determinista — los "irritantes"
    # son individuales (no hay un token-ofensor universal que swappear sin destruir el plato criollo). El
    # prompt cubre la dirección segura; precedencia baja (no choca con cardiometabólicas).
    ConditionRule(
        id="gastritis", label="Gastritis / reflujo (ERGE)", terms=GASTRITIS_CONDITION_TERMS,
        precedence=58, classification=CLINICAL_REFERRAL,
        prompt_block=(
            "🔥 REGLA CLÍNICA — GASTRITIS / REFLUJO (ERGE) (orientativa, requiere gastroenterólogo):\n"
            "   • EVITA IRRITANTES: picante (ají picante, salsas picantes), exceso de cítricos ácidos "
            "(naranja agria, limón en gran cantidad), tomate concentrado, café y bebidas con cafeína, "
            "chocolate, menta, bebidas gaseosas y alcohol.\n"
            "   • SIN FRITOS NI MUY GRASOSO: prefiere cocción al horno/plancha/hervido/al vapor; las frituras "
            "y comidas muy grasosas retrasan el vaciamiento y empeoran el reflujo.\n"
            "   • COMIDAS PEQUEÑAS Y FRECUENTES: porciones moderadas, comer despacio, NO acostarse hasta "
            "2-3 horas después de comer; evita las comidas muy abundantes de noche.\n"
            "   • PREFIERE: avena, arroz, viandas hervidas, vegetales cocidos no ácidos, proteínas magras "
            "(pollo/pescado/pavo a la plancha), lácteos bajos en grasa según tolerancia.\n"
            "   • Este plan es ORIENTATIVO; la tolerancia a cada alimento es individual y la define tu médico."),
    ),
    # [P1-BARIATRIC-CLINICAL-RULES · 2026-06-27] (audit corr=558af493: plan bariátrico RECHAZADO CRÍTICO por
    # el revisor médico → "5¼ lonjas de queso" (dumping/volumen), miel (azúcar simple), pescado en merienda
    # nocturna, "2.5 fresas solo la cáscara"). El chip 'Cirugía Bariátrica' (P2-FORM-FREETEXT-SATISFIES) ya
    # enrutaba 6 comidas (P1-CLINICAL-MEAL-COUNT) pero NO había reglas clínicas → el generador producía platos
    # inseguros y el reviewer los rechazaba → plan degradado. Ruleset diseñado + verificado adversarialmente
    # (workflow 3 lentes clínicas + crítico). SAFETY_HARD: substitutions deterministas (azúcar simple → estevia,
    # reusa _DM2_SUGAR_SUBS) + cap de porción determinista (cap_bariatric_portions, graph_orchestrator) + este
    # prompt_block. precedence 20 (alta: por encima de DM2/HTA; por debajo de renal/embarazo). El cap duro de
    # queso/lácteos es el backstop del rechazo literal #1; el prompt cubre dumping/volumen/proteína-primero.
    ConditionRule(
        id="bariatric", label="Cirugía bariátrica (sleeve/bypass/manga)", terms=BARIATRIC_CONDITION_TERMS,
        precedence=20, substitutions=_DM2_SUGAR_SUBS, sub_negatives=_DM2_SUGAR_NEGATIVES,
        prompt_block=(
            "🔻 REGLA CLÍNICA — CIRUGÍA BARIÁTRICA (sleeve/bypass/manga, fase de MANTENIMIENTO / dieta general, "
            ">6 meses post-op; prevención de SÍNDROME DE DUMPING y obstrucción del pouch):\n"
            "   • FASE MANTENIMIENTO: el paciente tolera DIETA GENERAL (vegetales y frutas enteras, fibra, granos "
            "integrales, leguminosas como acompañante, pescado/mariscos, especias suaves) — NO es dieta líquida/"
            "puré de fase temprana. El control va en CANTIDAD y AZÚCAR, no en prohibir alimentos generales.\n"
            "   • POUCH PEQUEÑO: cada comida cabe en ~150-200 mL. Comidas PRINCIPALES ≤200 g de sólidos; "
            "MERIENDAS ≤150 g. Ninguna comida voluminosa. Total del día 1400-1700 kcal en 6 comidas; NUNCA "
            "por debajo de 1400 kcal (desnutrición).\n"
            "   • PROTEÍNA PRIMERO E INVIOLABLE: piso 60-80 g/día (≈15-20 g por comida principal, 8-12 g por "
            "merienda). Sirve la proteína blanda (huevo, yogurt griego, pollo/res guisada y desmenuzada, "
            "pescado, queso) ANTES que cualquier almidón. Si no cabe todo, se sacrifica el almidón, NUNCA la "
            "proteína; si el piso no se alcanza, añade otra fuente proteica pequeña en vez de almidón.\n"
            "   • PROHIBIDO TODO AZÚCAR SIMPLE/LIBRE (dispara dumping): miel, azúcar, panela, melaza, sirope, "
            "leche condensada, jugos/refrescos/gaseosas/malta, frutas en almíbar y dulces (habichuelas con "
            "dulce, dulce de leche/coco/batata). Endulza con estevia, canela o fruta entera medida.\n"
            "   • SIN BEBIDAS CON LA COMIDA: no incluyas líquidos junto a sólidos en la misma comida; agua/té/"
            "café sin azúcar van ENTRE comidas. NADA de gaseosas ni alcohol. NADA de agua de coco/jugo (azúcar "
            "líquido → dumping).\n"
            "   • TOPES DUROS DE PORCIÓN POR COMIDA: queso ≤30 g (1 lonja entera, MÁX 1 comida con queso/día); "
            "yogurt griego natural sin azúcar ≤120 g; MÁX 2 porciones lácteas/día en total (queso+yogurt+leche); "
            "huevo 1-2 uds; pollo 60-90 g cocido; res 70 g; pescado 70-90 g; UN solo almidón pequeño (arroz/"
            "pasta ~45-50 g o víver hervido ~60 g) SOLO en almuerzo/cena; fruta 60-80 g, máx 2/día. En el plato "
            "criollo arroz+habichuela cuentan como UN almidón (no ambos en porción llena).\n"
            "   • CARGA GLUCÉMICA BAJA: ~≤30 g de carbohidrato por comida. NO combines fruta + almidón + lácteo "
            "en la misma comida; la fruta va con proteína, nunca sola (toda merienda DEBE llevar ≥6 g de "
            "proteína — prohibida la merienda de solo fruta o solo almidón).\n"
            "   • MERIENDA NOCTURNA = SOLO proteína láctea ligera o huevo (yogurt griego 120 g, 1 huevo duro, "
            "queso ≤30 g o leche descremada 200 ml). PROHIBIDO pescado, carne, almidón y grasa densa (maní/"
            "aguacate) de noche.\n"
            "   • COCCIÓN BLANDA Y HÚMEDA: horno/hervido/plancha/vapor/guisado; carnes guisadas o desmenuzadas, "
            "nunca secas/correosas (obstruyen el pouch); víveres hervidos. PROHIBIDAS las frituras (maduros "
            "fritos, tostones, pollo frito, chicharrón). Aceite ≤1 cdita (5 g) por comida.\n"
            "   • PORCIONES REALISTAS Y POSIBLES: usa unidades enteras y gramos/tazas claros (ej. '1 lonja de "
            "queso (30 g)', '4 fresas enteras', '120 g de yogurt'). PROHIBIDAS las fracciones absurdas "
            "('5¼ lonjas', '2.5 fresas', '0.25 cda') y las instrucciones imposibles ('fresa solo la cáscara').\n"
            "   • El paciente bariátrico requiere SUPLEMENTACIÓN (multivitamínico, B12, hierro, calcio citrato, "
            "vitamina D); NO intentes cubrir déficits con alimentos en exceso de volumen. Plan ORIENTATIVO que "
            "no sustituye al equipo bariátrico."),
    ),
)

_RULES_BY_ID = {r.id: r for r in CONDITION_RULES}


def _norm_conditions(form_data) -> list:
    """Lista normalizada (lower + strip_accents, sin sentinel) de las condiciones del form."""
    if not isinstance(form_data, dict):
        return []
    try:
        from constants import strip_accents as _sa
    except Exception:
        _sa = lambda x: x  # noqa: E731
    raw = form_data.get("medicalConditions") or form_data.get("medical_conditions") or []
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for c in raw:
        s = str(c).strip().lower()
        if not s or s in _MEDICAL_NONE_SENTINELS:
            continue
        try:
            s = _sa(s)
        except Exception:
            pass
        out.append(s)
    return out


def detect_active_rules(form_data) -> list:
    """Reglas activas para el perfil, ordenadas por precedencia (seguridad primero)."""
    conds = _norm_conditions(form_data)
    if not conds:
        return []
    active = [r for r in CONDITION_RULES
              if any(any(t in c for t in r.terms) for c in conds)]
    return sorted(active, key=lambda r: r.precedence)


def build_condition_prompt(form_data) -> str:
    """Bloque de reglas nutricionales por condición (registry-driven) + nota de comorbilidad."""
    active = detect_active_rules(form_data)
    if not active:
        return ""
    blocks = [r.prompt_block for r in active]
    ids = {r.id for r in active}
    if "dm2" in ids and "renal" in ids:
        blocks.append(
            "⚖️ PRECEDENCIA CLÍNICA — DIABETES + ENFERMEDAD RENAL JUNTAS (diabético-nefropatía):\n"
            "   • La regla RENAL MANDA sobre el target de fibra/leguminosas de la diabetes: las leguminosas y "
            "granos integrales (altos en potasio/fósforo) se MODERAN — prioriza vegetales BAJOS en potasio.\n"
            "   • Mantén proteína MODERADA (renal) y sodio bajo; NO subas la carga de carbohidrato. El balance "
            "fino lo define el nefrólogo.")
    # [P2-RENAL-HTA-POTASSIUM-PROMPT · 2026-06-19] (audit fresco P2, cluster S1) Árbitro renal+HTA del potasio.
    # El prompt_block de HTA pide "POTASIO/MAGNESIO/CALCIO ALTOS" (patrón DASH) mientras el renal pide MODERAR
    # potasio/fósforo → directivas contradictorias sin resolución (la rama genérica de abajo solo arbitra
    # sodio/proteína/grasa-saturada, nunca el potasio). En ERC la dirección segura es MODERAR el potasio
    # (hiperkalemia → arritmia): la regla renal MANDA. Espejo del bloque dm2+renal. El panel determinista ya
    # gatea el piso DASH-K en ERC (P2-RENAL-HTA-POTASSIUM-GUARD); esto cierra la mitad prompt al generador.
    # `elif` para preservar el encadenamiento: dm2+renal (que también modera potasio) tiene precedencia; un
    # perfil con las tres (dm2+hta+renal) cae en la rama dm2+renal, que ya modera potasio/leguminosas.
    elif "hta" in ids and "renal" in ids:
        blocks.append(
            "⚖️ PRECEDENCIA CLÍNICA — HIPERTENSIÓN + ENFERMEDAD RENAL JUNTAS:\n"
            "   • La regla RENAL MANDA sobre el potasio/magnesio ALTOS del patrón DASH de la hipertensión: en "
            "enfermedad renal el potasio se MODERA (riesgo de hiperkalemia → arritmia) — NO maximices guineo, "
            "aguacate, leguminosas ni vegetales de hoja verde; mantén porciones moderadas y parejas.\n"
            "   • SÍ conserva el SODIO BAJO de DASH (beneficia ambas condiciones). El balance fino del potasio "
            "lo define el nefrólogo.")
    elif len(active) >= 2:
        labels = ", ".join(r.label for r in active)
        blocks.append(
            f"⚖️ PRECEDENCIA CLÍNICA — CONDICIONES MÚLTIPLES ({labels}): cuando dos reglas chocan, gana la MÁS "
            "RESTRICTIVA en dirección de seguridad (el techo más bajo de sodio, proteína o grasa saturada). "
            "Este plan es orientativo; el balance individual lo define tu profesional de salud.")
    return ("\n--- REGLAS NUTRICIONALES POR CONDICIÓN MÉDICA (DETERMINISTAS, CITABLES) ---\n"
            + "\n\n".join(blocks)
            + "\n----------------------------------------\n")


# [P2-13 · 2026-06-16] (gap-audit P2-13) Las subs por condición/alérgeno reemplazaban a proteína ANIMAL
# (Pechuga de pollo / Filete de pescado blanco) sin mirar dietType → inyectaban animal en un plan vegano.
# Redirige el reemplazo a una proteína vegetal que RESUELVE al catálogo (Lentejas, verificado). Espejo
# mínimo de graph_orchestrator._canonicalize_diet_type (duplicado a propósito — import circular: condition_rules
# es importado POR graph_orchestrator). Solo redirige CARNE/PESCADO; lácteo/huevo (vegetariano los permite)
# NO se tocan — el residual lácteo en vegano lo cubre el backstop P1-DIET-HARD-GUARD → fallback diet-aware.
_VEG_PROTEIN_REDIRECT = {"Pechuga de pollo": "Lentejas", "Filete de pescado blanco": "Lentejas"}
_PESC_PROTEIN_REDIRECT = {"Pechuga de pollo": "Filete de pescado blanco"}  # pescetariano: permite pescado


def _canon_diet(diet) -> str:
    """[P2-13] dietType → {vegan|vegetarian|pescatarian|balanced} (EN + ES masc/fem). Acent-stripped."""
    try:
        from constants import strip_accents as _sa
    except Exception:
        _sa = lambda x: x  # noqa: E731
    d = _sa(str(diet or "").strip().lower())
    if d in ("vegano", "vegan", "vegana"):
        return "vegan"
    if d in ("vegetariano", "vegetarian", "vegetariana", "ovolactovegetariano"):
        return "vegetarian"
    if d in ("pescetariano", "pescatariano", "pescatarian", "pescetarian", "pescetariana", "pescatariana"):
        return "pescatarian"
    return "balanced"


def _redirect_replacement_for_diet(repl: str, diet_canon: str, allergen_cat: str = None) -> str:
    """[P2-13] Redirige un reemplazo animal a uno compatible con la dieta. Vegan/vegetarian → vegetal;
    pescatarian → solo carne-de-tierra→pescado; balanced → sin cambio.
    [P2-13 review-fix] `allergen_cat`: si la sustitución es por ALERGIA a pescado y la dieta es pescetariana,
    el target NO puede ser pescado (lo prohíbe la alergia) NI carne de tierra (lo prohíbe la dieta) → cae a
    vegetal, como vegano. Sin esto, pescetariano+alergia-pescado redirigía pollo→pescado (reintroduce el alérgeno)."""
    if allergen_cat == "fish" and diet_canon == "pescatarian":
        return _VEG_PROTEIN_REDIRECT.get(repl, repl)
    if diet_canon in ("vegan", "vegetarian"):
        return _VEG_PROTEIN_REDIRECT.get(repl, repl)
    if diet_canon == "pescatarian":
        return _PESC_PROTEIN_REDIRECT.get(repl, repl)
    return repl


def collect_substitutions(form_data, diet_type=None) -> list:
    """Sustituciones deterministas de ingredientes activas para el perfil, en orden de precedencia.
    Cada item: {tokens, replacement, label, negatives, condition, preserve_qty}. El guard las aplica
    en un solo pase. Tolera filas legacy de 3 elementos (preserve_qty → False por defecto).
    [P2-13] `diet_type` (opcional) redirige reemplazos animales a proteína vegetal/pescado para veg*."""
    _dc = _canon_diet(diet_type) if diet_type else "balanced"
    out = []
    for r in detect_active_rules(form_data):
        for sub in (r.substitutions or ()):
            tokens, repl, label = sub[0], sub[1], sub[2]
            preserve_qty = bool(sub[3]) if len(sub) > 3 else False
            repl = _redirect_replacement_for_diet(repl, _dc)  # [P2-13] diet-aware redirect
            out.append({"tokens": tokens, "replacement": repl, "label": label,
                        "negatives": r.sub_negatives or (), "condition": r.id,
                        "preserve_qty": preserve_qty})
    return out


def active_condition_labels(form_data) -> list:
    return [r.label for r in detect_active_rules(form_data)]


# ════════════════════════════════════════════════════════════════════════════════════════════════
# [P0-ALLERGEN-SUBS · 2026-06-14] Sustitución determinista de ALÉRGENOS IgE declarados
# ════════════════════════════════════════════════════════════════════════════════════════════════
# Cierra el gap del audit clínico (2026-06-14): las alergias declaradas (`form_data['allergies']`)
# dependían del prompt + el backstop romo `_scan_allergen_violations` (que NUKE el plan entero a
# fallback matemático). Este registro hace QUIRÚRGICA la defensa: para los alérgenos con un reemplazo
# SEGURO que RESUELVE al catálogo `master_ingredients`, sustituye el ingrediente ofensor in-place
# (preserva cantidad + recalcula macros por delta, vía el mismo motor que las sustituciones por
# condición), conservando el plan rico del LLM. `_scan_allergen_violations` queda como red de
# seguridad post-swap: cualquier residual sigue escalando a rechazo crítico → fallback.
#
# DECISIÓN HONESTA (documentada): lácteos, huevo, maní y frutos secos NO se sustituyen aquí — el
# catálogo es-DO NO tiene un target libre del alérgeno que resuelva (no hay leche/queso vegetal, ni
# mantequilla de semillas). Para esos, el path crítico→fallback existente sigue EXCLUYÉNDOLOS (cero
# regresión). Habilitar lácteos/huevo requeriría filas nuevas en el catálogo (palanca de DATOS).
#
# TOKENS accent-free + lowercase (se matchean contra `strip_accents(ingrediente).lower()`) y
# ESTRECHOS (lección del bug 'soya'/'pana'): nada de raíces ambiguas — 'pan de agua' (NO 'pan'
# desnudo, que matchea 'pana'=fruta de pan); 'tofu'/'salsa de soya' (NO 'soya' desnudo). Lo que un
# token estrecho no atrape (p.ej. 'pan' a secas) lo recoge el backstop `_scan_allergen_violations`.
# Cada fila: (tokens, reemplazo, etiqueta, preserve_qty). Los reemplazos son nombres del catálogo
# (con acentos) → resuelven en el delta de macros. preserve_qty=True para proteínas/almidones-staple.

_ALLERGEN_FISH_SUBS = (
    (("pescado", "bacalao", "atun", "salmon", "tilapia", "mero", "chillo", "sardina",
      "merluza", "carite", "dorado", "filete de pescado", "arenque", "arenque salado"),  # [P2-12] +arenque
     "Pechuga de pollo", "pescado", True),
)
_ALLERGEN_SHELLFISH_SUBS = (
    (("camaron", "langosta", "langostino", "cangrejo", "marisco", "pulpo", "calamar",
      "almeja", "ostra", "lambi", "gamba"),
     "Pechuga de pollo", "mariscos", True),
)
_ALLERGEN_SOY_SUBS = (
    # condimento (qty pequeña, 'al gusto') → reemplazo bare, NO preserva prefijo de cantidad.
    (("salsa de soya", "salsa de soja", "teriyaki"), "Limón con especias", "salsa de soya", False),
    # proteína de soya → swap a pollo preservando el peso comprable.
    (("tofu", "edamame", "proteina de soya", "carne de soya", "soja texturizada", "soya texturizada"),
     "Pechuga de pollo", "soya (tofu/edamame)", True),
)
_ALLERGEN_GLUTEN_SUBS = (
    (("harina de trigo",), "Harina de maíz precocida", "harina de trigo", True),
    (("pan de agua", "pan integral", "pan blanco", "pan de trigo", "pan tostado", "pan sandwich",
      "pan pita", "tostada", "tostadas"),
     "Casabe", "pan de trigo", True),
    (("pasta integral", "espagueti", "macarron", "coditos", "fideo", "lasana", "tallarin",
      "pasta de trigo", "penne", "ravioli", "ñoqui", "noqui"),
     "Arroz blanco", "pasta de trigo", True),
    # [P3-GALLETA-ARROZ-REMOVE · 2026-06-22] Target GF cambiado "Galletas de arroz"→"Casabe":
    # el owner confirmó que La Sirena no vende galletas de arroz → fuera del catálogo. Casabe
    # (cracker de yuca, GF nativo dominicano, en catálogo) es el reemplazo crujiente disponible
    # (ya se usa para el swap de pan GF arriba).
    (("galleta de soda", "galletas de soda", "galleta de trigo", "galletas de trigo"),
     "Casabe", "galletas de trigo", True),
    # [P0-ALLERGEN-SUBS live-fix · 2026-06-14] La avena es naturalmente sin gluten PERO el revisor
    # médico la rechaza por contaminación cruzada (estándar conservador) → sin este swap, un plan con
    # avena para un alérgico a gluten cae a fallback. Swap a Quinoa (GF nativo, en catálogo, alto en
    # proteína → ideal para volumen). 'avena' no es substring de 'avellana' (frutos secos) ni de otro
    # ingrediente → token seguro. Hallazgo de la prueba en vivo P0-ALLERGEN-SUBS.
    (("avena", "hojuelas de avena", "harina de avena", "salvado de avena"),
     "Quinoa", "avena (gluten por contaminación)", True),
    (("cebada", "centeno", "cuscus", "bulgur", "germen de trigo", "salvado de trigo", "tortilla de trigo",
      "tortilla de harina", "tortilla integral"),
     "Arroz blanco", "cereal con gluten", True),
)
# Vetos (accent-free): evitan swappear un alimento que YA es libre del alérgeno (queda SAFE igual,
# pero no lo cambiamos innecesariamente). Solo gluten los necesita (variantes GF en el catálogo es-DO).
_ALLERGEN_GLUTEN_NEGATIVES = ("sin gluten", "libre de gluten", "gluten free", "de maiz", "de arroz",
                              "de yuca", "de almendra", "casabe", "pana")

# Detección: alergia DECLARADA (texto del form, strip_accents+lower) → categoría con tabla de swaps.
# Términos accent-free; matching `term in declared` (substring) — over-detección es la dirección SEGURA.
_ALLERGEN_DETECT = {
    "fish": ("pescado", "fish", "atun", "salmon", "bacalao", "tilapia", "sardina"),
    "shellfish": ("marisco", "camaron", "langosta", "cangrejo", "shellfish", "crustaceo", "molusco"),
    "soy": ("soya", "soja", "soy", "tofu", "edamame"),
    "gluten": ("gluten", "trigo", "wheat", "celiac", "celiaqu", "tacc"),
}
_ALLERGEN_SUBS_BY_CAT = {
    "fish": _ALLERGEN_FISH_SUBS,
    "shellfish": _ALLERGEN_SHELLFISH_SUBS,
    "soy": _ALLERGEN_SOY_SUBS,
    "gluten": _ALLERGEN_GLUTEN_SUBS,
}
_ALLERGEN_NEGATIVES_BY_CAT = {
    "gluten": _ALLERGEN_GLUTEN_NEGATIVES,
}


def collect_allergen_substitutions(form_data, diet_type=None) -> list:
    """[P0-ALLERGEN-SUBS · 2026-06-14] Sustituciones deterministas para los alérgenos IgE DECLARADOS
    (`form_data['allergies']`) que tienen un reemplazo seguro que RESUELVE al catálogo es-DO
    (fish/shellfish/soy/gluten). Mismo shape que `collect_substitutions` → reusa el motor compartido
    `_apply_substitutions_core`. Cada item: {tokens, replacement, label, negatives, condition,
    preserve_qty}. Lácteos/huevo/maní/frutos secos NO se incluyen (sin target que resuelva) → siguen
    por el path crítico→fallback. Sentinel: P0-ALLERGEN-SUBS.
    [P2-13] `diet_type` (opcional) redirige el reemplazo animal (pollo) a vegetal/pescado para veg*."""
    if not isinstance(form_data, dict):
        return []
    _dc = _canon_diet(diet_type) if diet_type else "balanced"  # [P2-13] diet-aware redirect
    try:
        from constants import strip_accents as _sa
    except Exception:
        _sa = lambda x: x  # noqa: E731
    raw = form_data.get("allergies") or []
    if isinstance(raw, str):
        raw = [raw]
    declared = []
    for a in raw:
        s = str(a).strip().lower()
        if not s or s in _MEDICAL_NONE_SENTINELS:
            continue
        try:
            s = _sa(s)
        except Exception:
            pass
        declared.append(s)
    if not declared:
        return []
    out = []
    for cat, terms in _ALLERGEN_DETECT.items():
        if not any(t in d for t in terms for d in declared):
            continue
        negs = _ALLERGEN_NEGATIVES_BY_CAT.get(cat, ())
        for sub in _ALLERGEN_SUBS_BY_CAT[cat]:
            tokens, repl, label = sub[0], sub[1], sub[2]
            preserve_qty = bool(sub[3]) if len(sub) > 3 else False
            repl = _redirect_replacement_for_diet(repl, _dc, allergen_cat=cat)  # [P2-13] diet+allergen-aware
            out.append({"tokens": tokens, "replacement": repl, "label": label,
                        "negatives": negs, "condition": f"allergen:{cat}",
                        "preserve_qty": preserve_qty})
    return out


def active_allergen_labels(form_data) -> list:
    """Etiquetas es-DO de las categorías de alérgeno con swap determinista activo para el perfil."""
    seen, out = set(), []
    for s in collect_allergen_substitutions(form_data):
        cat = s["condition"]
        if cat not in seen:
            seen.add(cat)
            out.append(s["label"])
    return out
