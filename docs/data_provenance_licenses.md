# Proveniencia y licencias de datos nutricionales

[G16/G17 · 2026-06-15] Tabla canónica de las fuentes de datos nutricionales del catálogo + validación, con
su licencia y condiciones de uso COMERCIAL. Producto de una investigación multi-agente (32 datasets
verificados contra sus fuentes reales). Evita que un audit futuro confunda una decisión de licencia con
deuda, y documenta por qué lo "dominicano puro" sigue siendo curación manual.

## Conclusión estructural

Existe una tensión irreductible: puedes tener **2 de estas 3, nunca las 3 juntas** — (a) licencia comercial
libre, (b) máquina-legible/descargable, (c) cobertura real de comida dominicana. Lo gratis-y-comercial (USDA)
NO tiene cocina criolla; lo caribeño/latino afín (INCAP, LATINFOODS, ICBF, INHA-Cuba, CFNI) es **TODO
NonCommercial o no-descargable**. **No existe** tabla de composición OFICIAL dominicana con datos per-100g.

## Fuentes EN USO (gratis + comercial-OK)

| Fuente | Uso | Licencia | Comercial | Acceso | Condición |
|---|---|---|---|---|---|
| **USDA FoodData Central** (Foundation, SR Legacy) | Base del catálogo `master_ingredients` (G17) | **CC0 1.0** (dominio público) | ✅ sin restricción | API (key gratis data.gov) + bulk CSV/JSON sin login | Atribución solo "requested", NO obligatoria |
| **USDA FNDDS** (Survey foods, platos compuestos) | Ground-truth EXTERNO no-circular para validación (G16) | Dominio público (CC0) | ✅ sin restricción | API FDC `dataType=Survey (FNDDS)` + bulk | Ninguna |
| **TACO 4ª ed** (NEPA/UNICAMP, Brasil) | Víveres compartidos (yuca/ñame/auyama) — opcional | Solo-atribución (NO NonCommercial) | ✅ con cita | Excel oficial nepa.unicamp.br | **Citar NEPA/UNICAMP TACO** obligatorio |
| **Open Food Facts** | Embutidos empacados de marca DD (salami/longaniza) — opcional | ODbL (datos) + DbCL | ✅ con atribución | dump CSV/JSONL + API sin registro | Atribución "Open Food Facts contributors"; share-alike SOLO si redistribuyes una base derivada (uso interno NO lo dispara) |

## Fuentes RECHAZADAS (por qué NO se embeben)

| Fuente | Razón |
|---|---|
| **NutriBench** (benchmark LLM-nutrición) | CC-BY-**NC**-SA → comercial=NO; + cero países caribeños/RD. Solo referencia metodológica. |
| **INCAP TCA-Centroamérica** (2ª/3ª ed) | La más afín a RD, pero `© INCAP/OPS todos los derechos reservados` (sin CC); solo PDF; software NutrINCAP de pago. |
| **FAO/INFOODS + LATINFOODS + FAO/WHO GIFT** | NonCommercial (CC-BY-NC) o términos FAO que prohíben uso comercial; LATINFOODS no tiene capítulo RD. |
| **TBCA-USP / ICBF-Colombia / INHA-Cuba / CFNI-PAHO** | NonCommercial (CC-BY-NC-ND) y/o no-descargable (solo PDF/consulta web). INHA-Cuba sería la más afín culturalmente pero su acceso/licencia no se pudo confirmar. |
| **Food.com / Recipe1M+** | Nutrición estimada (circular) o licencia research-only NonCommercial. |
| **GABAs/Pilón MSP RD** | NO es tabla de composición (son guías dietéticas, sin valores per-100g). Confirma que no existe tabla oficial DR. |

## Decisión

- **Validación (G16):** USDA FNDDS como ground-truth externo NO-circular — `scripts/build_fndds_reference.py`
  (mapea plato DR → análogo FNDDS Survey, descarga macros per-100g CC0) → `data/fndds_dish_reference.json`;
  `clinical_validation_export.py` compara el PERFIL de macros (fracción de kcal P/C/F, independiente de
  porción) de la app vs FNDDS. Requiere una `USDA_API_KEY` gratis (DEMO_KEY rate-limited es insuficiente).
- **Curación (G17):** seguir con USDA FDC (CC0). Lo criollo PURO (yautía, sancocho/mangú como platos) sigue
  siendo **curación manual trazable** — ningún dataset gratis-comercial lo cubre. Marcado `nutrition_source`
  en `master_ingredients` (usda / manual / openfoodfacts) + footer de proveniencia en el PDF.
- **NO embeber** INCAP/LATINFOODS/ICBF/TBCA/INHA/CFNI en el producto. Si se quiere su cobertura criolla
  superior, abrir un track de PERMISO comercial escrito a INCAP/OPS — decisión de producto con costo legal,
  no integración técnica.
