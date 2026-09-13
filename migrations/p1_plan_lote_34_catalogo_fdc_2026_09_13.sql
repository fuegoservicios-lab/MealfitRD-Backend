-- [P1-PLAN-LOTE-34 · 2026-09-13] F6: el barrido descripción-USDA ↔ nombre sobre las 288 filas con fdc_id.
--
-- Un fdc_id es una AFIRMACIÓN («esta fila ES ese alimento de USDA»; P1-PROVENANCE-TRUTHFUL). La auditoría del 19-ago
-- cerró los ids compartidos y dejó escrito lo que no veía: un id ÚNICO mal apuntado. El barrido
-- (scripts/catalog_fdc_sweep.py, artefacto scripts/data/catalog_fdc_sweep_2026_09_13.json) comparó
-- identidad (nombre frente a la descripción de USDA) y valores (P/C/G/kcal) fila a fila: 22 ids apuntaban a OTRO
-- alimento con los valores del catálogo intactos (Tamarindo → verdolaga, Hígado de res → T-bone, Mapuey → col rizada
-- cocida, Leche de almendras → un experimento con tomates…). Cada corrección se VERIFICÓ contra USDA antes de
-- escribirla (P/C/G dentro de tolerancia y kcal igual o por Atwater general).
--
-- Lo que se escribe: el fdc_id correcto (22); proxies no declarados que pasan a declarados y valores sin fuente que
-- dejan de fingir una (18: fdc_id NULL, 'manual', la traza en nutrition_source_ref); 17 notas de kcal por
-- Atwater con el id correcto; 2 glosas en inglés que nombraban otra fruta/hierba.
--
-- Lo que NO se toca: ningún VALOR nutricional. Los micros de las filas corregidas NO son los del id equivocado
-- (comparados micro a micro: 0 de 19 filas los heredó), así que no hay nada que re-traer.
--
-- Idempotente: cada UPDATE filtra por `name` exacto (y la glosa, por su valor anterior).

-- Tamarindo: 169274 «Purslane, raw» → 167763 «Tamarinds, raw» · Tamarinds, raw
UPDATE public.master_ingredients SET fdc_id = 167763, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:169274 «Purslane, raw»' WHERE name = 'Tamarindo';

-- Ciruela pasa: 168168 «Rhubarb, frozen, uncooked» → 168162 «Plums, dried (prunes), uncooked» · Plums, dried (prunes), uncooked
UPDATE public.master_ingredients SET fdc_id = 168162, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:168168 «Rhubarb, frozen, uncooked»' WHERE name = 'Ciruela pasa';

-- Kéfir: 2257046 «Oat milk, unsweetened, plain, refrigerated» → 170904 «Kefir, lowfat, plain, LIFEWAY» · Kefir, lowfat, plain
UPDATE public.master_ingredients SET fdc_id = 170904, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:2257046 «Oat milk, unsweetened, plain, refrigerated»' WHERE name = 'Kéfir';

-- Mantequilla de almendras: 170579 «Nuts, coconut meat, dried (desiccated), toasted» → 168588 «Nuts, almond butter, plain, without salt added» · Nuts, almond butter, plain, without salt added (el sodio del catálogo, 7 mg, es el sin sal)
UPDATE public.master_ingredients SET fdc_id = 168588, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:170579 «Nuts, coconut meat, dried (desiccated), toasted»' WHERE name = 'Mantequilla de almendras';

-- Cereza maraschino: 173952 «Carissa, (natal-plum), raw» → 167766 «Maraschino cherries, canned, drained» · Maraschino cherries, canned, drained
UPDATE public.master_ingredients SET fdc_id = 167766, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:173952 «Carissa, (natal-plum), raw»' WHERE name = 'Cereza maraschino';

-- Chivo: 174375 «Lamb, loin, separable lean and fat, trimmed to 1/8" fat, choice, cooked, broiled» → 175303 «Game meat, goat, raw» · Game meat, goat, raw
UPDATE public.master_ingredients SET fdc_id = 175303, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:174375 «Lamb, loin, separable lean and fat, trimmed to 1/8" fat, choice, cooked, broiled»' WHERE name = 'Chivo';

-- Cebolla en polvo: 171324 «Spices, fenugreek seed» → 171327 «Spices, onion powder» · Spices, onion powder
UPDATE public.master_ingredients SET fdc_id = 171327, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:171324 «Spices, fenugreek seed»' WHERE name = 'Cebolla en polvo';

-- Mero: 173690 «Fish, salmon, pink, canned, total can contents» → 171962 «Fish, grouper, mixed species, raw» · Fish, grouper, mixed species, raw
UPDATE public.master_ingredients SET fdc_id = 171962, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:173690 «Fish, salmon, pink, canned, total can contents»' WHERE name = 'Mero';

-- Cundeamor: 169214 «Corn, sweet, yellow, canned, whole kernel, drained solids» → 168393 «Balsam-pear (bitter gourd), pods, raw» · Balsam-pear (bitter gourd), pods, raw
UPDATE public.master_ingredients SET fdc_id = 168393, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:169214 «Corn, sweet, yellow, canned, whole kernel, drained solids»' WHERE name = 'Cundeamor';

-- Conejo: 174374 «Lamb, leg, shank half, separable lean and fat, trimmed to 1/8" fat, choice, raw» → 172521 «Game meat, rabbit, domesticated, composite of cuts, raw» · Game meat, rabbit, domesticated, composite of cuts, raw
UPDATE public.master_ingredients SET fdc_id = 172521, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:174374 «Lamb, leg, shank half, separable lean and fat, trimmed to 1/8" fat, choice, raw»' WHERE name = 'Conejo';

-- Durazno en almíbar: 169123 «Pears, dried, sulfured, stewed, with added sugar» → 169112 «Peaches, canned, heavy syrup pack, solids and liquids» · Peaches, canned, heavy syrup pack, solids and liquids
UPDATE public.master_ingredients SET fdc_id = 169112, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:169123 «Pears, dried, sulfured, stewed, with added sugar»' WHERE name = 'Durazno en almíbar';

-- Tomillo: 170935 «Spices, sage, ground» → 170938 «Spices, thyme, dried» · Spices, thyme, dried
UPDATE public.master_ingredients SET fdc_id = 170938, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:170935 «Spices, sage, ground»' WHERE name = 'Tomillo';

-- Hígado de res: 168620 «Beef, short loin, t-bone steak, bone-in, separable lean only, trimmed to 1/8" fat, choice, cooked, grilled» → 169451 «Beef, variety meats and by-products, liver, raw» · Beef, variety meats and by-products, liver, raw
UPDATE public.master_ingredients SET fdc_id = 169451, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:168620 «Beef, short loin, t-bone steak, bone-in, separable lean only, trimmed to 1/8" fat, choice, cooked, grilled»' WHERE name = 'Hígado de res';

-- Muslo de pollo: 171077 «Chicken, broiler or fryers, breast, skinless, boneless, meat only, raw» → 173627 «Chicken, broilers or fryers, dark meat, thigh, meat only, raw» · Chicken, broilers or fryers, dark meat, thigh, meat only, raw
UPDATE public.master_ingredients SET fdc_id = 173627, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:171077 «Chicken, broiler or fryers, breast, skinless, boneless, meat only, raw»' WHERE name = 'Muslo de pollo';

-- Salmón: 175167 «Fish, salmon, Atlantic, farmed, raw» → 173686 «Fish, salmon, Atlantic, wild, raw» · Fish, salmon, Atlantic, WILD, raw (el id apuntaba al de granja: 208 kcal)
UPDATE public.master_ingredients SET fdc_id = 173686, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:175167 «Fish, salmon, Atlantic, farmed, raw»' WHERE name = 'Salmón';

-- Toronja: 174675 «Grapefruit, raw, pink and red, Florida» → 174673 «Grapefruit, raw, pink and red, all areas» · Grapefruit, raw, pink and red, all areas
UPDATE public.master_ingredients SET fdc_id = 174673, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:174675 «Grapefruit, raw, pink and red, Florida»' WHERE name = 'Toronja';

-- Pavo molido: 171506 «Turkey, Ground, cooked» → 171505 «Turkey, Ground, raw» · Turkey, Ground, RAW (el id apuntaba al cocido; el catálogo pesa en crudo)
UPDATE public.master_ingredients SET fdc_id = 171505, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:171506 «Turkey, Ground, cooked»' WHERE name = 'Pavo molido';

-- Aceite vegetal: 171426 «Oil, soybean lecithin» → 171411 «Oil, soybean, salad or cooking» · Oil, soybean, salad or cooking (el id apuntaba a la lecitina de soya)
UPDATE public.master_ingredients SET fdc_id = 171411, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:171426 «Oil, soybean lecithin» · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 884.0 kcal' WHERE name = 'Aceite vegetal';

-- Leche de avena: 2257045 «Almond milk, unsweetened, plain, refrigerated» → 2705412 «Oat milk» · Oat milk
UPDATE public.master_ingredients SET fdc_id = 2705412, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:2257045 «Almond milk, unsweetened, plain, refrigerated»' WHERE name = 'Leche de avena';

-- Leche de almendras: 1750336 «Manipulation of ZDS in tomato exposes carotenoid‐ and ABA‐specific effects on fruit development and ripening» → 174832 «Beverages, almond milk, unsweetened, shelf stable» · Beverages, almond milk, unsweetened, shelf stable
UPDATE public.master_ingredients SET fdc_id = 174832, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:1750336 «Manipulation of ZDS in tomato exposes carotenoid‐ and ABA‐specific effects on fruit development and ripening»' WHERE name = 'Leche de almendras';

-- Leche de coco: 1097542 «Soy milk» → 2705413 «Coconut milk» · Coconut milk (bebida)
UPDATE public.master_ingredients SET fdc_id = 2705413, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:1097542 «Soy milk»' WHERE name = 'Leche de coco';

-- Cangrejo: 171959 «Fish, mahimahi, raw» → 174204 «Crustaceans, crab, blue, raw» · Crustaceans, crab, blue, raw (el id apuntaba a mahimahi)
UPDATE public.master_ingredients SET fdc_id = 174204, nutrition_source = 'usda', nutrition_source_ref = 'id corregido 2026-09-13 (P1-PLAN-LOTE-34): antes usda:171959 «Fish, mahimahi, raw»' WHERE name = 'Cangrejo';

-- Percebes: los percebes no son cangrejo azul: valores copiados de él
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:174204 (proxy: Crustaceans, crab, blue, raw)' WHERE name = 'Percebes';

-- Guascas: la guasca (Galinsoga) no es diente de león: valores copiados de él
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:169226 (proxy: Dandelion greens, raw)' WHERE name = 'Guascas';

-- Huitlacoche: el huitlacoche es un hongo del maíz, no un champiñón crimini
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:168434 (proxy: Mushrooms, brown, italian, or crimini, raw)' WHERE name = 'Huitlacoche';

-- Arracacha: la arracacha no es chirivía (parsnip)
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:170417 (proxy: Parsnips, raw)' WHERE name = 'Arracacha';

-- Bacalaítos: el bacalaíto es fritura de bacalao, no corvina empanizada
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:171957 (proxy: Fish, croaker, Atlantic, cooked, breaded and fried)' WHERE name = 'Bacalaítos';

-- Leche de cabra en polvo: USDA no tiene leche de cabra en polvo: los valores son de leche de VACA entera en polvo (el id apuntaba a la descremada)
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:173454 (proxy: Milk, dry, whole, without added vitamin D)' WHERE name = 'Leche de cabra en polvo';

-- Panecillos de mantequilla: valores de masa de biscuit refrigerada, no del panecillo horneado
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:172668 (proxy: Biscuits, plain or buttermilk, refrigerated dough, higher fat)' WHERE name = 'Panecillos de mantequilla';

-- Membrillo dulce: la pasta de membrillo no es una mermelada genérica
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:169641 (proxy: Jams and preserves)' WHERE name = 'Membrillo dulce';

-- Chuleta ahumada: la chuleta ahumada no es jamón curado en rebanada
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:168283 (proxy: Pork, cured, ham, center slice, separable lean and fat, unheated)' WHERE name = 'Chuleta ahumada';

-- Alioli: el alioli se aproxima con mayonesa regular
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:171009 (proxy: Salad dressing, mayonnaise, regular)' WHERE name = 'Alioli';

-- Mapuey: el mapuey (Dioscorea trifida) se aproxima con el ñame genérico de USDA; ese id es de la fila Ñame (el id previo apuntaba a col rizada cocida)
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:170071 (proxy: Yam, raw)' WHERE name = 'Mapuey';

-- Flor de Jamaica: usda:168170 (id previo: Roselle, raw — cáliz FRESCO, 49 kcal; los valores del catálogo son de flor SECA, 329 kcal, fuente sin verificar)
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:168170 (id previo: Roselle, raw — cáliz FRESCO, 49 kcal; los valores del catálogo son de flor SECA, 329 kcal, fuente sin verificar)' WHERE name = 'Flor de Jamaica';

-- Yogur de coco: usda:2058937 (id previo mal apuntado: SUPER SWEET CORN WITH BUTTER SAUCE); valores propios sin fuente verificada
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:2058937 (id previo mal apuntado: SUPER SWEET CORN WITH BUTTER SAUCE); valores propios sin fuente verificada' WHERE name = 'Yogur de coco';

-- Harina de Negrito: usda:169706 (id previo mal apuntado: Rice, brown, medium-grain, raw); valores propios (vit. A 180: fortificada); el más cercano en USDA: 169740 Barley malt flour
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:169706 (id previo mal apuntado: Rice, brown, medium-grain, raw); valores propios (vit. A 180: fortificada); el más cercano en USDA: 169740 Barley malt flour' WHERE name = 'Harina de Negrito';

-- Guisantes secos: usda:172428 (identidad correcta: Peas, green, split, mature seeds, raw) pero los valores son de una edición anterior de SR (341 kcal, 1,16 g grasa; USDA hoy 364 / 3,89); el id previo apuntaba a lentejas cocidas (172421)
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:172428 (identidad correcta: Peas, green, split, mature seeds, raw) pero los valores son de una edición anterior de SR (341 kcal, 1,16 g grasa; USDA hoy 364 / 3,89); el id previo apuntaba a lentejas cocidas (172421)' WHERE name = 'Guisantes secos';

-- Soya texturizada: usda:174270 (id previo: Soybeans, mature seeds, raw — soya entera, 446 kcal); valores propios; el más cercano en USDA: 174275 Soy flour, defatted (la soya texturizada se hace de ella)
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:174270 (id previo: Soybeans, mature seeds, raw — soya entera, 446 kcal); valores propios; el más cercano en USDA: 174275 Soy flour, defatted (la soya texturizada se hace de ella)' WHERE name = 'Soya texturizada';

-- Costilla de cerdo: usda:167852 (id previo: Pork, fresh, shoulder, blade, boston — paleta, no costilla); valores propios; el más cercano en USDA: 167853 Pork, fresh, spareribs, raw (proteína 15,5 frente a 17,0)
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'usda:167852 (id previo: Pork, fresh, shoulder, blade, boston — paleta, no costilla); valores propios; el más cercano en USDA: 167853 Pork, fresh, spareribs, raw (proteína 15,5 frente a 17,0)' WHERE name = 'Costilla de cerdo';

-- Frijoles pintos: sinonimo:Judías pintas (usda:175199 Beans, pinto, mature seeds, raw — mismos valores; el id vive en esa fila; el id previo apuntaba a Beans, navy, cocidas, 173746)
UPDATE public.master_ingredients SET fdc_id = NULL, nutrition_source = 'manual', nutrition_source_ref = 'sinonimo:Judías pintas (usda:175199 Beans, pinto, mature seeds, raw — mismos valores; el id vive en esa fila; el id previo apuntaba a Beans, navy, cocidas, 173746)' WHERE name = 'Frijoles pintos';

-- Níspero: en RD el níspero es el zapote (Manilkara zapota): el id (Sapodilla) era correcto y el gloss no
UPDATE public.master_ingredients SET name_en = 'Sapodilla' WHERE name = 'Níspero' AND name_en = 'Loquat';

-- Cebollín: en RD el cebollín es la cebolla de verdeo: el id (spring onions or scallions) era correcto y el gloss no
UPDATE public.master_ingredients SET name_en = 'Scallions' WHERE name = 'Cebollín' AND name_en = 'Chives';

-- Aceite de oliva: kcal por Atwater general (900) frente a USDA (884); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:167737 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 884 kcal' WHERE name = 'Aceite de oliva' AND fdc_id = 167737;

-- Aceite de sésamo: kcal por Atwater general (900) frente a USDA (884); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:171016 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 884 kcal' WHERE name = 'Aceite de sésamo' AND fdc_id = 171016;

-- Aguacate: kcal por Atwater general (181) frente a USDA (167); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:171706 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 167 kcal' WHERE name = 'Aguacate' AND fdc_id = 171706;

-- Ajo en polvo: kcal por Atwater general (363.8) frente a USDA (331); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:171325 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 331 kcal' WHERE name = 'Ajo en polvo' AND fdc_id = 171325;

-- Albahaca seca: kcal por Atwater general (319.8) frente a USDA (233); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:171317 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 233 kcal' WHERE name = 'Albahaca seca' AND fdc_id = 171317;

-- Canela en polvo: kcal por Atwater general (349.5) frente a USDA (247); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:171320 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 247 kcal' WHERE name = 'Canela en polvo' AND fdc_id = 171320;

-- Limón: kcal por Atwater general (46.6) frente a USDA (30); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:168155 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 30 kcal' WHERE name = 'Limón' AND fdc_id = 168155;

-- Mantequilla: kcal por Atwater general (733.5) frente a USDA (717); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:173410 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 717 kcal' WHERE name = 'Mantequilla' AND fdc_id = 173410;

-- Mantequilla de maní: kcal por Atwater general (640.3) frente a USDA (598); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:172470 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 598 kcal' WHERE name = 'Mantequilla de maní' AND fdc_id = 172470;

-- Maní: kcal por Atwater general (630.1) frente a USDA (587); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:173806 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 587 kcal' WHERE name = 'Maní' AND fdc_id = 173806;

-- Maíz dulce en granos: kcal por Atwater general (100) frente a USDA (86); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:169998 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 86 kcal' WHERE name = 'Maíz dulce en granos' AND fdc_id = 169998;

-- Miel: kcal por Atwater general (330.8) frente a USDA (304); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:169640 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 304 kcal' WHERE name = 'Miel' AND fdc_id = 169640;

-- Orégano dominicano: kcal por Atwater general (350.1) frente a USDA (265); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:171328 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 265 kcal' WHERE name = 'Orégano dominicano' AND fdc_id = 171328;

-- Pimentón: kcal por Atwater general (388.5) frente a USDA (282); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:171329 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 282 kcal' WHERE name = 'Pimentón' AND fdc_id = 171329;

-- Pimienta negra: kcal por Atwater general (326.9) frente a USDA (251); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:170931 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 251 kcal' WHERE name = 'Pimienta negra' AND fdc_id = 170931;

-- Vainilla: kcal por Atwater general (51.2) frente a USDA (288); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:173471 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 288 kcal' WHERE name = 'Vainilla' AND fdc_id = 173471;

-- Vinagre de manzana: kcal por Atwater general (3.7) frente a USDA (21); P/C/G y el id son los de USDA
UPDATE public.master_ingredients SET nutrition_source_ref = 'usda:173469 · kcal = 4·P+4·C+9·G (Atwater general); USDA declara 21 kcal' WHERE name = 'Vinagre de manzana' AND fdc_id = 173469;

-- == Sanity 1: las 22 correcciones quedaron escritas ==========================================
DO $$
DECLARE _n int;
BEGIN
    SELECT COUNT(*) INTO _n FROM public.master_ingredients
    WHERE nutrition_source_ref LIKE 'id corregido 2026-09-13 (P1-PLAN-LOTE-34)%' AND fdc_id IS NOT NULL;
    IF _n <> 22 THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-34] % correcciones de fdc_id, esperadas 22', _n;
    END IF;
END $$;

-- == Sanity 2: ningún fdc_id compartido ==========================================================
DO $$
DECLARE _d int;
BEGIN
    SELECT COUNT(*) INTO _d FROM (SELECT fdc_id FROM public.master_ingredients WHERE fdc_id IS NOT NULL
                                  GROUP BY fdc_id HAVING COUNT(*) > 1) x;
    IF _d > 0 THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-34] % fdc_id compartidos tras la migración', _d;
    END IF;
END $$;

-- == Sanity 3: lo declarado como proxy o propio no conserva un id que lo contradiga ================
DO $$
DECLARE _m int;
BEGIN
    SELECT COUNT(*) INTO _m FROM public.master_ingredients
    WHERE name IN ('Percebes', 'Guascas', 'Huitlacoche', 'Arracacha', 'Bacalaítos', 'Leche de cabra en polvo', 'Panecillos de mantequilla', 'Membrillo dulce', 'Chuleta ahumada', 'Alioli', 'Mapuey', 'Flor de Jamaica', 'Yogur de coco', 'Harina de Negrito', 'Guisantes secos', 'Soya texturizada', 'Costilla de cerdo', 'Frijoles pintos')
      AND (fdc_id IS NOT NULL OR nutrition_source <> 'manual' OR nutrition_source_ref IS NULL);
    IF _m > 0 THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-34] % filas proxy/propias con procedencia incompleta', _m;
    END IF;
END $$;
