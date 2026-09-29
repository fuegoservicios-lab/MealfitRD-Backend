# Permiso para la IA de terceros · `ia-2026-10` · fr-FR

[P1-PLAN-LOTE-844 · ronda 1] Texto EXACTO de la hoja «Tus datos y la IA» que la app muestra en `fr-FR` para la versión
`ia-2026-10`. Es lo que la persona acepta (art. 7.1 del RGPD: el consentimiento tiene que poder demostrarse): la fila de
`user_consents` guarda la versión y el `text_sha256` de este texto.

- Versión: `ia-2026-10`
- Idioma: `fr-FR`
- SHA-256 (`text_sha256`): `14718025b67df2dcbc028dbee56a765d5d740fa74de40fb975afd89e4f4807f8`
- Fuente en la app: `frontend/src/consent/textoDeLaHoja.js` (los bloques, por `t()`) y el catálogo del idioma
  (`frontend/src/i18n/locales/<locale>.json`; `es-DO` es el texto base, sin catálogo). `textoPlanoDeLaHoja` une los
  bloques con un salto de línea: ese texto, en UTF-8, es el que se firma.
- Guardián: `frontend/src/__tests__/lote844.textos.test.js` fija el SHA-256 de cada idioma y comprueba que este fichero
  coincide con el catálogo. Cambiar el texto obliga a decidir si se sube la versión (y a escribir `ia-AAAA-MM/`).

El texto va entre las dos marcas; los dos saltos de línea que lo separan de ellas no forman parte de él.

<!-- texto:inicio -->
Vos données et l'IA
Pour créer votre plan, vous répondre dans le coach et analyser vos photos, Bioboros envoie certaines de vos données à des fournisseurs externes d'intelligence artificielle. Nous ne les utilisons pas pour la publicité et nous ne les vendons pas.
DeepSeek
Hangzhou DeepSeek, République populaire de Chine
Reçoit votre profil de santé (âge, poids, pathologies, médicaments, allergies, grossesse, restriction alimentaire religieuse), vos préférences, votre nom, vos messages avec le coach et ce que vous notez dans le journal.
Pour générer des parties de votre plan, vous répondre dans le coach et estimer ce que vous notez.
OpenAI
États-Unis
Reçoit votre profil de santé, vos préférences et, si nécessaire, une partie de la conversation.
Pour générer des jours de votre plan, vérifier qu'il est sûr et servir de solution de secours.
Google Gemini
États-Unis
Reçoit les photos que vous scannez ou envoyez dans le chat, ce que vous écrivez pour les préciser et votre pays ; en mode vocal, le texte qu'il lit à voix haute.
Pour reconnaître les aliments et générer la voix.
Cohere
Serveurs principalement aux États-Unis
Reçoit un résumé de votre profil (objectif, allergies, pathologies), les notes que le coach garde sur vous et la description de vos repas.
Pour que le coach retrouve ce qui est pertinent.
Transfert vers la Chine
DeepSeek traite vos données en République populaire de Chine. L'Union européenne ne reconnaît pas à la Chine un niveau de protection adéquat et nous n'avons pas signé de clauses contractuelles avec DeepSeek : vos données pourraient être accessibles aux autorités de ce pays et il vous serait plus difficile d'y exercer vos droits.
J'accepte que Bioboros utilise mes données de santé pour créer mon plan et me proposer le coach, et qu'il les envoie aux fournisseurs d'IA indiqués.
J'accepte que mes données soient envoyées à DeepSeek, en Chine, en connaissance des risques décrits.
Facultatif
Aidez-nous à nous améliorer : statistiques d'utilisation, sans données de santé.
Bioboros ne remplace pas votre médecin ni votre nutritionniste. Consultez-les avant de modifier votre alimentation, surtout si vous avez une pathologie, prenez des médicaments, êtes enceinte ou allaitez, ou avez subi une chirurgie bariatrique.
Vous pouvez retirer votre autorisation à tout moment dans Réglages → Confidentialité → IA tierce. La retirer arrête les nouveaux envois.
Politique de confidentialité
Utilisation de l'IA
<!-- texto:fin -->
