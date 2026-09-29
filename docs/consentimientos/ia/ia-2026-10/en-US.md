# Permiso para la IA de terceros · `ia-2026-10` · en-US

[P1-PLAN-LOTE-844 · ronda 1] Texto EXACTO de la hoja «Tus datos y la IA» que la app muestra en `en-US` para la versión
`ia-2026-10`. Es lo que la persona acepta (art. 7.1 del RGPD: el consentimiento tiene que poder demostrarse): la fila de
`user_consents` guarda la versión y el `text_sha256` de este texto.

- Versión: `ia-2026-10`
- Idioma: `en-US`
- SHA-256 (`text_sha256`): `a4e646fa4e6640f13515a2501b746abbc709cfd9c9a58fd0cbc087ac1a17cff2`
- Fuente en la app: `frontend/src/consent/textoDeLaHoja.js` (los bloques, por `t()`) y el catálogo del idioma
  (`frontend/src/i18n/locales/<locale>.json`; `es-DO` es el texto base, sin catálogo). `textoPlanoDeLaHoja` une los
  bloques con un salto de línea: ese texto, en UTF-8, es el que se firma.
- Guardián: `frontend/src/__tests__/lote844.textos.test.js` fija el SHA-256 de cada idioma y comprueba que este fichero
  coincide con el catálogo. Cambiar el texto obliga a decidir si se sube la versión (y a escribir `ia-AAAA-MM/`).

El texto va entre las dos marcas; los dos saltos de línea que lo separan de ellas no forman parte de él.

<!-- texto:inicio -->
Your data and AI
To build your plan, answer you in the coach and analyze your photos, Bioboros sends some of your data to external artificial intelligence providers. We don't use it for advertising and we don't sell it.
DeepSeek
Hangzhou DeepSeek, People's Republic of China
Receives your health profile (age, weight, conditions, medications, allergies, pregnancy, religious dietary restriction), your preferences, your name, your messages with the coach and what you log in your diary.
To generate parts of your plan, answer you in the coach and estimate what you log.
OpenAI
US
Receives your health profile, your preferences and, when needed, part of the conversation.
To generate days of your plan, check that it's safe and act as a backup.
Google Gemini
US
Receives the photos you scan or send to the chat, what you write to clarify them and your country; in voice mode, the text it reads aloud.
To recognize the food and generate the voice.
Cohere
Servers mainly in the US
Receives a summary of your profile (goal, allergies, conditions), the notes the coach remembers about you and the description of your meals.
So the coach can find what's relevant.
Transfer to China
DeepSeek processes your data in the People's Republic of China. The European Union does not recognize China as providing an adequate level of protection, and we have not signed contractual clauses with DeepSeek: your data could be accessible to that country's authorities, and it would be harder for you to exercise your rights there.
I agree that Bioboros may use my health data to create my plan and provide the coach, and send it to the AI providers listed.
I agree that my data may be sent to DeepSeek, in China, aware of the risks described.
Optional
Help us improve: usage analytics, with no health data.
Bioboros does not replace your doctor or dietitian. Consult them before changing your diet, especially if you have a medical condition, take medication, are pregnant or breastfeeding, or have had bariatric surgery.
You can withdraw your permission anytime in Settings → Privacy → Third-party AI. Withdrawing it stops new transfers.
Privacy Policy
AI use
<!-- texto:fin -->
