# Permiso para la IA de terceros · `ia-2026-10` · pt-BR

[P1-PLAN-LOTE-844 · ronda 1] Texto EXACTO de la hoja «Tus datos y la IA» que la app muestra en `pt-BR` para la versión
`ia-2026-10`. Es lo que la persona acepta (art. 7.1 del RGPD: el consentimiento tiene que poder demostrarse): la fila de
`user_consents` guarda la versión y el `text_sha256` de este texto.

- Versión: `ia-2026-10`
- Idioma: `pt-BR`
- SHA-256 (`text_sha256`): `181052cb0ec126be4c94a557bd7a955d3d49314bb74bd2e82133042617a6b695`
- Fuente en la app: `frontend/src/consent/textoDeLaHoja.js` (los bloques, por `t()`) y el catálogo del idioma
  (`frontend/src/i18n/locales/<locale>.json`; `es-DO` es el texto base, sin catálogo). `textoPlanoDeLaHoja` une los
  bloques con un salto de línea: ese texto, en UTF-8, es el que se firma.
- Guardián: `frontend/src/__tests__/lote844.textos.test.js` fija el SHA-256 de cada idioma y comprueba que este fichero
  coincide con el catálogo. Cambiar el texto obliga a decidir si se sube la versión (y a escribir `ia-AAAA-MM/`).

El texto va entre las dos marcas; los dos saltos de línea que lo separan de ellas no forman parte de él.

<!-- texto:inicio -->
Seus dados e a IA
Para criar seu plano, responder você no coach e analisar suas fotos, o Bioboros envia alguns dos seus dados a provedores externos de inteligência artificial. Não os usamos para publicidade nem os vendemos.
DeepSeek
Hangzhou DeepSeek, República Popular da China
Recebe seu perfil de saúde (idade, peso, condições, medicamentos, alergias, gravidez, restrição alimentar religiosa), suas preferências, seu nome, suas mensagens com o coach e o que você anota no diário.
Para gerar partes do seu plano, responder você no coach e estimar o que você anota.
OpenAI
EUA
Recebe seu perfil de saúde, suas preferências e, quando necessário, parte da conversa.
Para gerar dias do seu plano, revisar se é seguro e servir de reserva.
Google Gemini
EUA
Recebe as fotos que você escaneia ou envia ao chat, o que você escreve para esclarecê-las e seu país; no modo voz, o texto que lê em voz alta.
Para reconhecer os alimentos e gerar a voz.
Cohere
Servidores principalmente nos EUA
Recebe um resumo do seu perfil (objetivo, alergias, condições), as notas que o coach lembra sobre você e a descrição das suas refeições.
Para que o coach encontre o que é relevante.
Transferência para a China
A DeepSeek trata seus dados na República Popular da China. A União Europeia não reconhece à China um nível de proteção adequado e não temos cláusulas contratuais assinadas com a DeepSeek: seus dados poderiam ficar ao alcance das autoridades desse país e seria mais difícil para você exercer lá os seus direitos.
Aceito que o Bioboros use meus dados de saúde para criar meu plano e me oferecer o coach, e que os envie aos provedores de IA indicados.
Aceito que meus dados sejam enviados à DeepSeek, na China, ciente dos riscos descritos.
Opcional
Ajude-nos a melhorar: análise de uso, sem dados de saúde.
O Bioboros não substitui seu médico nem seu nutricionista. Consulte-os antes de mudar sua alimentação, sobretudo se você tem uma condição médica, toma medicamentos, está grávida ou amamentando, ou fez cirurgia bariátrica.
Você pode retirar sua permissão quando quiser em Configurações → Privacidade → IA de terceiros. Retirá-la interrompe os novos envios.
Política de Privacidade
Uso de IA
<!-- texto:fin -->
