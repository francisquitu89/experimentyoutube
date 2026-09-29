# Audio → Video
1. `npm install`  2. `cp .env.example .env` y pon `OPENAI_API_KEY`  3. `npm start` → http://localhost:3000
Requiere Node 18+, FFmpeg **y FFprobe** (o `FFMPEG_PATH`, con ffprobe en la misma carpeta). Todo se guarda en `projects/<id>/` y es reanudable.

## Flujo (nada se gasta en imágenes sin tu orden)
`UPLOAD AUDIO` → `ANALYZE` (transcribe + scene plan) → **revisión del plan** → `GENERATE IMAGES` → `RENDER VIDEO`

## AI Voice
El modo **AI VOICE** permite pegar un guion, elegir una de las voces compatibles y generar narración con `gpt-4o-mini-tts` (configurable con `OPENAI_TTS_MODEL`; voz inicial `marin`, configurable con `OPENAI_TTS_VOICE`). El test compara `marin` y `cedar` con el mismo texto. Las voces están optimizadas para inglés; escucha ambas antes de elegir para español.

- La voz se anuncia como **AI-generated voice**. La dirección de interpretación se guarda separada del texto hablado.
- Los guiones se dividen en bloques naturales de hasta 3.500 caracteres. FFmpeg concatena las partes en `audio/narration_final.mp3`; los timestamps se generan después con `whisper-1`.
- El proyecto, la voz, sus instrucciones y el audio se cachean para reanudar sin repetir llamadas idénticas.
- El storyboard mantiene a Maya como protagonista recurrente. No se generan imágenes durante la creación de voz/storyboard. Tras revisar, `GENERATE 10-FRAME TEST VIDEO` genera solo los primeros diez frames consecutivos y crea `output/test_10_frames.mp4` con audio y subtítulos.
- Los lotes siguientes son explícitos: `GENERATE NEXT 10`, `GENERATE NEXT 25` o confirmación para todos los pendientes. Las imágenes siguientes usan la anterior como referencia mediante edición GPT Image cuando el modelo configurado lo permite.
- El guion de demo atribuye los datos del acuerdo a Reuters y distingue el plan futuro de GeoPark de producción ya disponible. Fuentes: Reuters, [anuncio del 28 de agosto](https://www.reuters.com/business/energy/us-enters-into-oil-agreement-with-venezuela-trump-says-2026-08-28/), [dudas legales del 31 de agosto](https://www.reuters.com/legal/litigation/experts-lawyers-puzzled-over-us-venezuela-oil-deal-call-contract-transparency-2026-08-31/) y [plan de inversión de GeoPark del 29 de septiembre](https://www.reuters.com/business/energy/geopark-invest-7-billion-lift-output-venezuelan-oilfield-2026-09-29/).

- Tras ANALYZE se muestra duración, nº de escenas, imágenes por generar y cada escena (tiempos, narración, concepto, prompt). Puedes editar, duplicar o borrar escenas.
- `MAX_SCENES` (por defecto 50): si el plan lo supera, GENERATE IMAGES pide confirmación (y el servidor rechaza la petición sin ella).
- Reanudable: transcripción, plan e imágenes completadas no se repiten.
- El panel superior (`/api/health`) comprueba FFmpeg, FFprobe, la conexión con OpenAI y los tres modelos configurados.
- La API key nunca se muestra ni se registra en logs o respuestas.

## Pruebas
- `npm run test:render` — render FFmpeg sin API.
- `npm run test:flow -- audio.mp3 25` — upload → transcribe → analyze → review (usa OpenAI para transcribir/planificar; **no genera imágenes**).
- `npm run test:image -- <projectId> 1` — una sola imagen.
- `npm run test:voice` — exactamente dos archivos, `voice_test_marin.mp3` y `voice_test_cedar.mp3`; no genera imágenes.
- `npm run test:transcription -- <projectId>` — valida transcript y timestamps desde el audio final con Whisper 1.
- `npm run test:storyboard -- <projectId>` — valida acciones, beats, continuidad y tiempos; no genera imágenes.
- `npm run test:frames -- <projectId>` — genera solo los primeros diez frames del storyboard.
- `npm run test:full` — guion demo → voz → Whisper → storyboard → diez imágenes → MP4 de revisión. Puede consumir créditos TTS, razonamiento e imágenes.
