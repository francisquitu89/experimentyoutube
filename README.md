# Audio → Video
1. `npm install`  2. `cp .env.example .env` y pon `OPENAI_API_KEY`  3. `npm start` → http://localhost:3000
Requiere Node 18+, FFmpeg **y FFprobe** (o `FFMPEG_PATH`, con ffprobe en la misma carpeta). Todo se guarda en `projects/<id>/` y es reanudable.

## Flujo (nada se gasta en imágenes sin tu orden)
`UPLOAD AUDIO` → `ANALYZE` (transcribe + scene plan) → **revisión del plan** → `GENERATE IMAGES` → `RENDER VIDEO`

- Tras ANALYZE se muestra duración, nº de escenas, imágenes por generar y cada escena (tiempos, narración, concepto, prompt). Puedes editar, duplicar o borrar escenas.
- `MAX_SCENES` (por defecto 50): si el plan lo supera, GENERATE IMAGES pide confirmación (y el servidor rechaza la petición sin ella).
- Reanudable: transcripción, plan e imágenes completadas no se repiten.
- El panel superior (`/api/health`) comprueba FFmpeg, FFprobe, la conexión con OpenAI y los tres modelos configurados.
- La API key nunca se muestra ni se registra en logs o respuestas.

## Pruebas
- `npm run test:render` — render FFmpeg sin API.
- `npm run test:flow -- audio.mp3 25` — upload → transcribe → analyze → review (usa OpenAI para transcribir/planificar; **no genera imágenes**).
- `npm run test:image -- <projectId> 1` — una sola imagen.
