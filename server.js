import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import OpenAI, { toFile } from 'openai';
import fs from 'fs';
import path from 'path';
import { spawn, spawnSync } from 'child_process';
import { createHash, randomUUID } from 'crypto';

const ROOT = path.resolve('projects');
fs.mkdirSync(ROOT, { recursive: true });
// ---------- herramientas externas ----------
// FFMPEG_PATH puede ser la ruta a ffmpeg(.exe) o a la carpeta que lo contiene. ffprobe se busca en la misma carpeta
// (o en FFPROBE_PATH si se define). Sin FFMPEG_PATH se usan "ffmpeg" y "ffprobe" del PATH.
const exe = (n) => (process.platform === 'win32' ? `${n}.exe` : n);
const asFile = (p, name) => { try { if (fs.statSync(p).isDirectory()) return path.join(p, exe(name)); } catch {} return p; };
const FFMPEG = process.env.FFMPEG_PATH ? asFile(process.env.FFMPEG_PATH, 'ffmpeg') : 'ffmpeg';
const FFPROBE = process.env.FFPROBE_PATH ? asFile(process.env.FFPROBE_PATH, 'ffprobe')
  : process.env.FFMPEG_PATH ? path.join(path.dirname(FFMPEG), exe('ffprobe')) : 'ffprobe';
const MAX_SCENES = Math.max(1, parseInt(process.env.MAX_SCENES, 10) || 50);

// ---------- seguridad: nunca exponer la API key ----------
const KEY = process.env.OPENAI_API_KEY || '';
const redact = (x) => { let t = String(x ?? ''); if (KEY) t = t.split(KEY).join('[REDACTED]');
  return t.replace(/sk-[A-Za-z0-9_\-*.]{6,}/g, 'sk-[REDACTED]').replace(/Bearer\s+[A-Za-z0-9_\-.]+/g, 'Bearer [REDACTED]'); };
const M = {
  tr: process.env.OPENAI_TRANSCRIPTION_MODEL || 'gpt-transcribe',
  // Según la doc oficial, timestamp_granularities SOLO existe para whisper-1.
  ts: process.env.OPENAI_TIMESTAMP_MODEL || 'whisper-1',
  re: process.env.OPENAI_REASONING_MODEL || 'gpt-5.6-luna',
  im: process.env.OPENAI_IMAGE_MODEL || 'gpt-image-2',
};
const IMG_SIZE = process.env.OPENAI_IMAGE_SIZE || '2048x1152'; // 16:9 exacto, lados múltiplos de 16 (gpt-image-2)
const IMG_QUALITY = process.env.OPENAI_IMAGE_QUALITY || 'medium';
const TTS_MODEL = process.env.OPENAI_TTS_MODEL || 'gpt-4o-mini-tts';
const TTS_DEFAULT_VOICE = process.env.OPENAI_TTS_VOICE || 'marin';
const TTS_VOICES = ['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer', 'verse', 'marin', 'cedar'];
const TTS_TEST_TEXT = 'Hay una pregunta que parece bastante simple: ¿qué harías si de repente tuvieras acceso a una de las mayores reservas de petróleo del planeta? Porque ahí es donde la historia se pone interesante.';
const TTS_BASE_INSTRUCTIONS = 'Speak in Spanish with a natural Latin American conversational accent. Use a warm, intelligent, confident, lightly ironic adult masculine-sounding narrator delivery, with clear pronunciation, natural pauses, and varied intonation. Sound like an interesting person telling a story, not an advertisement or an overly solemn documentary. Do not add words or speak these instructions.';
const TTS_CHUNK_LIMIT = 3500;
const AI_VOICE_TRANSCRIPTION_MODEL = 'whisper-1';
const AI_VOICE_DEMO_TITLE = '¿Por qué Venezuela puede convertirse en una pieza clave del tablero petrolero mundial?';
const AI_VOICE_DEMO_SCRIPT = `Imagina una caja fuerte enorme bajo tu casa, llena de petróleo. No tienes la llave y tus vecinos también quieren decidir quién la abre. Así se entiende el acuerdo petrolero entre Estados Unidos y Venezuela.

El 28 de agosto, Donald Trump afirmó que Estados Unidos tendría control mayoritario sobre más de 65 mil millones de barriles de reservas probadas. Reuters reportó dudas legales y posibles desafíos constitucionales.

Petróleo bajo tierra no significa petróleo listo para vender.

El 29 de septiembre, Reuters informó que GeoPark planea invertir siete mil millones de dólares en el campo Bare. Su meta: producir 90 mil barriles diarios para 2038. Es un plan de largo plazo, no petróleo disponible mañana.

La pregunta no es solo quién tiene las reservas, sino quién produce, compra y transporta ese petróleo. Encontrar la caja fuerte es una cosa; controlar la llave, el transporte y el destino es otra.`;
const AI_VOICE_DEMO_SOURCES = [
  'https://www.reuters.com/business/energy/us-enters-into-oil-agreement-with-venezuela-trump-says-2026-08-28/',
  'https://www.reuters.com/legal/litigation/experts-lawyers-puzzled-over-us-venezuela-oil-deal-call-contract-transparency-2026-08-31/',
  'https://www.reuters.com/business/energy/geopark-invest-7-billion-lift-output-venezuelan-oilfield-2026-09-29/',
];
const openai = KEY ? new OpenAI({ apiKey: KEY }) : null;
const STYLE = {
  visual_style: 'Deliberately crude amateur drawing, as if made in 2 minutes in MS Paint by a total beginner using a mouse. Funny, naive, intentionally bad',
  line_style: 'thick black uneven wobbly lines, shaky hand-drawn strokes, slightly broken outlines',
  character_style: 'stick figures: circle head, single-line body, arms and legs as lines, mitten-like simple hands, dot eyes, primitive mouth expressions',
  color_policy: 'white background; flat solid fills in a few bright colors, applied sloppily; no gradients, no shading',
  background_policy: 'plain white, no scenery unless essential; lots of empty space',
  composition_policy: 'wide 16:9 frame; one clear idea; subjects centered with wide margins; nothing important touching or cut by the edges; visual humor and exaggeration welcome',
  text_policy: 'avoid text; if essential use 1-3 short words in wobbly handwriting, correctly spelled',
  negative: 'realism, photo, anime, manga, Disney, Pixar, 3D render, cinematic or dramatic lighting, realistic shadows, textures, polished vector or digital art, detailed backgrounds, sophisticated anatomy, professional illustration',
};

// ---------- storage helpers ----------
const pdir = (id) => path.join(ROOT, path.basename(id));
const sfile = (id) => path.join(pdir(id), 'state.json');
const rd = (f) => JSON.parse(fs.readFileSync(f, 'utf8'));
const load = (id) => rd(sfile(id));
const save = (id, s) => fs.writeFileSync(sfile(id), JSON.stringify(s, null, 2));
const log = (id, msg) => {
  const line = `[${new Date().toISOString().slice(11, 19)}] ${redact(msg)}`;
  console.log(line);
  try { const s = load(id); s.logs = [...(s.logs || []).slice(-200), line]; save(id, s); } catch {}
};
const patch = (id, fn) => { const s = load(id); fn(s); save(id, s); return s; };
const run = (cmd, args, opts = {}) => new Promise((res, rej) => {
  const p = spawn(cmd, args, opts); let err = '';
  p.stderr.on('data', (d) => (err += d));
  p.on('error', (e) => rej(new Error(`No se pudo ejecutar ${cmd}: ${e.message}`)));
  p.on('close', (c) => (c === 0 ? res() : rej(new Error(err.slice(-800)))));
});
const ffmpegOk = () => spawnSync(FFMPEG, ['-version']).status === 0;
const ffprobeOk = () => spawnSync(FFPROBE, ['-version']).status === 0;
const FFPROBE_HELP = `FFprobe no encontrado (se intentó ejecutar: ${FFPROBE}). FFprobe viene incluido en FFmpeg y debe estar en la misma carpeta que ffmpeg. `
  + 'Solución: en .env define FFMPEG_PATH con la ruta completa a ffmpeg.exe (sin comillas), por ejemplo '
  + 'FFMPEG_PATH=C:\\Users\\<usuario>\\AppData\\Local\\ffmpeg\\<carpeta>\\bin\\ffmpeg.exe ; ffprobe.exe se busca en esa misma carpeta bin. '
  + 'Si está en otra ubicación, usa FFPROBE_PATH. Después reinicia el servidor.';
const FFMPEG_HELP = `FFmpeg no encontrado (se intentó ejecutar: ${FFMPEG}). Instálalo (Windows: winget install ffmpeg | macOS: brew install ffmpeg | Linux: apt install ffmpeg) o define FFMPEG_PATH en .env con la ruta a ffmpeg.exe.`;
const assertTools = () => { if (!ffmpegOk()) throw new Error(FFMPEG_HELP); if (!ffprobeOk()) throw new Error(FFPROBE_HELP); };
// La duración es imprescindible (recorta el último plano y valida el plan): si FFprobe falla, se aborta con error explícito.
function audioDuration(file) {
  if (!ffprobeOk()) throw new Error(FFPROBE_HELP);
  const r = spawnSync(FFPROBE, ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', file]);
  const d = parseFloat(String(r.stdout || '').trim());
  if (r.status !== 0 || !isFinite(d) || d <= 0) throw new Error(`FFprobe no pudo leer la duración de ${path.basename(file)}: ${redact(String(r.stderr || '').trim().slice(-300)) || 'salida vacía'}. ¿El archivo está dañado o no es audio?`);
  return d;
}
const hashText = (value) => createHash('sha256').update(String(value)).digest('hex');
const voiceTestDir = () => path.join(ROOT, '_voice-test');
const voiceTestMetadataFile = () => path.join(voiceTestDir(), 'voice_test.json');
async function synthesizeSpeechFile(text, voice, instructions, filePath) {
  const response = await openai.audio.speech.create({
    model: TTS_MODEL, voice, input: text, instructions, response_format: 'mp3',
  });
  fs.writeFileSync(filePath, Buffer.from(await response.arrayBuffer()));
  if (!fs.statSync(filePath).size) throw new Error(`TTS devolvió un archivo vacío: ${path.basename(filePath)}`);
}
async function generateVoiceTest() {
  if (!openai) throw new Error('Falta OPENAI_API_KEY en .env');
  assertTools();
  const directory = voiceTestDir();
  fs.mkdirSync(directory, { recursive: true });
  const metadataFile = voiceTestMetadataFile();
  const metadata = fs.existsSync(metadataFile) ? rd(metadataFile) : { tests: {} };
  const scriptHash = hashText(TTS_TEST_TEXT), instructions = TTS_BASE_INSTRUCTIONS;
  let apiCalls = 0;
  const files = [];
  for (const voice of ['marin', 'cedar']) {
    const file = `voice_test_${voice}.mp3`, absoluteFile = path.join(directory, file);
    const prior = metadata.tests[voice];
    const cached = prior?.script_hash === scriptHash && prior.model === TTS_MODEL && prior.voice === voice
      && prior.instructions === instructions && fs.existsSync(absoluteFile) && fs.statSync(absoluteFile).size > 0;
    if (!cached) {
      await synthesizeSpeechFile(TTS_TEST_TEXT, voice, instructions, absoluteFile);
      apiCalls++;
      metadata.tests[voice] = { script_hash: scriptHash, model: TTS_MODEL, voice, instructions,
        instructions_hash: hashText(instructions), file, duration: +audioDuration(absoluteFile).toFixed(3), generated_at: new Date().toISOString() };
      fs.writeFileSync(metadataFile, JSON.stringify(metadata, null, 2));
    }
    files.push({ voice, file, duration: metadata.tests[voice].duration, cached });
  }
  return { model: TTS_MODEL, files, apiCalls, imageGenerationCalled: false };
}

function splitScriptIntoChunks(script, limit = TTS_CHUNK_LIMIT) {
  const paragraphs = String(script).replace(/\r\n/g, '\n').trim().split(/\n\s*\n/).map((paragraph) => paragraph.trim()).filter(Boolean);
  const units = [];
  for (const paragraph of paragraphs) {
    const sentences = paragraph.match(/[^.!?]+(?:[.!?]+|$)/g) || [paragraph];
    for (const rawSentence of sentences) {
      const sentence = rawSentence.trim();
      if (!sentence) continue;
      if (sentence.length <= limit) { units.push(sentence); continue; }
      const clauses = sentence.split(/(?<=[,;:])\s+/);
      for (const clause of clauses) {
        if (clause.length <= limit) { units.push(clause.trim()); continue; }
        let fragment = '';
        for (const word of clause.split(/\s+/)) {
          if (word.length > limit) throw new Error('El guion contiene una palabra más larga que el límite TTS.');
          const candidate = fragment ? `${fragment} ${word}` : word;
          if (candidate.length > limit) { units.push(fragment); fragment = word; }
          else fragment = candidate;
        }
        if (fragment) units.push(fragment);
      }
    }
  }
  const chunks = [];
  let current = '';
  for (const unit of units) {
    const candidate = current ? `${current} ${unit}` : unit;
    if (candidate.length > limit && current) { chunks.push(current); current = unit; }
    else current = candidate;
  }
  if (current) chunks.push(current);
  if (!chunks.length) throw new Error('El guion está vacío.');
  if (chunks.some((chunk) => chunk.length > limit)) throw new Error('No se pudo dividir el guion dentro del límite TTS.');
  return chunks;
}

const VOICE_DIRECTION_SCHEMA = {
  name: 'voice_directions', strict: true,
  schema: { type: 'object', additionalProperties: false, required: ['directions'], properties: { directions: {
    type: 'array', items: { type: 'object', additionalProperties: false, required: ['chunk_index', 'instructions'], properties: {
      chunk_index: { type: 'integer' }, instructions: { type: 'string' },
    } },
  } } },
};

function findAIVoiceProject(cacheKey) {
  for (const entry of fs.readdirSync(ROOT, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith('_') || !fs.existsSync(sfile(entry.name))) continue;
    try { if (load(entry.name).aiVoice?.cache_key === cacheKey) return entry.name; } catch {}
  }
  return null;
}

function createAIVoiceProject({ title, script, voice = TTS_DEFAULT_VOICE, sources = [] }) {
  const cleanScript = String(script || '').trim();
  if (!cleanScript) throw new Error('El guion está vacío.');
  if (!TTS_VOICES.includes(voice)) throw new Error(`Voz no válida. Elige una de estas: ${TTS_VOICES.join(', ')}.`);
  if (cleanScript.length > 100000) throw new Error('El guion supera el máximo de 100.000 caracteres.');
  const scriptHash = hashText(cleanScript), cacheKey = hashText(JSON.stringify({ scriptHash, model: TTS_MODEL, voice,
    direction_model: M.re, voice_profile_hash: hashText(TTS_BASE_INSTRUCTIONS), direction_version: 1 }));
  const existing = findAIVoiceProject(cacheKey);
  if (existing) return { id: existing, cached: true };
  const id = randomUUID().slice(0, 8), dir = pdir(id);
  for (const folder of ['audio', 'transcript', 'scenes', 'frames', 'output', 'script']) fs.mkdirSync(path.join(dir, folder), { recursive: true });
  fs.writeFileSync(path.join(dir, 'style_bible.json'), JSON.stringify(STYLE, null, 2));
  fs.writeFileSync(path.join(dir, 'script', 'script.txt'), cleanScript);
  const references = sources.length ? sources : cleanScript === AI_VOICE_DEMO_SCRIPT ? AI_VOICE_DEMO_SOURCES : [];
  fs.writeFileSync(path.join(dir, 'script', 'sources.json'), JSON.stringify(references, null, 2));
  const scriptChunks = splitScriptIntoChunks(cleanScript);
  const metadata = { version: 1, script_hash: scriptHash, model: TTS_MODEL, direction_model: M.re, voice, cache_key: cacheKey,
    instructions: null, instructions_hash: null, status: 'queued', chunks: scriptChunks.map((text, index) => ({ index, file: `audio_${String(index + 1).padStart(3, '0')}.mp3`, text_hash: hashText(text), instructions: null, duration: null })),
    file: 'audio/narration_final.mp3', duration: null };
  fs.writeFileSync(path.join(dir, 'script', 'voice_metadata.json'), JSON.stringify(metadata, null, 2));
  save(id, { id, name: title || 'AI Voice project', audioFile: 'narration_final.mp3', duration: 0, step: 2, scenes: [], logs: [], mode: 'ai_voice',
    aiVoice: { status: 'queued', stage: 'voice', voice, model: TTS_MODEL, script_hash: scriptHash, cache_key: cacheKey, voiceReady: false, generatedFrames: 0 } });
  return { id, cached: false };
}

async function createVoiceDirections(scriptChunks) {
  const input = scriptChunks.map((text, index) => ({ chunk_index: index, spoken_text: text }));
  const response = await openai.chat.completions.create({ model: M.re,
    response_format: { type: 'json_schema', json_schema: VOICE_DIRECTION_SCHEMA },
    messages: [
      { role: 'system', content: 'Write only performance directions for each supplied spoken-text chunk. Do not rewrite, summarize, translate, or add spoken words. Output one distinct short direction per chunk, in Spanish, reflecting the dramatic function: curious hook, clear explanation, surprise, light irony, punchline pause, or reflective conclusion as appropriate. The audio model receives these separately as instructions; they must never be included in the spoken text. Base the directions only on the supplied script.' },
      { role: 'user', content: JSON.stringify(input) },
    ] });
  const result = JSON.parse(response.choices[0].message.content).directions;
  if (result.length !== scriptChunks.length || result.some((entry, index) => entry.chunk_index !== index || !entry.instructions.trim()))
    throw new Error('GPT no devolvió una dirección de voz válida para cada bloque.');
  return result.map((entry) => entry.instructions.trim());
}

async function runAIVoicePipeline(id) {
  const dir = pdir(id), metadataFile = path.join(dir, 'script', 'voice_metadata.json');
  try {
    if (!openai) throw new Error('Falta OPENAI_API_KEY en .env');
    let metadata = rd(metadataFile), state = load(id);
    const script = fs.readFileSync(path.join(dir, 'script', 'script.txt'), 'utf8');
    const chunks = splitScriptIntoChunks(script);
    patch(id, (current) => { current.aiVoice.status = 'generating'; current.aiVoice.stage = 'voice directions'; current.error = null; });
    if (!metadata.instructions) {
      const directions = await createVoiceDirections(chunks);
      metadata.voice_directions = directions;
      metadata.instructions = directions.map((direction) => `${TTS_BASE_INSTRUCTIONS} ${direction}`);
      metadata.instructions_hash = hashText(JSON.stringify(metadata.instructions));
      metadata.chunks.forEach((chunk, index) => { chunk.instructions = metadata.instructions[index]; });
      metadata.status = 'generating';
      fs.writeFileSync(metadataFile, JSON.stringify(metadata, null, 2));
    }
    metadata.instructions.forEach((direction, index) => { metadata.chunks[index].instructions = direction; });
    const audioDir = path.join(dir, 'audio');
    for (const [index, text] of chunks.entries()) {
      const part = metadata.chunks[index], output = path.join(audioDir, part.file);
      if (!fs.existsSync(output) || !fs.statSync(output).size) {
        patch(id, (current) => { current.aiVoice.stage = `voice ${index + 1}/${chunks.length}`; });
        log(id, `[AI VOICE ${String(index + 1).padStart(3, '0')}] TTS ${metadata.voice}`);
        await synthesizeSpeechFile(text, metadata.voice, part.instructions, output);
      }
      part.duration ||= +audioDuration(output).toFixed(3);
      fs.writeFileSync(metadataFile, JSON.stringify(metadata, null, 2));
    }
    const finalAudio = path.join(audioDir, 'narration_final.mp3');
    if (!fs.existsSync(finalAudio) || !fs.statSync(finalAudio).size) {
      const concatFile = path.join(audioDir, 'concat.txt');
      fs.writeFileSync(concatFile, metadata.chunks.map((chunk) => `file '${chunk.file}'`).join('\n'));
      await run(FFMPEG, ['-y', '-f', 'concat', '-safe', '0', '-i', 'concat.txt', '-c:a', 'libmp3lame', '-q:a', '2', 'narration_final.mp3'], { cwd: audioDir });
    }
    metadata.file = 'audio/narration_final.mp3';
    metadata.duration = +audioDuration(finalAudio).toFixed(3);
    metadata.status = 'completed';
    metadata.cache_signature = hashText(JSON.stringify({ script_hash: metadata.script_hash, model: metadata.model, direction_model: metadata.direction_model,
      voice: metadata.voice, instructions_hash: metadata.instructions_hash }));
    fs.writeFileSync(metadataFile, JSON.stringify(metadata, null, 2));
    patch(id, (current) => { current.duration = metadata.duration; current.aiVoice.duration = metadata.duration; current.aiVoice.voiceReady = true; current.aiVoice.stage = 'transcription'; });
    log(id, `[AI VOICE] Completed (${metadata.voice}, ${metadata.duration}s, ${chunks.length} chunks)`);

    const transcript = await transcribe(id, { forceModel: AI_VOICE_TRANSCRIPTION_MODEL });
    patch(id, (current) => { current.aiVoice.stage = 'scene analysis'; });
    const scenePlan = await planScenes(id, transcript);
    patch(id, (current) => { current.aiVoice.stage = 'visual sequence planning'; });
    await planVisualSequences(id);
    patch(id, (current) => { current.aiVoice.stage = 'continuous storyboard'; });
    await planAIVoiceStoryboardActions(id);
    const storyboard = generateVisualStoryboard(id);
    if (storyboard.validation.status !== 'passed') throw new Error(`Auditoría storyboard: ${JSON.stringify(storyboard.validation.issues.slice(0, 8))}`);
    patch(id, (current) => { current.step = 4; current.aiVoice.status = 'review'; current.aiVoice.stage = 'storyboard review';
      current.aiVoice.sceneCount = scenePlan.scenes.length; current.aiVoice.totalFrames = storyboard.total_frames; current.aiVoice.generatedFrames = 0; });
    log(id, `[AI VOICE] Ready for review (${storyboard.sequences.length} sequences, ${storyboard.total_frames} frames; 0 images generated)`);
    return { id, duration: metadata.duration, storyboard_frames: storyboard.total_frames };
  } catch (error) {
    patch(id, (current) => { current.aiVoice.status = 'failed'; current.aiVoice.stage = 'error'; current.error = redact(error.message); });
    log(id, `[AI VOICE ERROR] ${error.message}`);
    throw error;
  }
}

function launchAIVoicePipeline(id) {
  if (running.has(id)) return;
  running.add(id);
  runAIVoicePipeline(id).catch(() => {}).finally(() => running.delete(id));
}
const running = new Set();

// ---------- pipeline ----------
async function transcribe(id, { forceModel = null } = {}) {
  const f = path.join(pdir(id), 'transcript', 'transcript.json');
  if (fs.existsSync(f)) {
    const saved = rd(f);
    if (!forceModel || saved.model === forceModel) return saved;
  }
  log(id, '[TRANSCRIBE] Started');
  const s = load(id);
  // La API limita a 25 MB: se envía una copia comprimida mono 16 kHz (~14 MB/hora). El audio original no se toca.
  const small = path.join(pdir(id), 'audio', 'for_transcription.mp3');
  if (!fs.existsSync(small)) await run(FFMPEG, ['-y', '-i', path.join(pdir(id), 'audio', s.audioFile), '-vn', '-ac', '1', '-ar', '16000', '-b:a', '32k', small]);
  if (fs.statSync(small).size > 25 * 1024 * 1024) throw new Error('Audio demasiado largo para una sola petición (>25 MB comprimido). Divídelo en partes.');
  const call = (model) => openai.audio.transcriptions.create({
    file: fs.createReadStream(small), model, response_format: 'verbose_json', timestamp_granularities: ['word', 'segment'] });
  let used = forceModel || M.tr, r;
  if (forceModel) r = await call(forceModel);
  else {
    try { r = await call(M.tr); if (!r.segments?.length) throw new Error('sin segmentos'); }
    catch (e) {
      if (M.tr === M.ts) throw e;
      log(id, `[TRANSCRIBE] ${M.tr} no devuelve timestamps (${String(e.message).slice(0, 120)}). Usando ${M.ts}, el modelo oficial con timestamps.`);
      used = M.ts; r = await call(M.ts);
    }
  }
  if (!r.segments?.length) throw new Error('La API no devolvió segmentos con timestamps');
  const t = {
    model: used, duration: r.duration ?? s.duration, language: r.language, text: r.text,
    segments: r.segments.map((x) => ({ start: +x.start, end: +x.end, text: x.text.trim() })),
    words: (r.words || []).map((w) => ({ word: w.word, start: +w.start, end: +w.end })),
  };
  fs.writeFileSync(f, JSON.stringify(t, null, 2));
  writeSubs(id, t);
  log(id, `[TRANSCRIBE] Completed (${used}, ${t.segments.length} segmentos, ${t.words.length} palabras)`);
  return t;
}

const ts = (x, sep) => { const h = Math.floor(x / 3600), m = Math.floor(x % 3600 / 60), s = Math.floor(x % 60), ms = Math.round((x % 1) * 1000);
  const p = (n, l = 2) => String(n).padStart(l, '0'); return `${p(h)}:${p(m)}:${p(s)}${sep}${p(ms, 3)}`; };
function writeSubs(id, t) {
  const dir = path.join(pdir(id), 'transcript');
  fs.writeFileSync(path.join(dir, 'subtitles.srt'), t.segments.map((g, i) => `${i + 1}\n${ts(g.start, ',')} --> ${ts(g.end, ',')}\n${g.text}\n`).join('\n'));
  fs.writeFileSync(path.join(dir, 'subtitles.vtt'), 'WEBVTT\n\n' + t.segments.map((g) => `${ts(g.start, '.')} --> ${ts(g.end, '.')}\n${g.text}\n`).join('\n'));
}

const SCHEMA = { name: 'scene_plan', strict: true, schema: { type: 'object', additionalProperties: false, required: ['characters', 'scenes'], properties: {
  characters: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['id', 'name', 'description'], properties: { id: { type: 'string' }, name: { type: 'string' }, description: { type: 'string' } } } },
  scenes: { type: 'array', items: { type: 'object', additionalProperties: false,
    required: ['start', 'end', 'narration', 'visual_concept', 'visual_reason', 'prompt', 'character_ids', 'importance'],
    properties: { start: { type: 'number' }, end: { type: 'number' }, narration: { type: 'string' }, visual_concept: { type: 'string' }, visual_reason: { type: 'string' },
      prompt: { type: 'string' }, character_ids: { type: 'array', items: { type: 'string' } }, importance: { type: 'number' } } } } } } };

async function planScenes(id, t) {
  const dir = pdir(id), f = path.join(dir, 'scene_plan.json');
  if (fs.existsSync(f)) return rd(f);
  log(id, '[SCENE ANALYSIS] Started');
  const bf = path.join(dir, 'style_bible.json');
  if (!fs.existsSync(bf)) fs.writeFileSync(bf, JSON.stringify(STYLE, null, 2));
  const segs = t.segments.map((g) => `[${g.start.toFixed(2)}-${g.end.toFixed(2)}] ${g.text}`).join('\n');
  const aiVoiceRules = load(id).mode === 'ai_voice'
    ? '\nThis is an AI-voice continuous story: include one recurring stick-figure narrator named Maya in every scene. She travels through the visual world and physically reacts to each narrated event. Keep her fixed appearance and clothes; other people may appear only when the narration requires them.' : '';
  const sys = `You are a human video editor planning static illustrations for an explainer YouTube video narrated by the user. Decide when the image should change: change of idea, subject or context, new metaphor, new main object, emotional shift, contrast, punchline, relevant question, conclusion. Keep one image while the same idea continues; do NOT create unnecessary scenes (typically one scene every 5-15 seconds).
Do not illustrate sentences literally: use visual metaphors, contrast, exaggeration, visual humor, simple symbols, cause/effect. A viewer should roughly understand the narration from the image alone.
Every scene "start" MUST equal the start time of one of the given segments; scenes must be in order and cover the whole audio.
"characters": recurring people/entities (may be empty). Describe each ONCE, precisely and simply: head, body, clothing, color. Scenes list the character_ids they contain.
"prompt": English, describes ONLY the scene content (who, doing what, which objects, expression, layout). Do NOT describe art style; the style bible is added automatically. Refer to characters by name only.${aiVoiceRules}`;
  const r = await openai.chat.completions.create({ model: M.re, response_format: { type: 'json_schema', json_schema: SCHEMA },
    messages: [{ role: 'system', content: sys }, { role: 'user', content: `Total duration: ${t.duration}s\n${segs}` }] });
  const out = JSON.parse(r.choices[0].message.content);
  const aiVoice = load(id).mode === 'ai_voice';
  if (aiVoice) out.characters = [{ id: 'narrator_maya', name: 'Maya', description: 'Recurring small stick-figure narrator with a round head, short dark hair, yellow sweater, blue trousers, and white shoes. Preserve the exact same face, outfit, silhouette, and proportions in every scene.' }];
  let sc = out.scenes.sort((a, b) => a.start - b.start);
  if (!sc.length) throw new Error('El modelo no devolvió escenas');
  if (aiVoice) sc.forEach((scene) => {
    scene.character_ids = ['narrator_maya'];
    scene.prompt = `Maya, the same recurring stick-figure narrator, physically moves through this scene: ${scene.prompt}`;
  });
  // Los cortes se ajustan al inicio de segmento real más cercano (timestamps siempre provienen de la transcripción)
  const starts = t.segments.map((g) => g.start);
  sc.forEach((x) => { x.start = starts.reduce((a, b) => (Math.abs(b - x.start) < Math.abs(a - x.start) ? b : a)); });
  sc[0].start = 0;
  sc = sc.filter((x, i) => i === 0 || x.start > sc[i - 1].start + 0.3);
  sc.forEach((x, i) => { x.end = i < sc.length - 1 ? sc[i + 1].start : t.duration; });
  fs.writeFileSync(path.join(dir, 'characters.json'), JSON.stringify(out.characters, null, 2));
  const plan = { scenes: sc.map((x, i) => ({
    scene_id: i + 1, start: +x.start.toFixed(2), end: +x.end.toFixed(2), duration: +(x.end - x.start).toFixed(2),
    narration: x.narration, visual_concept: x.visual_concept, visual_reason: x.visual_reason,
    prompt: x.prompt, character_ids: x.character_ids, importance: x.importance, status: 'pending' })) };
    plan.scenes.forEach((scene, index) => { scene.motion = chooseMotion(scene, index, plan.scenes[index - 1]?.motion?.type); });
  fs.writeFileSync(f, JSON.stringify(plan, null, 2));
  patch(id, (s) => { s.scenes = plan.scenes.map((x) => ({ scene_id: x.scene_id, status: 'pending', retry_count: 0, file: null, error: null })); });
  log(id, `[SCENE ANALYSIS] Completed (${plan.scenes.length} scenes, ${out.characters.length} personajes)`);
  return plan;
}

// Prompt = A) escena + B) style bible + C) composición + D) personajes + E) negativos
function fullPrompt(id, sc) {
  const dir = pdir(id), B = rd(path.join(dir, 'style_bible.json'));
  const cf = path.join(dir, 'characters.json');
  const chars = (fs.existsSync(cf) ? rd(cf) : []).filter((c) => (sc.character_ids || []).includes(c.id));
  return [
    `Widescreen 16:9 image. ${B.visual_style}.`,
    `Scene: ${sc.prompt}`,
    chars.length ? `Recurring characters (draw exactly like this in every image, same head, body, clothes, colors, proportions): ${chars.map((c) => `${c.name} — ${c.description}`).join('; ')}.` : '',
    `Lines: ${B.line_style}. Characters: ${B.character_style}. Color: ${B.color_policy}. Background: ${B.background_policy}.`,
    `Composition: ${B.composition_policy}. Text: ${B.text_policy}.`,
    `It must NOT look like: ${B.negative}. It should look intentionally crude and amateur.`,
  ].filter(Boolean).join('\n');
}
const pad = (n) => String(n).padStart(3, '0');
const MOTION_TYPES = ['zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'pan_up', 'pan_down', 'pan_diagonal', 'zoom_in_fast', 'static'];
function chooseMotion(sc, index, previous) {
  const text = `${sc.narration || ''} ${sc.visual_concept || ''} ${sc.visual_reason || ''} ${sc.prompt || ''}`.toLowerCase();
  let type;
  if (/punchline|remate|chiste|funny|joke|surprising/.test(text)) type = 'zoom_in_fast';
  else if (/loss|p[eé]rdida|retroceso|deja de crecer|stopped|fall(?:s|ing)?|declin/.test(text)) type = 'zoom_out';
  else if (/grow(?:th|ing)?|crec(?:e|imiento)|increase|rising|build(?:ing)?/.test(text)) type = 'zoom_in';
  else if (/between|enfrent|contrast|versus|balance|balanced/.test(text)) type = index % 2 ? 'pan_left' : 'pan_right';
  else if (/person|persona|stands|standing|holds|holding/.test(text)) type = index % 2 ? 'pan_right' : 'pan_left';
  else type = MOTION_TYPES[(index * 3 + 1) % (MOTION_TYPES.length - 1)];
  if (type === previous) type = MOTION_TYPES.find((candidate) => candidate !== previous && candidate !== 'static') || 'static';
  const intensity = type === 'zoom_in_fast' ? 0.12 : type === 'static' ? 0 : Number(sc.importance) >= 0.9 ? 0.15 : 0.08;
  return { type, intensity, direction: type.startsWith('pan_') ? type.slice(4) : 'center', duration: +(sc.end - sc.start).toFixed(2) };
}
function ensureMotionPlan(id) {
  const file = path.join(pdir(id), 'scene_plan.json'), plan = rd(file);
  let previous = null, changed = false;
  plan.scenes.forEach((sc, index) => {
    if (!sc.motion || !MOTION_TYPES.includes(sc.motion.type)) { sc.motion = chooseMotion(sc, index, previous); changed = true; }
    previous = sc.motion.type;
  });
  if (changed) fs.writeFileSync(file, JSON.stringify(plan, null, 2));
  return plan;
}
const SEQUENCE_SCHEMA = {
  name: 'visual_sequence_plan', strict: true,
  schema: {
    type: 'object', additionalProperties: false, required: ['sequences'],
    properties: { sequences: {
      type: 'array', items: {
        type: 'object', additionalProperties: false, required: ['sequence_id', 'frames'],
        properties: {
          sequence_id: { type: 'integer' },
          frames: {
            type: 'array', items: {
              type: 'object', additionalProperties: false,
              required: ['start_offset', 'visual', 'prompt', 'previous_frame_context', 'continuity_notes', 'motion_type', 'motion_amount', 'intentional_fast'],
              properties: {
                start_offset: { type: 'number' }, visual: { type: 'string' }, prompt: { type: 'string' },
                previous_frame_context: { type: 'string' }, continuity_notes: { type: 'string' },
                motion_type: { type: 'string', enum: MOTION_TYPES }, motion_amount: { type: 'number' }, intentional_fast: { type: 'boolean' },
              },
            },
          },
        },
      },
    } },
  },
};
const sequencePlanFile = (id) => path.join(pdir(id), 'visual_sequence_plan.json');
const storyboardFile = (id) => path.join(pdir(id), 'visual_storyboard.json');
const storyboardActionsFile = (id) => path.join(pdir(id), 'script', 'storyboard_actions.json');
const STORYBOARD_ACTION_SCHEMA = {
  name: 'continuous_storyboard_actions', strict: true,
  schema: { type: 'object', additionalProperties: false, required: ['sequences'], properties: { sequences: {
    type: 'array', items: { type: 'object', additionalProperties: false, required: ['sequence_id', 'beats'], properties: {
      sequence_id: { type: 'integer' },
      beats: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['source_plan_beat_id', 'actions'], properties: {
        source_plan_beat_id: { type: 'integer' },
        actions: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['action', 'object_state'], properties: {
          action: { type: 'string' }, object_state: { type: 'string' },
        } } },
      } } },
    } },
  } } },
};

function inferSequenceTitle(scene) {
  const text = (scene.visual_concept || scene.narration || '').trim();
  if (!text) return 'Sequence';
  const normalized = text.replace(/\s+/g, ' ').replace(/\.$/, '');
  return normalized.length <= 72 ? normalized : `${normalized.slice(0, 69).replace(/\s+\S*$/, '')}...`;
}

function buildCharacterBible(id) {
  const dir = pdir(id);
  const chars = fs.existsSync(path.join(dir, 'characters.json')) ? rd(path.join(dir, 'characters.json')) : [];
  const primary = chars[0] || { name: 'Maya', description: 'Woman with round head, short dark hair, yellow sweater, blue pants, and white shoes.' };
  return {
    version: 1,
    locked: true,
    principal_character: primary.name,
    primary_rule: 'The character must remain visually identical across every frame unless the story unequivocally requires a change.',
    characters: chars.length ? chars.map((character) => ({
      name: character.name,
      age_approx: 'adult',
      appearance: character.description,
      head: 'simple round head',
      body: 'stick-figure torso and limbs',
      clothing: character.description.includes('sweater') ? 'same sweater and pants as the approved character bible' : 'same approved outfit',
      colors: character.description.includes('yellow') ? ['yellow', 'blue', 'white', 'black'] : ['flat primary colors'],
      proportions: 'consistent, simple stick-figure proportions',
      continuity: 'Do not change face, clothes, silhouette or proportions unless the story requires it.',
    })) : [{
      name: primary.name,
      age_approx: 'adult',
      appearance: primary.description,
      head: 'simple round head',
      body: 'stick-figure torso and limbs',
      clothing: 'same approved outfit',
      colors: ['yellow', 'blue', 'white', 'black'],
      proportions: 'consistent, simple stick-figure proportions',
      continuity: 'Do not change face, clothes, silhouette or proportions unless the story requires it.',
    }],
  };
}

function buildWorldBible(id) {
  const dir = pdir(id);
  const plan = fs.existsSync(path.join(dir, 'scene_plan.json')) ? rd(path.join(dir, 'scene_plan.json')) : { scenes: [] };
  const visualObjects = [
    ['shopping cart', /\bshopping cart\b/i], ['scissors', /\bscissors\b/i], ['price tag', /\bprice tag\b/i],
    ['coins', /\bcoins?\b/i], ['income elevator', /\belevator\b/i], ['expenses staircase', /\bstaircase\b/i],
    ['bills', /\bbills?\b/i], ['warning symbols', /\bwarning symbols?\b/i], ['money sack', /\b(?:money )?sack\b/i],
    ['spending funnel', /\bfunnel\b/i], ['balance scale', /\bscale\b/i], ['savings jar', /\b(?:savings )?jar\b/i],
    ['arrow', /\barrow\b/i], ['Venezuela', /\bvenezuela\b/i], ['United States', /\b(?:united states|u\.s\.)\b/i],
    ['safe', /\b(?:safe|vault|strongbox)\b/i], ['key', /\bkeys?\b/i], ['map', /\bmaps?\b/i],
    ['oil field', /\b(?:oil|petroleum) field\b|\bbare oilfield\b/i], ['oil well', /\boil wells?\b/i],
    ['oil derrick', /\b(?:oil )?(?:rig|derrick|pumpjack)s?\b/i], ['oil barrels', /\bbarrels?\b/i],
    ['refinery', /\brefiner(?:y|ies)\b/i], ['oil tanker', /\btankers?\b/i], ['ship', /\bships?\b/i],
    ['pipeline', /\bpipelines?\b/i], ['ocean', /\b(?:ocean|sea)\b/i], ['power grid', /\b(?:electricity|power grid)\b/i],
    ['contract', /\bcontracts?\b/i], ['India', /\bindia\b/i], ['China', /\bchina\b/i], ['Asia', /\basia\b/i],
    ['world map', /\bworld map\b/i], ['office', /\boffices?\b/i],
  ];
  const objectsIn = (text) => visualObjects.filter(([, pattern]) => pattern.test(text)).map(([name]) => name);
  const sequenceObjects = plan.scenes.map((scene) => ({
    sequence_id: scene.scene_id,
    objects: [...new Set(objectsIn(`${scene.visual_concept || ''} ${scene.prompt || ''}`))],
  }));
  const objects = [...new Set(sequenceObjects.flatMap((sequence) => sequence.objects))];
  return {
    version: 1,
    locked: true,
    place: 'plain white background; each narrative sequence has its own stable minimal set',
    architecture: 'simple geometric shapes and symbolic objects; preserve the set within a sequence and allow a new set at a new sequence',
    objects,
    sequence_objects: sequenceObjects,
    colors: ['white', 'black', 'yellow', 'blue', 'red', 'green'],
    context: 'Deliberately crude amateur drawing, intentionally simple and readable, with huge margins and empty space.',
    continuity: 'Keep each sequence environment and its object layout stable; a new sequence may establish a new metaphorical set.',
  };
}

const STORYBOARD_FPS = 4.2;
const STORYBOARD_CHOREOGRAPHY = {
  1: [
    [
      ['Maya faces the cart and fixes her gaze on its oversized price tag.', 'Cart stays still; the price tag and expense coins are full-sized.'],
      ['Maya turns her shoulders toward the price tag.', 'The tag remains attached to the cart; scissors rest beside it.'],
      ['Maya reaches for the scissors beside the cart.', 'The tag is still uncut and the coins remain unchanged.'],
      ['Maya lifts the scissors to the level of the tag.', 'Open scissors approach one corner of the full-sized tag.'],
      ['Maya opens the blades around the tag corner.', 'The tag is held between open blades; no piece is cut yet.'],
      ['Maya aligns the blades with the tag edge.', 'The blades touch the edge while the tag remains full-sized.'],
      ['Maya closes the scissors for the first snip.', 'A small notch appears in the tag edge.'],
      ['Maya reopens the scissors after the snip.', 'The notch is visible; the rest of the tag is unchanged.'],
      ['Maya makes a second short snip beside the first notch.', 'A narrow paper strip separates from the tag.'],
      ['Maya pulls the scissors away from the cut edge.', 'The tag is visibly shorter; the cart stays fixed.'],
      ['Maya watches the nearby expense coins shrink one step.', 'The coins become slightly smaller while the tag stays shortened.'],
      ['Maya lowers the scissors and checks the smaller tag.', 'The cut tag and slightly smaller coins remain beside the cart.'],
    ],
    [
      ['Maya raises the scissors toward the remaining tag edge.', 'The already-shortened tag and reduced coins carry over unchanged.'],
      ['Maya rotates the scissors to line up another cut.', 'Open blades meet the remaining edge of the tag.'],
      ['Maya closes the blades for another deliberate snip.', 'A second narrow strip separates from the tag.'],
      ['Maya opens the blades and draws them back.', 'The tag is shorter again; no new object appears.'],
      ['Maya makes one more small cut at the tag edge.', 'A new notch makes the tag noticeably smaller.'],
      ['Maya pauses with the scissors above the cart.', 'The reduced tag stays attached; the coins have not changed in this instant.'],
      ['Maya follows the coins as they shrink another step.', 'The expense coins are now smaller than in the previous beat.'],
      ['Maya lowers the scissors toward her side.', 'The small tag and reduced coins remain in their same positions.'],
      ['Maya turns her head from the tag to the coins.', 'The cart, shortened tag, and coins stay fixed.'],
      ['Maya holds still while looking over the reduced expenses.', 'The second reduction is complete; the props do not move.'],
    ],
    [
      ['Maya notices how tiny the price tag has become.', 'The tag is tiny beneath the scissors; the cart remains unchanged.'],
      ['Maya lifts the tiny tag slightly for a closer look.', 'The tag moves a short distance; the coins keep their reduced size.'],
      ['Maya turns the tag toward herself.', 'The tiny tag rotates in Maya’s hand without changing size.'],
      ['Maya lowers the tag and looks toward the expense coins.', 'The tiny tag returns beside the cart; the coins remain small.'],
      ['Maya watches the coins become tiny as well.', 'Each expense coin shrinks to a small dot-like coin.'],
      ['Maya sets the scissors down beside the cart.', 'The tiny tag and coins stay in place; scissors leave Maya’s hand.'],
      ['Maya straightens beside the finished cart.', 'All reduced expenses remain visible in the same white set.'],
      ['Maya smiles proudly at the tiny tag and coins.', 'The final reduced state holds without adding props.'],
    ],
  ],
  2: [
    [
      ['Maya stands inside the income elevator as it begins to rise.', 'The elevator starts low on the expense staircase; the stairs are still.'],
      ['The elevator carries Maya a short distance upward.', 'The elevator rises slightly while the staircase stays in place.'],
      ['Maya looks toward the staircase above the elevator.', 'The car continues upward beneath the first expense steps.'],
      ['The elevator rises past the first step.', 'The car gains height; bills and warning symbols remain above it.'],
      ['Maya braces as the elevator slows.', 'The elevator decelerates before reaching the top of the staircase.'],
      ['The elevator stops halfway up.', 'The income elevator is now stationary at mid-height.'],
      ['Maya looks up from inside the stopped elevator.', 'The elevator remains fixed; the expense staircase is unchanged.'],
      ['The lowest expense step extends upward above the car.', 'One stair segment rises while the elevator stays stopped.'],
      ['A second staircase segment lifts above the first.', 'The expense staircase is now higher than before.'],
      ['A bill climbs onto the newly raised step.', 'One bill moves upward along the staircase.'],
      ['A warning symbol follows the bill upward.', 'The bill and warning symbol occupy higher steps.'],
      ['Another step rises above the warning symbol.', 'The staircase extends farther above the stationary elevator.'],
      ['Maya tilts her head to follow the growing staircase.', 'The elevator stays fixed while the steps continue upward.'],
      ['The top of the expense staircase moves farther away.', 'The gap between elevator and staircase top increases.'],
      ['Maya remains inside the stalled elevator and looks up.', 'The elevator is stopped; the staircase is visibly higher.'],
    ],
    [
      ['Maya keeps her place in the motionless elevator.', 'The elevator stays at the same height as the previous beat.'],
      ['The first expense step extends higher above the car.', 'The staircase rises; the elevator does not move.'],
      ['A bill moves up onto the new step.', 'The bill advances while the elevator remains fixed.'],
      ['A second step lifts above the first.', 'The staircase gains another segment.'],
      ['A warning symbol climbs to the higher step.', 'The symbol follows the rising expenses.'],
      ['The staircase extends one more short segment.', 'The gap above the elevator grows again.'],
      ['Two bills shift upward along the staircase.', 'Bills occupy higher steps; the car stays stationary.'],
      ['Maya raises her eyes toward the new top step.', 'The growing staircase remains above the stopped elevator.'],
      ['The expense staircase reaches farther overhead.', 'Its upper end is now noticeably higher than before.'],
      ['Maya follows the last bill as it reaches the upper step.', 'The bill settles high above the fixed income elevator.'],
      ['Maya looks across the widening gap.', 'The income car is still; the expenses tower above it.'],
    ],
    [
      ['Maya’s eyebrows lift as she looks up from the stopped elevator.', 'The tall expense staircase remains fixed above the car.'],
      ['Maya’s shoulders dip while the income elevator stays still.', 'The rising-expense gap remains unchanged in the final reaction.'],
    ],
  ],
  3: [
    [
      ['Maya braces the full money sack beside the spending funnel.', 'The sack is full; the funnel and bill outlet are empty.'],
      ['Maya tightens her grip as the funnel catches the first coin.', 'One coin leaves the sack and enters the funnel.'],
      ['The funnel pulls a short stream of coins from the sack.', 'Coins travel from the full sack toward the funnel outlet.'],
      ['Maya looks down as the first coins reach the bills.', 'A few coins enter the funnel; the sack is still mostly full.'],
    ],
    [
      ['The spending funnel draws another coin from the sack.', 'A fresh coin joins the stream; the sack loses a little fullness.'],
      ['Maya shifts her hands to keep hold of the sack.', 'The sack mouth tilts toward the unchanged funnel.'],
      ['A small stream of money slides into the funnel.', 'More coins pass through the funnel throat.'],
      ['The sack begins to sag at its upper edge.', 'The sack is less full; coins are visible inside the funnel.'],
      ['The funnel releases the first oversized bill.', 'A bill appears below the funnel outlet.'],
      ['A second bill unfolds beneath the first.', 'The bill pile grows by one visible layer.'],
      ['Maya tightens her grip as more money leaves.', 'The sack shrinks further while the funnel stays fixed.'],
      ['Several coins pass through the funnel in succession.', 'The coin stream continues toward the expanding bill pile.'],
      ['Maya tracks the outflow with an anxious glance.', 'The sack is partly emptied; bills collect below.'],
    ],
    [
      ['The money stream continues into the spending funnel.', 'Only part of the money remains in the sack.'],
      ['The sack’s sides fold inward as it empties.', 'The sack becomes visibly slack.'],
      ['The last few coins leave the sack.', 'A small cluster of coins enters the funnel.'],
      ['The funnel fills with the incoming coins.', 'Coins crowd the funnel while the sack sags.'],
      ['Another oversized bill drops beneath the funnel.', 'The bill pile rises beside the earlier bills.'],
      ['Maya’s eyes widen as the pile grows.', 'The nearly empty sack and larger bill pile stay in view.'],
      ['Maya braces the sagging sack with both hands.', 'The sack hangs low with only a few coins left.'],
      ['The final coin slips toward the funnel.', 'The sack is almost empty; the funnel holds most of the money.'],
      ['Maya looks from the emptying sack to the bills.', 'The spending pile is now larger than the remaining sack contents.'],
    ],
    [
      ['Maya holds the limp, nearly empty sack beside the funnel.', 'The sack hangs slack; the funnel and bills remain full.'],
      ['Maya lowers the sack until its bottom droops.', 'The limp sack drops slightly in Maya’s hands.'],
      ['Maya follows the money path from the sack to the funnel.', 'The funnel remains stuffed above the oversized bill pile.'],
      ['Maya turns her gaze from the funnel to the bills.', 'The bill pile remains the largest object in the set.'],
      ['Maya’s shoulders sink as she sees the result.', 'The nearly empty sack stays beside the full spending funnel.'],
      ['Maya opens the sack mouth to check what remains.', 'The sack interior is visibly empty.'],
      ['Maya holds the empty sack still and stares at the bills.', 'No money remains in the sack; the outflow pile is unchanged.'],
      ['Maya keeps her alarmed gaze on the spending pile.', 'The consequence holds before the next sequence begins.'],
    ],
  ],
  4: [
    [
      ['Maya turns toward the balanced scale.', 'The scale, income coins, spending coins, and savings jar form one fixed set.'],
      ['Maya points to the modest income coin stack.', 'The income stack remains modest on its side of the scale.'],
      ['Maya traces her hand along the level scale beam.', 'The beam stays level between the two sides.'],
      ['Maya shifts her pointing hand toward the spending coins.', 'The smaller spending stack stays opposite the income coins.'],
      ['Maya looks from the spending coins back to the beam.', 'The scale remains balanced; the savings jar stays beside it.'],
      ['Maya lowers her hand and compares both stacks.', 'Both coin stacks remain unchanged and visible.'],
      ['Maya glances toward the savings jar beside the scale.', 'The jar is beside the balanced scale and is not yet fuller.'],
      ['Maya holds the comparison pose beside the level scale.', 'The balance between income and spending remains clear.'],
    ],
    [
      ['Maya raises one arm toward the income side.', 'The scale beam remains level; income coins stay modest.'],
      ['Maya points directly at the income coin stack.', 'The income coins do not increase.'],
      ['Maya sweeps her hand across the level beam.', 'The balance scale stays in the same position.'],
      ['Maya brings her hand to the smaller spending stack.', 'The spending coins remain fewer than the income coins.'],
      ['Maya compares the two sides with a short back-and-forth gesture.', 'Both stacks stay fixed on the level scale.'],
      ['Maya finishes with her hand between spending and the jar.', 'The arrow still leads from controlled spending toward savings.'],
    ],
    [
      ['Maya reaches toward the savings jar.', 'The balanced scale and both coin stacks remain unchanged.'],
      ['A first coin drops into the jar.', 'One coin appears inside the previously less-full jar.'],
      ['Two more coins settle into the jar.', 'The jar is visibly fuller while the scale stays level.'],
      ['Maya smiles beside the fuller jar.', 'The modest income and smaller spending stacks remain unchanged.'],
    ],
  ],
};

function storyboardActionsForSequence(sequenceId) {
  if (sequenceId === 1) return STORYBOARD_CHOREOGRAPHY[1][0];
  if (sequenceId === 2) return [
    ['Maya lowers the scissors out of view and looks upward as the scene cuts.', 'The cart, price tag, and scissors leave the composition; a clean white stage opens.'],
    ['Maya spots a simple elevator marked INCOME in the new white space.', 'A single elevator labeled INCOME stands ahead of Maya.'],
    ['Maya takes her first step toward the open elevator.', 'The INCOME elevator stays fixed as Maya moves closer.'],
    ['Maya reaches the elevator doorway.', 'Maya is at the open elevator; no expenses staircase is visible yet.'],
    ['Maya steps inside the elevator and faces upward.', 'Maya is inside the INCOME elevator at its starting height.'],
    ['The elevator begins to rise with Maya inside.', 'The elevator moves upward from its starting height.'],
    ['The elevator stops halfway up.', 'The INCOME elevator is stationary at mid-height.'],
    ['Maya studies the INCOME label on the stopped elevator.', 'The elevator remains stopped; the INCOME label is clear.'],
    ['Maya looks toward the empty space above the elevator.', 'The car is still; the area above it is not yet occupied.'],
    ...STORYBOARD_CHOREOGRAPHY[2][0].slice(7),
    ...STORYBOARD_CHOREOGRAPHY[2][1],
    ...STORYBOARD_CHOREOGRAPHY[2][2],
  ];
  if (sequenceId === 3 || sequenceId === 4) return STORYBOARD_CHOREOGRAPHY[sequenceId].flat();
  throw new Error(`No hay microacciones definidas para la secuencia narrativa ${sequenceId}.`);
}

function significantActionWords(action) {
  const stopWords = new Set(['maya', 'the', 'and', 'with', 'from', 'into', 'while', 'beside', 'toward', 'her', 'its', 'one', 'as', 'at', 'to']);
  return new Set(action.toLowerCase().match(/[a-z]+/g).filter((word) => word.length > 2 && !stopWords.has(word)));
}

function actionSimilarity(left, right) {
  const leftWords = significantActionWords(left), rightWords = significantActionWords(right);
  const overlap = [...leftWords].filter((word) => rightWords.has(word)).length;
  const union = new Set([...leftWords, ...rightWords]).size;
  return union ? overlap / union : 1;
}

async function planAIVoiceStoryboardActions(id) {
  const file = storyboardActionsFile(id);
  if (fs.existsSync(file)) {
    const saved = rd(file);
    if (saved.version === 2) return saved;
  }
  if (!openai) throw new Error('Falta OPENAI_API_KEY en .env');
  const dir = pdir(id), scenePlan = rd(path.join(dir, 'scene_plan.json')), sequencePlan = rd(sequencePlanFile(id));
  const source = sequencePlan.sequences.map((sequence) => ({
    sequence_id: sequence.sequence_id,
    scene: scenePlan.scenes.find((scene) => scene.scene_id === sequence.source_scene_id)?.visual_concept,
    narration: sequence.narration,
    beats: sequence.frames.map((frame) => ({ source_plan_beat_id: frame.frame_id, start: frame.start, end: frame.end,
      target_actions: Math.max(1, Math.min(12, Math.round((frame.end - frame.start) * STORYBOARD_FPS))),
      visual: frame.visual, prompt: frame.prompt, previous_frame_context: frame.previous_frame_context, continuity_notes: frame.continuity_notes })),
  }));
  const system = `Create a hand-drawn flipbook action plan from the supplied visual beats and word-timed narration. Every beat must expand into a short ordered list of distinct static micro-actions; target roughly 4 actions per second of beat duration, but use fewer when the story is calm and do not pad with repeated poses. Each action answers what the recurring protagonist Maya does immediately after the previous action. Keep one same-design stick-figure Maya physically present throughout the story, moving through settings as narration motivates; preserve her identity, clothes, proportions, and drawing style across scene cuts. Inside a sequence keep the environment, objects, and composition fixed unless the narration explicitly changes them. At a new sequence boundary, a narrative cut may establish the new scene while Maya remains the same person. Never invent a narrated event: small preparatory and consequential motions are allowed only when they directly represent a source beat. Each action is one legible instant, not a list of multiple future events. Adjacent actions must change a meaningful body pose, gaze, prop state, or position; prompts cannot simply rephrase a static composition. Preserve exact source_plan_beat_id on each group. action is an English sentence describing only the current instant. object_state states the visible prop condition for that instant. Return all input sequences and all source beats in order.`;
  const response = await openai.chat.completions.create({ model: M.re,
    response_format: { type: 'json_schema', json_schema: STORYBOARD_ACTION_SCHEMA },
    messages: [{ role: 'system', content: `${system}\nThe target_actions value for each source beat is exact: return exactly that many distinct actions for that beat.` },
      { role: 'user', content: JSON.stringify(source) }] });
  const generated = JSON.parse(response.choices[0].message.content);
  const generatedSequences = new Map(generated.sequences.map((sequence) => [sequence.sequence_id, sequence]));
  const errors = [];
  for (const sequence of sequencePlan.sequences) {
    const generatedSequence = generatedSequences.get(sequence.sequence_id);
    if (!generatedSequence) { errors.push(`Falta sequence ${sequence.sequence_id}`); continue; }
    const beatsById = new Map(generatedSequence.beats.map((beat) => [beat.source_plan_beat_id, beat]));
    for (const beat of sequence.frames) {
      const expanded = beatsById.get(beat.frame_id);
      if (!expanded?.actions?.length) errors.push(`Beat ${beat.frame_id} sin acciones`);
      const actions = expanded?.actions || [];
      const expectedActions = Math.max(1, Math.min(12, Math.round((beat.end - beat.start) * STORYBOARD_FPS)));
      if (actions.length !== expectedActions) errors.push(`Beat ${beat.frame_id}: ${actions.length} acciones; se esperaban ${expectedActions}`);
      if (actions.some((action) => !action.action.trim() || !action.object_state.trim())) errors.push(`Beat ${beat.frame_id} contiene estado vacío`);
      for (let index = 1; index < actions.length; index++) {
        if (actions[index].action === actions[index - 1].action || actionSimilarity(actions[index].action, actions[index - 1].action) >= 0.9)
          errors.push(`Beat ${beat.frame_id} repite una acción consecutiva`);
      }
    }
  }
  if (errors.length) throw new Error(`El plan de microacciones no pasó validación: ${errors.slice(0, 8).join('; ')}`);
  const result = { version: 2, ...generated };
  fs.writeFileSync(file, JSON.stringify(result, null, 2));
  return result;
}

function validateVisualStoryboard(storyboard, sourceBeatIds = []) {
  const issues = [];
  const frames = storyboard.sequences.flatMap((sequence) => sequence.frames.map((frame) => ({ ...frame, parentSequenceId: sequence.sequence_id })));
  const issue = (severity, code, message, frameIds = []) => issues.push({ severity, code, message, frame_ids: frameIds });
  const seenPrompts = new Map();
  let previousFrame = null;

  for (const [index, frame] of frames.entries()) {
    const nextFrame = frames[index + 1];
    if (!frame.action?.trim()) issue('error', 'missing_action', `Frame ${frame.frame_id} no tiene una acción concreta.`, [frame.frame_id]);
    if (!frame.source_beat_id || (sourceBeatIds.length && !sourceBeatIds.includes(frame.source_beat_id)))
      issue('error', 'invalid_source_beat', `Frame ${frame.frame_id} no apunta a un beat fuente válido.`, [frame.frame_id]);
    for (const field of ['previous_frame_state', 'current_frame_state', 'next_action', 'character_state', 'environment_state', 'object_state', 'camera_state', 'prompt']) {
      if (!frame[field]?.trim()) issue('error', `missing_${field}`, `Frame ${frame.frame_id} no tiene ${field}.`, [frame.frame_id]);
    }
    if (seenPrompts.has(frame.prompt)) issue('error', 'duplicate_prompt', `Frames ${seenPrompts.get(frame.prompt)} y ${frame.frame_id} tienen el mismo prompt.`, [seenPrompts.get(frame.prompt), frame.frame_id]);
    else seenPrompts.set(frame.prompt, frame.frame_id);
    if (previousFrame && frame.previous_frame_id !== previousFrame.frame_id)
      issue('error', 'wrong_previous_frame', `Frame ${frame.frame_id} no referencia el frame temporal anterior.`, [previousFrame.frame_id, frame.frame_id]);
    if (previousFrame && frame.previous_frame_state !== previousFrame.current_frame_state)
      issue('error', 'previous_state_mismatch', `El estado previo del frame ${frame.frame_id} no coincide con el estado actual anterior.`, [previousFrame.frame_id, frame.frame_id]);
    if (previousFrame && frame.parentSequenceId === previousFrame.parentSequenceId) {
      if (frame.environment_state !== previousFrame.environment_state)
        issue('error', 'unjustified_environment_change', `El entorno cambia dentro de la secuencia ${frame.parentSequenceId}.`, [previousFrame.frame_id, frame.frame_id]);
      if (frame.action === previousFrame.action)
        issue('error', 'duplicate_action', `Frames consecutivos ${previousFrame.frame_id} y ${frame.frame_id} repiten la acción.`, [previousFrame.frame_id, frame.frame_id]);
      if (frame.current_frame_state === previousFrame.current_frame_state)
        issue('warning', 'no_state_delta', `Frames ${previousFrame.frame_id} y ${frame.frame_id} no cambian el estado.`, [previousFrame.frame_id, frame.frame_id]);
      if (actionSimilarity(previousFrame.action, frame.action) >= 0.9)
        issue('warning', 'near_duplicate_action', `Frames ${previousFrame.frame_id} y ${frame.frame_id} tienen acciones casi iguales.`, [previousFrame.frame_id, frame.frame_id]);
    }
    if (previousFrame && frame.source_beat_id !== previousFrame.source_beat_id) {
      if (!frame.sequence_transition) issue('error', 'missing_beat_cut', `El beat ${frame.source_beat_id} comienza sin marcar un corte narrativo.`, [previousFrame.frame_id, frame.frame_id]);
      if (frame.action === previousFrame.action || actionSimilarity(previousFrame.action, frame.action) >= 0.9)
        issue('error', 'same_action_across_beats', `El frame ${frame.frame_id} conserva la acción del beat anterior.`, [previousFrame.frame_id, frame.frame_id]);
      if (frame.environment_state === previousFrame.environment_state)
        issue('warning', 'beat_without_new_environment', `El beat ${frame.source_beat_id} cambia sin establecer un entorno nuevo.`, [previousFrame.frame_id, frame.frame_id]);
    }
    if (nextFrame && frame.next_action !== nextFrame.action)
      issue('error', 'next_action_mismatch', `next_action del frame ${frame.frame_id} no coincide con la acción del frame ${nextFrame.frame_id}.`, [frame.frame_id, nextFrame.frame_id]);
    if (previousFrame && frame.character_identity !== previousFrame.character_identity)
      issue('error', 'character_identity_changed', `La identidad del personaje cambia entre frames ${previousFrame.frame_id} y ${frame.frame_id}.`, [previousFrame.frame_id, frame.frame_id]);
    if (Math.abs(frame.duration - (frame.end - frame.start)) > 0.0005 || frame.end <= frame.start)
      issue('error', 'duration_mismatch', `La duración del frame ${frame.frame_id} no coincide con end - start.`, [frame.frame_id]);
    previousFrame = frame;
  }

  let cursorMs = 0;
  for (const sequence of storyboard.sequences) {
    for (const frame of sequence.frames) {
      const startMs = Math.round(frame.start * 1000), endMs = Math.round(frame.end * 1000);
      if (startMs !== cursorMs) issue('error', startMs < cursorMs ? 'timeline_overlap' : 'timeline_gap', `Hay un hueco o solape antes del frame ${frame.frame_id}.`, [frame.frame_id]);
      cursorMs = endMs;
    }
  }
  if (cursorMs !== Math.round(storyboard.duration * 1000)) issue('error', 'timeline_end_mismatch', 'El storyboard no termina exactamente con su duración total.');
  return { status: issues.some((entry) => entry.severity === 'error') ? 'failed' : 'passed', errors: issues.filter((entry) => entry.severity === 'error').length,
    warnings: issues.filter((entry) => entry.severity === 'warning').length, issues };
}

function generateVisualStoryboard(id) {
  const dir = pdir(id);
  const planFile = path.join(dir, 'scene_plan.json');
  const sourcePlanFile = sequencePlanFile(id);
  const transcriptFile = path.join(dir, 'transcript', 'transcript.json');
  if (!fs.existsSync(planFile)) throw new Error('Primero debe existir scene_plan.json para generar el storyboard.');
  if (!fs.existsSync(sourcePlanFile)) throw new Error('Primero debe existir visual_sequence_plan.json para expandir sus beats sin cambiarlo.');
  const scenePlan = rd(planFile);
  const sourcePlan = rd(sourcePlanFile);
  const aiVoice = load(id).mode === 'ai_voice';
  const actionPlan = aiVoice ? rd(storyboardActionsFile(id)) : null;
  const transcript = fs.existsSync(transcriptFile) ? rd(transcriptFile) : { duration: sourcePlan.duration };
  const style = fs.existsSync(path.join(dir, 'style_bible.json')) ? rd(path.join(dir, 'style_bible.json')) : STYLE;
  const characterBible = buildCharacterBible(id);
  const worldBible = buildWorldBible(id);
  let frameId = 1;
  let previousFrame = null;
  let previousCharacterPose = 'Maya stands in a relaxed neutral pose.';
  const character = characterBible.characters.find((entry) => entry.name === characterBible.principal_character) || characterBible.characters[0];
  const characterIdentity = `${character.name}: ${character.appearance}; fixed clothes, silhouette, and proportions.`;
  const scenesById = new Map(scenePlan.scenes.map((scene) => [scene.scene_id, scene]));
  const sourceSequencesByScene = new Map(sourcePlan.sequences.map((sequence) => [sequence.source_scene_id || sequence.sequence_id, sequence]));
  const sequences = scenePlan.scenes.map((scene) => {
    const sourceSequence = sourceSequencesByScene.get(scene.scene_id);
    if (!sourceSequence) throw new Error(`No existe el plan de micro-beats para la escena ${scene.scene_id}.`);
    const setObjects = worldBible.sequence_objects.find((entry) => entry.sequence_id === scene.scene_id)?.objects || [];
    const setDescription = setObjects.length ? setObjects.join(', ') : 'the source scene objects';
    const environmentState = `Plain white background; wide static composition; fixed set with ${setDescription}. No location change within this sequence.`;
    const sceneStartMs = Math.round(scene.start * 1000), sceneEndMs = Math.round(scene.end * 1000);
    const frames = [];
    const plannedSequence = aiVoice ? actionPlan.sequences.find((entry) => entry.sequence_id === scene.scene_id) : null;
    const actionGroups = aiVoice
      ? sourceSequence.frames.map((sourceBeat) => {
        const beatPlan = plannedSequence?.beats.find((entry) => entry.source_plan_beat_id === sourceBeat.frame_id);
        if (!beatPlan?.actions?.length) throw new Error(`Falta expansión para el beat ${sourceBeat.frame_id}.`);
        return { sourceBeat, steps: beatPlan.actions.map((entry) => [entry.action, entry.object_state]) };
      })
      : [{ sourceBeat: null, steps: storyboardActionsForSequence(scene.scene_id) }];
    let sceneActionIndex = 0;
    const sceneActionCount = actionGroups.reduce((total, group) => total + group.steps.length, 0);
    actionGroups.forEach(({ sourceBeat: groupBeat, steps }) => steps.forEach(([action, objectState], actionIndex) => {
      const beatStartMs = groupBeat ? Math.round(groupBeat.start * 1000) : sceneStartMs;
      const beatEndMs = groupBeat ? Math.round(groupBeat.end * 1000) : sceneEndMs;
      const startMs = groupBeat
        ? beatStartMs + Math.floor((beatEndMs - beatStartMs) * actionIndex / steps.length)
        : sceneStartMs + Math.floor((sceneEndMs - sceneStartMs) * sceneActionIndex / sceneActionCount);
      const endMs = groupBeat
        ? beatStartMs + Math.floor((beatEndMs - beatStartMs) * (actionIndex + 1) / steps.length)
        : sceneStartMs + Math.floor((sceneEndMs - sceneStartMs) * (sceneActionIndex + 1) / sceneActionCount);
      const sourceTime = (startMs + endMs) / 2000;
      const sourceBeat = groupBeat || sourceSequence.frames.find((beat) => sourceTime >= beat.start && sourceTime < beat.end) || sourceSequence.frames.at(-1);
      if (/^Maya\b/.test(action)) previousCharacterPose = action;
      const characterState = `${characterIdentity} Current pose/action: ${previousCharacterPose}`;
      const currentFrameState = `Character: ${characterState} Environment: ${environmentState} Objects: ${objectState}`;
      const sequenceTransition = !previousFrame || previousFrame.sequence_id !== scene.scene_id;
      const previousFrameState = previousFrame ? previousFrame.current_frame_state : `Opening state: ${characterState}`;
      const frame = {
        frame_id: frameId++,
        sequence_id: scene.scene_id,
        source_beat_id: scene.scene_id,
        source_plan_beat_id: sourceBeat.frame_id,
        previous_frame_id: previousFrame?.frame_id ?? null,
        previous_frame_state: previousFrameState,
        current_frame_state: currentFrameState,
        start: startMs / 1000,
        end: endMs / 1000,
        duration: (endMs - startMs) / 1000,
        action,
        visual_action: action,
        next_action: '',
        character_identity: character.name,
        character_state: characterState,
        object_state: objectState,
        environment_state: environmentState,
        camera_state: 'Static wide 16:9 framing; no pan, zoom, or camera movement.',
        continuity_notes: sequenceTransition
          ? 'Narrative cut: preserve Maya identity and proportions; establish the new environment and dominant action for this scene.'
          : `Keep the established environment and composition. Change only what this action requires; source visual: ${sourceBeat.visual}.`,
        sequence_transition: sequenceTransition,
        prompt: [
          `Draw one static frame in this exact approved style: ${style.visual_style}; ${style.line_style}; ${style.character_style}.`,
          `Action at this instant: ${action}`,
          `Character state: ${characterState}`,
          `Object state: ${objectState}`,
          `Environment: ${environmentState}`,
          previousFrame ? `Previous frame state: ${previousFrameState}` : 'Opening frame of this visual story.',
          'Preserve the same character design, simple flat colors, empty white background, and wide composition. Do not add a character, prop, location, or event not supported by the source scene.',
        ].join('\n'),
      };
      frames.push(frame);
      previousFrame = frame;
      sceneActionIndex++;
    }));
    return {
      sequence_id: scene.scene_id,
      source_scene_id: scene.scene_id,
      title: inferSequenceTitle(scene),
      start: scene.start,
      end: scene.end,
      duration: +(scene.end - scene.start).toFixed(3),
      story_arc: scene.visual_reason || scene.visual_concept,
      environment_state: environmentState,
      source_beat_count: 1,
      source_microbeat_count: sourceSequence.frames.length,
      frames,
    };
  });
  const allFrames = sequences.flatMap((sequence) => sequence.frames);
  allFrames.forEach((frame, index) => {
    frame.next_action = allFrames[index + 1]?.action || 'Hold the final pose as the narration ends.';
    const previous = allFrames[index - 1];
    const linked = !previous || (frame.previous_frame_id === previous.frame_id && frame.previous_frame_state === previous.current_frame_state);
    const sameCharacter = !previous || frame.character_identity === previous.character_identity;
    const sameSequence = !previous || frame.sequence_id === previous.sequence_id;
    const sameEnvironment = !previous || !sameSequence || frame.environment_state === previous.environment_state;
    const objectContinuity = !previous || (sameSequence ? 1 : 0.6);
    const changedState = !previous || frame.current_frame_state !== previous.current_frame_state;
    frame.continuity_score = Number((0.35 * Number(sameCharacter) + 0.25 * Number(sameEnvironment) + 0.2 * objectContinuity + 0.1 * Number(linked) + 0.1 * Number(changedState)).toFixed(2));
  });
  const totalFrames = sequences.reduce((total, sequence) => total + sequence.frames.length, 0);
  const sourceBeatIds = scenePlan.scenes.map((scene) => scene.scene_id);
  const sourceMicrobeatCount = sourcePlan.sequences.reduce((total, sequence) => total + sequence.frames.length, 0);
  const storyboard = {
    version: 1,
    model: 'deterministic-action-progression',
    created_at: new Date().toISOString(),
    project_id: id,
    duration: +(transcript.duration || sourcePlan.duration).toFixed(3),
    story_arc: 'Each source beat becomes a timed micro-sequence of concrete actions; character identity persists while each sequence controls its own environment.',
    style_lock: style,
    character_bible: characterBible,
    world_bible: worldBible,
    sequences,
    total_frames: totalFrames,
    average_frame_duration: totalFrames ? +(transcript.duration / totalFrames).toFixed(3) : 0,
    estimated_image_generations: totalFrames,
    source_beat_count: sourceBeatIds.length,
    source_microbeat_count: sourceMicrobeatCount,
    base_frames_per_second: STORYBOARD_FPS,
    continuity_score_method: 'Heuristic structural consistency only; this is not an overall image-quality score.',
    validation: validateVisualStoryboard({ duration: +(transcript.duration || sourcePlan.duration).toFixed(3), sequences }, sourceBeatIds),
    review_gate: 'No images were generated. Review the action progression before any image generation.',
  };
  fs.writeFileSync(path.join(dir, 'character_bible.json'), JSON.stringify(characterBible, null, 2));
  fs.writeFileSync(path.join(dir, 'world_bible.json'), JSON.stringify(worldBible, null, 2));
  fs.writeFileSync(storyboardFile(id), JSON.stringify(storyboard, null, 2));
  return storyboard;
}

function framePrompt(id, scene, beat) {
  const dir = pdir(id), bible = rd(path.join(dir, 'style_bible.json'));
  const characterFile = path.join(dir, 'characters.json');
  const characters = (fs.existsSync(characterFile) ? rd(characterFile) : []).filter((c) => (scene.character_ids || []).includes(c.id));
  return [
    `Sequence context: ${scene.visual_concept}. Narration: ${scene.narration}`,
    `Frame visual: ${beat.visual}. Draw this exact moment: ${beat.prompt}`,
    characters.length ? `Recurring characters, keep exactly consistent: ${characters.map((c) => `${c.name}: ${c.description}`).join('; ')}.` : '',
    beat.previous_frame_context ? `Previous frame: ${beat.previous_frame_context}` : 'Opening frame of this sequence.',
    beat.continuity_notes ? `Continuity: ${beat.continuity_notes}` : '',
    `Approved style bible: ${bible.visual_style}; ${bible.line_style}; ${bible.character_style}; ${bible.color_policy}; ${bible.background_policy}.`,
    `Composition: ${bible.composition_policy}. Text: ${bible.text_policy}. Avoid: ${bible.negative}.`,
  ].filter(Boolean).join('\n');
}
function summarizeVisualSequences(id) {
  const file = sequencePlanFile(id);
  if (!fs.existsSync(file)) return null;
  const plan = rd(file), state = load(id), frames = plan.sequences.flatMap((sequence) => sequence.frames);
  const statusOf = (frame) => {
    const saved = state.visualFrameImages?.[frame.frame_id];
    if (saved?.status === 'completed' && saved.file && fs.existsSync(path.join(pdir(id), saved.file))) return 'completed';
    return frame.reused && frame.file && fs.existsSync(path.join(pdir(id), frame.file)) ? 'completed' : saved?.status || 'pending';
  };
  const reusable = frames.filter((frame) => frame.reused && statusOf(frame) === 'completed').length;
  const generated = frames.filter((frame) => !frame.reused && statusOf(frame) === 'completed').length;
  const pending = frames.filter((frame) => !frame.reused && statusOf(frame) !== 'completed').length;
  return { duration: plan.duration, sequences: plan.sequences.length, frames: frames.length,
    averageFrameDuration: frames.length ? +(plan.duration / frames.length).toFixed(2) : 0,
    reusedFrames: reusable, generatedFrames: generated, imagesToGenerate: pending, estimatedImageGenerations: pending,
    testFramesRequested: state.visualFrameTest?.requested || 0, testFramesGenerated: state.visualFrameTest?.frame_ids?.filter((id) => statusOf(frames.find((frame) => frame.frame_id === id) || {}) === 'completed').length || 0,
    testRender: state.visualFrameTest?.rendered ? state.visualFrameTest.output : null };
}
function visualSequenceView(id) {
  const planFile = sequencePlanFile(id);
  if (!fs.existsSync(planFile)) return null;
  const plan = rd(planFile), state = load(id);
  return { ...plan, sequences: plan.sequences.map((sequence) => ({ ...sequence, frames: sequence.frames.map((frame) => {
    const saved = state.visualFrameImages?.[frame.frame_id];
    if (saved?.status === 'completed' && saved.file && fs.existsSync(path.join(pdir(id), saved.file)))
      return { ...frame, status: 'completed', file: saved.file, testGenerated: true, error: null };
    if (frame.reused && frame.file && fs.existsSync(path.join(pdir(id), frame.file)))
      return { ...frame, status: 'completed', reused: true, testGenerated: false };
    return { ...frame, status: saved?.status || 'pending', file: null, testGenerated: false, error: saved?.error || null };
  }) })) };
}
function chooseTestFrames(id, count = 5) {
  const plan = visualSequenceView(id);
  if (!plan) throw new Error('Primero crea el visual sequence plan');
  const pending = plan.sequences.flatMap((sequence) => sequence.frames.filter((frame) => !frame.reused && frame.status === 'pending')
    .map((frame) => ({ ...frame, sequence_id: sequence.sequence_id })));
  if (pending.length < count) throw new Error(`Sólo quedan ${pending.length} frames pendientes; no se puede generar ${count} nuevos`);
  const groups = new Map();
  pending.forEach((frame) => groups.set(frame.sequence_id, [...(groups.get(frame.sequence_id) || []), frame]));
  const largest = [...groups.entries()].sort((a, b) => b[1].length - a[1].length || a[0] - b[0])[0][1];
  const selected = [...largest].sort((a, b) => a.start - b.start);
  const rest = pending.filter((frame) => !selected.some((chosen) => chosen.frame_id === frame.frame_id));
  while (selected.length < count) {
    const center = (selected[0].start + selected.at(-1).end) / 2;
    rest.sort((a, b) => Math.abs((a.start + a.end) / 2 - center) - Math.abs((b.start + b.end) / 2 - center));
    selected.push(rest.shift());
  }
  return selected.sort((a, b) => a.start - b.start);
}
async function generateTestFrames(id) {
  if (running.has(id)) throw new Error('Ya hay un proceso en curso para este proyecto');
  if (!openai) throw new Error('Falta OPENAI_API_KEY en .env');
  const state = load(id);
  if (state.visualFrameTest) throw new Error('La generación de 5 frames de prueba ya fue ejecutada');
  const selected = chooseTestFrames(id, 5), dir = pdir(id);
  fs.mkdirSync(path.join(dir, 'frames'), { recursive: true });
  running.add(id);
  patch(id, (current) => {
    current.visualFrameImages ||= {};
    current.visualFrameTest = { requested: 5, frame_ids: selected.map((frame) => frame.frame_id), started_at: new Date().toISOString(), status: 'generating' };
    for (const frame of selected) current.visualFrameImages[frame.frame_id] = { status: 'generating', sequence_id: frame.sequence_id, error: null };
  });
  try {
    for (const frame of selected) {
      const rel = `frames/frame_${pad(frame.frame_id)}.png`;
      log(id, `[TEST FRAME ${pad(frame.frame_id)}] Generating once (${frame.sequence_id} sequence)`);
      try {
        const response = await openai.images.generate({ model: M.im, prompt: frame.prompt, size: IMG_SIZE, quality: IMG_QUALITY, n: 1 });
        const image = response.data[0]?.b64_json;
        if (!image) throw new Error('Respuesta sin b64_json');
        fs.writeFileSync(path.join(dir, rel), Buffer.from(image, 'base64'));
        patch(id, (current) => { current.visualFrameImages[frame.frame_id] = { status: 'completed', sequence_id: frame.sequence_id,
          file: rel, model: M.im, size: IMG_SIZE, timestamp: new Date().toISOString(), error: null }; });
        log(id, `[TEST FRAME ${pad(frame.frame_id)}] Completed`);
      } catch (error) {
        patch(id, (current) => { current.visualFrameImages[frame.frame_id] = { status: 'failed', sequence_id: frame.sequence_id, error: redact(error.message) }; });
        log(id, `[TEST FRAME ${pad(frame.frame_id)}] Failed: ${error.message}`);
      }
    }
    const current = load(id), generated = selected.filter((frame) => current.visualFrameImages[frame.frame_id]?.status === 'completed');
    patch(id, (next) => { next.visualFrameTest.status = generated.length === 5 ? 'completed' : 'partial'; next.visualFrameTest.completed_at = new Date().toISOString(); });
    return { requested: 5, generated: generated.length, frame_ids: selected.map((frame) => frame.frame_id), sequence_ids: [...new Set(selected.map((frame) => frame.sequence_id))] };
  } finally { running.delete(id); }
}
const srtTimestamp = (seconds) => {
  const ms = Math.max(0, Math.round(seconds * 1000)), h = Math.floor(ms / 3600000), m = Math.floor(ms % 3600000 / 60000), s = Math.floor(ms % 60000 / 1000), fraction = ms % 1000;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(fraction).padStart(3, '0')}`;
};
async function renderVisualTest(id) {
  if (!ffmpegOk() || !ffprobeOk()) assertTools();
  const dir = pdir(id), state = load(id), plan = visualSequenceView(id), test = state.visualFrameTest;
  if (!test || test.status !== 'completed' || test.frame_ids?.length !== 5) throw new Error('Primero deben completarse los 5 frames de prueba');
  const byId = new Map(plan.sequences.flatMap((sequence) => sequence.frames.map((frame) => [frame.frame_id, frame])));
  const frames = test.frame_ids.map((frameId) => byId.get(frameId));
  if (frames.some((frame) => !frame || frame.status !== 'completed' || !frame.file || !fs.existsSync(path.join(dir, frame.file))))
    throw new Error('Falta una imagen de los 5 frames de prueba');
  const sorted = frames.sort((a, b) => a.start - b.start), temp = path.join(dir, 'output', '_visual_test');
  fs.rmSync(temp, { recursive: true, force: true }); fs.mkdirSync(temp, { recursive: true });
  const transcriptFile = path.join(dir, 'transcript', 'transcript.json'), transcript = fs.existsSync(transcriptFile) ? rd(transcriptFile) : { segments: [] };
  const subtitleLines = [], videoLines = [], audioFilters = [];
  let outputCursor = 0;
  for (const frame of sorted) {
    const words = (transcript.words || []).filter((word) => word.end > frame.start && word.start < frame.end)
      .map((word) => ({ start: Math.max(frame.start, word.start), end: Math.min(frame.end, word.end), text: word.word.trim() }))
      .filter((word) => word.text && word.end > word.start);
    if (words.length) {
      let group = [];
      const writeGroup = () => {
        if (!group.length) return;
        const cueStart = outputCursor + group[0].start - frame.start, cueEnd = outputCursor + group.at(-1).end - frame.start;
        const text = group.map((word) => word.text).join(' ').replace(/\s+([,.;!?])/g, '$1');
        if (cueEnd > cueStart) subtitleLines.push(`${subtitleLines.length + 1}\n${srtTimestamp(cueStart)} --> ${srtTimestamp(cueEnd)}\n${text}\n`);
        group = [];
      };
      for (const word of words) {
        const gap = group.length ? word.start - group.at(-1).end : 0;
        const previous = group.at(-1)?.text || '';
        if (group.length && (group.length >= 7 || gap > 0.6 || /[,.;!?]$/.test(previous))) writeGroup();
        group.push(word);
      }
      writeGroup();
    } else {
      const cue = transcript.segments.find((segment) => segment.start >= frame.start && segment.start < frame.end);
      if (cue) subtitleLines.push(`${subtitleLines.length + 1}\n${srtTimestamp(outputCursor)} --> ${srtTimestamp(outputCursor + frame.duration)}\n${cue.text.trim()}\n`);
    }
    outputCursor += frame.duration;
  }
  const subtitleFile = path.join(dir, 'output', 'visual_sequence_test.srt');
  fs.writeFileSync(subtitleFile, subtitleLines.join('\n'));
  try {
    for (const [index, frame] of sorted.entries()) {
      const filename = `frame_${pad(frame.frame_id)}.mp4`, frameCount = Math.max(1, Math.round(frame.duration * 30));
      await run(FFMPEG, ['-y', '-loop', '1', '-framerate', '30', '-i', frame.file, '-vf', motionExpressions(frame.motion, frameCount),
        '-frames:v', String(frameCount), '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', '-r', '30', path.join(temp, filename)], { cwd: dir });
      videoLines.push(`file '${path.join(temp, filename).replace(/\\/g, '/')}'`);
      audioFilters.push(`[1:a]atrim=start=${frame.start.toFixed(3)}:end=${frame.end.toFixed(3)},asetpts=PTS-STARTPTS[a${index}]`);
    }
    const concatFile = path.join(temp, 'concat.txt');
    fs.writeFileSync(concatFile, videoLines.join('\n'));
    const filter = `${audioFilters.join(';')};${sorted.map((_frame, index) => `[a${index}]`).join('')}concat=n=5:v=0:a=1[aout]`;
    await run(FFMPEG, ['-y', '-f', 'concat', '-safe', '0', '-i', concatFile, '-i', `audio/${state.audioFile}`,
      '-filter_complex', filter, '-map', '0:v:0', '-map', '[aout]', '-vf', "subtitles=output/visual_sequence_test.srt:force_style='FontSize=18,Outline=2,MarginV=40'",
      '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-r', '30', '-c:a', 'aac', '-b:a', '192k',
      '-t', outputCursor.toFixed(3), '-movflags', '+faststart', 'output/visual_sequence_test.mp4'], { cwd: dir });
    const outputFile = path.join(dir, 'output', 'visual_sequence_test.mp4');
    const probe = spawnSync(FFPROBE, ['-v', 'error', '-show_entries', 'stream=codec_name,width,height,r_frame_rate,duration', '-of', 'json', outputFile]);
    if (probe.status !== 0) throw new Error('FFprobe no pudo validar el render de prueba');
    const streams = JSON.parse(probe.stdout.toString()).streams;
    if (!streams.some((stream) => stream.codec_name === 'h264' && stream.width === 1920 && stream.height === 1080 && stream.r_frame_rate === '30/1'))
      throw new Error('El render de prueba no quedó en H.264 1920x1080 30 fps');
    if (!streams.some((stream) => stream.codec_name === 'aac')) throw new Error('El render de prueba no contiene audio AAC');
    patch(id, (current) => { current.visualFrameTest.rendered = true; current.visualFrameTest.output = 'output/visual_sequence_test.mp4'; });
    log(id, '[VISUAL TEST RENDER] Completed (5 frames, source audio intervals and shifted subtitles)');
    return { output: 'output/visual_sequence_test.mp4', duration: +outputCursor.toFixed(3), frames: sorted.map((frame) => frame.frame_id) };
  } finally { fs.rmSync(temp, { recursive: true, force: true }); }
}

async function generateStoryboardSampleFrames(id) {
  if (!openai) throw new Error('Falta OPENAI_API_KEY en .env');
  const storyboard = rd(storyboardFile(id));
  if (storyboard.validation?.status !== 'passed') throw new Error('Corrige la auditoría del storyboard antes de generar imágenes.');
  const sequence = storyboard.sequences.find((entry) => entry.sequence_id === 1);
  const frames = sequence?.frames.slice(0, 10) || [];
  if (frames.length !== 10 || frames.some((frame) => frame.sequence_id !== 1))
    throw new Error('La primera secuencia debe tener diez frames consecutivos disponibles.');
  const relativeDir = 'frames/ai_voice';
  const outputDir = path.join(pdir(id), relativeDir);
  fs.mkdirSync(outputDir, { recursive: true });
  const generated = [];

  for (const [index, frame] of frames.entries()) {
    const imageName = `frame_${pad(index + 1)}.png`;
    const outputFile = path.join(outputDir, imageName);
    const previousFile = index > 0 ? path.join(outputDir, `frame_${pad(index)}.png`) : null;
    if (fs.existsSync(outputFile)) {
      if (!fs.statSync(outputFile).size) throw new Error(`La imagen existente está vacía: ${relativeDir}/${imageName}`);
      generated.push({ frame_id: frame.frame_id, file: `${relativeDir}/${imageName}`, reused: true });
      continue;
    }
    if (previousFile && !fs.existsSync(previousFile))
      throw new Error(`No se puede continuar sin la imagen de referencia anterior: ${path.basename(previousFile)}`);

    const referenceGuidance = frame.sequence_transition
      ? 'Use the preceding image only as a reference for Maya identity, proportions, and drawing style. This is a new narrated scene: reset the environment and props to the new scene while retaining the same Maya.'
      : 'Use the preceding image as the visual reference. Preserve Maya, the full composition, object positions, line style, colors, and scale. Change only what the stated action requires for this next instant. No camera movement, no new props, and no redesign.';
    const prompt = [
      frame.prompt,
      `Previous frame state: ${frame.previous_frame_state}`,
      `Current frame state: ${frame.current_frame_state}`,
      `Continuity notes: ${frame.continuity_notes}`,
      `Camera: ${frame.camera_state}`,
      referenceGuidance,
    ].join('\n');
    log(id, `[STORYBOARD SAMPLE ${pad(index + 1)}] ${previousFile ? 'Editing from previous frame' : 'Generating opening frame'}`);
    const response = previousFile
      ? await openai.images.edit({ model: M.im, image: await toFile(fs.createReadStream(previousFile), path.basename(previousFile), { type: 'image/png' }), prompt, size: IMG_SIZE, quality: IMG_QUALITY, n: 1 })
      : await openai.images.generate({ model: M.im, prompt, size: IMG_SIZE, quality: IMG_QUALITY, n: 1 });
    const image = response.data[0]?.b64_json;
    if (!image) throw new Error(`La respuesta para el frame ${index + 1} no contiene b64_json.`);
    fs.writeFileSync(outputFile, Buffer.from(image, 'base64'));
    generated.push({ frame_id: frame.frame_id, file: `${relativeDir}/${imageName}`, reused: false });
    log(id, `[STORYBOARD SAMPLE ${pad(index + 1)}] Completed`);
  }
  return { model: M.im, reference_mode: 'images.edit using the immediately previous PNG', frames: generated };
}

async function renderStoryboardSample(id) {
  if (!ffmpegOk() || !ffprobeOk()) assertTools();
  const dir = pdir(id), storyboard = rd(storyboardFile(id)), state = load(id);
  const sequence = storyboard.sequences.find((entry) => entry.sequence_id === 1);
  const frames = sequence?.frames.slice(0, 10) || [];
  if (frames.length !== 10) throw new Error('La primera secuencia debe contener diez frames consecutivos.');
  const relativeDir = 'frames/ai_voice';
  const imageFiles = frames.map((_frame, index) => path.join(relativeDir, `frame_${pad(index + 1)}.png`));
  if (imageFiles.some((file) => !fs.existsSync(path.join(dir, file)))) throw new Error('Faltan imágenes de la muestra de diez frames.');
  const outputName = 'output/test_10_frames.mp4';
  const subtitleName = 'output/test_10_frames.srt';
  const outputFile = path.join(dir, outputName);
  const sortedFrames = [...frames].sort((left, right) => left.start - right.start);
  const firstStart = sortedFrames[0].start, lastEnd = sortedFrames.at(-1).end;
  const duration = +(lastEnd - firstStart).toFixed(3);
  const transcriptFile = path.join(dir, 'transcript', 'transcript.json');
  const transcript = fs.existsSync(transcriptFile) ? rd(transcriptFile) : { words: [], segments: [] };
  const subtitleLines = [];
  const words = (transcript.words || []).filter((word) => word.end > firstStart && word.start < lastEnd)
    .map((word) => ({ start: Math.max(firstStart, word.start) - firstStart, end: Math.min(lastEnd, word.end) - firstStart, text: word.word.trim() }))
    .filter((word) => word.text && word.end > word.start);
  let group = [];
  const flushGroup = () => {
    if (!group.length) return;
    const text = group.map((word) => word.text).join(' ').replace(/\s+([,.;!?])/g, '$1');
    subtitleLines.push(`${subtitleLines.length + 1}\n${srtTimestamp(group[0].start)} --> ${srtTimestamp(group.at(-1).end)}\n${text}\n`);
    group = [];
  };
  for (const word of words) {
    const gap = group.length ? word.start - group.at(-1).end : 0;
    if (group.length && (group.length >= 7 || gap > 0.6 || /[,.;!?]$/.test(group.at(-1).text))) flushGroup();
    group.push(word);
  }
  flushGroup();
  if (!subtitleLines.length) {
    const segments = (transcript.segments || []).filter((segment) => segment.end > firstStart && segment.start < lastEnd);
    segments.forEach((segment) => subtitleLines.push(`${subtitleLines.length + 1}\n${srtTimestamp(Math.max(0, segment.start - firstStart))} --> ${srtTimestamp(Math.min(duration, segment.end - firstStart))}\n${segment.text.trim()}\n`));
  }
  fs.writeFileSync(path.join(dir, subtitleName), subtitleLines.join('\n'));

  const temp = path.join(dir, 'output', '_test_10_frames');
  fs.rmSync(temp, { recursive: true, force: true });
  fs.mkdirSync(temp, { recursive: true });
  try {
    const concatLines = [];
    for (const [index, frame] of sortedFrames.entries()) {
      const startFrame = Math.round((frame.start - firstStart) * 30);
      const endFrame = Math.round((frame.end - firstStart) * 30);
      const frameCount = Math.max(1, endFrame - startFrame);
      const clipName = `frame_${pad(index + 1)}.mp4`;
      await run(FFMPEG, ['-y', '-loop', '1', '-framerate', '30', '-i', imageFiles[index], '-frames:v', String(frameCount),
        '-vf', 'scale=1920:1080,setsar=1,format=yuv420p', '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
        '-r', '30', path.join(temp, clipName)], { cwd: dir });
      concatLines.push(`file '${path.join(temp, clipName).replace(/\\/g, '/')}'`);
    }
    const concatFile = path.join(temp, 'concat.txt');
    fs.writeFileSync(concatFile, concatLines.join('\n'));
    const audioEnd = (firstStart + duration).toFixed(3);
    await run(FFMPEG, ['-y', '-f', 'concat', '-safe', '0', '-i', concatFile, '-i', `audio/${state.audioFile}`,
      '-filter_complex', `[1:a]atrim=start=${firstStart.toFixed(3)}:end=${audioEnd},asetpts=PTS-STARTPTS[aout]`,
      '-map', '0:v:0', '-map', '[aout]', '-vf', `subtitles=${subtitleName}:force_style='FontSize=18,Outline=2,MarginV=40'`,
      '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-r', '30', '-c:a', 'aac', '-b:a', '192k',
      '-t', duration.toFixed(3), '-movflags', '+faststart', outputName], { cwd: dir });
  } finally { fs.rmSync(temp, { recursive: true, force: true }); }

  const probe = spawnSync(FFPROBE, ['-v', 'error', '-show_entries', 'stream=codec_name,width,height,r_frame_rate,duration:format=duration', '-of', 'json', outputFile]);
  if (probe.status !== 0) throw new Error('FFprobe no pudo validar el render de la muestra de storyboard.');
  const media = JSON.parse(probe.stdout.toString()), streams = media.streams;
  if (!streams.some((stream) => stream.codec_name === 'h264' && stream.width === 1920 && stream.height === 1080 && stream.r_frame_rate === '30/1'))
    throw new Error('La muestra debe quedar en H.264 1920x1080 a 30 fps.');
  if (!streams.some((stream) => stream.codec_name === 'aac')) throw new Error('La muestra debe conservar audio AAC.');
  return { output: outputName, subtitles: subtitleName, duration: +Number(media.format.duration).toFixed(3), frames: imageFiles };
}

async function planVisualSequences(id) {
  const file = sequencePlanFile(id);
  if (fs.existsSync(file)) return rd(file);
  const dir = pdir(id), sceneFile = path.join(dir, 'scene_plan.json'), transcriptFile = path.join(dir, 'transcript', 'transcript.json');
  if (!fs.existsSync(sceneFile) || !fs.existsSync(transcriptFile)) throw new Error('Primero deben existir scene_plan.json y transcript.json');
  const scenePlan = ensureMotionPlan(id), transcript = rd(transcriptFile), state = load(id);
  const styleFile = path.join(dir, 'style_bible.json');
  if (!fs.existsSync(styleFile)) fs.writeFileSync(styleFile, JSON.stringify(STYLE, null, 2));
  const charactersFile = path.join(dir, 'characters.json');
  const characters = fs.existsSync(charactersFile) ? rd(charactersFile) : [];
  const input = { duration: transcript.duration, characters, scenes: scenePlan.scenes.map((scene) => ({
    sequence_id: scene.scene_id, start: scene.start, end: scene.end, narration: scene.narration,
    visual_concept: scene.visual_concept, visual_reason: scene.visual_reason, current_visual_prompt: scene.prompt,
    character_ids: scene.character_ids, importance: scene.importance,
    word_timestamps: (transcript.words || []).filter((word) => word.start >= scene.start && word.start < scene.end)
      .map((word) => ({ start: +word.start.toFixed(3), text: word.word })),
  })) };
  const aiVoiceRules = state.mode === 'ai_voice'
    ? ' AI VOICE CONTINUITY: Maya is the one recurring stick-figure narrator and remains physically present through virtually the whole story. Across consecutive scenes, carry her position, gaze, pose, and carried props forward; when the narrative concept changes, stage a clear visual cut while keeping Maya identity, clothes, proportions, and drawing style unchanged. Never replace her with a collection of unrelated subject illustrations.' : '';
  const system = `You are an experienced YouTube editor planning static drawing frames for a hand-made flipbook sequence. Preserve each input scene as one narrative sequence; create multiple frames only for useful visual beats (action, reaction, object appearing/disappearing, consequence, metaphor, contrast, emphasis, joke). Never create a frame just because a word changed. Every prompt describes a static drawing, never video or character animation. Adjacent frames in one sequence must be small, coherent changes: same characters, clothes, colors, proportions, objects and composition unless the narration motivates a change. First frame is the opening state and should match current_visual_prompt. Use the exact approved style bible supplied in the input as visual constraints, but keep the frame prompt focused on content. Choose timing offsets relative to sequence start using the supplied word timestamps; frame 1 starts at offset 0, all later offsets should align near a spoken word start. Use variable pacing: calm explanation usually 2-3 s per frame, normal 1-2 s, visible actions 0.5-1.2 s, and only intentional punchline cuts may be shorter than 0.3 s. Do not force these rates when no new visual beat exists. Choose subtle camera motion, amount normally 0.01-0.03 and never above 0.04; camera is secondary to the drawing changes. Select different motion types for adjacent frames when useful; static is valid. Return one sequence for each input scene with its sequence_id unchanged. Frame prompt must be English and describe the single static drawing. previous_frame_context and continuity_notes should explain only what is needed to preserve continuity.${aiVoiceRules}`;
  log(id, '[VISUAL SEQUENCES] Planning started');
  const response = await openai.chat.completions.create({ model: M.re, response_format: { type: 'json_schema', json_schema: SEQUENCE_SCHEMA },
    messages: [{ role: 'system', content: system }, { role: 'user', content: `Approved style bible:\n${JSON.stringify(rd(styleFile))}\nExisting characters and scene/timestamp data:\n${JSON.stringify(input)}` }] });
  const generated = JSON.parse(response.choices[0].message.content);
  const generatedById = new Map(generated.sequences.map((sequence) => [sequence.sequence_id, sequence]));
  const outputSequences = [];
  let nextFrameId = 1, previousMotion = null;
  for (const scene of [...scenePlan.scenes].sort((a, b) => a.start - b.start)) {
    const raw = [...(generatedById.get(scene.scene_id)?.frames || [])].sort((a, b) => a.start_offset - b.start_offset);
    if (!raw.length) raw.push({ start_offset: 0, visual: scene.visual_concept, prompt: scene.prompt,
      previous_frame_context: '', continuity_notes: '', motion_type: scene.motion?.type || 'static', motion_amount: 0.02, intentional_fast: false });
    const startMs = Math.round(scene.start * 1000), endMs = Math.round(scene.end * 1000), durationMs = endMs - startMs;
    const wordStarts = (transcript.words || []).filter((word) => word.start >= scene.start && word.start < scene.end).map((word) => Math.round(word.start * 1000));
    const beatStarts = [{ beat: raw[0], at: startMs }];
    let cursor = startMs;
    for (const beat of raw.slice(1)) {
      const target = startMs + Math.max(0, Math.min(durationMs, Math.round(Number(beat.start_offset) * 1000)));
      if (target >= endMs) continue;
      const snapped = wordStarts.length ? wordStarts.reduce((best, value) => Math.abs(value - target) < Math.abs(best - target) ? value : best) : target;
      if (snapped <= cursor || (snapped - cursor < 300 && !beat.intentional_fast)) continue;
      beatStarts.push({ beat, at: snapped });
      cursor = snapped;
    }
    if (beatStarts.length > 1 && endMs - beatStarts.at(-1).at < 300 && !beatStarts.at(-1).beat.intentional_fast) beatStarts.pop();
    const sceneState = state.scenes.find((entry) => entry.scene_id === scene.scene_id);
    const reusableFile = sceneState?.status === 'completed' && sceneState.file && fs.existsSync(path.join(dir, sceneState.file)) ? sceneState.file : null;
    const frames = beatStarts.map(({ beat, at }, index) => {
      const first = index === 0, frameEndMs = index < beatStarts.length - 1 ? beatStarts[index + 1].at : endMs;
      const visual = first && reusableFile ? scene.visual_concept : beat.visual;
      const detail = first && reusableFile ? scene.prompt : beat.prompt;
      let motionType = MOTION_TYPES.includes(beat.motion_type) ? beat.motion_type : 'static';
      if (motionType === previousMotion && motionType !== 'static') motionType = MOTION_TYPES.find((type) => type !== motionType && type !== 'static') || 'static';
      previousMotion = motionType;
      const amount = motionType === 'static' ? 0 : Math.max(0, Math.min(0.04, Number(beat.motion_amount) || 0.02));
      const continuity = first && reusableFile ? 'Reuse the approved existing scene drawing as the opening state.' : beat.continuity_notes;
      const promptBeat = { visual, prompt: detail, previous_frame_context: first && reusableFile ? '' : beat.previous_frame_context, continuity_notes: continuity };
      const frame = { frame_id: nextFrameId++, start: +(at / 1000).toFixed(3), end: +(frameEndMs / 1000).toFixed(3),
        duration: +((frameEndMs - at) / 1000).toFixed(3), visual, prompt: framePrompt(id, scene, promptBeat),
        previous_frame_context: promptBeat.previous_frame_context, continuity_notes: continuity,
        motion: { type: motionType, amount }, intentional_fast: !!beat.intentional_fast, status: first && reusableFile ? 'completed' : 'pending',
        file: first && reusableFile ? reusableFile : null, reused: !!(first && reusableFile) };
      return frame;
    });
    outputSequences.push({ sequence_id: scene.scene_id, source_scene_id: scene.scene_id, start: scene.start, end: scene.end,
      duration: +(durationMs / 1000).toFixed(3), narration: scene.narration, visual_concept: scene.visual_concept, frames });
  }
  const plan = { version: 1, model: M.re, created_at: new Date().toISOString(), duration: transcript.duration, sequences: outputSequences };
  fs.writeFileSync(file, JSON.stringify(plan, null, 2));
  log(id, `[VISUAL SEQUENCES] Completed (${outputSequences.length} sequences, ${outputSequences.reduce((sum, sequence) => sum + sequence.frames.length, 0)} frames; no images generated)`);
  return plan;
}
function motionExpressions(motion, frames) {
  const type = MOTION_TYPES.includes(motion?.type) ? motion.type : 'static';
  const intensity = Math.max(0, Math.min(motion?.amount != null ? 0.04 : 0.15, Number(motion?.intensity ?? motion?.amount) || 0));
  const progress = `min(on/${Math.max(1, frames)}\\,1)`;
  let zoom = '1', x = '(iw-iw/zoom)/2', y = '(ih-ih/zoom)/2';
  if (type === 'zoom_in') zoom = `1+${intensity}*${progress}`;
  else if (type === 'zoom_out') zoom = `1+${intensity}*(1-${progress})`;
  else if (type === 'zoom_in_fast') zoom = `1+${intensity}*min(on/${Math.max(1, Math.min(frames, 21))}\\,1)`;
  else if (type.startsWith('pan_')) {
    zoom = `1+${intensity}`;
    if (type === 'pan_left') x = `(iw-iw/zoom)*(1-${progress})`;
    if (type === 'pan_right') x = `(iw-iw/zoom)*${progress}`;
    if (type === 'pan_up') y = `(ih-ih/zoom)*(1-${progress})`;
    if (type === 'pan_down') y = `(ih-ih/zoom)*${progress}`;
    if (type === 'pan_diagonal') { x = `(iw-iw/zoom)*${progress}`; y = `(ih-ih/zoom)*${progress}`; }
  }
  return `scale=1920:1080,zoompan=z='${zoom}':x='${x}':y='${y}':d=1:s=1920x1080:fps=30,setsar=1,format=yuv420p`;
}

async function genImage(id, sid) {
  const plan = rd(path.join(pdir(id), 'scene_plan.json'));
  const sc = plan.scenes.find((x) => x.scene_id === sid);
  if (!sc) throw new Error(`Escena ${sid} no existe`);
  const rel = `scenes/scene_${pad(sid)}.png`;
  for (let attempt = 1; attempt <= 3; attempt++) {
    patch(id, (s) => { const e = s.scenes.find((x) => x.scene_id === sid); e.status = 'generating'; e.retry_count = attempt - 1; });
    log(id, `[IMAGE ${pad(sid)}] Generating (attempt ${attempt})`);
    try {
      const r = await openai.images.generate({ model: M.im, prompt: fullPrompt(id, sc), size: IMG_SIZE, quality: IMG_QUALITY, n: 1 });
      const b64 = r.data[0].b64_json;
      if (!b64) throw new Error('Respuesta sin b64_json');
      fs.writeFileSync(path.join(pdir(id), rel), Buffer.from(b64, 'base64'));
      patch(id, (s) => { const e = s.scenes.find((x) => x.scene_id === sid); Object.assign(e, { status: 'completed', file: rel, error: null, model: M.im, size: IMG_SIZE, prompt: sc.prompt, timestamp: new Date().toISOString() }); });
      log(id, `[IMAGE ${pad(sid)}] Completed`);
      return;
    } catch (e) {
      log(id, `[IMAGE ${pad(sid)}] Error: ${e.message}`);
      patch(id, (s) => { const x = s.scenes.find((y) => y.scene_id === sid); x.error = redact(e.message); x.status = attempt === 3 ? 'failed' : 'pending'; });
    }
  }
}

// Etapa 1: TRANSCRIBE + ANALYZE. Nunca genera imágenes. Reanudable: si transcript.json / scene_plan.json ya existen, no se repiten.
// step: 1 upload · 2 transcribiendo · 3 analizando · 4 REVISIÓN del plan (espera al usuario) · 5 imágenes · 6 render · 7 completado
async function analyze(id) {
  if (running.has(id)) return;
  if (fs.existsSync(path.join(pdir(id), 'scene_plan.json'))) { log(id, '[ANALYZE] Ya completado: se reutiliza el plan existente (0 llamadas a OpenAI)'); return; }
  running.add(id);
  try {
    if (!openai) throw new Error('Falta OPENAI_API_KEY en .env');
    patch(id, (s) => { s.step = 2; s.error = null; });
    const t = await transcribe(id);
    patch(id, (s) => { s.step = 3; });
    await planScenes(id, t);
    patch(id, (s) => { s.step = 4; });
    log(id, '[REVIEW] Plan listo. Revísalo y pulsa GENERATE IMAGES cuando quieras gastar en imágenes.');
  } catch (e) {
    log(id, `[ERROR] ${e.message}`);
    patch(id, (s) => { s.error = redact(e.message); });
  } finally { running.delete(id); }
}

// Resumen del plan para la pantalla de revisión (siempre calculado en vivo desde scene_plan.json + state.json)
function summarize(id) {
  const pf = path.join(pdir(id), 'scene_plan.json'); if (!fs.existsSync(pf)) return null;
  const s = load(id), plan = ensureMotionPlan(id);
  const done = (sc) => { const e = (s.scenes || []).find((x) => x.scene_id === sc.scene_id); return e?.status === 'completed' && e.file && fs.existsSync(path.join(pdir(id), e.file)); };
  const scenes = plan.scenes.length, completed = plan.scenes.filter(done).length;
  return { duration: s.duration, scenes, imagesToGenerate: scenes - completed, imagesCompleted: completed,
    maxScenes: MAX_SCENES, overLimit: scenes > MAX_SCENES, model: M.im, size: IMG_SIZE, quality: IMG_QUALITY };
}

// Etapa 2: GENERATE IMAGES. Sólo se ejecuta si el usuario la pide explícitamente (POST /generate-images).
async function generateImages(id) {
  if (running.has(id)) return;
  running.add(id);
  try {
    if (!openai) throw new Error('Falta OPENAI_API_KEY en .env');
    patch(id, (s) => { s.step = 5; s.error = null; });
    // reanudable: sólo escenas no completadas (los "generating" huérfanos vuelven a pending)
    for (const e of load(id).scenes) if (e.status !== 'completed' || !e.file || !fs.existsSync(path.join(pdir(id), e.file))) await genImage(id, e.scene_id);
    const failed = load(id).scenes.filter((x) => x.status === 'failed').length;
    patch(id, (s) => { s.step = 5; s.error = failed ? `${failed} escena(s) fallaron. Usa REGENERATE.` : null; });
    log(id, failed ? `[IMAGES] Terminado con ${failed} fallo(s)` : '[IMAGES] Todas las imágenes completadas. Ya puedes hacer RENDER VIDEO.');
  } catch (e) {
    log(id, `[ERROR] ${e.message}`);
    patch(id, (s) => { s.error = redact(e.message); });
  } finally { running.delete(id); }
}

async function render(id) {
  if (!ffmpegOk()) throw new Error(FFMPEG_HELP);
  const dir = pdir(id), s = load(id), plan = ensureMotionPlan(id);
  const missing = plan.scenes.filter((sc) => { const e = s.scenes.find((x) => x.scene_id === sc.scene_id); return !e || e.status !== 'completed' || !e.file || !fs.existsSync(path.join(dir, e.file)); });
  if (missing.length) throw new Error(`Faltan imágenes en escenas: ${missing.map((x) => x.scene_id).join(', ')}`);
  log(id, '[RENDER] Started');
  fs.mkdirSync(path.join(dir, 'output'), { recursive: true });
  const temp = path.join(dir, 'output', '_render');
  fs.rmSync(temp, { recursive: true, force: true });
  fs.mkdirSync(temp, { recursive: true });
  try {
    const sorted = [...plan.scenes].sort((a, b) => a.start - b.start), lines = [];
    for (let i = 0; i < sorted.length; i++) {
      const sc = sorted[i], start = i === 0 ? 0 : sc.start;
      const end = i < sorted.length - 1 ? sorted[i + 1].start : s.duration;
      const frames = Math.max(1, Math.round(end * 30) - Math.round(start * 30));
      const file = `scene_${pad(sc.scene_id)}.mp4`;
      const image = s.scenes.find((x) => x.scene_id === sc.scene_id).file;
      await run(FFMPEG, ['-y', '-loop', '1', '-framerate', '30', '-i', image, '-vf', motionExpressions(sc.motion, frames),
        '-frames:v', String(frames), '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', '-r', '30', path.join(temp, file)], { cwd: dir });
      lines.push(`file '${path.join(temp, file).replace(/\\/g, '/')}'`);
    }
    const concatFile = path.join(temp, 'concat.txt');
    fs.writeFileSync(concatFile, lines.join('\n'));
    const args = ['-y', '-f', 'concat', '-safe', '0', '-i', concatFile, '-i', `audio/${s.audioFile}`, '-map', '0:v:0', '-map', '1:a:0'];
    if (fs.existsSync(path.join(dir, 'transcript', 'subtitles.srt')))
      args.push('-vf', "subtitles=transcript/subtitles.srt:force_style='FontSize=18,Outline=2,MarginV=40'");
    args.push('-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-r', '30',
      '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', 'output/final_video.mp4');
    await run(FFMPEG, args, { cwd: dir });
  } finally { fs.rmSync(temp, { recursive: true, force: true }); }
  patch(id, (x) => { x.rendered = true; });
  log(id, '[RENDER] Completed');
}

async function newProject(srcPath, name, ext = path.extname(srcPath).toLowerCase()) {
  assertTools(); // FFmpeg y FFprobe deben existir ANTES de crear nada
  const id = randomUUID().slice(0, 8), dir = pdir(id);
  try {
    for (const d of ['audio', 'transcript', 'scenes', 'output']) fs.mkdirSync(path.join(dir, d), { recursive: true });
    const audioFile = 'narration' + (ext === '.mp4' ? '.m4a' : ext);
    const dest = path.join(dir, 'audio', audioFile);
    if (ext === '.mp4') await run(FFMPEG, ['-y', '-i', srcPath, '-vn', '-c:a', 'aac', dest]); else fs.copyFileSync(srcPath, dest);
    const duration = audioDuration(dest); // lanza error explícito si FFprobe falla: no se crea un proyecto con duración 0
    save(id, { id, name, audioFile, duration, step: 1, scenes: [], logs: [] });
  } catch (e) { fs.rmSync(dir, { recursive: true, force: true }); throw e; }
  try { fs.unlinkSync(srcPath); } catch {}
  return id;
}

async function generateAIVoiceSampleVideo(id) {
  const state = load(id);
  if (state.mode !== 'ai_voice' || state.aiVoice?.status !== 'review') throw new Error('Primero genera y revisa el storyboard AI Voice.');
  if (state.aiVoice.sampleVideo?.status === 'completed') return state.aiVoice.sampleVideo;
  patch(id, (current) => { current.aiVoice.sampleVideo = { status: 'generating', started_at: new Date().toISOString() }; current.aiVoice.stage = 'generating first 10 frames'; });
  try {
    const generated = await generateStoryboardSampleFrames(id);
    const rendered = await renderStoryboardSample(id);
    const sampleVideo = { status: 'completed', requested_frames: 10, frame_ids: generated.frames.map((frame) => frame.frame_id),
      files: generated.frames.map((frame) => frame.file), output: rendered.output, subtitles: rendered.subtitles, duration: rendered.duration,
      model: generated.model, reference_mode: generated.reference_mode, completed_at: new Date().toISOString() };
    patch(id, (current) => { current.aiVoice.sampleVideo = sampleVideo; current.aiVoice.generatedFrames = Math.max(current.aiVoice.generatedFrames || 0, 10);
      current.aiVoice.stage = 'test video ready for review'; });
    log(id, '[AI VOICE TEST] Completed (10 consecutive frames, source audio, subtitles, direct cuts)');
    return sampleVideo;
  } catch (error) {
    patch(id, (current) => { current.aiVoice.sampleVideo = { ...(current.aiVoice.sampleVideo || {}), status: 'failed', error: redact(error.message) }; current.aiVoice.stage = 'test video failed'; });
    log(id, `[AI VOICE TEST ERROR] ${error.message}`);
    throw error;
  }
}

function summarizeAIVoiceFrames(id) {
  const storyboard = rd(storyboardFile(id)), dir = pdir(id);
  const frames = storyboard.sequences.flatMap((sequence) => sequence.frames);
  const complete = frames.filter((frame) => fs.existsSync(path.join(dir, 'frames', 'ai_voice', `frame_${pad(frame.frame_id)}.png`))).length;
  return { duration: storyboard.duration, sequences: storyboard.sequences.length, frames: frames.length,
    generatedFrames: complete, remainingFrames: frames.length - complete, estimatedImageGenerations: frames.length - complete };
}

async function generateAIVoiceFrames(id, requestedCount) {
  const state = load(id), dir = pdir(id), storyboard = rd(storyboardFile(id));
  if (state.mode !== 'ai_voice' || state.aiVoice?.status !== 'review') throw new Error('El storyboard AI Voice debe estar listo para revisión antes de generar frames.');
  const allFrames = storyboard.sequences.flatMap((sequence) => sequence.frames);
  const outputDir = path.join(dir, 'frames', 'ai_voice');
  fs.mkdirSync(outputDir, { recursive: true });
  const completed = (frame) => fs.existsSync(path.join(outputDir, `frame_${pad(frame.frame_id)}.png`));
  const firstPending = allFrames.findIndex((frame) => !completed(frame));
  if (firstPending < 0) return { requested: 0, generated: 0, remaining: 0, files: [] };
  const limit = requestedCount === 'all' ? allFrames.length : Number(requestedCount);
  if (!Number.isInteger(limit) || limit < 1) throw new Error('El lote debe ser 10, 25 o all.');
  const selected = allFrames.slice(firstPending, firstPending + limit);
  for (let index = 1; index < selected.length; index++) {
    if (selected[index].previous_frame_id !== selected[index - 1].frame_id)
      throw new Error('El lote seleccionado no es consecutivo; se detuvo para conservar la referencia visual.');
  }
  const generated = [];
  for (const frame of selected) {
    const imageName = `frame_${pad(frame.frame_id)}.png`, outputFile = path.join(outputDir, imageName);
    if (completed(frame)) { generated.push({ frame_id: frame.frame_id, file: `frames/ai_voice/${imageName}`, reused: true }); continue; }
    const previousFile = frame.previous_frame_id == null ? null : path.join(outputDir, `frame_${pad(frame.previous_frame_id)}.png`);
    if (previousFile && !fs.existsSync(previousFile)) throw new Error(`Falta la imagen de referencia frame ${frame.previous_frame_id}; se detuvo antes de saltar un frame.`);
    const continuity = frame.sequence_transition
      ? 'Use the previous image as a reference for Maya identity, proportions, and drawing style only. This is a new narrated sequence: establish the new environment and props while retaining the same Maya.'
      : 'Use the previous image as the reference. Preserve Maya, scene layout, props, line style, colors, and scale. Change only the small physical action described for this next instant.';
    const prompt = [frame.prompt, `Previous frame state: ${frame.previous_frame_state}`, `Current frame state: ${frame.current_frame_state}`,
      `Continuity notes: ${frame.continuity_notes}`, continuity, 'Keep a static camera; no transitions or camera motion.'].join('\n');
    log(id, `[AI VOICE FRAME ${pad(frame.frame_id)}] ${previousFile ? 'Editing from previous frame' : 'Generating opening frame'}`);
    const response = previousFile
      ? await openai.images.edit({ model: M.im, image: await toFile(fs.createReadStream(previousFile), path.basename(previousFile), { type: 'image/png' }), prompt, size: IMG_SIZE, quality: IMG_QUALITY, n: 1 })
      : await openai.images.generate({ model: M.im, prompt, size: IMG_SIZE, quality: IMG_QUALITY, n: 1 });
    const image = response.data[0]?.b64_json;
    if (!image) throw new Error(`La respuesta del frame ${frame.frame_id} no contiene b64_json.`);
    fs.writeFileSync(outputFile, Buffer.from(image, 'base64'));
    generated.push({ frame_id: frame.frame_id, file: `frames/ai_voice/${imageName}`, reused: false });
    log(id, `[AI VOICE FRAME ${pad(frame.frame_id)}] Completed`);
  }
  const summary = summarizeAIVoiceFrames(id);
  patch(id, (current) => { current.aiVoice.generatedFrames = summary.generatedFrames;
    current.aiVoice.frameBatches ||= [];
    current.aiVoice.frameBatches.push({ requested: requestedCount, frame_ids: generated.map((frame) => frame.frame_id), generated_at: new Date().toISOString() }); });
  return { requested: limit, generated: generated.filter((frame) => !frame.reused).length, remaining: summary.remainingFrames, files: generated };
}

// ---------- API ----------
const app = express();
app.use(express.json());
const upload = multer({ dest: path.join(ROOT, '_tmp'), limits: { fileSize: 500 * 1024 * 1024 } });
const OKEXT = ['.wav', '.mp3', '.m4a', '.mp4'];
const httpErr = (msg, http = 400, code, extra) => Object.assign(new Error(msg), { http, code, extra });
// fn(req,res) puede lanzar de forma síncrona (ej. los guardias con httpErr) o devolver una promesa que rechace;
// el try/catch cubre el caso síncrono y el .catch() el asíncrono. Sin el try/catch, un throw síncrono no lo
// captura Promise.resolve(fn(...)) porque el throw ocurre antes de llamarlo, y Express respondía HTML de error.
const sendErr = (res, e) => res.status(e.http || 400).json({ error: redact(e.message), code: e.code, ...(e.extra || {}) });
const wrap = (fn) => (req, res) => { try { Promise.resolve(fn(req, res)).catch((e) => sendErr(res, e)); } catch (e) { sendErr(res, e); } };

// ---------- health check (usa la API key real; jamás la devuelve) ----------
const explain = (e) => {
  if (e?.status === 401) return 'La API key fue rechazada (401). Revisa OPENAI_API_KEY en .env.';
  if (e?.status === 403) return 'Sin permiso (403): la key/proyecto no tiene acceso a este recurso.';
  if (e?.status === 404) return 'No existe o tu cuenta/proyecto no tiene acceso (404).';
  if (e?.status === 429) return 'Límite de uso o saldo agotado (429).';
  if (!e?.status) return `Sin conexión con la API de OpenAI: ${redact(e?.message || e)}`;
  return `Error ${e.status}: ${redact(e.message).slice(0, 200)}`;
};
async function openaiHealth() {
  if (!openai) return { ok: false, configured: false, connection: { ok: false, error: 'Falta OPENAI_API_KEY en .env' }, models: {} };
  const opt = { timeout: 10000, maxRetries: 0 };
  const connection = await openai.models.list(opt).then(() => ({ ok: true }), (e) => ({ ok: false, status: e.status ?? null, error: explain(e) }));
  const roles = { reasoning: M.re, transcription: M.tr, image: M.im, tts: TTS_MODEL };
  if (M.ts !== M.tr) roles.timestamps_fallback = M.ts;
  const models = {};
  // Con 401 o sin red, comprobar cada modelo no aporta nada. Un 403 en "list" sólo indica una key restringida: se sigue probando cada modelo.
  const skip = !connection.ok && (connection.status === 401 || connection.status === null);
  await Promise.all(Object.entries(roles).map(async ([role, id]) => {
    if (skip) { models[role] = { id, available: null, error: 'No comprobado (falla la conexión o la key)' }; return; }
    models[role] = await openai.models.retrieve(id, opt).then(() => ({ id, available: true }),
      (e) => ({ id, available: e.status === 403 ? null : false, error: explain(e) }));
  }));
  const critical = ['reasoning', 'transcription', 'image', 'tts'];
  return { ok: connection.ok && critical.every((r) => models[r].available === true), configured: true, connection, models,
    note: 'Comprobación gratuita (models.retrieve): confirma que el modelo existe y es visible para tu key. No prueba límites de uso ni verificación de organización para imágenes.' };
}
app.get('/api/health', wrap(async (_q, res) => {
  const fm = ffmpegOk(), fp = ffprobeOk();
  res.json({ ffmpeg: { ok: fm, command: FFMPEG, ...(fm ? {} : { help: FFMPEG_HELP }) }, ffprobe: { ok: fp, command: FFPROBE, ...(fp ? {} : { help: FFPROBE_HELP }) },
    openai: await openaiHealth(), maxScenes: MAX_SCENES, imageSettings: { size: IMG_SIZE, quality: IMG_QUALITY },
    voice: { model: TTS_MODEL, defaultVoice: TTS_DEFAULT_VOICE, voices: TTS_VOICES, disclosure: 'AI-generated voice' } });
}));
app.post('/api/voice-test', wrap(async (_req, res) => res.json(await generateVoiceTest())));
app.get('/api/voice-test/:voice', wrap((req, res) => {
  const voice = req.params.voice.toLowerCase();
  if (!['marin', 'cedar'].includes(voice)) throw httpErr('La prueba compara las voces marin y cedar.', 404);
  const file = path.join(voiceTestDir(), `voice_test_${voice}.mp3`);
  if (!fs.existsSync(file)) throw httpErr('Primero ejecuta TEST VOICE.', 404);
  res.type('audio/mpeg').sendFile(file);
}));
app.post('/api/ai-voice/projects', wrap((req, res) => {
  if (!openai) throw httpErr('Falta OPENAI_API_KEY en .env');
  const created = createAIVoiceProject(req.body || {});
  const current = load(created.id);
  if (current.aiVoice.status !== 'review' && !running.has(created.id)) launchAIVoicePipeline(created.id);
  res.json({ id: created.id, cached: created.cached, status: current.aiVoice.status, model: TTS_MODEL, voice: current.aiVoice.voice });
}));
app.get('/api/projects', (_q, res) => res.json(fs.readdirSync(ROOT).filter((d) => fs.existsSync(sfile(d))).map((d) => { const s = load(d); return { id: d, name: s.name }; })));

app.post('/api/projects', upload.single('audio'), wrap(async (req, res) => {
  const f = req.file; if (!f) throw new Error('No se recibió archivo');
  try {
    const ext = path.extname(f.originalname).toLowerCase();
    if (!OKEXT.includes(ext)) throw new Error(`Formato no permitido (${OKEXT.join(', ')})`);
    res.json({ id: await newProject(f.path, f.originalname, ext) }); // newProject valida FFmpeg + FFprobe y la duración
  } catch (e) { try { fs.unlinkSync(f.path); } catch {} throw e; }
}));

app.get('/api/projects/:id', wrap((req, res) => {
  const s = load(req.params.id);
  const pf = path.join(pdir(s.id), 'scene_plan.json'), tf = path.join(pdir(s.id), 'transcript', 'transcript.json');
  const vpf = sequencePlanFile(s.id), sbf = storyboardFile(s.id);
  const sourceFile = path.join(pdir(s.id), 'script', 'sources.json');
  res.json({ ...s, running: running.has(s.id), plan: fs.existsSync(pf) ? ensureMotionPlan(s.id) : null, summary: summarize(s.id),
    visualSequencePlan: fs.existsSync(vpf) ? visualSequenceView(s.id) : null, visualSequenceSummary: summarizeVisualSequences(s.id),
    visualStoryboard: fs.existsSync(sbf) ? rd(sbf) : null,
    ttsModel: TTS_MODEL,
    sourceReferences: fs.existsSync(sourceFile) ? rd(sourceFile) : [],
    subs: fs.existsSync(tf) ? rd(tf).segments : [], hasVideo: fs.existsSync(path.join(pdir(s.id), 'output', 'final_video.mp4')) });
}));
app.post('/api/projects/:id/ai-voice/test-video', wrap((req, res) => {
  const id = req.params.id, state = load(id);
  if (state.mode !== 'ai_voice') throw httpErr('Esta acción solo aplica a proyectos AI Voice.', 409);
  if (state.aiVoice?.status !== 'review') throw httpErr('Espera a que el storyboard esté listo para revisión.', 409);
  if (running.has(id)) throw httpErr('Ya hay un proceso en curso para este proyecto', 409);
  if (state.aiVoice.sampleVideo?.status === 'completed') return res.json({ ok: true, started: false, sampleVideo: state.aiVoice.sampleVideo });
  running.add(id);
  generateAIVoiceSampleVideo(id).catch(() => {}).finally(() => running.delete(id));
  res.json({ ok: true, started: true, frames: 10, imageModel: M.im, generatedImageCount: 10 });
}));
app.post('/api/projects/:id/ai-voice/frames', wrap((req, res) => {
  const id = req.params.id, state = load(id), count = req.body?.count;
  if (state.mode !== 'ai_voice' || state.aiVoice?.status !== 'review') throw httpErr('Primero genera el storyboard AI Voice y revísalo.', 409);
  if (![10, 25, 'all'].includes(count)) throw httpErr('El lote debe ser 10, 25 o all.', 400);
  if (count === 'all' && req.body?.confirmAll !== true) throw httpErr('Confirma explícitamente la generación de todos los frames pendientes.', 409, 'confirmation_required');
  if (running.has(id)) throw httpErr('Ya hay un proceso en curso para este proyecto', 409);
  const summary = summarizeAIVoiceFrames(id), requested = count === 'all' ? summary.remainingFrames : Math.min(count, summary.remainingFrames);
  if (!requested) return res.json({ ok: true, started: false, summary });
  running.add(id);
  generateAIVoiceFrames(id, count).catch((error) => patch(id, (current) => { current.aiVoice.lastFrameBatchError = redact(error.message); }))
    .finally(() => running.delete(id));
  res.json({ ok: true, started: true, requested, generatedFrames: summary.generatedFrames, remainingFrames: summary.remainingFrames, model: M.im });
}));
app.post('/api/projects/:id/storyboard', wrap((req, res) => {
  const id = req.params.id;
  load(id);
  if (!fs.existsSync(path.join(pdir(id), 'scene_plan.json'))) throw httpErr('Primero ejecuta ANALYZE para crear el plan de escenas.', 409, 'missing_scene_plan');
  const storyboard = generateVisualStoryboard(id);
  res.json({ ok: true, storyboard, file: `projects/${id}/visual_storyboard.json` });
}));

// UPLOAD → TRANSCRIBE → ANALYZE → (revisión del usuario). Esta ruta NUNCA genera imágenes.
app.post('/api/projects/:id/analyze', wrap((req, res) => {
  load(req.params.id);
  if (!openai) throw httpErr('Falta OPENAI_API_KEY en .env');
  if (running.has(req.params.id)) throw httpErr('Ya hay un proceso en curso para este proyecto', 409);
  analyze(req.params.id); res.json({ ok: true });
}));
app.post('/api/projects/:id/visual-sequences', wrap(async (req, res) => {
  const id = req.params.id;
  load(id);
  if (!openai) throw httpErr('Falta OPENAI_API_KEY en .env');
  if (running.has(id)) throw httpErr('Ya hay un proceso en curso para este proyecto', 409);
  if (!fs.existsSync(path.join(pdir(id), 'scene_plan.json')) || !fs.existsSync(path.join(pdir(id), 'transcript', 'transcript.json')))
    throw httpErr('Primero ejecuta ANALYZE para crear escenas y transcripción.', 409, 'missing_source');
  running.add(id);
  try {
    await planVisualSequences(id);
    res.json({ ok: true, summary: summarizeVisualSequences(id) });
  } finally { running.delete(id); }
}));
app.post('/api/projects/:id/visual-sequences/test-frames', wrap(async (req, res) => {
  const id = req.params.id;
  load(id);
  if (!fs.existsSync(sequencePlanFile(id))) throw httpErr('Primero crea el visual sequence plan.', 409, 'missing_visual_plan');
  if (running.has(id)) throw httpErr('Ya hay un proceso en curso para este proyecto', 409);
  if (load(id).visualFrameTest) throw httpErr('La prueba de cinco frames ya fue iniciada y no se repetirá.', 409, 'test_already_started');
  res.json(await generateTestFrames(id));
}));
app.post('/api/projects/:id/visual-sequences/test-render', wrap(async (req, res) => {
  const id = req.params.id;
  load(id);
  if (running.has(id)) throw httpErr('Ya hay un proceso en curso para este proyecto', 409);
  running.add(id);
  try { res.json(await renderVisualTest(id)); }
  finally { running.delete(id); }
}));
// GENERATE IMAGES: sólo tras el plan y sólo por petición explícita. Si el plan supera MAX_SCENES exige confirmOverLimit=true.
app.post('/api/projects/:id/generate-images', wrap((req, res) => {
  const id = req.params.id; load(id);
  const sum = summarize(id);
  if (!sum) throw httpErr('Primero ejecuta ANALYZE: todavía no hay scene plan que revisar.', 409, 'no_plan');
  if (!openai) throw httpErr('Falta OPENAI_API_KEY en .env');
  if (running.has(id)) throw httpErr('Ya hay un proceso en curso para este proyecto', 409);
  if (sum.overLimit && !req.body?.confirmOverLimit)
    throw httpErr(`El plan tiene ${sum.scenes} escenas y el límite MAX_SCENES es ${sum.maxScenes}. Confirma para generar de todos modos.`, 409, 'over_limit', { scenes: sum.scenes, maxScenes: sum.maxScenes });
  if (!sum.imagesToGenerate) return res.json({ ok: true, started: false, message: 'Todas las imágenes ya están generadas' });
  patch(id, (s) => { s.imagesApproved = true; });
  generateImages(id); res.json({ ok: true, started: true, images: sum.imagesToGenerate });
}));
app.post('/api/projects/:id/render', wrap(async (req, res) => {
  const id = req.params.id; load(id);
  if (running.has(id)) throw httpErr('Ya hay un proceso en curso para este proyecto', 409);
  running.add(id);
  try { patch(id, (s) => { s.step = 6; s.error = null; }); await render(id); patch(id, (s) => { s.step = 7; }); res.json({ ok: true }); }
  catch (e) { patch(id, (s) => { s.step = 5; }); throw e; }
  finally { running.delete(id); }
}));

function editPlan(id, fn) {
  const f = path.join(pdir(id), 'scene_plan.json'); const plan = rd(f); fn(plan);
  plan.scenes.sort((a, b) => a.start - b.start);
  plan.scenes.forEach((x) => { x.duration = +(x.end - x.start).toFixed(2); if (x.motion) x.motion.duration = x.duration; });
  fs.writeFileSync(f, JSON.stringify(plan, null, 2));
}
app.put('/api/projects/:id/scenes/:sid', wrap((req, res) => {
  const sid = +req.params.sid, { prompt, start, end, motion } = req.body;
  editPlan(req.params.id, (p) => { const sc = p.scenes.find((x) => x.scene_id === sid);
    if (typeof prompt === 'string') sc.prompt = prompt;
    if (isFinite(start)) sc.start = +start; if (isFinite(end)) sc.end = +end;
    if (motion && MOTION_TYPES.includes(motion.type)) {
      sc.motion = { type: motion.type, intensity: Math.max(0, Math.min(0.15, Number(motion.intensity) || 0)),
        direction: motion.type.startsWith('pan_') ? motion.type.slice(4) : 'center', duration: +(sc.end - sc.start).toFixed(2) };
    } });
  res.json({ ok: true });
}));
app.post('/api/projects/:id/scenes/:sid/regenerate', wrap((req, res) => {
  const id = req.params.id, sid = +req.params.sid;
  if (!load(id).imagesApproved) throw httpErr('Aún no has pulsado GENERATE IMAGES: no se genera ninguna imagen hasta entonces.', 409, 'not_approved');
  if (!openai) throw httpErr('Falta OPENAI_API_KEY en .env');
  if (running.has(id)) throw httpErr('Ya hay un proceso en curso para este proyecto', 409);
  running.add(id);
  genImage(id, sid).catch((e) => log(id, `[ERROR] ${e.message}`)).finally(() => running.delete(id));
  res.json({ ok: true });
}));
app.delete('/api/projects/:id/scenes/:sid', wrap((req, res) => {
  const sid = +req.params.sid;
  editPlan(req.params.id, (p) => { if (p.scenes.length < 2) throw new Error('Debe quedar al menos una escena');
    const i = p.scenes.findIndex((x) => x.scene_id === sid);
    const [d] = p.scenes.splice(i, 1);
    if (i > 0) p.scenes[i - 1].end = d.end; else p.scenes[0].start = d.start; });
  patch(req.params.id, (s) => { s.scenes = s.scenes.filter((x) => x.scene_id !== sid); });
  res.json({ ok: true });
}));
app.post('/api/projects/:id/scenes/:sid/duplicate', wrap((req, res) => {
  const sid = +req.params.sid; let nid;
  editPlan(req.params.id, (p) => {
    const sc = p.scenes.find((x) => x.scene_id === sid); nid = Math.max(...p.scenes.map((x) => x.scene_id)) + 1;
    const mid = +((sc.start + sc.end) / 2).toFixed(2);
    p.scenes.push({ ...sc, scene_id: nid, start: mid }); sc.end = mid; });
  const src = load(req.params.id).scenes.find((x) => x.scene_id === sid);
  const rel = `scenes/scene_${pad(nid)}.png`;
  if (src?.file) fs.copyFileSync(path.join(pdir(req.params.id), src.file), path.join(pdir(req.params.id), rel));
  patch(req.params.id, (s) => { s.scenes.push({ ...src, scene_id: nid, file: src?.file ? rel : null }); });
  res.json({ ok: true });
}));

app.use('/files/:id', (req, res, next) => express.static(pdir(req.params.id))(req, res, next));
app.use(express.static('public'));
const port = process.env.PORT || 3000;
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve('server.js')) app.listen(port, '127.0.0.1', () => console.log(`http://localhost:${port}  ffmpeg=${ffmpegOk()} key=${!!openai}`));

export { app, STYLE, newProject, transcribe, planScenes, planVisualSequences, planAIVoiceStoryboardActions, generateVisualStoryboard,
  generateStoryboardSampleFrames, renderStoryboardSample, generateVoiceTest, createAIVoiceProject, runAIVoicePipeline,
  generateAIVoiceSampleVideo, generateAIVoiceFrames, summarizeAIVoiceFrames, summarize, summarizeVisualSequences, chooseTestFrames,
  generateTestFrames, renderVisualTest, analyze, generateImages, genImage, render, load, pdir, patch, M, TTS_MODEL,
  TTS_DEFAULT_VOICE, TTS_VOICES, TTS_TEST_TEXT, TTS_BASE_INSTRUCTIONS, AI_VOICE_DEMO_TITLE, AI_VOICE_DEMO_SCRIPT,
  AI_VOICE_DEMO_SOURCES, IMG_SIZE, MAX_SCENES, ffmpegOk, ffprobeOk, audioDuration, FFMPEG, FFPROBE, run, redact };
