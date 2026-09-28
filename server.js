import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';
import { spawn, spawnSync } from 'child_process';
import { randomUUID } from 'crypto';

const ROOT = path.resolve('projects');
fs.mkdirSync(ROOT, { recursive: true });
const FFMPEG = process.env.FFMPEG_PATH || 'ffmpeg';
const FFPROBE = process.env.FFMPEG_PATH ? path.join(path.dirname(process.env.FFMPEG_PATH), 'ffprobe') : 'ffprobe';
const M = {
  tr: process.env.OPENAI_TRANSCRIPTION_MODEL || 'gpt-transcribe',
  // Según la doc oficial, timestamp_granularities SOLO existe para whisper-1.
  ts: process.env.OPENAI_TIMESTAMP_MODEL || 'whisper-1',
  re: process.env.OPENAI_REASONING_MODEL || 'gpt-5.6-luna',
  im: process.env.OPENAI_IMAGE_MODEL || 'gpt-image-2',
};
const IMG_SIZE = process.env.OPENAI_IMAGE_SIZE || '2048x1152'; // 16:9 exacto, lados múltiplos de 16 (gpt-image-2)
const IMG_QUALITY = process.env.OPENAI_IMAGE_QUALITY || 'medium';
const openai = process.env.OPENAI_API_KEY ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;

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
  const line = `[${new Date().toISOString().slice(11, 19)}] ${msg}`;
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
const running = new Set();

// ---------- pipeline ----------
async function transcribe(id) {
  const f = path.join(pdir(id), 'transcript', 'transcript.json');
  if (fs.existsSync(f)) return rd(f);
  log(id, '[TRANSCRIBE] Started');
  const s = load(id);
  // La API limita a 25 MB: se envía una copia comprimida mono 16 kHz (~14 MB/hora). El audio original no se toca.
  const small = path.join(pdir(id), 'audio', 'for_transcription.mp3');
  if (!fs.existsSync(small)) await run(FFMPEG, ['-y', '-i', path.join(pdir(id), 'audio', s.audioFile), '-vn', '-ac', '1', '-ar', '16000', '-b:a', '32k', small]);
  if (fs.statSync(small).size > 25 * 1024 * 1024) throw new Error('Audio demasiado largo para una sola petición (>25 MB comprimido). Divídelo en partes.');
  const call = (model) => openai.audio.transcriptions.create({
    file: fs.createReadStream(small), model, response_format: 'verbose_json', timestamp_granularities: ['word', 'segment'] });
  let used = M.tr, r;
  try { r = await call(M.tr); if (!r.segments?.length) throw new Error('sin segmentos'); }
  catch (e) {
    if (M.tr === M.ts) throw e;
    log(id, `[TRANSCRIBE] ${M.tr} no devuelve timestamps (${String(e.message).slice(0, 120)}). Usando ${M.ts}, el modelo oficial con timestamps.`);
    used = M.ts; r = await call(M.ts);
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
  const sys = `You are a human video editor planning static illustrations for an explainer YouTube video narrated by the user. Decide when the image should change: change of idea, subject or context, new metaphor, new main object, emotional shift, contrast, punchline, relevant question, conclusion. Keep one image while the same idea continues; do NOT create unnecessary scenes (typically one scene every 5-15 seconds).
Do not illustrate sentences literally: use visual metaphors, contrast, exaggeration, visual humor, simple symbols, cause/effect. A viewer should roughly understand the narration from the image alone.
Every scene "start" MUST equal the start time of one of the given segments; scenes must be in order and cover the whole audio.
"characters": recurring people/entities (may be empty). Describe each ONCE, precisely and simply: head, body, clothing, color. Scenes list the character_ids they contain.
"prompt": English, describes ONLY the scene content (who, doing what, which objects, expression, layout). Do NOT describe art style; the style bible is added automatically. Refer to characters by name only.`;
  const r = await openai.chat.completions.create({ model: M.re, response_format: { type: 'json_schema', json_schema: SCHEMA },
    messages: [{ role: 'system', content: sys }, { role: 'user', content: `Total duration: ${t.duration}s\n${segs}` }] });
  const out = JSON.parse(r.choices[0].message.content);
  let sc = out.scenes.sort((a, b) => a.start - b.start);
  if (!sc.length) throw new Error('El modelo no devolvió escenas');
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
      patch(id, (s) => { const x = s.scenes.find((y) => y.scene_id === sid); x.error = e.message; x.status = attempt === 3 ? 'failed' : 'pending'; });
    }
  }
}

async function generate(id) {
  if (running.has(id)) return;
  running.add(id);
  try {
    if (!openai) throw new Error('Falta OPENAI_API_KEY en .env');
    patch(id, (s) => { s.step = 1; s.error = null; });
    const t = await transcribe(id);
    patch(id, (s) => { s.step = 3; });
    await planScenes(id, t);
    patch(id, (s) => { s.step = 4; });
    // reanudable: sólo escenas no completadas (los "generating" huérfanos vuelven a pending)
    for (const e of load(id).scenes) if (e.status !== 'completed' || !e.file || !fs.existsSync(path.join(pdir(id), e.file))) await genImage(id, e.scene_id);
    const failed = load(id).scenes.filter((x) => x.status === 'failed').length;
    patch(id, (s) => { s.step = 5; s.error = failed ? `${failed} escena(s) fallaron. Usa REGENERATE.` : null; });
  } catch (e) {
    log(id, `[ERROR] ${e.message}`);
    patch(id, (s) => { s.error = e.message; });
  } finally { running.delete(id); }
}

async function render(id) {
  if (!ffmpegOk()) throw new Error('FFmpeg no encontrado. Instálalo (Windows: winget install ffmpeg | macOS: brew install ffmpeg | Linux: apt install ffmpeg) o define FFMPEG_PATH en .env');
  const dir = pdir(id), s = load(id), plan = rd(path.join(dir, 'scene_plan.json'));
  const missing = plan.scenes.filter((sc) => { const e = s.scenes.find((x) => x.scene_id === sc.scene_id); return !e || e.status !== 'completed'; });
  if (missing.length) throw new Error(`Faltan imágenes en escenas: ${missing.map((x) => x.scene_id).join(', ')}`);
  log(id, '[RENDER] Started');
  fs.mkdirSync(path.join(dir, 'output'), { recursive: true });
  const lines = [];
  const scs = [...plan.scenes].sort((a, b) => a.start - b.start);
  scs.forEach((sc, i) => {
    if (i === 0) sc.start = 0;
    const end = i < scs.length - 1 ? scs[i + 1].start : Math.max(sc.end, s.duration);
    lines.push(`file '${s.scenes.find((x) => x.scene_id === sc.scene_id).file}'`, `duration ${Math.max(0.05, end - sc.start).toFixed(3)}`);
  });
  lines.push(lines[lines.length - 2]); // ffmpeg concat: repetir el último archivo
  fs.writeFileSync(path.join(dir, 'concat.txt'), lines.join('\n'));
  const vf = 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white,setsar=1,fps=30,format=yuv420p'
    + (fs.existsSync(path.join(dir, 'transcript', 'subtitles.srt'))
      ? ",subtitles=transcript/subtitles.srt:force_style='FontSize=18,Outline=2,MarginV=40'" : '');
  await run(FFMPEG, ['-y', '-f', 'concat', '-safe', '0', '-i', 'concat.txt', '-i', `audio/${s.audioFile}`, '-vf', vf,
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', 'output/final_video.mp4'], { cwd: dir });
  patch(id, (x) => { x.rendered = true; });
  log(id, '[RENDER] Completed');
}

async function newProject(srcPath, name, ext = path.extname(srcPath).toLowerCase()) {
  const id = randomUUID().slice(0, 8), dir = pdir(id);
  for (const d of ['audio', 'transcript', 'scenes', 'output']) fs.mkdirSync(path.join(dir, d), { recursive: true });
  const audioFile = 'narration' + (ext === '.mp4' ? '.m4a' : ext);
  const dest = path.join(dir, 'audio', audioFile);
  if (ext === '.mp4') await run(FFMPEG, ['-y', '-i', srcPath, '-vn', '-c:a', 'aac', dest]); else fs.copyFileSync(srcPath, dest);
  try { fs.unlinkSync(srcPath); } catch {}
  const pr = spawnSync(FFPROBE, ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', dest]);
  const duration = parseFloat(pr.stdout.toString()) || 0;
  save(id, { id, name, audioFile, duration, step: 0, scenes: [], logs: [] });
  return id;
}

// ---------- API ----------
const app = express();
app.use(express.json());
const upload = multer({ dest: path.join(ROOT, '_tmp'), limits: { fileSize: 500 * 1024 * 1024 } });
const OKEXT = ['.wav', '.mp3', '.m4a', '.mp4'];
const wrap = (fn) => (req, res) => Promise.resolve(fn(req, res)).catch((e) => res.status(400).json({ error: e.message }));

app.get('/api/health', (_q, res) => res.json({ ffmpeg: ffmpegOk(), openai: !!openai, models: M }));
app.get('/api/projects', (_q, res) => res.json(fs.readdirSync(ROOT).filter((d) => fs.existsSync(sfile(d))).map((d) => { const s = load(d); return { id: d, name: s.name }; })));

app.post('/api/projects', upload.single('audio'), wrap(async (req, res) => {
  const f = req.file; if (!f) throw new Error('No se recibió archivo');
  const ext = path.extname(f.originalname).toLowerCase();
  if (!OKEXT.includes(ext)) { fs.unlinkSync(f.path); throw new Error(`Formato no permitido (${OKEXT.join(', ')})`); }
  if (!ffmpegOk()) { fs.unlinkSync(f.path); throw new Error('FFmpeg no encontrado. Instálalo o define FFMPEG_PATH en .env'); }
  const id = await newProject(f.path, f.originalname, ext);
  res.json({ id });
}));

app.get('/api/projects/:id', wrap((req, res) => {
  const s = load(req.params.id);
  const pf = path.join(pdir(s.id), 'scene_plan.json'), tf = path.join(pdir(s.id), 'transcript', 'transcript.json');
  res.json({ ...s, running: running.has(s.id), plan: fs.existsSync(pf) ? rd(pf) : null,
    subs: fs.existsSync(tf) ? rd(tf).segments : [], hasVideo: fs.existsSync(path.join(pdir(s.id), 'output', 'final_video.mp4')) });
}));

app.post('/api/projects/:id/generate', wrap((req, res) => { generate(req.params.id); res.json({ ok: true }); }));
app.post('/api/projects/:id/render', wrap(async (req, res) => {
  patch(req.params.id, (s) => { s.step = 5; });
  await render(req.params.id); patch(req.params.id, (s) => { s.step = 6; }); res.json({ ok: true });
}));

function editPlan(id, fn) {
  const f = path.join(pdir(id), 'scene_plan.json'); const plan = rd(f); fn(plan);
  plan.scenes.sort((a, b) => a.start - b.start);
  plan.scenes.forEach((x) => { x.duration = +(x.end - x.start).toFixed(2); });
  fs.writeFileSync(f, JSON.stringify(plan, null, 2));
}
app.put('/api/projects/:id/scenes/:sid', wrap((req, res) => {
  const sid = +req.params.sid, { prompt, start, end } = req.body;
  editPlan(req.params.id, (p) => { const sc = p.scenes.find((x) => x.scene_id === sid);
    if (typeof prompt === 'string') sc.prompt = prompt;
    if (isFinite(start)) sc.start = +start; if (isFinite(end)) sc.end = +end; });
  res.json({ ok: true });
}));
app.post('/api/projects/:id/scenes/:sid/regenerate', wrap((req, res) => { genImage(req.params.id, +req.params.sid); res.json({ ok: true }); }));
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

export { STYLE, newProject, transcribe, planScenes, genImage, render, load, pdir, patch, M, IMG_SIZE, ffmpegOk, FFMPEG, run };
