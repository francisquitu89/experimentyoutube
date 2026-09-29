// Tests por etapas. Uso:
//   npm run test:plan  -- audio.wav [segundos=25]   audio → transcripción → scene plan (SIN imágenes)
//   npm run test:image -- <projectId> [sceneId=1]   una sola escena → gpt-image-2
//   npm run test:image -- "a stick figure ..."      prompt suelto con el style bible
//   npm run test:render                             imagen → FFmpeg → MP4 (sin usar la API, coste 0)
//   npm run test:flow -- audio.wav [segundos=25]    UPLOAD → TRANSCRIBE → ANALYZE → REVIEW por la API HTTP. Usa OpenAI para transcribir y planificar
//                                                   (céntimos) pero NUNCA genera imágenes: comprueba que todo queda en 'pending'.
import 'dotenv/config';
import assert from 'node:assert/strict';
import fs from 'fs';
import path from 'path';
import { spawnSync } from 'child_process';
import * as S from './server.js';

const [, , cmd, a, b] = process.argv;
const sh = (args) => S.run(S.FFMPEG, ['-y', ...args]);
const show = (o) => console.log(JSON.stringify(o, null, 2));
if (!S.ffmpegOk()) { console.error(`FFmpeg no encontrado (${S.FFMPEG}): instálalo o define FFMPEG_PATH en .env`); process.exit(1); }
if (!S.ffprobeOk()) { console.error(`FFprobe no encontrado (${S.FFPROBE}). Define FFMPEG_PATH en .env con la ruta a ffmpeg.exe (ffprobe.exe debe estar en la misma carpeta) o FFPROBE_PATH.`); process.exit(1); }

if (cmd === 'plan') {
  if (!a) throw new Error('Uso: npm run test:plan -- audio.wav [segundos]');
  fs.mkdirSync('projects/_tmp', { recursive: true });
  const cut = path.resolve('projects/_tmp/test_cut.mp3');
  await sh(['-i', a, '-t', String(b || 25), '-vn', cut]);
  const id = await S.newProject(cut, path.basename(a), '.mp3');
  const t = await S.transcribe(id);
  const p = await S.planScenes(id, t);
  show({ project: id, transcribe_model: t.model, segments: t.segments.length, words: t.words.length, scenes: p.scenes.map((x) => ({ id: x.scene_id, start: x.start, end: x.end, concept: x.visual_concept })) });
  console.log(`\nSiguiente: npm run test:image -- ${id} 1`);
} else if (cmd === 'image') {
  if (!a) throw new Error('Uso: npm run test:image -- <projectId> [sceneId]  |  "prompt"');
  if (fs.existsSync(path.join('projects', a, 'state.json'))) {
    await S.genImage(a, +b || 1);
    show(S.load(a).scenes.find((x) => x.scene_id === (+b || 1)));
    console.log(`Imagen: projects/${a}/scenes/  (tamaño pedido ${S.IMG_SIZE})`);
  } else {
    const id = await S.newProject((await (async () => { const f = path.resolve('projects/_tmp/silence.mp3'); fs.mkdirSync('projects/_tmp', { recursive: true }); await sh(['-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono', '-t', '2', f]); return f; })()), 'prompt-test', '.mp3');
    fs.writeFileSync(path.join(S.pdir(id), 'scene_plan.json'), JSON.stringify({ scenes: [{ scene_id: 1, start: 0, end: 2, duration: 2, prompt: a, character_ids: [] }] }));
    fs.writeFileSync(path.join(S.pdir(id), 'style_bible.json'), JSON.stringify(S.STYLE));
    S.patch(id, (s) => { s.scenes = [{ scene_id: 1, status: 'pending', retry_count: 0 }]; });
    await S.genImage(id, 1);
    show(S.load(id).scenes[0]);
  }
} else if (cmd === 'sequence-plan') {
  if (!a || !fs.existsSync(path.join('projects', a, 'state.json'))) throw new Error('Uso: npm run test:sequence-plan -- <projectId>');
  if (!process.env.OPENAI_API_KEY) throw new Error('Falta OPENAI_API_KEY en .env');
  const before = S.load(a).scenes.map((scene) => ({ ...scene }));
  const imageFiles = fs.readdirSync(path.join(S.pdir(a), 'scenes')).filter((file) => /\.png$/i.test(file)).sort();
  const plan = await S.planVisualSequences(a);
  const transcript = JSON.parse(fs.readFileSync(path.join(S.pdir(a), 'transcript', 'transcript.json'), 'utf8'));
  const scenePlan = JSON.parse(fs.readFileSync(path.join(S.pdir(a), 'scene_plan.json'), 'utf8'));
  assert.equal(plan.sequences.length, scenePlan.scenes.length, 'Debe haber una secuencia por escena narrativa');
  let totalFrames = 0;
  for (let i = 0; i < plan.sequences.length; i++) {
    const sequence = plan.sequences[i], source = scenePlan.scenes.find((scene) => scene.scene_id === sequence.source_scene_id);
    assert(source, `Secuencia ${sequence.sequence_id} debe apuntar a una escena`);
    assert.equal(Math.round(sequence.start * 1000), Math.round(source.start * 1000));
    assert.equal(Math.round(sequence.end * 1000), Math.round(source.end * 1000));
    assert(sequence.frames.length > 0, `Secuencia ${sequence.sequence_id} debe tener frames`);
    let cursor = Math.round(sequence.start * 1000);
    for (const frame of sequence.frames) {
      const startMs = Math.round(frame.start * 1000), endMs = Math.round(frame.end * 1000);
      assert.equal(startMs, cursor, `Frame ${frame.frame_id} deja un hueco o solape`);
      assert.equal(Math.round(frame.duration * 1000), endMs - startMs, `Frame ${frame.frame_id} tiene duración inconsistente`);
      assert(frame.duration >= 0.3 || frame.intentional_fast, `Frame ${frame.frame_id} dura menos de 0.3 s sin intención`);
      if (frame.file) assert(fs.existsSync(path.join(S.pdir(a), frame.file)), `Falta imagen reutilizada ${frame.file}`);
      cursor = endMs;
      totalFrames++;
    }
    assert.equal(cursor, Math.round(sequence.end * 1000), `Los frames de secuencia ${sequence.sequence_id} no cubren su duración`);
  }
  assert.equal(Math.round(plan.duration * 1000), Math.round(transcript.duration * 1000));
  assert.deepEqual(S.load(a).scenes, before, 'El planner no debe alterar estados ni imágenes de escena');
  assert.deepEqual(fs.readdirSync(path.join(S.pdir(a), 'scenes')).filter((file) => /\.png$/i.test(file)).sort(), imageFiles,
    'El planner no debe crear ni reemplazar imágenes');
  show({ summary: S.summarizeVisualSequences(a), sequences: plan.sequences.map((sequence) => ({
    sequence_id: sequence.sequence_id, start: sequence.start, end: sequence.end,
    frames: sequence.frames.map((frame) => ({ frame_id: frame.frame_id, start: frame.start, end: frame.end,
      duration: frame.duration, visual: frame.visual, motion: frame.motion, reused: frame.reused })),
  })), imageGenerationCalled: false });
  console.log(`OK: ${totalFrames} frames cover ${plan.duration}s exactly; image files and scene states unchanged.`);
} else if (cmd === 'render') {
  const dir = path.resolve('projects/_tmp'); fs.mkdirSync(dir, { recursive: true });
  const audio = path.join(dir, 'tone.mp3');
  await sh(['-f', 'lavfi', '-i', 'sine=frequency=440:duration=6', audio]);
  const id = await S.newProject(audio, 'render-test', '.mp3');
  const pd = S.pdir(id);
  const [w, h] = S.IMG_SIZE.split('x');
  await sh(['-f', 'lavfi', '-i', `color=c=white:s=${w}x${h}`, '-vf', 'drawbox=x=300:y=300:w=500:h=400:color=red:t=fill', '-frames:v', '1', path.join(pd, 'scenes/scene_001.png')]);
  await sh(['-f', 'lavfi', '-i', `color=c=white:s=${w}x${h}`, '-vf', 'drawbox=x=1200:y=300:w=500:h=400:color=blue:t=fill', '-frames:v', '1', path.join(pd, 'scenes/scene_002.png')]);
  fs.writeFileSync(path.join(pd, 'scene_plan.json'), JSON.stringify({ scenes: [{ scene_id: 1, start: 0, end: 3, duration: 3 }, { scene_id: 2, start: 3, end: 6, duration: 3 }] }));
  fs.writeFileSync(path.join(pd, 'transcript/subtitles.srt'), '1\n00:00:00,000 --> 00:00:03,000\nEscena uno (rojo)\n\n2\n00:00:03,000 --> 00:00:06,000\nEscena dos (azul)\n');
  S.patch(id, (s) => { s.duration = 6; s.scenes = [1, 2].map((n) => ({ scene_id: n, status: 'completed', file: `scenes/scene_00${n}.png` })); });
  await S.render(id);
  console.log(`OK → projects/${id}/output/final_video.mp4 (debe durar 6 s, cambiar a los 3 s, 1920x1080)`);
  const plan = JSON.parse(fs.readFileSync(path.join(pd, 'scene_plan.json'), 'utf8'));
  if (plan.scenes.some((scene, i) => !scene.motion || (i > 0 && scene.motion.type === plan.scenes[i - 1].motion.type)))
    throw new Error('El plan debe guardar motion y variar entre escenas consecutivas');
  const pr = spawnSync(S.FFPROBE, ['-v', 'error', '-show_entries', 'stream=codec_name,width,height,r_frame_rate,duration:format=duration', '-of', 'json', path.join(pd, 'output/final_video.mp4')]);
  if (pr.status !== 0) throw new Error('FFprobe falló al verificar el MP4: ' + pr.stderr);
  const media = JSON.parse(pr.stdout.toString()), video = media.streams.find((x) => x.codec_name === 'h264'), audioStream = media.streams.find((x) => x.codec_name === 'aac');
  if (!video || video.width !== 1920 || video.height !== 1080 || video.r_frame_rate !== '30/1') throw new Error('El video debe ser H.264 1920x1080 a 30 fps');
  if (!audioStream || Math.abs(Number(media.format.duration) - 6) > 0.05) throw new Error('El MP4 debe conservar audio AAC y durar 6 s');
  console.log(JSON.stringify(media, null, 2));
} else if (cmd === 'flow') {
  if (!a) throw new Error('Uso: npm run test:flow -- audio.wav [segundos]');
  if (!process.env.OPENAI_API_KEY) throw new Error('Falta OPENAI_API_KEY en .env');
  fs.mkdirSync('projects/_tmp', { recursive: true });
  const cut = path.resolve('projects/_tmp/flow_cut.mp3');
  await sh(['-i', a, '-t', String(b || 25), '-vn', cut]);
  const srv = S.app.listen(0, '127.0.0.1'); await new Promise((r) => srv.once('listening', r));
  const api = `http://127.0.0.1:${srv.address().port}/api`;
  const call = async (u, o) => { const r = await fetch(api + u, o); return { status: r.status, body: await r.json() }; };
  const ok = (c, m) => { console.log(`${c ? '✓' : '✗'} ${m}`); if (!c) process.exitCode = 1; };
  const h = (await call('/health')).body; console.log('health:', JSON.stringify({ ffmpeg: h.ffmpeg.ok, ffprobe: h.ffprobe.ok, openai: h.openai.ok, models: h.openai.models }));
  const fd = new FormData(); fd.append('audio', new Blob([fs.readFileSync(cut)]), 'flow_test.mp3');
  const up = await call('/projects', { method: 'POST', body: fd }); const id = up.body.id; ok(up.status === 200 && id, `UPLOAD → proyecto ${id}`);
  ok((await call(`/projects/${id}/generate-images`, { method: 'POST' })).body.code === 'no_plan', 'GENERATE IMAGES bloqueado antes de ANALYZE');
  await call(`/projects/${id}/analyze`, { method: 'POST' });
  let st; do { await new Promise((r) => setTimeout(r, 1500)); st = (await call(`/projects/${id}`)).body; } while (st.running);
  ok(!st.error, `TRANSCRIBE + ANALYZE terminados${st.error ? ' — ' + st.error : ''}`); if (st.error) { srv.close(); process.exit(1); }
  show({ summary: st.summary, scenes: st.plan.scenes.map((x) => ({ id: x.scene_id, start: x.start, end: x.end, narration: x.narration, concept: x.visual_concept, prompt: x.prompt })) });
  ok(st.step === 4, 'REVIEW: el pipeline se detiene en el paso 4 (revisión)');
  ok(st.scenes.every((x) => x.status === 'pending' && !x.file), 'todas las escenas siguen en pending (0 imágenes generadas)');
  ok(fs.readdirSync(path.join(S.pdir(id), 'scenes')).length === 0, 'la carpeta scenes/ está vacía');
  ok((await call(`/projects/${id}/scenes/1/regenerate`, { method: 'POST' })).body.code === 'not_approved', 'REGENERATE bloqueado hasta pulsar GENERATE IMAGES');
  await call(`/projects/${id}/analyze`, { method: 'POST' }); await new Promise((r) => setTimeout(r, 500));
  ok((await call(`/projects/${id}`)).body.logs.some((l) => l.includes('Ya completado')), 'ANALYZE repetido reutiliza transcripción y plan (reanudable)');
  console.log(`\nRevisa: projects/${id}/scene_plan.json  ·  Sigue sin gastar en imágenes.`);
  srv.close(); process.exit(process.exitCode || 0);
} else console.log('Comandos: plan | image | render | flow');

