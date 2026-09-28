// Tests por etapas. Uso:
//   npm run test:plan  -- audio.wav [segundos=25]   audio → transcripción → scene plan (SIN imágenes)
//   npm run test:image -- <projectId> [sceneId=1]   una sola escena → gpt-image-2
//   npm run test:image -- "a stick figure ..."      prompt suelto con el style bible
//   npm run test:render                             imagen → FFmpeg → MP4 (sin usar la API, coste 0)
import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import * as S from './server.js';

const [, , cmd, a, b] = process.argv;
const sh = (args) => S.run(S.FFMPEG, ['-y', ...args]);
const show = (o) => console.log(JSON.stringify(o, null, 2));
if (!S.ffmpegOk()) { console.error('FFmpeg no encontrado: instálalo o define FFMPEG_PATH'); process.exit(1); }

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
  await S.run(S.FFMPEG.replace(/ffmpeg(\.exe)?$/, 'ffprobe$1'), ['-v', 'error', '-show_entries', 'stream=codec_name,width,height', '-of', 'default=nw=1', path.join(pd, 'output/final_video.mp4')]).catch(() => {});
} else console.log('Comandos: plan | image | render');

