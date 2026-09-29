// Tests por etapas. Uso:
//   npm run test:plan  -- audio.wav [segundos=25]   audio → transcripción → scene plan (SIN imágenes)
//   npm run test:image -- <projectId> [sceneId=1]   una sola escena → gpt-image-2
//   npm run test:image -- "a stick figure ..."      prompt suelto con el style bible
//   npm run test:render                             imagen → FFmpeg → MP4 (sin usar la API, coste 0)
//   npm run test:flow -- audio.wav [segundos=25]    UPLOAD → TRANSCRIBE → ANALYZE → REVIEW por la API HTTP. Usa OpenAI para transcribir y planificar
//                                                   (céntimos) pero NUNCA genera imágenes: comprueba que todo queda en 'pending'.
import 'dotenv/config';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
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
} else if (cmd === 'storyboard') {
  if (!a || !fs.existsSync(path.join('projects', a, 'state.json'))) throw new Error('Uso: npm run test:storyboard -- <projectId>');
  const projectDir = S.pdir(a);
  const sourcePlanFile = path.join(projectDir, 'visual_sequence_plan.json');
  const scenePlanFile = path.join(projectDir, 'scene_plan.json');
  const stateFile = path.join(projectDir, 'state.json');
  if (!fs.existsSync(sourcePlanFile)) throw new Error('Falta visual_sequence_plan.json; el test no lo va a generar ni modificar.');
  const protectedFiles = [sourcePlanFile, scenePlanFile, stateFile].map((file) => [file, fs.readFileSync(file)]);
  const imageSnapshot = ['scenes', 'frames'].flatMap((folder) => {
    const directory = path.join(projectDir, folder);
    return fs.existsSync(directory) ? fs.readdirSync(directory).filter((file) => /\.png$/i.test(file)).sort().map((file) => {
      const contents = fs.readFileSync(path.join(directory, file));
      return [`${folder}/${file}`, createHash('sha256').update(contents).digest('hex')];
    }) : [];
  });
  const sourcePlan = JSON.parse(fs.readFileSync(sourcePlanFile, 'utf8'));
  const scenePlan = JSON.parse(fs.readFileSync(scenePlanFile, 'utf8'));
  const previousStoryboardFile = path.join(projectDir, 'visual_storyboard.json');
  const previousFirstTwelve = fs.existsSync(previousStoryboardFile)
    ? JSON.parse(fs.readFileSync(previousStoryboardFile, 'utf8')).sequences[0].frames.slice(0, 12).map((frame) => ({
      action: frame.action, character_state: frame.character_state, object_state: frame.object_state, prompt: frame.prompt,
    })) : null;
  const storyboard = await S.generateVisualStoryboard(a);
  assert.ok(storyboard && Array.isArray(storyboard.sequences), 'Debe devolver un storyboard con secuencias');
  assert.equal(storyboard.sequences.length, scenePlan.scenes.length, 'Debe conservar una secuencia por cada escena narrativa');
  assert.equal(storyboard.source_beat_count, scenePlan.scenes.length, 'Los beats narrativos corresponden a escenas fuente');
  assert.equal(storyboard.source_microbeat_count, sourcePlan.sequences.reduce((total, sequence) => total + sequence.frames.length, 0));
  assert.ok(storyboard.total_frames >= 20, 'Debe haber suficientes frames para leer una progresión');
  const maxStoryboardFrames = S.load(a).mode === 'ai_voice' ? 360 : 180;
  assert.ok(storyboard.total_frames <= maxStoryboardFrames, 'Debe mantenerse en un rango razonable para el primer storyboard');
  const allFrames = storyboard.sequences.flatMap((sequence) => sequence.frames);
  const sourceBeatIds = scenePlan.scenes.map((scene) => scene.scene_id);
  const sourceMicrobeatIds = sourcePlan.sequences.flatMap((sequence) => sequence.frames.map((frame) => frame.frame_id));
  const prompts = new Set();
  let total = 0, previousFrame = null;
  for (const seq of storyboard.sequences) {
    assert.ok(Array.isArray(seq.frames), `La secuencia ${seq.sequence_id} debe contener frames`);
    total += seq.frames.length;
    const beats = [...new Set(seq.frames.map((frame) => frame.source_beat_id))];
    assert.equal(beats.length, 1, `Secuencia ${seq.sequence_id} debe pertenecer a un solo beat narrativo`);
    assert.equal(seq.source_beat_count, 1);
    assert.ok(seq.frames.every((frame) => sourceMicrobeatIds.includes(frame.source_plan_beat_id)),
      `Secuencia ${seq.sequence_id} debe conservar referencias a los micro-beats del plan`);
    assert.equal(seq.frames[0].start, seq.start, `La secuencia ${seq.sequence_id} debe comenzar en su timestamp original`);
    assert.equal(seq.frames.at(-1).end, seq.end, `La secuencia ${seq.sequence_id} debe terminar en su timestamp original`);
    for (const frame of seq.frames) {
      assert.ok(frame.prompt, `El frame ${frame.frame_id} debe llevar prompt`);
      assert.ok(frame.continuity_notes, `El frame ${frame.frame_id} debe explicar continuidad`);
      assert.ok(frame.action, `El frame ${frame.frame_id} debe tener una acción concreta`);
      assert.ok(sourceBeatIds.includes(frame.source_beat_id), `El frame ${frame.frame_id} debe referenciar un beat fuente existente`);
      assert.ok(frame.previous_frame_state && frame.current_frame_state, `El frame ${frame.frame_id} debe declarar estados antes y después`);
      assert.ok(frame.character_state && frame.environment_state && frame.object_state && frame.camera_state,
        `El frame ${frame.frame_id} debe declarar continuidad de personaje, entorno, objeto y cámara`);
      assert.ok(frame.continuity_score >= 0 && frame.continuity_score <= 1, `Frame ${frame.frame_id} tiene continuity_score fuera de 0..1`);
      assert.ok(!prompts.has(frame.prompt), `Prompt duplicado en el frame ${frame.frame_id}`);
      prompts.add(frame.prompt);
      if (previousFrame) assert.equal(frame.previous_frame_id, previousFrame.frame_id, `El frame ${frame.frame_id} no enlaza el anterior`);
      else assert.equal(frame.previous_frame_id, null, 'El primer frame debe abrir la secuencia global');
      const next = allFrames[allFrames.indexOf(frame) + 1];
      if (next) assert.equal(frame.next_action, next.action, `next_action no coincide con la acción del frame ${next.frame_id}`);
      else assert.ok(frame.next_action, 'El último frame debe indicar cómo termina la acción');
      assert.equal(Math.round(frame.duration * 1000), Math.round((frame.end - frame.start) * 1000), `Duración inconsistente en frame ${frame.frame_id}`);
      if (previousFrame) {
        assert.notEqual(frame.current_frame_state, previousFrame.current_frame_state, `Frame ${frame.frame_id} no cambia nada`);
        if (frame.sequence_id === previousFrame.sequence_id)
          assert.equal(frame.environment_state, previousFrame.environment_state, `El entorno cambia dentro de secuencia ${frame.sequence_id}`);
      }
      previousFrame = frame;
    }
  }
  const beatBoundaries = allFrames.filter((frame, index) => index > 0 && frame.source_beat_id !== allFrames[index - 1].source_beat_id);
  assert.deepEqual(beatBoundaries.map((frame) => frame.source_beat_id), scenePlan.scenes.slice(1).map((scene) => scene.scene_id),
    'Cada nuevo beat debe empezar en el límite de su escena narrativa');
  assert.equal(beatBoundaries[0].frame_id, 13, 'El beat del ascensor debe comenzar en el frame 13');
  for (const frame of beatBoundaries) {
    const previous = allFrames[allFrames.indexOf(frame) - 1];
    assert.equal(frame.sequence_transition, true, `Frame ${frame.frame_id} debe señalar el corte narrativo`);
    assert.notEqual(frame.environment_state, previous.environment_state, `Frame ${frame.frame_id} debe establecer el entorno del nuevo beat`);
    assert.notEqual(frame.action, previous.action, `Frame ${frame.frame_id} no debe repetir la acción anterior`);
  }
  if (previousFirstTwelve) assert.deepEqual(storyboard.sequences[0].frames.slice(0, 12).map((frame) => ({
    action: frame.action, character_state: frame.character_state, object_state: frame.object_state, prompt: frame.prompt,
  })), previousFirstTwelve, 'Las acciones, estados y prompts de los frames 1–12 deben conservarse');
  assert.equal(total, storyboard.total_frames, 'El total de frames debe coincidir con el conteo global');
  assert.equal(fs.existsSync(path.join(S.pdir(a), 'visual_storyboard.json')), true, 'Debe crearse el archivo visual_storyboard.json');
  const saved = JSON.parse(fs.readFileSync(path.join(S.pdir(a), 'visual_storyboard.json'), 'utf8'));
  assert.deepEqual(saved, storyboard, 'El archivo guardado debe coincidir con el storyboard generado');
  assert.equal(storyboard.validation.status, 'passed', `Auditoría de continuidad falló: ${JSON.stringify(storyboard.validation.issues)}`);
  for (const [file, contents] of protectedFiles) assert.deepEqual(fs.readFileSync(file), contents, `${path.basename(file)} no debe modificarse`);
  const imageSnapshotAfter = ['scenes', 'frames'].flatMap((folder) => {
    const directory = path.join(projectDir, folder);
    return fs.existsSync(directory) ? fs.readdirSync(directory).filter((file) => /\.png$/i.test(file)).sort().map((file) => {
      const image = fs.readFileSync(path.join(directory, file));
      return [`${folder}/${file}`, createHash('sha256').update(image).digest('hex')];
    }) : [];
  });
  assert.deepEqual(imageSnapshotAfter, imageSnapshot, 'Las imágenes existentes deben permanecer intactas');
  const firstTwenty = storyboard.sequences[0].frames.slice(0, 20).map((frame) => ({
    frame_id: frame.frame_id, source_beat_id: frame.source_beat_id, previous_frame_id: frame.previous_frame_id,
    action: frame.action, next_action: frame.next_action, character_state: frame.character_state, object_state: frame.object_state,
  }));
  show({ id: a, total_sequences: storyboard.sequences.length, source_beats: storyboard.source_beat_count,
    total_frames: storyboard.total_frames, average_frame_duration: storyboard.average_frame_duration,
    validation: storyboard.validation, imageGenerationCalled: false, first_sequence_first_20_frames: firstTwenty });
  console.log('OK: progresión de acciones validada; scene_plan, visual_sequence_plan, state e imágenes permanecen intactos.');
} else if (cmd === 'storyboard-sample') {
  if (!a || !fs.existsSync(path.join('projects', a, 'state.json'))) throw new Error('Uso: node test.js storyboard-sample <projectId>');
  const projectDir = S.pdir(a);
  const protectedFiles = ['scene_plan.json', 'visual_sequence_plan.json', 'visual_storyboard.json']
    .map((name) => [name, fs.readFileSync(path.join(projectDir, name))]);
  const stateBefore = JSON.parse(fs.readFileSync(path.join(projectDir, 'state.json'), 'utf8'));
  const oldFrameDirectory = path.join(projectDir, 'frames');
  const oldFrames = fs.readdirSync(oldFrameDirectory).filter((file) => /^frame_\d+\.png$/i.test(file)).sort().map((file) => {
    const image = fs.readFileSync(path.join(oldFrameDirectory, file));
    return [file, createHash('sha256').update(image).digest('hex')];
  });
  const generated = await S.generateStoryboardSampleFrames(a);
  assert.equal(generated.frames.length, 10, 'La muestra debe contener exactamente diez imágenes');
  assert.deepEqual(generated.frames.map((frame) => frame.frame_id), Array.from({ length: 10 }, (_, index) => index + 1));
  assert.ok(generated.frames.every((frame) => frame.file.startsWith('frames/ai_voice/')),
    'Las imágenes deben quedar aisladas de los cinco PNG previos');
  const rendered = await S.renderStoryboardSample(a);
  assert.equal(rendered.frames.length, 10);
  for (const [name, contents] of protectedFiles)
    assert.deepEqual(fs.readFileSync(path.join(projectDir, name)), contents, `${name} no debe modificarse durante la muestra`);
  const stateAfter = JSON.parse(fs.readFileSync(path.join(projectDir, 'state.json'), 'utf8'));
  const { logs: originalLogs, ...stateContentBefore } = stateBefore;
  const { logs: updatedLogs, ...stateContentAfter } = stateAfter;
  assert.deepEqual(stateContentAfter, stateContentBefore, 'La muestra no debe cambiar el estado del proyecto fuera del log');
  assert.deepEqual(updatedLogs.slice(0, originalLogs.length), originalLogs, 'La muestra no debe alterar logs anteriores');
  assert.ok(updatedLogs.slice(originalLogs.length).every((line) => line.includes('[STORYBOARD SAMPLE ')),
    'El único cambio permitido en state.json son los logs de esta muestra');
  const oldFramesAfter = fs.readdirSync(oldFrameDirectory).filter((file) => /^frame_\d+\.png$/i.test(file)).sort().map((file) => {
    const image = fs.readFileSync(path.join(oldFrameDirectory, file));
    return [file, createHash('sha256').update(image).digest('hex')];
  });
  assert.deepEqual(oldFramesAfter, oldFrames, 'Los cinco PNG anteriores deben permanecer intactos');
  show({ id: a, imageGenerationCalled: true, model: generated.model, reference_mode: generated.reference_mode,
    generated: generated.frames, render: rendered, oldTestImagesModified: false });
  console.log('OK: solo frames 1–10 de sequence 1; cortes directos, audio/subtítulos y cinco imágenes previas intactos.');
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
} else if (cmd === 'voice') {
  if (!process.env.OPENAI_API_KEY) throw new Error('Falta OPENAI_API_KEY en .env');
  const projectDirsBefore = fs.readdirSync('projects').filter((entry) => fs.existsSync(path.join('projects', entry, 'state.json'))).sort();
  const imageFilesBefore = fs.readdirSync('projects', { recursive: true }).filter((file) => /\.png$/i.test(file)).sort();
  const result = await S.generateVoiceTest();
  assert.deepEqual(result.files.map((entry) => entry.file).sort(), ['voice_test_cedar.mp3', 'voice_test_marin.mp3']);
  assert.ok(result.files.every((entry) => entry.duration > 0));
  assert.ok(result.apiCalls >= 0 && result.apiCalls <= 2, 'Como máximo genera una vez por voz');
  assert.equal(result.imageGenerationCalled, false);
  assert.deepEqual(fs.readdirSync('projects').filter((entry) => fs.existsSync(path.join('projects', entry, 'state.json'))).sort(), projectDirsBefore,
    'El test de voz no crea proyectos de video');
  assert.deepEqual(fs.readdirSync('projects', { recursive: true }).filter((file) => /\.png$/i.test(file)).sort(), imageFilesBefore,
    'El test de voz no debe generar imágenes');
  show({ ...result, test_text: S.TTS_TEST_TEXT, images_generated: 0 });
  console.log('OK: mismo texto en Marin y Cedar; test de voz sin proyectos ni imágenes.');
} else if (cmd === 'transcription') {
  if (!a || !fs.existsSync(path.join(S.pdir(a), 'state.json'))) throw new Error('Uso: npm run test:transcription -- <projectId>');
  const transcript = await S.transcribe(a, { forceModel: 'whisper-1' });
  assert.equal(transcript.model, 'whisper-1', 'AI Voice debe usar whisper-1 como fuente temporal');
  assert.ok(transcript.duration > 0 && transcript.segments.length > 0 && transcript.words.length > 0);
  const audio = path.join(S.pdir(a), 'audio', S.load(a).audioFile);
  assert.ok(Math.abs(transcript.duration - S.audioDuration(audio)) < 0.5, 'La transcripción debe corresponder al audio final');
  show({ project: a, model: transcript.model, duration: transcript.duration, segments: transcript.segments.length,
    words: transcript.words.length, imageGenerationCalled: false });
  console.log('OK: timestamps calculados desde narration_final.mp3 con whisper-1.');
} else if (cmd === 'frames') {
  if (!a || !fs.existsSync(path.join(S.pdir(a), 'state.json'))) throw new Error('Uso: npm run test:frames -- <projectId>');
  const state = S.load(a);
  if (state.mode !== 'ai_voice' || state.aiVoice?.status !== 'review') throw new Error('El proyecto AI Voice debe tener storyboard en revisión.');
  const generated = await S.generateStoryboardSampleFrames(a);
  assert.equal(generated.frames.length, 10);
  assert.deepEqual(generated.frames.map((frame) => frame.frame_id), Array.from({ length: 10 }, (_, index) => index + 1));
  assert.ok(generated.frames.every((frame) => frame.file.startsWith('frames/ai_voice/')));
  show({ project: a, frames: generated.frames, model: generated.model, reference_mode: generated.reference_mode, imageGenerationCalled: true });
  console.log('OK: generados/reutilizados exactamente los primeros diez frames consecutivos.');
} else if (cmd === 'full') {
  if (!process.env.OPENAI_API_KEY) throw new Error('Falta OPENAI_API_KEY en .env');
  const project = S.createAIVoiceProject({ title: S.AI_VOICE_DEMO_TITLE, script: S.AI_VOICE_DEMO_SCRIPT,
    voice: S.TTS_DEFAULT_VOICE, sources: S.AI_VOICE_DEMO_SOURCES });
  const id = project.id, before = S.load(id).aiVoice;
  if (before.status !== 'review') await S.runAIVoicePipeline(id);
  const current = S.load(id);
  const sampleVideo = current.aiVoice.sampleVideo?.status === 'completed'
    ? current.aiVoice.sampleVideo : await S.generateAIVoiceSampleVideo(id);
  const transcript = JSON.parse(fs.readFileSync(path.join(S.pdir(id), 'transcript', 'transcript.json'), 'utf8'));
  const storyboard = JSON.parse(fs.readFileSync(path.join(S.pdir(id), 'visual_storyboard.json'), 'utf8'));
  assert.equal(transcript.model, 'whisper-1');
  assert.equal(storyboard.validation.status, 'passed');
  assert.ok(storyboard.sequences[0].frames.length >= 10, 'La primera secuencia debe contener diez frames consecutivos para la prueba visual');
  assert.ok(storyboard.total_frames >= 180 && storyboard.total_frames <= 360, 'Unos 60–70 segundos deben tener densidad flipbook cercana a 4 fps');
  assert.equal(sampleVideo.requested_frames, 10);
  assert.equal(fs.readdirSync(path.join(S.pdir(id), 'frames', 'ai_voice')).filter((file) => /\.png$/i.test(file)).length, 10);
  show({ project: id, voice: current.aiVoice.voice, tts_model: current.aiVoice.model, duration: current.duration,
    transcription_model: transcript.model, scenes: storyboard.sequences.length, frames: storyboard.total_frames,
    sample_video: sampleVideo.output, sample_duration: sampleVideo.duration,
    imageGenerationCalled: true, generatedImages: 10, noAdditionalFramesGenerated: true });
  console.log('OK: GUIÓN → voz → Whisper → storyboard → 10 imágenes → test video.');
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

