import assert from 'node:assert/strict';
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import loadMujoco from '@mujoco/mujoco';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';
import { loadTrainingBundle } from '../src/training/bundle';

// Deterministic single-environment WASM gate. This does not measure contact slip
// or replace the native multi-environment first-episode gait assessment.
const files = process.argv.slice(2);
if (!files.length) files.push('public/rl/training/walking-reference.json');
const engine = new MujocoEngine();
(engine as any).mujoco = await loadMujoco();
const results = [];
try {
  for (const file of files) {
    const { replay } = loadTrainingBundle(engine, JSON.parse(readFileSync(file, 'utf8')));
    const model = engine.getModel(), data = engine.getData();
    const name = (address: number) => {
      let text = '';
      for (let i = address; i >= 0 && i < model.names.length && model.names[i]; i++) text += String.fromCharCode(model.names[i]);
      return text;
    };
    const sensor = Array.from(model.name_sensoradr as Int32Array).findIndex(address => name(address) === 'rl_base_pos');
    assert.ok(sensor >= 0, 'Training world must expose actual base position');
    const address = model.sensor_adr[sensor];
    const position = () => Array.from(data.sensordata.slice(address, address + 3)) as number[];
    const start = position();
    let minimumBaseHeight = start[2], maximumLateralDeviation = 0;
    for (let step = 0; step < 600; step++) {
      replay.step();
      const pos = position();
      minimumBaseHeight = Math.min(minimumBaseHeight, pos[2]);
      maximumLateralDeviation = Math.max(maximumLateralDeviation, Math.abs(pos[1] - start[1]));
    }
    const end = position();
    const result = { file, seconds: engine.getTime(), forwardMetres: end[0] - start[0], minimumBaseHeight, maximumLateralDeviation };
    results.push(result);
    assert.ok(result.forwardMetres > 3, 'Reference must travel forward more than3m');
    assert.ok(minimumBaseHeight >= .22, 'Base must remain above task fall threshold');
    assert.ok(maximumLateralDeviation < .5, 'Lateral deviation must remain within task bound');
  }
} finally { engine.destroy(); }
mkdirSync('results/training/walking', { recursive: true });
writeFileSync('results/training/walking/wasm-evaluation.json', JSON.stringify({ scope: '600 control steps, no reset; displacement/height/lateral checks only', results }, null, 2) + '\n');
console.log(JSON.stringify(results, null, 2));
