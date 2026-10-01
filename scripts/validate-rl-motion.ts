/** Reproduce 20-second policy motion in both shipped CPU WASM engines.
 * node --import tsx scripts/validate-rl-motion.ts [--output FILE] [--reference-dir DIR]
 * Native Isaac captures are optional evidence, never a runtime prerequisite.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import loadMujoco from '@mujoco/mujoco';
import loadPhysx from 'physx-js-webidl';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';
import { loadCheckpoint } from '../src/rl/checkpoint';
import { PhysxSimulation, type PhysxModel } from '../src/rl/physx';
import { CONTROL_DT, REPLAY_PROFILES, type ReplayProfile } from '../src/rl/profiles';
import { PolicyReplay } from '../src/rl/replay';

const root = fileURLToPath(new URL('../', import.meta.url));
const currentDurationSeconds = 20;
const migrationDurationSeconds = 2;
let output = resolve(root, 'results/sim-to-sim/validation.json');
let referenceDirectory: string | undefined;
for (let i = 2; i < process.argv.length; i++) {
  const option = process.argv[i];
  if (!['--output', '--reference-dir'].includes(option) || !process.argv[i + 1]) {
    throw new Error('Usage: validate-rl-motion.ts [--output FILE] [--reference-dir DIR]');
  }
  const value = resolve(process.argv[++i]);
  if (option === '--output') output = value;
  else referenceDirectory = value;
}
const hash = (bytes: string | Uint8Array) => createHash('sha256').update(bytes).digest('hex');
const sourceHashes = Object.fromEntries(['src/rl/checkpoint.ts', 'src/rl/replay.ts', 'src/rl/physx.ts', 'src/rl/profiles.ts', 'scripts/validate-rl-motion.ts']
  .map(path => [path, hash(readFileSync(resolve(root, path)))]));

const mujoco = await loadMujoco();
mujoco.FS.writeFile('ma.obj', readFileSync(resolve(root, 'public/assets/ma.obj')));
mujoco.FS.writeFile('mb.obj', readFileSync(resolve(root, 'public/assets/mb.obj')));
const originalProcess = globalThis.process;
const hadWindow = Object.hasOwn(globalThis, 'window');
const originalWindow = (globalThis as any).window;
let physx: any;
try {
  // Upstream targets browsers. Inject its local binary into the same WASM
  // runtime while selecting its browser bootstrap, exactly as the tests do.
  const wasmBinary = readFileSync(resolve(root, 'node_modules/physx-js-webidl/physx-js-webidl.wasm'));
  (globalThis as any).process = undefined;
  (globalThis as any).window = {};
  physx = await (loadPhysx as any)({ wasmBinary });
} finally {
  globalThis.process = originalProcess;
  if (hadWindow) (globalThis as any).window = originalWindow;
  else delete (globalThis as any).window;
}

const experimentalIds = new Set(['Zbot-Direct-8dof-bipedal-v3', 'Zbot-Direct-6dof-bipedal-velocity-v0', 'Zbot-Direct-8dof-bipedal-velocity-v0']);
const migratedIds = new Set(REPLAY_PROFILES.filter(profile => profile.origin === 'isaacgym').map(profile => profile.id));
const numericalOnlyIds = new Set([...experimentalIds, ...migratedIds]);
const motionRequirements: Record<string, { minHeight?: number; maxHeight?: number; xDirection?: number; minX?: number; minXY?: number; transition?: boolean }> = {
  'Zbot-Direct-8dof-bipedal-v0': { minHeight: .30, xDirection: -1, minX: 5 },
  'Zbot-Direct-8dof-snake-v0': { minHeight: .025, maxHeight: .12, minXY: 2 },
  'Zbot-Direct-6dof-bipedal-quat-v0': { minHeight: .22, xDirection: 1, minX: 2.5 },
  'Zbot-Direct-6dof-bipedal-to-snake-v0': { transition: true },
  'Zbot-Direct-6dof-bipedal-to-snake-v1': { transition: true, minXY: 1 },
  'Zbot-Direct-6dof-bipedal-v0': { minHeight: .22, xDirection: 1, minX: 1 },
  'Zbot-Direct-8dof-bipedal-v1': { minHeight: .30, xDirection: -1, minX: 5 },
  'Zbot-Direct-8dof-bipedal-v2': { minHeight: .30, xDirection: 1, minX: 2 },
  'Zbot-Direct-8dof-bird-v0': { minHeight: .25, xDirection: -1, minX: 5 },
  'Zbot-Direct-8dof-wheel-v0': { minHeight: .02, maxHeight: .5, minXY: 2 },
  'Zbot-Direct-6dof-bipedal-velocity-imu-v0': { minHeight: .22, minXY: .5 },
  'Zbot-Direct-8dof-bipedal-run-v0': { minHeight: .30, xDirection: -1, minX: 5 },
  'ZbotRlIsaaclab-6DOF-Periodic-Walking': { minHeight: .22, xDirection: 1, minX: 2 },
  'IsaacGym-ZBotSingleLeg': { minHeight: .18, minXY: .8 },
};
function walkingThreshold(profile: ReplayProfile): number | null {
  return motionRequirements[profile.id]?.minHeight ?? (numericalOnlyIds.has(profile.id) ? .15 : null);
}

function summarize(positions: number[][], threshold: number | null) {
  const start = positions[0];
  const final = positions.at(-1)!;
  const samples = positions.slice(1);
  const displacement = final.map((v, i) => v - start[i]);
  return {
    initialBasePosition: start,
    finalBasePosition: final,
    minBaseHeight: Math.min(...positions.map(p => p[2])),
    maxBaseHeight: Math.max(...positions.map(p => p[2])),
    standHeightThreshold: threshold,
    standFraction: threshold === null || !samples.length ? null : samples.filter(p => p[2] >= threshold).length / samples.length,
    netXYDisplacement: Math.hypot(displacement[0], displacement[1]),
    xProgress: displacement[0],
  };
}

function nativeEvidence(profile: ReplayProfile) {
  const captures: Record<string, string[]> = {
    'Zbot-Direct-8dof-bipedal-v0': ['zbot-isaac-human-cpu.json', 'zbot-isaac-human.json'],
    'Zbot-Direct-8dof-snake-v0': ['zbot-isaac-snake-cpu.json', 'zbot-isaac-snake.json'],
    'Zbot-Direct-6dof-bipedal-quat-v0': ['zbot-isaac-quat.json'],
    'Zbot-Direct-8dof-bipedal-v1': ['zbot-isaac-human_v1.json'],
    'Zbot-Direct-8dof-bipedal-v2': ['zbot-isaac-human_v2.json'],
    'Zbot-Direct-8dof-bipedal-v3': ['zbot-isaac-human_v3.json', 'zbot-isaac-human_v3-historical.json', 'zbot-isaac-human_v3-historical-exact.json'],
    'Zbot-Direct-8dof-bird-v0': ['zbot-isaac-bird.json'],
    'Zbot-Direct-8dof-wheel-v0': ['zbot-isaac-wheel.json'],
    'Zbot-Direct-8dof-bipedal-run-v0': ['zbot-isaac-run.json', 'zbot-isaac-run-sensor.json'],
    'Zbot-Direct-6dof-bipedal-velocity-v0': ['zbot-isaac-velocity6-commanded.json'],
    'Zbot-Direct-8dof-bipedal-velocity-v0': ['zbot-isaac-velocity8-commanded.json'],
    'Zbot-Direct-6dof-bipedal-velocity-imu-v0': ['zbot-isaac-imu-commanded.json'],
  };
  if (!referenceDirectory) return [];
  return [...(captures[profile.id] ?? []), `${profile.id}.json`].flatMap(filename => {
    const path = resolve(referenceDirectory!, filename);
    if (!existsSync(path)) return [];
    const bytes = readFileSync(path);
    const capture = JSON.parse(bytes.toString('utf8'));
    const base = capture.bodyNames?.indexOf('base');
    if (capture.task !== profile.id || base === undefined || base < 0 || !Array.isArray(capture.rows) || capture.rows.length < 2) {
      throw new Error(`Invalid optional native capture: ${path}`);
    }
    const positions: number[][] = capture.rows.map((row: any) => row.bodyPos[base]);
    if (positions.some(p => p.length !== 3 || !p.every(Number.isFinite))) throw new Error(`Non-finite native capture: ${path}`);
    return [{
      externalFile: path, sha256: hash(bytes), sourceImplementation: capture.source,
      provenance: 'Optional external native Isaac capture; not required to run or regenerate WASM validation.',
      controlSteps: positions.length - 1, controlDt: capture.controlDt,
      ...summarize(positions, walkingThreshold(profile)),
    }];
  });
}

function run(profile: ReplayProfile, engine: 'physx' | 'mujoco') {
  const display = new MujocoEngine();
  let dynamics: PhysxSimulation | undefined;
  const positions: number[][] = [];
  let completedSteps = 0;
  let error: string | null = null;
  let simulationTime = 0;
  let warnings: { index: number; count: number; lastInfo: number }[] = [];
  const started = performance.now();
  try {
    (display as any).mujoco = mujoco;
    display.loadModelFromXml(readFileSync(resolve(root, `public/rl/models/${profile.displayModel ?? profile.model}.xml`), 'utf8'));
    if (engine === 'physx') {
      const model: PhysxModel = JSON.parse(readFileSync(resolve(root, `public/rl/physx/${profile.model}.json`), 'utf8'));
      model.simulation = { ...model.simulation, ...profile.physxSimulation };
      if (profile.origin === 'isaacgym') model.defaultQ = [...profile.defaultAngles];
      model.displayJointNames = profile.displayJointNames;
      model.displayJointSigns = profile.displayJointSigns;
      model.displayRootQuaternionOffset = profile.displayRootQuaternionOffset;
      dynamics = new PhysxSimulation(physx, model, display,
        { physicsDt: profile.physicsDt, controlDt: profile.controlDt,
          rootPosition: profile.initialRootPosition, rootQuaternion: profile.initialRootQuaternion });
    }
    const bytes = readFileSync(resolve(root, `public/rl/checkpoints/${profile.id}/${profile.checkpoint}`));
    const policy = loadCheckpoint(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    const replay = new PolicyReplay(display, profile, policy, dynamics);
    const model = display.getModel();
    const nameAt = (address: number) => {
      let name = '';
      for (let i = address; i >= 0 && i < model.names.length && model.names[i]; i++) name += String.fromCharCode(model.names[i]);
      return name;
    };
    const base = dynamics ? -1 : Array.from(model.name_bodyadr as Int32Array).findIndex(address => nameAt(address) === 'base');
    if (!dynamics && base < 0) throw new Error('Missing base body');
    const position = () => dynamics?.getBasePosition() ?? Array.from(display.getData().xpos.slice(3 * base, 3 * base + 3)) as number[];
    positions.push(position());
    const durationSeconds = profile.origin === 'isaacgym' ? migrationDurationSeconds : currentDurationSeconds;
    const steps = Math.round(durationSeconds / (profile.controlDt ?? CONTROL_DT));
    for (let i = 0; i < steps; i++) {
      replay.step();
      const point = position();
      if (point.length !== 3 || !point.every(Number.isFinite)) throw new Error('Non-finite base position');
      positions.push(point);
      completedSteps++;
    }
  } catch (caught) {
    error = caught instanceof Error ? caught.message : String(caught);
  } finally {
    const data = display.getData();
    if (data) {
      simulationTime = data.time;
      warnings = Array.from({ length: data.warning.size() }, (_, index) => {
        const warning = data.warning.get(index);
        return { index, count: warning.number, lastInfo: warning.lastinfo };
      }).filter(warning => warning.count > 0);
    }
    dynamics?.dispose();
    display.destroy();
  }
  const metrics = positions.length ? summarize(positions, walkingThreshold(profile)) : null;
  const durationSeconds = profile.origin === 'isaacgym' ? migrationDurationSeconds : currentDurationSeconds;
  const controlDt = profile.controlDt ?? CONTROL_DT;
  const expectedSteps = Math.round(durationSeconds / controlDt);
  let accepted = completedSteps === expectedSteps && error === null && Number.isFinite(simulationTime)
    && Math.abs(simulationTime - expectedSteps * controlDt) < 1e-7;
  const failures: string[] = [];
  if (!accepted) failures.push(error ?? 'Incomplete or invalid simulation');
  const requirement = motionRequirements[profile.id];
  if (!requirement && !numericalOnlyIds.has(profile.id)) failures.push('Task has no explicit validation classification');
  if (metrics && requirement) {
    if (requirement.minHeight !== undefined && metrics.minBaseHeight <= requirement.minHeight) failures.push(`Base height must stay above ${requirement.minHeight}m`);
    if (requirement.maxHeight !== undefined && metrics.maxBaseHeight >= requirement.maxHeight) failures.push(`Base height must stay below ${requirement.maxHeight}m`);
    if (requirement.minX !== undefined && metrics.xProgress * requirement.xDirection! <= requirement.minX) failures.push(`Directed X progress must exceed ${requirement.minX}m`);
    if (requirement.minXY !== undefined && metrics.netXYDisplacement <= requirement.minXY) failures.push(`XY displacement must exceed ${requirement.minXY}m`);
    if (requirement.transition && !(metrics.initialBasePosition[2] > .22 && metrics.finalBasePosition[2] > .025 && metrics.finalBasePosition[2] < .12)) failures.push('Transition must start upright and finish with base height between .025m and .12m');
  }
  accepted &&= failures.length === 0;
  return { engine, completedSteps, controlDt: profile.controlDt ?? CONTROL_DT, simulationTime, elapsedMs: performance.now() - started,
    acceptanceScope: requirement ? 'motion-validated' : migratedIds.has(profile.id) ? 'migration-numerical-only' : 'experimental-numerical-only',
    motionRequirement: requirement ?? null, nativeWarning: profile.note ?? null,
    accepted, failures, error, mujocoWarnings: warnings, metrics };
}

const results = REPLAY_PROFILES.map(profile => {
  const checkpoint = `public/rl/checkpoints/${profile.id}/${profile.checkpoint}`;
  const physxResult = run(profile, 'physx');
  const mujocoResult = profile.mujocoCompatible === false ? { engine: 'mujoco', accepted: false, skipped: true, reason: 'Task requires PhysX velocity/contact state', metrics: null } : run(profile, 'mujoco');
  const physxLabel = physxResult.accepted ? physxResult.acceptanceScope === 'motion-validated' ? 'PASS'
    : `COMPLETE (${physxResult.acceptanceScope === 'migration-numerical-only' ? 'migration' : 'experimental'}; motion ungraded)` : 'FAIL';
  console.log(`${profile.label}: PhysX ${physxLabel}, MuJoCo ${'skipped' in mujocoResult ? 'N/A' : mujocoResult.accepted ? mujocoResult.acceptanceScope === 'experimental-numerical-only' ? 'COMPLETE (experimental)' : 'PASS' : 'FAIL'}; `
    + `PhysX x=${physxResult.metrics?.xProgress.toFixed(3)}, min height=${physxResult.metrics?.minBaseHeight.toFixed(3)}`);
  return { task: profile.id, checkpoint, checkpointSha256: hash(readFileSync(resolve(root, checkpoint))),
    physxModelSha256: hash(readFileSync(resolve(root, `public/rl/physx/${profile.model}.json`))),
    mujocoModelSha256: hash(readFileSync(resolve(root, `public/rl/models/${profile.displayModel ?? profile.model}.xml`))),
    runs: [physxResult, mujocoResult], optionalNativeReferences: nativeEvidence(profile) };
});
const passed = results.every(result => result.runs[0].accepted);
const artifact = {
  schemaVersion: 3, generatedAt: new Date().toISOString(), passed,
  summary: { totalPolicies: results.length, motionValidated: results.filter(r => r.runs[0].accepted && 'acceptanceScope' in r.runs[0] && r.runs[0].acceptanceScope === 'motion-validated').length,
    experimental: results.filter(r => experimentalIds.has(r.task)).length,
    migrated: results.filter(r => migratedIds.has(r.task)).length },
  command: 'node --import tsx scripts/validate-rl-motion.ts [--output FILE] [--reference-dir DIR]',
  durations: { currentPoliciesSeconds: currentDurationSeconds, migratedPoliciesSeconds: migrationDurationSeconds }, variableControlTiming: true,
  runtimes: { node: process.version, physxPackage: JSON.parse(readFileSync(resolve(root, 'node_modules/physx-js-webidl/package.json'), 'utf8')).version,
    physxSdk: [24, 16, 8].map(shift => (physx.PHYSICS_VERSION >> shift) & 255).join('.'),
    mujocoPackage: JSON.parse(readFileSync(resolve(root, 'node_modules/@mujoco/mujoco/package.json'), 'utf8')).version },
  sourceHashes,
  acceptance: {
    motion: '14 tasks must pass their explicit per-task height and displacement requirements, including IsaacLab periodic walking and native-aligned SingleLeg. Existing five baselines retain their original motion thresholds.',
    experimental: 'Three source checkpoints (8DOF v3, 6DOF velocity, 8DOF velocity) are graded for finite complete 20-second replay only. Native instability warnings remain visible; this is not a motion success claim. Their .15m height fraction is descriptive only.',
    migrated: 'Legacy Isaac Gym checkpoints are graded for actor dimensions, exact family-level network inference, and finite 2-second migration replay only until their original importer-level dynamics are captured.',
    comparison: 'MuJoCo results are diagnostic and do not gate exit status. Only PhysX acceptance gates exit status.',
  },
  limitations: ['A deterministic 20-second check is not a guarantee over arbitrary starts, checkpoints, or longer horizons.',
    'PhysX CPU WASM and native Isaac trajectories may diverge; this checks motion behavior, not bitwise solver equivalence.',
    'Optional native capture summaries have external-file provenance and can be omitted when regenerating elsewhere.'],
  results,
};
mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, JSON.stringify(artifact, null, 2) + '\n');
console.log(`Validation ${passed ? 'passed' : 'failed'}: ${output}`);
if (!passed) process.exitCode = 1;
