import initialize from 'physx-js-webidl';
import wasmUrl from 'physx-js-webidl/physx-js-webidl.wasm?url';
import { PhysxSimulation, type PhysxModel } from './physx';

let simulation: PhysxSimulation | null = null;
const reply = (id: number, result?: unknown, error?: unknown) => postMessage({ id, result, error: error ? String(error) : undefined });
const snapshot = () => ({ state: simulation!.state(), bodies: simulation!.getRenderBodies(),
  diagnostics: simulation!.getDiagnostics(), basePosition: simulation!.getBasePosition() });

self.onmessage = async event => {
  const { id, command, payload } = event.data;
  try {
    if (command === 'init') {
      postMessage({ log: 'physx.worker.imported' });
      const create = initialize as unknown as (options: Record<string, unknown>) => Promise<any>;
      const runtime = await create({
        locateFile: () => wasmUrl,
        monitorRunDependencies: (remaining: number) => postMessage({ log: 'physx.worker.dependencies', details: { remaining } }),
      });
      postMessage({ log: 'physx.worker.runtime-ready' });
      simulation = new PhysxSimulation(runtime, payload.model as PhysxModel, null, payload.timing);
      reply(id, snapshot());
    } else if (command === 'step') {
      simulation!.step(payload.targets);
      reply(id, snapshot());
    } else if (command === 'reset') {
      simulation!.reset(payload.elapsedTime, payload.targets);
      reply(id, snapshot());
    } else if (command === 'visualization') {
      simulation!.setDebugVisualization(!!payload.showForces);
      reply(id);
    } else if (command === 'dispose') {
      simulation?.dispose(); simulation = null; reply(id);
      close();
    } else throw new Error(`未知 PhysX Worker 命令 ${command}`);
  } catch (error) { reply(id, undefined, error instanceof Error ? error.message : error); }
};
