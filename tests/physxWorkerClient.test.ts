import assert from 'node:assert/strict';
import { test } from 'node:test';
import { PhysxWorkerSimulation } from '../src/rl/physxWorkerClient';

test('PhysX worker initialization timeout terminates the worker without blocking the caller', async () => {
  let terminated = false;
  class SilentWorker {
    onmessage: ((event: any) => void) | null = null;
    onerror: ((event: any) => void) | null = null;
    postMessage() { /* Deliberately never replies. */ }
    terminate() { terminated = true; }
  }
  const previousWorker = globalThis.Worker;
  const previousWindow = (globalThis as any).window;
  try {
    (globalThis as any).Worker = SilentWorker;
    (globalThis as any).window = globalThis;
    const started = performance.now();
    await assert.rejects(PhysxWorkerSimulation.create({ defaultQ: [] } as any, {} as any, {}, 20), /超过 0.02 秒/);
    assert.ok(performance.now() - started < 500, 'timeout must return control promptly');
    assert.equal(terminated, true);
  } finally {
    (globalThis as any).Worker = previousWorker;
    (globalThis as any).window = previousWindow;
  }
});
