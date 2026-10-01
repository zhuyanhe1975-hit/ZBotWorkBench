import wasmUrl from 'physx-js-webidl/physx-js-webidl.wasm?url';
import { logRlDebug } from './debugLog';

let runtime: Promise<any> | null = null;

/** Lazy, same-origin CPU WASM: no CUDA, worker server, or CDN. */
export function loadPhysx(): Promise<any> {
  if (!runtime) {
    logRlDebug('physx.import.start', { wasmUrl });
    runtime = import('physx-js-webidl').then(({ default: initialize }) => {
      logRlDebug('physx.import.done');
      // Upstream's generated declaration omits Emscripten Module options.
      const create = initialize as unknown as (options: {
        locateFile: (path: string) => string;
        monitorRunDependencies: (remaining: number) => void;
        onAbort: (reason: unknown) => void;
        printErr: (message: string) => void;
      }) => Promise<any>;
      logRlDebug('physx.wasm.initialize.start', { wasmUrl });
      let dependencies = -1;
      return create({
        locateFile: path => {
          logRlDebug('physx.wasm.locate-file', { path, wasmUrl });
          return wasmUrl;
        },
        monitorRunDependencies: remaining => {
          if (remaining !== dependencies) {
            dependencies = remaining;
            logRlDebug('physx.wasm.dependencies', { remaining });
          }
        },
        onAbort: reason => logRlDebug('physx.wasm.abort', { reason: String(reason) }),
        printErr: message => logRlDebug('physx.wasm.stderr', { message }),
      }).then(value => {
        logRlDebug('physx.wasm.initialize.done');
        return value;
      });
    }).catch(error => {
      logRlDebug('physx.initialize.error', { message: error instanceof Error ? error.message : String(error) });
      runtime = null;
      throw error;
    });
  } else {
    logRlDebug('physx.runtime.cache-hit');
  }
  return runtime;
}
