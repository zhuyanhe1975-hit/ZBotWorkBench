import wasmUrl from 'physx-js-webidl/physx-js-webidl.wasm?url';

let runtime: Promise<any> | null = null;

/** Lazy, same-origin CPU WASM: no CUDA, worker server, or CDN. */
export function loadPhysx(): Promise<any> {
  if (!runtime) {
    runtime = import('physx-js-webidl').then(({ default: initialize }) => {
      // Upstream's generated declaration omits Emscripten Module options.
      const create = initialize as unknown as (options: { locateFile: () => string }) => Promise<any>;
      return create({ locateFile: () => wasmUrl });
    }).catch(error => { runtime = null; throw error; });
  }
  return runtime;
}
