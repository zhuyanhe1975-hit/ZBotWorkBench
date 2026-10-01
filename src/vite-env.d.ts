/// <reference types="vite/client" />

declare module 'virtual:zbot-physx-models' {
  const models: Readonly<Record<string, import('./rl/physx').PhysxModel>>;
  export default models;
}
