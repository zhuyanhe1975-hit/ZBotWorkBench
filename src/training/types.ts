export interface TrainingTask {
  id: string;
  label: string;
  description: string;
  referenceBundleUrl?: string;
  taskCard: TrainingTaskCard;
}

export interface TrainingTaskCardSettings {
  stage1Rewards: Record<string, number>;
  stage2Rewards: Record<string, number>;
  terminatedRewardPenalty: number;
}

export interface TrainingTaskCard extends TrainingTaskCardSettings {
  physicsHz: number;
  controlHz: number;
  contactHistory: number;
  initialStage: 1 | 2;
  jointSpeedRange: [number, number];
}

export interface TrainingResources {
  hostname: string;
  platform: string;
  cpu: { model: string; logicalCores: number; availableThreads: number };
  memory: { totalBytes: number; availableBytes: number };
  gpus: { id: number; name: string; uuid: string; totalMemoryMiB: number; freeMemoryMiB: number; utilization: number; driver: string }[];
  runtime: { available: boolean; python: string; mjlabVersion?: string; torchVersion?: string; warpVersion?: string; cudaBuild?: boolean; error?: string };
  tasks: TrainingTask[];
  scannedRoots: string[];
  checkpoints: { name: string; path: string; bytes: number; modifiedAt: string; resumable: boolean }[];
}

export interface TrainingConfig {
  taskId: string;
  device: string; // cpu or cuda:<detected index>
  cpuThreads: number;
  numEnvs: number;
  iterations: number;
  saveInterval: number;
  seed: number;
  maxSeconds: number;
  taskCard?: TrainingTaskCardSettings;
  resumeJobId?: string;
  resumeCheckpoint?: string;
}

export interface TrainingLiveFrame {
  iteration: number;
  capturedAt: string;
  bodyNames: string[];
  environments: {
    /** World-frame poses sampled directly from the running mjlab batch. */
    environmentOrigin?: [number, number, number];
    bodyPositions: [number, number, number][];
    bodyQuaternions: [number, number, number, number][];
  }[];
}

export interface TrainingJob {
  id: string;
  status: 'queued' | 'starting' | 'running' | 'stopping' | 'stopped' | 'completed' | 'failed';
  config: TrainingConfig;
  createdAt: string;
  startedAt?: string;
  finishedAt?: string;
  metrics: { iteration: number; totalIterations: number; reward?: number; episodeLength?: number; fps?: number; loss?: number; rewardTerms?: Record<string, number> };
  history?: { iteration: number; reward?: number; loss?: number; rewardTerms?: Record<string, number> }[];
  liveFrame?: TrainingLiveFrame;
  livePreviewEnabled?: boolean;
  taskCard?: TrainingTaskCardSettings;
  taskCardRevision?: number;
  checkpoints: { name: string; bytes: number; modifiedAt: string }[];
  logTail: string;
  error?: string;
  bundleReady: boolean;
}

export interface TrainingReplayBundle {
  schemaVersion: 1;
  source: 'mjlab';
  taskId: string;
  name: string;
  xml: string;
  assets: Record<string, string>;
  checkpointBase64: string;
  profile: {
    jointNames: string[];
    defaultAngles: number[];
    observation: 'quaternion';
    inputSize: 26;
    policyFeatures?: 'quat-gravity-heading-v1';
    physicsDt: number;
    controlDt: number;
    jointSpeedLimit: number;
  };
  versions: Record<string, string>;
}
