export interface ZbotModule {
  id: string;
  name: string;
  parentId: string | null; // null for root module
  dockAngle: number; // 0, 90, 180, 270 degrees around docking axis (Euler Z)
  customEuler?: [number, number, number]; // [roll, pitch, yaw] in degrees
  initialAngle?: number; // candidate pose in degrees
  jointAxis: [number, number, number]; // default [0, -1, 1]
  jointRange: [number, number]; // [-180, 180]
  colorA?: string;
  colorB?: string;
}

export type GaitType = 'serpentine' | 'inchworm' | 'rolling' | 'sidewind' | 'trot' | 'manual';

export interface GaitConfig {
  type: GaitType;
  frequency: number; // in Hz (e.g. 0.8)
  amplitude: number; // in degrees (e.g. 45)
  phaseLag: number; // in degrees per module (e.g. 60)
  steering: number; // offset bias in degrees (e.g. -10 to +10 for turning)
  speed: number;
  manualAngles: Record<string, number>; // joint id -> angle in degrees
}

export interface ZbotConfiguration {
  id: string;
  name: string;
  category: 'planar' | 'orthogonal' | 'helical' | 'alternating' | 'leg' | 'zoned' | 'snake' | 'loop' | 'walker' | 'arm' | 'custom';
  baseMode?: 'fixed' | 'free';
  /** Missing means original OBJ; envelope is the legacy benchmark; mechanical adds shaft-aligned housings and connectors. */
  geometryMode?: 'cad' | 'envelope' | 'mechanical';
  hypothesis?: string;
  description: string;
  modules: ZbotModule[];
  rootPos: [number, number, number];
  rootEuler: [number, number, number];
  defaultGait: GaitConfig;
}

export interface SimMetrics {
  endEffectorPos?: [number, number, number];
  centerOfMass?: [number, number, number];
  contactCount?: number;
  time: number;
  fps: number;
  rootPos: [number, number, number];
  rootVelocity: [number, number, number];
  speed: number;
  totalDistance: number;
  jointAngles: Record<string, number>;
  jointTorques: Record<string, number>;
}
