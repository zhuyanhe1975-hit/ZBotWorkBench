export interface RlDebugRecord {
  session: string;
  sequence: number;
  elapsedMs: number;
  event: string;
  details?: Record<string, unknown>;
}

const session = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
const started = typeof performance === 'undefined' ? Date.now() : performance.now();
let sequence = 0;
const records: RlDebugRecord[] = [];
const listeners = new Set<(record: RlDebugRecord) => void>();

export function logRlDebug(event: string, details?: Record<string, unknown>): RlDebugRecord {
  const now = typeof performance === 'undefined' ? Date.now() : performance.now();
  const record: RlDebugRecord = { session, sequence: ++sequence, elapsedMs: Math.round(now - started), event, details };
  records.push(record);
  if (records.length > 200) records.shift();
  console.info(`[RL ${record.elapsedMs}ms] ${event}`, details ?? '');
  listeners.forEach(listener => listener(record));
  try {
    localStorage.setItem('zbot.rlDebugLog', JSON.stringify(records.slice(-100)));
  } catch { /* Diagnostics must never block replay. */ }
  try {
    const body = JSON.stringify(record);
    navigator.sendBeacon?.('/api/rl-debug-log', new Blob([body], { type: 'application/json' }));
  } catch { /* Console and in-memory logs remain available. */ }
  return record;
}

export function getRlDebugRecords(): readonly RlDebugRecord[] { return records; }

export function subscribeRlDebug(listener: (record: RlDebugRecord) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
