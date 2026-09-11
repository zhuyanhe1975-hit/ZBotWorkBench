import type { TrainingConfig, TrainingJob, TrainingReplayBundle, TrainingResources, TrainingTaskCardSettings } from './types';

const API = '/api/training';

/** Same-origin optional local service. Loading the UI never starts a training process. */
export class TrainingClient {
  private token: string | null = null;

  private async request<T>(path: string, signal?: AbortSignal, body?: unknown): Promise<T> {
    const controller = new AbortController();
    const abort = () => controller.abort();
    if (signal?.aborted) abort();
    else signal?.addEventListener('abort', abort, { once: true });
    const timeout = setTimeout(abort, path === '/resources' ? 30_000 : 15_000);
    try {
      const response = await fetch(`${API}${path}`, {
        signal: controller.signal, credentials: 'same-origin',
        method: body === undefined ? 'GET' : 'POST',
        headers: body === undefined ? {} : { 'Content-Type': 'application/json', 'X-Training-Token': this.token! },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      });
      if (!response.headers.get('content-type')?.includes('application/json')) {
        throw new Error('当前页面未连接本地训练服务。');
      }
      const result = await response.json();
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) this.token = null;
        throw new Error(typeof result?.error === 'string' ? result.error : `训练服务请求失败（HTTP ${response.status}）`);
      }
      return result as T;
    } catch (error) {
      if (controller.signal.aborted && !signal?.aborted) throw new Error('训练服务请求超时，请检查本地服务。');
      throw error;
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener('abort', abort);
    }
  }

  private async mutation<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
    if (!this.token) {
      const session = await this.request<{ token: string }>('/session', signal);
      if (typeof session.token !== 'string' || !session.token) throw new Error('训练服务没有提供有效会话。');
      this.token = session.token;
    }
    return this.request<T>(path, signal, body);
  }

  resources(signal?: AbortSignal) { return this.request<TrainingResources>('/resources', signal); }
  jobs(signal?: AbortSignal) { return this.request<TrainingJob[]>('/jobs', signal); }
  job(id: string, signal?: AbortSignal) { return this.request<TrainingJob>(`/jobs/${encodeURIComponent(id)}`, signal); }
  start(config: TrainingConfig, signal?: AbortSignal) { return this.mutation<TrainingJob>('/jobs', config, signal); }
  stop(id: string, signal?: AbortSignal) { return this.mutation<TrainingJob>(`/jobs/${encodeURIComponent(id)}/stop`, {}, signal); }
  preview(id: string, enabled: boolean, signal?: AbortSignal) { return this.mutation<TrainingJob>(`/jobs/${encodeURIComponent(id)}/preview`, { enabled }, signal); }
  taskCard(id: string, taskCard: TrainingTaskCardSettings, signal?: AbortSignal) { return this.mutation<TrainingJob>(`/jobs/${encodeURIComponent(id)}/task-card`, taskCard, signal); }
  live(id: string, onFrame: (frame: NonNullable<TrainingJob['liveFrame']>) => void, onError?: () => void) {
    const source = new EventSource(`${API}/jobs/${encodeURIComponent(id)}/live`);
    source.onmessage = event => {
      try { onFrame(JSON.parse(event.data)); } catch { onError?.(); }
    };
    source.onerror = () => onError?.();
    return () => source.close();
  }
  bundle(id: string, signal?: AbortSignal) { return this.request<TrainingReplayBundle>(`/jobs/${encodeURIComponent(id)}/bundle`, signal); }
}

export function trainingCheckpointUrl(id: string, name: string): string {
  return `${API}/jobs/${encodeURIComponent(id)}/checkpoints/${encodeURIComponent(name)}`;
}

export function trainingBundleUrl(id: string): string {
  return `${API}/jobs/${encodeURIComponent(id)}/bundle`;
}
