import express, { type NextFunction, type Request, type Response } from 'express';
import { randomBytes, randomUUID, timingSafeEqual } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createTrainingService, TrainingError, type TrainingServiceOptions } from './trainingService';

const loopbackHost = (host: string) => ['localhost', '127.0.0.1', '[::1]', '::1'].includes(host.toLowerCase());
const loopbackAddress = (address?: string) => address === '127.0.0.1' || address === '::1' || address === '::ffff:127.0.0.1';
export async function createTrainingApp(options: TrainingServiceOptions & { distDir?: string } = {}) {
  const service = await createTrainingService(options);
  const app = express();
  const token = randomBytes(32).toString('hex');
  const nativeReplays = new Map<string, { child: ReturnType<typeof spawn>; listeners: Set<(value: unknown) => void>; last?: unknown }>();
  const nativeReplayScript = path.resolve(options.distDir ? path.join(options.distDir, '..', 'training/native_replay.py') : 'training/native_replay.py');
  const python = process.env.ZBOT_TRAINING_PYTHON ?? path.join(process.env.HOME ?? '', 'mjlab', '.venv', 'bin', 'python');
  app.disable('x-powered-by');
  app.use((req, res, next) => {
    if (!loopbackAddress(req.socket.remoteAddress)) return res.status(403).json({ error: '训练服务仅接受本机连接' });
    let host: URL;
    try { host = new URL(`http://${req.headers.host}`); } catch { return res.status(403).json({ error: '无效请求主机' }); }
    if (!req.headers.host || !loopbackHost(host.hostname) || host.username || host.password || host.pathname !== '/') return res.status(403).json({ error: '训练服务仅允许 localhost 或回环地址' });
    const origin = req.headers.origin;
    if (req.headers['sec-fetch-site'] === 'cross-site') return res.status(403).json({ error: '不允许跨站训练请求' });
    if (origin) {
      let url: URL;
      try { url = new URL(origin); } catch { return res.status(403).json({ error: '无效请求来源' }); }
      if (!loopbackHost(url.hostname) || !['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.pathname !== '/') return res.status(403).json({ error: '不允许远程网页访问训练服务' });
      res.setHeader('Access-Control-Allow-Origin', origin);
      res.setHeader('Vary', 'Origin');
    }
    res.setHeader('Cache-Control', 'no-store');
    if (req.method === 'OPTIONS') {
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Training-Token');
      return res.sendStatus(204);
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      const supplied = req.headers['x-training-token'];
      if (typeof supplied !== 'string' || Buffer.byteLength(supplied) !== token.length || !timingSafeEqual(Buffer.from(supplied), Buffer.from(token))) return res.status(403).json({ error: '训练会话令牌无效，请刷新连接' });
      if (!req.is('application/json')) return res.status(415).json({ error: '训练请求必须使用 application/json' });
    }
    next();
  });
  app.use(express.json({ limit: '64kb', strict: true }));
  const asyncRoute = (handler: (req: Request, res: Response) => Promise<unknown>) => (req: Request, res: Response, next: NextFunction) => { void handler(req, res).catch(next); };
  app.get('/api/training/session', (_req, res) => res.json({ token }));
  app.get('/api/training/resources', asyncRoute(async (_req, res) => res.json(await service.resources())));
  app.post('/api/training/replay', asyncRoute(async (req, res) => {
    const frequency = Number(req.body?.frequency ?? 0.4);
    const steps = Number(req.body?.steps ?? 600);
    const device = typeof req.body?.device === 'string' && /^cuda:\d+$/.test(req.body.device) ? req.body.device : 'cuda:0';
    if (!Number.isFinite(frequency) || frequency < 0.25 || frequency > 1 || !Number.isInteger(steps) || steps < 1 || steps > 10000) return res.status(400).json({ error: 'invalid native replay parameters' });
    const id = randomUUID();
    const child = spawn(python, [nativeReplayScript, '--frequency', String(frequency), '--steps', String(steps), '--device', device], { cwd: path.dirname(nativeReplayScript), env: { ...process.env, PYTHONUNBUFFERED: '1' }, stdio: ['ignore', 'pipe', 'pipe'] });
    const record: { child: ReturnType<typeof spawn>; listeners: Set<(value: unknown) => void>; last?: unknown } = { child, listeners: new Set<(value: unknown) => void>() };
    nativeReplays.set(id, record);
    const lineReader = createInterface({ input: child.stdout });
    lineReader.on('line', line => {
      try { const value = JSON.parse(line); record.last = value; for (const listener of record.listeners) listener(value); } catch {}
    });
    child.stderr.on('data', chunk => console.error('[MJWarp replay]', String(chunk).trim()));
    child.once('close', () => { for (const listener of record.listeners) listener({ kind: 'closed' }); setTimeout(() => nativeReplays.delete(id), 60000).unref(); });
    return res.status(202).json({ id, backend: 'mjwarp', frequency, device });
  }));
  app.post('/api/training/replay/:id/stop', (req, res) => {
    const record = nativeReplays.get(req.params.id);
    if (!record) return res.status(404).json({ error: 'native replay not found' });
    record.child.kill('SIGTERM'); return res.json({ stopped: true });
  });
  app.get('/api/training/replay/:id/live', (req, res) => {
    const record = nativeReplays.get(req.params.id);
    if (!record) return res.status(404).json({ error: 'native replay not found' });
    res.status(200).set({ 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', Connection: 'keep-alive', 'X-Accel-Buffering': 'no' });
    res.flushHeaders(); res.write(': connected\n\n');
    const listener = (value: unknown) => res.write('data: ' + JSON.stringify(value) + '\n\n');
    record.listeners.add(listener); if (record.last) listener(record.last);
    const heartbeat = setInterval(() => res.write(': keepalive\n\n'), 15000); heartbeat.unref();
    res.once('close', () => { clearInterval(heartbeat); record.listeners.delete(listener); });
  });
  app.get('/api/training/jobs', (_req, res) => res.json(service.list()));
  app.get('/api/training/jobs/:id', asyncRoute(async (req, res) => res.json(service.get(req.params.id))));
  app.post('/api/training/jobs', asyncRoute(async (req, res) => res.status(201).json(await service.start(req.body))));
  app.post('/api/training/jobs/:id/stop', asyncRoute(async (req, res) => res.json(await service.stop(req.params.id))));
  app.post('/api/training/jobs/:id/preview', asyncRoute(async (req, res) => res.json(await service.setPreview(req.params.id, req.body?.enabled))));
  app.post('/api/training/jobs/:id/task-card', asyncRoute(async (req, res) => res.json(await service.setTaskCard(req.params.id, req.body))));
  app.get('/api/training/jobs/:id/live', (req, res, next) => {
    try {
      res.status(200).set({ 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', Connection: 'keep-alive', 'X-Accel-Buffering': 'no' });
      res.flushHeaders();
      res.write(': connected\n\n');
      const unsubscribe = service.subscribeLive(req.params.id, frame => res.write(`data: ${JSON.stringify(frame)}\n\n`));
      const heartbeat = setInterval(() => res.write(': keepalive\n\n'), 15000); heartbeat.unref();
      res.once('close', () => {
        clearInterval(heartbeat);
        if (unsubscribe() === 0) void service.setPreview(req.params.id, false).catch(() => {});
      });
    } catch (error) { next(error); }
  });
  app.get('/api/training/jobs/:id/bundle', asyncRoute(async (req, res) => {
    const file = await service.bundle(req.params.id);
    res.type('application/json').sendFile(file);
  }));
  app.get('/api/training/jobs/:id/checkpoints/:name', asyncRoute(async (req, res) => {
    const file = await service.checkpointPath(req.params.id, req.params.name);
    res.download(file, path.basename(file));
  }));
  app.use('/api/training', (_req, res) => res.status(404).json({ error: '训练接口不存在' }));
  app.use(express.static(path.resolve(options.distDir ?? 'dist')));
  app.use((error: unknown, _req: Request, res: Response, _next: NextFunction) => {
    const status = error instanceof TrainingError ? error.status : (error as { status?: number })?.status ?? 500;
    res.status(status >= 400 && status <= 599 ? status : 500).json({ error: error instanceof Error ? error.message : '训练服务出错' });
  });
  return { app, service, close: () => service.close() };
}

async function main() {
  const port = Number(process.env.ZBOT_TRAINING_PORT ?? 8767);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('无效的 ZBOT_TRAINING_PORT');
  const instance = await createTrainingApp();
  const server = instance.app.listen(port, '127.0.0.1', () => {
    console.log(`ZBot 本机训练服务：http://127.0.0.1:${port}`);
  });
  let closing = false;
  const close = async () => {
    if (closing) return; closing = true;
    await instance.close(); server.close();
  };
  process.once('SIGINT', () => { void close(); });
  process.once('SIGTERM', () => { void close(); });
  server.once('error', error => { console.error(error); void instance.close(); process.exitCode = 1; });
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  void main().catch(error => { console.error(error); process.exitCode = 1; });
}
