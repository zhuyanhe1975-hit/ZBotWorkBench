import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import {defineConfig, Plugin} from 'vite';

// Vite can serve the read-only workbench on the LAN; its training proxy must not
// turn a loopback-only process-control API into a network-accessible endpoint.
function localTrainingProxyGuard(): Plugin {
  return {
    name: 'local-training-proxy-guard',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url?.startsWith('/api/training') && !['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(req.socket.remoteAddress ?? '')) {
          res.statusCode = 403;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ error: '训练接口仅允许本机访问' }));
          return;
        }
        next();
      });
    },
  };
}

function rlBrowserDebugLog(): Plugin {
  const logPath = '/tmp/zbot-workbench-rl-browser.log';
  return {
    name: 'rl-browser-debug-log',
    configureServer(server) {
      server.middlewares.use('/api/rl-debug-log', (req, res, next) => {
        if (req.method === 'GET') {
          res.statusCode = 200;
          res.setHeader('Content-Type', 'text/plain; charset=utf-8');
          res.setHeader('Cache-Control', 'no-store');
          res.end(fs.existsSync(logPath) ? fs.readFileSync(logPath, 'utf8').slice(-200_000) : '');
          return;
        }
        if (req.method !== 'POST') { next(); return; }
        let body = '';
        req.setEncoding('utf8');
        req.on('data', chunk => {
          body += chunk;
          if (body.length > 16_384) req.destroy();
        });
        req.on('end', () => {
          try {
            const value = JSON.parse(body);
            if (!value || typeof value.event !== 'string' || value.event.length > 120) throw new Error('invalid debug record');
            fs.appendFileSync(logPath, `${new Date().toISOString()} ${JSON.stringify(value)}\n`);
            res.statusCode = 204;
            res.end();
          } catch {
            res.statusCode = 400;
            res.end('invalid debug record');
          }
        });
      });
    },
  };
}

// PhysX model JSON used to be fetched after the checkpoint and display XML.
// Some LAN browsers leave that standalone public-file request pending forever.
// Bundle the immutable catalog into the application so replay loading has no
// separate physical-model request and no browser decompression dependency.
function bundledPhysxModels(): Plugin {
  const publicId = 'virtual:zbot-physx-models';
  const resolvedId = `\0${publicId}`;
  return {
    name: 'bundled-zbot-physx-models',
    resolveId(id) { return id === publicId ? resolvedId : undefined; },
    load(id) {
      if (id !== resolvedId) return undefined;
      const directory = path.resolve(__dirname, 'public', 'rl', 'physx');
      const models = Object.fromEntries(fs.readdirSync(directory).filter(name => name.endsWith('.json')).sort().map(name => {
        const key = name.slice(0, -'.json'.length);
        return [key, JSON.parse(fs.readFileSync(path.join(directory, name), 'utf8'))];
      }));
      return `export default ${JSON.stringify(models)};`;
    },
  };
}

// LINT.IfChange(aistudio_media_plugin)
function aistudioMediaPlugin(): Plugin {
  return {
    name: 'vite-plugin-aistudio-media',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url && req.url.startsWith('/assets/aistudio/')) {
          const rawPath = req.url.split('?')[0].split('#')[0];
          try {
            const decodedPath = decodeURIComponent(rawPath);
            const relativePath = decodedPath.replace(/^\//, '');
            const aistudioDir = path.resolve(
              __dirname,
              'public',
              'assets',
              'aistudio',
            );
            const filePath = path.resolve(__dirname, 'public', relativePath);
            if (
              filePath.startsWith(aistudioDir + path.sep) &&
              fs.existsSync(filePath) &&
              fs.statSync(filePath).isFile()
            ) {
              const ext = path.extname(filePath).toLowerCase();
              const mimeMap: Record<string, string> = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.svg': 'image/svg+xml',
                '.bmp': 'image/bmp',
                '.ico': 'image/x-icon',
                '.mp4': 'video/mp4',
                '.webm': 'video/webm',
                '.ogv': 'video/ogg',
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.ogg': 'audio/ogg',
                '.pdf': 'application/pdf',
              };
              res.setHeader(
                'Content-Type',
                mimeMap[ext] || 'application/octet-stream',
              );
              res.setHeader('Cache-Control', 'no-cache');
              fs.createReadStream(filePath).pipe(res);
              return;
            }
          } catch {
            // Fall through if URI decoding or file access fails
          }
        }
        next();
      });
    },
  };
}
// LINT.ThenChange(//depot/google3/java/com/google/alkali/boq/makersuite/applet_dev_service/templates/initializers/react_theme/vite.config.ts:aistudio_media_plugin)

export default defineConfig(() => {
  return {
    plugins: [localTrainingProxyGuard(), rlBrowserDebugLog(), bundledPhysxModels(), react(), tailwindcss(), aistudioMediaPlugin()],
    resolve: {
      dedupe: ['react', 'react-dom'],
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    optimizeDeps: {
      include: ['react', 'react-dom', 'react-dom/client', 'react/jsx-runtime'],
    },
    server: {
      proxy: {
        '/api/training': { target: process.env.ZBOT_TRAINING_URL || 'http://127.0.0.1:8767', changeOrigin: true },
      },
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâfile watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : { ignored: ['**/.omx/**', '**/tests/**', '**/docs/**'] },
    },
  };
});
