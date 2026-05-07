#!/usr/bin/env node
// Visualizer server — no external dependencies (pure Node.js built-ins)
const http = require('http');
const path = require('path');
const fs   = require('fs');
const url  = require('url');

const PORT          = 8765;
const OUTPUT_DIR    = path.resolve(__dirname, '../output');
const EVAL_DIR      = path.resolve(__dirname, '../evaluation/output');
const GT_VIDEOS_DIR = path.resolve(__dirname, '../../../_hf_staging/data/test/videos');
const STATIC_DIR    = __dirname;

// ── MIME types ───────────────────────────────────────────────────────────────
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.mp4':  'video/mp4',
  '.json': 'application/json; charset=utf-8',
  '.txt':  'text/plain; charset=utf-8',
};

function mime(filePath) {
  return MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function send(res, statusCode, contentType, body) {
  const buf = typeof body === 'string' ? Buffer.from(body, 'utf8') : body;
  res.writeHead(statusCode, {
    'Content-Type': contentType,
    'Content-Length': buf.length,
    'Cache-Control': 'no-cache',
  });
  res.end(buf);
}

function json(res, data, status = 200) {
  send(res, status, 'application/json; charset=utf-8', JSON.stringify(data));
}

function notFound(res, msg = 'Not found') {
  json(res, { error: msg }, 404);
}

function err500(res, msg) {
  json(res, { error: msg }, 500);
}

function isDir(p) {
  try { return fs.statSync(p).isDirectory(); } catch { return false; }
}

function isFile(p) {
  try { return fs.statSync(p).isFile(); } catch { return false; }
}

// ── Video streaming with Range support ───────────────────────────────────────
function streamVideo(res, filePath) {
  if (!isFile(filePath)) return notFound(res, 'Video not found');
  // (Range requests handled by streaming; browsers will retry with range)
  const stat = fs.statSync(filePath);
  res.writeHead(200, {
    'Content-Type': 'video/mp4',
    'Content-Length': stat.size,
    'Accept-Ranges': 'bytes',
    'Cache-Control': 'no-cache',
  });
  fs.createReadStream(filePath).pipe(res);
}

// ── Route table ─────────────────────────────────────────────────────────────
const server = http.createServer((req, res) => {
  const parsed = url.parse(req.url, true);
  const pathname = parsed.pathname;
  const q = parsed.query;

  // ── API routes ─────────────────────────────────────────────────────────────

  // GET /api/folders
  if (pathname === '/api/folders') {
    try {
      const folders = fs.readdirSync(OUTPUT_DIR)
        .filter(f => isDir(path.join(OUTPUT_DIR, f)))
        .sort().reverse();
      return json(res, folders);
    } catch (e) { return err500(res, e.message); }
  }

  // GET /api/examples?folder=&page=&per_page=&search=
  if (pathname === '/api/examples') {
    const { folder, page = '1', per_page = '10', search = '' } = q;
    if (!folder) return json(res, { error: 'folder required' }, 400);
    try {
      const dir = path.join(OUTPUT_DIR, folder);
      let examples = fs.readdirSync(dir)
        .filter(f => isDir(path.join(dir, f)))
        .sort();
      if (search) {
        const lo = search.toLowerCase();
        examples = examples.filter(e => e.toLowerCase().includes(lo));
      }
      const total   = examples.length;
      const perPage = Math.max(1, parseInt(per_page, 10) || 10);
      const pageNum = Math.max(1, parseInt(page, 10) || 1);
      const start   = (pageNum - 1) * perPage;
      return json(res, { examples: examples.slice(start, start + perPage), total, page: pageNum, per_page: perPage });
    } catch (e) { return err500(res, e.message); }
  }

  // GET /api/trace?folder=&example=
  if (pathname === '/api/trace') {
    const { folder, example } = q;
    if (!folder || !example) return json(res, { error: 'folder and example required' }, 400);
    const tracePath = path.join(OUTPUT_DIR, folder, example, 'trace.json');
    if (!isFile(tracePath)) return notFound(res, 'trace.json not found');
    try {
      const content = fs.readFileSync(tracePath, 'utf8');
      return send(res, 200, 'application/json; charset=utf-8', content);
    } catch (e) { return err500(res, e.message); }
  }

  // GET /api/iters?folder=&example=
  if (pathname === '/api/iters') {
    const { folder, example } = q;
    if (!folder || !example) return json(res, { error: 'folder and example required' }, 400);
    try {
      const dir = path.join(OUTPUT_DIR, folder, example);
      const iters = fs.readdirSync(dir)
        .filter(f => /^iter_\d+$/.test(f) && isDir(path.join(dir, f)))
        .sort((a, b) => parseInt(a.split('_')[1]) - parseInt(b.split('_')[1]));
      return json(res, iters);
    } catch (e) { return err500(res, e.message); }
  }

  // GET /api/metrics?folder=&example=
  if (pathname === '/api/metrics') {
    const { folder, example } = q;
    if (!folder || !example) return json(res, { error: 'folder and example required' }, 400);
    const resultsPath = path.join(EVAL_DIR, folder, 'results.json');
    if (!isFile(resultsPath)) return json(res, null);
    try {
      const data = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
      const exData = (data.examples || {})[example] || null;
      return json(res, exData);
    } catch (e) { return err500(res, e.message); }
  }

  // GET /api/file?folder=&example=&iter=&filename=
  if (pathname === '/api/file') {
    const { folder, example, iter, filename } = q;
    if (!folder || !example || !iter || !filename)
      return json(res, { error: 'folder, example, iter, filename required' }, 400);
    const allowed = ['full.html', 'raw_output.txt', 'feedback.txt', 'raw_critic.txt', 'raw_editor.txt'];
    if (!allowed.includes(filename)) return json(res, { error: 'file not allowed' }, 403);
    const filePath = path.join(OUTPUT_DIR, folder, example, iter, filename);
    if (!isFile(filePath)) return notFound(res, 'File not found');
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      return send(res, 200, 'text/plain; charset=utf-8', content);
    } catch (e) { return err500(res, e.message); }
  }

  // ── Video routes ───────────────────────────────────────────────────────────

  // GET /video/gt/<example>.mp4
  const gtMatch = pathname.match(/^\/video\/gt\/(.+)\.mp4$/);
  if (gtMatch) {
    const example = decodeURIComponent(gtMatch[1]);
    // Prevent path traversal
    if (example.includes('..') || example.includes('/')) return notFound(res);
    return streamVideo(res, path.join(GT_VIDEOS_DIR, `${example}.mp4`));
  }

  // GET /video/output/<folder>/<example>/<iter>/animation.mp4
  const outMatch = pathname.match(/^\/video\/output\/([^/]+)\/([^/]+)\/(iter_\d+)\/animation\.mp4$/);
  if (outMatch) {
    const [, folder, example, iter] = outMatch.map(decodeURIComponent);
    if ([folder, example, iter].some(s => s.includes('..'))) return notFound(res);
    return streamVideo(res, path.join(OUTPUT_DIR, folder, example, iter, 'animation.mp4'));
  }

  // ── Static files ───────────────────────────────────────────────────────────
  let filePath = path.join(STATIC_DIR, pathname === '/' ? 'index.html' : pathname);
  // Prevent path traversal outside STATIC_DIR
  if (!filePath.startsWith(STATIC_DIR)) return notFound(res);
  if (isFile(filePath)) {
    try {
      const stat = fs.statSync(filePath);
      res.writeHead(200, { 'Content-Type': mime(filePath), 'Content-Length': stat.size });
      fs.createReadStream(filePath).pipe(res);
      return;
    } catch (e) { return err500(res, e.message); }
  }

  return notFound(res, `${pathname} not found`);
});

server.listen(PORT, () => {
  console.log(`Visualizer → http://localhost:${PORT}`);
});
