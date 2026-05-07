"""
Side-by-side determinism dashboard.
Reads comparison.csv and serves run1 vs run2 videos.
Filter by identical column; defaults to false (different videos only).

Run:
    uv run python data_processing/video_processing/v2/determinism_check/dashboard.py
Then open http://localhost:5500
"""

import csv
import http.server
import json
import os
import urllib.parse

PORT = 5500
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "determinism_comparison.csv")
PAGE_SIZE = 10

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Determinism Dashboard</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #111; color: #eee; padding: 20px; }
    header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
    h1 { font-size: 1rem; font-weight: 600; }
    #stats { font-size: 0.82rem; color: #888; }
    .counts { display: flex; gap: 12px; font-size: 0.82rem; }
    .count-diff  { color: #f87171; }
    .count-same  { color: #4ade80; }
    .toolbar { display: flex; gap: 8px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }
    .toolbar span { font-size: 0.82rem; color: #666; }
    .filter-btn {
      padding: 4px 12px; border-radius: 5px; font-size: 0.82rem; cursor: pointer;
      background: #1e1e1e; color: #aaa; border: 1px solid #333; text-decoration: none;
    }
    .filter-btn.active { background: #2563eb; color: #fff; border-color: #2563eb; }
    .filter-btn:hover:not(.active) { background: #2a2a2a; }
    .cards { display: flex; flex-direction: column; gap: 14px; }
    .card { background: #1c1c1c; border: 1px solid #2e2e2e; border-radius: 8px; overflow: hidden; }
    .card-header {
      display: flex; align-items: center; gap: 10px;
      padding: 8px 12px; background: #161616; border-bottom: 1px solid #2e2e2e;
    }
    .example-name { font-size: 0.78rem; font-family: monospace; color: #818cf8; flex: 1; }
    .badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 999px; font-weight: 600; }
    .badge-true  { background: #14532d; color: #4ade80; }
    .badge-false { background: #450a0a; color: #f87171; }
    .metrics { display: flex; gap: 12px; font-size: 0.72rem; color: #888; }
    .metric { display: flex; gap: 4px; }
    .metric-label { color: #555; }
    .metric-value { color: #cbd5e1; font-family: monospace; }
    .replay-btn {
      padding: 3px 10px; border-radius: 5px; font-size: 0.75rem; cursor: pointer;
      background: #1e293b; color: #94a3b8; border: 1px solid #334155;
    }
    .replay-btn:hover { background: #273549; color: #cbd5e1; }
    .videos { display: grid; grid-template-columns: 1fr 1fr; }
    .video-col { padding: 8px; }
    .label { font-size: 0.7rem; color: #555; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
    video { width: 100%; max-width: 320px; display: block; background: #000; border-radius: 5px; }
    .pagination { margin-top: 20px; display: flex; align-items: center; gap: 10px; }
    .pagination button {
      background: #1e1e1e; color: #ccc; border: 1px solid #333;
      border-radius: 5px; padding: 5px 14px; font-size: 0.82rem; cursor: pointer;
    }
    .pagination button:hover:not(:disabled) { background: #2a2a2a; }
    .pagination button:disabled { opacity: 0.35; cursor: default; }
    #page-info { font-size: 0.82rem; color: #666; }
    .empty { color: #555; text-align: center; padding: 40px; font-size: 0.9rem; }
  </style>
</head>
<body>
  <header>
    <h1>Determinism Dashboard</h1>
    <div id="stats"></div>
    <div class="counts">
      <span class="count-diff" id="count-diff"></span>
      <span class="count-same" id="count-same"></span>
    </div>
  </header>
  <div class="toolbar">
    <span>Filter:</span>
    <button class="filter-btn" data-val="false" onclick="setFilter('false')">Different</button>
    <button class="filter-btn" data-val="true"  onclick="setFilter('true')">Identical</button>
    <button class="filter-btn" data-val="all"   onclick="setFilter('all')">All</button>
  </div>
  <div class="cards" id="cards"></div>
  <div class="pagination">
    <button id="btn-prev" onclick="changePage(-1)">&#8592; Prev</button>
    <span id="page-info"></span>
    <button id="btn-next" onclick="changePage(1)">Next &#8594;</button>
  </div>

  <script>
    let page = 0;
    let filter = 'false';

    function setFilter(val) {
      filter = val;
      page = 0;
      load();
    }

    async function load() {
      document.querySelectorAll('.filter-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.val === filter);
      });
      const res = await fetch(`/api/rows?page=${page}&identical=${filter}`);
      const data = await res.json();
      render(data);
    }

    function render({ rows, page: pg, total, total_pages, total_different, total_identical }) {
      document.getElementById('stats').textContent =
        `${total} shown`;
      document.getElementById('count-diff').textContent =
        `${total_different} different`;
      document.getElementById('count-same').textContent =
        `${total_identical} identical`;
      document.getElementById('page-info').textContent =
        `Page ${pg + 1} of ${total_pages || 1}`;
      document.getElementById('btn-prev').disabled = pg === 0;
      document.getElementById('btn-next').disabled = pg >= total_pages - 1;

      const container = document.getElementById('cards');
      container.innerHTML = '';

      if (!rows.length) {
        container.innerHTML = '<div class="empty">No results.</div>';
        return;
      }

      for (const row of rows) {
        const card = document.createElement('div');
        card.className = 'card';
        const badgeClass = row.identical === 'true' ? 'badge-true' : 'badge-false';
        const badgeText  = row.identical === 'true' ? 'identical' : 'different';
        const v1src = `/video?path=${encodeURIComponent(row.run1)}`;
        const v2src = `/video?path=${encodeURIComponent(row.run2)}`;
        const psnr = row.psnr ? parseFloat(row.psnr).toFixed(1) + ' dB' : '—';
        const maxDiff = row.max_pixel_diff || '—';
        card.innerHTML = `
          <div class="card-header">
            <span class="example-name">${row.example}</span>
            <span class="badge ${badgeClass}">${badgeText}</span>
            <div class="metrics">
              <div class="metric"><span class="metric-label">PSNR</span><span class="metric-value">${psnr}</span></div>
              <div class="metric"><span class="metric-label">max px diff</span><span class="metric-value">${maxDiff}</span></div>
            </div>
            <button class="replay-btn" onclick="replayPair(this)">&#8635; Replay</button>
          </div>
          <div class="videos">
            <div class="video-col">
              <div class="label">Run 1</div>
              <video autoplay loop muted playsinline src="${v1src}"></video>
            </div>
            <div class="video-col">
              <div class="label">Run 2</div>
              <video autoplay loop muted playsinline src="${v2src}"></video>
            </div>
          </div>`;
        container.appendChild(card);
      }
    }

    function replayPair(btn) {
      const videos = btn.closest('.card').querySelectorAll('video');
      videos.forEach(v => { v.currentTime = 0; v.play(); });
    }

    function changePage(delta) {
      page = Math.max(0, page + delta);
      window.scrollTo(0, 0);
      load();
    }

    load();
  </script>
</body>
</html>
"""


def load_csv():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


ALL_ROWS = load_csv()


def get_page(identical_filter, page):
    if identical_filter == "all":
        rows = ALL_ROWS
    else:
        rows = [r for r in ALL_ROWS if r["identical"].lower() == identical_filter]
    total = len(rows)
    total_pages = max(1, -(-total // PAGE_SIZE))
    page = max(0, min(page, total_pages - 1))
    total_different = sum(1 for r in ALL_ROWS if r["identical"].lower() == "false")
    total_identical = len(ALL_ROWS) - total_different
    return {
        "rows": rows[page * PAGE_SIZE: (page + 1) * PAGE_SIZE],
        "page": page,
        "total": total,
        "total_pages": total_pages,
        "total_different": total_different,
        "total_identical": total_identical,
    }


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/":
            self._send(200, "text/html", HTML.encode())

        elif path == "/api/rows":
            identical = query.get("identical", ["false"])[0].lower()
            if identical not in ("true", "false", "all"):
                identical = "false"
            page = int(query.get("page", ["0"])[0])
            data = get_page(identical, page)
            self._send(200, "application/json", json.dumps(data).encode())

        elif path == "/video":
            rel = query.get("path", [""])[0]
            abs_path = os.path.normpath(os.path.join(SCRIPT_DIR, rel))
            # Only allow paths inside the v2 output directory
            allowed = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "output"))
            if not abs_path.startswith(allowed):
                self.send_error(403)
                return
            self._serve_file(abs_path)

        else:
            self.send_error(404)

    def _send(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, file_path):
        if not os.path.isfile(file_path):
            self.send_error(404)
            return

        file_size = os.path.getsize(file_path)
        range_header = self.headers.get("Range", "")

        if range_header.startswith("bytes="):
            parts = range_header[6:].split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", length)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(file_path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", file_size)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())


if __name__ == "__main__":
    total = len(ALL_ROWS)
    different = sum(1 for r in ALL_ROWS if r["identical"].lower() == "false")
    print(f"Loaded {total} rows ({different} different, {total - different} identical)")
    print(f"Dashboard → http://localhost:{PORT}")
    http.server.HTTPServer(("", PORT), Handler).serve_forever()
