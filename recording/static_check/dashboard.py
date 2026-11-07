"""Dashboard for reviewing static_videos.json.

Streams videos from the run directory. "Not static" removes a false positive from the JSON.
"""

import http.server
import json
import os
import urllib.parse

PORT = 8080
DATASET_DIR = os.environ.get("RUN_DIR", os.path.join(os.path.dirname(__file__), "..", "output", "run1"))
JSON_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "static_videos.json"))
PAGE_SIZE = 6

HTML = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Static Video Review</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #111; color: #eee; padding: 20px; }
    header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 20px; }
    h1 { font-size: 1rem; font-weight: 600; }
    #stats { font-size: 0.82rem; color: #888; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 12px;
    }
    .card {
      background: #1c1c1c;
      border: 1px solid #2e2e2e;
      border-radius: 8px;
      overflow: hidden;
      transition: opacity 0.3s;
    }
    .card.removed { opacity: 0.25; pointer-events: none; }
    video { width: 100%; display: block; background: #000; }
    .card-footer {
      padding: 8px 10px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .name {
      flex: 1;
      font-size: 0.7rem;
      color: #777;
      word-break: break-all;
    }
    .btn-remove {
      flex-shrink: 0;
      background: #8b1a1a;
      color: #eee;
      border: none;
      border-radius: 4px;
      padding: 4px 10px;
      font-size: 0.75rem;
      cursor: pointer;
      white-space: nowrap;
    }
    .btn-remove:hover:not(:disabled) { background: #c0392b; }
    .btn-remove:disabled { background: #333; color: #666; cursor: default; }
    .pagination {
      margin-top: 20px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .pagination button {
      background: #252525;
      color: #ccc;
      border: 1px solid #333;
      border-radius: 4px;
      padding: 5px 14px;
      font-size: 0.82rem;
      cursor: pointer;
    }
    .pagination button:hover:not(:disabled) { background: #333; }
    .pagination button:disabled { opacity: 0.35; cursor: default; }
    #page-info { font-size: 0.82rem; color: #666; }
  </style>
</head>
<body>
  <header>
    <h1>Static Video Review</h1>
    <div id="stats"></div>
  </header>
  <div class="grid" id="grid"></div>
  <div class="pagination">
    <button id="btn-prev" onclick="changePage(-1)">&#8592; Prev</button>
    <span id="page-info"></span>
    <button id="btn-next" onclick="changePage(1)">Next &#8594;</button>
  </div>

  <script>
    let page = 0;
    let total = 0;

    async function load() {
      const res = await fetch(`/api/videos?page=${page}`);
      const data = await res.json();
      total = data.total;
      render(data.videos, data.page, data.total, data.total_pages);
    }

    function render(videos, pg, tot, totalPages) {
      document.getElementById('stats').textContent =
        `${tot} video${tot !== 1 ? 's' : ''} flagged as static`;
      document.getElementById('page-info').textContent =
        `Page ${pg + 1} of ${totalPages || 1}`;
      document.getElementById('btn-prev').disabled = pg === 0;
      document.getElementById('btn-next').disabled = pg >= totalPages - 1;

      const grid = document.getElementById('grid');
      grid.innerHTML = '';
      for (const name of videos) {
        const card = document.createElement('div');
        card.className = 'card';
        card.dataset.name = name;
        card.innerHTML = `
          <video src="/video/${encodeURIComponent(name)}"
                 controls loop muted autoplay preload="auto"></video>
          <div class="card-footer">
            <span class="name">${name}</span>
            <button class="btn-remove">Not static</button>
          </div>`;
        card.querySelector('.btn-remove').addEventListener('click', () => remove(card, name));
        grid.appendChild(card);
      }
    }

    async function remove(card, name) {
      const btn = card.querySelector('.btn-remove');
      btn.disabled = true;
      const res = await fetch('/api/remove', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      if (res.ok) {
        card.classList.add('removed');
        total--;
        setTimeout(() => {
          // clamp page in case we removed the last item on this page
          const maxPage = Math.max(0, Math.ceil(total / """ + str(PAGE_SIZE) + r""") - 1);
          if (page > maxPage) page = maxPage;
          load();
        }, 500);
      } else {
        btn.disabled = false;
      }
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


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence per-request logs

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/":
            self._send(200, "text/html", HTML.encode())

        elif path == "/api/videos":
            with open(JSON_FILE) as f:
                videos = json.load(f)
            pg = int(query.get("page", ["0"])[0])
            total_pages = max(1, -(-len(videos) // PAGE_SIZE))  # ceil division
            start = pg * PAGE_SIZE
            payload = json.dumps({
                "videos": videos[start : start + PAGE_SIZE],
                "page": pg,
                "total": len(videos),
                "total_pages": total_pages,
            })
            self._send(200, "application/json", payload.encode())

        elif path.startswith("/video/"):
            name = urllib.parse.unquote(path[len("/video/"):])
            self._serve_video(name)

        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/remove":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            name = body.get("name", "")
            with open(JSON_FILE) as f:
                videos = json.load(f)
            if name in videos:
                videos.remove(name)
                with open(JSON_FILE, "w") as f:
                    json.dump(videos, f, indent=2)
                    f.write("\n")
            self._send(200, "application/json", b'{"ok":true}')
        else:
            self.send_error(404)

    def _send(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_video(self, name):
        video_path = os.path.join(DATASET_DIR, name, "animation.mp4")
        if not os.path.exists(video_path):
            self.send_error(404)
            return

        file_size = os.path.getsize(video_path)
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
            with open(video_path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", file_size)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(video_path, "rb") as f:
                self.wfile.write(f.read())


if __name__ == "__main__":
    with open(JSON_FILE) as f:
        count = len(json.load(f))
    print(f"Serving {count} videos from {JSON_FILE}")
    print(f"Dashboard → http://localhost:{PORT}")
    http.server.HTTPServer(("", PORT), Handler).serve_forever()
