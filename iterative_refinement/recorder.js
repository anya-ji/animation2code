#!/usr/bin/env node
/**
 * Record a single HTML file to MP4.
 *
 * Adapted from data_processing/video_processing/v3/record.js for use with
 * arbitrary HTML file paths (not tied to the dataset directory structure).
 *
 * Usage:
 *   node recorder.js --html /abs/path/to/full.html --output /abs/path/out.mp4
 *                    [--fps 12] [--duration 10] [--width 1024] [--height 768]
 *
 * node_modules must be available — run.py symlinks v3's node_modules here on first use.
 */
import { capture, launch, PuppeteerCaptureFormat } from "puppeteer-capture";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ─── CSS helpers (same logic as v3/record.js) ─────────────────────────────

function bodyIsAnimated(html) {
  return /body\s*\{[^}]*animation[\w-]*\s*:/i.test(html);
}

function bodyHasMarginReset(html) {
  return (
    /body\s*\{[^}]*\bmargin\s*:/i.test(html) ||
    /\*\s*\{[^}]*\bmargin\s*:/i.test(html)
  );
}

function buildFixCss(html) {
  const parts = ["html { overflow: hidden !important; }"];
  if (bodyIsAnimated(html)) {
    parts.push("body { overflow: hidden !important; }");
  } else {
    parts.push("body { overflow: hidden !important; min-height: 100vh !important; }");
  }
  if (!bodyHasMarginReset(html)) {
    parts.push("body { margin: 0 !important; }");
  }
  return parts.join(" ");
}

// ─── Find chrome-headless-shell ───────────────────────────────────────────

const CHROME_HEADLESS_SHELL = (() => {
  const base = path.join(
    process.env.HOME || "/home/anyaji",
    ".cache/puppeteer/chrome-headless-shell"
  );
  if (!fs.existsSync(base)) return undefined;
  const builds = fs.readdirSync(base).sort().reverse();
  for (const build of builds) {
    const p = path.join(base, build, "chrome-headless-shell-linux64", "chrome-headless-shell");
    if (fs.existsSync(p)) return p;
  }
  return undefined;
})();

// ─── CLI ──────────────────────────────────────────────────────────────────

function parseArgs() {
  const argv = process.argv.slice(2);
  const opts = {
    htmlPath: null,
    outputPath: null,
    fps: 12,
    duration: 10,
    width: 1024,
    height: 768,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if      (a === "--html"     && argv[i+1]) opts.htmlPath   = argv[++i];
    else if (a === "--output"   && argv[i+1]) opts.outputPath = argv[++i];
    else if (a === "--fps"      && argv[i+1]) opts.fps        = parseInt(argv[++i]) || opts.fps;
    else if (a === "--duration" && argv[i+1]) opts.duration   = parseFloat(argv[++i]) || opts.duration;
    else if (a === "--width"    && argv[i+1]) opts.width      = parseInt(argv[++i]) || opts.width;
    else if (a === "--height"   && argv[i+1]) opts.height     = parseInt(argv[++i]) || opts.height;
  }
  return opts;
}

// ─── Recording ────────────────────────────────────────────────────────────

async function recordHtml(opts) {
  const { htmlPath, outputPath, fps, duration, width, height } = opts;

  if (!htmlPath || !outputPath) {
    console.error("ERROR: --html and --output are required");
    process.exit(1);
  }
  if (!fs.existsSync(htmlPath)) {
    console.error(`ERROR: HTML file not found: ${htmlPath}`);
    process.exit(1);
  }

  const html = fs.readFileSync(htmlPath, "utf-8");
  const pageCss = buildFixCss(html);
  const tmpPath = outputPath + ".tmp.mp4";

  const launchOpts = {
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
    ...(CHROME_HEADLESS_SHELL ? { executablePath: CHROME_HEADLESS_SHELL } : {}),
  };
  const browser = await launch(launchOpts);
  let recorder = null;

  try {
    const page = await browser.newPage();
    await page.setViewport({ width, height, deviceScaleFactor: 1 });

    // Inject fix CSS before any page scripts run
    await page.evaluateOnNewDocument(`
      (() => {
        const css = ${JSON.stringify(pageCss)};
        function inject() {
          if (!document.head) return false;
          const s = document.createElement('style');
          s.textContent = css;
          document.head.insertBefore(s, document.head.firstChild);
          return true;
        }
        if (!inject()) document.addEventListener('DOMContentLoaded', inject, { once: true });
      })();
    `);

    recorder = await capture(page, {
      fps,
      size: `${width}x${height}`,
      format: PuppeteerCaptureFormat.MP4("fast", "libx264"),
    });

    await page.goto(`file://${htmlPath}`, { waitUntil: "networkidle0", timeout: 30000 });

    // Reinforce CSS after load
    await page.addStyleTag({ content: pageCss });

    // Pause all WAAPI animations at t=0 before recording starts
    await page.evaluate(() => {
      for (const anim of document.getAnimations({ subtree: true })) {
        try { anim.pause(); anim.currentTime = 0; } catch (_) {}
      }
    });

    await recorder.start(tmpPath);

    // Sync WAAPI animation time to the virtual rAF clock from puppeteer-capture
    await page.evaluate(() => {
      let startTs = null;
      let prevElapsed = 0;
      function syncFrame(ts) {
        if (startTs === null) startTs = ts;
        const elapsed = ts - startTs;
        const delta = elapsed - prevElapsed;
        prevElapsed = elapsed;
        if (window.__advanceTime) window.__advanceTime(delta);
        for (const anim of document.getAnimations({ subtree: true })) {
          try { anim.currentTime = elapsed; } catch (_) {}
        }
        requestAnimationFrame(syncFrame);
      }
      requestAnimationFrame(syncFrame);
    });

    await recorder.waitForTimeout(duration * 1000);
    await recorder.stop();
    await recorder.detach();
    recorder = null;

    fs.renameSync(tmpPath, outputPath);
    console.log(`OK recorded ${duration}s @ ${fps}fps → ${outputPath}`);
  } finally {
    if (recorder) {
      try { await recorder.stop(); } catch {}
      try { await recorder.detach(); } catch {}
    }
    await browser.close().catch(() => {});
    if (fs.existsSync(tmpPath)) { try { fs.unlinkSync(tmpPath); } catch {} }
  }
}

// ─── Main ─────────────────────────────────────────────────────────────────

const opts = parseArgs();
recordHtml(opts).catch((err) => {
  console.error(`FATAL: ${err.stack || err.message}`);
  process.exit(1);
});
