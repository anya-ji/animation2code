import { capture, launch, PuppeteerCaptureFormat } from "puppeteer-capture";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { execSync } from "child_process";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "../../..");
const DATASET_DIR = path.join(REPO_ROOT, "dataset");
const LOG_DIR = path.join(REPO_ROOT, "log");

const CONFIG = {
  width: 1024,
  height: 768,
  fps: 30,
  maxDuration: 8,
  defaultDurationCSS: 8,
  defaultDurationJS: 8,
  defaultDurationUnknown: 8,
  minOutputS: 2,
};

const JS_PATTERNS = [
  /gsap/i, /TweenMax/i, /TweenLite/i, /TimelineMax/i, /TimelineLite/i,
  /GreenSock/i, /bodymovin/i, /lottie/i, /anime\(/i, /velocity/i,
  /popmotion/i, /framer-motion/i, /mo\.js/i,
  /requestAnimationFrame\s*\(/, /\.getContext\s*\(/,
  /new\s+THREE\./i, /createjs/i, /pixi/i, /fabric/i, /paper/i,
  /p5\./i, /d3\.\w+transition/i, /Konva/i, /easeljs/i, /canvas/i,
  /setInterval\s*\(/, /setTimeout\s*\(/, /<animate/i,
];

// Returns true if the page's own CSS applies an animation directly to body.
// When true, we must NOT inject min-height: 100vh — it would change the body's
// aspect ratio and distort animations that use body itself as the visual shape
// (e.g. border-radius: 100% becoming elliptical instead of circular).
function bodyIsAnimated(html) {
  return /body\s*\{[^}]*animation[\w-]*\s*:/i.test(html);
}

// Returns true if the page already resets body/universal margin in its own CSS.
// When false, we inject margin:0 to prevent the default 8px body margin from
// creating white-border gutters around fullscreen canvas/flex animations.
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

// JS time injection for JS-type animations. Mocks Date/performance.now/
// setTimeout/setInterval/Math.random but NOT requestAnimationFrame —
// puppeteer-capture intercepts rAF at the browser level. Time advances via
// window.__advanceTime(ms), called from the rAF sync loop during recording.
const JS_TIME_INJECTION = `
(() => {
  let __t = 0;
  const __tos = new Map();
  const __ivs = new Map();
  let __tId = 1e6;

  let __seed = 123456789;
  Math.random = () => {
    __seed = (__seed + 0x6D2B79F5) | 0;
    let t = Math.imul(__seed ^ (__seed >>> 15), 1 | __seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  const __OrigDate = Date;
  window.Date = function (...args) {
    if (args.length === 0) return new __OrigDate(__t);
    return new __OrigDate(...args);
  };
  Object.assign(window.Date, __OrigDate);
  window.Date.now = () => __t;

  Object.defineProperty(performance, "now", {
    value: () => __t, writable: true, configurable: true,
  });

  window.setTimeout = (cb, d = 0, ...a) => {
    const fn = typeof cb === "function" ? cb : () => {};
    const id = ++__tId;
    __tos.set(id, { fn, at: __t + d, a });
    return id;
  };
  window.clearTimeout = (id) => __tos.delete(id);

  window.setInterval = (cb, iv = 0, ...a) => {
    const fn = typeof cb === "function" ? cb : () => {};
    const id = ++__tId;
    __ivs.set(id, { fn, iv: Math.max(iv, 1), next: __t + Math.max(iv, 1), a });
    return id;
  };
  window.clearInterval = (id) => __ivs.delete(id);

  window.__advanceTime = (ms) => {
    __t += ms;
    let maxIter = 100;
    while (maxIter-- > 0) {
      const expired = [];
      for (const [id, t] of __tos) { if (t.at <= __t) expired.push([id, t]); }
      if (expired.length === 0) break;
      for (const [id, t] of expired) {
        __tos.delete(id);
        try { t.fn(...t.a); } catch {}
      }
    }
    for (const [, iv] of __ivs) {
      let n = 10;
      while (iv.next <= __t && n-- > 0) {
        try { iv.fn(...iv.a); } catch {}
        iv.next += iv.iv;
      }
    }
  };
})();
`;

// Find puppeteer-bundled chrome-headless-shell (required by puppeteer-capture)
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

// ─── CLI ─────────────────────────────────────────────────────────────────

function parseArgs() {
  const argv = process.argv.slice(2);
  const opts = {
    target: null,
    rewrite: false,
    concurrency: 1,
    jsInject: false,
    output: "run1",
    htmlDir: null,
    start: 0,
    end: null,
    session: null,
    fps: null,
    outputName: "animation.mp4",
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--rewrite") opts.rewrite = true;
    else if (a === "--js-inject") opts.jsInject = true;
    else if (a === "--concurrency" && argv[i + 1]) opts.concurrency = Math.max(1, parseInt(argv[++i]) || 1);
    else if (a === "--output" && argv[i + 1]) opts.output = argv[++i];
    else if (a === "--output-name" && argv[i + 1]) opts.outputName = argv[++i];
    else if (a === "--html-dir" && argv[i + 1]) opts.htmlDir = path.resolve(argv[++i]);
    else if (a === "--start" && argv[i + 1]) opts.start = parseInt(argv[++i]) || 0;
    else if (a === "--end" && argv[i + 1]) opts.end = parseInt(argv[++i]);
    else if (a === "--session" && argv[i + 1]) opts.session = argv[++i];
    else if (a === "--fps" && argv[i + 1]) opts.fps = parseFloat(argv[++i]);
    else if (!a.startsWith("--")) opts.target = a;
  }
  return opts;
}

// ─── Animation type detection ─────────────────────────────────────────────

function detectAnimationType(html) {
  for (const pat of JS_PATTERNS) {
    if (pat.test(html)) return "js";
  }
  if (/@keyframes/i.test(html) || /animation[\s-]*:/i.test(html)) return "css";
  return "unknown";
}

function estimateDuration(html) {
  const type = detectAnimationType(html);
  if (type === "css") {
    const durations = [];
    const delays = [];
    let m;
    const durS = /animation(?:-duration)?:\s*[^;]*?(\d+\.?\d*)\s*s/gi;
    while ((m = durS.exec(html)) !== null) durations.push(parseFloat(m[1]));
    const durMs = /animation(?:-duration)?:\s*[^;]*?(\d+)\s*ms/gi;
    while ((m = durMs.exec(html)) !== null) durations.push(parseFloat(m[1]) / 1000);
    const delS = /animation-delay:\s*(\d+\.?\d*)\s*s/gi;
    while ((m = delS.exec(html)) !== null) delays.push(parseFloat(m[1]));
    if (durations.length > 0) {
      const hasAlternate = /animation(?:-direction)?:\s*[^;]*\balternate\b/i.test(html);
      const maxDur = Math.max(...durations) * (hasAlternate ? 2 : 1);
      const total = maxDur + (delays.length ? Math.max(...delays) : 0) /* + 1 */;
      return Math.min(total, CONFIG.maxDuration);
    }
    return CONFIG.defaultDurationCSS;
  }
  if (type === "js") return CONFIG.defaultDurationJS;
  return CONFIG.defaultDurationUnknown;
}

// ─── Loop short videos with ffmpeg ───────────────────────────────────────

function loopVideo(inputPath, outputPath, minDurationS) {
  execSync(
    `ffmpeg -y -stream_loop -1 -i "${inputPath}" -t ${minDurationS} ` +
    `-c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -movflags +faststart "${outputPath}"`,
    { stdio: "pipe" }
  );
}

// ─── Primary recording (puppeteer-capture) ────────────────────────────────

async function recordExample(browser, exampleName, sourceDir, outputDir, opts, staticLogStream) {
  const htmlPath = path.join(sourceDir, exampleName, "full.html");
  const html = fs.readFileSync(htmlPath, "utf-8");
  const animType = detectAnimationType(html);
  const duration = estimateDuration(html);

  if (animType === "unknown") {
    staticLogStream.write(`${new Date().toISOString()}\t${exampleName}\n`);
  }

  const exampleOutDir = path.join(outputDir, exampleName);
  fs.mkdirSync(exampleOutDir, { recursive: true });
  const finalPath = path.join(exampleOutDir, opts.outputName);
  const tmpPath = finalPath + ".tmp.mp4";

  const pageCss = buildFixCss(html);

  let recorder = null;
  const page = await browser.newPage();
  try {
    await page.setViewport({ width: CONFIG.width, height: CONFIG.height, deviceScaleFactor: 1 });

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

    // For JS-type animations: optionally freeze time during page load so
    // initialization is deterministic. rAF is NOT mocked — puppeteer-capture
    // owns it. Enable with --js-inject if puppeteer-capture alone is not enough.
    if (opts.jsInject && animType === "js") {
      await page.evaluateOnNewDocument(JS_TIME_INJECTION);
    }

    recorder = await capture(page, {
      fps: CONFIG.fps,
      size: `${CONFIG.width}x${CONFIG.height}`,
      format: PuppeteerCaptureFormat.MP4("fast", "libx264"),
    });

    await page.goto(`file://${htmlPath}`, { waitUntil: "networkidle0", timeout: 30000 });

    // Reinforce fix CSS after load (catches late-added style resets from page scripts)
    await page.addStyleTag({ content: pageCss });

    // Pause all WAAPI animations at t=0 before recording starts
    await page.evaluate(() => {
      for (const anim of document.getAnimations({ subtree: true })) {
        try { anim.pause(); anim.currentTime = 0; } catch (_) {}
      }
    });

    await recorder.start(tmpPath);

    // After start(), rAF is intercepted by puppeteer-capture's virtual clock.
    // This loop:
    //   1. Sets WAAPI animation.currentTime to virtual elapsed time
    //   2. Calls __advanceTime(delta) so setTimeout/setInterval mocks fire on schedule
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

    if (duration < CONFIG.minOutputS) {
      loopVideo(tmpPath, finalPath, CONFIG.minOutputS);
      fs.unlinkSync(tmpPath);
    } else {
      fs.renameSync(tmpPath, finalPath);
    }

    return { animType, duration, captureMethod: "puppeteer-capture" };
  } finally {
    if (recorder) {
      try { await recorder.stop(); } catch {}
      try { await recorder.detach(); } catch {}
    }
    await page.close().catch(() => {});
    if (fs.existsSync(tmpPath)) { try { fs.unlinkSync(tmpPath); } catch {} }
  }
}

// ─── Manifest ─────────────────────────────────────────────────────────────

function loadManifest(manifestPath) {
  if (!fs.existsSync(manifestPath)) return { results: [] };
  try { return JSON.parse(fs.readFileSync(manifestPath, "utf-8")); } catch { return { results: [] }; }
}

function saveManifest(manifestPath, results, opts, startTime) {
  const data = {
    config: {
      width: CONFIG.width,
      height: CONFIG.height,
      fps: CONFIG.fps,
      output: opts.output,
      flags: { jsInject: opts.jsInject },
      created: new Date().toISOString(),
      elapsed_ms: Date.now() - startTime,
    },
    results: [...results].sort((a, b) => a.folder.localeCompare(b.folder)),
  };
  fs.writeFileSync(manifestPath, JSON.stringify(data, null, 2));
}

// ─── Main ─────────────────────────────────────────────────────────────────

async function main() {
  const opts = parseArgs();
  if (opts.fps !== null) CONFIG.fps = opts.fps;
  const SOURCE_DIR = opts.htmlDir || DATASET_DIR;
  const OUTPUT_DIR = opts.htmlDir
    ? opts.htmlDir
    : path.join(__dirname, "output", opts.output);
  const MANIFEST_PATH = opts.htmlDir
    ? path.join(OUTPUT_DIR, "_record_manifest.json")
    : path.join(OUTPUT_DIR, "manifest.json");

  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  fs.mkdirSync(LOG_DIR, { recursive: true });

  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const logStream = fs.createWriteStream(path.join(LOG_DIR, `record-${ts}${opts.session !== null ? `-${opts.session}` : ""}.log`), { flags: "a" });
  const staticLogStream = fs.createWriteStream(path.join(LOG_DIR, `static-${ts}${opts.session !== null ? `-${opts.session}` : ""}.txt`), { flags: "a" });

  function log(msg) {
    const line = `[${new Date().toISOString()}] ${msg}`;
    process.stdout.write(line + "\n");
    logStream.write(line + "\n");
  }

  const examples = opts.target
    ? [opts.target]
    : fs.readdirSync(SOURCE_DIR)
        .filter((n) => {
          const p = path.join(SOURCE_DIR, n);
          return fs.statSync(p).isDirectory() && fs.existsSync(path.join(p, "full.html"));
        })
        .sort()
        .slice(opts.start, opts.end ?? undefined);

  const total = examples.length;

  // Resume: load existing successes from manifest
  const manifestData = loadManifest(MANIFEST_PATH);
  const succeededSet = new Set(
    (manifestData.results || [])
      .filter((r) => r.status === "success" &&
        fs.existsSync(path.join(OUTPUT_DIR, r.folder, opts.outputName)))
      .map((r) => r.folder)
  );
  const results = (manifestData.results || []).filter((r) => succeededSet.has(r.folder));

  log(`=== anim2code recorder v2 ===`);
  log(`Target:      ${opts.target || "all dataset"}`);
  log(`Examples:    ${total}`);
  log(`Output:      ${OUTPUT_DIR}`);
  log(`Concurrency: ${opts.concurrency}`);
  if (opts.rewrite)   log(`Rewrite:     yes`);
  if (opts.jsInject)  log(`JS-inject:   on`);
  log(`Log:         ${path.join(LOG_DIR, `record-${ts}.log`)}`);
  log(`Static log:  ${path.join(LOG_DIR, `static-${ts}.txt`)}`);
  if (CHROME_HEADLESS_SHELL) log(`Chrome:      ${CHROME_HEADLESS_SHELL}`);
  else log(`WARNING: chrome-headless-shell not found, using system default`);

  const startTime = Date.now();
  const launchOpts = CHROME_HEADLESS_SHELL ? { executablePath: CHROME_HEADLESS_SHELL } : {};

  let qi = 0;
  let doneCount = 0;
  const stats = { ok: 0, err: 0, skip: 0 };

  function postfix() {
    return `  [ok=${stats.ok} err=${stats.err} skip=${stats.skip}]`;
  }

  async function worker(workerId) {
    let browser = await launch(launchOpts);

    async function ensureBrowser() {
      try {
        if (!browser.isConnected()) throw new Error('disconnected');
        await browser.pages();
      } catch {
        log(`[W${workerId}] Browser crashed, relaunching...`);
        await browser.close().catch(() => {});
        browser = await launch(launchOpts);
      }
    }

    try {
      while (true) {
        const i = qi++;
        if (i >= examples.length) break;
        const example = examples[i];

        if (!opts.rewrite && (succeededSet.has(example) || fs.existsSync(path.join(OUTPUT_DIR, example, opts.outputName)))) {
          stats.skip++;
          log(`[${i + 1}/${total}] SKIP  ${example}${postfix()}`);
          doneCount++;
          continue;
        }

        await ensureBrowser();
        log(`[${i + 1}/${total}] START ${example}`);
        const t0 = Date.now();

        let timeoutId;
        try {
          const timeoutPromise = new Promise((_, reject) => {
            timeoutId = setTimeout(() => {
              browser.close().catch(() => {});
              reject(new Error("example timed out after 600s"));
            }, 600_000);
          });
          const { animType, duration, captureMethod } = await Promise.race([
            recordExample(browser, example, SOURCE_DIR, OUTPUT_DIR, opts, staticLogStream),
            timeoutPromise,
          ]);
          clearTimeout(timeoutId);
          const wall = ((Date.now() - t0) / 1000).toFixed(1);
          stats.ok++;
          log(`[${i + 1}/${total}] OK    ${example}  type=${animType}  dur=${duration}s  wall=${wall}s${postfix()}`);
          results.push({
            folder: example, status: "success",
            animationType: animType, durationS: duration,
            wall_ms: Date.now() - t0, captureMethod,
          });
          succeededSet.add(example);
        } catch (err) {
          clearTimeout(timeoutId);
          stats.err++;
          log(`[${i + 1}/${total}] ERROR ${example}: ${err.message?.slice(0, 120)}${postfix()}`);
          results.push({ folder: example, status: "error", reason: err.message?.slice(0, 200) });
        }

        doneCount++;
        if (doneCount % 5 === 0) saveManifest(MANIFEST_PATH, results, opts, startTime);
      }
    } finally {
      await browser.close().catch(() => {});
    }
  }

  await Promise.all(Array.from({ length: opts.concurrency }, (_, i) => worker(i)));

  saveManifest(MANIFEST_PATH, results, opts, startTime);

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  const successes = results.filter((r) => r.status === "success");
  const errors = results.filter((r) => r.status !== "success");

  log(`\n=== Done ===`);
  log(`  Recorded: ${successes.length}`);
  log(`  Errors:   ${errors.length}`);
  log(`  Time:     ${elapsed}s`);
  log(`  Manifest: ${MANIFEST_PATH}`);

  if (errors.length > 0) {
    log(`  Failed:`);
    for (const e of errors) log(`    ${e.folder}: ${e.reason}`);
  }

  logStream.end();
  staticLogStream.end();
}

main().catch((err) => {
  console.error(`FATAL: ${err.stack || err.message}`);
  process.exit(1);
});
