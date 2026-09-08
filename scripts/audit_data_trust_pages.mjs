#!/usr/bin/env node
/** Rendered acceptance audit for the five governed Data/Methods pages.
 *
 * Uses only Node built-ins plus a locally installed Chromium browser. It starts
 * an ephemeral static server and isolated headless browser profile, then checks
 * responsive overflow, no-JavaScript evidence, keyboard focus, anchor clearance,
 * readable text, computed contrast, and console errors.
 */
import { spawn } from "node:child_process";
import { createReadStream, existsSync } from "node:fs";
import { mkdtemp, rm, stat } from "node:fs/promises";
import { createServer } from "node:http";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..");
const argIndex = process.argv.indexOf("--docs");
const DOCS_ROOT = path.resolve(
  argIndex >= 0 ? process.argv[argIndex + 1] : path.join(REPO_ROOT, "docs"),
);
const VIEWPORTS = [390, 768, 1280];
const PAGES = [
  { file: "data-quality.html", anchor: "entities", evidence: ".quality-table" },
  { file: "methodology.html", anchor: "labeling", evidence: ".table-scroll table" },
  { file: "era-boundaries.html", anchor: "sensitivity", evidence: ".table-wrap table" },
  { file: "label-models.html", anchor: "agreement", evidence: ".rate-row, .agreement-row" },
  { file: "metrics.html", anchor: "jaccard", evidence: ".metric-card" },
];

if (process.argv.includes("--help")) {
  console.log("Usage: node scripts/audit_data_trust_pages.mjs [--docs /absolute/docs/path]");
  process.exit(0);
}

function invariant(condition, message) {
  if (!condition) throw new Error(message);
}

function contentType(filename) {
  return ({
    ".css": "text/css; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
  })[path.extname(filename)] || "application/octet-stream";
}

async function staticServer(root) {
  const resolvedRoot = path.resolve(root);
  const server = createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url || "/", "http://127.0.0.1");
      if (requestUrl.pathname === "/favicon.ico") {
        response.writeHead(204, { "Cache-Control": "no-store" });
        response.end();
        return;
      }
      const relative = decodeURIComponent(requestUrl.pathname).replace(/^\/+/, "") || "index.html";
      let filename = path.resolve(resolvedRoot, relative);
      invariant(
        filename === resolvedRoot || filename.startsWith(`${resolvedRoot}${path.sep}`),
        "path escapes docs root",
      );
      const info = await stat(filename);
      if (info.isDirectory()) filename = path.join(filename, "index.html");
      response.writeHead(200, {
        "Content-Type": contentType(filename),
        "Cache-Control": "no-store",
      });
      createReadStream(filename).pipe(response);
    } catch {
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("not found");
    }
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  invariant(address && typeof address !== "string", "static server did not bind a TCP port");
  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}

async function freePort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  invariant(address && typeof address !== "string", "could not reserve a debugging port");
  const port = address.port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

function chromeBinary() {
  const candidates = [
    process.env.CHROME_BINARY,
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].filter(Boolean);
  const found = candidates.find((candidate) => existsSync(candidate));
  invariant(found, "no local Chrome/Chromium binary found; set CHROME_BINARY");
  return found;
}

async function waitForDebugger(port, child, stderrText) {
  const endpoint = `http://127.0.0.1:${port}/json/version`;
  for (let attempt = 0; attempt < 300; attempt += 1) {
    if (child.exitCode !== null || child.signalCode !== null) {
      throw new Error(
        `Chrome stopped (exit=${child.exitCode}, signal=${child.signalCode}): ${stderrText()}`,
      );
    }
    try {
      const response = await fetch(endpoint);
      if (response.ok) return await response.json();
    } catch {
      // Browser startup is expected to race the first few requests.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(
    `timed out connecting to the isolated browser on port ${port}: ${stderrText()}`,
  );
}

class CdpClient {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    this.waiters = new Map();
    this.handlers = [];
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(String(event.data));
      if (message.id) {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        if (message.error) pending.reject(new Error(message.error.message));
        else pending.resolve(message.result || {});
        return;
      }
      for (const handler of this.handlers) handler(message.method, message.params || {});
      const queued = this.waiters.get(message.method);
      if (queued?.length) queued.shift().resolve(message.params || {});
    });
  }

  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  once(method, timeoutMs = 10_000) {
    return new Promise((resolve, reject) => {
      const queue = this.waiters.get(method) || [];
      const timer = setTimeout(() => reject(new Error(`timed out waiting for ${method}`)), timeoutMs);
      queue.push({
        resolve: (value) => {
          clearTimeout(timer);
          resolve(value);
        },
      });
      this.waiters.set(method, queue);
    });
  }

  onEvent(handler) {
    this.handlers.push(handler);
  }

  close() {
    this.socket.close();
  }
}

async function connectCdp(url) {
  const socket = new WebSocket(url);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });
  return new CdpClient(socket);
}

async function evaluate(client, expression) {
  const response = await client.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (response.exceptionDetails) {
    throw new Error(response.exceptionDetails.exception?.description || response.exceptionDetails.text);
  }
  return response.result?.value;
}

async function navigate(client, url) {
  const result = await client.send("Page.navigate", { url });
  invariant(!result.errorText, `navigation failed: ${result.errorText}`);
  for (let attempt = 0; attempt < 300; attempt += 1) {
    try {
      const ready = await evaluate(
        client,
        `document.readyState === "complete" && location.href === ${JSON.stringify(url)}`,
      );
      if (ready) {
        await new Promise((resolve) => setTimeout(resolve, 650));
        return;
      }
    } catch {
      // Navigation briefly destroys the prior execution context; retry in the new one.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`timed out waiting for the rendered page: ${url}`);
}

async function stopChild(child) {
  if (child.exitCode !== null || child.signalCode !== null) return;
  const waitForClose = () => new Promise((resolve) => child.once("close", resolve));
  let closed = waitForClose();
  child.kill("SIGTERM");
  await Promise.race([
    closed,
    new Promise((resolve) => setTimeout(resolve, 5_000)),
  ]);
  if (child.exitCode === null && child.signalCode === null) {
    closed = waitForClose();
    child.kill("SIGKILL");
    await Promise.race([
      closed,
      new Promise((resolve) => setTimeout(resolve, 2_000)),
    ]);
  }
}

const LAYOUT_EXPRESSION = String.raw`(() => {
  const visible = (element) => {
    const style = getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
  };
  const parseColor = (value) => {
    const match = value.match(/[\d.]+/g);
    if (!match || match.length < 3) return null;
    return [Number(match[0]), Number(match[1]), Number(match[2]), match[3] === undefined ? 1 : Number(match[3])];
  };
  const background = (element) => {
    for (let current = element; current; current = current.parentElement) {
      const style = getComputedStyle(current);
      if (style.backgroundImage !== "none") return null;
      const color = parseColor(style.backgroundColor);
      if (color && color[3] > 0.98) return color;
    }
    return [255, 255, 255, 1];
  };
  const luminance = ([r, g, b]) => {
    const channel = (value) => {
      const normalized = value / 255;
      return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
    };
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
  };
  const contrast = (left, right) => {
    const a = luminance(left);
    const b = luminance(right);
    return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
  };
  const directText = (element) => [...element.childNodes].some(
    (node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim(),
  );
  const textElements = [...document.querySelectorAll("body *")].filter(
    (element) => visible(element) && directText(element),
  );
  const contrastFailures = [];
  const tinyText = [];
  for (const element of textElements) {
    const style = getComputedStyle(element);
    const size = Number.parseFloat(style.fontSize);
    if (size < 10.5 && !["SUP", "SUB"].includes(element.tagName)) {
      tinyText.push({ tag: element.tagName, text: element.textContent.trim().slice(0, 80), size });
    }
    const foreground = parseColor(style.color);
    const behind = background(element);
    if (!foreground || !behind || foreground[3] < 0.98) continue;
    const weight = Number.parseInt(style.fontWeight, 10) || 400;
    const required = size >= 24 || (size >= 18.66 && weight >= 700) ? 3 : 4.5;
    const ratio = contrast(foreground, behind);
    if (ratio + 0.05 < required) {
      contrastFailures.push({
        tag: element.tagName,
        text: element.textContent.trim().slice(0, 80),
        ratio: Number(ratio.toFixed(2)),
        required,
        color: style.color,
        background: getComputedStyle(element.parentElement || document.body).backgroundColor,
      });
    }
  }
  return {
    viewport: window.innerWidth,
    documentWidth: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
    figures: document.querySelectorAll("figure[data-substantive-chart]").length,
    emptyFigures: [...document.querySelectorAll("figure[data-substantive-chart]")]
      .filter((figure) => visible(figure) && figure.innerText.trim().length < 20).length,
    undersizedDeepLinks: [...document.querySelectorAll("a.deep-link")]
      .filter((link) => {
        const rect = link.getBoundingClientRect();
        return visible(link) && (rect.width < 24 || rect.height < 24);
      })
      .map((link) => {
        const rect = link.getBoundingClientRect();
        return { href: link.getAttribute("href"), width: rect.width, height: rect.height };
      }),
    contrastFailures: contrastFailures.slice(0, 12),
    tinyText: tinyText.slice(0, 12),
  };
})()`;

async function audit() {
  invariant(existsSync(DOCS_ROOT), `docs directory does not exist: ${DOCS_ROOT}`);
  for (const page of PAGES) {
    invariant(existsSync(path.join(DOCS_ROOT, page.file)), `missing governed page ${page.file}`);
  }
  const server = await staticServer(DOCS_ROOT);
  const debugPort = await freePort();
  const profile = await mkdtemp(path.join(os.tmpdir(), "presidential-profiles-browser-audit-"));
  const chrome = spawn(chromeBinary(), [
    "--headless=new",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-default-apps",
    "--disable-extensions",
    "--disable-sync",
    "--metrics-recording-only",
    "--no-default-browser-check",
    "--no-first-run",
    "--remote-debugging-address=127.0.0.1",
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${profile}`,
    "about:blank",
  ], { stdio: ["ignore", "ignore", "pipe"] });
  const chromeStderr = [];
  chrome.stderr.on("data", (chunk) => {
    chromeStderr.push(String(chunk));
    if (chromeStderr.length > 40) chromeStderr.shift();
  });
  let client;
  const failures = [];
  const observations = [];
  try {
    await waitForDebugger(debugPort, chrome, () => chromeStderr.join("").slice(-8_000));
    const targetResponse = await fetch(
      `http://127.0.0.1:${debugPort}/json/new?${encodeURIComponent("about:blank")}`,
      { method: "PUT" },
    );
    invariant(targetResponse.ok, "could not create browser audit tab");
    const target = await targetResponse.json();
    client = await connectCdp(target.webSocketDebuggerUrl);
    await Promise.all([
      client.send("Page.enable"),
      client.send("Runtime.enable"),
      client.send("Log.enable"),
      client.send("Network.enable"),
    ]);
    await client.send("Network.setCacheDisabled", { cacheDisabled: true });
    let consoleErrors = [];
    client.onEvent((method, params) => {
      if (method === "Runtime.exceptionThrown") {
        consoleErrors.push(params.exceptionDetails?.exception?.description || params.exceptionDetails?.text || method);
      } else if (method === "Runtime.consoleAPICalled" && params.type === "error") {
        consoleErrors.push("console.error");
      } else if (method === "Log.entryAdded" && params.entry?.level === "error") {
        consoleErrors.push(params.entry.text || "log error");
      }
    });

    let navigationId = 0;
    for (const page of PAGES) {
      for (const width of VIEWPORTS) {
        await client.send("Emulation.setDeviceMetricsOverride", {
          width,
          height: 900,
          deviceScaleFactor: 1,
          mobile: width <= 768,
          screenWidth: width,
          screenHeight: 900,
        });
        await client.send("Emulation.setScriptExecutionDisabled", { value: false });
        consoleErrors = [];
        navigationId += 1;
        await navigate(client, `${server.baseUrl}/${page.file}?audit=${navigationId}`);
        const layout = await evaluate(client, LAYOUT_EXPRESSION);
        if (layout.documentWidth > layout.viewport + 1) {
          failures.push(`${page.file} at ${width}px overflows document: ${layout.documentWidth}px`);
        }
        if (layout.emptyFigures) {
          failures.push(`${page.file} at ${width}px has ${layout.emptyFigures} empty substantive figures`);
        }
        if (layout.undersizedDeepLinks.length) {
          failures.push(
            `${page.file} at ${width}px has deep links below 24px: ${JSON.stringify(layout.undersizedDeepLinks)}`,
          );
        }
        if (layout.tinyText.length) {
          failures.push(`${page.file} at ${width}px has text below 10.5px: ${JSON.stringify(layout.tinyText)}`);
        }
        if (layout.contrastFailures.length) {
          failures.push(`${page.file} at ${width}px has contrast failures: ${JSON.stringify(layout.contrastFailures)}`);
        }
        if (consoleErrors.length) {
          failures.push(`${page.file} at ${width}px console errors: ${consoleErrors.join(" | ")}`);
        }

        await evaluate(client, "document.activeElement?.blur(); document.body.focus(); true");
        await client.send("Input.dispatchKeyEvent", {
          type: "keyDown", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9,
        });
        await client.send("Input.dispatchKeyEvent", {
          type: "keyUp", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9,
        });
        const focus = await evaluate(client, String.raw`(() => {
          const element = document.activeElement;
          const style = getComputedStyle(element);
          return {
            tag: element?.tagName || "",
            text: element?.textContent?.trim().slice(0, 60) || "",
            outlineWidth: Number.parseFloat(style.outlineWidth) || 0,
            outlineStyle: style.outlineStyle,
            boxShadow: style.boxShadow,
          };
        })()`);
        if (["", "BODY", "HTML"].includes(focus.tag)) {
          failures.push(`${page.file} at ${width}px did not expose a keyboard focus target`);
        } else if (
          (focus.outlineStyle === "none" || focus.outlineWidth < 2)
          && (!focus.boxShadow || focus.boxShadow === "none")
        ) {
          failures.push(`${page.file} at ${width}px focus is not visibly styled: ${JSON.stringify(focus)}`);
        }
        observations.push({ page: page.file, width, documentWidth: layout.documentWidth, focus: focus.tag });
      }

      await client.send("Emulation.setDeviceMetricsOverride", {
        width: 390, height: 900, deviceScaleFactor: 1, mobile: true,
        screenWidth: 390, screenHeight: 900,
      });
      navigationId += 1;
      await client.send("Emulation.setScriptExecutionDisabled", { value: true });
      await navigate(client, `${server.baseUrl}/${page.file}?audit=${navigationId}`);
      const noJs = await evaluate(client, `(() => {
        const evidence = [...document.querySelectorAll(${JSON.stringify(page.evidence)})]
          .filter((element) => {
            const rect = element.getBoundingClientRect();
            const style = getComputedStyle(element);
            return rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden";
          });
        return { evidence: evidence.length, text: document.body.innerText.trim().length };
      })()`);
      if (!noJs.evidence || noJs.text < 500) {
        failures.push(`${page.file} loses its evidence when JavaScript is disabled`);
      }
      await client.send("Emulation.setScriptExecutionDisabled", { value: false });

      navigationId += 1;
      await navigate(
        client,
        `${server.baseUrl}/${page.file}?audit=${navigationId}#${encodeURIComponent(page.anchor)}`,
      );
      const anchor = await evaluate(client, `(() => {
        const target = document.getElementById(${JSON.stringify(page.anchor)});
        if (!target) return null;
        const stickyBottom = Math.max(0, ...[...document.querySelectorAll("body *")]
          .filter((element) => {
            const style = getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            return ["sticky", "fixed"].includes(style.position) && rect.bottom > 0 && rect.top < 250;
          }).map((element) => element.getBoundingClientRect().bottom));
        return { top: target.getBoundingClientRect().top, stickyBottom };
      })()`);
      if (!anchor) failures.push(`${page.file} is missing #${page.anchor}`);
      else if (anchor.top + 1 < anchor.stickyBottom) {
        failures.push(
          `${page.file} #${page.anchor} is obscured: top ${anchor.top.toFixed(1)} < sticky bottom ${anchor.stickyBottom.toFixed(1)}`,
        );
      }
    }
  } finally {
    if (client) client.close();
    await stopChild(chrome);
    await server.close();
    await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  }

  if (failures.length) {
    console.error(`Data Trust browser audit failed (${failures.length}):\n- ${failures.join("\n- ")}`);
    process.exitCode = 1;
  } else {
    console.log(
      `Data Trust browser audit passed: ${PAGES.length} pages × ${VIEWPORTS.length} widths; `
      + "JavaScript-disabled evidence, keyboard focus, contrast, anchors, and console checked.",
    );
    console.log(JSON.stringify(observations));
  }
}

await audit();
process.exit(process.exitCode || 0);
