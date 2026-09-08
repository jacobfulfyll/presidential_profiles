#!/usr/bin/env node
/** Whole-site rendered phone audit.
 *
 * Uses Node built-ins and a local Chromium binary. The audit discovers every
 * generated HTML page, serves docs from an ephemeral localhost port, and uses
 * the Chrome DevTools Protocol directly so the release gate has no browser
 * package dependency.
 */
import { spawn } from "node:child_process";
import { createReadStream, existsSync } from "node:fs";
import { mkdtemp, readFile, readdir, rm, stat } from "node:fs/promises";
import { createServer } from "node:http";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..");
const docsArg = process.argv.indexOf("--docs");
const baselineArg = process.argv.indexOf("--desktop-baseline");
const DOCS_ROOT = path.resolve(
  docsArg >= 0 ? process.argv[docsArg + 1] : path.join(REPO_ROOT, "docs"),
);
const DESKTOP_BASELINE = baselineArg >= 0 ? path.resolve(process.argv[baselineArg + 1]) : null;
const SPECIAL_ONLY = process.argv.includes("--special-only");
const VIEWPORTS = [
  { name: "phone-320", width: 320, height: 568, mobile: true, touch: true },
  { name: "phone-390", width: 390, height: 844, mobile: true, touch: true },
  { name: "phone-430", width: 430, height: 932, mobile: true, touch: true },
  { name: "phone-760", width: 760, height: 900, mobile: true, touch: true },
  { name: "desktop-boundary-768", width: 768, height: 1024, mobile: false, touch: false },
];
const REPRESENTATIVE_PAGES = [
  "index.html",
  "summary.html",
  "compare.html",
  "explorer.html",
  "presidents/index.html",
  "presidents/franklin-d-roosevelt.html",
  "presidents/william-harrison.html",
  "issues/index.html",
  "issues/foreign-policy.html",
  "issues/health-care.html",
  "data-quality.html",
  "methodology.html",
  "era-boundaries.html",
  "label-models.html",
  "metrics.html",
  "feedback.html",
];

if (process.argv.includes("--help")) {
  console.log("Usage: node scripts/audit_mobile_site.mjs [--docs /absolute/docs/path] [--desktop-baseline /tmp/baseline] [--special-only]");
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

async function htmlPages(root, relative = "") {
  const directory = path.join(root, relative);
  const pages = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const child = path.join(relative, entry.name);
    if (entry.isDirectory()) pages.push(...await htmlPages(root, child));
    else if (entry.isFile() && entry.name.endsWith(".html")) pages.push(child.split(path.sep).join("/"));
  }
  return pages.sort();
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
  invariant(address && typeof address !== "string", "static server did not bind");
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
      throw new Error(`Chrome stopped before audit startup: ${stderrText()}`);
    }
    try {
      const response = await fetch(endpoint);
      if (response.ok) return;
    } catch {
      // Browser startup races the first few requests.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`timed out connecting to Chromium: ${stderrText()}`);
}

class CdpClient {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
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
    });
    socket.addEventListener("close", () => {
      for (const pending of this.pending.values()) pending.reject(new Error("Chrome DevTools connection closed"));
      this.pending.clear();
    });
  }

  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Chrome DevTools command timed out: ${method}`));
      }, 20_000);
      this.pending.set(id, {
        resolve: (value) => { clearTimeout(timer); resolve(value); },
        reject: (error) => { clearTimeout(timer); reject(error); },
      });
      this.socket.send(JSON.stringify({ id, method, params }));
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

async function navigate(client, url, settleMs = 140) {
  const expected = new URL(url);
  const result = await client.send("Page.navigate", { url });
  invariant(!result.errorText, `navigation failed: ${result.errorText}`);
  let diagnostic = null;
  for (let attempt = 0; attempt < 300; attempt += 1) {
    try {
      diagnostic = await evaluate(
        client,
        `({readyState:document.readyState,hasBody:Boolean(document.body),href:location.href})`,
      );
      const actual = new URL(diagnostic.href);
      if (diagnostic.readyState !== "loading" && diagnostic.hasBody
          && actual.origin === expected.origin && actual.pathname === expected.pathname) {
        await evaluate(client, "scrollTo(0, 0); true");
        await new Promise((resolve) => setTimeout(resolve, settleMs));
        return;
      }
    } catch {
      // Navigation briefly replaces the prior execution context.
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`timed out waiting for ${url}: ${JSON.stringify(diagnostic)}`);
}

async function stopChild(child) {
  if (child.exitCode !== null || child.signalCode !== null) return;
  const waitForClose = () => new Promise((resolve) => child.once("close", resolve));
  let closed = waitForClose();
  child.kill("SIGTERM");
  await Promise.race([closed, new Promise((resolve) => setTimeout(resolve, 5_000))]);
  if (child.exitCode === null && child.signalCode === null) {
    closed = waitForClose();
    child.kill("SIGKILL");
    await Promise.race([closed, new Promise((resolve) => setTimeout(resolve, 2_000))]);
  }
}

async function emulate(client, viewport) {
  await client.send("Emulation.setDeviceMetricsOverride", {
    width: viewport.width,
    height: viewport.height,
    deviceScaleFactor: 1,
    mobile: viewport.mobile,
    screenWidth: viewport.width,
    screenHeight: viewport.height,
  });
  await client.send("Emulation.setTouchEmulationEnabled", viewport.touch
    ? { enabled: true, maxTouchPoints: 5 }
    : { enabled: false });
}

const LAYOUT_EXPRESSION = String.raw`((phone) => {
  const visible = (element) => {
    if (!element || element.closest("[hidden],[inert]")) return false;
    const style = getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== "none" && style.visibility !== "hidden"
      && Number(style.opacity || 1) > 0 && rect.width > 0 && rect.height > 0
      && rect.right > -1000 && rect.bottom > -1000;
  };
  const roundedRect = (rect) => ({
    width: Number(rect.width.toFixed(1)), height: Number(rect.height.toFixed(1)),
  });
  const directText = (element) => [...element.childNodes].some(
    (node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim(),
  );
  const effectiveRect = (element) => {
    if (element.matches('input[type="checkbox"],input[type="radio"]')) {
      const label = element.closest("label") || (element.id
        ? document.querySelector('label[for="' + CSS.escape(element.id) + '"]') : null);
      if (label && visible(label)) return label.getBoundingClientRect();
    }
    return element.getBoundingClientRect();
  };
  const approvedScroller = (element) => {
    for (let current = element.parentElement; current; current = current.parentElement) {
      const style = getComputedStyle(current);
      if (["auto", "scroll"].includes(style.overflowX)
          && current.scrollWidth > current.clientWidth + 1) return current;
    }
    return null;
  };
  const documentWidth = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth);
  const globalNav = document.querySelector(".global-nav");
  const mobileNav = document.querySelector(".mobile-nav");
  const desktopLinks = document.querySelector(".nav-links");
  const nav = {
    height: globalNav ? Number(globalNav.getBoundingClientRect().height.toFixed(1)) : 0,
    mobileVisible: visible(mobileNav), desktopVisible: visible(desktopLinks),
  };
  const targetSelector = [
    "button", "select", "input:not([type=hidden])", "summary", "[role=button]",
    "nav a", "a.deep-link",
  ].join(",");
  const undersizedTargets = phone ? [...document.querySelectorAll(targetSelector)]
    .filter((element) => visible(element) && !element.matches(":disabled")
      && !element.closest(".breadcrumbs,.crumbs"))
    .map((element) => ({ element, rect: effectiveRect(element) }))
    .filter(({ rect }) => rect.width < 43.5 || rect.height < 43.5)
    .slice(0, 16)
    .map(({ element, rect }) => ({
      tag: element.tagName, cls: String(element.className || "").slice(0, 70),
      text: (element.getAttribute("aria-label") || element.textContent || "").trim().slice(0, 70),
      ...roundedRect(rect),
    })) : [];
  const tinyText = phone ? [...document.querySelectorAll("body *")]
    .filter((element) => visible(element) && directText(element)
      && !["SUP", "SUB", "OPTION"].includes(element.tagName))
    .map((element) => ({ element, size: Number.parseFloat(getComputedStyle(element).fontSize) }))
    .filter(({ element, size }) => size + 0.01 < (element.closest("svg") ? 10.5 : 12))
    .slice(0, 16)
    .map(({ element, size }) => ({
      tag: element.tagName, cls: String(element.className || "").slice(0, 70),
      text: element.textContent.trim().slice(0, 70), size,
    })) : [];
  const clippedSubstantive = [...document.querySelectorAll(
    "table,pre,svg,.chart,.js-plotly-plot,.trend-svg,.pc-network-stage",
  )].filter((element) => {
    if (!visible(element)) return false;
    const rect = element.getBoundingClientRect();
    return (rect.left < -1 || rect.right > innerWidth + 1) && !approvedScroller(element);
  }).slice(0, 12).map((element) => {
    const rect = element.getBoundingClientRect();
    return { tag: element.tagName, cls: String(element.className || "").slice(0, 70),
      left: Number(rect.left.toFixed(1)), right: Number(rect.right.toFixed(1)) };
  });
  const scrollers = [...document.querySelectorAll("body *")].filter((element) => {
    if (!visible(element)) return false;
    const style = getComputedStyle(element);
    return ["auto", "scroll"].includes(style.overflowX)
      && element.scrollWidth > element.clientWidth + 1;
  }).map((element) => {
    const figure = element.closest("figure");
    const caption = figure?.querySelector("figcaption")?.textContent.trim() || "";
    const tableCaption = element.querySelector("table > caption")?.textContent.trim() || "";
    const named = Boolean(element.getAttribute("aria-label")
      || element.getAttribute("aria-labelledby") || caption || tableCaption
      || element.matches("nav") || element.querySelector("a,button,input,select,summary"));
    const original = element.scrollLeft;
    element.scrollLeft = 0;
    const start = element.scrollLeft;
    element.scrollLeft = element.scrollWidth;
    const reachable = element.scrollLeft > start;
    element.scrollLeft = original;
    return { cls: String(element.className || "").slice(0, 70), named, reachable,
      client: element.clientWidth, scroll: element.scrollWidth };
  });
  const blankFigures = [...document.querySelectorAll("figure[data-substantive-chart]")]
    .filter((figure) => visible(figure)
      && !figure.querySelector("svg,canvas,table,.js-plotly-plot")
      && figure.innerText.trim().length < 20).length;
  return {
    viewport: innerWidth, documentWidth, nav, undersizedTargets, tinyText,
    clippedSubstantive, unnamedScrollers: scrollers.filter((item) => !item.named),
    unreachableScrollers: scrollers.filter((item) => !item.reachable), blankFigures,
  };
})`;

const GEOMETRY_EXPRESSION = String.raw`(() => ({
  document: [document.documentElement.scrollWidth, document.documentElement.scrollHeight],
  elements: [...document.querySelectorAll("nav,header,main,section,article,footer")]
    .filter((element) => {
      const style = getComputedStyle(element); const rect = element.getBoundingClientRect();
      return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
    }).map((element) => {
      const style = getComputedStyle(element); const rect = element.getBoundingClientRect();
      return { cls: String(element.className || ""), display: style.display, font: style.fontSize,
        h: rect.height, id: element.id || "", position: style.position, tag: element.tagName,
        w: rect.width, x: rect.x, y: rect.y };
    }),
  viewport: [innerWidth, innerHeight],
}))()`;

async function desktopGeometryChecks(client, baseUrl, baselineRoot, failures) {
  const sets = [
    { key: "1280", width: 1280, height: 720 },
    { key: "1024", width: 1024, height: 768 },
    { key: "768", width: 768, height: 1024 },
    { key: "844x390", width: 844, height: 390 },
  ];
  const closeEnough = (left, right, tolerance = 1.25) => Math.abs(left - right) <= tolerance;
  for (const set of sets) {
    const baselineRows = JSON.parse(await readFile(path.join(baselineRoot, `geometry-${set.key}.json`), "utf8"));
    await emulate(client, { ...set, mobile: false, touch: false });
    for (const baseline of baselineRows) {
      // Lazy Plotly layouts briefly expand while their responsive resize settles.
      await navigate(client, `${baseUrl}/${baseline.route}`, 900);
      const current = await evaluate(client, GEOMETRY_EXPRESSION);
      const context = `${baseline.route} at desktop ${set.key}`;
      if (!closeEnough(current.document[0], baseline.geometry.document[0], 0.55)
          || !closeEnough(current.document[1], baseline.geometry.document[1], 12)) {
        failures.push(`${context} changed document geometry: ${JSON.stringify({ before: baseline.geometry.document, after: current.document })}`);
        continue;
      }
      if (current.elements.length !== baseline.geometry.elements.length) {
        failures.push(`${context} changed visible-element inventory: ${baseline.geometry.elements.length}/${current.elements.length}`);
        continue;
      }
      const drift = current.elements.findIndex((element, index) => {
        const before = baseline.geometry.elements[index];
        const stableClass = (value) => value.split(/\s+/).filter((name) => name && name !== "is-visible").sort().join(" ");
        const heightTolerance = Math.max(1.5, Math.min(30, before.h * .015));
        const yTolerance = stableClass(before.cls).split(" ").includes("story-section") ? 36 : 20;
        return stableClass(element.cls) !== stableClass(before.cls)
          || ["display", "font", "id", "position", "tag"].some((key) => element[key] !== before[key])
          || !closeEnough(element.h, before.h, heightTolerance)
          || ["w", "x"].some((key) => !closeEnough(element[key], before[key]))
          || !closeEnough(element.y, before.y, yTolerance);
      });
      if (drift >= 0) {
        failures.push(`${context} changed visible-element geometry at item ${drift}: ${JSON.stringify({ before: baseline.geometry.elements[drift], after: current.elements[drift] })}`);
      }
    }
  }
}

async function focusCheck(client) {
  await evaluate(client, "document.activeElement?.blur(); document.body.focus(); true");
  await client.send("Input.dispatchKeyEvent", {
    type: "keyDown", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9,
  });
  await client.send("Input.dispatchKeyEvent", {
    type: "keyUp", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9,
  });
  return evaluate(client, String.raw`(() => {
    const element = document.activeElement;
    const style = getComputedStyle(element);
    return { tag: element?.tagName || "", text: element?.textContent?.trim().slice(0, 60) || "",
      outlineWidth: Number.parseFloat(style.outlineWidth) || 0,
      outlineStyle: style.outlineStyle, boxShadow: style.boxShadow };
  })()`);
}

async function anchorCheck(client) {
  const target = await evaluate(client, String.raw`(() => {
    const visible = (element) => {
      const style = getComputedStyle(element); const rect = element.getBoundingClientRect();
      return style.display !== "none" && style.visibility !== "hidden" && rect.height > 0;
    };
    const candidates = [...document.querySelectorAll("main section[id]")].filter(visible);
    const element = candidates.find((candidate) => candidate.offsetTop > innerHeight) || candidates[0];
    if (!element) return null;
    element.scrollIntoView({ block: "start" });
    const rect = element.getBoundingClientRect();
    const stickyBottom = Math.max(0, ...[...document.querySelectorAll("body *")]
      .filter((candidate) => {
        const style = getComputedStyle(candidate); const box = candidate.getBoundingClientRect();
        return ["sticky", "fixed"].includes(style.position) && box.bottom > 0 && box.top < 220;
      }).map((candidate) => candidate.getBoundingClientRect().bottom));
    return { id: element.id, top: rect.top, stickyBottom, scrollY };
  })()`);
  if (!target || target.scrollY < 2) return null;
  return target;
}

async function touch(client, x, y) {
  await client.send("Input.dispatchTouchEvent", {
    type: "touchStart", touchPoints: [{ x, y, radiusX: 1, radiusY: 1, force: 1, id: 1 }],
  });
  await client.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
}

async function horizontalSwipe(client, x, y, distance = 180) {
  const point = (currentX) => [{ x: currentX, y, radiusX: 1, radiusY: 1, force: 1, id: 1 }];
  await client.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: point(x) });
  for (let step = 1; step <= 10; step += 1) {
    await client.send("Input.dispatchTouchEvent", {
      type: "touchMove", touchPoints: point(x - (distance * step / 10)),
    });
    await new Promise((resolve) => setTimeout(resolve, 16));
  }
  await client.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
}

async function menuChecks(client, baseUrl, page, navigationId) {
  const viewport = VIEWPORTS[1];
  await emulate(client, viewport);
  await client.send("Emulation.setScriptExecutionDisabled", { value: false });
  await navigate(client, `${baseUrl}/${page}?audit=menu-${navigationId}`);
  const summaryRect = await evaluate(client, String.raw`(() => {
    const summary = document.querySelector(".mobile-menu > summary");
    if (!summary) return null;
    const rect = summary.getBoundingClientRect();
    return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
  })()`);
  invariant(summaryRect, `${page} is missing the phone menu summary`);
  await touch(client, summaryRect.x, summaryRect.y);
  const opened = await evaluate(client, "document.querySelector('.mobile-menu')?.open === true");
  invariant(opened, `${page} phone menu did not open from a touch tap`);
  await touch(client, 3, 180);
  const outsideClosed = await evaluate(client, "document.querySelector('.mobile-menu')?.open === false");
  invariant(outsideClosed, `${page} phone menu did not close after outside activation`);
  await touch(client, summaryRect.x, summaryRect.y);
  await evaluate(client, "document.querySelector('.mobile-menu > summary')?.focus(); true");
  await client.send("Input.dispatchKeyEvent", {
    type: "keyDown", key: "Escape", code: "Escape", windowsVirtualKeyCode: 27,
  });
  await client.send("Input.dispatchKeyEvent", {
    type: "keyUp", key: "Escape", code: "Escape", windowsVirtualKeyCode: 27,
  });
  const escaped = await evaluate(client, String.raw`(() => ({
    closed: document.querySelector(".mobile-menu")?.open === false,
    restored: document.activeElement === document.querySelector(".mobile-menu > summary"),
  }))()`);
  invariant(escaped.closed && escaped.restored, `${page} Escape did not close and restore phone-menu focus`);

  await client.send("Emulation.setScriptExecutionDisabled", { value: true });
  await navigate(client, `${baseUrl}/${page}?audit=menu-no-js-${navigationId}`);
  const noJsRect = await evaluate(client, String.raw`(() => {
    const summary = document.querySelector(".mobile-menu > summary");
    const rect = summary?.getBoundingClientRect();
    return rect ? { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 } : null;
  })()`);
  invariant(noJsRect, `${page} loses the phone menu without JavaScript`);
  await touch(client, noJsRect.x, noJsRect.y);
  const noJsOpen = await evaluate(client, "document.querySelector('.mobile-menu')?.open === true");
  invariant(noJsOpen, `${page} native phone menu does not open without JavaScript`);
  await client.send("Emulation.setScriptExecutionDisabled", { value: false });
}

async function noJavaScriptChecks(client, baseUrl) {
  await emulate(client, VIEWPORTS[1]);
  await client.send("Emulation.setScriptExecutionDisabled", { value: true });
  let ordinal = 0;
  for (const page of REPRESENTATIVE_PAGES.filter((item) => item !== "feedback.html")) {
    ordinal += 1;
    await navigate(client, `${baseUrl}/${page}?audit=no-js-${ordinal}`, 80);
    const state = await evaluate(client, String.raw`(() => {
      const primary = document.querySelector("main") || document.querySelector("header");
      return { text: document.body.innerText.trim().length,
        primary: Boolean(primary), mobileMenu: Boolean(document.querySelector(".mobile-menu")),
        emptyPrimary: (primary?.innerText.trim().length || 0) < 120 };
    })()`);
    invariant(state.primary && state.mobileMenu && !state.emptyPrimary && state.text > 400,
      `${page} loses meaningful server-rendered evidence without JavaScript`);
  }
  await client.send("Emulation.setScriptExecutionDisabled", { value: false });
}

async function explorerFailureCheck(client, baseUrl) {
  await emulate(client, VIEWPORTS[1]);
  await client.send("Network.setBlockedURLs", { urls: ["*/explorer/index_v2.json*"] });
  await navigate(client, `${baseUrl}/explorer.html?audit=blocked-explorer`, 500);
  const state = await evaluate(client, String.raw`(() => ({
    fallback: Boolean(document.querySelector(".explore-fallback, [data-explore-fallback]")),
    retry: [...document.querySelectorAll("button")].some((button) => /retry/i.test(button.textContent)),
    text: document.querySelector("main")?.innerText.trim().length || 0,
  }))()`);
  await client.send("Network.setBlockedURLs", { urls: [] });
  invariant(state.text > 500 && (state.fallback || state.retry),
    "Explore request failure removed its fallback instead of preserving evidence/retry");
}

async function representativeInteractionChecks(client, baseUrl) {
  await emulate(client, VIEWPORTS[0]);
  await client.send("Emulation.setScriptExecutionDisabled", { value: false });

  await navigate(client, `${baseUrl}/index.html?audit=story-stage`, 250);
  const storyStage = await evaluate(client, String.raw`(() => {
    const tab = [...document.querySelectorAll('[data-era-workspace-tab][aria-selected="false"]')]
      .find((candidate) => !candidate.closest('[hidden]') && candidate.getClientRects().length);
    const panel = tab && document.getElementById(tab.getAttribute("aria-controls"));
    if (!tab || !panel) return null;
    tab.click();
    return { selected: tab.getAttribute("aria-selected"), hidden: panel.hidden };
  })()`);
  invariant(storyStage?.selected === "true" && !storyStage.hidden,
    "Story staged controls did not reveal their associated panel");

  await navigate(client, `${baseUrl}/compare.html?audit=radar`, 250);
  await evaluate(client, "document.querySelector('#rhetoric')?.scrollIntoView({block:'center'}); true");
  let radar = null;
  for (let attempt = 0; attempt < 60; attempt += 1) {
    radar = await evaluate(client, String.raw`(() => {
      const path = document.querySelector(".radar-chart .polarsublayer.plotbg path");
      if (!path) return null;
      const box = path.getBoundingClientRect();
      return { width: box.width, height: box.height };
    })()`);
    if (radar?.width >= 219.5 && radar?.height >= 219.5) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  invariant(radar?.width >= 219.5 && radar?.height >= 219.5,
    `Compare phone radar is smaller than 220px: ${JSON.stringify(radar)}`);

  await navigate(client, `${baseUrl}/explorer.html?audit=search-history`, 500);
  const exploreSearch = await evaluate(client, String.raw`(() => {
    const search = document.getElementById("catalog-search");
    if (!search) return null;
    search.value = "economy";
    search.dispatchEvent(new Event("input", {bubbles:true}));
    const rows = [...document.querySelectorAll(".catalog-row")].filter((row) => !row.hidden
      && getComputedStyle(row).display !== "none");
    const before = location.href;
    const choice = rows.map((row) => row.querySelector('input:not(:disabled)')).find(Boolean);
    choice?.click();
    return { matches: rows.length, labels: rows.map((row) => row.textContent.trim()).slice(0, 4),
      before, clicked: Boolean(choice) };
  })()`);
  let exploreHistoryChanged = false;
  for (let attempt = 0; attempt < 60; attempt += 1) {
    exploreHistoryChanged = await evaluate(client,
      `location.href !== ${JSON.stringify(exploreSearch?.before || "")}`);
    if (exploreHistoryChanged) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  invariant(exploreSearch?.matches > 0
      && exploreSearch.labels.some((label) => /econom/i.test(label))
      && exploreSearch.clicked && exploreHistoryChanged,
  `Explore search/history interaction failed: ${JSON.stringify({...exploreSearch, historyChanged: exploreHistoryChanged})}`);
}

async function touchScrollerCheck(client, baseUrl, page, selector) {
  await emulate(client, VIEWPORTS[1]);
  await navigate(client, `${baseUrl}/${page}?audit=touch-scroll`, 500);
  const start = await evaluate(client, `(() => {
    const element = document.querySelector(${JSON.stringify(selector)});
    if (!element) return null;
    element.scrollIntoView({block:"center"}); element.scrollLeft = 0;
    const rect = element.getBoundingClientRect();
    return {x:Math.min(innerWidth - 24, rect.right - 24), y:Math.max(80, Math.min(innerHeight - 24, rect.top + rect.height / 2)),
      client:element.clientWidth, scroll:element.scrollWidth};
  })()`);
  invariant(start && start.scroll > start.client + 1,
    `${page} is missing its expected touch scroller ${selector}`);
  await horizontalSwipe(client, start.x, start.y);
  await new Promise((resolve) => setTimeout(resolve, 180));
  const moved = await evaluate(client,
    `document.querySelector(${JSON.stringify(selector)})?.scrollLeft || 0`);
  invariant(moved > 1, `${page} did not respond to a horizontal touch gesture in ${selector}`);
}

async function touchScrollChecks(client, baseUrl) {
  await touchScrollerCheck(client, baseUrl, "explorer.html", ".trend-scroll");
  await touchScrollerCheck(client, baseUrl, "presidents/franklin-d-roosevelt.html", ".pc-network-scroll");
  await touchScrollerCheck(client, baseUrl, "era-boundaries.html", ".sensitivity-scroll");
}

async function textZoomChecks(client, baseUrl) {
  await emulate(client, VIEWPORTS[1]);
  let ordinal = 0;
  for (const page of REPRESENTATIVE_PAGES.filter((item) => item !== "feedback.html")) {
    ordinal += 1;
    await navigate(client, `${baseUrl}/${page}?audit=text-zoom-${ordinal}`, 80);
    const state = await evaluate(client, String.raw`(() => {
      document.documentElement.style.fontSize = "32px";
      const width = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth);
      const primary = document.querySelector("main") || document.querySelector("header");
      return { width, viewport: innerWidth, primary: primary?.getBoundingClientRect().width || 0 };
    })()`);
    invariant(state.width <= state.viewport + 1 && state.primary > 0,
      `${page} loses reflow at 200% text size: ${JSON.stringify(state)}`);
  }
}

async function landscapeChecks(client, baseUrl) {
  const landscape = { width: 844, height: 390, mobile: true, touch: true };
  await emulate(client, landscape);
  await client.send("Emulation.setScriptExecutionDisabled", { value: false });
  let ordinal = 0;
  for (const page of REPRESENTATIVE_PAGES) {
    ordinal += 1;
    await navigate(client, `${baseUrl}/${page}?audit=landscape-${ordinal}`, 80);
    const state = await evaluate(client, String.raw`(() => {
      const visible = (element) => {
        if (!element) return false;
        const style = getComputedStyle(element); const rect = element.getBoundingClientRect();
        return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
      };
      const secondary = [...document.querySelectorAll(
        ".story-meter,.summary-chapter-route,.compare-nav,.profile-nav,.issue-local-nav,.quality-nav,.methods-local-nav,.local-nav",
      )].filter(visible).filter((element) => ["sticky", "fixed"].includes(getComputedStyle(element).position));
      const globalNav = document.querySelector(".global-nav")?.getBoundingClientRect();
      return { coarse: matchMedia("(pointer:coarse)").matches,
        documentWidth: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
        mobileNav: visible(document.querySelector(".mobile-nav")),
        desktopNav: visible(document.querySelector(".nav-links")),
        navHeight: globalNav?.height || 0, secondarySticky: secondary.map((element) => element.className) };
    })()`);
    invariant(state.coarse, `${page} landscape emulation did not expose a coarse pointer`);
    invariant(state.documentWidth <= landscape.width + 1, `${page} overflows in coarse phone landscape`);
    invariant(state.mobileNav && !state.desktopNav && Math.abs(state.navHeight - 56) <= 1,
      `${page} does not use compact navigation in coarse phone landscape`);
    invariant(!state.secondarySticky.length,
      `${page} retains secondary sticky rails in short phone landscape: ${state.secondarySticky.join(", ")}`);
  }
}

async function preferenceChecks(client, baseUrl) {
  await emulate(client, VIEWPORTS[1]);
  await client.send("Emulation.setEmulatedMedia", { features: [
    { name: "prefers-reduced-motion", value: "reduce" },
    { name: "forced-colors", value: "active" },
  ] });
  await navigate(client, `${baseUrl}/index.html?audit=preferences`, 80);
  const state = await evaluate(client, String.raw`(() => {
    document.querySelector(".mobile-menu").open = true;
    const arrow = getComputedStyle(document.querySelector(".mobile-menu > summary"), "::after");
    const panel = getComputedStyle(document.querySelector(".nav-links"));
    return { reduced: matchMedia("(prefers-reduced-motion:reduce)").matches,
      forced: matchMedia("(forced-colors:active)").matches,
      transition: arrow.transitionDuration, border: Number.parseFloat(panel.borderTopWidth) || 0 };
  })()`);
  invariant(state.reduced && state.forced, "browser preference emulation did not activate");
  invariant(state.transition === "0s" && state.border >= 2,
    `phone navigation loses reduced-motion or forced-color treatment: ${JSON.stringify(state)}`);
  await client.send("Emulation.setEmulatedMedia", { features: [] });
}

async function audit() {
  invariant(existsSync(DOCS_ROOT), `docs directory does not exist: ${DOCS_ROOT}`);
  const pages = await htmlPages(DOCS_ROOT);
  invariant(pages.length === 74, `expected 74 physical HTML pages, found ${pages.length}`);
  for (const page of REPRESENTATIVE_PAGES) {
    invariant(pages.includes(page), `missing representative page ${page}`);
  }
  const server = await staticServer(DOCS_ROOT);
  const debugPort = await freePort();
  const profile = await mkdtemp(path.join(os.tmpdir(), "presidential-profiles-mobile-audit-"));
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
    invariant(targetResponse.ok, "could not create mobile audit tab");
    const target = await targetResponse.json();
    client = await connectCdp(target.webSocketDebuggerUrl);
    await Promise.all([
      client.send("Page.enable"), client.send("Runtime.enable"),
      client.send("Log.enable"), client.send("Network.enable"),
    ]);
    await client.send("Network.setCacheDisabled", { cacheDisabled: true });
    let consoleErrors = [];
    let localResourceErrors = [];
    client.onEvent((method, params) => {
      if (method === "Runtime.exceptionThrown") {
        consoleErrors.push(params.exceptionDetails?.exception?.description || params.exceptionDetails?.text || method);
      } else if (method === "Runtime.consoleAPICalled" && params.type === "error") {
        consoleErrors.push((params.args || []).map((arg) => arg.value || arg.description || "").join(" "));
      } else if (method === "Log.entryAdded" && params.entry?.level === "error") {
        consoleErrors.push(params.entry.text || "log error");
      } else if (method === "Network.responseReceived"
          && params.response?.url?.startsWith(server.baseUrl) && params.response.status >= 400) {
        localResourceErrors.push(`${params.response.status} ${params.response.url}`);
      }
    });

    let navigationId = 0;
    // Exercise trusted keyboard/touch paths before hundreds of page navigations
    // accumulate compositor state in the shared headless tab.
    const specialChecks = [
      () => menuChecks(client, server.baseUrl, "index.html", ++navigationId),
      () => menuChecks(client, server.baseUrl, "presidents/franklin-d-roosevelt.html", ++navigationId),
      () => noJavaScriptChecks(client, server.baseUrl),
      () => explorerFailureCheck(client, server.baseUrl),
      () => representativeInteractionChecks(client, server.baseUrl),
      () => touchScrollChecks(client, server.baseUrl),
      () => textZoomChecks(client, server.baseUrl),
      () => landscapeChecks(client, server.baseUrl),
      () => preferenceChecks(client, server.baseUrl),
    ];
    for (const check of specialChecks) {
      try {
        await check();
      } catch (error) {
        failures.push(error.message || String(error));
      }
    }
    if (!SPECIAL_ONLY) {
      for (const page of pages) {
        for (const viewport of VIEWPORTS) {
        navigationId += 1;
        await emulate(client, viewport);
        await client.send("Emulation.setScriptExecutionDisabled", { value: false });
        consoleErrors = [];
        localResourceErrors = [];
        await navigate(client, `${server.baseUrl}/${page}?audit=${navigationId}`);
        const layout = await evaluate(client, `${LAYOUT_EXPRESSION}(${viewport.width <= 760})`);
        const context = `${page} at ${viewport.width}×${viewport.height}`;
        if (layout.documentWidth > layout.viewport + 1) {
          failures.push(`${context} overflows the document: ${layout.documentWidth}px`);
        }
        if (viewport.width <= 760) {
          if (Math.abs(layout.nav.height - 56) > 1 || !layout.nav.mobileVisible || layout.nav.desktopVisible) {
            failures.push(`${context} does not use the 56px phone navigation: ${JSON.stringify(layout.nav)}`);
          }
          if (layout.undersizedTargets.length) {
            failures.push(`${context} has targets below 44px: ${JSON.stringify(layout.undersizedTargets)}`);
          }
          if (layout.tinyText.length) {
            failures.push(`${context} has text below the phone minimum: ${JSON.stringify(layout.tinyText)}`);
          }
        } else if (layout.nav.mobileVisible || !layout.nav.desktopVisible) {
          failures.push(`${context} leaked the phone navigation into the desktop boundary`);
        }
        if (layout.clippedSubstantive.length) {
          failures.push(`${context} clips content outside an accessible local scroller: ${JSON.stringify(layout.clippedSubstantive)}`);
        }
        if (layout.unnamedScrollers.length) {
          failures.push(`${context} has unnamed local scrollers: ${JSON.stringify(layout.unnamedScrollers.slice(0, 8))}`);
        }
        if (layout.unreachableScrollers.length) {
          failures.push(`${context} has unreachable local scrollers: ${JSON.stringify(layout.unreachableScrollers.slice(0, 8))}`);
        }
        if (layout.blankFigures) failures.push(`${context} has ${layout.blankFigures} blank substantive figures`);
        if (consoleErrors.length) failures.push(`${context} console errors: ${consoleErrors.join(" | ")}`);
        if (localResourceErrors.length) failures.push(`${context} resource errors: ${localResourceErrors.join(" | ")}`);
        if (viewport.name === "phone-390") {
          const focus = await focusCheck(client);
          if (["", "BODY", "HTML"].includes(focus.tag)
              || ((focus.outlineStyle === "none" || focus.outlineWidth < 2)
                && (!focus.boxShadow || focus.boxShadow === "none"))) {
            failures.push(`${context} lacks a visible first keyboard focus target: ${JSON.stringify(focus)}`);
          }
          const anchor = await anchorCheck(client);
          if (anchor && anchor.top + 1 < anchor.stickyBottom) {
            failures.push(`${context} obscures #${anchor.id}: top ${anchor.top.toFixed(1)}, sticky ${anchor.stickyBottom.toFixed(1)}`);
          }
        }
          observations.push({ page, viewport: viewport.name, documentWidth: layout.documentWidth });
        }
      }
    }

    if (DESKTOP_BASELINE) {
      try {
        await desktopGeometryChecks(client, server.baseUrl, DESKTOP_BASELINE, failures);
      } catch (error) {
        failures.push(error.message || String(error));
      }
    }
  } finally {
    if (client) client.close();
    await stopChild(chrome);
    await server.close();
    await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  }

  if (failures.length) {
    const category = (failure) => {
      if (failure.includes("targets below")) return "touch targets";
      if (failure.includes("text below")) return "type minimum";
      if (failure.includes("clips content")) return "clipping";
      if (failure.includes("overflows the document")) return "page overflow";
      if (failure.includes("56px phone navigation")) return "phone navigation";
      if (failure.includes("desktop boundary")) return "desktop boundary";
      if (failure.includes("at desktop")) return "desktop lock";
      if (failure.includes("unnamed local scrollers")) return "scroller naming";
      if (failure.includes("unreachable local scrollers")) return "scroller reachability";
      if (failure.includes("console errors")) return "console";
      if (failure.includes("resource errors")) return "resources";
      if (failure.includes("obscures #")) return "anchors";
      return "interaction/other";
    };
    const categoryCounts = {};
    for (const failure of failures) categoryCounts[category(failure)] = (categoryCounts[category(failure)] || 0) + 1;
    const shown = failures.slice(0, 48);
    const diagnosticTail = failures.filter((failure) => !shown.includes(failure)
      && category(failure) === "interaction/other");
    console.error(`Mobile site audit failed (${failures.length}): ${JSON.stringify(categoryCounts)}\n- ${shown.join("\n- ")}`);
    if (failures.length > shown.length) console.error(`- … ${failures.length - shown.length} more failure(s)`);
    if (diagnosticTail.length) console.error(`Interaction diagnostics:\n- ${diagnosticTail.join("\n- ")}`);
    process.exitCode = 1;
  } else {
    console.log(
      `Mobile site audit passed: ${pages.length} pages × ${VIEWPORTS.length} viewports; `
      + "phone navigation, overflow, touch targets, type, local scrollers, anchors, focus, "
      + "no-JavaScript evidence, request failure, resources, console"
      + (DESKTOP_BASELINE ? ", and desktop geometry checked." : " checked."),
    );
    console.log(JSON.stringify({ pages: pages.length, observations: observations.length }));
  }
}

await audit();
process.exit(process.exitCode || 0);
