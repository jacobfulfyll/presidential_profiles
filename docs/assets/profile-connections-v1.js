"use strict";

export const PROFILE_CONNECTIONS_REQUIRED_CONFIG = Object.freeze([
  "indexUrl", "shardUrl", "presidentProfileId"
]);

const NS = "http://www.w3.org/2000/svg";
const INDEX_SCHEMA = "president-profile-context-index-v1";
const SHARD_SCHEMA = "president-profile-context-president-v1";

function array(value) { return Array.isArray(value) ? value : []; }
function object(value) { return value && typeof value === "object" && !Array.isArray(value) ? value : {}; }
function finite(value) { const number = Number(value); return Number.isFinite(number) ? number : 0; }
function percent(value) { return `${(finite(value) * 100).toFixed(2)}%`; }
function byId(rows, key) { return new Map(array(rows).map(row => [String(row[key]), row])); }

function safeDataUrl(value, kind, doc) {
  const url = new URL(String(value || ""), doc.baseURI);
  if (url.origin !== doc.location.origin || url.search || url.hash) throw new Error("unsafe context URL");
  const path = url.pathname;
  const valid = kind === "index"
    ? /\/data\/profile-context\/index_v1\.json$/.test(path)
    : /\/data\/profile-context\/presidents\/[a-z0-9]+(?:-[a-z0-9]+)*_v1\.json$/.test(path);
  if (!valid) throw new Error("context URL is outside the governed projection");
  return url.href;
}

function make(doc, tag, className, value) {
  const node = doc.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = String(value);
  return node;
}

function svg(doc, tag, attrs = {}) {
  const allowed = new Set(["viewBox", "role", "aria-label", "aria-pressed", "tabindex", "focusable", "class", "x", "y", "x1", "y1",
    "x2", "y2", "cx", "cy", "r", "d", "markerWidth", "markerHeight", "refX", "refY", "orient",
    "markerUnits", "id", "marker-end"]);
  const node = doc.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => { if (allowed.has(key)) node.setAttribute(key, String(value)); });
  return node;
}

function textSvg(doc, label, x, y, className) {
  const node = svg(doc, "text", {x, y, class: className});
  node.textContent = String(label || "");
  return node;
}

function setRelation(node, relation) { node.dataset.pcRelation = relation; }
function edgeWidth(value) { return 1.25 + Math.min(Math.max(finite(value), 0), 1) * 26; }
function topicRadius(value) { return Math.sqrt(196 + Math.min(Math.max(finite(value), 0), 1) * 1900); }
function presidentName(row) { return String(row?.president_display_name || row?.president_name || "Unknown president"); }

function resolvedTopicSelection(selection, profileId, topics, presidents, edges) {
  const nodeIds = array(selection.node_ids).map(String), edgeIds = array(selection.edge_ids).map(String);
  if (nodeIds.some(id => id !== profileId && !topics.has(id) && !presidents.has(id))
      || edgeIds.some(id => !edges.has(id))) throw new Error("unknown precomputed topic selection ID");
  const topicIds = array(selection.topic_node_ids).length ? array(selection.topic_node_ids).map(String)
    : nodeIds.filter(id => topics.has(id));
  const peerIds = array(selection.peer_president_profile_ids).length
    ? array(selection.peer_president_profile_ids).map(String)
    : nodeIds.filter(id => id !== profileId && presidents.has(id));
  const focalIds = array(selection.focal_edge_ids).length ? array(selection.focal_edge_ids).map(String)
    : edgeIds.filter(id => String(edges.get(id)?.president_profile_id) === profileId);
  const focalSet = new Set(focalIds);
  const peerEdgeIds = array(selection.peer_edge_ids).length ? array(selection.peer_edge_ids).map(String)
    : edgeIds.filter(id => !focalSet.has(id));
  const mobileIds = array(selection.mobile_topic_peer_node_ids).map(String);
  if (new Set([...focalIds, ...peerEdgeIds]).size !== new Set(edgeIds).size)
    throw new Error("typed topic selection does not cover its precomputed edges");
  if (mobileIds.length > 6 || mobileIds.some(id => !topicIds.includes(id) && !peerIds.includes(id)))
    throw new Error("invalid precomputed mobile topic selection");
  return {node_ids: nodeIds, edge_ids: edgeIds, topic_node_ids: topicIds,
    peer_president_profile_ids: peerIds, focal_edge_ids: focalIds, peer_edge_ids: peerEdgeIds,
    mobile_topic_peer_node_ids: mobileIds};
}

function relationButton(doc, relation, title, summary) {
  const item = make(doc, "li", "");
  const button = make(doc, "button", "pc-relationship-button");
  button.type = "button"; button.setAttribute("aria-pressed", "false"); setRelation(button, relation);
  button.append(make(doc, "strong", "", title), make(doc, "span", "", summary));
  item.append(button); return item;
}

function buildTopicSvg(doc, profileId, displayName, selection, topics, presidents, edges, observed) {
  const graphic = svg(doc, "svg", {class: "pc-network-svg", viewBox: "0 0 720 560", role: "group",
    "aria-label": `Actual-speaker topic ego network for ${displayName}`, focusable: "false"});
  const resolved = resolvedTopicSelection(selection, profileId, topics, presidents, edges);
  const topicIds = resolved.topic_node_ids, peerIds = resolved.peer_president_profile_ids;
  const focalIds = resolved.focal_edge_ids, peerEdgeIds = resolved.peer_edge_ids;
  const relationByTopic = new Map(topicIds.map((id, index) => [id, `t${index}`]));
  const mobileIds = new Set(resolved.mobile_topic_peer_node_ids.length
    ? resolved.mobile_topic_peer_node_ids : [...topicIds, ...peerIds].slice(0, 6));
  const center = [360, 280], topicPositions = new Map(), peerPositions = new Map();
  topicIds.forEach((id, index) => { const angle = -Math.PI / 2 + Math.PI * 2 * index / Math.max(1, topicIds.length);
    topicPositions.set(id, [center[0] + Math.cos(angle) * 132, center[1] + Math.sin(angle) * 132]); });
  peerIds.forEach((id, index) => { const angle = -Math.PI / 2 + Math.PI * 2 * index / Math.max(1, peerIds.length);
    peerPositions.set(id, [center[0] + Math.cos(angle) * 246, center[1] + Math.sin(angle) * 224]); });
  const focalByTopic = new Map();
  focalIds.forEach(id => {
    const edge = edges.get(id); if (!edge) throw new Error("unknown focal topic edge");
    const topicId = String(edge.topic_id), position = topicPositions.get(topicId);
    if (!position || String(edge.president_profile_id) !== profileId) throw new Error("invalid focal topic selection");
    focalByTopic.set(topicId, edge);
    const line = svg(doc, "line", {class: `pc-topic-edge${observed ? " pc-observed-edge" : ""}${mobileIds.has(topicId) ? "" : " pc-mobile-hidden"}`,
      x1: center[0], y1: center[1], x2: position[0], y2: position[1]});
    line.style.setProperty("--pc-edge-width", `${edgeWidth(edge.speaker_paragraph_share)}px`);
    setRelation(line, relationByTopic.get(topicId)); graphic.append(line);
  });
  peerEdgeIds.forEach(id => {
    const edge = edges.get(id); if (!edge) throw new Error("unknown peer topic edge");
    const topicId = String(edge.topic_id), peerId = String(edge.president_profile_id);
    const start = peerPositions.get(peerId), end = topicPositions.get(topicId);
    if (!start || !end) throw new Error("invalid peer topic selection");
    const line = svg(doc, "line", {class: `pc-topic-edge pc-peer-edge${mobileIds.has(peerId) && mobileIds.has(topicId) ? "" : " pc-mobile-hidden"}`,
      x1: start[0], y1: start[1], x2: end[0], y2: end[1]});
    line.style.setProperty("--pc-edge-width", `${edgeWidth(edge.speaker_paragraph_share)}px`);
    setRelation(line, relationByTopic.get(topicId)); graphic.append(line);
  });
  topicIds.forEach(id => {
    const topic = topics.get(id), edge = focalByTopic.get(id), position = topicPositions.get(id);
    if (!topic || !edge) throw new Error("unknown topic node");
    const label = String(topic.topic_label || id);
    const group = svg(doc, "g", {class: `pc-topic-node pc-topic-control${mobileIds.has(id) ? "" : " pc-mobile-hidden"}`,
      role: "button", tabindex: "0", "aria-pressed": "false",
      "aria-label": `Select ${label} shared-topic paths, ${percent(edge.speaker_paragraph_share)} of eligible actual-speaker paragraphs`});
    setRelation(group, relationByTopic.get(id));
    const radius = topicRadius(edge.speaker_paragraph_share);
    group.append(svg(doc, "circle", {class: "pc-topic-hit-area", cx: position[0], cy: position[1], r: Math.max(22, radius)}),
      svg(doc, "circle", {cx: position[0], cy: position[1], r: radius}),
      textSvg(doc, label, position[0], position[1] + 4, "pc-svg-label pc-topic-label"));
    graphic.append(group);
  });
  peerIds.forEach(id => {
    const row = presidents.get(id), position = peerPositions.get(id); if (!row) throw new Error("unknown peer president");
    const group = svg(doc, "g", {class: `pc-president-node pc-peer-node${mobileIds.has(id) ? "" : " pc-mobile-hidden"}`});
    group.dataset.pcRelations = peerEdgeIds.map(edgeId => edges.get(edgeId)).filter(edge =>
      edge && String(edge.president_profile_id) === id).map(edge => relationByTopic.get(String(edge.topic_id))).join(" ");
    group.append(svg(doc, "circle", {cx: position[0], cy: position[1], r: 22}),
      textSvg(doc, presidentName(row), position[0], position[1] + 38, "pc-svg-label pc-peer-label")); graphic.append(group);
  });
  graphic.append(svg(doc, "circle", {class: "pc-focal-node", cx: center[0], cy: center[1], r: 22}),
    textSvg(doc, displayName, center[0], center[1] + 4, "pc-svg-label pc-focal-label"));
  return graphic;
}

function renderTopic(root, index, shard, mode) {
  const doc = root.ownerDocument, selection = object(object(shard.topic_selections)[mode]);
  const topics = byId(index.level1_topics, "topic_id"), presidents = byId(index.presidents, "president_profile_id");
  const edges = byId([...array(index.topic_edges), ...array(shard.topic_edges)], "edge_id");
  const profileId = String(root.dataset.presidentProfileId), displayName = presidentName(presidents.get(profileId));
  const resolved = resolvedTopicSelection(selection, profileId, topics, presidents, edges);
  const stage = root.querySelector("[data-pc-topic-stage]"), oldGraphic = stage?.querySelector("svg");
  if (!stage || !oldGraphic) throw new Error("topic stage is unavailable");
  oldGraphic.replaceWith(buildTopicSvg(doc, profileId, displayName, selection, topics, presidents, edges, mode === "observed_thin"));
  const list = root.querySelector("[data-pc-topic-list]");
  if (!list) throw new Error("topic semantic controls are unavailable");
  list.replaceChildren();
  const focalByTopic = new Map(resolved.focal_edge_ids.map(id => {
    const edge = edges.get(String(id)); return [String(edge?.topic_id), edge]; }));
  const peerEdges = resolved.peer_edge_ids.map(id => edges.get(String(id))).filter(Boolean);
  resolved.topic_node_ids.forEach((topicId, topicIndex) => {
    const topic = topics.get(topicId), focal = focalByTopic.get(topicId); if (!topic || !focal) throw new Error("invalid topic selection");
    const peers = peerEdges.filter(edge => String(edge.topic_id) === topicId);
    const relation = `t${topicIndex}`;
    list.append(relationButton(doc, relation, String(topic.topic_label || topicId),
      `${percent(focal.speaker_paragraph_share)} of eligible actual-speaker paragraphs · ${peers.length} comparison presidents`));
  });
}

export async function enhanceProfileConnections(root, config) {
  if (!root || typeof root.querySelector !== "function") throw new TypeError("Connections root is required");
  if (root.dataset.pcEnhanced === "true") return root.__profileConnectionsController;
  const options = object(config);
  PROFILE_CONNECTIONS_REQUIRED_CONFIG.forEach(key => { if (!options[key]) throw new TypeError(`Missing ${key}`); });
  const doc = root.ownerDocument;
  const indexUrl = safeDataUrl(options.indexUrl, "index", doc), shardUrl = safeDataUrl(options.shardUrl, "shard", doc);
  const profileId = String(options.presidentProfileId);
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(profileId) || profileId !== root.dataset.presidentProfileId)
    throw new Error("Connections president identity mismatch");
  const fetchImpl = options.fetchImpl || globalThis.fetch;
  if (typeof fetchImpl !== "function") throw new TypeError("fetch is unavailable");
  const status = root.querySelector("[data-pc-status]"), retry = root.querySelector("[data-pc-retry]");
  let loaded = null, loading = null, retryUsed = false, pendingAction = null;
  let pinnedKey = null, pinnedControl = null;

  const compactStage = root.querySelector("[data-pc-topic-stage]"),
    compactList = root.querySelector("[data-pc-topic-list]");
  const compactGraphic = compactStage?.querySelector("svg")?.cloneNode(true);
  const compactListChildren = compactList ? [...compactList.childNodes].map(node => node.cloneNode(true)) : [];
  if (!compactStage || !compactList || !compactGraphic)
    throw new Error("compact topic view is unavailable");

  function announce(message) { if (status) status.textContent = message; }
  function payloadValid(index, shard) {
    if (index?.schema_version !== INDEX_SCHEMA || shard?.schema_version !== SHARD_SCHEMA
        || String(shard?.president?.president_profile_id) !== profileId) throw new Error("unsupported Connections payload");
  }
  async function requestData(isRetry) {
    if (loaded) return loaded;
    if (loading) return loading;
    if (isRetry) retryUsed = true;
    announce(isRetry ? "Retrying Connections data…" : "Loading expanded Connections data…");
    loading = Promise.all([fetchImpl(indexUrl, {credentials: "same-origin"}),
      fetchImpl(shardUrl, {credentials: "same-origin"})])
      .then(async responses => {
        if (responses.some(response => !response.ok)) throw new Error("Connections request failed");
        const payloads = await Promise.all(responses.map(response => response.json()));
        payloadValid(payloads[0], payloads[1]); loaded = {index: payloads[0], shard: payloads[1]};
        if (retry) retry.hidden = true; announce("Expanded Connections data loaded."); return loaded;
      }).catch(error => {
        announce(retryUsed ? "Connections data still could not load; the complete server-rendered fallback remains available."
          : "Connections data could not load; the complete server-rendered fallback remains available. Retry is available once.");
        if (retry) retry.hidden = retryUsed; throw error;
      }).finally(() => { loading = null; });
    return loading;
  }

  function showRelation(key) {
    root.querySelectorAll("[data-pc-relation]").forEach(node => {
      const active = node.dataset.pcRelation === key;
      node.classList.toggle("is-pinned", active);
    });
    root.querySelectorAll("[data-pc-relations]").forEach(node => {
      const active = String(node.dataset.pcRelations || "").split(/\s+/).includes(key);
      node.classList.toggle("is-pinned", active);
    });
  }
  function clearPin(restoreFocus) {
    const control = pinnedControl; pinnedKey = null; pinnedControl = null;
    root.querySelectorAll("[data-pc-relation][aria-pressed]").forEach(node => node.setAttribute("aria-pressed", "false"));
    showRelation(null); if (restoreFocus && control) control.focus(); announce("Topic emphasis cleared.");
  }
  function activateRelation(control) {
    const key = control.dataset.pcRelation; if (!key) return;
    if (pinnedKey === key) { clearPin(false); return; }
    pinnedKey = key; pinnedControl = control;
    root.querySelectorAll("[data-pc-relation][aria-pressed]").forEach(node =>
      node.setAttribute("aria-pressed", node.dataset.pcRelation === key ? "true" : "false"));
    showRelation(key); announce("Topic paths emphasized. Select again or press Escape to clear.");
  }

  function restoreCompactTopic() {
    const stage = root.querySelector("[data-pc-topic-stage]"), oldGraphic = stage?.querySelector("svg");
    const list = root.querySelector("[data-pc-topic-list]");
    if (!oldGraphic || !list) throw new Error("topic semantic controls are unavailable");
    oldGraphic.replaceWith(compactGraphic.cloneNode(true));
    list.replaceChildren(...compactListChildren.map(node => node.cloneNode(true)));
  }
  function setToggleState(button, expanded) {
    button.setAttribute("aria-expanded", String(expanded));
    if (!expanded) button.textContent = button.dataset.pcExpand === "topic-observed"
      ? "Show observed thin-record topics" : "Show expanded topic network";
    else button.textContent = button.dataset.pcExpand === "topic-observed"
      ? "Hide observed thin-record topics" : "Show compact topic network";
  }
  function applyExpansion(action, payload, button) {
    const mode = action === "topic-observed" ? "observed_thin" : "expanded";
    clearPin(false); renderTopic(root, payload.index, payload.shard, mode);
    setToggleState(button, true); pendingAction = null;
    announce(action === "topic-observed"
      ? "Observed thin-record topics are shown."
      : "Expanded topic relationships are shown. Use Show compact topic network to return.");
  }
  async function toggleTopic(button) {
    if (button.getAttribute("aria-expanded") === "true") {
      clearPin(false); restoreCompactTopic(); setToggleState(button, false); pendingAction = null;
      announce(button.dataset.pcExpand === "topic-observed"
        ? "Observed thin-record topics are hidden." : "Compact topic relationships are shown.");
      return;
    }
    const action = String(button.dataset.pcExpand || "");
    if (action !== "topic-expanded" && action !== "topic-observed") return;
    const restoreFocus = doc.activeElement === button;
    pendingAction = action; button.disabled = true;
    try { applyExpansion(action, await requestData(false), button); }
    catch (_) { /* The status and one retry preserve the server fallback. */ }
    finally { button.disabled = false; if (restoreFocus) button.focus(); }
  }

  root.addEventListener("click", event => {
    const relation = event.target.closest?.("[data-pc-relation][aria-pressed]");
    if (relation && root.contains(relation)) { activateRelation(relation); return; }
    const button = event.target.closest?.("[data-pc-expand]");
    if (button && root.contains(button)) toggleTopic(button);
  });
  root.addEventListener("keydown", event => {
    const relation = event.target.closest?.("[data-pc-relation][aria-pressed]");
    if (relation && root.contains(relation) && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault(); activateRelation(relation); return;
    }
    if (event.key === "Escape" && pinnedKey) { event.preventDefault(); clearPin(true); }
  });
  if (retry) retry.addEventListener("click", async () => {
    if (retryUsed || !pendingAction) return;
    const action = pendingAction, button = root.querySelector(`[data-pc-expand="${action}"]`);
    if (!button) return;
    button.disabled = true;
    try { applyExpansion(action, await requestData(true), button);
    } catch (_) { /* A second failure is final and the fallback remains. */ }
    finally { button.disabled = false; button.focus(); }
  });

  const invocationSwitch = root.querySelector("[data-pc-invocation-switch]");
  const invocationDirections = root.querySelector("[data-pc-invocation-directions]");
  function showInvocationDirection(direction, shouldAnnounce) {
    if (!invocationDirections || (direction !== "outgoing" && direction !== "incoming")) return;
    invocationDirections.querySelectorAll("[data-pc-invocation-direction]").forEach(panel => {
      panel.hidden = panel.dataset.pcInvocationDirection !== direction;
    });
    if (shouldAnnounce) announce(direction === "outgoing"
      ? "Most invoked former presidents shown." : "Most frequent later invokers shown.");
  }
  if (invocationSwitch && invocationDirections) {
    const initialDirection = String(invocationDirections.dataset.defaultInvocationDirection || "outgoing");
    showInvocationDirection(initialDirection, false); invocationSwitch.hidden = false;
    invocationSwitch.addEventListener("change", event => {
      const choice = event.target.closest?.("[data-pc-invocation-choice]");
      if (choice?.checked) showInvocationDirection(String(choice.value), true);
    });
    invocationSwitch.addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      const choices = [...invocationSwitch.querySelectorAll("[data-pc-invocation-choice]")];
      const current = event.target.closest?.("[data-pc-invocation-choice]");
      if (!current || !choices.includes(current)) return;
      event.preventDefault();
      const currentIndex = choices.indexOf(current);
      const nextIndex = event.key === "Home" ? 0 : event.key === "End" ? choices.length - 1
        : event.key === "ArrowLeft" ? (currentIndex - 1 + choices.length) % choices.length
        : (currentIndex + 1) % choices.length;
      const next = choices[nextIndex]; next.checked = true; next.focus();
      showInvocationDirection(String(next.value), true);
    });
  }
  root.querySelectorAll("[data-pc-enhancement-controls]").forEach(node => { node.hidden = false; });
  root.classList.add("is-enhanced"); root.dataset.pcEnhanced = "true";
  const controller = Object.freeze({clearPin: () => clearPin(false)});
  root.__profileConnectionsController = controller; return controller;
}
