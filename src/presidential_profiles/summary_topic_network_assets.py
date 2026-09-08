"""Scoped assets for the reusable topic relationship browser."""

SUMMARY_TOPIC_NETWORK_CSS = r"""
.summary-topic-section{overflow:clip}.summary-topic-population{display:flex;flex-wrap:wrap;
  gap:8px;align-items:baseline;margin:12px 0 18px;color:var(--muted);font-size:.84rem}
.summary-topic-population strong{color:var(--ink);font:700 1.14rem/1 Georgia,serif}
.summary-topic-browser{display:grid;grid-template-columns:minmax(0,1fr);gap:8px;margin-top:18px}
.summary-topic-controls,.summary-topic-stage{min-width:0;margin:0;padding:12px;border:1px solid var(--border);
  border-radius:12px;background:var(--surface)}.summary-topic-controls{display:grid;gap:8px}
.summary-topic-controls h3{margin:0;font-size:1rem}.summary-topic-control-heading{display:flex;min-height:44px;
  align-items:center;justify-content:space-between;gap:12px}.summary-topic-picker{display:flex;flex-wrap:nowrap;
  gap:7px;max-width:100%;padding:2px 0 7px;overflow-x:auto;overscroll-behavior-inline:contain}
.summary-topic-picker button{flex:0 0 auto;min-width:155px;min-height:44px;padding:8px 10px;border:1px solid #c9b39f;
  border-radius:7px;background:#fbfaf7;color:#543a27;font:700 .7rem/1.24 inherit;cursor:pointer}
.summary-topic-picker button[aria-pressed="true"]{border-color:#704421;background:#704421;color:#fff;
  box-shadow:inset 0 0 0 2px #f0d7bd}.summary-topic-picker button:disabled{cursor:not-allowed;opacity:.38}
.summary-topic-selection-tools{display:flex;min-height:44px;align-items:center;gap:10px;white-space:nowrap}
.summary-topic-selection-tools strong{font-size:.7rem}.summary-topic-selection-tools button,
.summary-topic-president-row>button{min-height:44px;padding:8px 11px;
  border:1px solid #b8895d;border-radius:8px;background:#fbf7f1;color:#5d351b;font:700 .72rem/1.25 inherit;
  cursor:pointer}.summary-topic-selection-tools button:disabled{cursor:not-allowed;opacity:.5}
.summary-topic-president-filter{min-width:0;padding-top:8px;border-top:1px solid var(--grid)}
.summary-topic-president-line{display:flex;align-items:center;gap:12px}.summary-topic-president-filter label{flex:0 0 auto;
  color:#51351f;font-size:.7rem;font-weight:800;text-transform:uppercase;letter-spacing:.035em}
.summary-topic-president-filter select{flex:0 1 300px;min-width:0;width:300px;min-height:44px;padding:7px 30px 7px 9px;
  border:1px solid #bfa58e;border-radius:7px;background:#fff;color:var(--ink);font:inherit}
.summary-topic-president-row{display:flex;min-width:0;align-items:center;gap:7px;margin-top:7px}
.summary-topic-president-row>button{flex:0 0 auto}.summary-topic-president-chips{display:flex;min-width:0;gap:6px;
  padding:1px 0 6px;overflow-x:auto;overscroll-behavior-inline:contain}.summary-topic-president-chips button{display:flex;
  flex:0 0 auto;
  min-height:44px;align-items:center;justify-content:space-between;gap:8px;padding:7px 9px;border:1px solid #9b663b;
  border-radius:7px;background:#fffaf3;color:#4b2e1b;font:700 .68rem/1.25 inherit;cursor:pointer}
.summary-topic-president-chips button::after{content:"Remove";color:var(--muted);font-size:.56rem;font-weight:650;
  text-transform:uppercase}.summary-topic-status{position:absolute;width:1px;height:1px;padding:0;margin:-1px;
  overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}.summary-topic-picker button:focus-visible,
.summary-topic-selection-tools button:focus-visible,.summary-topic-president-filter select:focus-visible,
.summary-topic-president-filter button:focus-visible,.summary-topic-download a:focus-visible{
  outline:3px solid #d08b45;outline-offset:2px}
.summary-topic-stage{width:100%;padding:10px}.summary-topic-stage-tools{display:grid;
  grid-template-columns:minmax(0,1fr) minmax(260px,.72fr);gap:8px;margin-bottom:8px}.summary-topic-stage-tools p{
  min-width:0;margin:0;padding:8px 10px;border-radius:7px;background:#f8f1e8;color:#5b4636;font-size:.69rem;
  line-height:1.4}.summary-topic-stage-tools p:first-child{border-left:3px solid #6c3516;color:#432b1a;font-weight:710}
.summary-topic-stage-tools strong{color:#5d351b}.summary-topic-stage svg{display:block;width:100%;height:auto;
  min-height:500px;border-radius:9px;background:radial-gradient(circle at center,#fff 0,#f9f6f0 48%,#efe5d7 100%)}
.summary-topic-stage figcaption{margin:8px 5px 2px;color:var(--muted);font-size:.68rem;line-height:1.45}
.summary-topic-edge{stroke:#86684e;stroke-linecap:round;opacity:.08;pointer-events:none}
.summary-topic-edge.is-active{stroke:#6c3516;opacity:.76}.summary-topic-anchor{cursor:default}
.summary-topic-anchor .anchor-halo{fill:#f3e7d8;stroke:#d3b393;stroke-width:1}
.summary-topic-anchor .anchor-focus{fill:none;stroke:transparent}.summary-topic-anchor .anchor-shape{fill:#fffaf3;
  stroke:#8d562c;stroke-width:2.2}.summary-topic-anchor text{fill:#4a2e1b;font:760 11px/1.2 system-ui,sans-serif;
  text-anchor:middle;pointer-events:none}.summary-topic-anchor.is-active .anchor-shape{fill:#f0d9bf;stroke:#5e3217;
  stroke-width:3.5}.summary-topic-anchor:focus{outline:none}.summary-topic-anchor:focus .anchor-focus{fill:none;
  stroke:#165a78;stroke-width:4}.summary-topic-president{cursor:pointer}.summary-topic-president .president-ring{
  fill:#fff;stroke:#315f78;stroke-width:2}.summary-topic-president .president-focus{fill:none;stroke:transparent}
.summary-topic-president .president-image{pointer-events:none}.summary-topic-president.is-active .president-ring{
  stroke:#6c3516;stroke-width:4}.summary-topic-president:focus{outline:none}.summary-topic-president:focus .president-focus{
  fill:none;stroke:#165a78;stroke-width:4}.summary-topic-president-label{fill:#2c231c;
  font:720 10.5px/1.2 system-ui,sans-serif;paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round;
  text-anchor:middle;pointer-events:none}.summary-topic-empty{fill:#6e6258;font:650 16px/1.4 system-ui,sans-serif;
  text-anchor:middle}.summary-topic-gravity-center{fill:#8a6a50;opacity:.35}.summary-topic-gravity-label{fill:#77685c;
  font:650 10px/1 system-ui,sans-serif}.summary-topic-download{margin:12px 0 0;font-size:.73rem}
.summary-topic-download a{display:inline-flex;min-height:44px;align-items:center}.summary-topic-section [hidden]{display:none!important}
@media(max-width:760px){.summary-topic-control-heading{align-items:stretch;flex-direction:column}
  .summary-topic-selection-tools{width:100%;justify-content:space-between}.summary-topic-president-line{
    align-items:stretch;flex-direction:column;gap:7px}.summary-topic-president-filter select{width:100%;flex-basis:auto}
  .summary-topic-picker button,.summary-topic-selection-tools button,
  .summary-topic-president-row>button,.summary-topic-president-chips button{font-size:.875rem}
  .summary-topic-picker button,.summary-topic-selection-tools button,
  .summary-topic-president-row>button,.summary-topic-president-chips button,
  .summary-topic-president-filter select{min-height:44px;min-width:44px}
  .summary-topic-selection-tools strong,.summary-topic-president-filter label,
  .summary-topic-president-chips button::after,.summary-topic-stage figcaption,
  .summary-topic-download{font-size:.75rem}.summary-topic-stage-tools p{font-size:1rem;line-height:1.55}
  .summary-topic-stage{overflow-x:auto;overscroll-behavior-inline:contain;scrollbar-width:auto}
  .summary-topic-stage:has(.summary-topic-president)::before{content:"Swipe horizontally to inspect →";
    position:sticky;left:6px;z-index:3;display:block;width:max-content;margin:0 0 7px;padding:5px 8px;
    border-radius:999px;background:#eee7dc;color:#5d5144;font:750 .75rem/1.2 system-ui,sans-serif}
  .summary-topic-stage svg{min-height:410px}.summary-topic-stage:has(.summary-topic-president) svg{
    width:900px;min-width:900px}.summary-topic-empty{font-size:44px}
  .summary-topic-anchor text,.summary-topic-president-label{font-size:14px}
  .summary-topic-gravity-label{font-size:12px}.summary-topic-stage-tools{grid-template-columns:1fr}}
@media(max-width:390px){.summary-topic-control-heading{gap:7px}.summary-topic-selection-tools{gap:6px}
  .summary-topic-controls,.summary-topic-stage{padding:9px}.summary-topic-stage{padding:6px}}
@media(prefers-reduced-motion:reduce){.summary-topic-section *,.summary-topic-section *::before,
  .summary-topic-section *::after{scroll-behavior:auto!important;animation:none!important;transition:none!important}}
.reduced-motion .summary-topic-section *,.reduced-motion .summary-topic-section *::before,
.reduced-motion .summary-topic-section *::after{scroll-behavior:auto!important;animation:none!important;transition:none!important}
"""


SUMMARY_TOPIC_NETWORK_JS = r"""
(() => {
  "use strict";
  const root = document.querySelector("[data-summary-topic-browser]");
  if (!root) return;
  const svg = root.querySelector("[data-summary-topic-svg]");
  const status = root.querySelector("[data-summary-topic-status]");
  const picker = root.querySelector("[data-summary-topic-picker]");
  const selectionCount = root.querySelector("[data-summary-topic-selection-count]");
  const clearButton = root.querySelector("[data-summary-topic-clear]");
  const presidentFilter = root.querySelector("[data-summary-topic-president-filter]");
  const presidentRow = root.querySelector("[data-summary-topic-president-row]");
  const presidentChips = root.querySelector("[data-summary-topic-president-chips]");
  const presidentClear = root.querySelector("[data-summary-topic-president-clear]");
  const stageReadout = root.querySelector("[data-summary-topic-stage-readout]");
  const sizeKey = root.querySelector("[data-summary-topic-size-key]");
  const NS = "http://www.w3.org/2000/svg";
  const MAX_TOPICS = 8;
  const MAX_PRESIDENTS = 8;
  const state = {selectedTopicIds: [], selectedPresidentIds: [], previewPresidentId: null,
    pinnedPresidentId: null, previewTopicId: null};
  let index = null;
  let loading = null;
  const pct = value => value == null ? "N/A" : `${(Number(value) * 100).toFixed(2)}%`;
  const count = value => Number(value || 0).toLocaleString();
  const setStatus = message => { status.textContent = message; };
  const svgEl = (name, attrs = {}) => {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  };
  const textNode = (tag, value, className) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = value;
    return node;
  };
  const presidentForId = id => index?.presidents.find(row => row.president_profile_id === id);
  const presidentSupport = id => index?.president_support.find(row => row.president_profile_id === id);
  const topicForId = id => index?.level1_topics.find(row => row.topic_id === id);
  const activePresidentId = () => state.previewPresidentId || state.pinnedPresidentId;
  const selectedTopics = () => {
    const selected = new Set(state.selectedTopicIds);
    return (index?.level1_topics || []).filter(topic => selected.has(topic.topic_id));
  };

  async function ensureIndex() {
    if (index) return index;
    if (loading) return loading;
    setStatus("Loading the recurring all-corpus topic relationships…");
    loading = fetch(root.dataset.indexUrl, {credentials: "same-origin"})
      .then(response => {
        if (!response.ok) throw new Error(`index request failed (${response.status})`);
        return response.json();
      })
      .then(payload => {
        if (payload.schema_version !== "summary-actual-speaker-topic-network-index-v1"
            || payload.edge_width_measure !== "speaker_paragraph_share") throw new Error("unsupported projection");
        index = payload;
        renderAll();
        return payload;
      })
      .catch(error => {
        setStatus("The topic field could not load. The exact recurring-topic CSV remains available below.");
        throw error;
      })
      .finally(() => { loading = null; });
    return loading;
  }

  function qualifyingEdges() {
    if (!index || !state.selectedTopicIds.length) return [];
    const selected = new Set(state.selectedTopicIds);
    return index.edges.filter(edge => selected.has(edge.topic_id)
      && edge.default_visible === true && edge.president_support_status === "supported");
  }

  function presidentAggregates(applyPresidentFilter = true) {
    if (!index || !state.selectedTopicIds.length) return [];
    const presidentOrder = new Map(index.presidents.map(row => [row.president_profile_id, row.display_order]));
    const topicOrder = new Map(index.level1_topics.map(row => [row.topic_id, row.display_order]));
    const groups = new Map();
    qualifyingEdges().forEach(edge => {
      if (!groups.has(edge.president_profile_id)) groups.set(edge.president_profile_id, {
        president: presidentForId(edge.president_profile_id), edges: [], memberships: 0, maxShare: 0});
      const row = groups.get(edge.president_profile_id);
      row.edges.push(edge);
      row.memberships += Number(edge.topic_paragraph_count);
      row.maxShare = Math.max(row.maxShare, Number(edge.speaker_paragraph_share));
    });
    groups.forEach(row => row.edges.sort((a, b) => topicOrder.get(a.topic_id) - topicOrder.get(b.topic_id)));
    const rows = [...groups.values()].sort((a, b) => b.memberships - a.memberships || b.maxShare - a.maxShare
      || presidentOrder.get(a.president.president_profile_id) - presidentOrder.get(b.president.president_profile_id));
    if (!applyPresidentFilter || !state.selectedPresidentIds.length) return rows;
    const selected = new Set(state.selectedPresidentIds);
    return rows.filter(row => selected.has(row.president.president_profile_id));
  }

  function anchorPositions(topics, width, height) {
    const centerX = width / 2, centerY = height / 2;
    if (topics.length === 1) return [{x: centerX, y: 74}];
    if (topics.length === 2) return [{x: 105, y: centerY}, {x: width - 105, y: centerY}];
    const radiusX = width / 2 - 92, radiusY = height / 2 - 62;
    return topics.map((_, i) => {
      const angle = -Math.PI / 2 + Math.PI * 2 * i / topics.length;
      return {x: centerX + Math.cos(angle) * radiusX, y: centerY + Math.sin(angle) * radiusY};
    });
  }

  function deterministicSeed(value) {
    return [...String(value)].reduce((total, character, i) => total + character.charCodeAt(0) * (i + 1), 0);
  }

  function gravityLayout(aggregates, topics, mobile, sizeDomainMax) {
    const width = mobile ? 600 : 900, height = mobile ? 720 : 610;
    const centerX = width / 2, centerY = height / 2;
    const positions = anchorPositions(topics, width, height);
    const anchors = topics.map((topic, i) => ({...topic, ...positions[i]}));
    const anchorMap = new Map(anchors.map(row => [row.topic_id, row]));
    const shown = aggregates.slice(0, mobile ? 18 : 45);
    const maxMemberships = Math.max(1, Number(sizeDomainMax) || 0);
    const nodes = shown.map((aggregate, rank) => {
      const radius = Math.max(mobile ? 9 : 10,
        Math.sqrt(aggregate.memberships / maxMemberships) * (mobile ? 23 : 31));
      const pulls = aggregate.edges.map(edge => ({anchor: anchorMap.get(edge.topic_id), edge,
        weight: Number(edge.speaker_paragraph_share)})).filter(row => row.anchor && row.weight > 0);
      const weightTotal = pulls.reduce((total, row) => total + row.weight, 0);
      const semanticX = weightTotal ? pulls.reduce((total, row) => total + row.anchor.x * row.weight, 0)
        / weightTotal : centerX;
      const semanticY = weightTotal ? pulls.reduce((total, row) => total + row.anchor.y * row.weight, 0)
        / weightTotal : centerY;
      const blend = pulls.length === 1 ? .66 : .82;
      const seed = deterministicSeed(aggregate.president.president_profile_id);
      const angle = seed % 360 * Math.PI / 180;
      const jitter = 10 + seed % 21;
      const targetX = centerX + (semanticX - centerX) * blend + Math.cos(angle) * jitter;
      const targetY = centerY + (semanticY - centerY) * blend + Math.sin(angle) * jitter;
      return {...aggregate, rank, radius, targetX, targetY, x: targetX, y: targetY, pulls};
    }).sort((a, b) => b.radius - a.radius || a.president.display_order - b.president.display_order);
    if (nodes.length > 1) {
      const spanX = Math.max(1, Math.max(...nodes.map(node => node.targetX))
        - Math.min(...nodes.map(node => node.targetX)));
      const spanY = Math.max(1, Math.max(...nodes.map(node => node.targetY))
        - Math.min(...nodes.map(node => node.targetY)));
      const scaleX = Math.min(mobile ? 1.8 : 2.3, Math.max(1, (mobile ? 330 : 540) / spanX));
      const scaleY = Math.min(mobile ? 1.7 : 2.1, Math.max(1, (mobile ? 380 : 390) / spanY));
      nodes.forEach(node => {
        node.targetX = centerX + (node.targetX - centerX) * scaleX;
        node.targetY = centerY + (node.targetY - centerY) * scaleY;
        node.x = node.targetX;
        node.y = node.targetY;
      });
    }
    const keepInBounds = node => {
      const margin = node.radius + (mobile ? 6 : 8);
      node.x = Math.min(Math.max(node.x, margin), width - margin);
      node.y = Math.min(Math.max(node.y, margin), height - margin);
    };
    const separate = spring => {
      nodes.forEach(node => {
        node.x += (node.targetX - node.x) * spring;
        node.y += (node.targetY - node.y) * spring;
        anchors.forEach(anchor => {
          let dx = node.x - anchor.x, dy = node.y - anchor.y;
          let distance = Math.hypot(dx, dy);
          const minimum = node.radius + (mobile ? 64 : 78);
          if (distance >= minimum) return;
          if (distance < .01) {
            const a = deterministicSeed(node.president.president_profile_id + anchor.topic_id)
              % 360 * Math.PI / 180;
            dx = Math.cos(a); dy = Math.sin(a); distance = 1;
          }
          node.x += dx / distance * (minimum - distance);
          node.y += dy / distance * (minimum - distance);
        });
        keepInBounds(node);
      });
      nodes.forEach((left, i) => nodes.slice(i + 1).forEach(right => {
        let dx = right.x - left.x, dy = right.y - left.y;
        let distance = Math.hypot(dx, dy);
        const minimum = left.radius + right.radius + (mobile ? 12 : 18);
        if (distance >= minimum) return;
        if (distance < .01) {
          const a = deterministicSeed(left.president.president_profile_id + right.president.president_profile_id)
            % 360 * Math.PI / 180;
          dx = Math.cos(a); dy = Math.sin(a); distance = 1;
        }
        const shift = (minimum - distance) / 2;
        left.x -= dx / distance * shift; left.y -= dy / distance * shift;
        right.x += dx / distance * shift; right.y += dy / distance * shift;
        keepInBounds(left); keepInBounds(right);
      }));
    };
    for (let i = 0; i < 190; i += 1) separate(.012);
    for (let i = 0; i < 30; i += 1) separate(0);
    return {width, height, anchors, nodes};
  }

  function addWrappedSvgText(group, label, x, y, className, maxLength = 18, maxLines = 3) {
    const text = svgEl("text", {x, y, class: className || ""});
    const lines = [];
    let line = "";
    String(label).split(/\s+/).forEach(word => {
      if ((line + " " + word).trim().length > maxLength && line) { lines.push(line); line = word; }
      else line = (line + " " + word).trim();
    });
    if (line) lines.push(line);
    lines.slice(0, maxLines).forEach((value, i) => {
      const span = svgEl("tspan", {x, dy: i === 0 ? 0 : 13});
      span.textContent = value;
      text.append(span);
    });
    group.append(text);
  }

  function updateActiveStyles() {
    const presidentId = activePresidentId();
    const topicId = state.previewTopicId;
    root.querySelectorAll(".summary-topic-edge").forEach(node => {
      const active = topicId ? node.dataset.topicId === topicId : node.dataset.presidentId === presidentId;
      node.classList.toggle("is-active", active);
    });
    root.querySelectorAll(".summary-topic-president[data-president-id]").forEach(node => {
      node.classList.toggle("is-active", node.dataset.presidentId === presidentId);
    });
    root.querySelectorAll(".summary-topic-anchor[data-topic-id]").forEach(node => {
      node.classList.toggle("is-active", node.dataset.topicId === topicId);
    });
  }

  function updateStageReadout() {
    const presidentId = activePresidentId();
    const aggregate = presidentAggregates(false).find(row => row.president.president_profile_id === presidentId);
    if (aggregate) {
      const strongest = aggregate.edges.slice().sort((a, b) => Number(b.speaker_paragraph_share)
        - Number(a.speaker_paragraph_share))[0];
      stageReadout.textContent = `${aggregate.president.president_name}: ${count(aggregate.memberships)} `
        + `selected-topic paragraph memberships across ${aggregate.edges.length} topics. Strongest pull: `
        + `${topicForId(strongest.topic_id)?.topic_label}, ${pct(strongest.speaker_paragraph_share)}.`;
      return;
    }
    if (state.previewTopicId) {
      const connected = presidentAggregates().filter(row => row.edges.some(edge => edge.topic_id === state.previewTopicId));
      stageReadout.textContent = `${topicForId(state.previewTopicId)?.topic_label}: ${connected.length} `
        + "presidents meet the recurring threshold in the current view.";
      return;
    }
    stageReadout.textContent = state.selectedTopicIds.length
      ? "Hover, focus, or select a president to see its topic memberships and strongest pull."
      : "Choose topics, then hover, focus, or select a president to inspect it here.";
  }

  function updateSizeKey() {
    sizeKey.replaceChildren(textNode("strong", "Portrait area"));
    if (!state.selectedTopicIds.length || !index) {
      sizeKey.append(document.createTextNode(" = selected-topic paragraph memberships. One paragraph can count under more than one topic."));
      return;
    }
    const maximum = Math.max(0, ...presidentAggregates(false).map(row => row.memberships));
    sizeKey.append(document.createTextNode(` = selected-topic paragraph memberships. Largest in this selection: `
      + `${count(maximum)}. One paragraph can count under more than one topic.`));
  }

  function previewPresident(presidentId) {
    state.previewPresidentId = presidentId;
    updateActiveStyles();
    updateStageReadout();
  }

  function pinPresident(presidentId) {
    state.pinnedPresidentId = state.pinnedPresidentId === presidentId ? null : presidentId;
    state.previewPresidentId = null;
    updateActiveStyles();
    updateStageReadout();
  }

  function wirePresidentTarget(node, presidentId) {
    node.addEventListener("pointerenter", () => previewPresident(presidentId));
    node.addEventListener("pointerleave", () => previewPresident(null));
    node.addEventListener("focus", () => previewPresident(presidentId));
    node.addEventListener("blur", () => previewPresident(null));
    node.addEventListener("click", () => pinPresident(presidentId));
    node.addEventListener("keydown", event => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      pinPresident(presidentId);
    });
  }

  function wireTopicTarget(node, topicId) {
    const preview = value => {state.previewTopicId = value; updateActiveStyles(); updateStageReadout();};
    node.addEventListener("pointerenter", () => preview(topicId));
    node.addEventListener("pointerleave", () => preview(null));
    node.addEventListener("focus", () => preview(topicId));
    node.addEventListener("blur", () => preview(null));
  }

  function renderDiagram() {
    svg.replaceChildren();
    const title = svgEl("title", {id: "summary-topic-svg-title"});
    const desc = svgEl("desc", {id: "summary-topic-svg-desc"});
    title.textContent = "President topic-gravity field";
    svg.append(title, desc);
    const mobile = matchMedia("(max-width: 760px)").matches;
    const width = mobile ? 600 : 900, height = mobile ? 720 : 610;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    if (!state.selectedTopicIds.length || !index) {
      desc.textContent = "No topics are selected. Choose up to eight broad topics.";
      const prompt = svgEl("text", {x: width / 2, y: height / 2 - 8, class: "summary-topic-empty"});
      prompt.textContent = "Choose up to eight topics";
      const prompt2 = svgEl("text", {x: width / 2, y: height / 2 + 20, class: "summary-topic-empty"});
      prompt2.textContent = "Presidents will gather between them.";
      svg.append(prompt, prompt2);
      return;
    }
    const topics = selectedTopics();
    const aggregates = presidentAggregates();
    const allAggregates = presidentAggregates(false);
    const sizeDomainMax = Math.max(1, ...allAggregates.map(row => row.memberships));
    const layout = gravityLayout(aggregates, topics, mobile, sizeDomainMax);
    const anchorMap = new Map(layout.anchors.map(row => [row.topic_id, row]));
    desc.textContent = `${topics.length} selected topics anchor ${layout.nodes.length} displayed presidents. `
      + "President area shows selected-topic paragraph memberships.";
    const defs = svgEl("defs");
    layout.nodes.forEach(node => {
      const clip = svgEl("clipPath", {id: `summary-topic-clip-${node.president.president_profile_id}`});
      clip.append(svgEl("circle", {cx: node.x, cy: node.y, r: Math.max(1, node.radius - 1)}));
      defs.append(clip);
    });
    svg.append(defs);
    layout.nodes.forEach(node => node.edges.forEach(edge => {
      const anchor = anchorMap.get(edge.topic_id);
      if (!anchor) return;
      const widthValue = 1.5 + 5.5 * Math.max(0, Math.min(1, Number(edge.speaker_paragraph_share)));
      svg.append(svgEl("line", {x1: node.x, y1: node.y, x2: anchor.x, y2: anchor.y,
        class: "summary-topic-edge", "stroke-width": widthValue, "data-edge-id": edge.edge_id,
        "data-president-id": edge.president_profile_id, "data-topic-id": edge.topic_id}));
    }));
    svg.append(svgEl("circle", {cx: layout.width / 2, cy: layout.height / 2, r: 3,
      class: "summary-topic-gravity-center"}));
    const centerLabel = svgEl("text", {x: layout.width / 2 + 8, y: layout.height / 2 + 3,
      class: "summary-topic-gravity-label"});
    centerLabel.textContent = "shared pull";
    svg.append(centerLabel);
    layout.anchors.forEach(anchor => {
      const group = svgEl("g", {tabindex: "0", role: "img",
        "aria-label": `${anchor.topic_label}, selected topic anchor`, class: "summary-topic-anchor",
        "data-topic-id": anchor.topic_id});
      group.append(svgEl("rect", {x: anchor.x - 76, y: anchor.y - 32, width: 152, height: 64,
        rx: 13, class: "anchor-focus"}), svgEl("rect", {x: anchor.x - 72, y: anchor.y - 28,
        width: 144, height: 56, rx: 12, class: "anchor-halo"}), svgEl("rect", {
        x: anchor.x - 67, y: anchor.y - 23, width: 134, height: 46, rx: 10, class: "anchor-shape"}));
      addWrappedSvgText(group, anchor.topic_label, anchor.x, anchor.y - 5, "", mobile ? 16 : 18, 3);
      wireTopicTarget(group, anchor.topic_id);
      svg.append(group);
    });
    layout.nodes.forEach(node => {
      const presidentId = node.president.president_profile_id;
      const aria = `${node.president.president_name}: ${count(node.memberships)} selected-topic paragraph memberships `
        + `across ${node.edges.length} selected topics.`;
      const group = svgEl("g", {tabindex: "0", role: "button", "aria-label": aria,
        class: "summary-topic-president", "data-president-id": presidentId});
      group.append(svgEl("circle", {cx: node.x, cy: node.y, r: node.radius + 5, class: "president-focus"}),
        svgEl("circle", {cx: node.x, cy: node.y, r: node.radius + 1.5, class: "president-ring"}),
        svgEl("image", {href: `portraits/${presidentId}.png`, x: node.x - node.radius,
          y: node.y - node.radius, width: node.radius * 2, height: node.radius * 2,
          class: "president-image", "clip-path": `url(#summary-topic-clip-${presidentId})`,
          preserveAspectRatio: "xMidYMid slice"}));
      if (node.rank < (mobile ? 3 : 8)) addWrappedSvgText(group, node.president.president_name, node.x,
        node.y + node.radius + 13, "summary-topic-president-label", mobile ? 15 : 18, 2);
      wirePresidentTarget(group, presidentId);
      svg.append(group);
    });
    updateActiveStyles();
  }

  function renderPicker() {
    if (!index) return;
    picker.replaceChildren();
    const atLimit = state.selectedTopicIds.length >= MAX_TOPICS;
    index.level1_topics.forEach(topic => {
      const selected = state.selectedTopicIds.includes(topic.topic_id);
      const button = textNode("button", topic.topic_label);
      button.type = "button";
      button.dataset.summaryTopicChoice = topic.topic_id;
      button.setAttribute("aria-pressed", String(selected));
      button.disabled = atLimit && !selected;
      button.title = topic.topic_definition || "AI-assigned topic";
      picker.append(button);
    });
  }

  function renderPresidentFilter() {
    if (!index) return;
    presidentFilter.replaceChildren();
    const prompt = document.createElement("option");
    prompt.value = "";
    prompt.textContent = state.selectedPresidentIds.length >= MAX_PRESIDENTS
      ? `${MAX_PRESIDENTS} presidents selected` : "Add a president…";
    presidentFilter.append(prompt);
    const selected = new Set(state.selectedPresidentIds);
    index.presidents.filter(president => presidentSupport(president.president_profile_id)?.president_support_status
      === "supported").forEach(president => {
      const option = document.createElement("option");
      option.value = president.president_profile_id;
      option.textContent = president.president_name;
      option.disabled = selected.has(president.president_profile_id)
        || state.selectedPresidentIds.length >= MAX_PRESIDENTS;
      presidentFilter.append(option);
    });
    presidentFilter.disabled = state.selectedPresidentIds.length >= MAX_PRESIDENTS;
    presidentChips.replaceChildren();
    state.selectedPresidentIds.forEach(presidentId => {
      const president = presidentForId(presidentId);
      if (!president) return;
      const chip = textNode("button", president.president_name);
      chip.type = "button";
      chip.dataset.presidentId = presidentId;
      chip.setAttribute("aria-label", `Remove ${president.president_name} from the president comparison`);
      chip.addEventListener("click", () => removePresidentFilter(presidentId));
      presidentChips.append(chip);
    });
    presidentRow.hidden = !state.selectedPresidentIds.length;
  }

  function clearPins() {
    state.previewPresidentId = null;
    state.pinnedPresidentId = null;
    state.previewTopicId = null;
  }

  function addPresidentFilter(presidentId) {
    if (!presidentForId(presidentId) || state.selectedPresidentIds.includes(presidentId)) return;
    if (state.selectedPresidentIds.length >= MAX_PRESIDENTS) {
      setStatus(`Choose no more than ${MAX_PRESIDENTS} presidents.`);
      return;
    }
    state.selectedPresidentIds = [...state.selectedPresidentIds, presidentId];
    clearPins();
    renderAll();
  }

  function removePresidentFilter(presidentId) {
    state.selectedPresidentIds = state.selectedPresidentIds.filter(id => id !== presidentId);
    clearPins();
    renderAll();
    presidentFilter.focus();
  }

  function updateControls() {
    renderPicker();
    renderPresidentFilter();
    selectionCount.textContent = `${state.selectedTopicIds.length} of ${MAX_TOPICS} selected`;
    clearButton.disabled = !state.selectedTopicIds.length;
    if (!state.selectedTopicIds.length) {
      const selectedPresidents = state.selectedPresidentIds.length
        ? ` ${state.selectedPresidentIds.length} selected presidents will be compared.` : "";
      setStatus(`Choose up to eight topics. No presidents are drawn yet.${selectedPresidents}`);
      return;
    }
    const rows = presidentAggregates();
    const mobile = matchMedia("(max-width: 760px)").matches;
    const connected = new Set(rows.map(row => row.president.president_profile_id));
    const missing = state.selectedPresidentIds.filter(id => !connected.has(id)).length;
    const selectedCount = state.selectedPresidentIds.length;
    const presidentScope = selectedCount
      ? `${selectedCount} president${selectedCount === 1 ? "" : "s"} selected · ${rows.length} `
        + `${rows.length === 1 ? "has" : "have"} recurring relationships`
      : `${rows.length} presidents meet the recurring threshold`;
    const missingText = missing
      ? ` · ${missing} ${missing === 1 ? "does" : "do"} not meet it for these topics` : "";
    setStatus(`${state.selectedTopicIds.length} topics selected · ${presidentScope}${missingText} · `
      + `${Math.min(rows.length, mobile ? 18 : 45)} shown in the field.`);
  }

  function renderAll() {
    renderDiagram();
    updateControls();
    updateStageReadout();
    updateSizeKey();
  }

  function toggleTopic(topicId) {
    const selected = new Set(state.selectedTopicIds);
    if (selected.has(topicId)) selected.delete(topicId);
    else if (selected.size < MAX_TOPICS) selected.add(topicId);
    else { setStatus(`Choose no more than ${MAX_TOPICS} topics.`); return; }
    state.selectedTopicIds = index.level1_topics.filter(topic => selected.has(topic.topic_id))
      .map(topic => topic.topic_id);
    clearPins();
    renderAll();
  }

  picker.addEventListener("click", async event => {
    const button = event.target.closest("[data-summary-topic-choice]");
    if (!button || button.disabled) return;
    try {await ensureIndex(); toggleTopic(button.dataset.summaryTopicChoice);}
    catch (error) { /* The exact recurring-topic CSV remains available. */ }
  });
  presidentFilter.addEventListener("change", async () => {
    const presidentId = presidentFilter.value;
    if (!presidentId) return;
    try {await ensureIndex(); addPresidentFilter(presidentId);}
    catch (error) { /* The exact recurring-topic CSV remains available. */ }
  });
  presidentClear.addEventListener("click", () => {
    state.selectedPresidentIds = [];
    clearPins();
    renderAll();
    presidentFilter.focus();
  });
  clearButton.addEventListener("click", () => {state.selectedTopicIds = []; clearPins(); renderAll();});
  root.addEventListener("keydown", event => {
    if (event.key !== "Escape") return;
    if (state.pinnedPresidentId) {
      event.preventDefault(); state.pinnedPresidentId = null; updateActiveStyles(); updateStageReadout(); return;
    }
    if (state.selectedPresidentIds.length) {
      event.preventDefault(); state.selectedPresidentIds = []; clearPins(); renderAll(); return;
    }
    if (state.selectedTopicIds.length) {event.preventDefault(); state.selectedTopicIds = []; renderAll();}
  });
  root.addEventListener("focusin", () => {ensureIndex().catch(() => {});}, {once: true});
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      observer.disconnect();
      ensureIndex().catch(() => {});
    }, {rootMargin: "600px 0px"});
    observer.observe(root);
  }
  matchMedia("(max-width: 760px)").addEventListener?.("change", () => {if (index) renderAll();});
})();
"""
