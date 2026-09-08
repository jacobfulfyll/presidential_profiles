(() => {
"use strict";
const root = document.getElementById("explore-app");
if (!root) return;
const NS = "http://www.w3.org/2000/svg";
const SERIES_LIMIT = 6;
const SLOT = [
  {letter:"A", color:"#2A78D6", shape:"circle"},
  {letter:"B", color:"#9B6200", shape:"square"},
  {letter:"C", color:"#008300", shape:"diamond"},
  {letter:"D", color:"#7A4DA3", shape:"cross"},
  {letter:"E", color:"#B33B58", shape:"triangle"},
  {letter:"F", color:"#006E73", shape:"kite"}
];
const CSV_FIELDS = ["selection_order","series_id","series_label","series_kind","instrument","query","grouping_active","family_label","family_form_count","family_forms","period_kind","period","period_order","period_start","period_end","x","display_value","display_unit","numerator_count","denominator_count","denominator_unit","point","lo_sampling","hi_sampling","lo","hi","n_paragraphs","n_speeches","n_paired_paragraphs","ci_status","interval_unresolvable","disagreement_band_applied","disagreement_half_width","ci_components","disagreement_status","agreement_source","value_status","source_scope"];
let INDEX = null;
let FAMILIES = null;
let TOPICS = null;
let selections = [];
let uncertainty = true;
let contextKey = "";
let activePreset = "";
let hoverId = "";
let focusId = "";
let pinnedId = "";
let inspected = null;
let restoreSerial = 0;
let selectionEpoch = 0;
const fetchCache = new Map();
const catalogMap = new Map();
const savedDisclosureState = new Map();

const byId = id => document.getElementById(id);
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function svg(tag, attrs = {}) {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([name, value]) => node.setAttribute(name, String(value)));
  return node;
}
function status(text, announce = true) {
  const node = byId("explore-status");
  node.textContent = text;
  if (!announce) node.setAttribute("aria-live", "off");
  else node.setAttribute("aria-live", "polite");
}
function safeError(message, retry) {
  const panel = byId("explore-failure");
  panel.replaceChildren();
  panel.appendChild(element("span", "", message));
  if (retry) {
    const button = element("button", "retry-action", "Retry");
    button.type = "button";
    button.addEventListener("click", () => { panel.hidden = true; retry(); });
    panel.appendChild(button);
  }
  const dismiss = element("button", "retry-action", "Dismiss");
  dismiss.type = "button";
  dismiss.addEventListener("click", () => { panel.hidden = true; });
  panel.appendChild(dismiss);
  panel.hidden = false;
}
async function loadJSON(url, cacheKey = url) {
  if (!fetchCache.has(cacheKey)) {
    const request = fetch(url).then(response => {
      if (!response.ok) throw new Error(`${response.status} ${url}`);
      return response.json();
    }).catch(error => { fetchCache.delete(cacheKey); throw error; });
    fetchCache.set(cacheKey, request);
  }
  return fetchCache.get(cacheKey);
}
async function ensureFamilies() {
  if (FAMILIES) return FAMILIES;
  try {
    FAMILIES = await loadJSON(root.dataset.familiesUrl, "families");
    byId("word-mode-family").disabled = false;
    return FAMILIES;
  } catch (error) {
    byId("word-mode-family").disabled = true;
    safeError("Word-family data could not be loaded. Exact-form search still works.", ensureFamilies);
    throw error;
  }
}
async function ensureTopics() {
  if (TOPICS) return TOPICS;
  try {
    TOPICS = await loadJSON(root.dataset.topicsUrl, "topics");
    return TOPICS;
  } catch (error) {
    safeError("Topic values could not be loaded. Current selections were left unchanged.", ensureTopics);
    throw error;
  }
}
function fnv1aBucket(value) {
  let hash = 2166136261;
  const bytes = new TextEncoder().encode(value);
  bytes.forEach(byte => { hash = Math.imul(hash ^ byte, 16777619) >>> 0; });
  return (hash & 63).toString(16).padStart(2, "0");
}
function shardUrl(namespace, key) {
  const directory = INDEX.sharding.namespaces[namespace];
  return `explorer/${directory}/${fnv1aBucket(key)}.json`;
}
async function loadLexicalRecord(namespace, key) {
  const url = shardUrl(namespace, key);
  const shard = await loadJSON(url, `shard:${url}`);
  return Object.prototype.hasOwnProperty.call(shard.records, key) ? shard.records[key] : null;
}
function decodeLexical(record) {
  const offsets = [];
  let cursor = 0;
  record[1].forEach((delta, index) => {
    cursor = index === 0 ? delta : cursor + delta;
    offsets.push(cursor);
  });
  const sparse = new Map(offsets.map((offset, index) => [offset, [record[2][index], record[3][index]]]));
  return INDEX.lexical_axis.map((period, index) => {
    const available = period.value_status === "observed_or_zero";
    const pair = sparse.get(index) || [0, 0];
    const display = available ? pair[1] / INDEX.fixed_point_scale : null;
    return {
      period_kind:period.period_kind, period:period.period, period_order:period.period_order,
      period_start:period.period_start, period_end:period.period_end, x:period.x,
      display_value:display, point:display, numerator_count:available ? pair[0] : null,
      denominator_count:available ? period.denominator_words : null,
      value_status:available ? (pair[0] ? "observed" : "observed_zero") : "unknown_low_words",
      status_label:period.status_label
    };
  });
}
function lexicalRows(meta, record) {
  const forms = meta.familyForms ? JSON.stringify(meta.familyForms) : null;
  return decodeLexical(record).map(row => ({
    ...row, series_id:meta.id, series_label:meta.label, series_kind:meta.kind,
    instrument:meta.instrument, query:meta.query, grouping_active:meta.grouping,
    family_label:meta.familyLabel || null, family_form_count:meta.familyFormCount || null,
    family_forms:forms, display_unit:"uses per 10,000 indexed words",
    denominator_unit:"indexed source-document words", lo_sampling:null, hi_sampling:null,
    lo:null, hi:null, n_paragraphs:null, n_speeches:null, n_paired_paragraphs:null,
    ci_status:"not_applicable_lexical_rate", interval_unresolvable:false,
    disagreement_band_applied:false, disagreement_half_width:null,
    ci_components:"not_applicable_lexical_rate", disagreement_status:"not_applicable_lexical_rate",
    agreement_source:"not applicable", source_scope:"Complete source-document transcripts",
    interval_label:"Not published for this measure",
    support_label:row.denominator_count == null ? "Window does not clear the word floor" : `${row.denominator_count.toLocaleString()} indexed words`
  }));
}
function asciiLetterTokens(value) {
  return value.match(/[a-z']+/g) || [];
}
function hasUnsupportedLetter(value) {
  return Array.from(value).some(char => /[^\x00-\x7F]/.test(char) && /\p{L}/u.test(char));
}
function escapeRegExp(value) { return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
function normalizeGrouped(value, normalizations) {
  let result = value;
  Object.entries(normalizations).sort((a,b) => b[0].length - a[0].length).forEach(([from,to]) => {
    const pattern = new RegExp(`(^|[^a-z])${escapeRegExp(from)}(?=$|[^a-z])`, "g");
    result = result.replace(pattern, (match, prefix) => prefix + to);
  });
  return result;
}
function parseTerm(raw, grouping, familyPayload = FAMILIES) {
  let value = String(raw).trim().replace(/[’‘]/g, "'");
  if (hasUnsupportedLetter(value)) return {error:"Use English A–Z letters; this index does not support other scripts."};
  value = value.toLowerCase();
  if (grouping && familyPayload) value = normalizeGrouped(value, familyPayload.normalizations);
  const tokens = asciiLetterTokens(value).filter(token => /[a-z]/.test(token));
  if (!tokens.length) return {error:"Enter a word or two-word phrase."};
  if (tokens.length > 2) return {error:"Use no more than two indexed words."};
  const query = tokens.join(" ");
  if (!grouping) return {query, key:query, tokens, grouping:false};
  const nodes = tokens.map(token => familyPayload.resolver[token] || token);
  return {query, key:nodes.join(" "), tokens, nodes, grouping:true};
}
function familyDescription(parsed) {
  const perPosition = parsed.nodes.map(node => {
    const family = FAMILIES.families[node];
    return family ? family.forms.slice() : [node];
  });
  return {
    familyLabel:parsed.key,
    familyForms:perPosition,
    familyFormCount:perPosition.reduce((sum, forms) => sum + forms.length, 0)
  };
}
function lexicalNamespace(parsed) {
  if (parsed.grouping) return parsed.tokens.length === 1 ? "lexical-family-unigram" : "lexical-family-bigram";
  return parsed.tokens.length === 1 ? "lexical-exact-unigram" : "lexical-exact-bigram";
}
async function makeLexicalSelection(raw, grouping) {
  if (grouping) await ensureFamilies();
  const parsed = parseTerm(raw, grouping);
  if (parsed.error) throw new Error(parsed.error);
  const namespace = lexicalNamespace(parsed);
  let record;
  try { record = await loadLexicalRecord(namespace, parsed.key); }
  catch (error) { error.userMessage = `Could not load corpus data for “${parsed.query}”.`; throw error; }
  if (!record) {
    const floor = parsed.tokens.length === 1 ? 30 : 15;
    throw new Error(`“${parsed.query}” appears fewer than ${floor} times in the indexed corpus for this mode.`);
  }
  const family = grouping ? familyDescription(parsed) : {};
  const meta = {
    id:`${grouping ? "lexical-family" : "lexical-exact"}:${parsed.query}`,
    identity:`${grouping ? "lexical-family" : "lexical-exact"}:${parsed.key}`,
    kind:grouping ? "lexical-family" : "lexical-exact", query:parsed.query,
    label:grouping ? parsed.key : parsed.query, grouping,
    instrument:grouping ? "Audited algorithmic word family" : "Exact lexical form",
    ...family
  };
  return {...meta, record, rows:lexicalRows(meta, record)};
}
async function makeCatalogSelection(id) {
  const item = catalogMap.get(id);
  if (!item) throw new Error(`Unknown series skipped: “${id}”.`);
  if (item.series_kind === "acronym") {
    const meta = {id, identity:id, kind:"acronym", query:item.query, label:item.label,
      grouping:false, instrument:item.instrument};
    return {...meta, record:item.record, rows:lexicalRows(meta, item.record)};
  }
  const payload = await ensureTopics();
  if (!payload.series[id]) throw new Error(`No governed values were published for “${item.label}”.`);
  return {id, identity:id, kind:item.series_kind, query:item.source_label,
    label:item.label, grouping:null, instrument:item.instrument,
    definition:item.definition, rows:payload.series[id].map(row => ({...row}))};
}
async function makeSelection(encoded) {
  const split = encoded.indexOf(":");
  if (split < 1) throw new Error(`Invalid series state: “${encoded}”.`);
  const type = encoded.slice(0, split);
  const value = encoded.slice(split + 1);
  if (type === "lexical-exact") return makeLexicalSelection(value, false);
  if (type === "lexical-family") return makeLexicalSelection(value, true);
  if (["corex","llm","acronym"].includes(type)) return makeCatalogSelection(encoded);
  throw new Error(`Unknown series type skipped: “${type}”.`);
}
function duplicateReason(candidate, current = selections) {
  if (current.some(item => item.id === candidate.id || item.identity === candidate.identity)) {
    return `“${candidate.label}” is already selected in that mode.`;
  }
  return "";
}
function encodedSelection(item) { return item.id; }
function setControlsEnabled(enabled) {
  root.querySelectorAll("[data-hydration-control]").forEach(control => { control.disabled = !enabled; });
}
function buildCatalogMap() {
  INDEX.catalog.broad_issues.forEach(item => catalogMap.set(item.series_id, item));
  INDEX.catalog.detailed_topic_groups.forEach(group => group.topics.forEach(item => catalogMap.set(item.series_id, item)));
  INDEX.catalog.acronyms.forEach(item => catalogMap.set(item.series_id, item));
}
function normalizedSearch(value) {
  return value.normalize("NFKC").toLocaleLowerCase("en-US").replace(/[^a-z0-9]+/g, " ").trim().split(/\s+/).filter(Boolean);
}
function catalogMatches(item, tokens) {
  if (!tokens.length) return true;
  const words = item.search_terms.flatMap(value => normalizedSearch(String(value)));
  return tokens.every(token => words.some(word => word.startsWith(token)));
}
function catalogRow(item) {
  const label = element("label", "catalog-row");
  label.dataset.catalogRow = item.series_id;
  const input = document.createElement("input");
  input.type = "checkbox";
  input.value = item.series_id;
  input.dataset.hydrationControl = "";
  const copy = element("span");
  copy.appendChild(element("strong", "", item.label));
  copy.appendChild(element("small", "", item.definition));
  label.append(input, copy);
  input.addEventListener("change", async () => {
    if (!input.checked) {
      removeSelection(item.series_id, input);
      return;
    }
    input.disabled = true;
    label.setAttribute("aria-busy", "true");
    const epoch = selectionEpoch;
    try {
      const candidate = await makeCatalogSelection(item.series_id);
      if (epoch !== selectionEpoch) return;
      const duplicate = duplicateReason(candidate);
      if (duplicate) throw new Error(duplicate);
      if (selections.length >= SERIES_LIMIT) throw new Error(`Choose no more than ${SERIES_LIMIT} series.`);
      selections.push(candidate);
      activePreset = "";
      commit("push", `${candidate.label} added.`);
    } catch (error) {
      input.checked = false;
      status(error.userMessage || error.message);
      if (error.userMessage) safeError(error.userMessage, () => input.click());
    } finally {
      label.removeAttribute("aria-busy");
      updateCatalogChecks();
    }
  });
  return label;
}
function catalogGroup(labelText, items, open = false, nested = false) {
  const details = document.createElement("details");
  details.open = open;
  details.dataset.catalogGroup = nested ? `nested:${labelText}` : labelText;
  details.appendChild(element("summary", "", `${labelText} (${items.length})`));
  const list = element("div", "catalog-list");
  items.forEach(item => list.appendChild(catalogRow(item)));
  details.appendChild(list);
  return details;
}
function renderCatalog() {
  const box = byId("catalog-groups");
  box.replaceChildren();
  box.appendChild(catalogGroup("Broad issues · deterministic model", INDEX.catalog.broad_issues, true));
  const detailed = document.createElement("details");
  detailed.dataset.catalogGroup = "Detailed topics · AI labels";
  detailed.appendChild(element("summary", "", "Detailed topics · AI labels (50)"));
  INDEX.catalog.detailed_topic_groups.forEach(group => detailed.appendChild(catalogGroup(group.label, group.topics, false, true)));
  box.appendChild(detailed);
  box.appendChild(catalogGroup("Case-sensitive named acronyms", INDEX.catalog.acronyms));
  updateCatalogChecks();
}
function filterCatalog() {
  const tokens = normalizedSearch(byId("catalog-search").value);
  const searching = tokens.length > 0;
  const rows = [...root.querySelectorAll("[data-catalog-row]")];
  if (searching && !savedDisclosureState.size) {
    root.querySelectorAll("[data-catalog-group]").forEach(group => savedDisclosureState.set(group.dataset.catalogGroup, group.open));
  }
  let shown = 0;
  rows.forEach(row => {
    const item = catalogMap.get(row.dataset.catalogRow);
    const visible = catalogMatches(item, tokens);
    row.hidden = !visible;
    if (visible) shown += 1;
  });
  root.querySelectorAll("[data-catalog-group]").forEach(group => {
    const hasMatch = [...group.querySelectorAll(":scope > .catalog-list > [data-catalog-row], :scope > details [data-catalog-row]")].some(row => !row.hidden);
    group.hidden = searching && !hasMatch;
    if (searching && hasMatch) group.open = true;
    if (!searching && savedDisclosureState.has(group.dataset.catalogGroup)) group.open = savedDisclosureState.get(group.dataset.catalogGroup);
  });
  if (!searching) savedDisclosureState.clear();
  const none = byId("catalog-no-results");
  none.hidden = shown !== 0;
  none.textContent = shown ? "" : "No governed series matches every search term. Try a shorter word or a parent topic.";
}
function updateCatalogChecks() {
  const full = selections.length >= SERIES_LIMIT;
  root.querySelectorAll("[data-catalog-row]").forEach(row => {
    const input = row.querySelector("input");
    const selected = selections.some(item => item.id === input.value);
    input.checked = selected;
    input.disabled = !selected && full;
  });
}
function slotMark(index) {
  const mark = element("span", `slot-mark slot-${index}`);
  mark.style.color = SLOT[index].color;
  mark.appendChild(element("span", "", SLOT[index].letter));
  mark.setAttribute("aria-hidden", "true");
  return mark;
}
function focusAfterRender(preferredId, fallback) {
  requestAnimationFrame(() => {
    const target = preferredId ? root.querySelector(`[data-focus-id="${CSS.escape(preferredId)}"]`) : null;
    (target || fallback)?.focus();
  });
}
function removeSelection(id, sourceControl) {
  const index = selections.findIndex(item => item.id === id);
  if (index < 0) return;
  const next = selections[index + 1] || selections[index - 1];
  const removed = selections[index];
  selections.splice(index, 1);
  if (pinnedId === removed.id) pinnedId = "";
  if (inspected && inspected.id === removed.id) inspected = null;
  activePreset = "";
  commit("push", `${removed.label} removed.`);
  focusAfterRender(next ? `remove:${next.id}` : "", sourceControl || byId("word-query"));
}
async function switchGrouping(item, control) {
  const epoch = selectionEpoch;
  control.disabled = true;
  control.setAttribute("aria-busy", "true");
  try {
    const replacement = await makeLexicalSelection(item.query, !item.grouping);
    if (epoch !== selectionEpoch || !selections.includes(item)) return;
    const others = selections.filter(candidate => candidate !== item);
    const duplicate = duplicateReason(replacement, others);
    if (duplicate) throw new Error(duplicate);
    const index = selections.indexOf(item);
    selections.splice(index, 1, replacement);
    if (pinnedId === item.id) pinnedId = replacement.id;
    activePreset = "";
    commit("push", `${replacement.label} now uses ${replacement.grouping ? "grouped forms" : "the exact form"}.`);
    focusAfterRender(`switch:${replacement.id}`, byId("word-query"));
  } catch (error) {
    status(error.userMessage || error.message);
    if (error.userMessage) safeError(error.userMessage, () => switchGrouping(item, control));
  } finally {
    control.removeAttribute("aria-busy");
    control.disabled = false;
  }
}
function renderSelections() {
  byId("selected-count").textContent = `${selections.length} of ${SERIES_LIMIT} selected`;
  const list = byId("selected-list");
  list.replaceChildren();
  selections.forEach((item, index) => {
    const card = element("article", "series-chip");
    card.dataset.seriesChip = item.id;
    const head = element("div", "chip-head");
    head.appendChild(slotMark(index));
    const copy = element("div");
    copy.appendChild(element("div", "chip-label", item.label));
    const mode = item.kind === "lexical-family" ? "Grouping on" : item.kind === "lexical-exact" ? "Exact form" : item.instrument;
    copy.appendChild(element("span", "chip-mode", mode));
    head.appendChild(copy);
    const remove = element("button", "chip-action", "Remove");
    remove.type = "button";
    remove.dataset.focusId = `remove:${item.id}`;
    remove.setAttribute("aria-label", `Remove ${item.label}.`);
    remove.addEventListener("click", () => removeSelection(item.id, byId("word-query")));
    head.appendChild(remove);
    card.appendChild(head);
    if (item.kind === "lexical-family") {
      const details = document.createElement("details");
      details.className = "family-members";
      details.appendChild(element("summary", "", `${item.familyFormCount} forms combined`));
      const listNode = document.createElement("ol");
      item.familyForms.forEach((forms, position) => {
        const li = document.createElement("li");
        li.textContent = item.familyForms.length === 1 ? forms.join(", ") : `Position ${position + 1}: ${forms.join(", ")}`;
        listNode.appendChild(li);
      });
      details.appendChild(listNode);
      card.appendChild(details);
    }
    if (item.kind === "lexical-family" || item.kind === "lexical-exact") {
      const controls = element("div", "chip-controls");
      const switcher = element("button", "chip-action", item.grouping ? "Use exact form" : "Group word forms");
      switcher.type = "button";
      switcher.dataset.focusId = `switch:${item.id}`;
      switcher.addEventListener("click", () => switchGrouping(item, switcher));
      controls.appendChild(switcher);
      card.appendChild(controls);
    }
    list.appendChild(card);
  });
  byId("clear-series").disabled = !selections.length;
  byId("download-selected").disabled = !selections.length;
  byId("add-word").disabled = selections.length >= SERIES_LIMIT;
  updateCatalogChecks();
}
function selectedPresetKey() {
  const exact = selections.every(item => item.kind === "lexical-exact");
  if (!exact) return "";
  const words = selections.map(item => item.query);
  return Object.keys(INDEX.presets).find(key => JSON.stringify(INDEX.presets[key].words) === JSON.stringify(words)) || "";
}
function updatePresetNote() {
  const note = byId("preset-note");
  if (activePreset && INDEX.presets[activePreset]) {
    const preset = INDEX.presets[activePreset];
    note.textContent = `${preset.label}. Exact words: ${preset.words.join(", ")}. Why included: ${preset.rationale} Known ambiguities: ${preset.ambiguities}`;
  } else {
    note.textContent = "Loading a guide replaces the current selection with its editable exact words.";
  }
}
function writeURL(mode) {
  const url = new URL(location.href);
  ["state","s","words","topics","entities","grouped","combine","periods","scenario","from","preset","uncertainty","context"].forEach(key => url.searchParams.delete(key));
  url.searchParams.set("state", "2");
  selections.forEach(item => url.searchParams.append("s", encodedSelection(item)));
  if (!uncertainty) url.searchParams.set("uncertainty", "off");
  if (contextKey) url.searchParams.set("context", contextKey);
  if (activePreset) url.searchParams.set("preset", activePreset);
  history[mode === "replace" ? "replaceState" : "pushState"](null, "", url);
}
function commit(historyMode, message) {
  selectionEpoch += 1;
  activePreset = selectedPresetKey();
  writeURL(historyMode);
  renderAll();
  if (message) status(message);
}
function chartGroups() {
  return [
    {key:"lexical", title:"Words and named acronyms", unit:"Uses per 10,000 indexed words", items:selections.filter(item => ["lexical-exact","lexical-family","acronym"].includes(item.kind))},
    {key:"corex", title:"Broad issues · deterministic model", unit:"Percent of eligible paragraphs · five-year periods", items:selections.filter(item => item.kind === "corex")},
    {key:"llm", title:"Detailed topics · AI labels", unit:"Percent of eligible paragraphs · nine named eras", items:selections.filter(item => item.kind === "llm")}
  ].filter(group => group.items.length);
}
function validRows(item) { return item.rows.filter(row => row.display_value !== null && Number.isFinite(Number(row.display_value))); }
function pathFor(rows, xScale, yScale, predicate = () => true) {
  let path = "";
  let drawing = false;
  rows.forEach(row => {
    const valid = row.display_value !== null && predicate(row);
    if (!valid) { drawing = false; return; }
    const point = `${xScale(Number(row.x)).toFixed(2)} ${yScale(Number(row.display_value)).toFixed(2)}`;
    path += `${drawing ? "L" : "M"}${point}`;
    drawing = true;
  });
  return path;
}
function bandPaths(rows, xScale, yScale) {
  const paths = [];
  let run = [];
  const flush = () => {
    if (!run.length) return;
    if (run.length === 1) {
      const row = run[0];
      paths.push(`M${xScale(Number(row.x))} ${yScale(Number(row.lo) * 100)}L${xScale(Number(row.x))} ${yScale(Number(row.hi) * 100)}`);
    } else {
      const high = run.map(row => `${xScale(Number(row.x))} ${yScale(Number(row.hi) * 100)}`);
      const low = run.slice().reverse().map(row => `${xScale(Number(row.x))} ${yScale(Number(row.lo) * 100)}`);
      paths.push(`M${high.join("L")}L${low.join("L")}Z`);
    }
    run = [];
  };
  rows.forEach(row => {
    if (row.lo === null || row.hi === null) flush();
    else run.push(row);
  });
  flush();
  return paths;
}
function marker(x, y, shape, color, className) {
  let node;
  if (shape === "square") node = svg("rect", {x:x-4,y:y-4,width:8,height:8});
  else if (shape === "diamond") node = svg("path", {d:`M${x} ${y-5}L${x+5} ${y}L${x} ${y+5}L${x-5} ${y}Z`});
  else if (shape === "triangle") node = svg("path", {d:`M${x} ${y-5}L${x+5} ${y+4}L${x-5} ${y+4}Z`});
  else node = svg("circle", {cx:x,cy:y,r:4});
  node.setAttribute("class", className);
  node.style.color = color;
  return node;
}
function emphasisId() { return focusId || hoverId || pinnedId; }
function applyEmphasis() {
  const active = emphasisId();
  byId("chart-panels").classList.toggle("has-emphasis", Boolean(active));
  root.querySelectorAll("[data-series-layer]").forEach(node => node.classList.toggle("is-dimmed", Boolean(active) && node.dataset.seriesLayer !== active));
  root.querySelectorAll("[data-legend-series]").forEach(node => {
    node.classList.toggle("is-dimmed", Boolean(active) && node.dataset.legendSeries !== active);
    node.classList.toggle("is-emphasized", node.dataset.legendSeries === active);
    node.setAttribute("aria-pressed", String(node.dataset.legendSeries === pinnedId));
  });
}
function nearestRow(item, targetX) {
  const rows = validRows(item);
  return rows.reduce((best, row) => !best || Math.abs(Number(row.x)-targetX) < Math.abs(Number(best.x)-targetX) ? row : best, null);
}
function readableValue(row) { return row.display_value === null ? "Unknown" : `${Number(row.display_value).toFixed(2)} ${row.display_unit || ""}`.trim(); }
function readoutText(item, row) {
  const period = row.period_start === row.period_end ? row.period : `${row.period_start}–${row.period_end}`;
  const interval = row.interval_label || "Not published for this measure";
  const support = row.support_label || row.status_label || "Support not available";
  const state = row.status_label || row.value_status;
  return {heading:`${item.label} · ${period}`, detail:`${readableValue(row)} · Uncertainty: ${interval} · ${support} · ${state}`};
}
function inspect(item, row, announce = false) {
  if (!row) return;
  inspected = {id:item.id, period_order:row.period_order};
  const text = readoutText(item, row);
  const box = byId("chart-readout");
  box.replaceChildren(element("strong", "", text.heading), element("span", "", text.detail));
  if (announce) byId("chart-live").textContent = `${text.heading}. ${text.detail}`;
}
function inspectByIndex(item, direction) {
  const rows = validRows(item);
  if (!rows.length) return;
  let index = rows.findIndex(row => inspected && inspected.id === item.id && row.period_order === inspected.period_order);
  if (direction === "home") index = 0;
  else if (direction === "end") index = rows.length - 1;
  else index = Math.max(0, Math.min(rows.length - 1, (index < 0 ? rows.length - 1 : index) + direction));
  inspect(item, rows[index], true);
}
function togglePin(item, row, announce = true) {
  pinnedId = pinnedId === item.id ? "" : item.id;
  if (row) inspect(item, row, announce);
  applyEmphasis();
  if (announce) status(pinnedId ? `${item.label} pinned.` : "Pinned series cleared.");
}
function legendGlyph(index) {
  const icon = svg("svg", {viewBox:"0 0 30 14", class:"legend-symbol", "aria-hidden":"true"});
  const line = svg("line", {x1:1,y1:7,x2:29,y2:7,class:`trajectory slot-line-${index}`});
  line.style.color = SLOT[index].color;
  icon.appendChild(line);
  return icon;
}
function panelYMax(items) {
  const values = [];
  items.forEach(item => item.rows.forEach(row => {
    if (row.display_value !== null) values.push(Number(row.display_value));
    if (row.ci_status === "ok" && row.hi !== null) values.push(Number(row.hi) * 100);
  }));
  const maximum = Math.max(0, ...values);
  return maximum ? maximum * 1.05 : 1;
}
function renderPanel(group) {
  const panel = element("section", "trend-panel");
  panel.appendChild(element("h3", "", group.title));
  panel.appendChild(element("p", "panel-unit", group.unit));
  const legend = element("div", "series-legend");
  group.items.forEach(item => {
    const index = selections.indexOf(item);
    const button = element("button", "legend-button");
    button.type = "button";
    button.dataset.legendSeries = item.id;
    button.setAttribute("aria-pressed", String(pinnedId === item.id));
    button.setAttribute("aria-label", `${SLOT[index].letter}, ${item.label}. Focus to highlight; arrow keys inspect periods; Enter pins.`);
    button.append(legendGlyph(index), element("span", "legend-label", `${SLOT[index].letter} · ${item.label}`));
    button.addEventListener("pointerenter", () => { hoverId = item.id; applyEmphasis(); });
    button.addEventListener("pointerleave", () => { hoverId = ""; applyEmphasis(); });
    button.addEventListener("focus", () => { focusId = item.id; applyEmphasis(); const rows=validRows(item); inspect(item, rows[rows.length-1], false); });
    button.addEventListener("blur", () => { focusId = ""; applyEmphasis(); });
    button.addEventListener("click", () => { const rows=validRows(item); const row=rows.find(value => inspected && inspected.id===item.id && value.period_order===inspected.period_order) || rows[rows.length-1]; togglePin(item,row); });
    button.addEventListener("keydown", event => {
      if (event.key === "ArrowLeft") { event.preventDefault(); inspectByIndex(item, -1); }
      else if (event.key === "ArrowRight") { event.preventDefault(); inspectByIndex(item, 1); }
      else if (event.key === "Home") { event.preventDefault(); inspectByIndex(item, "home"); }
      else if (event.key === "End") { event.preventDefault(); inspectByIndex(item, "end"); }
      else if (event.key === "Enter" || event.key === " ") { event.preventDefault(); const rows=validRows(item); const row=rows.find(value => inspected && inspected.id===item.id && value.period_order===inspected.period_order) || rows[rows.length-1]; togglePin(item,row); }
      else if (event.key === "Escape") { event.preventDefault(); pinnedId=""; applyEmphasis(); status("Pinned series cleared."); }
    });
    legend.appendChild(button);
  });
  panel.appendChild(legend);
  const width=900, height=310, left=58, right=14, top=14, bottom=38;
  const plotWidth=width-left-right, plotHeight=height-top-bottom;
  const chart = svg("svg", {viewBox:`0 0 ${width} ${height}`, class:"trend-svg", role:"img"});
  const titleId=`panel-title-${group.key}`, descId=`panel-desc-${group.key}`;
  chart.setAttribute("aria-labelledby", `${titleId} ${descId}`);
  const title=svg("title", {id:titleId}); title.textContent=group.title;
  const desc=svg("desc", {id:descId}); desc.textContent=`${group.unit}. Lines use independent zero-based vertical scales and share the 1785 to 2026 horizontal scale. Use the legend buttons for keyboard inspection.`;
  chart.append(title,desc);
  const xScale=value=>left+(value-1785)/(2026-1785)*plotWidth;
  const yMax=panelYMax(group.items), yScale=value=>top+plotHeight-Math.max(0,Math.min(yMax,value))/yMax*plotHeight;
  const context=INDEX.historical_contexts.find(period=>period.key===contextKey);
  if(context){const rect=svg("rect",{x:xScale(context.period_start),y:top,width:Math.max(1,xScale(context.period_end)-xScale(context.period_start)),height:plotHeight,class:"context-band"});rect.style.color="#6f7890";chart.appendChild(rect);}
  [0,.5,1].forEach(fraction=>{const y=top+plotHeight-fraction*plotHeight;chart.appendChild(svg("line",{x1:left,y1:y,x2:width-right,y2:y,class:"grid-line"}));const label=svg("text",{x:left-8,y:y+4,"text-anchor":"end"});label.textContent=(yMax*fraction).toFixed(fraction===0?0:1);chart.appendChild(label);});
  [1785,1850,1900,1950,2000,2026].forEach(year=>{const x=xScale(year);chart.appendChild(svg("line",{x1:x,y1:top,x2:x,y2:top+plotHeight,class:"grid-line"}));const label=svg("text",{x,y:height-12,"text-anchor":year===1785?"start":year===2026?"end":"middle"});label.textContent=year;chart.appendChild(label);});
  chart.appendChild(svg("line",{x1:left,y1:top+plotHeight,x2:width-right,y2:top+plotHeight,class:"axis-line"}));
  group.items.forEach(item=>{
    const index=selections.indexOf(item), slot=SLOT[index];
    const layer=svg("g",{class:"series-layer","data-series-layer":item.id});
    layer.dataset.seriesLayer=item.id;
    if(uncertainty && (item.kind==="corex"||item.kind==="llm")) bandPaths(item.rows,xScale,yScale).forEach(path=>{const band=svg("path",{d:path,class:"band-path"});band.style.color=slot.color;layer.appendChild(band);});
    const path=pathFor(item.rows,xScale,yScale);
    if(path){const under=svg("path",{d:path,class:`${item.kind==="corex"||item.kind==="llm"?"trajectory-low":"trajectory"} slot-line-${index}`});under.style.color=slot.color;layer.appendChild(under);}
    if(item.kind==="corex"||item.kind==="llm"){
      const solid=pathFor(item.rows,xScale,yScale,row=>row.ci_status==="ok");
      if(solid){const node=svg("path",{d:solid,class:`trajectory slot-line-${index}`});node.style.color=slot.color;layer.appendChild(node);}
      item.rows.forEach(row=>{if(row.display_value===null)return;const x=xScale(Number(row.x)),y=yScale(Number(row.display_value));if(row.interval_unresolvable)layer.appendChild(marker(x,y,slot.shape,slot.color,"unresolved-mark"));else if(row.ci_status!=="ok")layer.appendChild(marker(x,y,slot.shape,slot.color,"low-mark"));});
      if(item.rows.some(row=>row.ci_status!=="ok"&&row.hi!==null&&Number(row.hi)*100>yMax)){const cue=svg("text",{x:width-right-2,y:top+12,"text-anchor":"end",class:"overflow-cue"});cue.textContent="† range exceeds scale";layer.appendChild(cue);}
    }
    if(path){const hit=svg("path",{d:path,class:"hit-path"});hit.dataset.seriesId=item.id;hit.addEventListener("pointerenter",()=>{hoverId=item.id;applyEmphasis();});hit.addEventListener("pointerleave",()=>{hoverId="";applyEmphasis();});hit.addEventListener("pointermove",event=>{const rect=chart.getBoundingClientRect();const local=(event.clientX-rect.left)/rect.width*width;const target=1785+(local-left)/plotWidth*(2026-1785);inspect(item,nearestRow(item,target),false);});hit.addEventListener("click",event=>{const rect=chart.getBoundingClientRect();const local=(event.clientX-rect.left)/rect.width*width;const target=1785+(local-left)/plotWidth*(2026-1785);togglePin(item,nearestRow(item,target),true);});layer.appendChild(hit);}
    chart.appendChild(layer);
  });
  const swipeCue=element("p","chart-swipe-cue","Swipe chart horizontally →");
  const scroller=element("div","trend-scroll");
  scroller.tabIndex=0;
  scroller.setAttribute("role","region");
  scroller.setAttribute("aria-label",`${group.title} chart; scroll horizontally on narrow screens`);
  scroller.appendChild(chart);
  panel.append(swipeCue,scroller);
  return panel;
}
function renderCharts() {
  const mount=byId("chart-panels"); mount.replaceChildren();
  const groups=chartGroups();
  if(!groups.length){mount.appendChild(element("div","empty-chart","Choose a word, broad issue, detailed topic, or named acronym to begin."));byId("chart-readout").replaceChildren(element("strong","","No series selected"),element("span","","Add up to six series above."));return;}
  groups.forEach(group=>mount.appendChild(renderPanel(group)));
  const current=inspected&&selections.find(item=>item.id===inspected.id);
  if(current){const row=current.rows.find(value=>value.period_order===inspected.period_order&&value.display_value!==null);if(row)inspect(current,row,false);else inspected=null;}
  if(!inspected){const item=selections[0],rows=validRows(item);inspect(item,rows[rows.length-1],false);}
  applyEmphasis();
  byId("chart-summary").textContent=`${groups.length} aligned ${groups.length===1?"panel":"panels"}; each measure keeps its native unit and independent zero baseline. Topic lines use governed published periods; word rates use centered five-year windows.`;
}
function exactEstimate(row){return row.display_value===null?"Unknown":`${Number(row.display_value).toFixed(2)} ${row.display_unit}`;}
function renderExactValues(){
  const select=byId("exact-series");const previous=select.value;select.replaceChildren();
  selections.forEach(item=>{const option=document.createElement("option");option.value=item.id;option.textContent=item.label;select.appendChild(option);});
  select.disabled=!selections.length;select.value=selections.some(item=>item.id===previous)?previous:(selections[0]?.id||"");
  const mount=byId("exact-table-mount");mount.replaceChildren();
  const item=selections.find(value=>value.id===select.value);if(!item){mount.appendChild(element("p","","Choose a series to inspect exact values."));return;}
  const table=element("table","exact-table");
  const caption=document.createElement("caption");caption.textContent=`${item.label}. ${item.instrument}. ${item.rows[0]?.period_kind||"Published periods"}; ${item.rows[0]?.display_unit||"native unit"}; ${item.grouping===true?"grouped with the audited word-family map":item.grouping===false?"exact form":"source label"}. Scope: ${item.rows[0]?.source_scope||"published corpus scope"}.`;table.appendChild(caption);
  const head=document.createElement("thead"),headRow=document.createElement("tr");["Period","Estimate","Published uncertainty","Evidence / support","Status"].forEach(text=>{const th=element("th","",text);th.scope="col";headRow.appendChild(th);});head.appendChild(headRow);table.appendChild(head);
  const body=document.createElement("tbody");item.rows.forEach(row=>{const tr=document.createElement("tr");const period=`${row.period_start}–${row.period_end}`;const uncertaintyText=row.interval_label||"Not published for this measure";const supportText=row.support_label||(row.denominator_count==null?"Not published":`${Number(row.denominator_count).toLocaleString()} ${row.denominator_unit}`);[period,exactEstimate(row),uncertaintyText,supportText,row.status_label||row.value_status].forEach(text=>tr.appendChild(element("td","",text)));body.appendChild(tr);});table.appendChild(body);const wrap=element("div","table-wrap");wrap.appendChild(table);mount.appendChild(wrap);
}
function csvCell(value){if(value===null||value===undefined)return "";let text=typeof value==="boolean"?(value?"true":"false"):String(value);if(/^[=+\-@]/.test(text))text="'"+text;return /[",\n\r]/.test(text)?`"${text.replaceAll('"','""')}"`:text;}
function downloadSelected(){
  const rows=[CSV_FIELDS.join(",")];selections.forEach((item,index)=>item.rows.forEach(source=>{const row={...source,selection_order:index+1};rows.push(CSV_FIELDS.map(field=>csvCell(row[field])).join(","));}));
  const blob=new Blob([rows.join("\n")+"\n"],{type:"text/csv;charset=utf-8"});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download="presidential-profiles-explore.csv";link.click();URL.revokeObjectURL(link.href);
}
function renderAll(){renderSelections();updatePresetNote();renderCharts();if(byId("exact-values").open)renderExactValues();byId("uncertainty-toggle").checked=uncertainty;byId("context-select").value=contextKey;}
async function addWord(){
  const input=byId("word-query"),button=byId("add-word"),grouping=byId("word-mode-family").checked,epoch=selectionEpoch;button.disabled=true;input.setAttribute("aria-busy","true");
  try{if(selections.length>=SERIES_LIMIT)throw new Error(`Choose no more than ${SERIES_LIMIT} series.`);const candidate=await makeLexicalSelection(input.value,grouping);if(epoch!==selectionEpoch)return;const duplicate=duplicateReason(candidate);if(duplicate)throw new Error(duplicate);selections.push(candidate);activePreset="";input.value="";commit("push",`${candidate.label} added.`);}
  catch(error){status(error.userMessage||error.message);if(error.userMessage)safeError(error.userMessage,addWord);}
  finally{input.removeAttribute("aria-busy");button.disabled=selections.length>=SERIES_LIMIT;}
}
async function applyPreset(key,button){
  const preset=INDEX.presets[key];if(!preset){status(`Unknown preset skipped: “${key}”.`);return;}const epoch=++selectionEpoch;button.disabled=true;button.setAttribute("aria-busy","true");
  try{const next=await Promise.all(preset.words.map(word=>makeLexicalSelection(word,false)));if(epoch!==selectionEpoch)return;selections=next;activePreset=key;pinnedId="";inspected=null;writeURL("push");renderAll();status(`${preset.label} loaded with ${next.length} exact-word series.`);}
  catch(error){status(error.userMessage||`The ${preset.label} guide could not be loaded; the prior selection was retained.`);safeError(`The ${preset.label} guide could not be loaded; the prior selection was retained.`,()=>applyPreset(key,button));}
  finally{button.removeAttribute("aria-busy");button.disabled=false;}
}
function renderPresets(){const box=byId("preset-list");box.replaceChildren();Object.entries(INDEX.presets).forEach(([key,preset])=>{const card=element("div","preset-card");card.appendChild(element("p","",preset.question));const button=element("button","preset-button",`Load ${preset.label} (${preset.words.length} series)`);button.type="button";button.addEventListener("click",()=>applyPreset(key,button));card.appendChild(button);box.appendChild(card);});}
function renderContexts(){const select=byId("context-select");select.replaceChildren();const none=document.createElement("option");none.value="";none.textContent="None";select.appendChild(none);INDEX.historical_contexts.forEach(period=>{const option=document.createElement("option");option.value=period.key;option.textContent=`${period.label} · ${period.period_start}–${period.period_end}`;select.appendChild(option);});}
function legacyTopicId(value){
  const cleaned=value.startsWith("AI topic · ")?value.slice(11):value;
  for(const item of catalogMap.values())if(item.label===cleaned||item.source_label===cleaned)return item.series_id;
  return "";
}
function requestedState(){
  const params=new URLSearchParams(location.search),messages=[];let legacy=false,explicit=false,encoded=[];
  activePreset="";
  uncertainty=params.get("uncertainty")!=="off";contextKey=params.get("context")||"";
  if(contextKey&&!INDEX.historical_contexts.some(period=>period.key===contextKey)){messages.push(`unknown historical context “${contextKey}”`);contextKey="";}
  if(params.get("state")==="2"){explicit=true;encoded=params.getAll("s");}
  else if(params.has("state")||params.has("s")||params.has("words")||params.has("topics")||params.has("entities")||params.has("grouped")||params.has("combine")||params.has("periods")){legacy=true;explicit=params.has("state")||params.has("words")||params.has("topics")||params.has("entities");const grouped=params.get("grouped")==="1";params.getAll("s").forEach(value=>{const split=value.indexOf(":");const type=split>0?value.slice(0,split):"",payload=split>0?value.slice(split+1):"";if(type==="word")encoded.push(`${grouped?"lexical-family":"lexical-exact"}:${payload}`);else if(type==="topic"){const id=legacyTopicId(payload);if(id)encoded.push(id);else messages.push(`unknown topic “${payload}”`);}else if(type==="entity"){const entry=INDEX.catalog.acronyms.find(item=>item.label===payload||item.query===payload);if(entry)encoded.push(entry.series_id);else messages.push(`unknown named acronym “${payload}”`);}});(params.get("words")||"").split(",").filter(Boolean).forEach(value=>encoded.push(`${grouped?"lexical-family":"lexical-exact"}:${value}`));(params.get("topics")||"").split("|").filter(Boolean).forEach(value=>{const id=legacyTopicId(value);if(id)encoded.push(id);else messages.push(`unknown topic “${value}”`);});(params.get("entities")||"").split("|").filter(Boolean).forEach(value=>{const entry=INDEX.catalog.acronyms.find(item=>item.label===value||item.query===value);if(entry)encoded.push(entry.series_id);else messages.push(`unknown named acronym “${value}”`);});if(params.has("combine"))messages.push("combined-word state retired; individual series restored");const periods=(params.get("periods")||"").split(",").filter(Boolean);if(periods.length){const first=periods.find(key=>INDEX.historical_contexts.some(period=>period.key===key));if(first)contextKey=first;if(periods.length>1)messages.push("multiple era highlights reduced to the first recognized historical context");}}
  else if(params.has("preset")&&INDEX.presets[params.get("preset")]){legacy=true;encoded=INDEX.presets[params.get("preset")].words.map(word=>`lexical-exact:${word}`);activePreset=params.get("preset");}
  else encoded=INDEX.default_series.slice();
  return {params,messages,legacy,explicit,encoded};
}
async function restoreFromLocation(fromHistory=false){
  const serial=++restoreSerial,epoch=++selectionEpoch,state=requestedState(),next=[],messages=state.messages.slice();
  for(const encoded of state.encoded){
    if(next.length>=SERIES_LIMIT){messages.push(`selection limit ${SERIES_LIMIT} reached`);break;}
    try{const candidate=await makeSelection(encoded);const duplicate=duplicateReason(candidate,next);if(duplicate)messages.push(duplicate);else next.push(candidate);}
    catch(error){messages.push(error.userMessage||error.message);}
    if(serial!==restoreSerial||epoch!==selectionEpoch)return;
  }
  selections=next;pinnedId="";inspected=null;activePreset=activePreset||selectedPresetKey();renderAll();
  const corrected=state.legacy||messages.length>0;
  if(corrected)writeURL("replace");
  if(messages.length)status(`Skipped or adjusted: ${messages.join("; ")}.`);
  else if(fromHistory)status(`Restored ${selections.length} selected series from browser history.`);
}
function bind(){
  byId("add-word").addEventListener("click",addWord);byId("word-query").addEventListener("keydown",event=>{if(event.key==="Enter"){event.preventDefault();addWord();}});
  byId("catalog-search").addEventListener("input",filterCatalog);
  byId("clear-series").addEventListener("click",()=>{selections=[];activePreset="";pinnedId="";inspected=null;commit("push","All series cleared.");focusAfterRender("",byId("word-query"));});
  byId("reset-explore").addEventListener("click",async()=>{
    const button=byId("reset-explore"),epoch=++selectionEpoch;button.disabled=true;button.setAttribute("aria-busy","true");
    try{const next=[];for(const encoded of INDEX.default_series)next.push(await makeSelection(encoded));if(epoch!==selectionEpoch)return;selections=next;uncertainty=true;contextKey="";activePreset="";pinnedId="";inspected=null;commit("push","Explore reset to tariff, freedom, and border.");}
    catch(error){status("The default series could not be restored; the current selection was retained.");safeError("The default series could not be restored; the current selection was retained.",()=>button.click());}
    finally{button.removeAttribute("aria-busy");button.disabled=false;}
  });
  byId("uncertainty-toggle").addEventListener("change",event=>{uncertainty=event.target.checked;commit("push",uncertainty?"Uncertainty ranges shown.":"Uncertainty ranges hidden; status markers and exact values remain.");});
  byId("context-select").addEventListener("change",event=>{contextKey=event.target.value;commit("push",contextKey?"Historical context shown.":"Historical context cleared.");});
  byId("download-selected").addEventListener("click",downloadSelected);
  byId("exact-values").addEventListener("toggle",()=>{if(byId("exact-values").open)renderExactValues();});byId("exact-series").addEventListener("change",renderExactValues);
  addEventListener("popstate",()=>restoreFromLocation(true));
  root.addEventListener("keydown",event=>{if(event.key==="Escape"&&pinnedId){pinnedId="";applyEmphasis();status("Pinned series cleared.");}});
}
async function initialize(){
  try{
    INDEX=await loadJSON(root.dataset.indexUrl,"index");
    if(INDEX.schema_version!=="explore-index-v2")throw new Error("Explore index schema is incompatible.");
    buildCatalogMap();renderCatalog();renderContexts();renderPresets();bind();setControlsEnabled(true);root.classList.add("is-hydrated");await restoreFromLocation(false);
  }catch(error){setControlsEnabled(false);status("Explore controls could not be loaded. The default chart, exact tables, and downloads remain available below.");safeError("Explore controls could not be loaded. The default fallback remains available.",()=>{fetchCache.delete("index");initialize();});}
}
initialize();
})();
