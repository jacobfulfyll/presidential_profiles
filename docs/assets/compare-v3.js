(() => {
  "use strict";
  const node = document.getElementById("compare-page-data");
  if (!node) return;
  const model = JSON.parse(node.textContent);
  const byId = model.presidents;
  const orderedIds = model.order;
  const slots = ["a", "b", "c"];
  const slotNames = ["A", "B", "C"];
  const selects = slots.map(key => document.getElementById(`compare-${key}`));
  const status = document.getElementById("compare-status");
  const selectionStrip = document.getElementById("compare-selection-strip");
  const agendaRoot = document.getElementById("compare-agenda-stage");
  const evidenceRoot = document.getElementById("compare-evidence-content");
  const state = {
    selectedPresidentIds: [],
    activeRhetoricLayer: "corpus",
    agenda: null,
    evidence: null,
  };

  function make(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function selected() { return state.selectedPresidentIds.filter(Boolean); }
  function fallback(excluded, preferred) {
    if (preferred && !excluded.has(preferred)) return preferred;
    return orderedIds.find(id => !excluded.has(id));
  }
  function normalize(search) {
    const params = new URLSearchParams(search);
    let corrected = false;
    const hasA = params.has("a");
    let a = byId[params.get("a")] ? params.get("a") : model.defaults[0];
    if (params.has("a") && !byId[params.get("a")]) corrected = true;
    let b = byId[params.get("b")] ? params.get("b") : null;
    if (!b && hasA && byId[params.get("a")] && !params.has("b")) {
      const position = orderedIds.indexOf(a);
      b = orderedIds.slice(position + 1).concat(orderedIds.slice(0, position).reverse(), orderedIds).find(id => id !== a);
    }
    if (!b || b === a) {
      if (params.has("b")) corrected = true;
      b = fallback(new Set([a]), model.defaults[1]);
    }
    let c = params.get("c");
    if (c && (!byId[c] || c === a || c === b)) {
      c = null;
      corrected = true;
    }
    Array.from(params.keys()).forEach(key => {
      if (!slots.includes(key)) corrected = true;
    });
    return {selected: [a, b, c || null], corrected};
  }
  function canonicalUrl() {
    const url = new URL(location.href);
    const params = new URLSearchParams();
    params.set("a", state.selectedPresidentIds[0]);
    params.set("b", state.selectedPresidentIds[1]);
    if (state.selectedPresidentIds[2]) params.set("c", state.selectedPresidentIds[2]);
    url.search = params.toString();
    return `${url.pathname}${url.search}${url.hash}`;
  }
  function announce(text) { status.textContent = text; }
  function syncSelects() {
    selects.forEach((select, index) => {
      select.disabled = false;
      select.value = state.selectedPresidentIds[index] || "";
      Array.from(select.options).forEach(option => {
        if (!option.value) { option.disabled = index < 2; return; }
        const usedElsewhere = state.selectedPresidentIds.some((id, slot) => slot !== index && id === option.value);
        option.disabled = usedElsewhere;
        option.textContent = byId[option.value].display_name + (usedElsewhere ? " — already selected" : "");
      });
    });
  }
  function renderSelection() {
    const fragment = document.createDocumentFragment();
    selected().forEach((id, index) => {
      const president = byId[id];
      const article = make("article", `selection-card slot-${index + 1}`);
      const head = make("div", "selection-card-head");
      const img = document.createElement("img");
      img.src = president.portrait_url;
      img.alt = "";
      img.width = 44;
      img.height = 44;
      head.append(img, make("strong", "", `${slotNames[index]} · ${president.display_name}`));
      const meta = make("p", "selection-meta", `${president.party} · ${president.years.first}–${president.years.last}`);
      const support = make("p", "selection-support");
      support.append(
        make("span", "", `${president.source_document_speech_count} source-document speeches`),
        make("span", "", `${president.actual_speaker_appearance_count} actual-speaker appearances`),
      );
      const links = make("p", "selection-links");
      const profile = make("a", "", "Profile"); profile.href = president.profile_url;
      const data = make("a", "", "Profile V3 JSON"); data.href = president.profile_data_url;
      links.append(profile, document.createTextNode(" · "), data);
      article.append(head, meta, support, links);
      fragment.append(article);
    });
    selectionStrip.replaceChildren(fragment);
  }
  function measureRows(layer) {
    const catalog = model.measure_catalogs[layer];
    return selected().map(id => ({president: byId[id], values: byId[id].measures[layer], catalog}));
  }
  function renderExactTables() {
    ["corpus", "ai"].forEach(layer => {
      const body = document.querySelector(`[data-rhetoric-table="${layer}"] tbody`);
      if (!body) return;
      const fragment = document.createDocumentFragment();
      const catalog = model.measure_catalogs[layer];
      catalog.forEach((measure, measureIndex) => {
        const row = document.createElement("tr");
        const heading = make("th", "", measure.label); heading.scope = "row";
        row.append(heading);
        selected().forEach(id => {
          const value = byId[id].measures[layer][measureIndex];
          row.append(make("td", "", `${value.absolute_display} · ${value.rank_display}`));
        });
        fragment.append(row);
      });
      body.replaceChildren(fragment);
      const head = document.querySelector(`[data-rhetoric-table="${layer}"] thead tr`);
      const headerFragment = document.createDocumentFragment();
      const metricHead = make("th", "", "Measure"); metricHead.scope = "col"; headerFragment.append(metricHead);
      selected().forEach(id => { const th = make("th", "", byId[id].display_name); th.scope = "col"; headerFragment.append(th); });
      head.replaceChildren(headerFragment);
    });
  }
  function wrapLabel(value) {
    const words = value.split(" ");
    if (words.length < 3) return value;
    const split = Math.ceil(words.length / 2);
    return `${words.slice(0, split).join(" ")}<br>${words.slice(split).join(" ")}`;
  }
  let plotlyPromise = null;
  function loadPlotly() {
    if (window.Plotly) return Promise.resolve(window.Plotly);
    if (plotlyPromise) return plotlyPromise;
    plotlyPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = model.urls.plotly;
      script.onload = () => resolve(window.Plotly);
      script.onerror = () => reject(new Error("Plotly failed to load"));
      document.head.append(script);
    });
    return plotlyPromise;
  }
  function drawRadar(layer) {
    const target = document.getElementById(`compare-radar-${layer}`);
    if (!target || !window.Plotly || target.closest("[hidden]")) return;
    const catalog = model.measure_catalogs[layer];
    const theta = catalog.map(row => wrapLabel(row.label));
    const traces = [];
    selected().forEach((id, index) => {
      const president = byId[id];
      const values = president.measures[layer].map(row => row.percentile);
      if (values.some(value => value === null)) return;
      traces.push({
        type: "scatterpolar", mode: "lines+markers", name: `${slotNames[index]} · ${president.display_name}`,
        theta: theta.concat(theta.slice(0, 1)), r: values.concat(values.slice(0, 1)),
        line: {color: model.slot_styles[index].color, dash: model.slot_styles[index].dash, width: 3},
        marker: {color: model.slot_styles[index].color, symbol: model.slot_styles[index].plotly_symbol, size: 9},
        fill: "none", hovertemplate: "%{theta}: %{r:.0f}th percentile<extra>%{fullData.name}</extra>",
      });
    });
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const sideMargin = Math.min(112, Math.max(86, target.clientWidth * .29));
    window.Plotly.react(target, traces, {
      margin: {l: sideMargin, r: sideMargin, t: 30, b: 54}, height: 500,
      paper_bgcolor: "#fcfcfb", plot_bgcolor: "#fcfcfb", showlegend: true,
      legend: {orientation: "h", x: 0, y: -0.12, font: {color: "#262522"}},
      polar: {radialaxis: {range: [0, 100], tickvals: [0, 25, 50, 75, 100], gridcolor: "#d8d6d0"}, angularaxis: {gridcolor: "#d8d6d0"}},
      font: {family: model.font, color: "#262522", size: 12}, transition: reduced ? {duration: 0} : {duration: 180},
    }, {responsive: true, displayModeBar: false, staticPlot: false}).then(() => window.Plotly.Plots.resize(target));
  }
  function drawRadars() { drawRadar(state.activeRhetoricLayer); }
  function prepareRhetoric() {
    loadPlotly().then(drawRadars).catch(() => announce("Rhetoric charts could not load; exact tables remain available."));
  }
  function bindRhetoricTabs() {
    const tabs = Array.from(document.querySelectorAll('[data-rhetoric-tab]'));
    function activate(layer, focus) {
      state.activeRhetoricLayer = layer;
      tabs.forEach(tab => {
        const active = tab.dataset.rhetoricTab === layer;
        tab.setAttribute("aria-selected", String(active)); tab.tabIndex = active ? 0 : -1;
        document.getElementById(tab.getAttribute("aria-controls")).hidden = !active;
      });
      if (focus) tabs.find(tab => tab.dataset.rhetoricTab === layer).focus();
      prepareRhetoric();
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => activate(tab.dataset.rhetoricTab, false));
      tab.addEventListener("keydown", event => {
        let next = null;
        if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
        if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = tabs.length - 1;
        if (next !== null) { event.preventDefault(); activate(tabs[next].dataset.rhetoricTab, true); }
      });
    });
  }
  function renderEvidence(index) {
    const recordMap = new Map(index.presidents.map(row => [row.president_profile_id, row]));
    const fragment = document.createDocumentFragment();
    const footprint = make("div", "footprint-grid");
    selected().forEach((id, slot) => {
      const record = recordMap.get(id);
      const article = make("article", `footprint-card slot-${slot + 1}`);
      article.append(make("h3", "", `${slotNames[slot]} · ${byId[id].display_name}`));
      article.append(make("p", "", `${record.footprint.source_document_speech_count} speeches · ${record.footprint.source_document_paragraph_count.toLocaleString()} paragraphs · ${record.footprint.source_document_word_count.toLocaleString()} words`));
      if (record.footprint.support_state === "thin") article.append(make("span", "support-pill thin", "Thin record"));
      footprint.append(article);
    });
    fragment.append(footprint);
    const families = [
      ["adversarial_entities", "Adversarial entities", "Exploratory AI extraction · source-document paragraphs · raw mention counts"],
      ["presidential_invocations", "Presidential invocations", "Invocation v2 · source-document paragraphs · raw classified mentions"],
      ["distinctive_vocabulary", "Distinctive vocabulary", "Ranked terms that are statistically distinctive against all other president records"],
      ["signature_speeches", "Signature speeches", "Speech-level era-adjusted embedding alignment · cosine score"],
    ];
    families.forEach(([key, label, method]) => {
      const details = make("details", "evidence-family");
      details.append(make("summary", "", label));
      details.append(make("p", "method-label", method));
      const grid = make("div", "evidence-grid");
      selected().forEach((id, slot) => {
        const section = make("section", `evidence-president slot-${slot + 1}`);
        section.append(make("h3", "", `${slotNames[slot]} · ${byId[id].display_name}`));
        const values = recordMap.get(id)[key];
        if (!values.length) {
          section.append(make("p", "empty-state", key === "presidential_invocations" ? "None observed in the completed v2 artifact." : "None observed in the completed artifact."));
        } else {
          const list = make("ol", "evidence-list");
          values.forEach(item => {
            const li = document.createElement("li");
            if (key === "adversarial_entities") li.append(make("strong", "", `${item.label} · ${item.mention_count} mentions`));
            if (key === "presidential_invocations") li.append(make("strong", "", `${item.target} · ${item.function} · ${item.stance} · ${item.mention_count}`));
            if (key === "distinctive_vocabulary") li.append(make("strong", "", `#${Number(item.rank) + 1} ${item.term}`));
            if (key === "signature_speeches") {
              const link = make("a", "", `${item.title} (${item.year}) · ${Number(item.cosine_score).toFixed(3)}`);
              link.href = item.source_url; link.target = "_blank"; link.rel = "noopener"; li.append(link);
            }
            list.append(li);
          });
          section.append(list);
        }
        grid.append(section);
      });
      details.append(grid); fragment.append(details);
    });
    evidenceRoot.replaceChildren(fragment);
  }
  function loadEvidence() {
    if (state.evidence) { renderEvidence(state.evidence); return; }
    evidenceRoot.setAttribute("aria-busy", "true");
    fetch(model.urls.evidence_index).then(response => {
      if (!response.ok) throw new Error(String(response.status));
      return response.json();
    }).then(index => {
      if (index.schema_version !== "compare-evidence-index-v1") throw new Error("schema");
      state.evidence = index; evidenceRoot.removeAttribute("aria-busy"); renderEvidence(index);
    }).catch(() => {
      evidenceRoot.removeAttribute("aria-busy");
      const alert = make("div", "load-error"); alert.setAttribute("role", "alert");
      alert.append(make("p", "", "Evidence details could not load. Profile and CSV links remain available."));
      const retry = make("button", "", "Retry"); retry.type = "button"; retry.addEventListener("click", loadEvidence); alert.append(retry);
      evidenceRoot.replaceChildren(alert);
    });
  }
  function applyState(next, options) {
    state.selectedPresidentIds = next.selected;
    syncSelects(); renderSelection(); renderExactTables();
    if (window.Plotly) drawRadars();
    if (state.agenda) state.agenda.selectionChanged();
    if (state.evidence) renderEvidence(state.evidence);
    if (options.history === "push") history.pushState(null, "", canonicalUrl());
    if (options.history === "replace") history.replaceState(null, "", canonicalUrl());
    if (options.announce) announce(options.announce);
  }
  selects.forEach((select, index) => select.addEventListener("change", () => {
    const next = state.selectedPresidentIds.slice(); next[index] = select.value || null;
    applyState({selected: next}, {
      history: "push", announce: `Comparison updated: ${selected().map(id => byId[id].display_name).join(", ")}.`,
    });
  }));
  window.addEventListener("popstate", () => applyState(normalize(location.search), {history: "none"}));
  bindRhetoricTabs();
  const initial = normalize(location.search);
  applyState(initial, {history: initial.corrected ? "replace" : "none", announce: initial.corrected ? "Invalid comparison options were corrected." : ""});

  state.agenda = window.CompareAgenda.mount({
    root: agendaRoot,
    indexUrl: model.urls.agenda_index,
    selected,
    president: id => byId[id],
    slotStyles: model.slot_styles,
    announce,
  });
  const rhetoric = document.getElementById("rhetoric");
  const evidence = document.getElementById("evidence");
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      if (entry.target === rhetoric) prepareRhetoric();
      if (entry.target === agendaRoot) state.agenda.load();
      if (entry.target === evidence) loadEvidence();
      observer.unobserve(entry.target);
    }), {rootMargin: "600px 0px"});
    observer.observe(rhetoric); observer.observe(agendaRoot); observer.observe(evidence);
  } else { prepareRhetoric(); state.agenda.load(); loadEvidence(); }
  [rhetoric, agendaRoot, evidence].forEach(target => target.addEventListener("focusin", () => {
    if (target === rhetoric) prepareRhetoric();
    if (target === agendaRoot) state.agenda.load();
    if (target === evidence) loadEvidence();
  }, {once: true}));

  const download = document.getElementById("compare-download");
  download.addEventListener("click", () => {
    download.disabled = true; download.textContent = "Preparing…";
    Promise.all(selected().map(id => fetch(byId[id].profile_data_url).then(response => {
      if (!response.ok) throw new Error(String(response.status)); return response.json();
    }))).then(records => {
      const output = {schema_version: "president-comparison-v3", population: "source-document Profile V3 records", profile_schema_version: model.profile_schema_version, selected_president_ids: selected(), source_urls: selected().map(id => byId[id].profile_data_url), profiles: records};
      const blob = new Blob([JSON.stringify(output, null, 2) + "\n"], {type: "application/json"});
      const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "president-comparison-v3.json"; link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 0);
      announce("Selected source-document Profile V3 bundle prepared.");
    }).catch(() => announce("The selected Profile V3 bundle could not be prepared; direct Profile JSON links remain available."))
      .finally(() => { download.disabled = false; download.textContent = "Download selected Profile V3 records"; });
  });
  document.documentElement.classList.add("compare-hydrated");
})();