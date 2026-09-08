"""Page-specific, safe-DOM assets for Compare V3."""

COMPARE_V3_JS = r'''(() => {
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
  const compactRadarMedia = matchMedia("(max-width: 760px)");
  let radarResizeTimer = null;
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
    const compactRadar = compactRadarMedia.matches;
    const compactDiameter = Math.max(220, Math.min(430, target.clientWidth - 64));
    const sideMargin = compactRadar
      ? Math.max(18, (target.clientWidth - compactDiameter) / 2)
      : Math.min(112, Math.max(86, target.clientWidth * .29));
    const topMargin = compactRadar ? 24 : 30;
    const bottomMargin = compactRadar ? 84 : 54;
    const radarHeight = compactRadar
      ? Math.ceil(compactDiameter + topMargin + bottomMargin)
      : 500;
    const revision = Number(target.dataset.radarRevision || "0") + 1;
    target.dataset.radarRevision = String(revision);
    return window.Plotly.react(target, traces, {
      margin: {l: sideMargin, r: sideMargin, t: topMargin, b: bottomMargin}, height: radarHeight,
      paper_bgcolor: "#fcfcfb", plot_bgcolor: "#fcfcfb", showlegend: true,
      legend: {orientation: "h", x: 0, y: compactRadar ? -0.16 : -0.12, font: {color: "#262522"}},
      polar: {radialaxis: {range: [0, 100], tickvals: [0, 25, 50, 75, 100], gridcolor: "#d8d6d0"}, angularaxis: {gridcolor: "#d8d6d0"}},
      font: {family: model.font, color: "#262522", size: 12}, transition: reduced ? {duration: 0} : {duration: 180},
    }, {responsive: true, displayModeBar: false, staticPlot: false}).then(() => {
      if (!target.isConnected || target.closest("[hidden]") || Number(target.dataset.radarRevision) !== revision) return;
      return window.Plotly.Plots.resize(target);
    });
  }
  function drawRadars() { return drawRadar(state.activeRhetoricLayer); }
  function scheduleRadarRedraw(event) {
    if (!compactRadarMedia.matches && (!event || event.type !== "change")) return;
    if (radarResizeTimer !== null) clearTimeout(radarResizeTimer);
    radarResizeTimer = setTimeout(() => {
      radarResizeTimer = null;
      if (!window.Plotly) return;
      Promise.resolve(drawRadars()).catch(() => announce("Rhetoric charts could not redraw; exact tables remain available."));
    }, 120);
  }
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
  window.addEventListener("resize", scheduleRadarRedraw, {passive: true});
  window.addEventListener("orientationchange", scheduleRadarRedraw, {passive: true});
  if (compactRadarMedia.addEventListener) compactRadarMedia.addEventListener("change", scheduleRadarRedraw);
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
})();'''


AGENDA_COMPARISON_JS = r'''(() => {
  "use strict";

  const FOCUSED_TOPIC_LIMIT = 3;
  const SLOT_NAMES = ["A", "B", "C"];
  const SLOT_SYMBOLS = ["●", "■", "◆"];

  function make(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function percent(value) {
    return (Number(value) * 100).toFixed(2) + "%";
  }

  function supportText(state) {
    return ({
      supported: "supported",
      thin_edge: "thin relationship",
      thin_president: "thin president record",
      supported_zero: "supported zero",
      thin_zero: "thin zero",
      unavailable: "unavailable",
    })[state] || state;
  }

  function mount(options) {
    const root = options.root;
    const broadRoot = document.getElementById("agenda-broad");
    const runtimeStatus = document.getElementById("agenda-runtime-status");
    const state = {
      index: null,
      shards: new Map(),
      pinned: null,
      loading: null,
    };

    function supportMap() {
      return new Map(
        state.index.president_support.map(row => [row.president_profile_id, row])
      );
    }

    function cellMap(cells) {
      return new Map(
        cells.map(row => [row.topic_id + "|" + row.president_profile_id, row])
      );
    }

    function focusedReasons(cells) {
      const reasons = new Map();
      options.selected().forEach((presidentId, slot) => {
        const ranked = state.index.level1_topics.map((topic, order) => ({
          topic,
          order,
          cell: cells.get(topic.topic_id + "|" + presidentId),
        })).filter(item => (
          item.cell
          && item.cell.speaker_paragraph_share !== null
          && Number.isFinite(Number(item.cell.speaker_paragraph_share))
        )).sort((left, right) => (
          Number(right.cell.speaker_paragraph_share)
          - Number(left.cell.speaker_paragraph_share)
          || left.order - right.order
        )).slice(0, FOCUSED_TOPIC_LIMIT);
        ranked.forEach(item => {
          const slots = reasons.get(item.topic.topic_id) || [];
          slots.push(slot);
          reasons.set(item.topic.topic_id, slots);
        });
      });
      return reasons;
    }

    function reasonText(slots) {
      const names = slots.map(slot => {
        const president = options.president(options.selected()[slot]);
        return president.short_name || president.display_name;
      });
      return "Top 3 for " + names.join(" + ");
    }

    function markLabel(topic, presidentId, cell) {
      const president = options.president(presidentId);
      if (!cell || cell.state === "unavailable") {
        return topic.topic_label + " · " + president.display_name
          + " · unavailable · actual-speaker population";
      }
      const support = supportMap().get(presidentId);
      return topic.topic_label + " · " + president.display_name + " · "
        + percent(cell.speaker_paragraph_share) + " · "
        + cell.topic_paragraph_count + " topic paragraphs of "
        + cell.eligible_president_paragraph_count + " eligible paragraphs · "
        + cell.topic_appearance_count + " topic-bearing appearances of "
        + support.eligible_president_appearance_count
        + " actual-speaker appearances · " + supportText(cell.state);
    }

    function plotMark(slot, cell) {
      const mark = make(
        "span",
        "agenda-plot-mark slot-" + (slot + 1)
          + " state-" + (cell ? cell.state : "unavailable"),
        SLOT_SYMBOLS[slot]
      );
      const share = cell && cell.speaker_paragraph_share !== null
        ? Number(cell.speaker_paragraph_share) * 100
        : 0;
      mark.style.setProperty("--position", Math.min(100, share) + "%");
      mark.setAttribute("aria-hidden", "true");
      return mark;
    }

    function emptyDetail(detail) {
      detail.hidden = true;
      detail.removeAttribute("aria-busy");
      detail.replaceChildren();
    }

    function firstReceipt(shard, topic, presidentId) {
      const receipts = shard.evidence_receipts.filter(row => (
        row.topic_id === topic.topic_id
        && row.president_profile_id === presidentId
      ));
      return receipts.find(row => row.selection_role === "first") || receipts[0] || null;
    }

    function showDetail(detail, topic, presidentId, cell, shard, trigger) {
      detail.hidden = false;
      detail.removeAttribute("aria-busy");
      const heading = make(
        "h4",
        "",
        topic.topic_label + " · " + options.president(presidentId).display_name
      );
      const summary = make("p", "detail-value");
      if (!cell || cell.speaker_paragraph_share === null) {
        summary.textContent = "N/A · actual-speaker paragraphs";
      } else {
        summary.textContent = percent(cell.speaker_paragraph_share) + " · "
          + cell.topic_paragraph_count + " topic paragraphs of "
          + cell.eligible_president_paragraph_count + " eligible paragraphs · "
          + cell.topic_appearance_count + " topic-bearing appearances · "
          + supportText(cell.state);
      }
      const exampleLabel = make("p", "example-label", "One topic example");
      const receipt = firstReceipt(shard, topic, presidentId);
      if (!receipt) {
        detail.replaceChildren(
          heading,
          summary,
          exampleLabel,
          make("p", "empty-state", "No example is published for this zero or unavailable relationship.")
        );
      } else {
        const block = make("blockquote", "receipt");
        block.append(make("p", "", receipt.evidence_excerpt));
        const link = make(
          "a",
          "",
          receipt.speech_date + " · " + receipt.speech_title
            + " · paragraph " + receipt.para_idx
        );
        link.href = receipt.source_url;
        link.target = "_blank";
        link.rel = "noopener";
        block.append(link);
        detail.replaceChildren(heading, summary, exampleLabel, block);
      }
      detail.dataset.returnTarget = trigger.dataset.markKey || "";
    }

    function parentMeta(parentId) {
      return state.index.level1_topics.find(row => row.topic_id === parentId);
    }

    function loadShard(parentId) {
      if (state.shards.has(parentId)) {
        return Promise.resolve(state.shards.get(parentId));
      }
      return fetch(parentMeta(parentId).shard_url)
        .then(response => {
          if (!response.ok) throw new Error(String(response.status));
          return response.json();
        })
        .then(shard => {
          if (
            shard.schema_version !== "compare-agenda-topic-v1"
            || shard.parent_topic.topic_id !== parentId
          ) {
            throw new Error("schema");
          }
          state.shards.set(parentId, shard);
          return shard;
        });
    }

    function requestDetail(detail, topic, presidentId, cell, trigger, key) {
      detail.hidden = false;
      detail.setAttribute("aria-busy", "true");
      detail.replaceChildren(make("p", "loading", "Loading one topic example…"));
      loadShard(topic.topic_id).then(shard => {
        if (state.pinned !== key) return;
        showDetail(detail, topic, presidentId, cell, shard, trigger);
      }).catch(() => {
        if (state.pinned !== key) return;
        detail.removeAttribute("aria-busy");
        const alert = make("div", "load-error");
        alert.setAttribute("role", "alert");
        alert.append(make("p", "", "The topic example could not load. Exact counts remain available."));
        const retry = make("button", "", "Retry");
        retry.type = "button";
        retry.addEventListener("click", () => (
          requestDetail(detail, topic, presidentId, cell, trigger, key)
        ));
        alert.append(retry);
        detail.replaceChildren(alert);
      });
    }

    function activateMark(detail, topic, presidentId, cell, button, key) {
      if (state.pinned === key) {
        state.pinned = null;
        root.querySelectorAll(".agenda-mark").forEach(mark => (
          mark.setAttribute("aria-pressed", "false")
        ));
        root.querySelectorAll(".agenda-detail").forEach(emptyDetail);
        return;
      }
      state.pinned = key;
      root.querySelectorAll(".agenda-mark").forEach(mark => (
        mark.setAttribute("aria-pressed", String(mark.dataset.markKey === key))
      ));
      root.querySelectorAll(".agenda-detail").forEach(emptyDetail);
      requestDetail(detail, topic, presidentId, cell, button, key);
    }

    function markButton(slot, presidentId, label, cell, key, activate) {
      const president = options.president(presidentId);
      const button = make(
        "button",
        "agenda-mark slot-" + (slot + 1)
          + " state-" + (cell ? cell.state : "unavailable")
      );
      button.type = "button";
      button.dataset.markKey = key;
      button.setAttribute("aria-label", label);
      button.setAttribute("aria-pressed", String(state.pinned === key));
      button.title = label;
      button.append(
        make("span", "mark-symbol", SLOT_SYMBOLS[slot]),
        make(
          "span",
          "mark-president",
          SLOT_NAMES[slot] + " · " + (president.short_name || president.display_name)
        ),
        make(
          "span",
          "mark-value",
          cell && cell.speaker_paragraph_share !== null
            ? percent(cell.speaker_paragraph_share)
              + (cell.state.includes("thin") ? " · thin" : "")
            : "N/A"
        )
      );
      button.addEventListener("click", () => activate(button, key));
      button.addEventListener("keydown", event => {
        if (event.key !== "Escape" || state.pinned !== key) return;
        event.preventDefault();
        state.pinned = null;
        root.querySelectorAll(".agenda-mark").forEach(mark => (
          mark.setAttribute("aria-pressed", "false")
        ));
        root.querySelectorAll(".agenda-detail").forEach(emptyDetail);
        button.focus();
      });
      return button;
    }

    function valueGrid(cells, topic, detail) {
      const plot = make("div", "agenda-row-plot");
      const values = make("div", "agenda-row-values");
      values.style.setProperty("--president-count", String(cells.length));
      cells.forEach((cell, slot) => {
        const presidentId = options.selected()[slot];
        const key = topic.topic_level + "|" + topic.topic_id + "|" + presidentId;
        plot.append(plotMark(slot, cell));
        values.append(markButton(
          slot,
          presidentId,
          markLabel(topic, presidentId, cell),
          cell,
          key,
          button => activateMark(detail, topic, presidentId, cell, button, key)
        ));
      });
      return {plot, values};
    }

    function renderBroad() {
      const cells = cellMap(state.index.broad_cells);
      const reasons = focusedReasons(cells);
      const visibleTopics = state.index.level1_topics.filter(topic => (
        reasons.has(topic.topic_id)
      ));
      const fragment = document.createDocumentFragment();
      const chart = make("div", "agenda-chart broad-chart");
      visibleTopics.forEach(topic => {
        const row = make("section", "agenda-row broad-row");
        const detail = make("aside", "agenda-detail");
        detail.setAttribute("aria-live", "polite");
        emptyDetail(detail);
        const heading = make("h3", "agenda-row-label");
        heading.append(
          make("span", "topic-label", topic.topic_label),
          make("span", "agenda-reason", reasonText(reasons.get(topic.topic_id)))
        );
        row.append(heading);
        const selectedCells = options.selected().map(
          id => cells.get(topic.topic_id + "|" + id)
        );
        const pieces = valueGrid(selectedCells, topic, detail);
        row.append(pieces.plot, pieces.values, detail);
        chart.append(row);
      });
      fragment.append(chart);
      broadRoot.replaceChildren(fragment);
    }

    function load() {
      if (state.index) {
        renderBroad();
        return Promise.resolve(state.index);
      }
      if (state.loading) return state.loading;
      root.setAttribute("aria-busy", "true");
      state.loading = fetch(options.indexUrl)
        .then(response => {
          if (!response.ok) throw new Error(String(response.status));
          return response.json();
        })
        .then(index => {
          if (index.schema_version !== "compare-agenda-index-v1") {
            throw new Error("schema");
          }
          state.index = index;
          root.removeAttribute("aria-busy");
          runtimeStatus.replaceChildren();
          renderBroad();
          return index;
        })
        .catch(error => {
          root.removeAttribute("aria-busy");
          state.loading = null;
          const alert = make("div", "load-error");
          alert.setAttribute("role", "alert");
          alert.append(
            make(
              "p",
              "",
              "Topic data could not load. The server-rendered default and CSV download remain available."
            )
          );
          const retry = make("button", "", "Retry");
          retry.type = "button";
          retry.addEventListener("click", load);
          alert.append(retry);
          runtimeStatus.replaceChildren(alert);
          throw error;
        });
      return state.loading;
    }

    return {
      load,
      selectionChanged() {
        state.pinned = null;
        if (state.index) renderBroad();
      },
    };
  }

  window.CompareAgenda = {mount};
})();'''

COMPARE_V3_CSS = r'''
body.compare-page {
  overflow-x: clip;
}
.compare-page :where(header, main, section, article, nav, div) {
  min-width: 0;
}
.compare-page .global-nav :where(a, button) {
  min-height: 44px;
}
.compare-hero {
  padding-top: 28px;
}
.compare-hero h1 {
  max-width: 760px;
  margin: 8px 0;
  font: 700 clamp(2.2rem, 6vw, 4rem)/1.02 Georgia, serif;
  letter-spacing: -.035em;
}
.compare-hero .sub {
  max-width: 760px;
  color: var(--ink2);
  font-size: 1.03rem;
}
.compare-controls {
  margin-top: 24px;
  padding: 18px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--surface);
}
.compare-controls legend {
  padding: 0 7px;
  font-weight: 800;
}
.selector-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
.selector-grid label {
  display: grid;
  gap: 6px;
  color: var(--ink2);
  font-size: .78rem;
  font-weight: 750;
}
.selector-grid select {
  width: 100%;
  min-height: 44px;
  padding: 8px 10px;
  border: 1px solid #6e6b65;
  border-radius: 8px;
  background: #fff;
  color: var(--ink);
  font: inherit;
}
.interactive-note,
#compare-status {
  margin: 10px 0 0;
  color: var(--muted);
  font-size: .78rem;
}
.selection-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-top: 14px;
}
.selection-card {
  padding: 13px;
  border: 1px solid var(--border);
  border-top: 4px solid var(--slot-color);
  border-radius: 10px;
  background: var(--surface);
}
.slot-1 {
  --slot-color: #2a78d6;
}
.slot-2 {
  --slot-color: #9b6200;
}
.slot-3 {
  --slot-color: #008300;
}
.selection-card-head {
  display: flex;
  align-items: center;
  gap: 9px;
}
.selection-card-head img {
  border-radius: 50%;
  object-fit: cover;
}
.selection-meta,
.selection-support,
.selection-links {
  margin: 6px 0 0;
  font-size: .76rem;
}
.selection-support {
  display: grid;
  color: var(--ink2);
}
.compare-nav {
  position: sticky;
  z-index: 8;
  top: calc(var(--global-nav-height, 0px) + 5px);
  width: max-content;
  max-width: 100%;
  margin: 18px auto;
}
.compare-nav ul {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 5px;
  margin: 0;
  padding: 5px;
  border: 1px solid var(--border);
  border-radius: 11px;
  background: rgba(252, 252, 251, .96);
  list-style: none;
}
.compare-nav a {
  display: grid;
  min-height: 44px;
  place-items: center;
  padding: 7px 14px;
  border-radius: 7px;
  color: var(--ink2);
  font-weight: 750;
  text-decoration: none;
}
.compare-nav a:hover,
.compare-nav a:focus-visible {
  background: #f0efeb;
  color: var(--ink);
}
.compare-page main {
  max-width: 1120px;
  margin: auto;
  padding: 0 24px 80px;
}
.compare-page main > section {
  padding-top: 44px;
  scroll-margin-top: 100px;
}
.section-kicker {
  margin: 0 0 5px;
  color: #6a421b;
  font-size: .7rem;
  font-weight: 800;
  letter-spacing: .09em;
  text-transform: uppercase;
}
.compare-page h2 {
  margin: 0;
  font: 700 clamp(1.65rem, 4vw, 2.35rem)/1.12 Georgia, serif;
}
.section-intro {
  max-width: 760px;
  color: var(--ink2);
}
.tablist {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 18px 0 14px;
}
.rhetoric-tabs {
  width: max-content;
  max-width: 100%;
  justify-content: center;
  margin-inline: auto;
}
.tablist button {
  min-height: 44px;
  padding: 8px 14px;
  border: 1px solid #817d75;
  border-radius: 8px;
  background: #fff;
  color: var(--ink);
  font: inherit;
  font-weight: 720;
}
.tablist button[aria-selected=true] {
  border-color: #315f78;
  background: #eaf2f8;
  color: #183c50;
}
.tablist button:focus-visible,
.agenda-mark:focus-visible,
button:focus-visible,
select:focus-visible,
a:focus-visible,
summary:focus-visible {
  outline: 3px solid #0b5cad;
  outline-offset: 3px;
}
.rhetoric-panel {
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--surface);
}
.radar-chart {
  min-height: 500px;
}
.rhetoric-fallback {
  max-width: 700px;
  margin: 32px auto;
  color: var(--muted);
  text-align: center;
}
.exact-disclosure,
.evidence-family {
  margin-top: 14px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface);
}
.exact-disclosure > summary,
.evidence-family > summary {
  min-height: 44px;
  padding: 12px 15px;
  font-weight: 780;
  cursor: pointer;
}
.exact-wrap {
  overflow-x: auto;
  padding: 0 14px 14px;
}
.exact-table {
  width: 100%;
  border-collapse: collapse;
  font-size: .82rem;
}
.exact-table th,
.exact-table td {
  padding: 9px;
  border-top: 1px solid var(--grid);
  text-align: left;
  vertical-align: top;
}
.exact-table thead th {
  color: var(--muted);
  font-size: .7rem;
}
.method-label {
  max-width: 820px;
  padding: 10px 12px;
  border-left: 4px solid #557b9c;
  background: #f2f6f8;
  color: #354b59;
  font-size: .8rem;
}
.loading,
.empty-state {
  color: var(--muted);
}
.footprint-grid,
.evidence-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.footprint-card,
.evidence-president {
  padding: 13px;
  border: 1px solid var(--border);
  border-top: 4px solid var(--slot-color);
  border-radius: 9px;
  background: var(--surface);
}
.footprint-card h3,
.evidence-president h3 {
  margin: 0;
  font-size: .9rem;
}
.footprint-card p {
  margin: 6px 0;
}
.support-pill {
  display: inline-flex;
  padding: 2px 7px;
  border-radius: 99px;
  background: #edf5ee;
  color: #264f32;
  font-size: .7rem;
}
.support-pill.thin {
  background: #fff2cf;
  color: #614b19;
}
.evidence-family {
  padding-bottom: 14px;
}
.evidence-family > .method-label {
  margin: 0 14px 12px;
}
.evidence-grid {
  padding: 0 14px;
}
.evidence-list {
  display: grid;
  gap: 9px;
  margin: 8px 0 0;
  padding-left: 20px;
  font-size: .78rem;
}
.download-panel {
  display: flex;
  flex-wrap: wrap;
  gap: 9px;
  margin-top: 18px;
  padding: 15px;
  border: 1px solid var(--border);
  border-radius: 10px;
}
.download-panel a,
.download-panel button {
  display: inline-flex;
  min-height: 44px;
  align-items: center;
  padding: 8px 12px;
  border: 1px solid #315f78;
  border-radius: 8px;
  background: #fff;
  color: #214f68;
  font: inherit;
  font-weight: 730;
  text-decoration: none;
}
.load-error {
  padding: 14px;
  border: 1px solid #9d4f48;
  border-radius: 9px;
  background: #fff4f2;
  color: #652f2a;
}
.load-error button {
  min-height: 44px;
}
.compare-page footer {
  max-width: 1120px;
  margin: auto;
  padding: 30px 24px;
  color: var(--muted);
  font-size: .8rem;
}
.compare-hydrated .interactive-note {
  display: none;
}
@media (max-width: 760px) {
  .selector-grid,
  .selection-strip,
  .footprint-grid,
  .evidence-grid {
    grid-template-columns: 1fr;
  }
  .compare-page main {
    padding-inline: 14px;
  }
  .radar-chart {
    min-height: 328px;
  }
  .compare-nav {
    position: static;
  }
}
@media (max-width: 430px) {
  .compare-page main {
    padding-inline: 10px;
  }
  .rhetoric-panel {
    padding: 5px;
  }
  .exact-table {
    font-size: .72rem;
  }
}
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    scroll-behavior: auto !important;
    transition-duration: 0s !important;
    animation-duration: 0s !important;
  }
}
@media (forced-colors: active) {
  .agenda-mark,
  .selection-card,
  .footprint-card,
  .evidence-president {
    border-color: CanvasText;
  }
  .agenda-mark .mark-symbol {
    color: CanvasText;
  }
}
'''


AGENDA_COMPARISON_CSS = r'''
.agenda-chart {
  display: grid;
  gap: 7px;
  margin-top: 18px;
}
.agenda-row {
  display: grid;
  grid-template-columns: minmax(190px, 28%) minmax(150px, 32%) minmax(260px, 40%);
  align-items: center;
  gap: 12px;
  min-height: 58px;
  margin: 0;
  padding: 7px 10px;
  border: 1px solid #dedbd4;
  border-radius: 9px;
  background: var(--surface, Canvas);
}
.agenda-row-label {
  margin: 0;
  font-size: .82rem;
  overflow-wrap: anywhere;
}
.agenda-row-label .topic-label {
  display: block;
  font-weight: 780;
}
.agenda-reason {
  display: block;
  width: max-content;
  max-width: 100%;
  margin-top: 4px;
  padding: 2px 6px;
  border-radius: 999px;
  background: #edf3f7;
  color: #3e647d;
  font-size: .62rem;
  font-weight: 720;
  line-height: 1.35;
  overflow-wrap: anywhere;
}
.agenda-row-plot {
  position: relative;
  min-height: 50px;
  background:
    linear-gradient(to right, #ddd 1px, transparent 1px) 0 0 / 25% 100%;
}
.agenda-row-plot::before {
  position: absolute;
  inset: 24px 0 auto;
  height: 2px;
  background: #c7c5bf;
  content: "";
}
.agenda-plot-mark {
  position: absolute;
  z-index: 1;
  left: clamp(8px, var(--position), calc(100% - 8px));
  top: var(--mark-y);
  color: var(--slot-color, CanvasText);
  font-size: 1rem;
  line-height: 1;
  transform: translate(-50%, -50%);
}
.agenda-plot-mark.slot-1 {
  --mark-y: 8px;
}
.agenda-plot-mark.slot-2 {
  --mark-y: 25px;
}
.agenda-plot-mark.slot-3 {
  --mark-y: 42px;
}
.agenda-plot-mark.slot-1::after,
.agenda-plot-mark.slot-3::after {
  position: absolute;
  z-index: -1;
  left: 50%;
  height: 17px;
  border-left: 1px solid var(--slot-color, CanvasText);
  content: "";
}
.agenda-plot-mark.slot-1::after {
  top: 50%;
}
.agenda-plot-mark.slot-3::after {
  bottom: 50%;
}
.agenda-plot-mark.state-thin_edge,
.agenda-plot-mark.state-thin_president,
.agenda-plot-mark.state-thin_zero {
  opacity: .72;
  text-shadow: 0 0 0 var(--surface, Canvas);
}
.agenda-row-values {
  display: grid;
  grid-template-columns: repeat(var(--president-count), minmax(0, 1fr));
  gap: 5px;
}
.agenda-mark {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  min-width: 0;
  min-height: 44px;
  align-items: center;
  gap: 1px 5px;
  padding: 5px 6px;
  border: 1px solid transparent;
  border-radius: 6px;
  background: transparent;
  color: var(--ink, CanvasText);
  font: inherit;
  font-size: .7rem;
  font-variant-numeric: tabular-nums;
  text-align: left;
}
button.agenda-mark {
  cursor: pointer;
}
button.agenda-mark:hover {
  border-color: var(--slot-color, CanvasText);
  background: #f7f8f8;
}
.agenda-mark[aria-pressed=true] {
  border-color: var(--slot-color, CanvasText);
  background: #f4f7f9;
}
.agenda-mark .mark-symbol {
  grid-row: 1 / 3;
  color: var(--slot-color, CanvasText);
  font-size: .95rem;
}
.agenda-mark .mark-president {
  min-width: 0;
  font-weight: 800;
  overflow-wrap: anywhere;
}
.agenda-mark .mark-value {
  grid-column: 2;
}
.agenda-mark.state-thin_edge,
.agenda-mark.state-thin_president,
.agenda-mark.state-thin_zero {
  border-bottom-color: var(--slot-color, CanvasText);
  border-bottom-style: dashed;
}
.agenda-mark.state-unavailable {
  border-bottom-color: var(--muted, GrayText);
  border-bottom-style: dotted;
  color: var(--muted, GrayText);
}
.agenda-static-value {
  border-color: transparent;
}
.agenda-detail {
  grid-column: 1 / -1;
  width: 100%;
  margin: 4px 0 2px;
  padding: 15px;
  border: 1px solid #8ca5b7;
  border-radius: 10px;
  background: #eef6fb;
}
.agenda-detail h4,
.detail-value {
  margin: 0 0 7px;
}
.example-label {
  margin: 14px 0 0;
  color: #315f78;
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.receipt {
  margin: 7px 0 0;
  padding: 10px 12px;
  border-left: 3px solid #8ca5b7;
  background: #fff;
  color: var(--ink2, CanvasText);
  font-size: .78rem;
}
.receipt p {
  margin: 0 0 5px;
}
.agenda-runtime-status:empty {
  display: none;
}
@media (max-width: 820px) {
  .agenda-row {
    grid-template-columns: minmax(155px, 32%) minmax(130px, 28%) minmax(240px, 40%);
  }
}
@media (max-width: 760px) {
  .agenda-row {
    grid-template-columns: 1fr;
    gap: 5px;
    padding: 8px;
  }
  .agenda-row-values {
    grid-template-columns: repeat(var(--president-count), minmax(0, 1fr));
  }
}
@media (max-width: 430px) {
  .agenda-mark {
    padding: 4px;
    font-size: .64rem;
  }
}
@media (prefers-reduced-motion: reduce) {
  #compare-agenda-stage * {
    scroll-behavior: auto !important;
    transition-duration: 0s !important;
    animation-duration: 0s !important;
  }
}
@media (forced-colors: active) {
  .agenda-mark,
  .agenda-row {
    border-color: CanvasText;
  }
  .agenda-plot-mark,
  .agenda-mark .mark-symbol {
    color: CanvasText;
  }
  .agenda-detail {
    background: Canvas;
  }
}
'''
