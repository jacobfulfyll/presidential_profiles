"""Shared generated assets for the Issues V2 directory and detail pages."""

ISSUES_CSS = r"""
.skip-link{position:fixed;z-index:1000;left:12px;top:10px;transform:translateY(-160%);
  padding:9px 13px;border-radius:8px;background:#172a38;color:#fff;font-weight:750}
.skip-link:focus{transform:none}.issues-page{overflow-x:clip}.issues-page :where(header,main,
  section,article,nav,div,figure){min-width:0}.issues-hero{padding-top:30px}
.breadcrumbs{display:flex;flex-wrap:wrap;gap:5px;color:var(--muted);font-size:.82rem}
.breadcrumbs a{color:var(--ink2)}.issues-eyebrow{margin-top:18px;color:#6a421b;font-size:.7rem;
  font-weight:820;letter-spacing:.09em;text-transform:uppercase}.issues-hero h1{margin-top:4px;
  font:700 clamp(2rem,5vw,3.1rem)/1.04 Georgia,serif;letter-spacing:-.025em}
.issues-hero .sub{max-width:53rem}.source-badge{display:inline-flex;align-items:center;min-height:28px;
  margin-top:12px;padding:3px 9px;border:1px solid;border-radius:999px;font-size:.68rem;
  font-weight:820;letter-spacing:.045em;text-transform:uppercase}.source-badge.anchored{
  border-color:#7d6849;background:#f8f2e8;color:#604519}.source-badge.surfaced{
  border-color:#557b9c;background:#eef6ff;color:#254e70}.issue-facts{display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:20px}.issue-fact{
  padding:13px 14px;border:1px solid var(--border);border-radius:10px;background:var(--surface)}
.issue-fact span,.issue-fact strong,.issue-fact small{display:block}.issue-fact span{color:var(--muted);
  font-size:.7rem;font-weight:760}.issue-fact strong{margin-top:3px;line-height:1.3}.issue-fact small{
  margin-top:3px;color:var(--muted);font-size:.72rem}.issue-local-nav{position:sticky;z-index:8;
  top:calc(var(--global-nav-height,0px) + 6px);max-width:980px;margin:16px auto 0;padding:0 20px}
.issue-local-nav{contain:inline-size}.issue-local-nav ul{display:flex;max-width:100%;gap:4px;overflow-x:auto;margin:0;padding:5px;border:1px solid
  var(--border);border-radius:12px;background:rgba(252,252,251,.97);box-shadow:0 4px 16px
  rgba(11,11,11,.06);list-style:none}.issue-local-nav a{display:grid;min-height:38px;place-items:center;
  padding:5px 9px;border-radius:8px;color:var(--ink2);font-size:.76rem;font-weight:720;
  text-align:center;text-decoration:none;white-space:nowrap}.issue-local-nav a:hover{background:var(--page);
  color:var(--ink)}.issues-page main{padding-top:1px}.issues-page main>section{scroll-margin-top:
  calc(var(--global-nav-height,0px) + 72px)}.section-kicker{margin-bottom:5px;color:#6a421b;
  font-size:.68rem;font-weight:820;letter-spacing:.09em;text-transform:uppercase}
.issues-page section>h2{font:700 clamp(1.4rem,3vw,1.8rem)/1.15 Georgia,serif}
.issues-page section>p{max-width:53rem}.issue-figure{margin:18px 0 0}.chart-scroll{position:relative;
  overflow-x:auto;overscroll-behavior-inline:contain;padding:10px 6px 6px;scrollbar-gutter:stable}
.chart-scroll>.chart{min-width:700px}.chart-scroll[tabindex]{outline-offset:4px}.chart-loading{display:grid;
  min-height:340px;place-items:center;padding:24px;color:var(--muted);text-align:center}
.chart-summary{max-width:53rem;margin:8px 0 0;color:var(--ink2);font-size:.82rem}
.scroll-cue{margin:7px 0 0;color:#6c6256;font-size:.72rem;font-weight:680}.exact-values{
  margin-top:14px;border:1px solid var(--border);border-radius:10px;background:var(--surface)}
.exact-values summary{padding:11px 13px;cursor:pointer;font-weight:720}.table-scroll{width:100%;max-width:100%;contain:inline-size;overflow-x:auto;
  border-top:1px solid var(--grid)}.evidence-table{width:100%;border-collapse:collapse;font-size:.78rem}
.evidence-table caption{padding:9px 11px;color:var(--muted);font-size:.7rem;text-align:left}
.evidence-table th,.evidence-table td{padding:8px 10px;border-top:1px solid var(--grid);
  text-align:left;vertical-align:top;font-variant-numeric:tabular-nums}.evidence-table thead th{
  color:var(--muted);font-size:.66rem;letter-spacing:.04em;text-transform:uppercase}
.evidence-table tbody tr:first-child th,.evidence-table tbody tr:first-child td{border-top:0}
.status-ok{color:#315c45}.status-caution{color:#775411}.status-unresolved{color:#704d0c;
  font-weight:720}.download-links{display:flex;flex-wrap:wrap;gap:8px;margin-top:11px}
.download-links a{display:inline-flex;align-items:center;min-height:40px;padding:7px 11px;
  border:1px solid var(--border);border-radius:8px;background:var(--surface);color:var(--ink2);
  font-size:.8rem;font-weight:700;text-decoration:none}.excerpt-grid{display:grid;grid-template-columns:
  repeat(3,minmax(0,1fr));gap:12px;margin-top:18px}.issue-excerpt{display:flex;flex-direction:column;
  margin:0;padding:15px 16px;border:1px solid var(--border);border-left:4px solid #8c877e;
  border-radius:10px;background:var(--surface)}.excerpt-label{color:#6a421b;font-size:.67rem;
  font-weight:820;letter-spacing:.065em;text-transform:uppercase}.issue-excerpt p{margin:7px 0 12px;
  color:var(--ink);font-size:.9rem}.issue-excerpt cite{margin-top:auto;color:var(--muted);
  font-size:.76rem;font-style:normal}.issue-excerpt cite a{color:var(--ink2)}.receipt{display:block;
  margin-top:4px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.65rem}
.owner-scale{display:flex;justify-content:space-between;margin:14px 0 4px;padding-left:230px;
  color:var(--muted);font-size:.67rem}.owner-list{display:grid;gap:8px;margin:0;padding:0;list-style:none}
.owner-row{display:grid;grid-template-columns:210px minmax(150px,1fr) 112px;gap:12px;
  align-items:center;padding:9px 11px;border:1px solid var(--border);border-radius:10px;
  background:var(--surface)}.owner-person{display:grid;grid-template-columns:36px 1fr;gap:9px;
  align-items:center;color:var(--ink);font-weight:720;text-decoration:none}.owner-person img{width:36px;
  height:36px;border:1px solid var(--border);border-radius:50%;object-fit:cover}.owner-person small{
  display:block;color:var(--muted);font-size:.68rem;font-weight:500}.owner-track{height:12px;
  overflow:hidden;border:1px solid var(--border);border-radius:99px;background:#e8e6e1}.owner-track span{
  display:block;height:100%;background:#315f78}.owner-value{text-align:right;
  font-variant-numeric:tabular-nums}.owner-value strong,.owner-value small{display:block}.owner-value small{
  color:var(--muted);font-size:.66rem}.ai-note,.method-caveat{max-width:none;padding:13px 15px;
  border:1px solid #c9def3;border-radius:11px;background:#eef6ff;color:#254e70}
.method-caveat{border-color:#c59a43;background:#fff7df;color:#4e3b13}.fine-definitions{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:9px;margin:15px 0}
.fine-definition{padding:12px 13px;border:1px solid var(--border);border-left:4px solid
  var(--topic-color,#557b9c);border-radius:9px;background:var(--surface)}.fine-definition span{
  color:var(--muted);font-size:.66rem;font-weight:760;text-transform:uppercase}.fine-definition h3{
  margin-top:2px;font-size:.9rem}.fine-definition p{margin:5px 0 0;color:var(--ink2);font-size:.76rem}
.fine-controls{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:end;
  margin-top:16px}.fine-selects{display:flex;flex-wrap:wrap;gap:8px}.fine-selects label{display:grid;
  gap:3px;color:var(--muted);font-size:.68rem;font-weight:720}.fine-selects select{min-height:42px;
  padding:7px 28px 7px 9px;border:1px solid var(--border);border-radius:8px;background:var(--surface);
  color:var(--ink);font:inherit}.topic-picker{margin:12px 0;border:1px solid var(--border);
  border-radius:10px;background:var(--surface)}.topic-picker summary{min-height:42px;padding:10px 12px;
  cursor:pointer;font-weight:720}.topic-picker summary span{margin-left:7px;color:var(--muted);
  font-size:.75rem;font-weight:540}.topic-picker-actions{display:flex;gap:8px;padding:0 12px 10px}
.topic-picker-actions button{min-height:42px;padding:7px 11px;border:1px solid var(--border);
  border-radius:8px;background:var(--surface);color:var(--ink);font:inherit;font-weight:680;cursor:pointer}
.topic-controls{display:flex;flex-wrap:wrap;gap:8px;padding:0 12px 13px;border:0}.topic-controls legend{
  position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}.topic-controls label{
  display:flex;align-items:center;min-height:42px;padding:7px 10px;border:1px solid var(--border);
  border-left:4px solid var(--topic-color,#557b9c);border-radius:9px;background:var(--page);
  font-size:.78rem;cursor:pointer}.topic-controls input{width:18px;height:18px;margin-right:7px;
  accent-color:#275d8c}.topic-controls label:has(input:focus-visible){outline:3px solid var(--focus);
  outline-offset:3px}.fine-status{min-height:1.5em;margin:8px 0;color:var(--ink2);font-size:.78rem}
.fine-figure.is-empty .chart-scroll,.fine-figure.is-empty .scroll-cue{display:none}.fine-figure .chart-loading{
  min-height:410px}.provenance-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
  gap:10px;margin-top:15px}.provenance-card{padding:13px 14px;border:1px solid var(--border);
  border-radius:10px;background:var(--surface)}.provenance-card h3{font-size:.92rem}.provenance-card p{
  margin:5px 0 0;color:var(--ink2);font-size:.78rem}.provenance-card code{overflow-wrap:anywhere;
  word-break:break-word}.issue-adjacent{display:grid;
  grid-template-columns:1fr 1fr;gap:10px;margin-top:22px}.issue-adjacent a{display:flex;
  min-height:70px;flex-direction:column;justify-content:center;padding:12px 14px;border:1px solid
  var(--border);border-radius:10px;background:var(--surface);text-decoration:none}.issue-adjacent .next{
  text-align:right}.issue-adjacent span{color:var(--muted);font-size:.7rem}.issue-adjacent strong{
  margin-top:3px;color:var(--ink)}.directory-main{padding-top:12px}.issue-directory{display:grid;
  grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px;margin:10px 0 0;padding:0;
  list-style:none}.directory-card{height:100%;padding:16px;border:1px solid var(--border);
  border-radius:12px;background:var(--surface)}.directory-card:focus-within{border-color:var(--focus);
  box-shadow:0 0 0 1px var(--focus)}.directory-card>a{color:var(--ink);text-decoration:none}
.directory-card>a::after{content:" →"}.directory-card h2{font:700 1.12rem/1.2 Georgia,serif}
.directory-card dl{display:grid;gap:9px;margin-top:13px}.directory-card dt{color:var(--muted);
  font-size:.65rem;font-weight:780;letter-spacing:.04em;text-transform:uppercase}.directory-card dd{
  margin-top:2px;color:var(--ink2);font-size:.78rem}.directory-card .source-badge{margin:0 0 10px}
@media(max-width:760px){.issue-facts{grid-template-columns:1fr}.issue-local-nav{margin-top:11px;
  padding:0 8px}.issue-local-nav ul{padding:3px}.issue-local-nav a{min-height:40px;font-size:.7rem}
  .issues-page main>section{scroll-margin-top:calc(var(--global-nav-height,0px) + 112px)}
  .excerpt-grid{grid-template-columns:1fr}.owner-scale{display:none}.owner-row{grid-template-columns:1fr;
  gap:8px}.owner-track{height:10px}.owner-value{text-align:left}.owner-person{grid-template-columns:42px 1fr}
  .owner-person img{width:42px;height:42px}.fine-controls{grid-template-columns:1fr}.fine-selects{
  display:grid;grid-template-columns:1fr 1fr}.fine-selects select,.topic-picker-actions button,
  .topic-controls label{min-height:44px}.provenance-grid{grid-template-columns:1fr}.chart-scroll{
  margin-inline:-2px}.issue-adjacent{grid-template-columns:1fr}.issue-adjacent .next{text-align:left}}
@media(max-width:420px){.issues-hero{padding-top:25px}.issues-hero h1{font-size:1.85rem}
  .fine-selects{grid-template-columns:1fr}.topic-picker-actions{display:grid;grid-template-columns:1fr 1fr}
  .download-links{display:grid}.download-links a{width:100%}.directory-main{padding-inline:14px}
  .issue-directory{grid-template-columns:1fr}.evidence-table{font-size:.74rem}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;
  transition-duration:.01ms!important}}
"""


ISSUES_JS = r"""
(() => {
  "use strict";
  const page = document.querySelector("[data-issue-page]");
  if (!page) return;
  const broadTarget = document.getElementById("issue-trend-chart");
  const fineTarget = document.getElementById("fine-topic-chart");
  const removeLoadingPlaceholder = target => {
    target?.querySelector(".chart-loading")?.remove();
  };
  const setFailure = (target, message) => {
    if (!target) return;
    target.replaceChildren();
    target.removeAttribute("role");
    const note = document.createElement("p");
    note.className = "chart-loading";
    note.setAttribute("role", "status");
    note.setAttribute("aria-live", "polite");
    note.textContent = message;
    target.append(note);
  };
  const inline = document.getElementById("issue-inline-data");
  const loadPayload = async () => {
    if (inline) return JSON.parse(inline.textContent);
    const url = page.dataset.issueView;
    if (!url) throw new Error("Issue data URL is missing.");
    const response = await fetch(url, {credentials: "same-origin"});
    if (!response.ok) throw new Error(`Issue data request failed (${response.status}).`);
    return response.json();
  };
  const resolveMode = (requested, count, periodCount) => requested === "auto"
    ? (count <= 3 && count * periodCount <= 48 ? "bars" : "heatmap") : requested;
  const modeLabel = mode => mode === "heatmap" ? "Heatmap" : "Grouped bars";
  const formatShare = value => `${Number(value).toFixed(3).replace(/0+$/, "").replace(/\.$/, "")}%`;
  const formatCount = value => Number(value).toLocaleString();
  const formatRange = (start, end) => start == null || end == null ? "N/A" : `${start}\u2013${end}`;

  function renderBroad(payload) {
    if (!broadTarget || !payload.trend_figure) return;
    if (!window.Plotly) {
      setFailure(broadTarget, "The interactive trend could not load. The exact-value table remains available below.");
      return;
    }
    removeLoadingPlaceholder(broadTarget);
    Plotly.newPlot(
      broadTarget,
      payload.trend_figure.data,
      payload.trend_figure.layout,
      {displayModeBar: false, responsive: true}
    );
  }

  function setupFineTopics(payload) {
    if (!fineTarget) return;
    const rows = payload.fine_topic_rows || [];
    const controls = [...document.querySelectorAll("#fine-topic-controls input")];
    const bucketControl = document.getElementById("fine-topic-bucket");
    const modeControl = document.getElementById("fine-topic-mode");
    const count = document.getElementById("fine-topic-count");
    const status = document.getElementById("fine-topic-status");
    const summary = document.getElementById("fine-chart-summary");
    const tableBody = document.querySelector("#fine-topic-table tbody");
    const tableCaption = document.querySelector("#fine-topic-table caption");
    const figure = fineTarget.closest("figure");
    let initialized = false;

    controls.forEach(control => { control.disabled = false; });
    if (bucketControl) bucketControl.disabled = false;
    if (modeControl) modeControl.disabled = false;
    const allButton = document.getElementById("fine-topic-all");
    const clearButton = document.getElementById("fine-topic-clear");
    if (allButton) allButton.disabled = false;
    if (clearButton) clearButton.disabled = false;

    const updateTable = selectedRows => {
      if (!tableBody) return;
      const fragment = document.createDocumentFragment();
      selectedRows.forEach(row => {
        const tr = document.createElement("tr");
        const cells = [
          row.topic,
          formatRange(row.requested_start, row.requested_end),
          formatRange(row.observed_start, row.observed_end),
          formatCount(row.numerator),
          formatCount(row.denominator),
          formatShare(row.share_percent),
          row.partial_period ? "Partial observed bucket" : "Complete observed bucket",
          row.support_status === "low_support" ? "Low support" : "Descriptive",
        ];
        cells.forEach((value, index) => {
          const cell = document.createElement(index === 0 ? "th" : "td");
          if (index === 0) cell.scope = "row";
          cell.textContent = value;
          tr.append(cell);
        });
        fragment.append(tr);
      });
      tableBody.replaceChildren(fragment);
    };

    const draw = () => {
      initialized = true;
      const selected = controls.filter(control => control.checked).map(control => control.value);
      const bucket = Number(bucketControl?.value || 20);
      const requested = modeControl?.value || "auto";
      if (count) count.textContent = `${selected.length} ${selected.length === 1 ? "topic" : "topics"} selected`;
      const selectedRows = rows.filter(row => row.bucket_years === bucket && selected.includes(row.topic));
      const periodRows = rows.filter(row => row.bucket_years === bucket && row.topic === selected[0]);
      const periods = periodRows.map(row => row.period_label);
      const resolved = resolveMode(requested, selected.length, periods.length);
      const resolvedText = requested === "auto" ? `Auto → ${modeLabel(resolved)}` : modeLabel(resolved);
      const partialPeriodCount = new Set(
        selectedRows.filter(row => row.partial_period).map(row => row.period_label)
      ).size;
      const lowSupportPeriodCount = new Set(
        selectedRows.filter(row => row.support_status === "low_support").map(row => row.period_label)
      ).size;
      const partialText = `${partialPeriodCount} partial edge ${partialPeriodCount === 1 ? "period" : "periods"}`;
      const supportText = `${lowSupportPeriodCount} low-support ${lowSupportPeriodCount === 1 ? "period" : "periods"}`;
      if (tableCaption) tableCaption.textContent = selected.length
        ? `${selected.length} selected ${selected.length === 1 ? "topic" : "topics"} in ${bucket}-year descriptive buckets.`
        : `No fine topics selected in the ${bucket}-year view.`;
      updateTable(selectedRows);
      if (!selected.length) {
        if (window.Plotly) Plotly.purge(fineTarget);
        fineTarget.hidden = true;
        figure?.classList.add("is-empty");
        if (status) status.textContent = `No topics selected · ${bucket}-year descriptive buckets · ${resolvedText} · ${partialText} · ${supportText} · plot removed.`;
        if (summary) summary.textContent = "No fine topics are selected. Use the topic controls to restore the plot and exact table.";
        return;
      }
      fineTarget.hidden = false;
      figure?.classList.remove("is-empty");
      if (status) status.textContent = `${selected.length} ${selected.length === 1 ? "topic" : "topics"} selected · ${bucket}-year descriptive buckets · ${resolvedText} · ${partialText} · ${supportText}.`;
      if (summary) summary.textContent = `${modeLabel(resolved)} of ${selected.length} selected fine ${selected.length === 1 ? "topic" : "topics"} in ${bucket}-year descriptive buckets. Edge labels show the years actually observed; ${supportText}; this grain has no confidence interval.`;
      if (!window.Plotly) {
        setFailure(fineTarget, "The interactive fine-topic chart could not load. The exact-value table remains available below.");
        return;
      }
      removeLoadingPlaceholder(fineTarget);
      const lookup = new Map(selectedRows.map(row => [`${row.topic}\u0000${row.period_label}`, row]));
      const colors = payload.fine_topic_colors || {};
      const traces = resolved === "heatmap" ? [{
        type: "heatmap",
        x: periods,
        y: selected,
        z: selected.map(topic => periods.map(period => lookup.get(`${topic}\u0000${period}`)?.share_percent ?? null)),
        customdata: selected.map(topic => periods.map(period => {
          const row = lookup.get(`${topic}\u0000${period}`);
          return row ? [row.numerator, row.denominator, row.observed_start, row.observed_end, row.partial_period ? "partial" : "complete", row.support_status === "low_support" ? "low support" : "descriptive"] : [0, 0, "", "", "", ""];
        })),
        colorscale: [[0,"#f6f1e9"],[.35,"#b9cfde"],[1,"#275d8c"]],
        colorbar: {title: "share of eligible<br>paragraphs (%)"},
        hovertemplate: "%{y}<br>%{x}<br>%{z:.3f}% of eligible paragraphs<br>%{customdata[0]}/%{customdata[1]} paragraphs · %{customdata[4]} · %{customdata[5]}<extra></extra>",
      }] : selected.map(topic => ({
        type: "bar",
        name: topic,
        x: periods,
        y: periods.map(period => lookup.get(`${topic}\u0000${period}`)?.share_percent ?? null),
        customdata: periods.map(period => {
          const row = lookup.get(`${topic}\u0000${period}`);
          return row ? [row.numerator, row.denominator, row.partial_period ? "partial" : "complete", row.support_status === "low_support" ? "low support" : "descriptive"] : [0, 0, "", ""];
        }),
        marker: {color: colors[topic] || "#557b9c"},
        hovertemplate: "%{x}<br>%{y:.3f}% of eligible paragraphs<br>%{customdata[0]}/%{customdata[1]} paragraphs · %{customdata[2]} · %{customdata[3]}<extra>" + topic + "</extra>",
      }));
      Plotly.react(fineTarget, traces, {
        template: "simple_white",
        paper_bgcolor: "#fcfcfb",
        plot_bgcolor: "#fcfcfb",
        font: {family: "system-ui, -apple-system, 'Segoe UI', sans-serif", color: "#43505a"},
        barmode: "group",
        height: Math.max(480, resolved === "heatmap" ? 190 + selected.length * 34 : 500),
        margin: {l: resolved === "heatmap" ? 230 : 62, r: 24, t: 30, b: 118},
        xaxis: {type: "category", tickangle: -45, gridcolor: "#e3e2de"},
        yaxis: resolved === "heatmap"
          ? {automargin: true}
          : {title: "share of eligible paragraphs (%)", rangemode: "tozero", gridcolor: "#e3e2de"},
        legend: {orientation: "h", y: 1.1},
        showlegend: resolved !== "heatmap",
      }, {displayModeBar: false, responsive: true});
    };

    controls.forEach(control => control.addEventListener("change", draw));
    bucketControl?.addEventListener("change", draw);
    modeControl?.addEventListener("change", draw);
    allButton?.addEventListener("click", () => {
      controls.forEach(control => { control.checked = true; });
      draw();
    });
    clearButton?.addEventListener("click", () => {
      controls.forEach(control => { control.checked = false; });
      draw();
    });
    const initialize = () => { if (!initialized) draw(); };
    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(entries => {
        if (!entries.some(entry => entry.isIntersecting)) return;
        observer.disconnect();
        initialize();
      }, {rootMargin: "220px"});
      observer.observe(fineTarget);
    } else initialize();
  }

  loadPayload().then(payload => {
    renderBroad(payload);
    setupFineTopics(payload);
  }).catch(error => {
    console.error(error);
    setFailure(broadTarget, "The interactive trend data could not load. The exact-value table remains available below.");
    setFailure(fineTarget, "The interactive fine-topic data could not load. The exact-value table remains available below.");
  });
})();
"""
