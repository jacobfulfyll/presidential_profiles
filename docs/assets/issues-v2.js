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
