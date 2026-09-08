(() => {
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
})();