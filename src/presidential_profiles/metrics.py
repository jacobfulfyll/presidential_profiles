"""Central definitions and illustrated-lesson copy for public measures."""
from __future__ import annotations

from html import escape


def _metric(label: str, question: str, definition: str, example: str,
            steps: list[str], formula: str, unit: str, limitations: str,
            source: str, status: str) -> dict:
    return {"label": label, "question": question, "definition": definition,
            "example": example, "steps": steps, "formula": formula, "unit": unit,
            "limitations": limitations, "source": source, "status": status}


METRICS = {
    "effective_topics": _metric("Effective topics", "How many topics received meaningful attention?",
        "A diversity score translated back into an intuitive number of equally attended topics.",
        "Four equal topics score 4; one dominant topic scores close to 1.",
        ["Split words across paragraph topics", "Find each topic's speech share",
         "Calculate Shannon entropy", "Exponentiate the result"],
        "exp(−Σ pᵢ ln pᵢ)", "effective topics",
        "Depends on AI topic labels and paragraph boundaries.",
        "coverage_pressure.parquet", "confirmatory"),
    "depth_words": _metric("Topic depth", "How long did a topic hold the floor?",
        "The word length of a typical topic episode, giving long episodes proportionally more weight.",
        "A 50-word and 150-word episode produce (50²+150²)/(50+150)=125 words.",
        ["Find consecutive paragraphs carrying a topic", "Sum equivalent words in each run",
         "Weight longer runs by their length"],
        "Σ run_words² / Σ run_words", "equivalent words",
        "Multi-label paragraphs divide their words equally across topics.",
        "coverage_pressure.parquet", "confirmatory"),
    "leading_topic_share": _metric("Leading-topic share", "How concentrated was the speech?",
        "The fraction of topic exposure assigned to the speech's largest topic.",
        "If the largest topic receives 300 of 1,000 equivalent words, the share is 30%.",
        ["Find topic exposures", "Choose the largest", "Divide by all exposed words"],
        "max(topic words) / all topic words", "share",
        "A low value can mean broad coverage or noisy labeling.",
        "coverage_pressure.parquet", "secondary"),
    "adjacent_similarity": _metric("Adjacent-paragraph similarity", "Did neighboring paragraphs stay on the same subjects?",
        "Average Jaccard overlap between the topic sets of neighboring paragraphs.",
        "Topics {A,B} beside {B,C} overlap by 1 of 3, or 0.33.",
        ["Compare neighboring topic sets", "Divide overlap by union", "Average pairs"],
        "|A∩B| / |A∪B|", "Jaccard similarity",
        "Paragraph segmentation changes which pairs are neighbors.",
        "coverage_pressure.parquet", "secondary"),
    "rate_10k": _metric("Rate per 10,000 words", "How often does wording appear after allowing for length?",
        "A count divided by all words and scaled to a same-sized 10,000-word container.",
        "Five mentions in 10,000 words equals 5 per 10,000.",
        ["Count included words", "Count all words", "Divide and multiply by 10,000"],
        "count / words × 10,000", "mentions per 10,000 words",
        "Words can carry different meanings in context.", "speech_markers.parquet", "descriptive"),
    "paragraph_share": _metric("Share of paragraphs", "How much of the record carries a label?",
        "The number of labeled paragraphs divided by every paragraph in the selected record.",
        "Twenty labeled paragraphs among 100 total paragraphs is 20%.",
        ["Count paragraphs carrying the label", "Count every eligible paragraph",
         "Divide the labeled count by the total"],
        "labeled paragraphs / eligible paragraphs", "percent of paragraphs",
        "Paragraphs vary greatly in length, and AI labels can disagree.",
        "paragraph_annotations.parquet", "exploratory AI descriptive"),
    "president_percentile": _metric("President percentile", "Where does a president sit among comparable records?",
        "The percentage of eligible presidents whose measured value is at or below this one.",
        "The 80th percentile means the value is at least as high as about 80% of eligible presidents.",
        ["Calculate the same measure for every eligible president",
         "Exclude records below the five-speech precision floor", "Order the values",
         "Report the president's relative position"],
        "rank among eligible presidents / number eligible", "percentile from 0 to 100",
        "A percentile is a relative position, not an absolute amount or a quality ranking.",
        "president profile payload", "descriptive"),
    "era_relative": _metric("Era-relative difference", "What did a president emphasize more than contemporaries?",
        "A president's rate minus the average rate among nearby presidents.",
        "12% versus an 8% era baseline is +4 percentage points.",
        ["Estimate the president", "Estimate the contemporary baseline", "Subtract baseline"],
        "observed − era baseline", "percentage points",
        "The answer depends on the chosen comparison window.", "issues_president.parquet", "descriptive"),
    "confidence_interval": _metric("Confidence interval", "Which estimates remain plausible under repeated samples?",
        "A range produced by repeatedly resampling whole speeches.",
        "A 7% estimate with a 4–10% interval says repeat samples often land in that range.",
        ["Resample speeches", "Recalculate the estimate", "Keep the middle 95%"],
        "bootstrap quantiles 2.5% and 97.5%", "measure's native unit",
        "It is not a 95% probability that this one fixed interval contains truth.",
        "bands.parquet", "frequentist"),
    "reading_level": _metric("Flesch–Kincaid grade level", "How difficult is the sentence structure and vocabulary?",
        "An estimate of the U.S. school grade needed to read a passage, based on sentence length and syllables per word.",
        "A score of 8 suggests eighth-grade sentence and word complexity; it does not mean only eighth graders can understand it.",
        ["Count words and sentences", "Estimate syllables in each word",
         "Calculate average sentence length and syllables per word",
         "Combine them on the Flesch–Kincaid grade scale"],
        "0.39(words/sentences) + 11.8(syllables/words) − 15.59", "U.S. grade level",
        "It estimates surface difficulty, not accuracy, eloquence, intelligence, or the complexity of the ideas. Names, quotations, and formulaic legal language can move the score.",
        "speech_stats.parquet", "descriptive"),
    "primary_medium": _metric(
        "Primary medium assigned in the corpus",
        "What single communication form did the AI taxonomy assign to each speech?",
        "A mutually exclusive primary-form label used to describe this corpus's composition.",
        "A televised spoken address receives one primary label here, not both a spoken and broadcast count.",
        ["Read the speech title and opening context", "Assign one declared medium label",
         "Count the labels within the displayed period"],
        "speeches with label / corpus speeches in period", "share of corpus speeches",
        "It is not a complete inventory of delivery channels. Spoken addresses may also have "
        "been broadcast, and the label is an AI factual classification rather than archival metadata.",
        "llm_annotations/speech_annotations.parquet", "descriptive"),
    "assigned_audience": _metric(
        "Primary audience assigned in the corpus",
        "Who did the speech's framing most directly address?",
        "A mutually exclusive primary-audience label assigned from the speech title and opening context.",
        "An annual message addressed to Congress receives the Congress label even if newspapers later carried it to the public.",
        ["Read the title and opening context", "Assign one declared audience label",
         "Count labels within the displayed period"],
        "speeches with label / corpus speeches in period", "share of corpus speeches",
        "It does not inventory indirect audiences, circulation, reception, or rebroadcasting.",
        "llm_annotations/speech_annotations.parquet", "descriptive"),
    "proposal_values": _metric(
        "Proposal and values stance",
        "Is a paragraph proposing action, invoking values, doing both, or doing neither?",
        "A mutually exclusive projection of the frozen proposal/values paragraph judgment.",
        "A paragraph assigned both proposal and values appears in the mixed category once.",
        ["Join paragraphs to annotations by document and paragraph index",
         "Project both to mixed", "Divide each category count by all era paragraphs"],
        "category paragraphs / all era paragraphs", "percent of paragraphs",
        "The categories do not measure enactment, sincerity, persuasion, or policy quality.",
        "llm_annotations/paragraph_annotations.parquet", "exploratory AI descriptive"),
    "enemy_identity": _metric(
        "Adversarial framing and named adversaries",
        "Who or what is framed as an opponent, and how often does conflict framing appear?",
        "AI labels for enemy naming, zero-sum framing, partisan attack, and adversarial entity mentions.",
        "China and Democrats can both appear in one era's ranked list when distinct paragraphs frame them adversarially.",
        ["Join annotations on keyed paragraphs", "Canonicalize only declared aliases",
         "Count distinct supporting paragraphs", "Report rates and raw labels together"],
        "flagged paragraphs / eligible paragraphs", "percent of paragraphs",
        "Context errors remain possible; a name in the list is not a judgment that it was a legitimate enemy.",
        "combat/combativeness.parquet + llm_annotations/paragraph_entities.parquet",
        "exploratory AI descriptive"),
    "enemy_category_share": _metric(
        "Category of named adversary",
        "What kind of entity is framed as an adversary in each era?",
        "The share of extracted adversarial entity mentions assigned to nation, group, person, institution, or other.",
        "If 60 of 100 adversarial entity mentions are nations, the nation category is 60%.",
        ["Keep entity mentions with adversarial stance", "Assign the frozen entity type",
         "Count mentions by type within an era", "Divide by all adversarial entity mentions in that era"],
        "category adversarial mentions / all adversarial mentions", "percent of adversarial mentions",
        "Entity extraction and type labels can be wrong; the distribution excludes conflict wording without an extracted adversarial entity.",
        "combat/adversary_mix.parquet", "exploratory AI descriptive"),
    "temporal_orientation": _metric(
        "Future and nostalgia vocabulary",
        "How often does presidential wording point toward tomorrow or recover yesterday?",
        "Declared word families for forward-looking and backward-looking temporal appeal.",
        "A speech can contain both families and therefore score highly on both.",
        ["Match declared word families", "Count all eligible words",
         "Scale each family to 10,000 words"],
        "family count / words × 10,000", "mentions per 10,000 words",
        "A match cannot establish whether a promise is feasible or a memory is accurate.",
        "register/trends.parquet", "descriptive"),
    "hope_doom_ratio": _metric(
        "Hope divided by doom",
        "How much hope vocabulary appears for each doom term in one supported time window?",
        "The aggregate NRC hope count divided by the aggregate doom-family count in a centered five-year window.",
        "A value of 80 means eighty NRC hope matches for each doom-family match in that window.",
        ["Sum NRC hope, doom, and words within the window", "Require at least 20,000 words",
         "Suppress a zero doom denominator", "Divide aggregate counts"],
        "Σ NRC hope matches / Σ doom matches", "hope matches per doom match",
        "The two dictionaries differ greatly in breadth. A ratio can rise when hope increases or doom falls, so absolute rates remain in tooltips.",
        "speech_markers.parquet", "descriptive"),
    "uncertainty_envelope": _metric("Uncertainty envelope", "How much uncertainty comes from sampling and AI disagreement?",
        "The combined interval expands sampling variation by measured annotator disagreement.",
        "A sampling-only 4–8% range may become 3–10% after model disagreement is added.",
        ["Bootstrap speeches", "Measure paired-model disagreement", "Combine the two components"],
        "sampling interval ± disagreement half-width", "share",
        "Disagreement measured on a sample is assumed to transfer to the corpus.",
        "bands.parquet", "quality-aware descriptive"),
    "p_value": _metric("No-change surprise rate", "How surprising is this under a no-change model?",
        "The share of no-change repetitions at least as extreme as the observed result.",
        "4.2% means about four of 100 no-change repetitions were this extreme—not a 4.2% chance the finding is random.",
        ["Specify the no-change model", "Repeat the test", "Count equally or more extreme results"],
        "(extreme draws + 1) / (draws + 1)", "percent",
        "It is not the probability that a finding is random, real, or important.",
        "inference_receipts.parquet", "frequentist"),
    "kappa": _metric("Cohen’s κ", "Do two labelers agree beyond easy chance agreement?",
        "Agreement corrected for the labels each model uses most often.",
        "Two models saying 'no' on nearly everything can have high raw agreement but modest κ.",
        ["Build a confusion matrix", "Measure observed agreement",
         "Estimate agreement from each model's label rates", "Correct the observed value"],
        "(observed − expected) / (1 − expected)", "−1 to 1",
        "Rare labels can depress κ even when most rows match.",
        "agreement_v1.parquet", "quality audit"),
    "network_edge": _metric("Network edge", "Why are two nodes connected?",
        "An edge aggregates source records shared by the two nodes.",
        "Twenty paragraphs and eight speeches carrying two topics create one supported edge.",
        ["Find shared paragraphs", "Count distinct speeches", "Calculate Jaccard and lift"],
        "edge weight = shared evidence", "records or similarity",
        "Thresholds hide weak evidence but do not make visible edges causal.",
        "network_atlas.json", "exploratory"),
    "topic_lifecycle": _metric("Topic lifecycle", "When did an issue appear, disappear, persist, or return?",
        "A classification based on substantive years and long internal gaps.",
        "A topic absent for 35 years and later returning is marked revived.",
        ["Build a smoothed attention curve", "Apply the substantive threshold",
         "Find first, last, and long gaps", "Assign born/died/persistent/revived"],
        "thresholded attention timeline", "lifecycle class",
        "Boundary years depend on corpus coverage and the declared threshold.",
        "topic_lifecycles.parquet", "exploratory"),
    "legal_procedural": _metric("Legal and procedural vocabulary", "How often did presidents use formal governing terms?",
        "A rate for public wording such as act, bill, treaty, section, and appropriation.",
        "It measures what speeches say, not how competently government operated.",
        ["Match the declared word list", "Count all speech words", "Report per 10,000"],
        "included terms / words × 10,000", "mentions per 10,000 words",
        "It does not measure competence, productivity, policy depth, or enacted law.",
        "speech_markers.parquet", "descriptive"),
    "hype_doom": _metric("Hype and doom vocabulary", "How often did speeches use extreme positive or negative wording?",
        "Declared word families for superlative praise and catastrophic decline.",
        "greatest is included in hype; difficult is not automatically doom.",
        ["Match included word families", "Keep hype and doom separate", "Scale per 10,000 words"],
        "family count / words × 10,000", "mentions per 10,000 words",
        "Dictionary matches cannot fully resolve sarcasm or quoted speech.",
        "speech_markers.parquet", "descriptive"),
    "similarity": _metric("Cosine similarity",
        "Which presidents align inside one named, finite feature set?",
        "The angle between two vectors built from declared inputs such as 50 topic shares "
        "or eight rhetorical measures.",
        "Two presidents can score highly on AI topics and much lower on rhetoric; neither "
        "score is an overall likeness.",
        ["Choose one declared feature space", "Standardize unlike rhetorical units when needed",
         "Normalize vector lengths", "Take the dot product"],
        "A·B / (||A|| ||B||)", "−1 to 1",
        "A high score is resemblance within the named inputs, not influence, ideology, "
        "agreement, or general similarity.",
        "president_edges.parquet", "exploratory"),
}

CHART_METRICS = {
    "institution_grows": {"effective_topics"},
    "certainty": {"rate_10k"}, "hypedoom": {"hype_doom", "rate_10k"},
    "quadrant": {"hype_doom", "legal_procedural"}, "hopefear": {"rate_10k"},
    "readability": {"reading_level"}, "issues": {"uncertainty_envelope"},
    "llm_topics": {"paragraph_share"}, "keywords": {"rate_10k"},
    "progressive_crises": {"paragraph_share"}, "new_deal": {"paragraph_share"},
    "kinship_radar": {"similarity"},
    "map": {"similarity"}, "heatmap": {"similarity"},
    "families": {"rate_10k"}, "naming": {"rate_10k"},
    "written_republic": {"primary_medium", "legal_procedural"},
    "written_full": {"primary_medium", "legal_procedural"},
    "expansion_story": {"paragraph_share", "confidence_interval"},
    "fear_early": {"rate_10k"}, "fear_full": {"rate_10k"},
    "fear_index_era": {"rate_10k"}, "fear_index_full": {"rate_10k"},
    "civil_rights_crisis": {"paragraph_share"},
    "civil_rights_yearly": {"paragraph_share"},
    "civil_rights_full": {"paragraph_share", "confidence_interval"},
    "procedural_gilded": {"legal_procedural", "hype_doom"},
    "procedural_later": {"legal_procedural", "hype_doom"},
    "procedural_present": {"legal_procedural", "hype_doom"},
    "procedural_eras": {"legal_procedural", "hype_doom"},
    "progressive_1": {"paragraph_share"}, "progressive_2": {"paragraph_share"},
    "progressive_3": {"paragraph_share"}, "progressive_4": {"paragraph_share"},
    "progressive_heatmap": {"paragraph_share"},
    "naming_progressive": {"rate_10k"},
    "broadcast": {"primary_medium", "reading_level"},
    "platform_signals": {"rate_10k", "hype_doom", "legal_procedural"},
    "platform_era": {"rate_10k", "hype_doom", "legal_procedural"},
    "platform_full": {"rate_10k", "hype_doom", "legal_procedural"},
    "weather_map": {"rate_10k", "hype_doom"},
    "summary_communication": {"primary_medium", "assigned_audience"},
    "summary_enemy_categories": {"enemy_category_share"},
    "summary_enemy_presidents": {"enemy_identity", "paragraph_share"},
    "summary_combat": {"enemy_identity", "paragraph_share"},
    "summary_temporal": {"temporal_orientation", "rate_10k"},
    "summary_hope_doom_ratio": {"hope_doom_ratio", "hype_doom", "rate_10k"},
}


def validate_metric_names(names: set[str]) -> None:
    missing = names - set(METRICS)
    if missing:
        raise ValueError(f"substantive metrics missing registry definitions: {sorted(missing)}")


def validate_charts(chart_names: set[str]) -> None:
    missing = chart_names - set(CHART_METRICS)
    if missing:
        raise ValueError(f"substantive charts lack registered metrics: {sorted(missing)}")
    validate_metric_names(set().union(*(CHART_METRICS[name] for name in chart_names)))


def lesson_html(metric_name: str) -> str:
    metric = METRICS[metric_name]
    steps = "".join(f"<li>{escape(step)}</li>" for step in metric["steps"])
    return (
        f'<details class="metric-lesson"><summary>Explain this measure</summary>'
        f'<p><strong>{escape(metric["label"])}</strong> — {escape(metric["definition"])}</p>'
        f'<p class="toy">{escape(metric["example"])}</p><ol>{steps}</ol>'
        f'<details><summary>Formula and limitations</summary>'
        f'<p><code>{escape(metric["formula"])}</code> · {escape(metric["unit"])}</p>'
        f'<p>{escape(metric["limitations"])}</p></details></details>'
    )


def measure_tile_html(metric_name: str, label: str | None = None) -> str:
    """Compact, single-level measure disclosure for Story Page V3 charts."""
    metric = METRICS[metric_name]
    steps = "".join(f"<li>{escape(step)}</li>" for step in metric["steps"])
    visible_label = label or metric["label"]
    return (
        '<details class="inspector-tile measure-tile">'
        f'<summary><span aria-hidden="true">∑</span> Measure · '
        f'{escape(visible_label)}</summary>'
        f'<div class="inspector-panel"><p><strong>Definition.</strong> '
        f'{escape(metric["definition"])}</p>'
        f'<p><strong>Unit.</strong> {escape(metric["unit"])}.</p>'
        f'<p><strong>Calculation.</strong></p><ol>{steps}</ol>'
        f'<p><strong>Example.</strong> {escape(metric["example"])}</p>'
        f'<p><strong>Limitation.</strong> {escape(metric["limitations"])}</p>'
        f'<p><code>{escape(metric["formula"])}</code></p></div></details>'
    )
