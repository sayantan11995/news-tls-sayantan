"""
Agentic Clustering Timeline Generator using LangGraph.

Replaces the extractive summarization and timeline construction steps of the
classical clustering method with a multi-agent LLM pipeline:

1. Topic Classification  - Classify the topic type (disaster, org, person, ...)
2. Summary Generation    - Generate candidate summaries for each cluster
3. Summary Judging       - Judge quality with topic-type-aware criteria
4. Summary Verification  - Verify against source material
5. Timeline Finalization - Construct coherent timeline from verified summaries

Usage:
    python experiments/evaluate.py \
        --dataset $DATASETS/t17 \
        --method agentic_clust \
        --output $RESULTS/t17.agentic_clust.json
"""

import json
import datetime
import collections
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from sklearn.feature_extraction.text import TfidfVectorizer

from news_tls import data, clust


# ---------------------------------------------------------------------------
# Topic-type specific judging criteria
# ---------------------------------------------------------------------------
TOPIC_CRITERIA = {
    "disaster": (
        "For a DISASTER timeline, prioritize:\n"
        "- Impact and severity (casualties, damage extent, affected area)\n"
        "- Progression of the disaster (onset, peak, aftermath)\n"
        "- Emergency response and rescue efforts\n"
        "- Official statements and warnings\n"
        "- Recovery and long-term consequences\n"
        "Penalize summaries that focus on unrelated background information."
    ),
    "conflict": (
        "For a CONFLICT/WAR timeline, prioritize:\n"
        "- Key military operations and battles\n"
        "- Diplomatic negotiations and peace efforts\n"
        "- Casualties and humanitarian impact\n"
        "- Strategic developments and territorial changes\n"
        "- International responses and alliances\n"
        "Penalize summaries that are vague about specific events or outcomes."
    ),
    "organization": (
        "For an ORGANIZATION timeline, prioritize:\n"
        "- Key business decisions and strategic shifts\n"
        "- Leadership changes and appointments\n"
        "- Financial milestones (earnings, acquisitions, IPO)\n"
        "- Product launches and innovations\n"
        "- Regulatory and legal developments\n"
        "Penalize summaries that focus on generic industry trends rather than "
        "specific organizational events."
    ),
    "person": (
        "For a PERSON timeline, prioritize:\n"
        "- Career milestones and achievements\n"
        "- Key decisions and public statements\n"
        "- Controversies and challenges\n"
        "- Awards and recognitions\n"
        "- Personal events that had public significance\n"
        "Penalize summaries that are generic and don't specifically relate "
        "to the person's actions."
    ),
    "political": (
        "For a POLITICAL timeline, prioritize:\n"
        "- Policy decisions and legislative actions\n"
        "- Election events and results\n"
        "- Diplomatic relations and agreements\n"
        "- Public opinion shifts and protests\n"
        "- Government appointments and reorganizations\n"
        "Penalize summaries that editorialize rather than report specific events."
    ),
    "science": (
        "For a SCIENCE/TECHNOLOGY timeline, prioritize:\n"
        "- Research breakthroughs and discoveries\n"
        "- Publication of key findings\n"
        "- Technology releases and updates\n"
        "- Regulatory decisions (approvals, bans)\n"
        "- Impact on industry or public life\n"
        "Penalize summaries that lack specificity about what was discovered "
        "or developed."
    ),
    "legal": (
        "For a LEGAL timeline, prioritize:\n"
        "- Filing of charges and lawsuits\n"
        "- Court proceedings and testimony\n"
        "- Rulings and verdicts\n"
        "- Settlements and penalties\n"
        "- Appeals and procedural developments\n"
        "Penalize summaries that speculate rather than report factual legal "
        "developments."
    ),
    "health": (
        "For a HEALTH/EPIDEMIC timeline, prioritize:\n"
        "- Outbreak developments and spread\n"
        "- Case counts and mortality data\n"
        "- Medical responses and treatments\n"
        "- Public health measures and policies\n"
        "- Vaccine/drug development milestones\n"
        "Penalize summaries that focus on fear rather than factual developments."
    ),
    "general": (
        "For a GENERAL news timeline, prioritize:\n"
        "- Specific, newsworthy events\n"
        "- Key developments and turning points\n"
        "- Official statements and actions\n"
        "- Measurable impacts and outcomes\n"
        "Penalize summaries that are vague or don't capture concrete events."
    ),
}


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------
class TimelineState(TypedDict):
    topic_name: str
    topic_keywords: List[str]
    topic_type: str
    cluster_data: List[Dict[str, Any]]
    candidate_summaries: Dict[str, List[Dict[str, Any]]]
    judged_summaries: Dict[str, Dict[str, Any]]
    verified_summaries: Dict[str, Dict[str, Any]]
    rejected_dates: List[str]
    timeline_items: List[Dict[str, Any]]
    max_dates: int
    max_summary_sents: int
    iteration: int


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------
class AgenticClusteringTimelineGenerator:
    """Timeline generator that uses LangGraph to orchestrate an agentic
    pipeline for cluster summarization and timeline construction.

    The pipeline keeps the classical clustering and cluster-ranking steps
    unchanged, but replaces CentroidOpt summarization and timeline
    construction with:

        classify_topic  ->  generate_summaries  ->  judge_summaries
                                ^                        |
                                |                        v
                           (regenerate)          verify_summaries
                                ^                        |
                                |________________________|
                                                         |
                                                         v
                                                 finalize_timeline
    """

    def __init__(
        self,
        clusterer=None,
        cluster_ranker=None,
        clip_sents=5,
        unique_dates=True,
        llm=None,
        max_iterations=2,
    ):
        self.clusterer = clusterer or clust.TemporalMarkovClusterer()
        self.cluster_ranker = (
            cluster_ranker or clust.ClusterDateMentionCountRanker()
        )
        self.clip_sents = clip_sents
        self.unique_dates = unique_dates
        self.max_iterations = max_iterations

        if llm is None:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        else:
            self.llm = llm

        self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_graph(self):
        workflow = StateGraph(TimelineState)

        workflow.add_node("classify_topic", self._classify_topic)
        workflow.add_node("generate_summaries", self._generate_summaries)
        workflow.add_node("judge_summaries", self._judge_summaries)
        workflow.add_node("verify_summaries", self._verify_summaries)
        workflow.add_node("finalize_timeline", self._finalize_timeline)

        workflow.set_entry_point("classify_topic")
        workflow.add_edge("classify_topic", "generate_summaries")
        workflow.add_edge("generate_summaries", "judge_summaries")
        workflow.add_edge("judge_summaries", "verify_summaries")
        workflow.add_conditional_edges(
            "verify_summaries",
            self._should_regenerate,
            {"regenerate": "generate_summaries", "finalize": "finalize_timeline"},
        )
        workflow.add_edge("finalize_timeline", END)

        self.graph = workflow.compile()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_llm_json(text: str):
        """Robustly extract a JSON object or array from LLM output."""
        text = text.strip()
        # Strip markdown fences
        if "```" in text:
            lines = text.split("\n")
            cleaned = []
            inside = False
            for line in lines:
                if line.strip().startswith("```"):
                    inside = not inside
                    continue
                cleaned.append(line)
            text = "\n".join(cleaned).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to locate a JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        # Try a JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        return None

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------
    def _classify_topic(self, state: TimelineState) -> dict:
        """Determine topic type so that downstream nodes can use the right
        judging criteria (disaster vs. org vs. person, etc.)."""

        prompt = (
            "Given the following news topic information:\n"
            f"Topic name: {state['topic_name']}\n"
            f"Keywords: {', '.join(state['topic_keywords'])}\n\n"
            "Classify this topic into exactly ONE of these categories:\n"
            "  disaster, conflict, organization, person, political,\n"
            "  science, legal, health, general\n\n"
            "Respond with ONLY the category name, nothing else."
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        topic_type = response.content.strip().lower().split()[0]

        if topic_type not in TOPIC_CRITERIA:
            topic_type = "general"

        print(f"  [agentic] topic classified as: {topic_type}")
        return {"topic_type": topic_type}

    # ---- Generate ----
    def _generate_summaries(self, state: TimelineState) -> dict:
        """For every cluster that still needs a summary, ask the LLM to
        produce two candidates: one extractive and one abstractive."""

        candidate_summaries: Dict[str, list] = {}
        verified_dates = set(state.get("verified_summaries", {}).keys())
        rejected_dates = set(state.get("rejected_dates", []))
        iteration = state.get("iteration", 0)

        for cd in state["cluster_data"]:
            date_str = cd["date"]

            # Skip dates already verified
            if date_str in verified_dates:
                continue

            # After the first pass only regenerate for rejected dates
            if iteration > 0 and date_str not in rejected_dates:
                continue

            sentences_text = "\n".join(
                f"  - {s}" for s in cd["sentences"][:20]
            )
            titles_text = "\n".join(
                f"  - {t}" for t in cd["article_titles"][:10]
            )
            max_sents = state["max_summary_sents"]

            prompt = (
                f"You are a news timeline summarization expert working on a "
                f"{state['topic_type']} topic: \"{state['topic_name']}\".\n\n"
                f"Date: {date_str}\n"
                f"Number of articles in this cluster: {cd['num_articles']}\n\n"
                f"Article titles:\n{titles_text}\n\n"
                f"Key sentences from articles:\n{sentences_text}\n\n"
                f"Generate {max_sents} summary sentence(s) for this date "
                f"entry in a timeline.\n"
                f"The summary MUST mention at least one of these keywords: "
                f"{', '.join(state['topic_keywords'])}\n\n"
                f"Provide 2 candidates:\n"
                f"1. EXTRACTIVE – select the {max_sents} best verbatim "
                f"sentence(s) from the list above.\n"
                f"2. ABSTRACTIVE – write {max_sents} original concise "
                f"summary sentence(s).\n\n"
                f"Return ONLY valid JSON:\n"
                '{\n'
                '  "extractive": ["sentence1"],\n'
                '  "abstractive": ["sentence1"]\n'
                '}'
            )

            response = self.llm.invoke([HumanMessage(content=prompt)])
            parsed = self._parse_llm_json(response.content)

            if parsed and "extractive" in parsed and "abstractive" in parsed:
                candidate_summaries[date_str] = [
                    {"type": "extractive", "sentences": parsed["extractive"]},
                    {"type": "abstractive", "sentences": parsed["abstractive"]},
                ]
            else:
                # Fallback: use top source sentences as the only candidate
                fallback = cd["sentences"][: max_sents]
                candidate_summaries[date_str] = [
                    {"type": "extractive", "sentences": fallback},
                ]

        return {"candidate_summaries": candidate_summaries, "rejected_dates": []}

    # ---- Judge ----
    def _judge_summaries(self, state: TimelineState) -> dict:
        """Score each candidate on relevance, informativeness, topic fit,
        and conciseness; pick the best one per date."""

        judged: Dict[str, dict] = {}
        topic_type = state["topic_type"]
        criteria_text = TOPIC_CRITERIA.get(topic_type, TOPIC_CRITERIA["general"])

        for date_str, candidates in state["candidate_summaries"].items():
            # Only one candidate – use it directly
            if len(candidates) <= 1:
                judged[date_str] = {
                    "selected": candidates[0],
                    "score": 5.0,
                    "reasoning": "single candidate",
                }
                continue

            candidates_block = ""
            for idx, c in enumerate(candidates, 1):
                sents = " | ".join(c["sentences"])
                candidates_block += (
                    f"\nCandidate {idx} ({c['type']}): {sents}"
                )

            prompt = (
                f"You are a news timeline quality judge evaluating summaries "
                f"for a {topic_type} timeline about "
                f"\"{state['topic_name']}\".\n\n"
                f"Date: {date_str}\n"
                f"Keywords: {', '.join(state['topic_keywords'])}\n"
                f"\nCandidates:{candidates_block}\n\n"
                f"TOPIC-SPECIFIC CRITERIA:\n{criteria_text}\n\n"
                "GENERAL CRITERIA:\n"
                "- relevance: relates to the topic keywords\n"
                "- informativeness: conveys specific, important information\n"
                "- topic_fit: appropriate for a "
                f"{topic_type} timeline\n"
                "- conciseness: clear and not overly verbose\n\n"
                "Score each candidate 1-10 on each criterion.\n\n"
                "Return ONLY valid JSON:\n"
                '{\n'
                '  "scores": [\n'
                '    {"candidate": 1, "relevance": N, "informativeness": N, '
                '"topic_fit": N, "conciseness": N, "total": N},\n'
                '    {"candidate": 2, "relevance": N, "informativeness": N, '
                '"topic_fit": N, "conciseness": N, "total": N}\n'
                '  ],\n'
                '  "selected_candidate": 1,\n'
                '  "reasoning": "brief explanation"\n'
                '}'
            )

            response = self.llm.invoke([HumanMessage(content=prompt)])
            parsed = self._parse_llm_json(response.content)

            if parsed and "selected_candidate" in parsed:
                idx = parsed["selected_candidate"] - 1
                idx = max(0, min(idx, len(candidates) - 1))
                score = 5.0
                if "scores" in parsed and len(parsed["scores"]) > idx:
                    score = parsed["scores"][idx].get("total", 5.0)
                judged[date_str] = {
                    "selected": candidates[idx],
                    "score": score,
                    "reasoning": parsed.get("reasoning", ""),
                }
            else:
                judged[date_str] = {
                    "selected": candidates[0],
                    "score": 5.0,
                    "reasoning": "JSON parse fallback",
                }

        return {"judged_summaries": judged}

    # ---- Verify ----
    def _verify_summaries(self, state: TimelineState) -> dict:
        """Check each chosen summary against source sentences for factual
        grounding and keyword presence."""

        verified = dict(state.get("verified_summaries", {}))
        rejected: List[str] = []
        cluster_lookup = {cd["date"]: cd for cd in state["cluster_data"]}

        for date_str, judgment in state["judged_summaries"].items():
            if date_str in verified:
                continue

            selected = judgment["selected"]
            summary_text = " | ".join(selected["sentences"])

            source = cluster_lookup.get(date_str, {})
            source_sents = source.get("sentences", [])[:15]
            source_text = "\n".join(f"  - {s}" for s in source_sents)

            prompt = (
                "You are a fact-checker for news timeline summaries.\n\n"
                f"Topic: {state['topic_name']}\n"
                f"Date: {date_str}\n"
                f"Summary to verify: {summary_text}\n\n"
                "Source sentences from news articles:\n"
                f"{source_text}\n\n"
                "Check:\n"
                "1. Is the summary factually supported by the sources?\n"
                "2. Does it contain any unsupported claims (hallucination)?\n"
                "3. Does it contain at least one topic keyword "
                f"({', '.join(state['topic_keywords'])})?\n"
                "4. Is it appropriate for a timeline entry?\n\n"
                "Return ONLY valid JSON:\n"
                '{\n'
                '  "verified": true,\n'
                '  "confidence": 0.9,\n'
                '  "issues": [],\n'
                '  "corrected_summary": null\n'
                '}\n'
                'Set "corrected_summary" to a list of corrected sentence(s) '
                "only if correction is needed; otherwise null."
            )

            response = self.llm.invoke([HumanMessage(content=prompt)])
            parsed = self._parse_llm_json(response.content)

            if parsed:
                is_verified = parsed.get("verified", True)
                confidence = parsed.get("confidence", 0.5)

                if is_verified or confidence >= 0.6:
                    corrected = parsed.get("corrected_summary")
                    if (
                        corrected
                        and isinstance(corrected, list)
                        and len(corrected) > 0
                    ):
                        final_sents = corrected
                    else:
                        final_sents = selected["sentences"]
                    verified[date_str] = {
                        "sentences": final_sents,
                        "confidence": confidence,
                        "type": selected["type"],
                    }
                else:
                    rejected.append(date_str)
            else:
                # Parse failure – accept the summary as-is
                verified[date_str] = {
                    "sentences": selected["sentences"],
                    "confidence": 0.5,
                    "type": selected["type"],
                }

        return {
            "verified_summaries": verified,
            "rejected_dates": rejected,
            "iteration": state["iteration"] + 1,
        }

    # ---- Conditional edge ----
    def _should_regenerate(self, state: TimelineState) -> str:
        n_verified = len(state.get("verified_summaries", {}))
        n_needed = state["max_dates"]
        n_rejected = len(state.get("rejected_dates", []))
        iteration = state.get("iteration", 0)

        if (
            n_rejected > 0
            and n_verified < n_needed
            and iteration < self.max_iterations
        ):
            print(
                f"  [agentic] regenerating: {n_verified}/{n_needed} verified, "
                f"{n_rejected} rejected, iteration {iteration}"
            )
            return "regenerate"
        return "finalize"

    # ---- Finalize ----
    def _finalize_timeline(self, state: TimelineState) -> dict:
        """Let the LLM review the whole timeline for coherence, remove
        redundancies, and produce the final ordered output."""

        verified = state.get("verified_summaries", {})
        max_dates = state["max_dates"]

        if not verified:
            return {"timeline_items": []}

        sorted_entries = sorted(verified.items(), key=lambda x: x[0])

        # Trim to max_dates by score if needed
        if len(sorted_entries) > max_dates:
            judged = state.get("judged_summaries", {})
            scored = [
                (d, v, judged.get(d, {}).get("score", 5.0))
                for d, v in sorted_entries
            ]
            scored.sort(key=lambda x: x[2], reverse=True)
            sorted_entries = [(d, v) for d, v, _ in scored[:max_dates]]
            sorted_entries.sort(key=lambda x: x[0])

        entries_block = "\n".join(
            f"  {d}: {' | '.join(v['sentences'])}"
            for d, v in sorted_entries
        )

        prompt = (
            f"You are a news timeline editor finalizing a {state['topic_type']} "
            f"timeline for \"{state['topic_name']}\".\n\n"
            f"Current timeline entries:\n{entries_block}\n\n"
            "Review for:\n"
            "1. Chronological coherence\n"
            "2. Redundancy – remove near-duplicate entries\n"
            "3. Each entry captures a distinct, important event\n"
            "4. The timeline tells a coherent story of the topic\n\n"
            "Return the finalized timeline as JSON.  Keep original sentences "
            "but you may remove redundant entries.\n\n"
            "Return ONLY valid JSON:\n"
            '{\n'
            '  "timeline": [\n'
            '    {"date": "YYYY-MM-DD", "summary": ["sentence1"]},\n'
            '    ...\n'
            '  ],\n'
            '  "changes_made": "description of changes or none"\n'
            '}'
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        parsed = self._parse_llm_json(response.content)

        if parsed and "timeline" in parsed:
            timeline_items = [
                {"date": e["date"], "summary": e["summary"]}
                for e in parsed["timeline"][:max_dates]
            ]
            if timeline_items:
                changes = parsed.get("changes_made", "none")
                print(f"  [agentic] finalization changes: {changes}")
                return {"timeline_items": timeline_items}

        # Fallback: use entries as-is
        timeline_items = [
            {"date": d, "summary": v["sentences"]}
            for d, v in sorted_entries[:max_dates]
        ]
        return {"timeline_items": timeline_items}

    # ------------------------------------------------------------------
    # Public interface (same as ClusteringTimelineGenerator)
    # ------------------------------------------------------------------
    def predict(
        self,
        collection,
        max_dates=10,
        max_summary_sents=1,
        ref_tl=None,
        input_titles=False,
        output_titles=False,
        output_body_sents=True,
    ):
        """Generate a timeline using the agentic LangGraph pipeline.

        Steps 1-3 (clustering, time assignment, ranking) are identical to
        the classical ``clust`` method.  Steps 4-5 (summarization and
        timeline construction) are replaced by the agentic workflow.
        """

        # --- classical clustering (unchanged) ---
        print("clustering articles...")
        doc_vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english"
        )
        clusters = self.clusterer.cluster(collection, doc_vectorizer)

        print("assigning cluster times...")
        for c in clusters:
            c.time = c.most_mentioned_time()
            if c.time is None:
                c.time = c.earliest_pub_time()

        print("ranking clusters...")
        ranked_clusters = self.cluster_ranker.rank(clusters, collection)

        # --- prepare serialisable cluster data ---
        print("preparing cluster data for agentic pipeline...")
        cluster_data = self._prepare_cluster_data(
            ranked_clusters, collection, max_dates
        )

        # --- run LangGraph workflow ---
        print("running agentic pipeline...")
        initial_state: TimelineState = {
            "topic_name": collection.name,
            "topic_keywords": collection.keywords,
            "topic_type": "",
            "cluster_data": cluster_data,
            "candidate_summaries": {},
            "judged_summaries": {},
            "verified_summaries": {},
            "rejected_dates": [],
            "timeline_items": [],
            "max_dates": max_dates,
            "max_summary_sents": max_summary_sents,
            "iteration": 0,
        }

        result = self.graph.invoke(initial_state)

        return self._build_timeline(result)

    def _prepare_cluster_data(self, ranked_clusters, collection, max_dates):
        """Convert ranked ``Cluster`` objects into plain dicts that can be
        passed through the LangGraph state."""

        cluster_data = []
        seen_dates: set = set()
        n_to_process = min(len(ranked_clusters), max_dates * 3)

        for c in ranked_clusters[:n_to_process]:
            date = c.time.date()
            date_str = date.isoformat()

            if self.unique_dates and date_str in seen_dates:
                continue
            seen_dates.add(date_str)

            # Collect keyword-matching sentences
            sents = []
            for a in c.articles:
                for s in a.sentences[: self.clip_sents]:
                    lower = s.raw.lower()
                    if any(kw in lower for kw in collection.keywords):
                        sents.append(s.raw)

            # If nothing matched keywords, take a few sentences anyway
            if not sents:
                for a in c.articles:
                    for s in a.sentences[:2]:
                        sents.append(s.raw)

            titles = [a.title for a in c.articles]

            cluster_data.append(
                {
                    "date": date_str,
                    "sentences": sents[:30],
                    "article_titles": titles[:15],
                    "num_articles": len(c.articles),
                }
            )

        return cluster_data

    @staticmethod
    def _build_timeline(result) -> data.Timeline:
        """Convert the LangGraph result dict into a ``data.Timeline``."""

        timeline = []
        for item in result.get("timeline_items", []):
            try:
                dt = datetime.datetime.strptime(item["date"], "%Y-%m-%d")
                summary = item["summary"]
                if isinstance(summary, str):
                    summary = [summary]
                timeline.append((dt, summary))
            except (ValueError, KeyError):
                continue

        timeline.sort(key=lambda x: x[0])
        return data.Timeline(timeline)

    def load(self, ignored_topics):
        """No external models to load."""
        pass
