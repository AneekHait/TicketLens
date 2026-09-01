"""
KBA (Knowledge Base Article) generator.
Creates structured knowledge articles from clustering results using the local LLM.
"""

from src.logger import get_logger
from src.llm_utils import parse_llm_sections

logger = get_logger()

_KBA_HEADERS = ["TITLE", "SYMPTOMS", "CAUSE", "RESOLUTION", "PREVENTION"]


class KBAGenerator:
    """Generates Knowledge Base Articles from cluster data using a local LLM.

    Args:
        llm: llama-cpp Llama instance.
        cluster_data: Dict from TicketClusterer.get_cluster_data().
        settings: KBA settings from config.
    """

    def __init__(self, llm, cluster_data, settings=None):
        self.llm = llm
        self.cluster_data = cluster_data
        self.settings = settings or {}
        self.max_samples = self.settings.get("max_sample_tickets", 5)
        self.max_keywords = self.settings.get("max_keywords", 8)

    def generate_article(self, cluster_id):
        """Generate a single KBA article for a cluster.

        Args:
            cluster_id: Topic/cluster ID.

        Returns:
            Dict with keys: cluster_id, title, symptoms, cause, resolution,
            prevention, keywords, subcategory, category.
        """
        cdata = self.cluster_data.get(cluster_id)
        if not cdata:
            return None

        keywords = cdata.get("keywords", [])[:self.max_keywords]
        samples = cdata.get("sample_docs", [])[:self.max_samples]
        subcategory = cdata.get("subcategory", "")
        category = cdata.get("category", "")
        resolution_notes = cdata.get("resolution_notes", [])[:3]

        keyword_str = ", ".join(str(k) for k in keywords)
        sample_str = "\n".join(
            f"{i+1}. {str(s)[:200]}" for i, s in enumerate(samples)
        )
        res_str = "\n".join(
            f"- {str(r)[:150]}" for r in resolution_notes if r
        ) if resolution_notes else "N/A"

        # Single user message, letting the GGUF's own chat template supply the role/turn
        # tokens — the model-agnostic approach used in clustering.py. This used to be a
        # raw completion with hardcoded Phi-3 markup (<|system|>/<|end|>/<|assistant|>)
        # and stop=["<|end|>"]: every curated model is Gemma-4 or Qwen2.5, none of which
        # emit those tokens, so they were read as literal text and the stops never fired.
        # No system role — Gemma's template raises on one.
        user_content = f"""You are a senior IT knowledge manager. Write a Knowledge Base Article.

Topic: {subcategory}
Category: {category}
Keywords: {keyword_str}

Sample tickets:
{sample_str}

Resolution patterns:
{res_str}

Write a KBA using this EXACT format, one section per line, no other text:
TITLE: (clear, specific title)
SYMPTOMS: (what the user experiences)
CAUSE: (root cause or common reasons)
RESOLUTION: (step-by-step fix)
PREVENTION: (how to avoid in future)"""

        try:
            output = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": user_content}],
                max_tokens=400,
                temperature=0.3,
            )
            raw = output["choices"][0]["message"]["content"].strip()
            sections = parse_llm_sections(raw, _KBA_HEADERS)

            article = {
                "cluster_id": cluster_id,
                "title": sections.get("TITLE", subcategory or "Untitled"),
                "symptoms": sections.get("SYMPTOMS", ""),
                "cause": sections.get("CAUSE", ""),
                "resolution": sections.get("RESOLUTION", ""),
                "prevention": sections.get("PREVENTION", ""),
                "keywords": keywords,
                "subcategory": subcategory,
                "category": category,
            }

            # Validate: at least title + one body section
            body_sections = [article["symptoms"], article["cause"],
                             article["resolution"]]
            if not any(s.strip() for s in body_sections):
                logger.warning(f"KBA for cluster {cluster_id}: LLM returned no body sections, using fallback")
                article["symptoms"] = f"Users report issues related to: {keyword_str}"
                article["resolution"] = "Refer to the relevant support documentation."

            return article

        except Exception as e:
            logger.error(f"KBA generation failed for cluster {cluster_id}: {e}")
            return {
                "cluster_id": cluster_id,
                "title": subcategory or f"Topic {cluster_id}",
                "symptoms": f"Issues related to: {keyword_str}",
                "cause": "",
                "resolution": "Contact support team for assistance.",
                "prevention": "",
                "keywords": keywords,
                "subcategory": subcategory,
                "category": category,
            }

    def generate_all(self, callback=None):
        """Generate KBA articles for all non-outlier clusters.

        Args:
            callback: Optional function(msg, progress) for progress updates.

        Returns:
            List of article dicts.
        """
        cluster_ids = sorted(
            cid for cid in self.cluster_data if cid != -1
        )
        total = len(cluster_ids)
        articles = []

        for i, cid in enumerate(cluster_ids):
            if callback:
                callback(f"Generating KBA {i+1}/{total}...", (i / total))
            article = self.generate_article(cid)
            if article:
                articles.append(article)

        if callback:
            callback(f"Generated {len(articles)} KBA articles.", 1.0)
        logger.info(f"Generated {len(articles)} KBA articles")
        return articles
