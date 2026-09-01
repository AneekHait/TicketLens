"""
SOP (Standard Operating Procedure) generator.
Creates structured SOPs from clustering results grouped by macro-category.
"""

from src.logger import get_logger
from src.llm_utils import parse_llm_sections

logger = get_logger()

_SOP_HEADERS = ["TITLE", "PURPOSE", "SCOPE", "PREREQUISITES", "PROCEDURE",
                "ESCALATION", "QUALITY CHECKS"]


class SOPGenerator:
    """Generates Standard Operating Procedures from cluster data using a local LLM.

    Args:
        llm: llama-cpp Llama instance.
        cluster_data: Dict from TicketClusterer.get_cluster_data().
        settings: SOP settings from config.
    """

    def __init__(self, llm, cluster_data, settings=None):
        self.llm = llm
        self.cluster_data = cluster_data
        self.settings = settings or {}
        self.max_clusters = self.settings.get("max_clusters_per_sop", 10)

    def _group_by_category(self):
        """Group cluster data by macro-category.

        Returns dict: category -> {subcategories, keywords, sample_docs}.
        """
        groups = {}
        for cid, cdata in self.cluster_data.items():
            if cid == -1:
                continue
            cat = cdata.get("category", "General IT Support")
            if cat not in groups:
                groups[cat] = {
                    "subcategories": [],
                    "keywords": set(),
                    "sample_docs": [],
                }
            groups[cat]["subcategories"].append(cdata.get("subcategory", ""))
            for kw in cdata.get("keywords", [])[:6]:
                groups[cat]["keywords"].add(str(kw))
            for doc in cdata.get("sample_docs", [])[:2]:
                if len(groups[cat]["sample_docs"]) < 5:
                    groups[cat]["sample_docs"].append(doc)
        return groups

    def generate_sop(self, category, category_data):
        """Generate a single SOP for a macro-category.

        Args:
            category: Category name.
            category_data: Dict with subcategories, keywords, sample_docs.

        Returns:
            Dict with SOP sections.
        """
        subcats = ", ".join(
            s for s in category_data["subcategories"][:self.max_clusters] if s
        )
        kw_str = ", ".join(list(category_data["keywords"])[:12])
        samples = "\n".join(
            f"{i+1}. {str(d)[:180]}"
            for i, d in enumerate(category_data["sample_docs"][:3])
        )

        # See the note in kba_generator: a single user message so the model's own chat
        # template supplies the turn tokens, instead of hardcoded Phi-3 markup that the
        # curated Gemma/Qwen models treat as literal text.
        user_content = f"""You are an IT process manager. Write a Standard Operating Procedure.

Process Area: {category}
Sub-processes: {subcats}
Common issues: {kw_str}

Sample tickets:
{samples}

Write an SOP using this EXACT format, one section per line, no other text:
TITLE: (SOP title)
PURPOSE: (why this SOP exists)
SCOPE: (what it covers)
PREREQUISITES: (what's needed before starting)
PROCEDURE: (numbered step-by-step instructions)
ESCALATION: (when and how to escalate)
QUALITY CHECKS: (how to verify the work)"""

        try:
            output = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": user_content}],
                max_tokens=500,
                temperature=0.3,
            )
            raw = output["choices"][0]["message"]["content"].strip()
            sections = parse_llm_sections(raw, _SOP_HEADERS)

            sop = {
                "category": category,
                "title": sections.get("TITLE", f"SOP: {category}"),
                "purpose": sections.get("PURPOSE", ""),
                "scope": sections.get("SCOPE", ""),
                "prerequisites": sections.get("PREREQUISITES", ""),
                "procedure": sections.get("PROCEDURE", ""),
                "escalation": sections.get("ESCALATION", ""),
                "quality_checks": sections.get("QUALITY CHECKS", ""),
                "subcategories": category_data["subcategories"],
                "keywords": list(category_data["keywords"]),
            }

            # Validate: at least title + procedure
            if not sop["procedure"].strip():
                logger.warning(f"SOP for '{category}': LLM returned no procedure, using fallback")
                sop["procedure"] = (
                    "1. Receive and acknowledge the ticket\n"
                    "2. Review the issue details\n"
                    "3. Apply the relevant resolution steps\n"
                    "4. Verify the fix with the user\n"
                    "5. Close the ticket with documentation"
                )

            return sop

        except Exception as e:
            logger.error(f"SOP generation failed for '{category}': {e}")
            return {
                "category": category,
                "title": f"SOP: {category}",
                "purpose": f"Standard procedure for handling {category} issues.",
                "scope": subcats,
                "prerequisites": "",
                "procedure": (
                    "1. Receive and acknowledge the ticket\n"
                    "2. Review the issue details\n"
                    "3. Apply the relevant resolution steps\n"
                    "4. Verify the fix with the user\n"
                    "5. Close the ticket with documentation"
                ),
                "escalation": "Escalate to the relevant team lead if unresolved.",
                "quality_checks": "Confirm resolution with the user before closing.",
                "subcategories": category_data["subcategories"],
                "keywords": list(category_data["keywords"]),
            }

    def generate_all(self, callback=None):
        """Generate SOPs for all macro-categories.

        Args:
            callback: Optional function(msg, progress) for progress updates.

        Returns:
            List of SOP dicts.
        """
        groups = self._group_by_category()
        categories = sorted(groups.keys())
        total = len(categories)
        sops = []

        for i, cat in enumerate(categories):
            if callback:
                callback(f"Generating SOP {i+1}/{total}: {cat}...", (i / total))
            sop = self.generate_sop(cat, groups[cat])
            if sop:
                sops.append(sop)

        if callback:
            callback(f"Generated {len(sops)} SOPs.", 1.0)
        logger.info(f"Generated {len(sops)} SOPs")
        return sops
