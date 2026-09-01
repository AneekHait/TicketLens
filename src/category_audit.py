"""
LLM category audit — reviews how clusters are filed into macro-categories and
proposes reassignments so near-duplicate / misassigned categories collapse.

Design (deliberately plays to a small local model's strengths):
  * The model is never asked to invent a taxonomy. Each cluster is a *bounded
    classification*: given the cluster's subcategory + keywords + a few sample
    tickets, pick the single best-fitting category from the EXISTING set
    (standard categories plus whatever the run already discovered).
  * The current category is shown and "KEEP" is the cheap default, so a group
    only moves when the model is confident another bucket fits better. This
    limits churn from an uncertain model.
  * Nothing is applied here — audit() returns a list of *proposed* changes for
    the GUI to show in a review-and-approve dialog. Merges emerge naturally:
    if every cluster under a tiny category is reassigned elsewhere, that
    category simply disappears from the output.

Mirrors the (llm, cluster_data, settings) generator pattern of
disposition.py / kba_generator.py.
"""

import re

from src.logger import get_logger
from src.clustering import STANDARD_CATEGORIES, _cat_signature, ACCESS_INTENT_RE

logger = get_logger()

ACCESS_CATEGORY = "Access & Authorization"


class CategoryAuditor:
    """Propose per-cluster category reassignments using the local LLM."""

    def __init__(self, llm, cluster_data, settings=None):
        """
        Args:
            llm: a loaded llama-cpp ``Llama`` instance, or ``None`` (no-op audit).
            cluster_data: {cluster_id: {"keywords", "subcategory", "category",
                          "sample_docs", ...}} as built by
                          TicketClusterer.get_cluster_data().
            settings: optional dict — max_sample_tickets, max_keywords, temperature.
        """
        self.llm = llm
        self.cluster_data = cluster_data or {}
        self.settings = settings or {}
        self.max_samples = int(self.settings.get("max_sample_tickets", 3))
        self.max_keywords = int(self.settings.get("max_keywords", 8))
        self.temperature = float(self.settings.get("temperature", 0.1))

    # -- helpers ------------------------------------------------------------
    def _available_categories(self):
        """The category set the model may choose from: the standard taxonomy
        plus any categories the current run actually produced. Deduplicated by
        canonical signature so the option list itself has no near-duplicates."""
        seen = {}
        for cat in STANDARD_CATEGORIES:
            seen.setdefault(_cat_signature(cat), cat)
        for cdata in self.cluster_data.values():
            cat = (cdata.get("category") or "").strip()
            if not cat or cat == "Non-Repetitive":
                continue
            seen.setdefault(_cat_signature(cat), cat)
        return list(seen.values())

    @staticmethod
    def _truncate(text, max_chars=150):
        text = str(text or "").strip()
        return text[:max_chars].rsplit(" ", 1)[0] + "…" if len(text) > max_chars else text

    @staticmethod
    def _has_access_intent(subcategory, keywords):
        """True if the subcategory label / keywords carry access-provisioning
        intent per the same high-precision rule the clustering pipeline trusts
        as Tier-1 (ACCESS_INTENT_RE). The label matters most: it reliably names
        the action ("Provisioning", "Account Creation", "2FA Reset") even when
        the BERTopic keywords are dominated by system/app noise — which is exactly
        what makes the LLM misjudge these clusters."""
        signal = f"{subcategory or ''} {' '.join(str(k) for k in (keywords or []))}"
        return bool(ACCESS_INTENT_RE.search(signal))

    def _resolve_reply(self, reply, available):
        """Map a raw LLM reply to one of ``available`` (order/plural-insensitive),
        or None if it said KEEP / gave nothing usable."""
        reply = (reply or "").strip()
        if not reply:
            return None
        # First non-empty line, minus any echoed "Category:/Answer:/Label:" prefix
        # and surrounding quotes.
        first = reply.splitlines()[0].strip()
        first = re.sub(r"^(category|answer|label|reply)\s*:\s*", "", first, flags=re.IGNORECASE)
        first = first.strip().strip('"').strip("'")
        if first.upper().startswith("KEEP") or first.upper() == "NONE":
            return None
        sig = _cat_signature(first)
        if not sig:
            return None
        for cat in available:
            if _cat_signature(cat) == sig:
                return cat
        return None

    def _build_prompt(self, subcategory, keywords, samples, current, available):
        kw = ", ".join(str(k) for k in keywords[: self.max_keywords] if k)
        sample_lines = "\n".join(
            f"  {i + 1}. {self._truncate(s)}" for i, s in enumerate(samples[: self.max_samples]) if s
        ) or "  (none)"
        cat_lines = "\n".join(f"- {c}" for c in available)
        return f"""You are an IT service management taxonomy expert reviewing how a ticket group is categorized.

Ticket group
- Subcategory: {subcategory or 'N/A'}
- Keywords: {kw or 'N/A'}
- Sample tickets:
{sample_lines}

Currently filed under: "{current}"

Available categories:
{cat_lines}

Pick the single best-fitting category for this group.
- If "{current}" is already a reasonable fit, reply exactly: KEEP
- Otherwise reply with ONLY the exact category name from the list above.

Answer:"""

    # -- main ---------------------------------------------------------------
    def audit(self, callback=None, should_stop=None):
        """Return a list of proposed reassignments.

        Each item: {"cluster_id", "subcategory", "current_category",
                    "proposed_category", "size", "source"} where source is
                    "rule" (deterministic access-intent) or "ai" (LLM). Only
                    clusters whose proposed category differs from the current one
                    are included.

        Two layers, deliberately ordered:
          1. The high-precision access-intent rule owns the Access & Authorization
             boundary in BOTH directions — access-claimed clusters go to / stay in
             A&A (the LLM is not consulted for them), and the LLM is never allowed
             to move a non-access cluster INTO A&A. This is what stops the observed
             failure where provisioning / account / 2FA clusters were moved out.
          2. For everything the access rule is silent on, the LLM decides. Shallow
             keyword template matches are intentionally NOT trusted to override the
             LLM — those are what produced the original misfilings we want fixed.
        """
        proposals = []
        if not self.llm:
            logger.info("[category-audit] No LLM loaded — nothing to audit.")
            return proposals

        available = self._available_categories()
        real = [cid for cid in self.cluster_data if cid != -1]
        total = len(real)
        n_rule = 0
        logger.info(f"[category-audit] Auditing {total} clusters against {len(available)} categories.")

        for i, cid in enumerate(real):
            if should_stop and should_stop():
                logger.info("[category-audit] Stopped by user.")
                break
            cdata = self.cluster_data[cid]
            current = (cdata.get("category") or "").strip()
            subcat = (cdata.get("subcategory") or "").strip()
            if not current or current == "Non-Repetitive":
                continue

            if callback and (i % 3 == 0 or i == total - 1):
                callback(f"Auditing categories ({i + 1}/{total})…", i / total if total else 1.0)

            keywords = cdata.get("keywords") or []
            size = int(cdata.get("size", 0)) if cdata.get("size") else None

            # Layer 1 — deterministic access-intent guard (overrides the LLM).
            if self._has_access_intent(subcat, keywords):
                if _cat_signature(current) != _cat_signature(ACCESS_CATEGORY):
                    # Access intent but filed elsewhere → the rule corrects it.
                    proposals.append({
                        "cluster_id": cid,
                        "subcategory": subcat or f"Cluster {cid}",
                        "current_category": current,
                        "proposed_category": ACCESS_CATEGORY,
                        "size": size,
                        "source": "rule",
                    })
                    n_rule += 1
                # Already in A&A → keep; never let the LLM move it out.
                continue

            # Layer 2 — LLM decides for clusters the access rule is silent on.
            prompt = self._build_prompt(
                subcat, keywords, cdata.get("sample_docs") or [], current, available
            )
            try:
                output = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.settings.get("max_tokens", 16),
                    temperature=self.temperature,
                    stop=["\n\n"],
                )
                reply = output["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"[category-audit] LLM failed for cluster {cid}: {e}")
                continue

            proposed = self._resolve_reply(reply, available)
            if not proposed or _cat_signature(proposed) == _cat_signature(current):
                continue  # KEEP / unresolved / same-category-different-spelling
            # The access rule already ruled out access intent here, so don't let
            # the LLM dump a non-access cluster INTO Access & Authorization.
            if _cat_signature(proposed) == _cat_signature(ACCESS_CATEGORY):
                continue
            proposals.append({
                "cluster_id": cid,
                "subcategory": subcat or f"Cluster {cid}",
                "current_category": current,
                "proposed_category": proposed,
                "size": size,
                "source": "ai",
            })

        if callback:
            callback(f"Audit complete — {len(proposals)} suggestion(s).", 1.0)
        logger.info(
            f"[category-audit] {len(proposals)} reassignment(s) proposed "
            f"({n_rule} by rule, {len(proposals) - n_rule} by AI)."
        )
        return proposals
