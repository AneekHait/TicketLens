"""
Shared utilities for LLM output parsing.
Used by kba_generator.py and sop_generator.py.
"""

import re


def parse_llm_sections(text, expected_headers):
    """Parse LLM output into named sections based on header patterns.

    Looks for lines matching any of the expected headers (e.g. "TITLE:",
    "SYMPTOMS:") and splits the text into a dict keyed by header name
    (uppercased).

    Tolerates the markdown LLMs habitually add around headers — ``**TITLE:**``,
    ``### TITLE:``, ``- TITLE:``, ``1. TITLE:`` — and a trailing ``**``. Anchoring
    strictly at the start of the line meant any of those failed to match, so the header
    line was appended to the *previous* section's body instead. The consumers then hid
    the damage behind defaults (``sections.get("TITLE", subcategory)``), which made a
    total parse failure look like a plausible article whose title happened to equal the
    subcategory.

    Args:
        text: Raw LLM output string.
        expected_headers: Iterable of header names to look for
                          (e.g. ["TITLE", "SYMPTOMS", "CAUSE"]).

    Returns:
        Dict mapping uppercase header names to their content strings.
    """
    # Optional leading list/heading markup, then optional bold/italic, then the header.
    header_pattern = re.compile(
        r'^\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s*)?'      # "- ", "1. ", "### "
        r'(?:\*{1,3}|_{1,3})?\s*'                       # "**", "__"
        r'(' + '|'.join(re.escape(h) for h in expected_headers) + r')'
        r'\s*(?:\*{1,3}|_{1,3})?\s*:\s*'                # "**:" or ":"
        r'(?:\*{1,3}|_{1,3})?\s*'                       # bold opening the content
        r'(.*)',
        re.IGNORECASE,
    )

    sections = {}
    current_key = None
    current_lines = []

    for line in text.split("\n"):
        match = header_pattern.match(line)
        if match:
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = match.group(1).upper()
            first_content = match.group(2).strip()
            current_lines = [first_content] if first_content else []
        else:
            current_lines.append(line)

    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections
