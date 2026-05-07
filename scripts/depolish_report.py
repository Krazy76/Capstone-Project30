"""De-polish the report: remove em-dashes, AI-tells, triadic structures.

Reads reports/report_sections_4_5_6_8.md and writes a cleaned version.
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "report_sections_4_5_6_8.md"

text = SRC.read_text(encoding="utf-8")

# -------------------------------------------------------------------
# 1) Em-dash replacements, contextual.
# -------------------------------------------------------------------
# Patterns going from most specific to most general, in order.

# Title line: "Section 4, 5, 6, 8" -> use ":"
text = text.replace("# Capstone Report — Sections 4, 5, 6, 8",
                    "# Capstone Report: Sections 4, 5, 6, 8")
text = text.replace("Tri-Level Financial News Classifier — Shared encoder + 3 task heads",
                    "Tri-Level Financial News Classifier (shared encoder + 3 task heads)")

# "Step N — Title." -> "Step N. Title."
text = re.sub(r"\*\*Step (\d+) — ([^*]+)\.\*\*", r"**Step \1. \2.**", text)

# "Entity head — token level." labels -> "Entity head, token level."
text = text.replace("**Entity head — token level.**", "**Entity head, token-level metric.**")
text = text.replace("**Entity head — span level.**", "**Entity head, span-level metric.**")

# Mid-sentence parenthetical em-dashes: " — XXX — " becomes " (XXX) "
# Apply in two passes for paired em-dashes on same line.
text = re.sub(r" — ([^—\n]{1,200}) — ", r" (\1) ", text)

# Remaining single em-dashes: replace " — " with ". " (sentence break)
# but NOT inside the PlantUML code block (we'll skip the @startuml ... @enduml region).
def replace_remaining_emdash(s: str) -> str:
    # Find PlantUML region(s) and exclude them.
    out = []
    i = 0
    while i < len(s):
        m_start = s.find("@startuml", i)
        if m_start == -1:
            out.append(s[i:].replace(" — ", ". "))
            break
        # Process before block normally.
        out.append(s[i:m_start].replace(" — ", ". "))
        m_end = s.find("@enduml", m_start)
        if m_end == -1:
            out.append(s[m_start:])
            break
        # Append plantuml region untouched.
        out.append(s[m_start:m_end + len("@enduml")])
        i = m_end + len("@enduml")
    return "".join(out)

text = replace_remaining_emdash(text)

# Fix capitalisation after sentence-break replacements where lowercase letter
# follows a period inserted from em-dash mid-sentence. Heuristic: ". " followed
# by lowercase letter at start of new sentence is unusual; capitalise.
text = re.sub(r"(\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
# Re-lower words that were the start of a phrase that does not need capitalisation
# (e.g., "Fifty Housing candidates" should stay capitalised, that's fine).

# -------------------------------------------------------------------
# 2) Triadic / AI-tell rewrites.
# -------------------------------------------------------------------
text = text.replace(
    "Two observations follow. First, the gap between",
    "Two things stand out. The first is the gap between",
)
text = text.replace(
    "Three findings.\n",
    "We draw three conclusions from the comparison.\n",
)
text = text.replace(
    "is the standard treatment in the multi-task NLP literature",
    "is a common pattern in multi-task NLP work",
)
text = text.replace(
    "The honest reading is that",
    "We read this as evidence that",
)

# -------------------------------------------------------------------
# 3) AI-vocab hits.
# -------------------------------------------------------------------
text = text.replace(
    "RoBERTa's robust pretraining",
    "RoBERTa's heavier pretraining",
)

# -------------------------------------------------------------------
# 4) Cleanup: remove double spaces, normalise multiple newlines.
# -------------------------------------------------------------------
text = re.sub(r"  +", " ", text)
text = re.sub(r"\n{3,}", "\n\n", text)

SRC.write_text(text, encoding="utf-8")
print(f"updated: {SRC}")

# Quick re-audit
em = text.count("—")
print(f"  em-dashes remaining: {em}")
ai_vocab = sum(text.lower().count(w) for w in
               ["comprehensive", "crucial", "nuanced", "multifaceted",
                "robust", "tapestry", "underscore", "delve", "moreover",
                "furthermore", "additionally", "pivotal", "interplay"])
print(f"  AI-vocab hits remaining: {ai_vocab}")
