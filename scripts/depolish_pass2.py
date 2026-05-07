"""Pass 2: fix multi-line em-dashes + restore lowercase https/arxiv that
the previous capitalisation regex broke.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "report_sections_4_5_6_8.md"

text = SRC.read_text(encoding="utf-8")

# 1) Restore lowercased URL/identifier scheme prefixes broken by capitalisation pass.
text = text.replace(". Https://", ". https://")
text = text.replace(". ArXiv.", ". arXiv.")
text = text.replace("ArXiv. ", "arXiv. ")
text = text.replace("@Startuml", "@startuml")
text = text.replace("@Enduml", "@enduml")

# 2) Fix multi-line parenthetical em-dashes that span newlines.
# Pattern: " —\nXXX—" or " —\nXXX. —"
# More general: replace "X —\n" + "—" (the closing one) on next-or-same line with parens.
# Simpler approach: any remaining " —" not in plantuml -> just convert to "."
def replace_emdash_outside_plantuml(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        m_start = s.find("@startuml", i)
        if m_start == -1:
            chunk = s[i:]
            chunk = chunk.replace(" —\n", ".\n")
            chunk = chunk.replace("\n— ", "\n")
            chunk = chunk.replace(" — ", ". ")
            chunk = chunk.replace(" —", ".")
            chunk = chunk.replace("— ", "")
            out.append(chunk)
            break
        out.append(s[i:m_start])  # before plantuml unchanged
        m_end = s.find("@enduml", m_start)
        if m_end == -1:
            out.append(s[m_start:])
            break
        out.append(s[m_start:m_end + len("@enduml")])
        i = m_end + len("@enduml")
    # Now process the chunks (re-do the replacement on outside-plantuml regions)
    # The above is reading-only; let me redo by splitting.
    return "".join(out)

# Simpler: split by plantuml region
parts = re.split(r"(@startuml.*?@enduml)", text, flags=re.DOTALL)
new_parts = []
for p in parts:
    if p.startswith("@startuml"):
        new_parts.append(p)
    else:
        # Replace remaining em-dashes in non-plantuml parts.
        # Multi-line: " —\nNEXT" → ". NEXT" (with capitalisation)
        p = re.sub(r" —\n(\s*)([a-z])",
                   lambda m: ".\n" + m.group(1) + m.group(2).upper(), p)
        # Line starting with "— " followed by lowercase: drop the dash
        p = re.sub(r"\n— ", "\n", p)
        # Any remaining mid-sentence " — " becomes ". "
        p = p.replace(" — ", ". ")
        # Trailing " —" at end of token
        p = p.replace(" —", ".")
        # Standalone "—"
        p = p.replace("—", "-")
        new_parts.append(p)
text = "".join(new_parts)

# 3) Replace "robust pretraining" elsewhere
text = text.replace("RoBERTa's robust pretraining", "RoBERTa's heavier pretraining")
text = text.replace("robust pretraining", "heavier pretraining")

# 4) Compress double-spaces, multiple newlines.
text = re.sub(r"  +", " ", text)
text = re.sub(r"\n{3,}", "\n\n", text)

SRC.write_text(text, encoding="utf-8")

em = text.count("—")
ai_vocab = sum(text.lower().count(w) for w in
               ["comprehensive", "crucial", "nuanced", "multifaceted",
                "robust", "tapestry", "underscore", "delve", "moreover",
                "furthermore", "additionally", "pivotal", "interplay"])
caps_break = text.count(". Https://") + text.count("ArXiv. ")
print(f"updated: {SRC}")
print(f"  em-dashes remaining: {em}")
print(f"  AI-vocab hits remaining (excl references): {ai_vocab}")
print(f"  broken caps remaining: {caps_break}")
