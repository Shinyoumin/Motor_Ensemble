"""
49 — Build the revision deliverables as Word documents.

  revised/ADDITIONAL_EXPERIMENTS.docx   every analysis added or re-run for the revision
  revised/RESPONSE_TO_REVIEWERS.docx    the point-by-point response

Both are rendered with the same markdown renderer used for the manuscript.
"""
import importlib.util
from docx import Document
from docx.shared import Pt

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
_s = importlib.util.spec_from_file_location(
    "b21", "/mnt/hdd/MEG_Motor/motor_comparison_claude/scripts/21_build_docx.py")
b21 = importlib.util.module_from_spec(_s); _s.loader.exec_module(b21)

for stem in ("ADDITIONAL_EXPERIMENTS", "RESPONSE_TO_REVIEWERS"):
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
    b21.render_md(doc, f"{PROJ}/revised/{stem}.md", PROJ)
    out = f"{PROJ}/revised/{stem}.docx"
    doc.save(out)
    print("SAVED", out)
