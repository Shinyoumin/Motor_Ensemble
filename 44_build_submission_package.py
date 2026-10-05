"""
44 — Assemble the submission package for the ensemble paper.

Builds submission/ containing:
    PAPER_submission.docx      main text + supplement, WITHOUT the internal review note
    PAPER_submission.md        the markdown it was built from (same content)
    figures/                   the 300 dpi full figures (PNG + vector PDF)
    figures/panels/            each subplot exported separately at 300 dpi
    FIGURE_LIST.md             figure/panel -> file map for the submission system

The internal note ("review items not implemented") is stripped here, so the working manuscript can
keep it while the submission copy never carries it.

Run:  python 44_build_submission_package.py
"""
import os, re, shutil, importlib.util
from docx import Document
from docx.shared import Pt

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
SUB = f"{PROJ}/submission"
SRC300 = f"{PROJ}/figures_submission_300dpi"

_s = importlib.util.spec_from_file_location("b22", f"{PROJ}/scripts/22_build_docx.py")
b22 = importlib.util.module_from_spec(_s); _s.loader.exec_module(b22)

os.makedirs(f"{SUB}/figures/panels", exist_ok=True)

# ---- 1. strip the internal note ----
md = open(f"{PROJ}/PAPER.md", encoding="utf8").read()
cut = md.find("## Internal note")
assert cut > 0, "internal note marker not found"
md_sub = md[:cut].rstrip().rstrip("-").rstrip() + "\n"
open(f"{SUB}/PAPER_submission.md", "w", encoding="utf8").write(md_sub)

# ---- 2. render the submission docx ----
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
b22.render_paper(doc, f"{SUB}/PAPER_submission.md", PROJ)
doc.save(f"{SUB}/PAPER_submission.docx")

# ---- 3. copy the 300 dpi figures ----
for f in sorted(os.listdir(SRC300)):
    if f.endswith((".png", ".pdf")):
        shutil.copy2(f"{SRC300}/{f}", f"{SUB}/figures/{f}")
for f in sorted(os.listdir(f"{SRC300}/panels")):
    shutil.copy2(f"{SRC300}/panels/{f}", f"{SUB}/figures/panels/{f}")

# ---- 4. figure list ----
STEMS = {"Figure 1": "fig1_benchmark", "Figure 2": "fig2_bandwidth", "Figure 3": "fig3_architecture",
         "Figure 4": "fig4_realtime", "Figure S1": "figS1_supplement"}
rows = []
for name, stem in STEMS.items():
    panels = sorted(p for p in os.listdir(f"{SUB}/figures/panels") if p.startswith(stem + "_"))
    rows.append(f"| {name} | `figures/{stem}.png` (300 dpi) · `figures/{stem}.pdf` (vector) | " +
                " · ".join(f"`{p}`" for p in panels) + " |")
open(f"{SUB}/FIGURE_LIST.md", "w", encoding="utf8").write(
    "# Figure files for submission\n\nFull figures are 300 dpi PNG plus a vector PDF of the same "
    "figure; each subplot is additionally exported on its own at 300 dpi under `figures/panels/`, "
    "with the letter matching the panel letter used in the caption.\n\n"
    "| Figure | Full figure | Individual panels |\n|---|---|---|\n" + "\n".join(rows) + "\n")

print("SUBMISSION PACKAGE ->", SUB)
for root, _, files in os.walk(SUB):
    for f in sorted(files):
        fp = os.path.join(root, f)
        print(f"  {os.path.relpath(fp, SUB):46s} {os.path.getsize(fp)/1024:8.0f} KB")
