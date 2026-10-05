"""
22 — Compile PAPER.md into PAPER.docx (motor_ensemble_claude).

Reuses the sibling's pure-python-docx markdown renderer (scripts/21_build_docx.py in
motor_comparison_claude) and adds one hook: figure captions of the form
    **Figure N. ... (`figures/xxx.png`)** ...
embed the referenced image just above the caption (the captions are not ![]() embeds).

Run:  python 22_build_docx.py   ->  PAPER.docx
"""
import re, os, importlib.util
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
_s = importlib.util.spec_from_file_location(
    "b21", "/mnt/hdd/MEG_Motor/motor_comparison_claude/scripts/21_build_docx.py")
b21 = importlib.util.module_from_spec(_s); _s.loader.exec_module(b21)

# Submission-facing captions carry no internal file paths (R4 review), so the figure files are
# mapped by figure number here instead of being parsed out of the caption text.
FIGCAP = re.compile(r"^\*\*(Figure\s+S?\d+)\.")
FIGMAP = {
    "Figure 1":  "figures/fig1_benchmark.png",
    "Figure 2":  "figures/fig2_bandwidth.png",
    "Figure 3":  "figures/fig3_architecture.png",
    "Figure 4":  "figures/fig4_realtime.png",
    "Figure S1": "figures/figS1_supplement.png",
}


def render_paper(doc, path, base):
    """Same loop as b21.render_md but embeds the image referenced in a Figure caption."""
    lines = open(path).read().split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        m = FIGCAP.search(ln)
        if m and re.sub(r"\s+", " ", m.group(1)) in FIGMAP:
            fp = os.path.join(base, FIGMAP[re.sub(r"\s+", " ", m.group(1))])
            if os.path.exists(fp):
                doc.add_picture(fp, width=Inches(6.5))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            b21.add_runs(doc.add_paragraph(), ln)
            i += 1
            continue
        # table
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|?$", lines[i + 1].strip()):
            blk = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                blk.append(lines[i]); i += 1
            b21.emit_table(doc, blk); continue
        if ln.startswith("#"):
            lvl = len(ln) - len(ln.lstrip("#")); txt = ln[lvl:].strip()
            h = doc.add_heading("", level=0 if lvl == 1 else min(lvl - 1, 4)); b21.add_runs(h, txt)
            i += 1; continue
        if re.match(r"^---+$", ln.strip()):
            doc.add_page_break(); i += 1; continue
        if re.match(r"^\s*[-*]\s+", ln):
            b21.add_runs(doc.add_paragraph(style="List Bullet"), re.sub(r"^\s*[-*]\s+", "", ln)); i += 1; continue
        if re.match(r"^\s*\d+\.\s+", ln):
            b21.add_runs(doc.add_paragraph(style="List Number"), re.sub(r"^\s*\d+\.\s+", "", ln)); i += 1; continue
        if ln.startswith(">"):
            b21.add_runs(doc.add_paragraph(style="Intense Quote"), ln.lstrip("> ").strip()); i += 1; continue
        if not ln.strip():
            i += 1; continue
        buf = [ln]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||!\[|\s*[-*]\s|\s*\d+\.\s|>|---+$)", lines[i]):
            buf.append(lines[i].rstrip()); i += 1
        b21.add_runs(doc.add_paragraph(), " ".join(buf))


if __name__ == "__main__":
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
    render_paper(doc, f"{PROJ}/PAPER.md", PROJ)
    out = f"{PROJ}/PAPER.docx"
    doc.save(out)
    # sanity: count embedded images
    n_img = sum(1 for s in doc.inline_shapes)
    print(f"SAVED {out}  ({len(doc.paragraphs)} paras, {n_img} images)")
