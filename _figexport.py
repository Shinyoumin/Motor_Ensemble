"""
Shared submission-figure exporter.

Every manuscript figure script calls export_submission(fig, axes, stem) after building its figure.
It writes, into figures_submission_300dpi/:
    <stem>.png / <stem>.pdf   — the full multi-panel figure at 300 dpi (PDF is vector)
    panels/<stem>_a.png ...   — each subplot cropped to its own tight bounding box, 300 dpi
Panel letters follow the order the axes are passed in, i.e. the order they appear in the caption.
"""
import os, string

DPI = 300
OUT = "/mnt/hdd/MEG_Motor/motor_ensemble_claude/figures_submission_300dpi"


def export_submission(fig, axes, stem, pad=(1.10, 1.14)):
    os.makedirs(f"{OUT}/panels", exist_ok=True)
    fig.savefig(f"{OUT}/{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{OUT}/{stem}.pdf", bbox_inches="tight")          # vector, for journals wanting EPS/PDF
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    written = []
    for letter, ax in zip(string.ascii_lowercase, axes):
        bb = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fp = f"{OUT}/panels/{stem}_{letter}.png"
        fig.savefig(fp, dpi=DPI, bbox_inches=bb.expanded(*pad))
        written.append(os.path.basename(fp))
    print(f"  submission export: {stem}.png/.pdf (300 dpi) + panels {written}")
