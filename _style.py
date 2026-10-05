"""
Shared print geometry for the manuscript figures.

The earlier figures were drawn 12–17 inches wide and then scaled to the journal's column width at
typesetting, which shrank a nominal 10 pt label to about 4 pt on the printed page. Figures are now
drawn at the size they will actually be printed, so a point is a point.

JNE prints a full-width figure at roughly 180 mm (7.1 in); we draw at 7.2 in and never rely on the
typesetter scaling anything down.
"""
import matplotlib

FULL_W = 7.2          # inches, full page width
HALF_W = 3.5          # inches, single column
BASE_PT = 11          # body text inside a figure
TICK_PT = 10          # tick labels
TITLE_PT = 11.5       # panel titles
SMALL_PT = 9.5        # the smallest type permitted anywhere in a panel


def apply(dpi_screen=150, dpi_save=300):
    matplotlib.rcParams.update({
        "font.size": BASE_PT,
        "axes.titlesize": TITLE_PT,
        "axes.labelsize": BASE_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "legend.fontsize": TICK_PT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": dpi_screen,
        "savefig.dpi": dpi_save,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })
