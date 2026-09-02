"""
Figure 9 — Country-level hazard vs environment scatterplots.

Perbaikan terhadap versi sebelumnya, menjawab Komentar 24 Reviewer #1:

1. Typo judul panel: "Landslideean vs Ecosystem" -> "Landslide vs Ecosystem".
2. Label negara keluar bingkai. annotate() secara default tidak dipotong pada
   batas sumbu, sedangkan Cambodia dan Philippines berada persis di tepi rentang
   data. Diperbaiki dengan menambah margin sumbu dan mengaktifkan clip_on.
3. Label saling menimpa ("ThaiCambodia", "Vietnam" di atas "Indonesia").
   Diperbaiki dengan adjustText bila tersedia; kalau tidak, dipakai penggeseran
   sederhana berbasis tabrakan kotak teks.
4. cm.get_cmap() dihapus di matplotlib 3.9. Diganti plt.get_cmap().
5. tab10 hanya punya 10 warna untuk 11 negara. Diganti tab20.
6. Ekspor 600 dpi untuk keperluan cetak.

    pip install adjustText     # opsional, hasil label lebih rapi
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

try:
    from adjustText import adjust_text
    HAS_ADJUST = True
except ImportError:
    HAS_ADJUST = False
    print("adjustText tidak terpasang; memakai penggeseran label sederhana.")

# ------------------------------------------------------------------ konfigurasi
CSV = ('/content/drive/MyDrive/02. Database Data/01. Penelitian/06. Hazard/'
       '02. Hazard SEA Tesis Viola/00. Data/ZonalHazardGDPCS.csv')
OUT = 'Figure9_hazard_vs_environment.png'

CROPLAND_THRESHOLD = 36
ECONOMIC_THRESHOLD = 265
ECOSYSTEM_THRESHOLD = 6
HAZARD_THRESHOLD = 0.8          # ambang sumbu-x untuk elips mitigasi

MARGIN = 0.14                   # ruang di tepi agar label tidak menyentuh bingkai
LABEL_SIZE = 9
DPI = 600


# ------------------------------------------------------------------ data
df = pd.read_csv(CSV)

cols = {
    'country':   'COUNTRY',
    'flood':     '_Floodmean',
    'landslide': '_Landsli_1',
    'wildfire':  '_Wildfirem',
    'economic':  '_Econmean',
    'ecosystem': '_Ecosmean',
    'cropland':  '_Cropmean',
}
missing = [c for c in cols.values() if c not in df.columns]
if missing:
    raise SystemExit(f'kolom tidak ada di CSV: {missing}')

d = {k: df[v].values for k, v in cols.items()}
numeric = [k for k in d if k != 'country']
valid = np.logical_and.reduce([np.isfinite(d[k].astype(float)) for k in numeric])
for k in d:
    d[k] = d[k][valid]
print(f'{valid.sum()} dari {len(valid)} baris dipakai')

countries = d['country']
unique_countries = np.unique(countries)
# tab20 dipakai karena tab10 hanya menyediakan 10 warna untuk 11 negara,
# sehingga dua negara akan berwarna sama.
cmap = plt.get_cmap('tab20', max(len(unique_countries), 20))
color_of = {c: cmap(i) for i, c in enumerate(unique_countries)}
colors = [color_of[c] for c in countries]


# ------------------------------------------------------------------ elips
def draw_mitigation_ellipse(ax, x, y, threshold, color='green'):
    """Elips kovarians untuk unit yang melewati ambang bahaya dan lingkungan."""
    sel = (x > HAZARD_THRESHOLD) & (y > threshold)
    if sel.sum() < 2:
        print(f'  {ax.get_title()}: hanya {sel.sum()} titik, elips dilewati')
        return
    fx, fy = x[sel], y[sel]
    cov = np.cov(fx, fy)
    vals, vecs = np.linalg.eig(cov)
    width, height = 2 * np.sqrt(np.abs(vals))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse((fx.mean(), fy.mean()), width, height,
                         angle=angle, color=color, alpha=0.45, zorder=1))


# ------------------------------------------------------------------ label
def place_labels(ax, x, y, names):
    """Tempatkan nama negara tanpa saling menimpa dan tanpa keluar bingkai."""
    if HAS_ADJUST:
        texts = [ax.text(xi, yi, n, fontsize=LABEL_SIZE, alpha=0.85)
                 for xi, yi, n in zip(x, y, names)]
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='0.6', lw=0.5),
                    expand_points=(1.4, 1.4))
        return

    # Cadangan tanpa adjustText: coba beberapa posisi, terima yang pertama
    # tidak menimpa label lain DAN tetap berada di dalam bingkai. clip_on
    # sengaja dibiarkan mati, karena memotong teks (mis. "East Timor" jadi
    # "East Tim") alih-alih memindahkannya.
    fig = ax.figure
    fig.canvas.draw()
    ax_bb = ax.get_window_extent()
    placed = []
    order = np.argsort(-np.asarray(y, dtype=float))       # dari atas ke bawah
    for i in order:
        xr = (x[i] - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        # titik di sisi kanan diberi label ke kiri lebih dulu, dan sebaliknya
        cands = [(-6, 5), (-6, -10), (0, 12), (0, -14), (6, 5), (6, -10)] if xr > 0.6 \
            else [(6, 5), (6, -10), (0, 12), (0, -14), (-6, 5), (-6, -10)]
        for dx, dy in cands:
            t = ax.annotate(names[i], (x[i], y[i]), fontsize=LABEL_SIZE, alpha=0.85,
                            textcoords='offset points', xytext=(dx, dy),
                            ha='left' if dx > 0 else ('right' if dx < 0 else 'center'),
                            clip_on=False)
            fig.canvas.draw()
            bb = t.get_window_extent()
            inside = (bb.x0 >= ax_bb.x0 and bb.x1 <= ax_bb.x1
                      and bb.y0 >= ax_bb.y0 and bb.y1 <= ax_bb.y1)
            if inside and not any(bb.overlaps(p) for p in placed):
                placed.append(bb)
                break
            t.remove()
        else:
            t = ax.annotate(names[i], (x[i], y[i]), fontsize=LABEL_SIZE, alpha=0.6,
                            textcoords='offset points', xytext=(6, 5), clip_on=False)
            placed.append(t.get_window_extent())


# ------------------------------------------------------------------ gambar
PANELS = [
    ('flood',     'cropland',  'Flood vs Cropland',      CROPLAND_THRESHOLD),
    ('flood',     'economic',  'Flood vs Economic',      ECONOMIC_THRESHOLD),
    ('flood',     'ecosystem', 'Flood vs Ecosystem',     ECOSYSTEM_THRESHOLD),
    ('landslide', 'cropland',  'Landslide vs Cropland',  CROPLAND_THRESHOLD),
    ('landslide', 'economic',  'Landslide vs Economic',  ECONOMIC_THRESHOLD),
    # typo lama: "Landslideean vs Ecosystem"
    ('landslide', 'ecosystem', 'Landslide vs Ecosystem', ECOSYSTEM_THRESHOLD),
    ('wildfire',  'cropland',  'Wildfire vs Cropland',   CROPLAND_THRESHOLD),
    ('wildfire',  'economic',  'Wildfire vs Economic',   ECONOMIC_THRESHOLD),
    ('wildfire',  'ecosystem', 'Wildfire vs Ecosystem',  ECOSYSTEM_THRESHOLD),
]

fig, axs = plt.subplots(3, 3, figsize=(18, 12))

for ax, (xk, yk, title, thr) in zip(axs.flat, PANELS):
    x, y = d[xk].astype(float), d[yk].astype(float)
    ax.set_facecolor('#f0f0f0')
    ax.grid(True, color='white', linewidth=0.8, zorder=0)
    ax.scatter(x, y, color=colors, zorder=3, alpha=0.75, s=110,
               edgecolors='white', linewidths=0.6)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xk.capitalize(), fontsize=12)
    ax.set_ylabel(yk.capitalize(), fontsize=12)
    draw_mitigation_ellipse(ax, x, y, thr)
    # Margin lebih lebar memberi ruang bagi label di tepi rentang data,
    # yang sebelumnya membuat "Cambodia" dan "Philippines" keluar bingkai.
    ax.margins(MARGIN)
    place_labels(ax, x, y, countries)

fig.tight_layout()
fig.savefig(OUT, dpi=DPI, bbox_inches='tight')
print(f'{OUT} tersimpan ({DPI} dpi)')
