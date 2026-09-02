"""
Multi-Hazard SEA — pipeline lengkap untuk dijalankan di Colab

Titik latih dari Drive, parameter dari katalog GEE, sampling ditarik langsung
ke memori per potongan kecil. Tanpa upload asset, tanpa download raster, tanpa
Export.table.toDrive.

CARA PAKAI
----------
Berkas ini dibagi jadi sel dengan penanda "# %%".

  Colab   : buka berkas ini, lalu salin tiap blok "# %%" ke sel terpisah.
            Atau unggah ke Drive dan jalankan sekaligus:
                %run /content/drive/MyDrive/multihazard.py
  VS Code : penanda "# %%" langsung dikenali sebagai sel interaktif.
  Terminal: python multihazard.py  (menjalankan seluruh alur berurutan)

Sel 6 menyimpan hasil sampling ke Drive, jadi sesi yang putus tidak memaksa
pengulangan dari awal.
"""

# %% [markdown]
# # Multi-Hazard SEA — seluruhnya di Colab
#
# Tanpa upload asset ke GEE, tanpa download raster, tanpa `Export.table.toDrive`.
#
# - **Titik latih** dibaca dari shapefile di Drive
# - **Parameter** diambil dari katalog GEE, dihitung di server
# - **Sampling** ditarik langsung ke memori per potongan kecil, jadi tidak kena
#   batas waktu `getInfo()` dan tidak perlu antre batch job
# - **Validasi** dijalankan lokal dengan scikit-learn
#
# Hanya nilai prediktor di titik yang berpindah dari GEE ke Colab — sekitar
# 7.000 baris, bukan raster.
#
# Jalankan sel berurutan. Sel 5 menyimpan hasil sementara ke Drive, jadi kalau
# sesi Colab putus, sampling tidak perlu diulang dari nol.

# %% [markdown]
# ## Sel 1 — Setup

# %%
# Di Colab, dua blok berikut memasang dependensi dan memasang Drive.
# Di luar Colab, keduanya dilewati dan diasumsikan sudah terpasang:
#   pip install earthengine-api geopandas statsmodels scikit-learn scipy
try:
    from google.colab import drive
    drive.mount('/content/drive')
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "earthengine-api", "geopandas", "statsmodels",
                    "scikit-learn", "scipy"], check=False)
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("bukan Colab — pastikan dependensi sudah terpasang")

import ee
EE_PROJECT = "gisact2026"
try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)
print("Earth Engine siap:", EE_PROJECT)

# %% [markdown]
# ## Sel 2 — Konfigurasi

# %%
import os, glob, json, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

DRIVE_DIR = "/content/drive/MyDrive/02. Hazard SEA Tesis Viola/00. Data/01. Point Hazard"
CACHE_DIR = "/content/drive/MyDrive/GEE_MultiHazard_Revisi"
os.makedirs(CACHE_DIR, exist_ok=True)

DRIVE_FILES = {
    "landslide": {"presence": "Copy of titik_tanahlongsorvioterbaru.zip",
                  "legacy_absence": "Copy of TitikNonLongsor.zip"},
    "wildfire":  {"presence": "Copy of api.zip",
                  "legacy_absence": "Copy of nonapi2.zip"},
    "flood":     {"presence": "Copy of titikbanjirvio.zip",
                  "legacy_absence": "Copy of nonbanjir3.zip"},
}

ABSENCE_MODE  = "legacy"      # "legacy" atau "buffered"
TARGET_SCALE  = 250
DATE_START    = "2014-01-01"  # SESUAIKAN — ini yang bentrok di komentar #4
DATE_END      = "2020-12-31"
CHUNK         = 250           # titik per panggilan getInfo
RANDOM_SEED   = 42
N_FOLDS       = 5
N_REPEATS     = 3
BLOCK_DEG     = 1.0           # ~110 km, untuk spatial block CV
BUFFER_M      = 5000          # dipakai kalau ABSENCE_MODE = "buffered"

HAZARD_BANDS = {
    "landslide": ["rainfall", "slope", "elevation", "soil_texture",
                  "soil_water", "landcover", "ndvi"],
    "wildfire":  ["rainfall", "slope", "elevation", "landcover", "lst",
                  "wind_speed", "ndvi"],
    "flood":     ["rainfall", "slope", "elevation", "landcover", "ndvi",
                  "dist_river", "dist_road"],
}
CATEGORICAL_BANDS = {"landcover", "soil_texture"}
# Jumlah pembanding: apa yang benar-benar ada di berkas Drive.
FILE_COUNTS = {"landslide": (2483, 1100), "wildfire": (1919, 1089),
               "flood": (150, 146)}

# Penyaringan atribut sebelum titik dipakai.
#
# Berkas kebakaran adalah produk MODIS/FIRMS dan memuat kolom "type":
#   0 = presumed vegetation fire   1 = active volcano
#   2 = other static land source   3 = offshore
# (NASA Earthdata, active fire data attributes)
#
# Berkas Anda berisi 1.762 bertipe 0, 65 bertipe 1, dan 92 bertipe 2. Gunung api
# dan suar gas bukan kebakaran vegetasi; membiarkannya membuat model belajar
# bahwa lokasi gunung api rawan kebakaran hutan. Set None untuk mematikan.
POINT_FILTERS = {
    ("wildfire", "presence"): ("type", [0]),
}

# LSIB menamai Myanmar sebagai "Burma". Memakai "Myanmar" menghapus seluruh
# negara itu dari AOI dan membuang ribuan titik lewat filterBounds.
SEA = ["Brunei", "Cambodia", "Timor-Leste", "Indonesia", "Laos", "Malaysia",
       "Burma", "Philippines", "Singapore", "Thailand", "Vietnam"]

print("mode absence :", ABSENCE_MODE)
print("skala        :", TARGET_SCALE, "m")

# %% [markdown]
# ## Sel 3 — Baca titik dari Drive
#
# Shapefile dibaca lokal dengan geopandas. Yang keluar dari sel ini hanya daftar
# koordinat, jadi tidak ada asset yang perlu diunggah.

# %%
import zipfile
import geopandas as gpd

def find_vector(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
    for ext in (".shp", ".gpkg", ".geojson", ".json", ".kml"):
        hits = sorted([n for n in names if n.lower().endswith(ext)
                       and not os.path.basename(n).startswith((".", "__MACOSX"))],
                      key=len)
        if hits:
            return hits[0]
    raise RuntimeError(f"tidak ada file vektor di {os.path.basename(path)}: {names[:12]}")


def read_points(hazard, role):
    path = os.path.join(DRIVE_DIR, DRIVE_FILES[hazard][role])
    gdf = gpd.read_file(f"zip://{path}!{find_vector(path)}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")

    flt = POINT_FILTERS.get((hazard, role))
    if flt:
        col, keep = flt
        if col not in gdf.columns:
            print(f"    PERINGATAN {hazard}/{role}: kolom '{col}' tidak ada, "
                  f"filter dilewati")
        else:
            before = len(gdf)
            vc = pd.to_numeric(gdf[col], errors="coerce").value_counts().sort_index()
            gdf = gdf[pd.to_numeric(gdf[col], errors="coerce").isin(keep)]
            print(f"    filter {hazard}/{role}: '{col}' in {keep} -> "
                  f"{len(gdf)}/{before} dipertahankan")
            print(f"      sebaran: {dict(vc)}")
    pts = []
    for g in gdf.geometry:
        if g is None or g.is_empty:
            continue
        p = g if g.geom_type == "Point" else g.representative_point()
        pts.append((float(p.x), float(p.y)))
    return np.array(pts)


points = {}
for hz in DRIVE_FILES:
    pres = read_points(hz, "presence")
    absn = read_points(hz, "legacy_absence")
    points[hz] = {"presence": pres, "legacy_absence": absn}
    fp, fa = FILE_COUNTS[hz]
    note_p = "" if len(pres) == fp else f"  <- berkas utuh {fp}"
    print(f"  {hz:<10s} presence {len(pres):>5d}{note_p}")
    print(f"  {'':<10s} absence  {len(absn):>5d}"
          f"{'' if len(absn) == fa else f'  <- berkas utuh {fa}'}")

# %% [markdown]
# ## Sel 4 — Stack prediktor
#
# Dibangun sebagai objek GEE yang belum dihitung. Tidak ada `reproject()` —
# memanggilnya di sini memaksa GEE menghitung seluruh piksel AOI dan itulah
# penyebab utama *Computation timed out*. Skala cukup ditetapkan saat sampling.
#
# Koleksi harian juga diganti dengan agregat yang lebih jarang (CHIRPS pentad,
# MOD11A2 8-harian, ERA5-Land bulanan). Rata-rata jangka panjangnya setara, tapi
# jumlah citra yang diproses turun dari ribuan ke ratusan.

# %%
lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
sel = lsib.filter(ee.Filter.inList("country_na", SEA))
found = sel.aggregate_array("country_na").distinct().getInfo()
missing = [c for c in SEA if c not in found]
if missing:
    print("PERINGATAN negara tidak ketemu:", missing)
AOI = sel.geometry()
print(f"luas AOI: {AOI.area(maxError=1000).divide(1e6).getInfo():,.0f} km2")


def build_stack():
    dem   = ee.Image("USGS/SRTMGL1_003").rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    lc    = (ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019")
             .select("discrete_classification").rename("landcover"))
    sw    = (ee.Image("OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01")
             .select("b0").rename("soil_water"))
    stx   = (ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")
             .select("b0").rename("soil_texture"))
    rain  = (ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD")
             .filterDate(DATE_START, DATE_END).select("precipitation")
             .mean().rename("rainfall"))
    ndvi  = (ee.ImageCollection("MODIS/061/MOD13Q1")
             .filterDate(DATE_START, DATE_END).select("NDVI")
             .mean().multiply(0.0001).rename("ndvi"))
    lst   = (ee.ImageCollection("MODIS/061/MOD11A2")
             .filterDate(DATE_START, DATE_END).select("LST_Day_1km")
             .mean().multiply(0.02).subtract(273.15).rename("lst"))
    w     = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
             .filterDate(DATE_START, DATE_END)
             .select(["u_component_of_wind_10m", "v_component_of_wind_10m"]).mean())
    wind  = (w.select("u_component_of_wind_10m").pow(2)
             .add(w.select("v_component_of_wind_10m").pow(2))
             .sqrt().rename("wind_speed"))
    riv   = (ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
             .distance(searchRadius=50000).unmask(50000).rename("dist_river"))
    road  = ee.Image.constant(0).rename("dist_road")   # ganti bila punya groads

    cont = [dem, slope, sw, rain, ndvi, lst, wind, riv, road]
    cat  = [lc, stx]
    return ee.Image.cat([i.resample("bilinear") for i in cont] + cat)

STACK = build_stack()
print("stack siap:", STACK.bandNames().getInfo())

# %% [markdown]
# ## Sel 5 — Pseudo-absence buffered (lewati jika `ABSENCE_MODE = "legacy"`)
#
# Membangkitkan titik non-kejadian acak di luar buffer semua titik kejadian,
# sehingga absence tidak lagi ditentukan oleh prediktor yang sama yang dipakai
# melatih model. Hanya koordinat yang ditarik, tanpa perhitungan raster, jadi
# cepat.

# %%
def buffered_absences(pres_coords, n, buffer_m=BUFFER_M, seed=RANDOM_SEED):
    pres = ee.FeatureCollection(ee.List([list(map(float, c)) for c in pres_coords])
                                .map(lambda c: ee.Feature(ee.Geometry.Point(ee.List(c)))))
    area = AOI.difference(pres.geometry().buffer(buffer_m), maxError=1000)
    fc = ee.FeatureCollection.randomPoints(region=area, points=int(n * 1.2), seed=seed)
    geo = fc.limit(int(n)).geometry().getInfo()["coordinates"]
    return np.array(geo)


if ABSENCE_MODE == "buffered":
    for hz in points:
        n = len(points[hz]["presence"])
        t0 = time.time()
        points[hz]["absence_used"] = buffered_absences(points[hz]["presence"], n)
        print(f"  {hz:<10s} {len(points[hz]['absence_used'])} absence "
              f"({time.time()-t0:.0f} s)")
else:
    for hz in points:
        points[hz]["absence_used"] = points[hz]["legacy_absence"]
    print("  memakai titik non-kejadian lama dari Drive")

# %% [markdown]
# ## Sel 6 — Sampling bertahap, langsung ke memori
#
# Inti dari pendekatan ini. Titik dipotong jadi kelompok kecil, tiap kelompok
# ditarik dengan `getInfo()`. Karena tiap panggilan hanya menghitung stack di
# beberapa ratus titik, tidak ada yang mendekati batas waktu, dan tidak ada batch
# job yang perlu diantre.
#
# Hasil tiap bahaya disimpan ke Drive begitu selesai, jadi sesi yang putus tidak
# memaksa pengulangan dari awal.

# %%
def sample_chunk(coords, idx, bands, retries=3):
    """Tarik nilai prediktor untuk satu potongan titik."""
    payload = [[float(x), float(y), int(i)] for (x, y), i in zip(coords, idx)]
    fc = ee.FeatureCollection(ee.List(payload).map(
        lambda p: ee.Feature(ee.Geometry.Point(ee.List(p).slice(0, 2)),
                             {"idx": ee.List(p).get(2)})))
    sampled = STACK.select(bands).sampleRegions(
        collection=fc, properties=["idx"], scale=TARGET_SCALE,
        geometries=False, tileScale=4)
    for a in range(retries):
        try:
            return [f["properties"] for f in sampled.getInfo()["features"]]
        except Exception as e:
            if a == retries - 1:
                raise
            wait = 5 * (a + 1)
            print(f"      gagal ({type(e).__name__}), ulang dalam {wait}s")
            time.sleep(wait)


def sample_hazard(hazard, mode=ABSENCE_MODE):
    tag = "_filtered" if any(k[0] == hazard for k in POINT_FILTERS) else ""
    cache = os.path.join(
        CACHE_DIR, f"sample_{hazard}_{mode}{tag}_{TARGET_SCALE}m.csv")
    if os.path.exists(cache):
        print(f"  {hazard}: memakai cache {os.path.basename(cache)}")
        return pd.read_csv(cache)

    bands = HAZARD_BANDS[hazard]
    pres, absn = points[hazard]["presence"], points[hazard]["absence_used"]
    coords = np.vstack([pres, absn])
    labels = np.r_[np.ones(len(pres)), np.zeros(len(absn))].astype(int)

    rows, t0 = [], time.time()
    n_chunks = int(np.ceil(len(coords) / CHUNK))
    for c in range(n_chunks):
        lo, hi = c * CHUNK, min((c + 1) * CHUNK, len(coords))
        rows += sample_chunk(coords[lo:hi], range(lo, hi), bands)
        print(f"    {hazard} {c+1}/{n_chunks}  ({len(rows)} baris, "
              f"{time.time()-t0:.0f}s)", end="\r")

    df = pd.DataFrame(rows).set_index("idx").sort_index()
    df["Kelas"] = labels[df.index.values]
    df["lon"] = coords[df.index.values, 0]
    df["lat"] = coords[df.index.values, 1]
    df = df.dropna(subset=bands)

    # fold spasial dihitung lokal dari koordinat
    bi = np.floor(df.lon / BLOCK_DEG).astype(int)
    bj = np.floor(df.lat / BLOCK_DEG).astype(int)
    df["block_id"] = bi * 1000 + bj
    df["spatial_fold"] = np.abs(df.block_id) % N_FOLDS

    df.to_csv(cache, index=False)
    print(f"\n  {hazard}: {len(df)} baris tersimpan "
          f"({time.time()-t0:.0f}s) -> {os.path.basename(cache)}")
    return df


samples = {}
for hz in HAZARD_BANDS:
    samples[hz] = sample_hazard(hz)
    n1 = int((samples[hz].Kelas == 1).sum())
    n0 = int((samples[hz].Kelas == 0).sum())
    print(f"    {n1} presence / {n0} absence")

# %% [markdown]
# ## Sel 7 — Audit jarak pseudo-absence (komentar #8)
#
# Menghitung jarak tiap titik non-kejadian ke kejadian terdekat, lokal dengan
# scipy. Tidak menyentuh GEE, selesai seketika.

# %%
from scipy.spatial import cKDTree

def audit(hazard):
    pres = points[hazard]["presence"]
    absn = points[hazard]["absence_used"]
    lat0 = np.radians(np.mean(pres[:, 1]))
    # proyeksi kasar derajat -> km, cukup untuk statistik ringkasan
    to_km = lambda a: np.c_[a[:, 0] * 111.32 * np.cos(lat0), a[:, 1] * 110.57]
    d, _ = cKDTree(to_km(pres)).query(to_km(absn), k=1)
    return {"hazard": hazard, "n": len(d), "min_km": d.min(),
            "p05_km": np.percentile(d, 5), "median_km": np.median(d),
            "max_km": d.max()}

rows = [audit(h) for h in points]
audit_df = pd.DataFrame(rows).round(2)
print(audit_df.to_string(index=False))
audit_df.to_csv(f"absence_distance_audit_{ABSENCE_MODE}.csv", index=False)

# %% [markdown]
# ## Sel 8 — VIF dan matriks korelasi (komentar #5 dan #9)

# %%
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

def collinearity(df, bands, hazard):
    """VIF plus matriks korelasi Pearson DAN Spearman.

    VIF dan Pearson dipertahankan karena keduanya koheren satu sama lain: VIF
    diturunkan dari R2 regresi linear, dan itu pula yang diminta reviewer.

    Spearman dilaporkan berdampingan karena model yang dipakai berbasis pohon,
    yang invarian terhadap transformasi monoton. Untuk pertanyaan sebenarnya,
    yaitu apakah importance terbagi di antara prediktor redundan, keterkaitan
    monoton lebih relevan daripada linear. Spearman juga tahan terhadap
    kemencengan curah hujan dan terhadap tumpukan nilai pada dist_river yang
    berasal dari unmask(50000).

    PENTING: matriks desain HARUS memuat kolom konstanta. Tanpa itu,
    variance_inflation_factor memakai R2 uncentered, sehingga variabel dengan
    rasio mean terhadap sd yang besar mendapat VIF raksasa meskipun saling
    bebas. Sebagian versi statsmodels menambahkannya sendiri dan sebagian
    tidak, jadi konstanta ditambahkan eksplisit agar hasilnya konsisten.
    """
    num = [b for b in bands if b not in CATEGORICAL_BANDS]
    X = df[num].astype(float)
    dropped = [c for c in X.columns if X[c].std() == 0]
    if dropped:
        print(f"    ({hazard}: {dropped} konstan, dikeluarkan)")
    X = X.loc[:, X.std() > 0]

    Xc = add_constant(X)
    vif = pd.DataFrame({"variable": X.columns,
                        "VIF": [variance_inflation_factor(Xc.values, i)
                                for i in range(1, Xc.shape[1])]}
                       ).sort_values("VIF", ascending=False).reset_index(drop=True)

    pear = X.corr(method="pearson")
    spear = X.corr(method="spearman")

    print(f"\n  {hazard}")
    for _, r in vif.iterrows():
        print(f"    {r.variable:<14s} VIF {r.VIF:8.2f}"
              + ("   <-- di atas ambang 5" if r.VIF > 5 else ""))

    pairs = []
    for i, a in enumerate(pear.columns):
        for b in pear.columns[i + 1:]:
            pairs.append((a, b, pear.loc[a, b], spear.loc[a, b]))
    flagged = [p for p in pairs if max(abs(p[2]), abs(p[3])) > 0.7]
    if flagged:
        print("    pasangan dengan |r| > 0.7 pada salah satu ukuran:")
        for a, b, rp, rs in sorted(flagged, key=lambda t: -max(abs(t[2]), abs(t[3]))):
            print(f"      {a} - {b}:  Pearson {rp:+.3f}   Spearman {rs:+.3f}")

    # Selisih besar berarti hubungannya monoton tapi tidak linear, atau Pearson
    # terseret nilai ekstrem. Keduanya perlu disebut kalau muncul.
    diverge = [p for p in pairs if abs(p[2] - p[3]) > 0.15]
    if diverge:
        print("    Pearson dan Spearman berbeda jauh (hubungan non-linear "
              "atau pengaruh nilai ekstrem):")
        for a, b, rp, rs in sorted(diverge, key=lambda t: -abs(t[2] - t[3])):
            print(f"      {a} - {b}:  Pearson {rp:+.3f}   Spearman {rs:+.3f}"
                  f"   selisih {rp - rs:+.3f}")

    vif.to_csv(f"vif_{hazard}_{ABSENCE_MODE}.csv", index=False)
    pear.to_csv(f"corr_pearson_{hazard}_{ABSENCE_MODE}.csv")
    spear.to_csv(f"corr_spearman_{hazard}_{ABSENCE_MODE}.csv")
    return vif, pear, spear


pearsons, spearmans = {}, {}
for hz, df in samples.items():
    _, pearsons[hz], spearmans[hz] = collinearity(df, HAZARD_BANDS[hz], hz)

n = len(pearsons)
fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 9.4), squeeze=False)
for row, (label, mats) in enumerate((("Pearson", pearsons),
                                     ("Spearman", spearmans))):
    for k, (hz, corr) in enumerate(mats.items()):
        ax = axes[row][k]
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        ax.set_title(f"({'abcdef'[row * n + k]}) {hz} — {label}", fontsize=10)
        for i in range(len(corr)):
            for j in range(len(corr)):
                v = corr.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.55 else "black")
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="correlation")
fig.savefig(f"FigS1_correlation_{ABSENCE_MODE}.png", dpi=600, bbox_inches="tight")
print(f"\n  FigS1_correlation_{ABSENCE_MODE}.png tersimpan (600 dpi)")
plt.show()

# %% [markdown]
# ## Sel 9 — Validasi silang (komentar #3 dan #25)
#
# Selisih antara `random_fold` dan `spatial_fold` adalah angka yang diminta
# reviewer. Penurunan AUC pada skema spasial itu wajar dan justru menunjukkan
# validasinya benar.

# %%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

def model(name, seed):
    return {"RF": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
            "GTB": GradientBoostingClassifier(n_estimators=100, random_state=seed),
            "CART": DecisionTreeClassifier(random_state=seed)}[name]

def prep(df, bands):
    X = df[bands].copy()
    cats = [b for b in bands if b in CATEGORICAL_BANDS]
    if cats:
        X = pd.get_dummies(X, columns=cats, prefix=cats)
    return X.astype(float).values, df.Kelas.astype(int).values

def ev(X, y, name, tr, te, seed):
    if te.sum() < 20 or len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
        return None
    p = model(name, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y[te], p), accuracy_score(y[te], (p >= .5).astype(int))

records = []
for hz, df in samples.items():
    X, y = prep(df, HAZARD_BANDS[hz])
    print(f"\n  {hz}")
    for name in ("RF", "GTB", "CART"):
        r = np.random.RandomState(RANDOM_SEED).rand(len(df))
        res = ev(X, y, name, r < .7, r >= .7, RANDOM_SEED)
        if res:
            records.append(dict(hazard=hz, classifier=name, scheme="holdout_70_30",
                                n_folds=1, auc_mean=res[0], auc_sd=np.nan,
                                acc_mean=res[1], absence_mode=ABSENCE_MODE))
        for scheme in ("random_fold", "spatial_fold"):
            aucs, accs = [], []
            for rep in range(N_REPEATS if scheme == "random_fold" else 1):
                folds = (df.spatial_fold.values if scheme == "spatial_fold"
                         else np.random.RandomState(RANDOM_SEED+rep).randint(0, N_FOLDS, len(df)))
                for k in range(N_FOLDS):
                    res = ev(X, y, name, folds != k, folds == k, RANDOM_SEED)
                    if res:
                        aucs.append(res[0]); accs.append(res[1])
            if aucs:
                records.append(dict(hazard=hz, classifier=name, scheme=scheme,
                                    n_folds=len(aucs), auc_mean=float(np.mean(aucs)),
                                    auc_sd=float(np.std(aucs, ddof=1)) if len(aucs) > 1 else np.nan,
                                    acc_mean=float(np.mean(accs)), absence_mode=ABSENCE_MODE))
        for rec in [x for x in records if x["hazard"] == hz and x["classifier"] == name]:
            sd = "" if np.isnan(rec["auc_sd"]) else f" +/- {rec['auc_sd']:.3f}"
            print(f"    {name:<5s} {rec['scheme']:<16s} AUC {rec['auc_mean']:.3f}{sd}")

metrics = pd.DataFrame(records)
metrics.to_csv(f"validation_metrics_{ABSENCE_MODE}.csv", index=False)
print(f"\nvalidation_metrics_{ABSENCE_MODE}.csv tersimpan")

# %% [markdown]
# ## Sel 10 — Tabel untuk naskah

# %%
m = metrics.copy()
m["AUC"] = m.apply(lambda r: f"{r.auc_mean:.3f}" if np.isnan(r.auc_sd)
                   else f"{r.auc_mean:.3f} +/- {r.auc_sd:.3f}", axis=1)
tab = m.pivot_table(index=["hazard", "classifier"], columns="scheme",
                    values="AUC", aggfunc="first")
tab = tab[[c for c in ("holdout_70_30", "random_fold", "spatial_fold") if c in tab.columns]]
print(tab.to_string())
tab.to_csv(f"TableS1_validation_{ABSENCE_MODE}.csv")

d = metrics.pivot_table(index=["hazard", "classifier"], columns="scheme", values="auc_mean")
if {"random_fold", "spatial_fold"} <= set(d.columns):
    d["delta"] = d.spatial_fold - d.random_fold
    print("\nSelisih AUC (spasial - acak):")
    for (h, c), r in d.iterrows():
        print(f"  {h:<10s} {c:<5s} {r.random_fold:.3f} -> {r.spatial_fold:.3f}  ({r.delta:+.3f})")
    print(f"\n  rata-rata: {d.delta.mean():+.3f}")
