
import os
import zipfile

import pandas as pd

DRIVE_DIR = "/content/drive/MyDrive/02. Hazard SEA Tesis Viola/00. Data/01. Point Hazard"

SOURCES = {
    "longsor":   ("Copy of titik_tanahlongsorvioterbaru.zip", ["event_date"]),
    "kebakaran": ("Copy of api.zip",                          ["acq_date"]),
    "banjir":    ("Copy of titikbanjirvio.zip",               ["dfo_began", "dfo_ended"]),
}

# Filter yang dipakai di pipeline, agar rentang sesuai data yang benar-benar dilatih.
FILTER = {"kebakaran": ("type", [0])}


def find_vector(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
    for ext in (".shp", ".gpkg", ".geojson", ".json"):
        hits = sorted([n for n in names if n.lower().endswith(ext)
                       and not os.path.basename(n).startswith((".", "__MACOSX"))],
                      key=len)
        if hits:
            return hits[0]
    raise RuntimeError(f"tidak ada file vektor: {names[:10]}")


def year_range(hazard, fname, date_cols):
    import geopandas as gpd

    path = os.path.join(DRIVE_DIR, fname)
    if not os.path.exists(path):
        print(f"\n  {hazard}: TIDAK ADA - {fname}")
        return None

    gdf = gpd.read_file(f"zip://{path}!{find_vector(path)}")
    n_all = len(gdf)

    flt = FILTER.get(hazard)
    if flt and flt[0] in gdf.columns:
        col, keep = flt
        gdf = gdf[pd.to_numeric(gdf[col], errors="coerce").isin(keep)]
        print(f"\n  {hazard}: {len(gdf)}/{n_all} titik setelah filter {col} in {keep}")
    else:
        print(f"\n  {hazard}: {n_all} titik")

    out = {}
    for c in date_cols:
        if c not in gdf.columns:
            print(f"    kolom '{c}' tidak ada. Kolom tersedia: {list(gdf.columns)[:25]}")
            continue
        d = pd.to_datetime(gdf[c], errors="coerce")
        valid = d.dropna()
        if valid.empty:
            print(f"    '{c}': tidak ada tanggal yang bisa diurai. "
                  f"Contoh nilai mentah: {gdf[c].dropna().unique()[:5]}")
            continue
        print(f"    '{c}': {valid.min().date()}  s/d  {valid.max().date()}"
              f"   ({len(valid)}/{len(gdf)} terbaca)")
        # sebaran per tahun, untuk melihat apakah ada ekor yang tipis
        yrs = valid.dt.year.value_counts().sort_index()
        print(f"      per tahun: {dict(yrs)}")
        out[c] = (int(valid.dt.year.min()), int(valid.dt.year.max()))
    return out


def main():
    print("=" * 70)
    print("RENTANG TAHUN INVENTARIS BAHAYA")
    print("=" * 70)

    hasil = {}
    for hazard, (fname, cols) in SOURCES.items():
        r = year_range(hazard, fname, cols)
        if r:
            hasil[hazard] = r

    print("\n" + "=" * 70)
    print("RINGKASAN untuk Tabel 2 dan Bagian 2.1")
    print("=" * 70)
    for hazard, cols in hasil.items():
        lo = min(v[0] for v in cols.values())
        hi = max(v[1] for v in cols.values())
        print(f"  {hazard:<10s} {lo}-{hi}")
    print("\n  Pakai rentang ini di KEDUA tempat, dan hapus angka lama.")
    print("  Kalau tahun terakhir jauh lebih tipis dari tahun lain pada")
    print("  sebaran di atas, pertimbangkan memotongnya dan sebutkan alasannya.")


if __name__ == "__main__":
    main()
else:
    main()
