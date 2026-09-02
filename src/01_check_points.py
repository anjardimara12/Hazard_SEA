

import glob
import os
import re
import sys
import zipfile

import pandas as pd

# ------------------------------------------------------------------ konfigurasi
DRIVE_DIR = "/content/drive/MyDrive/02. Hazard SEA Tesis Viola/00. Data/01. Point Hazard"

FILES = {
    "landslide": {"presence": "Copy of titik_tanahlongsorvioterbaru.zip",
                  "absence":  "Copy of TitikNonLongsor.zip"},
    "wildfire":  {"presence": "Copy of api.zip",
                  "absence":  "Copy of nonapi2.zip"},
    "flood":     {"presence": "Copy of titikbanjirvio.zip",
                  "absence":  "Copy of nonbanjir3.zip"},
}

# Angka pembanding dari tesis (Tabel III.2): kejadian, non-kejadian
THESIS = {"landslide": (2483, 553), "wildfire": (4050, 748), "flood": (150, 134)}

# Nama kolom yang mungkin dipakai untuk label kelas.
CLASS_NAMES = ["kelas", "class", "label", "target", "y", "value", "gridcode",
               "kls", "tipe", "type", "status", "hazard", "presence", "occur"]


# ------------------------------------------------------------------ pembacaan
def find_vector(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
    for ext in (".shp", ".gpkg", ".geojson", ".json", ".kml", ".csv"):
        hits = sorted([n for n in names if n.lower().endswith(ext)
                       and not os.path.basename(n).startswith((".", "__MACOSX"))],
                      key=len)
        if hits:
            return hits[0], names
    raise RuntimeError(f"tidak ada file vektor. Isi zip: {names[:12]}")


def read_zip(path):
    import geopandas as gpd
    inner, _ = find_vector(path)
    uri = f"zip://{path}!{inner}"
    try:
        return gpd.read_file(uri), inner
    except Exception:
        return gpd.read_file(uri, engine="fiona"), inner


def binary_like(sr):
    """True kalau kolom hanya berisi dua nilai yang terlihat seperti 0/1."""
    vals = set(pd.Series(sr).dropna().unique().tolist())
    if len(vals) != 2:
        return False
    try:
        nums = {float(v) for v in vals}
    except (TypeError, ValueError):
        return False
    return nums == {0.0, 1.0}


# ------------------------------------------------------------------ pemeriksaan
def inspect(path, label):
    gdf, inner = read_zip(path)
    n = len(gdf)
    cols = [c for c in gdf.columns if c != gdf.geometry.name]

    print(f"\n  {label}")
    print(f"    berkas dalam zip : {inner}")
    print(f"    jumlah fitur     : {n}")
    print(f"    CRS              : {gdf.crs}")
    print(f"    geometri         : {sorted(gdf.geom_type.unique().tolist())}")
    print(f"    kolom atribut    : {cols if cols else '(tidak ada)'}")

    # Kolom bernama seperti kelas
    named = [c for c in cols if c.strip().lower() in CLASS_NAMES]
    # Kolom apa pun yang isinya biner 0/1
    binaries = [c for c in cols if binary_like(gdf[c])]
    candidates = list(dict.fromkeys(named + binaries))

    result = {"label": label, "n": n, "class_col": None,
              "n_ones": None, "n_zeros": None}

    if not candidates:
        print("    kolom kelas      : tidak ditemukan")
        if cols:
            print("    sebaran nilai tiap kolom (maks 6 nilai unik):")
            for c in cols:
                u = pd.Series(gdf[c]).dropna().unique()
                shown = u[:6].tolist()
                print(f"      {c}: {len(u)} nilai unik {shown}"
                      f"{' ...' if len(u) > 6 else ''}")
        return result

    for c in candidates:
        vc = pd.Series(gdf[c]).value_counts(dropna=False).sort_index()
        tag = " (nama cocok)" if c in named else " (isi biner 0/1)"
        print(f"    kolom '{c}'{tag}:")
        for v, k in vc.items():
            print(f"      nilai {v!r}: {k}")
        if result["class_col"] is None and binary_like(gdf[c]):
            s = pd.to_numeric(gdf[c], errors="coerce")
            result.update(class_col=c, n_ones=int((s == 1).sum()),
                          n_zeros=int((s == 0).sum()))
    return result


def main():
    if not os.path.isdir(DRIVE_DIR):
        sys.exit(f"folder tidak ada: {DRIVE_DIR}")

    print("=" * 72)
    print("PEMERIKSAAN ATRIBUT KELAS")
    print("=" * 72)

    found = {}
    for hazard, roles in FILES.items():
        print(f"\n{'-' * 72}\n{hazard.upper()}")
        for role, fname in roles.items():
            path = os.path.join(DRIVE_DIR, fname)
            if not os.path.exists(path):
                print(f"\n  {role}: TIDAK ADA — {fname}")
                continue
            try:
                found[(hazard, role)] = inspect(path, f"{role}: {fname}")
            except Exception as e:
                print(f"\n  {role}: GAGAL — {type(e).__name__}: {e}")

    # ------------------------------------------------------------- kesimpulan
    print(f"\n{'=' * 72}\nKESIMPULAN\n{'=' * 72}")
    rows = []
    for hazard in FILES:
        te, tn = THESIS[hazard]
        p = found.get((hazard, "presence"))
        a = found.get((hazard, "absence"))
        if not (p and a):
            continue

        # Hipotesis: berkas "absence" sebenarnya gabungan kejadian + non-kejadian
        combined = a["class_col"] is not None and a["n_ones"] > 0
        if combined:
            verdict = (f"GABUNGAN: berisi {a['n_ones']} kelas 1 dan "
                       f"{a['n_zeros']} kelas 0")
            eff_abs = a["n_zeros"]
        else:
            verdict = "murni non-kejadian (tidak ada kelas 1)"
            eff_abs = a["n"]

        rows.append({
            "bahaya": hazard,
            "presence berkas": p["n"],
            "tesis kejadian": te,
            "absence berkas": a["n"],
            "non-kejadian efektif": eff_abs,
            "tesis non-kejadian": tn,
            "cocok tesis?": "ya" if (p["n"] == te and eff_abs == tn) else "TIDAK",
        })
        print(f"\n  {hazard}: {verdict}")

    if rows:
        print()
        print(pd.DataFrame(rows).to_string(index=False))
        print("\n  Kalau kolom 'non-kejadian efektif' cocok dengan kolom tesis,")
        print("  angka 553 / 748 / 134 di naskah sudah benar dan berkas Drive")
        print("  memang gabungan. Kalau tidak cocok, keduanya harus dijelaskan.")


if __name__ == "__main__":
    main()
else:
    main()
