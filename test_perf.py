import pandas as pd
import numpy as np
import time

# Create dummy data
np.random.seed(42)
n1, n2 = 1000, 5000

in1 = pd.DataFrame({
    'ra': np.random.uniform(0, 360, n1),
    'dec': np.random.uniform(-90, 90, n1),
    'mag1': np.random.uniform(15, 20, n1),
    'ra_err': np.random.uniform(0.1, 0.5, n1),
    'dec_err': np.random.uniform(0.1, 0.5, n1)
})

in2 = pd.DataFrame({
    'ra': np.random.uniform(0, 360, n2),
    'dec': np.random.uniform(-90, 90, n2),
    'mag2': np.random.uniform(15, 20, n2),
    'ra_err': np.random.uniform(0.1, 0.5, n2),
    'dec_err': np.random.uniform(0.1, 0.5, n2)
})

# original
def skymatch_orig(in1, in2, ra1='ra', dec1='dec', ra2='ra', dec2='dec', error=1.0, matcher='skyerr', ra_err1='ra_err', dec_err1='dec_err', ra_err2='ra_err', dec_err2='dec_err'):
    rows = []
    target_ra = in2[ra2].to_numpy()
    target_dec = in2[dec2].to_numpy()

    for _, row1 in in1.iterrows():
        ra1_val = float(row1[ra1])
        dec1_val = float(row1[dec1])
        dra_arcsec = (target_ra - ra1_val) * np.cos(np.deg2rad(dec1_val)) * 3600.0
        ddec_arcsec = (target_dec - dec1_val) * 3600.0
        sep_arcsec = np.hypot(dra_arcsec, ddec_arcsec)
        best_idx = int(np.argmin(sep_arcsec))
        best_sep = float(sep_arcsec[best_idx])

        if matcher == "sky":
            threshold = float(error) * 3600.0
        else:
            sigma_thresh = float(error) * 20.0
            if ra_err1 and dec_err1 and ra_err2 and dec_err2:
                combined = np.sqrt(
                    float(row1[ra_err1]) ** 2
                    + float(row1[dec_err1]) ** 2
                    + float(in2.iloc[best_idx][ra_err2]) ** 2
                    + float(in2.iloc[best_idx][dec_err2]) ** 2
                )
                sigma_thresh = max(sigma_thresh, float(error) * combined)
            threshold = sigma_thresh

        if best_sep > threshold:
            continue

        out_row = {}
        for c in in1.columns:
            out_row[c] = row1[c]
        for c in in2.columns:
            if c in out_row:
                out_row[f"{c}_2"] = in2.iloc[best_idx][c]
            else:
                out_row[c] = in2.iloc[best_idx][c]
        out_row["separation"] = best_sep
        rows.append(out_row)
    return pd.DataFrame(rows)

def skymatch_opt(in1, in2, ra1='ra', dec1='dec', ra2='ra', dec2='dec', error=1.0, matcher='skyerr', ra_err1='ra_err', dec_err1='dec_err', ra_err2='ra_err', dec_err2='dec_err'):
    rows = []
    in1_cols = {c: in1[c].to_numpy() for c in in1.columns}
    in2_cols = {c: in2[c].to_numpy() for c in in2.columns}

    target_ra = in2_cols[ra2]
    target_dec = in2_cols[dec2]

    in1_len = len(in1)
    for i in range(in1_len):
        ra1_val = float(in1_cols[ra1][i])
        dec1_val = float(in1_cols[dec1][i])
        dra_arcsec = (target_ra - ra1_val) * np.cos(np.deg2rad(dec1_val)) * 3600.0
        ddec_arcsec = (target_dec - dec1_val) * 3600.0
        sep_arcsec = np.hypot(dra_arcsec, ddec_arcsec)
        best_idx = int(np.argmin(sep_arcsec))
        best_sep = float(sep_arcsec[best_idx])

        if matcher == "sky":
            threshold = float(error) * 3600.0
        else:
            sigma_thresh = float(error) * 20.0
            if ra_err1 and dec_err1 and ra_err2 and dec_err2:
                combined = np.sqrt(
                    float(in1_cols[ra_err1][i]) ** 2
                    + float(in1_cols[dec_err1][i]) ** 2
                    + float(in2_cols[ra_err2][best_idx]) ** 2
                    + float(in2_cols[dec_err2][best_idx]) ** 2
                )
                sigma_thresh = max(sigma_thresh, float(error) * combined)
            threshold = sigma_thresh

        if best_sep > threshold:
            continue

        out_row = {}
        for c in in1.columns:
            out_row[c] = in1_cols[c][i]
        for c in in2.columns:
            if c in out_row:
                out_row[f"{c}_2"] = in2_cols[c][best_idx]
            else:
                out_row[c] = in2_cols[c][best_idx]
        out_row["separation"] = best_sep
        rows.append(out_row)
    return pd.DataFrame(rows)

t0 = time.time()
res1 = skymatch_orig(in1, in2)
t1 = time.time()
print(f"Original: {t1-t0:.4f}s")

t0 = time.time()
res2 = skymatch_opt(in1, in2)
t1 = time.time()
print(f"Optimized: {t1-t0:.4f}s")

pd.testing.assert_frame_equal(res1, res2)
print("Results match.")
