# src/predict.py
import argparse
import json
import joblib
import pandas as pd
import numpy as np

def build_input_row(d, meta):
    # d = dict com campos conforme enunciado
    num_features = meta["num_features"]
    genre_cols = meta["genre_cols"]
    out = {}
    out["Meta_score"] = d.get("Meta_score", np.nan)
    runtime = d.get("Runtime", None)
    if isinstance(runtime, str):
        import re
        m = re.search(r"(\d+)", runtime)
        runtime = int(m.group(1)) if m else None
    out["Runtime_min"] = runtime
    ry = d.get("Released_Year", d.get("Released_Year_num", np.nan))
    out["Released_Year_num"] = int(ry) if ry else np.nan
    votes = d.get("No_of_Votes", np.nan)
    out["log_votes"] = np.log1p(votes) if pd.notna(votes) else np.nan
    gross = d.get("Gross", None)
    if isinstance(gross, str):
        gross = float(gross.replace(",",""))
    out["log_gross"] = np.log1p(gross) if gross else np.nan

    out["Certificate"] = d.get("Certificate", np.nan)
    # Director/Stars top mapping (if present in meta)
    # We'll rely on pipeline's imputer/onehot -> "Other" not strictly necessary here

    # genres: multi-hot
    genres = d.get("Genre", "")
    if isinstance(genres, str):
        genres_list = [g.strip() for g in genres.split(",") if g.strip()]
    else:
        genres_list = list(genres)
    for gcol in genre_cols:
        gname = gcol.replace("genre_","")
        out[gcol] = 1 if gname in genres_list else 0

    # ensure all columns present
    Xcols = num_features + ["Certificate","Director_top","Star1_top","Star2_top","Star3_top","Star4_top"] + genre_cols
    for c in Xcols:
        if c not in out:
            out[c] = np.nan if c not in genre_cols else 0
    return pd.DataFrame([out])[Xcols]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="JSON file with movie fields")
    args = parser.parse_args()

    m = joblib.load(args.model)
    pipeline = m["pipeline"]
    meta = m["meta"]
    with open(args.input) as f:
        d = json.load(f)
    Xnew = build_input_row(d, meta)
    pred = pipeline.predict(Xnew)[0]
    print(f"Predicted IMDb_Rating: {pred:.3f}")

if __name__ == "__main__":
    main()
