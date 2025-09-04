# src/main.py
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/desafio_indicium_imdb.csv")
    p.add_argument("--out-model", default="models/imdb_rating_model.pkl")
    return p.parse_args()

def load_and_clean(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["Released_Year_num"] = df["Released_Year"].astype(str).str.extract(r"(\d{4})").astype(float)
    df["Runtime_min"] = df["Runtime"].astype(str).str.extract(r"(\d+)").astype(float)
    df["Gross_num"] = df["Gross"].astype(str).replace({",": ""}, regex=True).replace("nan", np.nan).apply(lambda x: float(x) if x and x != "nan" else np.nan)
    df["Genre_list"] = df["Genre"].fillna("").apply(lambda s: [g.strip() for g in s.split(",") if g.strip()])
    df["Genre_main"] = df["Genre_list"].apply(lambda lst: lst[0] if len(lst) > 0 else np.nan)
    df["log_votes"] = np.log1p(df["No_of_Votes"])
    df["log_gross"] = np.log1p(df["Gross_num"])
    return df

def train_rating_model(df):
    # Multi-hot genres
    mlb2 = MultiLabelBinarizer()
    genre_mh2 = mlb2.fit_transform(df["Genre_list"])
    genre_cols2 = [f"genre_{g}" for g in mlb2.classes_]
    df2 = pd.concat([df.reset_index(drop=True), pd.DataFrame(genre_mh2, columns=genre_cols2)], axis=1)

    # Top-k for categorical high-cardinality
    def top_k_or_other(s, k=50):
        top = s.value_counts().head(k).index
        return s.where(s.isin(top), "Other")

    df2["Director_top"] = top_k_or_other(df2["Director"], k=50)
    for col in ["Star1","Star2","Star3","Star4"]:
        df2[col+"_top"] = top_k_or_other(df2[col], k=50)

    num_features = ["Meta_score","Runtime_min","Released_Year_num","log_votes","log_gross"]
    cat_features = ["Certificate","Director_top","Star1_top","Star2_top","Star3_top","Star4_top"]

    X = df2[num_features + cat_features + genre_cols2]
    y = df2["IMDB_Rating"]

    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
        ("genres", "passthrough", genre_cols2)
    ])

    gbr_model = Pipeline([("pre", preprocess), ("gbr", GradientBoostingRegressor(random_state=42))])
    gbr_model.fit(X, y)

    meta = {
        "pipeline": gbr_model,
        "genre_labels": mlb2.classes_.tolist(),
        "num_features": num_features,
        "cat_features": cat_features,
        "genre_cols": genre_cols2
    }
    return meta

def train_nlp_genre(df):
    nlp = df.dropna(subset=["Overview","Genre_main"]).copy()
    counts = nlp["Genre_main"].value_counts()
    valid_genres = counts[counts >= 20].index
    nlp = nlp[nlp["Genre_main"].isin(valid_genres)]
    X_text = nlp["Overview"].astype(str)
    y_genre = nlp["Genre_main"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X_text, y_genre, test_size=0.2, stratify=y_genre, random_state=42)
    pipe_nlp = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=3000)),
        ("clf", LogisticRegression(max_iter=200, solver="liblinear"))
    ])
    pipe_nlp.fit(X_train, y_train)
    # classification report saved in notebook/reports
    return pipe_nlp

def main():
    args = parse_args()
    p = Path(args.data)
    assert p.exists(), f"Arquivo não encontrado: {p}"
    df = load_and_clean(str(p))
    meta = train_rating_model(df)
    pipe_nlp = train_nlp_genre(df)

    # salvar modelo
    outp = Path(args.out_model)
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": meta["pipeline"], "meta": meta}, outp.as_posix())
    print(f"Modelo salvo em: {outp}")

if __name__ == "__main__":
    main()
