
"""
DSA0402 – FUNDAMENTALS OF DATA SCIENCE
Bank Customer Subscription Prediction and Customer Segmentation System

This implementation follows the uploaded assignment specification:
Data Retrieval -> Cleaning -> Transformation -> EDA -> Descriptive Statistics
-> Statistical Inference -> kNN/Decision Tree/Logistic Regression
-> Evaluation -> Model Selection -> K-means -> Segmentation -> Recommendations.

Dataset:
UCI Machine Learning Repository – Bank Marketing (bank-full.csv)
Expected file: bank-full.csv, semicolon separated.

The script can also download the dataset through ucimlrepo if the local CSV
is not present. Run:
    pip install pandas numpy scipy scikit-learn matplotlib python-docx ucimlrepo

Then:
    python bank_marketing_solution.py

All generated tables and figures are written to ./outputs.
"""

from __future__ import annotations

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score, roc_auc_score
)

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_DATA = DATA_DIR / "bank-full.csv"

# Official UCI download endpoint. The ucimlrepo fallback is preferred when
# available because it directly identifies dataset 222.
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip"


def load_dataset() -> pd.DataFrame:
    """Load bank-full.csv locally, otherwise retrieve it from UCI."""
    if LOCAL_DATA.exists():
        df = pd.read_csv(LOCAL_DATA, sep=";", quotechar='"')
        return clean_column_names(df)

    # Preferred programmatic retrieval.
    try:
        from ucimlrepo import fetch_ucirepo
        bank_marketing = fetch_ucirepo(id=222)
        X = bank_marketing.data.features.copy()
        y = bank_marketing.data.targets.copy()
        if isinstance(y, pd.DataFrame):
            target_name = y.columns[0]
            df = X.copy()
            df[target_name] = y[target_name]
        else:
            df = X.copy()
            df["y"] = y
        # UCI's dataset 222 exposes bank-full as one of the supplied files.
        # If the package returns the 45,211-row bank dataset, save it locally.
        df.to_csv(LOCAL_DATA, sep=";", index=False)
        return clean_column_names(df)
    except Exception as exc:
        print("Automatic UCI retrieval failed:", exc)

    raise FileNotFoundError(
        "bank-full.csv was not found. Download the UCI Bank Marketing dataset "
        "and place bank-full.csv inside the data folder, then rerun."
    )


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().str.replace('"', '', regex=False)
    return df


def basic_preprocessing(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean duplicates, validate missingness, and create modelling target."""
    df = clean_column_names(df)
    before = len(df)

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # UCI reports no formal missing values. 'unknown' is retained as an
    # explicit category because it is information-bearing rather than NA.
    missing_before = int(df.isna().sum().sum())
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].replace({"": np.nan, "None": np.nan})
    missing_after = int(df.isna().sum().sum())

    # Only rows with a missing target are unusable. For predictor missingness,
    # numeric values use the median and categorical values use the mode.
    if "y" not in df.columns:
        raise ValueError("Target column 'y' is missing.")
    df = df.dropna(subset=["y"]).copy()

    for c in df.columns:
        if c == "y":
            continue
        if df[c].isna().any():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                mode = df[c].mode(dropna=True)
                df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "unknown")

    df["target"] = df["y"].map({"yes": 1, "no": 0})
    if df["target"].isna().any():
        raise ValueError("Unexpected target values found in y.")

    # Basic range checks. We do not delete legitimate financial outliers.
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["campaign"] = pd.to_numeric(df["campaign"], errors="coerce")
    df["pdays"] = pd.to_numeric(df["pdays"], errors="coerce")
    df["previous"] = pd.to_numeric(df["previous"], errors="coerce")
    df["day"] = pd.to_numeric(df["day"], errors="coerce")

    report = {
        "rows_before": before,
        "duplicates_removed": duplicate_count,
        "missing_values_before": missing_before,
        "missing_values_after_cleaning": int(df.isna().sum().sum()),
        "rows_after": len(df),
    }
    return df, report


def save_preprocessing_report(report: Dict[str, int]) -> None:
    pd.DataFrame([report]).to_csv(OUT_DIR / "preprocessing_summary.csv", index=False)


def descriptive_analysis(df: pd.DataFrame) -> None:
    numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    desc = df[numeric_cols].describe().T
    desc["variance"] = df[numeric_cols].var()
    desc.to_csv(OUT_DIR / "descriptive_statistics.csv")

    covariance = df[numeric_cols].cov()
    correlation = df[numeric_cols].corr()
    covariance.to_csv(OUT_DIR / "covariance_matrix.csv")
    correlation.to_csv(OUT_DIR / "correlation_matrix.csv")

    # A compact set of pairwise values for the report.
    pairs = []
    for a, b in [("age", "balance"), ("age", "duration"),
                 ("balance", "duration"), ("duration", "campaign"),
                 ("pdays", "previous")]:
        if a in df.columns and b in df.columns:
            pairs.append({
                "variable_1": a,
                "variable_2": b,
                "covariance": covariance.loc[a, b],
                "correlation": correlation.loc[a, b],
            })
    pd.DataFrame(pairs).to_csv(OUT_DIR / "selected_covariance_correlation.csv", index=False)


def statistical_inference(df: pd.DataFrame) -> Dict[str, float]:
    """95% confidence interval for the population subscription proportion."""
    n = len(df)
    yes = int(df["target"].sum())
    p_hat = yes / n
    z = stats.norm.ppf(0.975)
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    lower = max(0.0, p_hat - z * se)
    upper = min(1.0, p_hat + z * se)

    result = {
        "sample_size": n,
        "subscribers": yes,
        "estimated_subscription_rate": p_hat,
        "confidence_level": 0.95,
        "ci_lower": lower,
        "ci_upper": upper,
    }
    pd.DataFrame([result]).to_csv(OUT_DIR / "confidence_interval.csv", index=False)
    return result


def save_eda_figures(df: pd.DataFrame) -> None:
    # Numerical distributions.
    for col in ["age", "balance", "duration", "campaign"]:
        if col not in df.columns:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(df[col], bins=30)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"distribution_{col}.png", dpi=180)
        plt.close()

    # Categorical frequencies.
    for col in ["job", "education", "housing", "poutcome"]:
        if col not in df.columns:
            continue
        counts = df[col].value_counts().sort_values(ascending=True)
        plt.figure(figsize=(9, 5))
        plt.barh(counts.index.astype(str), counts.values)
        plt.xlabel("Number of customers")
        plt.ylabel(col)
        plt.title(f"Frequency of {col}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"frequency_{col}.png", dpi=180)
        plt.close()

    # Subscriber vs non-subscriber comparison.
    if "age" in df.columns:
        grouped = df.groupby("y")["age"].mean().reindex(["no", "yes"])
        plt.figure(figsize=(6, 4))
        plt.bar(grouped.index, grouped.values)
        plt.xlabel("Subscription outcome")
        plt.ylabel("Mean age")
        plt.title("Mean age by subscription outcome")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "subscriber_non_subscriber_age.png", dpi=180)
        plt.close()

    if "balance" in df.columns:
        grouped = df.groupby("y")["balance"].median().reindex(["no", "yes"])
        plt.figure(figsize=(6, 4))
        plt.bar(grouped.index, grouped.values)
        plt.xlabel("Subscription outcome")
        plt.ylabel("Median balance")
        plt.title("Median balance by subscription outcome")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "subscriber_non_subscriber_balance.png", dpi=180)
        plt.close()

    # Outlier analysis using boxplots.
    cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    cols = [c for c in cols if c in df.columns]
    for col in cols:
        plt.figure(figsize=(7, 4))
        plt.boxplot(df[col].dropna(), vert=False)
        plt.xlabel(col)
        plt.title(f"Outlier analysis: {col}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"outlier_{col}.png", dpi=180)
        plt.close()

    # Correlation matrix.
    numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(9, 7))
    plt.imshow(corr, interpolation="nearest", aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_matrix.png", dpi=180)
    plt.close()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )


def prepare_modelling_data(df: pd.DataFrame):
    # 'duration' is intentionally excluded from the predictive model because
    # it is only known after a call and can create target leakage in a
    # pre-campaign targeting system. It remains in EDA and descriptive analysis.
    drop_cols = ["y", "target", "duration"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df["target"].astype(int).copy()
    return X, y


def train_models(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "kNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="liblinear",
            random_state=RANDOM_STATE
        ),
    }

    results = []
    fitted = {}
    predictions = {}

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor(X_train)),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        else:
            auc = np.nan

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1_score": f1_score(y_test, pred, zero_division=0),
            "roc_auc": auc,
        })
        fitted[name] = pipe
        predictions[name] = pred

        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation="nearest")
        plt.colorbar()
        plt.xticks([0, 1], ["Not Subscribed", "Subscribed"], rotation=20)
        plt.yticks([0, 1], ["Not Subscribed", "Subscribed"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, int(cm[i, j]), ha="center", va="center")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"Confusion Matrix – {name}")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(FIG_DIR / f"confusion_matrix_{safe_name}.png", dpi=180)
        plt.close()

    metrics = pd.DataFrame(results).sort_values(
        ["f1_score", "roc_auc", "recall"], ascending=False
    )
    metrics.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    # Performance comparison plot.
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(metrics))
    width = 0.18
    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(metric_names):
        plt.bar(x + (i - 1.5) * width, metrics[metric], width, label=metric.upper())
    plt.xticks(x, metrics["model"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Classification model performance comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_performance_comparison.png", dpi=180)
    plt.close()

    best_name = metrics.iloc[0]["model"]
    return metrics, fitted, X_train, X_test, y_train, y_test, predictions, best_name


def cluster_customers(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Choose K by silhouette score and profile the resulting customer groups."""
    # Clustering uses pre-campaign demographic, financial and behaviour fields.
    numeric = ["age", "balance", "campaign", "pdays", "previous"]
    categorical = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]

    Xc = df[numeric + categorical].copy()
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    Z = pre.fit_transform(Xc)

    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(5000, len(df))
    sample_idx = rng.choice(len(df), size=sample_size, replace=False)
    Z_sample = Z[sample_idx]

    scores = []
    models = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Z_sample)
        score = silhouette_score(Z_sample, labels)
        scores.append((k, score))
        models[k] = km

    score_df = pd.DataFrame(scores, columns=["k", "silhouette_score"])
    score_df.to_csv(OUT_DIR / "kmeans_silhouette_scores.csv", index=False)
    best_k = int(score_df.loc[score_df["silhouette_score"].idxmax(), "k"])

    plt.figure(figsize=(7, 4))
    plt.plot(score_df["k"], score_df["silhouette_score"], marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette score")
    plt.title("K selection using silhouette analysis")
    plt.xticks(score_df["k"])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kmeans_silhouette.png", dpi=180)
    plt.close()

    final_km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    df = df.copy()
    df["cluster"] = final_km.fit_predict(Z)

    profile = df.groupby("cluster").agg(
        customers=("cluster", "size"),
        mean_age=("age", "mean"),
        mean_balance=("balance", "mean"),
        median_balance=("balance", "median"),
        mean_campaign_contacts=("campaign", "mean"),
        mean_previous_contacts=("previous", "mean"),
        subscription_rate=("target", "mean"),
    ).reset_index()
    profile["subscription_rate_percent"] = profile["subscription_rate"] * 100
    profile.to_csv(OUT_DIR / "cluster_profiles.csv", index=False)

    # Cluster visualisation using age and balance. These are interpretable
    # dimensions rather than a claim that the clustering used only two fields.
    plt.figure(figsize=(8, 5))
    for cluster_id in sorted(df["cluster"].unique()):
        part = df[df["cluster"] == cluster_id]
        plt.scatter(part["age"], part["balance"], s=8, alpha=0.35, label=f"Cluster {cluster_id}")
    plt.xlabel("Age")
    plt.ylabel("Balance")
    plt.title(f"Customer segments from K-means (K={best_k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kmeans_customer_clusters.png", dpi=180)
    plt.close()

    return df, best_k


def generate_text_summary(df: pd.DataFrame, prep_report, inference, metrics, best_name, best_k):
    best_row = metrics.loc[metrics["model"] == best_name].iloc[0]
    lines = [
        "BANK CUSTOMER SUBSCRIPTION PROJECT – EXECUTION SUMMARY",
        "",
        f"Rows before cleaning: {prep_report['rows_before']}",
        f"Duplicate rows removed: {prep_report['duplicates_removed']}",
        f"Rows after cleaning: {prep_report['rows_after']}",
        f"Missing values after cleaning: {prep_report['missing_values_after_cleaning']}",
        f"Estimated subscription rate: {inference['estimated_subscription_rate']:.4f}",
        f"95% CI for subscription rate: ({inference['ci_lower']:.4f}, {inference['ci_upper']:.4f})",
        "",
        "MODEL COMPARISON",
        metrics.to_string(index=False),
        "",
        f"Selected best model: {best_name}",
        f"Best-model F1: {best_row['f1_score']:.4f}",
        f"Best-model recall: {best_row['recall']:.4f}",
        f"Best-model ROC-AUC: {best_row['roc_auc']:.4f}",
        "",
        f"Selected K for K-means: {best_k}",
        "",
        "Important design choice: duration is retained for EDA but excluded from the",
        "pre-campaign predictive model because it is only known after a call.",
    ]
    (OUT_DIR / "execution_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def menu(df, metrics, best_name, fitted):
    """Menu-driven functionality required by the assignment rubric."""
    while True:
        print("\n===== BANK CUSTOMER ANALYTICS MENU =====")
        print("1. Show dataset summary")
        print("2. Show subscription distribution")
        print("3. Show model comparison")
        print("4. Show cluster profiles")
        print("5. Show correlation matrix")
        print("6. Predict subscription for a new customer")
        print("0. Exit")

        choice = input("Enter choice: ").strip()
        if choice == "1":
            print(df.info())
            print(df.head())
        elif choice == "2":
            print(df["y"].value_counts())
            print(df["y"].value_counts(normalize=True).mul(100).round(2))
        elif choice == "3":
            print(metrics.to_string(index=False))
            print(f"\nBest model: {best_name}")
        elif choice == "4":
            path = OUT_DIR / "cluster_profiles.csv"
            if path.exists():
                print(pd.read_csv(path).to_string(index=False))
            else:
                print("Cluster profile is not available.")
        elif choice == "5":
            print(pd.read_csv(OUT_DIR / "correlation_matrix.csv").to_string(index=False))
        elif choice == "6":
            print("\n--- New Customer Subscription Prediction ---")
            print(f"Using best model: {best_name}")
            print("Enter customer details. Do not enter 'duration' because it is excluded")
            print("from prediction to avoid target leakage.\n")

            def get_text(prompt):
                return input(prompt).strip().lower()

            def get_int(prompt, minimum=None):
                while True:
                    try:
                        value = int(input(prompt).strip())
                        if minimum is not None and value < minimum:
                            print(f"Please enter a value >= {minimum}.")
                            continue
                        return value
                    except ValueError:
                        print("Please enter a valid integer.")

            new_customer = pd.DataFrame([{
                "age": get_int("Age: ", 18),
                "job": get_text("Job: "),
                "marital": get_text("Marital (married/single/divorced): "),
                "education": get_text("Education (primary/secondary/tertiary/unknown): "),
                "default": get_text("Default (yes/no): "),
                "balance": get_int("Balance: "),
                "housing": get_text("Housing loan (yes/no): "),
                "loan": get_text("Personal loan (yes/no): "),
                "contact": get_text("Contact (cellular/telephone/unknown): "),
                "day": get_int("Last contact day (1-31): ", 1),
                "month": get_text("Last contact month (jan-dec): "),
                "campaign": get_int("Number of contacts in this campaign: ", 1),
                "pdays": get_int("Days since previous campaign contact (-1 if never contacted): ", -1),
                "previous": get_int("Number of previous contacts: ", 0),
                "poutcome": get_text("Previous outcome (failure/other/success/unknown): ")
            }])

            selected_model = fitted[best_name]
            prediction = int(selected_model.predict(new_customer)[0])

            if hasattr(selected_model, "predict_proba"):
                probability = float(selected_model.predict_proba(new_customer)[0][1])
            else:
                probability = None

            result_text = "SUBSCRIBED" if prediction == 1 else "NOT SUBSCRIBED"

            print("\n==============================")
            print("       PREDICTION RESULT")
            print("==============================")
            print(f"Customer is predicted to: {result_text}")
            if probability is not None:
                print(f"Probability of subscription: {probability:.2%}")
            print(f"Best model used: {best_name}")
            print("==============================")

            result_row = new_customer.copy()
            result_row["predicted_subscription"] = result_text
            result_row["subscription_probability"] = probability
            result_row["model_used"] = best_name

            out_file = OUTPUT_DIR / "new_customer_prediction.csv"
            result_row.to_csv(out_file, index=False)
            print(f"\nPrediction saved to: {out_file}")

        elif choice == "0":
            print("Exiting.")
            break
        else:
            print("Invalid choice.")


def main():
    df_raw = load_dataset()
    df, prep_report = basic_preprocessing(df_raw)
    save_preprocessing_report(prep_report)

    # Pandas operations explicitly required by the assignment.
    # Filtering and grouping:
    high_balance = df[df["balance"] > df["balance"].median()]
    job_summary = (
        df.groupby("job")
          .agg(customers=("job", "size"),
               subscription_rate=("target", "mean"),
               mean_balance=("balance", "mean"))
          .sort_values("subscription_rate", ascending=False)
    )
    job_summary.to_csv(OUT_DIR / "job_group_summary_sorted.csv")
    high_balance.sort_values("balance", ascending=False).head(100).to_csv(
        OUT_DIR / "top_100_high_balance_customers.csv", index=False
    )

    descriptive_analysis(df)
    inference = statistical_inference(df)
    save_eda_figures(df)

    X, y = prepare_modelling_data(df)
    metrics, fitted, X_train, X_test, y_train, y_test, predictions, best_name = train_models(X, y)

    clustered_df, best_k = cluster_customers(df)
    clustered_df[["age", "job", "marital", "education", "balance", "housing",
                  "loan", "campaign", "pdays", "previous", "poutcome", "y",
                  "cluster"]].to_csv(OUT_DIR / "customer_segments.csv", index=False)

    generate_text_summary(df, prep_report, inference, metrics, best_name, best_k)

    print("\nAnalysis completed successfully.")
    print(f"Outputs: {OUT_DIR}")
    print(f"Best model: {best_name}")
    print(f"Selected K: {best_k}")

    # Menu can be enabled when interactive use is required.
    if sys.stdin.isatty():
        try:
            menu(df, metrics, best_name, fitted)
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
