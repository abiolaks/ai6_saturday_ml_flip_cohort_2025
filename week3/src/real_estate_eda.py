# --- Stakeholder HTML Report (single-file with embedded charts) ---
# Paste this cell into your notebook and run it.
import io, base64, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score


def _fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return b64


def _is_categorical(s, thresh=30):
    if s.dtype == "O":
        return True
    uniq = s.nunique(dropna=True)
    return uniq <= min(thresh, max(2, int(0.05 * len(s))))


def _try_parse_dates(_df):
    out = _df.copy()
    for c in out.columns:
        lc = str(c).lower()
        if ("date" in lc) or ("time" in lc):
            out[c] = pd.to_datetime(out[c], errors="ignore")
    return out


def _guess_target(_df, numeric_candidates):
    keys = [
        "saleprice",
        "price",
        "soldprice",
        "listprice",
        "askingprice",
        "rent",
        "rentalprice",
        "value",
        "valuation",
        "target",
        "label",
        "y",
    ]
    lm = {c.lower(): c for c in numeric_candidates}
    for k in keys:
        if k in lm:
            return lm[k]
        for lc, orig in lm.items():
            if k in lc:
                return orig
    return None


def _find_main_datetime_col(cols):
    priority = [
        "date",
        "saledate",
        "listdate",
        "posteddate",
        "transactiondate",
        "tx_date",
        "txdate",
    ]
    lm = {c.lower(): c for c in cols}
    for key in priority:
        for lc, orig in lm.items():
            if key in lc:
                return orig
    return cols[0] if cols else None


def _find_geo_cols(columns):
    lat_keys = ["latitude", "lat", "y"]
    lon_keys = ["longitude", "lon", "lng", "long", "x"]
    lm = {c.lower(): c for c in columns}
    lat = next(
        (orig for lc, orig in lm.items() if any(k in lc for k in lat_keys)), None
    )
    lon = next(
        (orig for lc, orig in lm.items() if any(k in lc for k in lon_keys)), None
    )
    return lat, lon


def _iqr_outlier_rate(s):
    s = s.dropna()
    if s.empty:
        return 0.0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float((((s < lower) | (s > upper)).mean() * 100).round(2))


def generate_stakeholder_report(
    df, output_path="real_estate_eda_stakeholder_report.html", sample_geo_n=5000
):
    # --- Type inference and preparation ---
    df = _try_parse_dates(df)

    # Optional: convert common year-only columns to datetime (YYYY-01-01)
    for col in ["year_built", "tx_year", "transaction_year"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str), format="%Y", errors="coerce")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [
        c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    categorical_cols = [
        c
        for c in df.columns
        if (c not in numeric_cols + datetime_cols) and _is_categorical(df[c])
    ]

    # Missingness
    missing_df = (
        (df.isna().mean() * 100).round(2).sort_values(ascending=False).reset_index()
    )
    missing_df.columns = ["Column", "Missing_%"]
    top_missing = missing_df.set_index("Column")["Missing_%"].head(20)

    # Outliers
    outlier_rates = pd.DataFrame(
        {
            "Column": numeric_cols,
            "Outlier_%": [_iqr_outlier_rate(df[c]) for c in numeric_cols],
        }
    ).sort_values("Outlier_%", ascending=False)

    # Correlation
    corr = df[numeric_cols].corr(numeric_only=True) if len(numeric_cols) >= 2 else None

    # Guess target and build quick model + feature importance
    target_col = _guess_target(df, numeric_cols)
    feat_importances_df = None
    model_summary_rows = []
    if target_col and target_col in df.columns:
        y = df[target_col]
        task_type = "regression"
        if not pd.api.types.is_numeric_dtype(y):
            task_type = "classification"
        else:
            if y.nunique(dropna=True) <= 10 and set(
                pd.Series(y.dropna().unique()).astype(float)
            ).issubset({0.0, 1.0}):
                task_type = "classification"

        # X = numeric + limited one-hot cats
        X_num = df[[c for c in numeric_cols if c != target_col]].copy()
        X_cat = pd.DataFrame(index=df.index)
        for c in categorical_cols:
            if df[c].nunique(dropna=True) <= 30:
                X_cat = pd.concat(
                    [X_cat, pd.get_dummies(df[c], prefix=c, dummy_na=True)], axis=1
                )
        X = pd.concat([X_num, X_cat], axis=1)

        valid = ~y.isna()
        X, y = X.loc[valid], y.loc[valid]
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

        if len(X) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if task_type == "regression":
                lr = LinearRegression().fit(X_train, y_train)
                ypl = lr.predict(X_test)
                model_summary_rows.append(
                    [
                        "LinearRegression",
                        f"RMSE={math.sqrt(mean_squared_error(y_test, ypl)):.2f}",
                        f"R2={r2_score(y_test, ypl):.3f}",
                    ]
                )

                rf = RandomForestRegressor(
                    n_estimators=300, random_state=42, n_jobs=-1
                ).fit(X_train, y_train)
                ypr = rf.predict(X_test)
                model_summary_rows.append(
                    [
                        "RandomForestRegressor",
                        f"RMSE={math.sqrt(mean_squared_error(y_test, ypr)):.2f}",
                        f"R2={r2_score(y_test, ypr):.3f}",
                    ]
                )

                feat_importances_df = pd.DataFrame(
                    {"Feature": X.columns, "Importance": rf.feature_importances_}
                ).sort_values("Importance", ascending=False)

            else:
                if not pd.api.types.is_numeric_dtype(y):
                    y = pd.factorize(y)[0]

                lg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                ypl = lg.predict(X_test)
                model_summary_rows.append(
                    [
                        "LogisticRegression",
                        f"ACC={accuracy_score(y_test, ypl):.3f}",
                        f"F1={f1_score(y_test, ypl, average='weighted'):.3f}",
                    ]
                )

                rf = RandomForestClassifier(
                    n_estimators=300, random_state=42, n_jobs=-1
                ).fit(X_train, y_train)
                ypr = rf.predict(X_test)
                model_summary_rows.append(
                    [
                        "RandomForestClassifier",
                        f"ACC={accuracy_score(y_test, ypr):.3f}",
                        f"F1={f1_score(y_test, ypr, average='weighted'):.3f}",
                    ]
                )

                # FIXED LINE: removed stray quote after rf.feature_importances_
                feat_importances_df = pd.DataFrame(
                    {"Feature": X.columns, "Importance": rf.feature_importances_}
                ).sort_values("Importance", ascending=False)

    # Time series
    main_dt = _find_main_datetime_col(datetime_cols) if datetime_cols else None
    monthly = None
    if main_dt and target_col and target_col in df.columns:
        tmp = df[[main_dt, target_col]].copy().dropna(subset=[main_dt])
        if not pd.api.types.is_datetime64_any_dtype(tmp[main_dt]):
            tmp[main_dt] = pd.to_datetime(tmp[main_dt], errors="coerce")
        tmp = tmp.dropna(subset=[main_dt])
        if not tmp.empty:
            tmp["Month"] = tmp[main_dt].dt.to_period("M").dt.to_timestamp()
            monthly = (
                tmp.groupby("Month")
                .agg(Volume=(target_col, "count"), MedianTarget=(target_col, "median"))
                .reset_index()
            )

    # Geo chart skipped: no latitude/longitude columns in dataset
    geo_img = None

    # --- Recreate key charts as base64 ---
    images = {}

    # Missingness chart
    if not top_missing.empty:
        plt.figure()
        top_missing.plot(kind="bar")
        plt.title("Top 20 Columns by Missingness (%)")
        plt.ylabel("Missing %")
        plt.tight_layout()
        images["missingness"] = _fig_to_base64()

    # Outliers chart
    if not outlier_rates.empty:
        plt.figure()
        outlier_rates.head(15).set_index("Column")["Outlier_%"].plot(kind="bar")
        plt.title("Top 15 Outlier-Prone Numeric Columns (IQR %)")
        plt.ylabel("Outlier %")
        plt.tight_layout()
        images["outliers"] = _fig_to_base64()

    # Correlation heatmap
    if corr is not None:
        plt.figure()
        plt.imshow(corr, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation Matrix (Numeric Features)")
        plt.tight_layout()
        images["correlation"] = _fig_to_base64()

    # Feature importances
    if feat_importances_df is not None and not feat_importances_df.empty:
        plt.figure()
        feat_importances_df.head(15).set_index("Feature")["Importance"].plot(kind="bar")
        plt.title("Top 15 Feature Importances (Random Forest)")
        plt.ylabel("Importance")
        plt.tight_layout()
        images["feat_importance"] = _fig_to_base64()

    # Time series charts
    if monthly is not None and not monthly.empty:
        plt.figure()
        plt.plot(monthly["Month"], monthly["Volume"])
        plt.title("Monthly Transaction Volume")
        plt.xlabel("Month")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        images["volume"] = _fig_to_base64()

        plt.figure()
        plt.plot(monthly["Month"], monthly["MedianTarget"])
        plt.title(f"Monthly Median {target_col}")
        plt.xlabel("Month")
        plt.ylabel(f"Median {target_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        images["median_target"] = _fig_to_base64()

    # --- Insights ---
    insights = []
    # Missingness
    high_miss = missing_df[missing_df["Missing_%"] >= 20.0]["Column"].tolist()
    if high_miss:
        insights.append(
            f"{len(high_miss)} fields have ≥20% missing values (e.g., {', '.join(high_miss[:5])}{'...' if len(high_miss) > 5 else ''}). Prioritize imputation or mandatory capture."
        )
    # Outliers
    heavy = outlier_rates[outlier_rates["Outlier_%"] >= 5.0]["Column"].tolist()
    if heavy:
        insights.append(
            f"{len(heavy)} numeric features show ≥5% outliers. Consider trimming/winsorization or robust scalers. Notable: {', '.join(heavy[:5])}{'...' if len(heavy) > 5 else ''}."
        )
    # Correlations
    if corr is not None:
        ac = corr.abs()
        pairs = []
        cols = ac.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], float(ac.iloc[i, j])))
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:3]
        if pairs:
            insights.append(
                "Top numeric correlations: "
                + "; ".join([f"{a}–{b}: {v:.2f}" for a, b, v in pairs])
                + "."
            )
    # Feature drivers
    if feat_importances_df is not None and not feat_importances_df.empty and target_col:
        insights.append(
            "Key drivers of {}: {}.".format(
                target_col, ", ".join(feat_importances_df.head(5)["Feature"].tolist())
            )
        )
    # Time momentum
    if monthly is not None and len(monthly) >= 6 and target_col:
        recent = monthly.tail(3)["MedianTarget"].median()
        prev = monthly.tail(6).head(3)["MedianTarget"].median()
        if not (pd.isna(recent) or pd.isna(prev)):
            pct = ((recent - prev) / (prev + 1e-9)) * 100.0
            direction = "up" if pct > 0 else "down"
            insights.append(
                f"Median {target_col} is {direction} ~{abs(pct):.1f}% over the last 3 months vs prior 3."
            )

    # --- Build HTML ---
    html = []
    html.append("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Real Estate EDA – Stakeholder Report</title>
<style>
body { font-family: Arial, sans-serif; margin: 24px; line-height: 1.5; }
h1, h2, h3 { margin-top: 24px; }
.section { margin-bottom: 28px; }
.kpi { display: inline-block; padding: 12px 16px; margin: 8px 8px 8px 0; border: 1px solid #ddd; border-radius: 8px; }
img { max-width: 100%; height: auto; border: 1px solid #eee; padding: 6px; border-radius: 6px; }
.caption { font-size: 0.9em; color: #444; margin-top: 4px; }
.callout { background: #f7f7f7; border-left: 4px solid #999; padding: 12px; margin: 12px 0; }
</style>
</head>
<body>
<h1>Real Estate EDA – Stakeholder Report</h1>
<p>This report summarizes patterns discovered in the dataset and provides data-driven recommendations to drive business impact.</p>
""")

    html.append(f"""
<div class="section">
  <h2>Snapshot</h2>
  <div class="kpi"><strong>Rows</strong><br>{len(df):,}</div>
  <div class="kpi"><strong>Columns</strong><br>{len(df.columns)}</div>
  <div class="kpi"><strong>Duplicates</strong><br>{int(df.duplicated().sum())}</div>
  <div class="kpi"><strong>Target</strong><br>{target_col if target_col else "Not detected"}</div>
  <div class="kpi"><strong>Date Field</strong><br>{main_dt if main_dt else "Not detected"}</div>
</div>
""")

    def _tag(key, caption):
        return (
            f'<div class="section"><img src="data:image/png;base64,{images[key]}"/><div class="caption">{caption}</div></div>'
            if key in images
            else ""
        )

    html.append('<div class="section"><h2>Key Visuals</h2>')
    html.append(
        _tag("missingness", "Top missing fields – prioritize data quality fixes.")
    )
    html.append(_tag("outliers", "Outlier-prone features – consider robust treatment."))
    html.append(_tag("correlation", "Correlation map – monitor multicollinearity."))
    html.append(_tag("feat_importance", "Top feature drivers from Random Forest."))
    html.append(_tag("volume", "Transaction volume by month."))
    html.append(_tag("median_target", f"Median {target_col or 'target'} by month."))
    # Geo chart skipped
    html.append("</div>")

    html.append('<div class="section"><h2>Insights That Matter</h2><ul>')
    for ins in insights:
        html.append(f"<li>{ins}</li>")
    html.append("</ul></div>")

    # Actions
    recs = []
    if target_col:
        recs.append(
            f"Use {target_col} drivers (top features) to refine pricing models and agent playbooks."
        )
    if monthly is not None and not monthly.empty:
        recs.append(
            "Time promotions with months showing rising prices and higher volume."
        )
    if corr is not None:
        recs.append(
            "Monitor multicollinearity in linear models; prefer trees or regularization if needed."
        )
    if (missing_df["Missing_%"] > 0).any():
        recs.append(
            "Fix high-missing fields at collection; add validation and mandatory capture for critical attributes."
        )
    if not recs:
        recs.append(
            "Standardize data collection and monitor leading indicators across time and location."
        )

    html.append('<div class="section"><h2>Recommended Actions</h2><ol>')
    for r in recs:
        html.append(f"<li>{r}</li>")
    html.append("</ol></div>")

    html.append("""
<div class="section">
  <h2>Notes</h2>
  <div class="callout">
    The analysis is exploratory. Validate with out-of-sample data and perform stability tests across time and sub-markets before production use.
  </div>
</div>
</body></html>
""")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html))

    print(f"Stakeholder report written to: {output_path}")


# --- Run it (change path if you want a different filename) ---

# Example: Load your data into a DataFrame named df before running the report.
# Replace the file path with your actual data file.
df = pd.read_csv("../data/real_estate_data.csv")

generate_stakeholder_report(df, output_path="real_estate_eda_stakeholder_report.html")
