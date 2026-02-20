# -*- coding: utf-8 -*-
"""
Classification_Model.py
Production-Ready CLI Pipeline for Census Segmentation & Classification with MLOps
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from IPython.display import display, HTML

# --- MLOps Integration ---
import dagshub
import mlflow
import mlflow.sklearn

# Initialize DagsHub & MLflow (This connects your script to the remote server)
dagshub.init(repo_owner='skarthiksubramanian0704', repo_name='Income-Classification', mlflow=True)
mlflow.set_experiment("Income_Classification")
# -------------------------

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==========================================================
# 1. ARGUMENT PARSER SETUP
# ==========================================================
parser = argparse.ArgumentParser(description="Census Classification Pipeline")
parser.add_argument("--pipeline", type=int, choices=[1, 2], default=1, 
                    help="1: Engineered Hybrid Model (32 feats) | 2: Baseline Model (40 feats)")
parser.add_argument("--algo", type=str, choices=["xgb", "rf", "logreg", "all"], default="all", 
                    help="Algorithm to train (xgb, rf, logreg, or all)")
parser.add_argument("--cv", action="store_true", 
                    help="Trigger Cross-Validation mode (skips standard test eval)")
parser.add_argument("--verbose", action="store_true", 
                    help="Print all preprocessing metadata and shapes")
parser.add_argument("--fast", action="store_true", 
                    help="Sample 10% of data for fast pipeline testing")
parser.add_argument("--save_plots", action="store_true", 
                    help="Silent mode: Save plots/metrics without displaying or printing them")

args, unknown = parser.parse_known_args()

VERBOSE = args.verbose
FAST_MODE = args.fast
SAVE_PLOTS = args.save_plots

# ==========================================================
# HEADLESS PLOTTING FIX & DIRECTORY SETUP
# ==========================================================
RESULTS_DIR = "results_classification"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

import matplotlib
if SAVE_PLOTS:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# 2. CONSOLE HEADERS
# ==========================================================
print("\n" + "="*71)
if args.pipeline == 1:
    print("[ RUNNING CUSTOM ENGINEERED HYBRID MODEL ]")
    print("="*71)
    print("Dataset:   Optimized Hybrid Pipeline")
else:
    print("[ RUNNING BASELINE CLASSIFICATION MODEL ]")
    print("="*71)
    print("Dataset:   Original Raw Data")
print("-"*71 + "\n")

# ==========================================================
# 3. DATA LOADING & CLEANING
# ==========================================================
if VERBOSE: print("[INFO] Loading dataset...")

PROJECT_DIR = "./"
if os.path.exists(PROJECT_DIR):
    os.chdir(PROJECT_DIR)

COLUMNS_FILE = "census-bureau.columns"
DATA_FILE = "census-bureau.data"

with open(COLUMNS_FILE, "r", encoding="utf-8") as f:
    cols = [line.strip() for line in f if line.strip()]

df = pd.read_csv(
    DATA_FILE, header=None, names=cols, sep=",",
    na_values=["?", "NA", "NAN", "NaN", "na", "nan", ""],
    skipinitialspace=True, engine="python"
)

# ==========================================================
# PREPROCESSING: FEATURE CLASSIFICATION
# ==========================================================
TARGET_COL = "label"
WEIGHT_COL = "weight"
feature_cols = [c for c in df.columns if c not in [TARGET_COL, WEIGHT_COL]]

numeric_cols, categorical_cols = [], []
for c in feature_cols:
    coverage = pd.to_numeric(df[c], errors="coerce").notna().mean()
    if coverage >= 0.95: numeric_cols.append(c)
    else: categorical_cols.append(c)

NUMERIC_FEATURES = ["age", "wage per hour", "capital gains", "capital losses", "dividends from stocks", "num persons worked for employer", "weeks worked in year", "year"]
CATEGORICAL_FEATURES = ["class of worker", "detailed industry recode", "detailed occupation recode", "education", "enroll in edu inst last wk", "marital stat", "major industry code", "major occupation code", "race", "hispanic origin", "sex", "member of a labor union", "reason for unemployment", "full or part time employment stat", "tax filer stat", "region of previous residence", "state of previous residence", "detailed household and family stat", "detailed household summary in household", "migration code-change in msa", "migration code-change in reg", "migration code-move within reg", "live in this house 1 year ago", "migration prev res in sunbelt", "family members under 18", "country of birth father", "country of birth mother", "country of birth self", "citizenship", "own business or self employed", "fill inc questionnaire for veteran's admin", "veterans benefits"]

# ==========================================================
# PREPROCESSING: MISSING VALUE IMPUTATION
# ==========================================================
categorical_missing_cols = df[CATEGORICAL_FEATURES].columns[df[CATEGORICAL_FEATURES].isna().any()].tolist()
cols_to_impute_unknown = [col for col in categorical_missing_cols if col not in ["hispanic origin", "country of birth self"]]
df[cols_to_impute_unknown] = df[cols_to_impute_unknown].fillna("NA")

ct = pd.crosstab(df["country of birth self"], df["hispanic origin"], dropna=True)
deterministic_map = {birthplace: row[row > 0].index[0] for birthplace, row in ct.iterrows() if len(row[row > 0]) == 1}
mask = (df["hispanic origin"].isna() & df["country of birth self"].isin(deterministic_map.keys()))
df.loc[mask, "hispanic origin"] = df.loc[mask, "country of birth self"].map(deterministic_map)
df["hispanic origin"] = df["hispanic origin"].fillna("Unknown")
df["country of birth self"] = df["country of birth self"].fillna("Unknown")

# ==========================================================
# PREPROCESSING: DATA CLEANSING & TARGET MAPPING
# ==========================================================
df["label"] = df["label"].str.strip()
df["income_binary"] = df["label"].map({"- 50000.": 0, "50000+.": 1})
df = df.drop_duplicates()

for col in CATEGORICAL_FEATURES:
    if df[col].dtype.name in ["object", "category"]: df[col] = df[col].str.strip()
    df[col] = df[col].astype("category")

if FAST_MODE:
    print("[WARNING] FAST MODE ENABLED: Running on a 10% random sample.\n")
    df = df.sample(frac=0.1, random_state=42)

# ==========================================================
# 4. FEATURE DEFINITIONS
# ==========================================================
if args.pipeline == 1:
    for col in ["wage per hour", "capital gains", "capital losses", "dividends from stocks"]:
        df[f"{col}_nonzero"] = (df[col] > 0).astype(int)
        df[f"{col}_log1p"] = np.log1p(df[col])
    
    df["work_intensity"] = df["weeks worked in year"] / 52.0
    df["investment_income"] = df["capital gains"] + df["dividends from stocks"]
    
    raw_gains = np.expm1(df["capital gains_log1p"])
    raw_dividends = np.expm1(df["dividends from stocks_log1p"])
    df["net_investment_income"] = raw_gains + raw_dividends - df["capital losses"]
    df["net_investment_income_log1p"] = np.sign(df["net_investment_income"]) * np.log1p(np.abs(df["net_investment_income"]))

    df["marital_tax_combo"] = (df["marital stat"].astype(str) + "__" + df["tax filer stat"].astype(str)).astype("category")
    df["veteran_affiliation"] = (df["veterans benefits"].astype(str) + "__" + df["fill inc questionnaire for veteran's admin"].astype(str)).astype("category")
    mapped_business = df["own business or self employed"].map({0: "Not_in_universe", 1: "Yes", 2: "No"}).astype(str)
    df["class_business_combo"] = (df["class of worker"].astype(str) + "__" + mapped_business).astype("category")

    num_cols = ['age', 'wage per hour', 'num persons worked for employer', 'weeks worked in year',  "capital gains", "capital losses", "dividends from stocks"]
    cat_cols = ['detailed occupation recode', 'education', 'marital_tax_combo', 'major industry code', 'race', 'hispanic origin', 'sex', 'member of a labor union', 'reason for unemployment', 'full or part time employment stat', 'region of previous residence', 'state of previous residence', 'detailed household summary in household', 'migration code-change in msa', 'migration code-change in reg', 'migration code-move within reg', 'live in this house 1 year ago', 'migration prev res in sunbelt', 'family members under 18', 'country of birth father', 'country of birth mother', 'country of birth self', 'citizenship', 'class_business_combo', 'veteran_affiliation']

else:
    num_cols = ["age", "wage per hour", "capital gains", "capital losses", "dividends from stocks", "num persons worked for employer", "weeks worked in year", "year"]
    cat_cols = CATEGORICAL_FEATURES.copy()

# ==========================================================
# 5. PREPROCESSING & SPLITTING
# ==========================================================
for col in cat_cols: df[col] = df[col].astype("category")
X, y, w = df[num_cols + cat_cols].copy(), df["income_binary"].astype(int), df["weight"].astype(float)
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.2, stratify=y, random_state=42)

# ==========================================================
# 6. ALGORITHM DEFINITIONS
# ==========================================================
xgb_scale_weight = np.sum(w_train[y_train == 0]) / np.sum(w_train[y_train == 1])
models_to_run = {}

if args.algo in ["logreg", "all"]:
    models_to_run["Logistic Regression"] = {
        "Unbalanced": LogisticRegression(C=0.1, max_iter=3000, random_state=42),
        "Balanced": LogisticRegression(C=10.0, class_weight="balanced", max_iter=3000, random_state=42)
    }
if args.algo in ["rf", "all"]:
    models_to_run["Random Forest"] = {
        "Unbalanced": RandomForestClassifier(min_samples_leaf=2, n_estimators=500, n_jobs=-1, random_state=42),
        "Balanced": RandomForestClassifier(min_samples_leaf=2, n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=42)
    }
if args.algo in ["xgb", "all"]:
    models_to_run["XGBoost"] = {
        "Unbalanced": XGBClassifier(colsample_bytree=0.8, learning_rate=0.05, max_depth=6, n_estimators=500, subsample=0.8, scale_pos_weight=1, tree_method="hist", eval_metric="auc", n_jobs=-1, random_state=42),
        "Balanced": XGBClassifier(colsample_bytree=1.0, learning_rate=0.05, max_depth=4, n_estimators=500, subsample=0.8, scale_pos_weight=xgb_scale_weight, tree_method="hist", eval_metric="auc", n_jobs=-1, random_state=42)
    }

# ==========================================================
# 7. EXECUTION ENGINE (WITH MLFLOW TRACKING)
# ==========================================================
all_results = []
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for algo_name, modes in models_to_run.items():
    print(f"\n[{algo_name.upper()}]")
    print("="*50)

    card_thresh = 45 if algo_name == "XGBoost" else 40
    HIGH_CARD_COLS = [c for c in cat_cols if X_train[c].nunique() > card_thresh]
    LOW_CARD_COLS = [c for c in cat_cols if c not in HIGH_CARD_COLS]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat_low", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOW_CARD_COLS),
        ("cat_high", TargetEncoder(target_type='binary', smooth="auto"), HIGH_CARD_COLS)
    ])
    
    for mode, model in modes.items():
        print(f" -> Training {mode}...")
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        cv_result_str = "N/A"
        
        # --- MLOps: START RUN ---
        run_name = f"{algo_name.replace(' ', '_')}_{mode}_Pipeline_{args.pipeline}"
        with mlflow.start_run(run_name=run_name):
            
            # Log Parameters
            mlflow.log_params({
                "Algorithm": algo_name, "Mode": mode, "Pipeline": args.pipeline, "Fast_Mode": args.fast
            })

            if args.cv:
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_strategy, scoring='roc_auc', params={'model__sample_weight': w_train}, n_jobs=1)
                cv_result_str = f"{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
                mlflow.log_metric("CV_ROC_AUC_Mean", cv_scores.mean())
                if not SAVE_PLOTS: print(f"    CV ROC-AUC: {cv_result_str}")
            
            # Train and Predict
            pipe.fit(X_train, y_train, model__sample_weight=w_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            y_pred = pipe.predict(X_test)

            # Calculate Metrics
            acc = accuracy_score(y_test, y_pred, sample_weight=w_test)
            prec = precision_score(y_test, y_pred, sample_weight=w_test)
            rec = recall_score(y_test, y_pred, sample_weight=w_test)
            f1 = f1_score(y_test, y_pred, sample_weight=w_test)
            auc_score = roc_auc_score(y_test, y_prob, sample_weight=w_test)

            # Log Metrics
            mlflow.log_metrics({"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "Test_ROC_AUC": auc_score})

            # Save the fully trained pipeline directly to DagsHub
            mlflow.sklearn.log_model(pipe, artifact_path="model")

            if not SAVE_PLOTS:
                print(f"  Mode: {mode} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | Test ROC-AUC: {auc_score:.4f}")
            
            result_dict = {"Algorithm": algo_name, "Mode": mode, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC-AUC": auc_score}
            if cv_result_str != "N/A": result_dict["CV ROC-AUC"] = cv_result_str
            all_results.append(result_dict)

            # Plot Dashboards
            if mode == "Balanced" and algo_name in ["Random Forest", "XGBoost"]:
                y_prob_0, y_test_0 = 1.0 - y_prob, (y_test == 0).astype(int)
                plt.figure(figsize=(16, 12))
                
                # PR Class 1
                plt.subplot(2, 2, 1)
                precision_1, recall_1, _ = precision_recall_curve(y_test, y_prob, sample_weight=w_test)
                plt.plot(recall_1, precision_1, color='purple', lw=2.5, label=f'>$50k (AP={average_precision_score(y_test, y_prob, sample_weight=w_test):.4f})')
                plt.title(f'{algo_name} PR Curve (>$50k Class)', fontweight='bold'); plt.legend()

                # PR Class 0
                plt.subplot(2, 2, 2)
                precision_0, recall_0, _ = precision_recall_curve(y_test_0, y_prob_0, sample_weight=w_test)
                plt.plot(recall_0, precision_0, color='crimson', lw=2.5, label=f'<$50k (AP={average_precision_score(y_test_0, y_prob_0, sample_weight=w_test):.4f})')
                plt.title(f'{algo_name} PR Curve (<$50k Class)', fontweight='bold'); plt.legend()

                # Gains
                plt.subplot(2, 1, 2)
                gains_df = pd.DataFrame({'actual': y_test, 'prob': y_prob, 'weight': w_test}).sort_values(by='prob', ascending=False)
                plt.plot((gains_df['weight'].cumsum() / gains_df['weight'].sum()) * 100, ((gains_df['actual'] * gains_df['weight']).cumsum() / (gains_df['actual'] * gains_df['weight']).sum()) * 100, color='green', lw=2.5, label='Model Gains')
                plt.title(f'{algo_name} Cumulative Gains', fontweight='bold'); plt.legend()
                
                plt.tight_layout()
                plot_path = f"{RESULTS_DIR}/{algo_name.replace(' ', '_')}_Dashboard.png"
                plt.savefig(plot_path)
                
                # Log the visual artifact to DagsHub
                mlflow.log_artifact(plot_path, artifact_path="dashboards")
                if not SAVE_PLOTS: plt.show()

# ==========================================================
# 8. FINAL METRICS EXPORT
# ==========================================================
if all_results:
    results_df = pd.DataFrame(all_results).round(4)
    results_df.to_csv(f"{RESULTS_DIR}/classification_metrics.txt", index=False, sep="\t")
    if not SAVE_PLOTS:
        print("\n" + "="*58 + "\nFINAL PERFORMANCE METRICS\n" + "="*58)
        print(results_df.to_string(index=False))