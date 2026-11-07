"""
Follow-up analysis: which feature family helps most?

The original project trained every model on all 51 features together.
This script trains SVM and Random Forest again, but with only one family
of features at a time, and compares the results:

    All        -> the 51 features (same as the original project)
    Frequency  -> total power + band power                    (21 features)
    Temporal   -> kurtosis, skewness, Katz, AR, perm. entropy (18 features)
    Spatial    -> cross-correlation, coherence (MSC), PLV     (12 features)

Everything else is the same as in the original pipeline: same 80/20 split,
same random seed, same 5-fold cross-validation and same hyperparameter grids.
So the "All" results should match the original SVM and Random Forest results.

Usage:
    python follow_up/feature_family_ablation.py FEATURES_CSV [RANKING_CSV]

    FEATURES_CSV  the feature table created by bci_motor_imagery_EN.py or
                  bci_motor_imagery.py (data/features_extraction/Cho2017/...)
    RANKING_CSV   optional. The BCI ranking file (models/bci_ranking.csv or
                  models/ranking_bci.csv). If you give it, the summary is also
                  split into eligible subjects and hard subjects.

Outputs (in follow_up/results/):
    ablation_results.csv   one row per subject, model and feature family
    ablation_summary.csv   mean test accuracy per model and feature family
    ablation_boxplot.png   test accuracy per feature family
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# =============================================================================
# Settings (the same as in the original pipeline)
# =============================================================================

RANDOM_SEED = 45
TEST_SIZE = 0.20
K_FOLDS = 5

# Each feature family is defined by the start of its column names
FEATURE_FAMILIES = {
    "All": ["TotalPow_", "BandPow_", "Kurtosis_", "Skewness_", "Katz_", "AR1_", "AR2_",
            "PermEnt_", "CrossCorr_", "MSC_", "PLV_"],
    "Frequency": ["TotalPow_", "BandPow_"],
    "Temporal": ["Kurtosis_", "Skewness_", "Katz_", "AR1_", "AR2_", "PermEnt_"],
    "Spatial": ["CrossCorr_", "MSC_", "PLV_"],
}

# Models and hyperparameter grids. To save time, you can remove one of them.
MODELS = {
    "SVM": (SVC(random_state=RANDOM_SEED),
            {"clf__C": [0.1, 1.0, 10.0], "clf__kernel": ["linear", "rbf"]}),
    "RandomForest": (RandomForestClassifier(random_state=RANDOM_SEED),
                     {"clf__n_estimators": [100, 300], "clf__max_depth": [5, 10, None]}),
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")


# =============================================================================
# Helper functions
# =============================================================================

def get_family_columns(all_columns, prefixes):
    """Returns the columns whose name starts with one of the prefixes."""
    selected = []
    for column in all_columns:
        for prefix in prefixes:
            if column.startswith(prefix):
                selected.append(column)
                break
    return selected


def train_one(model, grid, X, y):
    """80/20 split, grid search with 5-fold CV on the 80%, test on the 20%.
    The split only depends on y and the seed, so every feature family is
    tested on exactly the same trials."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    # The scaler is inside the Pipeline, so it only learns from training folds
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(pipe, param_grid=grid, cv=cv, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, search.best_estimator_.predict(X_test))
    return search.best_score_, test_accuracy


def read_eligible_subjects(ranking_path):
    """Reads the BCI ranking (English or Spanish version) and returns
    a dictionary: subject -> True (eligible) or False (hard subject)."""
    ranking = pd.read_csv(ranking_path)
    if "Subject_ID" in ranking.columns:
        subject_column, eligible_column = "Subject_ID", "BCI_Eligible"
    else:
        subject_column, eligible_column = "Sujeto_ID", "Apto_BCI"
    eligible = {}
    for i in range(len(ranking)):
        subject = int(ranking[subject_column].iloc[i])
        eligible[subject] = bool(ranking[eligible_column].iloc[i])
    return eligible


def mean_table(df):
    """Mean test accuracy: one row per model, one column per feature family."""
    table = df.pivot_table(index="Model", columns="Features", values="Test_Accuracy", aggfunc="mean")
    return table[list(FEATURE_FAMILIES.keys())]


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python follow_up/feature_family_ablation.py FEATURES_CSV [RANKING_CSV]")
        sys.exit(1)
    features_path = sys.argv[1]
    ranking_path = None
    if len(sys.argv) > 2:
        ranking_path = sys.argv[2]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(features_path)
    # The English script calls the subject column "Subject_ID", the Spanish one "Sujeto_ID"
    if "Subject_ID" in df.columns:
        subject_column = "Subject_ID"
    else:
        subject_column = "Sujeto_ID"

    # Check the feature families and print how many columns each one has
    family_columns = {}
    for family in FEATURE_FAMILIES:
        columns = get_family_columns(df.columns, FEATURE_FAMILIES[family])
        family_columns[family] = columns
        print(family, "->", len(columns), "features")
    if len(family_columns["All"]) != 51:
        print("Warning: expected 51 features in 'All', found", len(family_columns["All"]))

    subjects = df[subject_column].unique()
    print("Subjects:", len(subjects))

    rows = []
    for s in range(len(subjects)):
        subject = subjects[s]
        df_subject = df[df[subject_column] == subject]
        y = df_subject["Target"].values
        if np.min(np.bincount(y)) < K_FOLDS:
            print("Subject", subject, ": too few trials. Skipping.")
            continue

        for model_name in MODELS:
            model, grid = MODELS[model_name]
            for family in FEATURE_FAMILIES:
                X = df_subject[family_columns[family]].values
                cv_accuracy, test_accuracy = train_one(model, grid, X, y)
                rows.append({"Subject_ID": int(subject), "Model": model_name, "Features": family,
                             "CV_Accuracy": cv_accuracy, "Test_Accuracy": test_accuracy})

        print("Subject", subject, "done (", s + 1, "of", len(subjects), ")")
        # Save after every subject, so nothing is lost if the run stops
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"), index=False)

    results = pd.DataFrame(rows)

    # --- Summary tables --------------------------------------------------------
    summaries = []
    table = mean_table(results)
    print("\nMean test accuracy, all subjects:")
    print(table.round(3))
    summaries.append(table.assign(Group="All subjects"))

    if ranking_path is not None:
        eligible = read_eligible_subjects(ranking_path)
        group = []
        for i in range(len(results)):
            if eligible.get(results["Subject_ID"].iloc[i], False):
                group.append("Eligible")
            else:
                group.append("Hard (near chance)")
        results["Group"] = group
        results.to_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"), index=False)

        for group_name in ["Eligible", "Hard (near chance)"]:
            subset = results[results["Group"] == group_name]
            if len(subset) == 0:
                continue
            table = mean_table(subset)
            n_subjects = subset["Subject_ID"].nunique()
            print("\nMean test accuracy,", group_name, "(", n_subjects, "subjects ):")
            print(table.round(3))
            summaries.append(table.assign(Group=group_name))

    summary = pd.concat(summaries).reset_index()
    summary.to_csv(os.path.join(OUTPUT_DIR, "ablation_summary.csv"), index=False)

    # --- Plot ------------------------------------------------------------------
    plot_groups = ["All subjects"]
    if "Group" in results.columns:
        plot_groups.append("Hard (near chance)")
    fig, axes = plt.subplots(1, len(plot_groups), figsize=(7 * len(plot_groups), 6), sharey=True)
    if len(plot_groups) == 1:
        axes = [axes]
    for i in range(len(plot_groups)):
        if plot_groups[i] == "All subjects":
            data = results
        else:
            data = results[results["Group"] == plot_groups[i]]
        sns.boxplot(data=data, x="Features", y="Test_Accuracy", hue="Model",
                    order=list(FEATURE_FAMILIES.keys()), palette="Set2", ax=axes[i])
        axes[i].axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance level (0.5)")
        axes[i].set_title(plot_groups[i] + " (" + str(data["Subject_ID"].nunique()) + " subjects)",
                          fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Test accuracy")
        axes[i].legend(loc="lower right")
    fig.suptitle("Test accuracy by feature family", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\nResults saved in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
