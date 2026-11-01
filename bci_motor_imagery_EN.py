"""
EEG motor imagery classification (left hand vs. right hand) on the Cho2017 dataset.

English version of bci_motor_imagery.py. The logic is the same; variable names,
comments, output files and plot labels are in English.

Pipeline:
    1. Feature extraction: 51 features per trial from channels C3, Cz and C4.
    2. Exploratory data analysis (EDA): boxplots per subject, channel and class.
    3. Within-subject training: Decision Tree, Random Forest and SVM on the
       features, and CSP+LDA on the raw EEG signal (baseline).
    4. BCI illiteracy filter: binomial chance threshold on the CSP+LDA test
       accuracy, and comparison of all subjects vs. eligible subjects.

Usage:
    BCI_REPO_PATH=/path/to/BCI-MENTORIA-UNC python bci_motor_imagery_EN.py

Outputs (written inside BCI_REPO_PATH):
    data/features_extraction/Cho2017/   feature dataset (CSV)
    results/figures/                    all plots
    models/                             per-subject models and metrics (all subjects)
    models_eligible/                    the same, for the eligible subjects only
"""

import os
import sys
import json
import math
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch, coherence, hilbert
from scipy.stats import skew, kurtosis, binom
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import mne
from mne.decoding import CSP

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
sns.set_theme(style="whitegrid")

# The PreprocessedDataset class (EEG loading and preprocessing) lives in the
# mentorship repository: https://github.com/bzorzet98/BCI-MENTORIA-UNC
REPO_PATH = os.environ.get("BCI_REPO_PATH", ".")
os.chdir(REPO_PATH)
sys.path.append(REPO_PATH)

from src.eeg_datasets.preprocessed_dataset import PreprocessedDataset


# =============================================================================
# Configuration
# =============================================================================

DATASET_NAME = "Cho2017"
SUBJECTS = None                      # None = all available subjects
SESSIONS = ["session_1"]
DATA_TO_LOAD = ["motor_imagery"]
SFREQ = 128                          # sampling frequency (Hz)

FEATURE_CHANNELS = ["C3", "Cz", "C4"]          # channels used for the 51 features
CSP_CHANNELS = ["C3", "C1", "Cz", "C2", "C4"]  # channels used by CSP+LDA
ELECTRODE_PAIRS = ["C3-Cz", "C4-Cz", "C3-C4"]  # pairs for the connectivity features
PAIR_INDICES = [(0, 1), (2, 1), (0, 2)]        # the same pairs, as channel indices
PSD_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (28, 30)]  # Hz
MSC_BANDS = [(8, 12), (12, 30)]                                        # Hz (mu, beta)

RANDOM_SEED = 45
TEST_SIZE = 0.20                     # 80/20 train/test split
K_FOLDS = 5                          # folds of the cross-validation
SCORING = "accuracy"
ALPHA = 0.05                         # significance level of the chance threshold

META_COLUMNS = ["Subject_ID", "Target", "Class_Name"]
CLASS_NAMES = {0: "Left hand", 1: "Right hand"}
FIGURES_DIR = os.path.join("results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def binarize_y(y):
    """Turns the two class labels into 0 and 1 (0 = left hand, 1 = right hand)."""
    classes = np.unique(y)
    assert len(classes) == 2, "y must have exactly two classes"
    class_to_bin = {classes[0]: 0, classes[1]: 1}
    bin_to_class = {v: k for k, v in class_to_bin.items()}
    return np.vectorize(class_to_bin.get)(y), bin_to_class


# =============================================================================
# 1. Features
# Every function receives X with shape (trials, channels, time samples)
# and returns one row per trial.
# =============================================================================

def feature_total_power(X):
    """Total power: mean of the squared signal, per channel."""
    return np.mean(np.square(X), axis=-1)


def feature_band_power(X, sfreq=SFREQ, fbands=PSD_BANDS):
    """Mean power in each frequency band, from the Welch power spectral density."""
    n_trials, n_channels, _ = X.shape
    n_bands = len(fbands)
    freqs, psd = welch(X, fs=sfreq, axis=-1, nperseg=int(sfreq))
    out = np.zeros((n_trials, n_channels * n_bands))
    for b, (f_min, f_max) in enumerate(fbands):
        idx = (freqs >= f_min) & (freqs <= f_max)
        power = np.mean(psd[:, :, idx], axis=-1)
        for c in range(n_channels):
            out[:, c * n_bands + b] = power[:, c]
    return out


def feature_kurtosis(X):
    """Kurtosis: how heavy the tails of the amplitude distribution are."""
    return kurtosis(X, axis=-1)


def feature_skewness(X):
    """Skewness: asymmetry of the amplitude distribution."""
    return skew(X, axis=-1)


def feature_katz_fractal_dim(X):
    """Katz fractal dimension: a measure of the signal's complexity."""
    n_trials, n_channels, n_times = X.shape
    out = np.zeros((n_trials, n_channels))
    N = n_times - 1
    for t in range(n_trials):
        for c in range(n_channels):
            signal = X[t, c, :]
            L = np.sum(np.abs(np.diff(signal)))        # total length of the curve
            d = np.max(np.abs(signal - signal[0]))     # largest distance from the first point
            out[t, c] = np.log10(N) / (np.log10(d / L) + np.log10(N))
    return out


def feature_autoregressive_coefs(X):
    """Order-2 autoregressive coefficients (Yule-Walker equations)."""
    n_trials, n_channels, n_times = X.shape
    out = np.zeros((n_trials, n_channels * 2))
    for t in range(n_trials):
        for c in range(n_channels):
            signal = X[t, c, :]
            # Autocorrelation at lags 0, 1 and 2, normalised by lag 0
            r = np.correlate(signal, signal, mode="full")[n_times - 1:n_times + 2]
            r = r / r[0]
            R = np.array([[r[0], r[1]], [r[1], r[0]]])
            b = np.array([r[1], r[2]])
            try:
                coefs = np.linalg.solve(R, b)
            except np.linalg.LinAlgError:
                coefs = np.zeros(2)
            out[t, c * 2:c * 2 + 2] = coefs
    return out


def feature_permutation_entropy(X, order=3, delay=1):
    """Normalised permutation entropy: how unpredictable the order of the samples is."""
    n_trials, n_channels, n_times = X.shape
    out = np.zeros((n_trials, n_channels))
    n_words = n_times - (order - 1) * delay
    for t in range(n_trials):
        for c in range(n_channels):
            x = X[t, c, :]
            # Each "word" is a group of `order` consecutive samples
            words = np.stack([x[i * delay:i * delay + n_words] for i in range(order)], axis=1)
            # Count how often each ordering pattern appears
            _, counts = np.unique(np.argsort(words, axis=1), axis=0, return_counts=True)
            probs = counts / counts.sum()
            out[t, c] = -np.sum(probs * np.log2(probs)) / np.log2(math.factorial(order))
    return out


def feature_cross_correlation(X):
    """Pearson correlation between the two channels of each electrode pair."""
    out = np.zeros((X.shape[0], len(PAIR_INDICES)))
    for t in range(X.shape[0]):
        for p, (a, b) in enumerate(PAIR_INDICES):
            corr = np.corrcoef(X[t, a, :], X[t, b, :])[0, 1]
            out[t, p] = 0.0 if np.isnan(corr) else corr
    return out


def feature_magnitude_squared_coherence(X, sfreq=SFREQ, fbands=MSC_BANDS):
    """Magnitude-squared coherence (MSC) of each pair, averaged in each band."""
    out = np.zeros((X.shape[0], len(PAIR_INDICES) * len(fbands)))
    for t in range(X.shape[0]):
        col = 0
        for a, b in PAIR_INDICES:
            f, msc = coherence(X[t, a, :], X[t, b, :], fs=sfreq, nperseg=int(sfreq))
            for f_min, f_max in fbands:
                idx = (f >= f_min) & (f <= f_max)
                out[t, col] = np.mean(msc[idx]) if np.any(idx) else 0.0
                col += 1
    return out


def feature_phase_locking_value(X):
    """Phase Locking Value (PLV): how stable the phase difference of each pair is."""
    out = np.zeros((X.shape[0], len(PAIR_INDICES)))
    phases = np.angle(hilbert(X, axis=-1))
    for p, (a, b) in enumerate(PAIR_INDICES):
        out[:, p] = np.abs(np.mean(np.exp(1j * (phases[:, a, :] - phases[:, b, :])), axis=-1))
    return out


# Features included in the dataset. Set "active" to False to leave one out.
PIPELINE_CONFIG = {
    "total_power": {"active": True, "method": feature_total_power, "params": {}},
    "band_power": {"active": True, "method": feature_band_power, "params": {"sfreq": SFREQ}},
    "kurtosis": {"active": True, "method": feature_kurtosis, "params": {}},
    "skewness": {"active": True, "method": feature_skewness, "params": {}},
    "katz_fractal": {"active": True, "method": feature_katz_fractal_dim, "params": {}},
    "ar_coefficients": {"active": True, "method": feature_autoregressive_coefs, "params": {}},
    "permutation_entropy": {"active": True, "method": feature_permutation_entropy, "params": {"order": 3}},
    "cross_correlation": {"active": True, "method": feature_cross_correlation, "params": {}},
    "msc_coherence": {"active": True, "method": feature_magnitude_squared_coherence, "params": {"sfreq": SFREQ}},
    "phase_plv": {"active": True, "method": feature_phase_locking_value, "params": {}},
}


def band_str(band):
    return f"{band[0]}-{band[1]}Hz"


def build_column_names(methods):
    """Column names of the dataset, in the same order the features are computed."""
    names = []
    for method in methods:
        if method == "total_power":
            names += [f"TotalPow_{c}" for c in FEATURE_CHANNELS]
        elif method == "band_power":
            names += [f"BandPow_{c}_{band_str(b)}" for c in FEATURE_CHANNELS for b in PSD_BANDS]
        elif method == "kurtosis":
            names += [f"Kurtosis_{c}" for c in FEATURE_CHANNELS]
        elif method == "skewness":
            names += [f"Skewness_{c}" for c in FEATURE_CHANNELS]
        elif method == "katz_fractal":
            names += [f"Katz_{c}" for c in FEATURE_CHANNELS]
        elif method == "ar_coefficients":
            names += [f"AR{i}_{c}" for c in FEATURE_CHANNELS for i in (1, 2)]
        elif method == "permutation_entropy":
            names += [f"PermEnt_{c}" for c in FEATURE_CHANNELS]
        elif method == "cross_correlation":
            names += [f"CrossCorr_{p}" for p in ELECTRODE_PAIRS]
        elif method == "msc_coherence":
            names += [f"MSC_{p}_{band_str(b)}" for p in ELECTRODE_PAIRS for b in MSC_BANDS]
        elif method == "phase_plv":
            names += [f"PLV_{p}" for p in ELECTRODE_PAIRS]
        else:
            raise ValueError(f"No naming rule for method: {method}")
    return names


def load_dataset(channels):
    return PreprocessedDataset(
        db_name=DATASET_NAME,
        sessions=SESSIONS,
        subjects=SUBJECTS,
        data_to_load=DATA_TO_LOAD,
        channels=channels,
    )


def extract_features(dataset):
    """Builds the feature table: one row per trial, all subjects together."""
    methods = [n for n, cfg in PIPELINE_CONFIG.items() if cfg["active"]]
    if not methods:
        raise ValueError("Activate at least one feature in PIPELINE_CONFIG.")
    columns = build_column_names(methods)
    blocks = []
    for subject in dataset.subject_list:
        try:
            X_eeg, y, _ = dataset.flatten_pool_data(subject_ids=[subject])
            y, _ = binarize_y(y)
        except Exception as e:
            print(f"[{subject}] Loading error: {e}. Skipping.")
            continue
        feats = []
        for n in methods:
            res = PIPELINE_CONFIG[n]["method"](X_eeg, **PIPELINE_CONFIG[n]["params"])
            feats.append(res[:, np.newaxis] if res.ndim == 1 else res)
        df_s = pd.DataFrame(np.hstack(feats), columns=columns)
        df_s.insert(0, "Class_Name", pd.Series(y).map(CLASS_NAMES))
        df_s.insert(0, "Target", y)
        df_s.insert(0, "Subject_ID", subject)
        blocks.append(df_s)
        print(f"[{subject}] {df_s.shape[0]} trials x {len(columns)} features")
    df = pd.concat(blocks, ignore_index=True)
    export_dir = os.path.join("data", "features_extraction", DATASET_NAME)
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, f"dataset_features_{'_'.join(methods)}.csv")
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}: {df.shape} ({df['Subject_ID'].nunique()} subjects)")
    return df


# =============================================================================
# 2. Exploratory data analysis
# =============================================================================

def save_figure(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature(df, prefix, band_filter=None):
    """Boxplot + stripplot of one feature, per subject, channel and class."""
    cols = [c for c in df.columns if c.startswith(prefix) and (band_filter is None or c.endswith(band_filter))]
    if not cols:
        raise ValueError(f"No columns for '{prefix}'")
    df_long = pd.melt(df[META_COLUMNS + cols], id_vars=META_COLUMNS, value_vars=cols,
                      var_name="Feature_Channel", value_name="Value")
    # The channel is the 2nd part of "BandPow_C3_8-12Hz" and the last part of "TotalPow_C3"
    pos = 1 if prefix == "BandPow" else -1
    df_long["Channel"] = df_long["Feature_Channel"].str.split("_").str[pos]
    df_long["Channel_Condition"] = df_long["Channel"] + " - " + df_long["Class_Name"]
    plt.figure(figsize=(16, 7))
    sns.boxplot(data=df_long, x="Subject_ID", y="Value", hue="Channel_Condition",
                palette="Set2", fliersize=0, boxprops=dict(alpha=0.6))
    sns.stripplot(data=df_long, x="Subject_ID", y="Value", hue="Channel_Condition",
                  palette="Set2", dodge=True, alpha=0.4, size=3, jitter=0.2, legend=False)
    suffix = f" ({band_filter})" if band_filter else ""
    plt.title(f"Distribution of {prefix}{suffix} by channel and condition", fontsize=14, fontweight="bold")
    plt.xlabel("Subject")
    plt.ylabel("Value")
    plt.xticks(rotation=90)
    plt.legend(title="Channel & Task", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(f"eda_{prefix}{'_' + band_filter if band_filter else ''}")


# =============================================================================
# 3. Within-subject training
# =============================================================================

def fit_and_save(pipe, grid, X, y, subject, model_name, base_dir):
    """80/20 split, grid search with 5-fold CV on the 80%, final test on the 20%."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(pipe, param_grid=grid, cv=cv, scoring=SCORING, n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    test_acc = accuracy_score(y_test, best.predict(X_test))
    print(f"[{model_name}] Subject {subject}: CV={gs.best_score_:.3f} Test={test_acc:.3f} {gs.best_params_}")
    subject_dir = os.path.join(base_dir, model_name, f"subject_{subject}")
    os.makedirs(subject_dir, exist_ok=True)
    with open(os.path.join(subject_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(best, f)
    with open(os.path.join(subject_dir, "metrics_hyperparameters.json"), "w") as f:
        json.dump({"best_hyperparameters": gs.best_params_,
                   "cv_accuracy": float(gs.best_score_),
                   "test_metrics": {"accuracy": float(test_acc)}}, f, indent=4)
    return {"Subject_ID": subject, "Model": model_name, "Best_Params": str(gs.best_params_),
            "CV_Accuracy": gs.best_score_, "Test_Accuracy": test_acc}


# Models trained on the features, with their hyperparameter grids
FEATURE_MODELS = {
    "DecisionTree": (DecisionTreeClassifier(random_state=RANDOM_SEED),
                     {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 5, 10]}),
    "RandomForest": (RandomForestClassifier(random_state=RANDOM_SEED),
                     {"clf__n_estimators": [100, 300], "clf__max_depth": [5, 10, None]}),
    "SVM": (SVC(random_state=RANDOM_SEED),
            {"clf__C": [0.1, 1.0, 10.0], "clf__kernel": ["linear", "rbf"]}),
}

# Grid of the CSP+LDA baseline (fitted on the raw EEG signal, not on the features)
CSP_GRID = {"csp__n_components": [2, 4], "csp__component_order": ["mutual_info", "alternate"]}


def train_feature_model(df, subjects, name, base_dir):
    """Trains one feature-based model for each subject. The scaler sits inside the
    Pipeline, so it is fitted on the training folds only (no data leakage)."""
    estimator, grid = FEATURE_MODELS[name]
    X_columns = [c for c in df.columns if c not in META_COLUMNS]
    results = []
    for subject in subjects:
        df_s = df[df["Subject_ID"] == subject]
        if df_s.empty:
            continue
        X, y = df_s[X_columns].values, df_s["Target"].values
        if np.min(np.bincount(y)) < K_FOLDS:
            print(f"[{name}] Subject {subject}: too few trials. Skipping.")
            continue
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        results.append(fit_and_save(pipe, grid, X, y, subject, name, base_dir))
    df_res = pd.DataFrame(results)
    os.makedirs(os.path.join(base_dir, name), exist_ok=True)
    df_res.to_csv(os.path.join(base_dir, name, f"df_global_{name}.csv"), index=False)
    return df_res


def train_csp(dataset, subjects, base_dir):
    """Trains the CSP+LDA baseline for each subject, on the raw EEG signal."""
    results = []
    for subject in subjects:
        try:
            X, y, _ = dataset.flatten_pool_data(subject_ids=[subject])
            y, _ = binarize_y(y)
            if np.min(np.bincount(y)) < K_FOLDS:
                print(f"[CSP] Subject {subject}: too few trials. Skipping.")
                continue
            pipe = Pipeline([("csp", CSP()), ("lda", LinearDiscriminantAnalysis())])
            results.append(fit_and_save(pipe, CSP_GRID, X, y, subject, "CSP", base_dir))
        except Exception as e:
            print(f"[CSP] Subject {subject} failed: {e}")
    df_res = pd.DataFrame(results)
    os.makedirs(os.path.join(base_dir, "CSP"), exist_ok=True)
    df_res.to_csv(os.path.join(base_dir, "CSP", "df_global_CSP.csv"), index=False)
    return df_res


def train_all(df, csp_dataset, subjects, base_dir):
    """Trains the 4 models for the given subjects and returns one results table."""
    results = [train_feature_model(df, subjects, n, base_dir) for n in FEATURE_MODELS]
    results.append(train_csp(csp_dataset, subjects, base_dir))
    return pd.concat(results, ignore_index=True)


def plot_tree_and_forest(df, base_dir):
    """For the first subject: the decision tree, one tree of the forest and the
    15 most important features of the Random Forest."""
    subject = df["Subject_ID"].unique()[0]
    X_columns = [c for c in df.columns if c not in META_COLUMNS]
    for name in ["DecisionTree", "RandomForest"]:
        with open(os.path.join(base_dir, name, f"subject_{subject}", "best_model.pkl"), "rb") as f:
            clf = pickle.load(f).named_steps["clf"]
        tree = clf if name == "DecisionTree" else clf.estimators_[0]
        plt.figure(figsize=(20, 10))
        plot_tree(tree, feature_names=X_columns, class_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
                  filled=True, rounded=True, fontsize=8, max_depth=3)
        plt.title(f"{name} — Subject {subject}", fontsize=14, fontweight="bold")
        save_figure(f"tree_{name}_subject_{subject}")
        if name == "RandomForest":
            df_imp = pd.DataFrame({"Feature": X_columns, "Importance": clf.feature_importances_})
            df_imp = df_imp.sort_values("Importance", ascending=False).head(15)
            plt.figure(figsize=(10, 7))
            sns.barplot(data=df_imp, x="Importance", y="Feature", hue="Feature", palette="Set2", legend=False)
            plt.title(f"Top 15 features — Random Forest — Subject {subject}", fontweight="bold")
            save_figure(f"rf_importance_subject_{subject}")


# =============================================================================
# 4. BCI illiteracy filter
# =============================================================================

def chance_threshold(n_test, p=0.5, alpha=ALPHA):
    """Smallest accuracy that a random classifier (p = 0.5) reaches with
    probability <= alpha on n_test trials (binomial distribution)."""
    for k in range(n_test + 1):
        if binom.sf(k - 1, n_test, p) <= alpha:
            return k / n_test
    return 1.0


def main():
    # --- 1. Features ---------------------------------------------------------
    feature_dataset = load_dataset(FEATURE_CHANNELS)
    df = extract_features(feature_dataset)

    # --- 2. EDA --------------------------------------------------------------
    plot_feature(df, "TotalPow")
    plot_feature(df, "BandPow", band_filter="8-12Hz")

    # --- 3. Training on all subjects -----------------------------------------
    subjects = df["Subject_ID"].unique().tolist()
    csp_dataset = load_dataset(CSP_CHANNELS)
    df_comparison = train_all(df, csp_dataset, subjects, "models")
    df_comparison.to_csv(os.path.join("models", "model_comparison.csv"), index=False)
    plot_tree_and_forest(df, "models")

    order = ["DecisionTree", "SVM", "RandomForest", "CSP"]
    plt.figure(figsize=(10, 7))
    sns.boxplot(x="Model", y="Test_Accuracy", data=df_comparison, order=order, palette="Set2")
    sns.stripplot(x="Model", y="Test_Accuracy", data=df_comparison, order=order,
                  color="black", alpha=0.4, jitter=True, size=4)
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance level (0.5)")
    plt.title("Model comparison — Test accuracy", fontweight="bold", fontsize=14)
    plt.ylabel("Test accuracy")
    plt.xlabel("")
    plt.legend()
    save_figure("model_comparison")
    print(df_comparison.groupby("Model")["Test_Accuracy"].agg(["mean", "std", "median", "count"]).reindex(order))

    # --- 4. BCI illiteracy filter --------------------------------------------
    n_test = int(round(len(df[df["Subject_ID"] == subjects[0]]) * TEST_SIZE))
    threshold = chance_threshold(n_test)
    print(f"Test trials per subject: {n_test} | Accuracy threshold (alpha={ALPHA}): {threshold:.3f}")

    # Subjects ranked by CSP+LDA test accuracy; above the threshold = eligible
    df_ranking = (df_comparison[df_comparison["Model"] == "CSP"][["Subject_ID", "Test_Accuracy"]]
                  .sort_values("Test_Accuracy", ascending=False).reset_index(drop=True))
    df_ranking["Rank"] = df_ranking.index + 1
    df_ranking["BCI_Eligible"] = df_ranking["Test_Accuracy"] > threshold
    df_ranking.to_csv(os.path.join("models", "bci_ranking.csv"), index=False)
    print(f"Eligible subjects: {df_ranking['BCI_Eligible'].sum()} / {len(df_ranking)}")

    plt.figure(figsize=(14, 6))
    colors = df_ranking["BCI_Eligible"].map({True: "#1D9E75", False: "#E24B4A"})
    plt.bar(df_ranking["Subject_ID"].astype(str), df_ranking["Test_Accuracy"], color=colors)
    plt.axhline(threshold, color="black", linestyle="--", label=f"Threshold alpha={ALPHA} ({threshold:.2f})")
    plt.axhline(0.5, color="gray", linestyle=":", label="Chance level (0.5)")
    plt.xticks(rotation=90)
    plt.ylabel("Test accuracy (CSP+LDA)")
    plt.title("Subjects ranked by BCI eligibility")
    plt.legend()
    save_figure("bci_ranking")

    # Same 4 models, same split and seed, only for the eligible subjects
    eligible_subjects = [int(s) for s in df_ranking.loc[df_ranking["BCI_Eligible"], "Subject_ID"]]
    df_eligible = train_all(df, csp_dataset, eligible_subjects, "models_eligible")

    df_all = pd.concat([df_comparison.assign(Group="All"), df_eligible.assign(Group="Eligible")],
                       ignore_index=True)
    plt.figure(figsize=(12, 7))
    sns.boxplot(x="Model", y="Test_Accuracy", hue="Group", data=df_all, order=order, palette="Set2")
    plt.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, label="Theoretical chance (0.50)")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Significance threshold ({threshold:.2f})")
    plt.title("Test accuracy: all subjects vs. eligible subjects", fontweight="bold", fontsize=14)
    plt.ylabel("Test accuracy")
    plt.xlabel("Model")
    plt.legend(loc="lower right")
    save_figure("all_vs_eligible")


if __name__ == "__main__":
    main()
