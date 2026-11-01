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

REPO_PATH = os.environ.get("BCI_REPO_PATH", ".")
os.chdir(REPO_PATH)
sys.path.append(REPO_PATH)

from src.eeg_datasets.preprocessed_dataset import PreprocessedDataset

DB_SELECCIONADA = "Cho2017"
SUBJECTS = None
SESSIONS = ["session_1"]
DATA_TO_LOAD = ["motor_imagery"]
SFREQ = 128
CANALES_FEATURES = ["C3", "Cz", "C4"]
CANALES_CSP = ["C3", "C1", "Cz", "C2", "C4"]
PARES_ELECTRODOS = ["C3-Cz", "C4-Cz", "C3-C4"]
PARES_IDX = [(0, 1), (2, 1), (0, 2)]
BANDAS_PSD = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (28, 30)]
BANDAS_MSC = [(8, 12), (12, 30)]

RANDOM_SEED = 45
SPLIT_TEST_SIZE = 0.20
K_FOLDS = 5
SCORING = "accuracy"
ALPHA = 0.05

COLUMNAS_META = ["Sujeto_ID", "Target", "Clase_Nombre"]
DIR_FIGURAS = os.path.join("results", "figures")
os.makedirs(DIR_FIGURAS, exist_ok=True)


def binarize_y(y):
    clases = np.unique(y)
    assert len(clases) == 2, "y debe tener exactamente dos clases"
    class_to_bin = {clases[0]: 0, clases[1]: 1}
    bin_to_class = {v: k for k, v in class_to_bin.items()}
    return np.vectorize(class_to_bin.get)(y), bin_to_class


def feature_total_power(X):
    return np.mean(np.square(X), axis=-1)


def feature_band_power(X, sfreq=SFREQ, fbands=BANDAS_PSD):
    n_ensayos, n_canales, _ = X.shape
    n_bandas = len(fbands)
    frecuencias, psd = welch(X, fs=sfreq, axis=-1, nperseg=int(sfreq))
    out = np.zeros((n_ensayos, n_canales * n_bandas))
    for b, (f_min, f_max) in enumerate(fbands):
        idx = (frecuencias >= f_min) & (frecuencias <= f_max)
        potencia = np.mean(psd[:, :, idx], axis=-1)
        for c in range(n_canales):
            out[:, c * n_bandas + b] = potencia[:, c]
    return out


def feature_kurtosis(X):
    return kurtosis(X, axis=-1)


def feature_skewness(X):
    return skew(X, axis=-1)


def feature_katz_fractal_dim(X):
    n_ensayos, n_canales, n_tiempos = X.shape
    out = np.zeros((n_ensayos, n_canales))
    N = n_tiempos - 1
    for e in range(n_ensayos):
        for c in range(n_canales):
            senal = X[e, c, :]
            L = np.sum(np.abs(np.diff(senal)))
            d = np.max(np.abs(senal - senal[0]))
            out[e, c] = np.log10(N) / (np.log10(d / L) + np.log10(N))
    return out


def feature_autoregressive_coefs(X):
    n_ensayos, n_canales, n_tiempos = X.shape
    out = np.zeros((n_ensayos, n_canales * 2))
    for e in range(n_ensayos):
        for c in range(n_canales):
            senal = X[e, c, :]
            r = np.correlate(senal, senal, mode="full")[n_tiempos - 1:n_tiempos + 2]
            r = r / r[0]
            R = np.array([[r[0], r[1]], [r[1], r[0]]])
            b = np.array([r[1], r[2]])
            try:
                coefs = np.linalg.solve(R, b)
            except np.linalg.LinAlgError:
                coefs = np.zeros(2)
            out[e, c * 2:c * 2 + 2] = coefs
    return out


def feature_permutation_entropy(X, order=3, delay=1):
    n_ensayos, n_canales, n_tiempos = X.shape
    out = np.zeros((n_ensayos, n_canales))
    n_words = n_tiempos - (order - 1) * delay
    for e in range(n_ensayos):
        for c in range(n_canales):
            x = X[e, c, :]
            words = np.stack([x[i * delay:i * delay + n_words] for i in range(order)], axis=1)
            _, counts = np.unique(np.argsort(words, axis=1), axis=0, return_counts=True)
            probs = counts / counts.sum()
            out[e, c] = -np.sum(probs * np.log2(probs)) / np.log2(math.factorial(order))
    return out


def feature_cross_correlation(X):
    out = np.zeros((X.shape[0], len(PARES_IDX)))
    for e in range(X.shape[0]):
        for p, (a, b) in enumerate(PARES_IDX):
            corr = np.corrcoef(X[e, a, :], X[e, b, :])[0, 1]
            out[e, p] = 0.0 if np.isnan(corr) else corr
    return out


def feature_magnitude_squared_coherence(X, sfreq=SFREQ, fbands=BANDAS_MSC):
    out = np.zeros((X.shape[0], len(PARES_IDX) * len(fbands)))
    for e in range(X.shape[0]):
        col = 0
        for a, b in PARES_IDX:
            f, msc = coherence(X[e, a, :], X[e, b, :], fs=sfreq, nperseg=int(sfreq))
            for f_min, f_max in fbands:
                idx = (f >= f_min) & (f <= f_max)
                out[e, col] = np.mean(msc[idx]) if np.any(idx) else 0.0
                col += 1
    return out


def feature_phase_locking_value(X):
    out = np.zeros((X.shape[0], len(PARES_IDX)))
    fases = np.angle(hilbert(X, axis=-1))
    for p, (a, b) in enumerate(PARES_IDX):
        out[:, p] = np.abs(np.mean(np.exp(1j * (fases[:, a, :] - fases[:, b, :])), axis=-1))
    return out


PIPELINE_CONFIG = {
    "potencia_total": {"active": True, "method": feature_total_power, "params": {}},
    "bandas_frecuencia": {"active": True, "method": feature_band_power, "params": {"sfreq": SFREQ}},
    "kurtosis": {"active": True, "method": feature_kurtosis, "params": {}},
    "skewness": {"active": True, "method": feature_skewness, "params": {}},
    "katz_fractal": {"active": True, "method": feature_katz_fractal_dim, "params": {}},
    "coeficientes_ar": {"active": True, "method": feature_autoregressive_coefs, "params": {}},
    "entropia_perm": {"active": True, "method": feature_permutation_entropy, "params": {"order": 3}},
    "correlacion_cruzada": {"active": True, "method": feature_cross_correlation, "params": {}},
    "coherencia_msc": {"active": True, "method": feature_magnitude_squared_coherence, "params": {"sfreq": SFREQ}},
    "plv_fase": {"active": True, "method": feature_phase_locking_value, "params": {}},
}


def banda_str(banda):
    return f"{banda[0]}-{banda[1]}Hz"


def generar_nombres_columnas(metodos):
    nombres = []
    for metodo in metodos:
        if metodo == "potencia_total":
            nombres += [f"TotalPow_{c}" for c in CANALES_FEATURES]
        elif metodo == "bandas_frecuencia":
            nombres += [f"BandPow_{c}_{banda_str(b)}" for c in CANALES_FEATURES for b in BANDAS_PSD]
        elif metodo == "kurtosis":
            nombres += [f"Kurtosis_{c}" for c in CANALES_FEATURES]
        elif metodo == "skewness":
            nombres += [f"Skewness_{c}" for c in CANALES_FEATURES]
        elif metodo == "katz_fractal":
            nombres += [f"Katz_{c}" for c in CANALES_FEATURES]
        elif metodo == "coeficientes_ar":
            nombres += [f"AR{i}_{c}" for c in CANALES_FEATURES for i in (1, 2)]
        elif metodo == "entropia_perm":
            nombres += [f"PermEnt_{c}" for c in CANALES_FEATURES]
        elif metodo == "correlacion_cruzada":
            nombres += [f"CrossCorr_{p}" for p in PARES_ELECTRODOS]
        elif metodo == "coherencia_msc":
            nombres += [f"MSC_{p}_{banda_str(b)}" for p in PARES_ELECTRODOS for b in BANDAS_MSC]
        elif metodo == "plv_fase":
            nombres += [f"PLV_{p}" for p in PARES_ELECTRODOS]
        else:
            raise ValueError(f"Método sin regla de nombrado: {metodo}")
    return nombres


def cargar_dataset(canales):
    return PreprocessedDataset(
        db_name=DB_SELECCIONADA,
        sessions=SESSIONS,
        subjects=SUBJECTS,
        data_to_load=DATA_TO_LOAD,
        channels=canales,
    )


def extraer_features(dataset):
    metodos = [n for n, cfg in PIPELINE_CONFIG.items() if cfg["active"]]
    if not metodos:
        raise ValueError("Activá al menos una característica en PIPELINE_CONFIG.")
    columnas = generar_nombres_columnas(metodos)
    bloques = []
    for sujeto in dataset.subject_list:
        try:
            X_eeg, y, _ = dataset.flatten_pool_data(subject_ids=[sujeto])
            y, _ = binarize_y(y)
        except Exception as e:
            print(f"[{sujeto}] Error al cargar: {e}. Saltando.")
            continue
        feats = []
        for n in metodos:
            res = PIPELINE_CONFIG[n]["method"](X_eeg, **PIPELINE_CONFIG[n]["params"])
            feats.append(res[:, np.newaxis] if res.ndim == 1 else res)
        df_s = pd.DataFrame(np.hstack(feats), columns=columnas)
        df_s.insert(0, "Clase_Nombre", pd.Series(y).map({0: "Mano Izquierda", 1: "Mano Derecha"}))
        df_s.insert(0, "Target", y)
        df_s.insert(0, "Sujeto_ID", sujeto)
        bloques.append(df_s)
        print(f"[{sujeto}] {df_s.shape[0]} ensayos x {len(columnas)} features")
    df = pd.concat(bloques, ignore_index=True)
    dir_exp = os.path.join("data", "features_extraction", DB_SELECCIONADA)
    os.makedirs(dir_exp, exist_ok=True)
    ruta = os.path.join(dir_exp, f"dataset_features_{'_'.join(metodos)}.csv")
    df.to_csv(ruta, index=False)
    print(f"Dataset guardado en {ruta}: {df.shape} ({df['Sujeto_ID'].nunique()} sujetos)")
    return df


def guardar_figura(nombre):
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURAS, f"{nombre}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def graficar_feature(df, prefijo, band_filter=None):
    cols = [c for c in df.columns if c.startswith(prefijo) and (band_filter is None or c.endswith(band_filter))]
    if not cols:
        raise ValueError(f"No hay columnas para '{prefijo}'")
    df_long = pd.melt(df[COLUMNAS_META + cols], id_vars=COLUMNAS_META, value_vars=cols,
                      var_name="Feature_Canal", value_name="Valor")
    pos = 1 if prefijo == "BandPow" else -1
    df_long["Canal"] = df_long["Feature_Canal"].str.split("_").str[pos]
    df_long["Canal_Condicion"] = df_long["Canal"] + " - " + df_long["Clase_Nombre"]
    plt.figure(figsize=(16, 7))
    sns.boxplot(data=df_long, x="Sujeto_ID", y="Valor", hue="Canal_Condicion",
                palette="Set2", fliersize=0, boxprops=dict(alpha=0.6))
    sns.stripplot(data=df_long, x="Sujeto_ID", y="Valor", hue="Canal_Condicion",
                  palette="Set2", dodge=True, alpha=0.4, size=3, jitter=0.2, legend=False)
    sufijo = f" ({band_filter})" if band_filter else ""
    plt.title(f"Distribución de {prefijo}{sufijo} por canal y condición", fontsize=14, fontweight="bold")
    plt.xlabel("Sujeto")
    plt.ylabel("Valor")
    plt.xticks(rotation=90)
    plt.legend(title="Canal & Tarea", bbox_to_anchor=(1.02, 1), loc="upper left")
    guardar_figura(f"eda_{prefijo}{'_' + band_filter if band_filter else ''}")


def ajustar_y_guardar(pipe, grid, X, y, sujeto, nombre_modelo, dir_base):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SPLIT_TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(pipe, param_grid=grid, cv=cv, scoring=SCORING, n_jobs=-1)
    gs.fit(X_train, y_train)
    mejor = gs.best_estimator_
    test_acc = accuracy_score(y_test, mejor.predict(X_test))
    print(f"[{nombre_modelo}] Sujeto {sujeto}: CV={gs.best_score_:.3f} Test={test_acc:.3f} {gs.best_params_}")
    dir_s = os.path.join(dir_base, nombre_modelo, f"subject_{sujeto}")
    os.makedirs(dir_s, exist_ok=True)
    with open(os.path.join(dir_s, "mejor_modelo.pkl"), "wb") as f:
        pickle.dump(mejor, f)
    with open(os.path.join(dir_s, "metricas_hiperparametros.json"), "w") as f:
        json.dump({"mejor_hiperparametros": gs.best_params_,
                   "metricas_cv_accuracy": float(gs.best_score_),
                   "metricas_test": {"accuracy": float(test_acc)}}, f, indent=4)
    return {"Sujeto_ID": sujeto, "Modelo": nombre_modelo, "Best_Params": str(gs.best_params_),
            "CV_Accuracy": gs.best_score_, "Test_Accuracy": test_acc}


MODELOS_FEATURES = {
    "DecisionTree": (DecisionTreeClassifier(random_state=RANDOM_SEED),
                     {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 5, 10]}),
    "RandomForest": (RandomForestClassifier(random_state=RANDOM_SEED),
                     {"clf__n_estimators": [100, 300], "clf__max_depth": [5, 10, None]}),
    "SVM": (SVC(random_state=RANDOM_SEED),
            {"clf__C": [0.1, 1.0, 10.0], "clf__kernel": ["linear", "rbf"]}),
}

GRID_CSP = {"csp__n_components": [2, 4], "csp__component_order": ["mutual_info", "alternate"]}


def entrenar_modelo_features(df, sujetos, nombre, dir_base):
    estimador, grid = MODELOS_FEATURES[nombre]
    columnas_X = [c for c in df.columns if c not in COLUMNAS_META]
    resultados = []
    for sujeto in sujetos:
        df_s = df[df["Sujeto_ID"] == sujeto]
        if df_s.empty:
            continue
        X, y = df_s[columnas_X].values, df_s["Target"].values
        if np.min(np.bincount(y)) < K_FOLDS:
            print(f"[{nombre}] Sujeto {sujeto}: pocos ensayos. Saltando.")
            continue
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimador)])
        resultados.append(ajustar_y_guardar(pipe, grid, X, y, sujeto, nombre, dir_base))
    df_res = pd.DataFrame(resultados)
    os.makedirs(os.path.join(dir_base, nombre), exist_ok=True)
    df_res.to_csv(os.path.join(dir_base, nombre, f"df_global_{nombre}.csv"), index=False)
    return df_res


def entrenar_csp(dataset, sujetos, dir_base):
    resultados = []
    for sujeto in sujetos:
        try:
            X, y, _ = dataset.flatten_pool_data(subject_ids=[sujeto])
            y, _ = binarize_y(y)
            if np.min(np.bincount(y)) < K_FOLDS:
                print(f"[CSP] Sujeto {sujeto}: pocos ensayos. Saltando.")
                continue
            pipe = Pipeline([("csp", CSP()), ("lda", LinearDiscriminantAnalysis())])
            resultados.append(ajustar_y_guardar(pipe, GRID_CSP, X, y, sujeto, "CSP", dir_base))
        except Exception as e:
            print(f"[CSP] Sujeto {sujeto} falló: {e}")
    df_res = pd.DataFrame(resultados)
    os.makedirs(os.path.join(dir_base, "CSP"), exist_ok=True)
    df_res.to_csv(os.path.join(dir_base, "CSP", "df_global_CSP.csv"), index=False)
    return df_res


def entrenar_todos(df, dataset_csp, sujetos, dir_base):
    resultados = [entrenar_modelo_features(df, sujetos, n, dir_base) for n in MODELOS_FEATURES]
    resultados.append(entrenar_csp(dataset_csp, sujetos, dir_base))
    return pd.concat(resultados, ignore_index=True)


def graficar_arbol_y_bosque(df, dir_base):
    sujeto = df["Sujeto_ID"].unique()[0]
    columnas_X = [c for c in df.columns if c not in COLUMNAS_META]
    for nombre in ["DecisionTree", "RandomForest"]:
        with open(os.path.join(dir_base, nombre, f"subject_{sujeto}", "mejor_modelo.pkl"), "rb") as f:
            clf = pickle.load(f).named_steps["clf"]
        arbol = clf if nombre == "DecisionTree" else clf.estimators_[0]
        plt.figure(figsize=(20, 10))
        plot_tree(arbol, feature_names=columnas_X, class_names=["Mano Izquierda", "Mano Derecha"],
                  filled=True, rounded=True, fontsize=8, max_depth=3)
        plt.title(f"{nombre} — Sujeto {sujeto}", fontsize=14, fontweight="bold")
        guardar_figura(f"arbol_{nombre}_sujeto_{sujeto}")
        if nombre == "RandomForest":
            df_imp = pd.DataFrame({"Feature": columnas_X, "Importancia": clf.feature_importances_})
            df_imp = df_imp.sort_values("Importancia", ascending=False).head(15)
            plt.figure(figsize=(10, 7))
            sns.barplot(data=df_imp, x="Importancia", y="Feature", hue="Feature", palette="Set2", legend=False)
            plt.title(f"Top 15 features — Random Forest — Sujeto {sujeto}", fontweight="bold")
            guardar_figura(f"importancia_RF_sujeto_{sujeto}")


def umbral_azar(n_test, p=0.5, alpha=ALPHA):
    for k in range(n_test + 1):
        if binom.sf(k - 1, n_test, p) <= alpha:
            return k / n_test
    return 1.0


def main():
    dataset_feat = cargar_dataset(CANALES_FEATURES)
    df = extraer_features(dataset_feat)

    graficar_feature(df, "TotalPow")
    graficar_feature(df, "BandPow", band_filter="8-12Hz")

    sujetos = df["Sujeto_ID"].unique().tolist()
    dataset_csp = cargar_dataset(CANALES_CSP)
    df_comparacion = entrenar_todos(df, dataset_csp, sujetos, "models")
    df_comparacion.to_csv(os.path.join("models", "df_comparacion_modelos.csv"), index=False)
    graficar_arbol_y_bosque(df, "models")

    orden = ["DecisionTree", "SVM", "RandomForest", "CSP"]
    plt.figure(figsize=(10, 7))
    sns.boxplot(x="Modelo", y="Test_Accuracy", data=df_comparacion, order=orden, palette="Set2")
    sns.stripplot(x="Modelo", y="Test_Accuracy", data=df_comparacion, order=orden,
                  color="black", alpha=0.4, jitter=True, size=4)
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance level (0.5)")
    plt.title("Comparación de modelos — Accuracy de test", fontweight="bold", fontsize=14)
    plt.ylabel("Accuracy de test")
    plt.xlabel("")
    plt.legend()
    guardar_figura("comparacion_modelos")
    print(df_comparacion.groupby("Modelo")["Test_Accuracy"].agg(["mean", "std", "median", "count"]).reindex(orden))

    n_test = int(round(len(df[df["Sujeto_ID"] == sujetos[0]]) * SPLIT_TEST_SIZE))
    umbral = umbral_azar(n_test)
    print(f"N_test por sujeto: {n_test} | Umbral de accuracy (alpha={ALPHA}): {umbral:.3f}")

    df_ranking = (df_comparacion[df_comparacion["Modelo"] == "CSP"][["Sujeto_ID", "Test_Accuracy"]]
                  .sort_values("Test_Accuracy", ascending=False).reset_index(drop=True))
    df_ranking["Ranking"] = df_ranking.index + 1
    df_ranking["Apto_BCI"] = df_ranking["Test_Accuracy"] > umbral
    df_ranking.to_csv(os.path.join("models", "ranking_bci.csv"), index=False)
    print(f"Sujetos aptos: {df_ranking['Apto_BCI'].sum()} / {len(df_ranking)}")

    plt.figure(figsize=(14, 6))
    colores = df_ranking["Apto_BCI"].map({True: "#1D9E75", False: "#E24B4A"})
    plt.bar(df_ranking["Sujeto_ID"].astype(str), df_ranking["Test_Accuracy"], color=colores)
    plt.axhline(umbral, color="black", linestyle="--", label=f"Umbral alpha={ALPHA} ({umbral:.2f})")
    plt.axhline(0.5, color="gray", linestyle=":", label="Chance level (0.5)")
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy de test (CSP+LDA)")
    plt.title("Ranking de sujetos por aptitud BCI")
    plt.legend()
    guardar_figura("ranking_bci")

    sujetos_aptos = [int(s) for s in df_ranking.loc[df_ranking["Apto_BCI"], "Sujeto_ID"]]
    df_aptos = entrenar_todos(df, dataset_csp, sujetos_aptos, "models_aptos")

    df_todo = pd.concat([df_comparacion.assign(Grupo="Todos"), df_aptos.assign(Grupo="Aptos")],
                        ignore_index=True)
    plt.figure(figsize=(12, 7))
    sns.boxplot(x="Modelo", y="Test_Accuracy", hue="Grupo", data=df_todo, order=orden, palette="Set2")
    plt.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, label="Azar teórico (0.50)")
    plt.axhline(umbral, color="red", linestyle="--", linewidth=1.5, label=f"Umbral de significancia ({umbral:.2f})")
    plt.title("Accuracy de test: todos los sujetos vs. sujetos aptos", fontweight="bold", fontsize=14)
    plt.ylabel("Accuracy de test")
    plt.xlabel("Modelo")
    plt.legend(loc="lower right")
    guardar_figura("todos_vs_aptos")


if __name__ == "__main__":
    main()
