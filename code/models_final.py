import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ============================================================
# 1. CARGA DEL DATASET
# ============================================================

df = pd.read_csv("cleanup_cdm_history_2015-2019.csv")


# ============================================================
# 2. SELECCIÓN DEL ÚLTIMO CDM POR EVENTO
# ============================================================

idx_last_cdm = df.groupby("event_id")["time_to_tca"].idxmin()
df_event = df.loc[idx_last_cdm].copy()

df_event = df_event.sort_values("event_id").reset_index(drop=True)

print("Shape original:", df.shape)
print("Eventos únicos en original:", df["event_id"].nunique())
print("Shape por evento:", df_event.shape)
print("Eventos únicos en dataset reducido:", df_event["event_id"].nunique())


# ============================================================
# 3. CREACIÓN DE VARIABLE OBJETIVO BINARIA
# ============================================================

df_prep = df_event.copy()

df_prep["risk_binary"] = (df_prep["risk"] > -30).astype(int)

print("\nDistribución risk_binary:")
print(df_prep["risk_binary"].value_counts())
print(df_prep["risk_binary"].value_counts(normalize=True))


# ============================================================
# 4. TRATAMIENTO DE VALORES NULOS
# ============================================================

df_prep = df_prep.drop(columns=["c_rcs_estimate"])

numeric_cols = df_prep.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols:
    if df_prep[col].isnull().sum() > 0:
        df_prep[col] = df_prep[col].fillna(df_prep[col].median())

remaining_nulls = df_prep[numeric_cols].isnull().sum().sum()

print("\nNulos restantes en variables numéricas:", remaining_nulls)
print("c_rcs_estimate en columnas:", "c_rcs_estimate" in df_prep.columns)


# ============================================================
# 5. TRATAMIENTO DE VARIABLE CATEGÓRICA
# ============================================================

print("\nDistribución original de c_object_type:")
print(df_prep["c_object_type"].value_counts())

df_prep["c_object_type"] = df_prep["c_object_type"].replace({
    "ROCKET BODY": "OTHER",
    "TBA": "OTHER"
})

print("\nDistribución agrupada de c_object_type:")
print(df_prep["c_object_type"].value_counts())

df_prep = pd.get_dummies(
    df_prep,
    columns=["c_object_type"],
    drop_first=True
)

print("\nColumnas tras one-hot encoding:")
print(df_prep.columns)


# ============================================================
# 6. REDUCCIÓN DE REDUNDANCIA ENTRE VARIABLES
# ============================================================

exclude_cols = ["event_id", "risk", "risk_binary"]

feature_cols = [
    col for col in df_prep.columns
    if col not in exclude_cols
]

numeric_feature_cols = df_prep[feature_cols].select_dtypes(
    include=[np.number, "bool"]
).columns.tolist()

corr_matrix = df_prep[numeric_feature_cols].corr().abs()

upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

threshold = 0.95

cols_to_drop_corr = [
    column for column in upper_triangle.columns
    if any(upper_triangle[column] > threshold)
]

print("\nNúmero de variables eliminadas por alta correlación:", len(cols_to_drop_corr))
print(cols_to_drop_corr)

df_prep_reduced = df_prep.drop(columns=cols_to_drop_corr)

print("Shape antes de reducción:", df_prep.shape)
print("Shape después de reducción:", df_prep_reduced.shape)


# ============================================================
# 7. ESCALADO Y NORMALIZACIÓN
# ============================================================

df_model = df_prep_reduced.copy()

exclude_cols = ["event_id", "risk", "risk_binary"]
one_hot_cols = [
    col for col in df_model.columns
    if col.startswith("c_object_type_")
]

feature_cols_to_scale = [
    col for col in df_model.columns
    if col not in exclude_cols + one_hot_cols
]

scaler = StandardScaler()

df_model[feature_cols_to_scale] = scaler.fit_transform(
    df_model[feature_cols_to_scale]
)

print("\nPrimeras filas tras escalado:")
print(df_model.head())

print("\nMedia variables escaladas:")
print(df_model[feature_cols_to_scale].mean().head())

print("\nDesviación estándar variables escaladas:")
print(df_model[feature_cols_to_scale].std().head())


# ============================================================
# 8. PREPARACIÓN FINAL DE DATOS
# ============================================================

# Eliminación de variables con posible fuga de información
leakage_cols = ["max_risk_scaling", "max_risk_estimate"]
df_model = df_model.drop(columns=leakage_cols)

X = df_model.drop(columns=["event_id", "risk", "risk_binary"])
y = df_model["risk_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nX_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nDistribución en entrenamiento:")
print(y_train.value_counts(normalize=True))

print("\nDistribución en prueba:")
print(y_test.value_counts(normalize=True))

# ============================================================
# 9. ENTRENAMIENTO DE MODELOS
# ============================================================

# Modelos con mejores parámetros
rf_best = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

gb_best = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

svm_best = SVC(
    C=10,
    gamma="scale",
    kernel="rbf",
    random_state=42
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Modelos optimizados
models = {
    "Random Forest": rf_best,
    "Gradient Boosting": gb_best,
    "SVM": svm_best
}

results_test = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    results_test[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

# Tabla de resultados
results_df = pd.DataFrame(results_test).T
print(results_df)

### Gráfica comparativa de métricas ###

results_df_plot = results_df.reset_index().rename(columns={"index": "Modelo"})

results_melted = results_df_plot.melt(
    id_vars="Modelo",
    var_name="Métrica",
    value_name="Valor"
)

plt.figure(figsize=(10, 6))
sns.barplot(data=results_melted, x="Modelo", y="Valor", hue="Métrica")
plt.title("Comparativa de métricas finales en el conjunto de prueba")
plt.ylim(0, 1)
plt.ylabel("Valor")
plt.xlabel("Modelo")
plt.legend(title="Métrica")
plt.tight_layout()
plt.show()

### Matrices de confusión ###

for name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Bajo riesgo", "Riesgo potencial"]
    )

    disp.plot(values_format="d")
    plt.title(f"Matriz de confusión - {name}")
    plt.tight_layout()
    plt.show()