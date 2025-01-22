# Realizamos los imports necesarios
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy import stats
import logging as log
import pandas as pd
import numpy as np

# Definimos la configuración básica de los logs
log.basicConfig(
    level=log.INFO,
    filename="logs/main.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M"
)

# Variables globales definidas
FILEPATH = "./csvs/stroke_prediction_dataset.csv"

def csvfile_to_dataframe(filepath: str) -> pd.DataFrame:
    """
    Attempts to convert a given CSV file into a pandas DataFrame.
    :param filepath: (str) the filepath to the CSV file to convert.
    :return: (pd.DataFrame) The DataFrame created from the given CSV file.
    """
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        log.error("There was an error trying to convert the CSV file into a DataFrame.")
        log.error("The error was " + str(e))
        return pd.DataFrame()

def replace_numeric_values(df: pd.DataFrame, column_name: str, method: str) -> None:
    """
    Attempts to replace the missed numeric values in the given DataFrame column with the given method.
    :param df: (pd.DataFrame) The DataFrame which the given column has missed values.
    :param column_name: (str) The column name to replace the missed values.
    :param method: (str) The method that will be used to replace the missed values.
    :return: It has no return.
    """
    try:
        # If is a valid method to use, then apply the proper replacement
        if method in ("median", "mean", "mode"):
            if  method == "median":
                df[column_name] = df[column_name].fillna(df[column_name].median())
            if method == "mean":
                df[column_name] = df[column_name].fillna(df[column_name].mean())
            if method == "mode":
                df[column_name] = df[column_name].fillna(df[column_name].mode().iloc[0])
    except Exception as e:
        log.error("There was an error trying to replace the missed numeric values in the DataFrame.")
        log.error("The error was " + str(e))
        return None

def replace_categorical_values(df: pd.DataFrame, column_name: str) -> None:
    """
    Attempts to replace the missed categorical values in the given DataFrame column.
    :param df: (pd.DataFrame) The DataFrame which the given column has missed values.
    :param column_name: (str) The column name to replace the missed values.
    :return: It has no return.
    """
    try:
        # If the quantity of missed values is less or equal to the 5% of the total number of rows, the column mode will be used
        # otherwise, the "NO_VALUE" String will be used
        if df[column_name].count <= (df.shape[0] * 0.05):
            df[column_name] = df[column_name].fillna(df[column_name].mode().iloc[0])
        else:
            df[column_name] = df[column_name].fillna("NO_VALUE")
    except Exception as e:
        log.error("There was an error trying to replace the missed categorical values in the DataFrame.")
        log.error("The error was " + str(e))
        return None

def handling_outliers_iqr(df: pd.DataFrame, column_name: str) -> list:
    """
    Attempts to handle the outliers with the interquartile range.
    :param df: (pd.DataFrame) The DataFrame which the given column outliers will be treated.
    :param column_name: (str) The column name to treat its outliers.
    :return: (list) A list with the 10 most bigger outliers of the column (if there are).
    """
    try:
        # Calculate the 1st and 3rd quantiles, also the interquartile range
        quartile_1 = df[column_name].quantile(0.25)
        quartile_3 = df[column_name].quantile(0.75)
        interquartile_range = quartile_3 - quartile_1

        # Calculate the lower and upper bounds
        lower_bound = quartile_1 - 1.5 * interquartile_range
        upper_bound = quartile_3 + 1.5 * interquartile_range

        # Get the 10 most extreme outliers (sorted by deviation from the median), just to have an example of them, if they exist
        outliers_detected = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)][column_name]
        outliers_detected10 = outliers_detected.reindex((outliers_detected - outliers_detected.median()).abs().sort_values(ascending=False).index).head(10).tolist()

        # Handle the outliers by capping them
        df[column_name] = df[column_name].clip(lower=lower_bound, upper=upper_bound)

        # Return the outliers found in the column
        return outliers_detected10
    except Exception as e:
        log.error("There was an error trying to handle the outliers values with the interquartile range in the DataFrame.")
        log.error("The error was " + str(e))
        return []

def handling_outliers_zscore(df: pd.DataFrame, column_name: str) -> list:
    """
    Attempts to handle the outliers with the Z-Score.
    :param df: (pd.DataFrame) The DataFrame which the given column outliers will be treated.
    :param column_name: (str) The column name to treat its outliers.
    :return: (list) A list with the 10 most bigger outliers of the column (if there are).
    """
    try:
        # Calculate the Z-Scores
        z_scores = stats.zscore(df[column_name])
        z_score_threshold = 3

        # Get outliers and sort by absolute z-score
        outliers_detected = df[abs(z_scores) > z_score_threshold][column_name]
        outliers_detected10 = outliers_detected.reindex((outliers_detected - outliers_detected.mean()).abs().sort_values(ascending=False).index).head(10).tolist()

        # Treat the outliers by capping them using the threshold (mean +/- 3 * std)
        mean = df[column_name].mean()
        std = df[column_name].std()
        lower_bound = mean - z_score_threshold * std
        upper_bound = mean + z_score_threshold * std

        # Handle the outliers by capping them
        df[column_name] = df[column_name].clip(lower=lower_bound, upper=upper_bound)

        # Return the outliers found in the column
        return outliers_detected10
    except Exception as e:
        log.error("There was an error trying to handle the outliers values with the Z-Score in the DataFrame.")
        log.error("The error was " + str(e))
        return []

def handling_outliers_percentile(df: pd.DataFrame, column_name: str) -> list:
    """
    Attempts to handle the outliers with the percentile method.
    :param df: (pd.DataFrame) The DataFrame which the given column outliers will be treated.
    :param column_name: (str) The column name to treat its outliers.
    :return: (list) A list with the 10 most bigger outliers of the column (if there are).
    """
    try:
        # Define percentiles
        lower_percentile = df[column_name].quantile(0.01)
        upper_percentile = df[column_name].quantile(0.99)

        # Get the 10 most extreme outliers (sorted by deviation from the median), just to have an example of them, if they exist
        outliers_detected= df[(df[column_name] < lower_percentile) | (df[column_name] > upper_percentile)][column_name]
        outliers_detected10 = outliers_detected.reindex((outliers_detected - outliers_detected.median()).abs().sort_values(ascending=False).index).head(10).tolist()

        # Handle the outliers by capping them
        df[column_name] = df[column_name].clip(lower=lower_percentile, upper=upper_percentile)

        # Return the outliers found in the column
        return outliers_detected10
    except Exception as e:
        log.error("There was an error trying to handle the outliers values with the percentile method in the DataFrame.")
        log.error("The error was " + str(e))
        return []

def handling_outliers_methods(df: pd.DataFrame, column_name: str, method: str) -> list:
    """
    Attempts to handle the outliers with the given method.
    :param df: (pd.DataFrame) The DataFrame which the given column outliers will be treated.
    :param column_name: (str) The column name to treat its outliers.
    :param method: (str) The method to use for handling the outliers
    :return: (list) A list with the 10 most bigger outliers of the column (if there are).
    """
    try:
        if method in ("iqr", "zscore", "percentile"):
            if method == "iqr":
                return handling_outliers_iqr(df=df, column_name=column_name)
            if method == "zscore":
                return handling_outliers_zscore(df=df, column_name=column_name)
            if method == "percentile":
                return handling_outliers_percentile(df=df, column_name=column_name)
    except Exception as e:
        log.error("There was an error trying to handle the outliers values with the " + method + " method (this is the general method) in the DataFrame.")
        log.error("The error was " + str(e))
        return []

def apply_supervised_learning_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, model: str, random_state: int) -> tuple:
    """
    Attempts to apply a supervised learning model with the given parameters.
    :param X_train: (pd.DataFrame) The training features data
    :param y_train: (pd.DataFrame) The training target data
    :param X_test: (pd.DataFrame) The test features data
    :param model: (str) The supervised model to use
    :param random_state: (int) It controls the pseudo-random generation
    :return: (tuple) The accuracy score of the model and its classification report
    """
    try:
        if model in ("random_forest", "logistic_regression", "support_vector_machine"):
            supervised_model = None
            if model == "random_forest":
                supervised_model = RandomForestClassifier(random_state=random_state)
            if model == "logistic_regression":
                supervised_model = LogisticRegression(random_state=random_state)
            if model == "support_vector_machine":
                supervised_model = SVC(random_state=random_state)

            if supervised_model != None:
                supervised_model.fit(X_train, y_train)
                y_pred = supervised_model.predict(X_test)

                return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)
    except Exception as e:
        log.error("There was an error trying to apply the supervised model " + model + ".")
        log.error("The error was " + str(e))
        return 0.0, None

def apply_unsupervised_learning_algorithm(df: pd.DataFrame, algorithm: str, n_clusters: int = 3, eps: float = 0.5, min_samples: int = 5, random_state: int = 42) -> tuple:
    """
    Attempts to apply an unsupervised learning algorithm with the given parameters.
    :param df: (pd.DataFrame) The DataFrame that contains the features to cluster
    :param algorithm: (str) The algorithm name to use
    :param n_clusters: (int) The number of cluster for K-Means (default: 3)
    :param eps: (float) Epsilon parameter for DBSCAN (default: 0.5)
    :param min_samples: (int) Minimum samples for DBSCAN (default: 5)
    :param random_state: (int) It controls the pseudo-random generation (default: 42)
    :return: (tuple)
    """
    try:
        if algorithm in ("kmeans", "dbscan"):
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df)

            pca = PCA()
            pca_data = pca.fit_transform(scaled_data)

            unsupervised_alg = None
            if algorithm == "kmeans":
                unsupervised_alg = KMeans(n_clusters=n_clusters, random_state=random_state)
            if algorithm == "dbscan":
                unsupervised_alg = DBSCAN(eps=eps, min_samples=min_samples)

            return unsupervised_alg.fit_predict(scaled_data), pca_data

    except Exception as e:
        log.error("There was an error trying to apply the unsupervised algorithm " + algorithm + ".")
        log.error("The error was " + str(e))
        return np.zeros(shape=(1,1)), np.zeros(shape=(1,1))

def plot_clusters(pca_data: np.ndarray, clusters: np.ndarray, title: str) -> None:
    """
    Attempts to plot the clusters in a graph
    :param pca_data: (np.ndarray) The PCA for visualization
    :param clusters: (np.ndarray) The clusters to display
    :param title: (str) The title of the graph
    :return: It has no return
    """
    try:
        plt.figure()
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap="plasma")
        plt.title(title)
        plt.xlabel("Primer Componente Principal")
        plt.ylabel("Segundo Componente Principal")
        plt.colorbar(label="Cluster")
        plt.show()
    except Exception as e:
        print(e)
        log.error("There was an error trying to plot the clusters.")
        log.error("The error was " + str(e))
        return None

def check_and_perform_ttest(num_var: str, group1: pd.DataFrame, group2: pd.DataFrame, alpha: float = 0.05) -> None:
    """
    Attempts to perform a proper test by hypothesis contrast
    :param num_var: (str) The variable name to analyze
    :param group1: (pd.DataFrame) The DataFrame with the data corresponding to the group 1 (True)
    :param group2: (pd.DataFrame) The DataFrame with the data corresponding to the group 2 (False)
    :param alpha: (float) The alpha parameter to compare with the p-value of the tests
    :return: It has no return.
    """
    try:
        print("\nAnálisis de la variable", num_var)

        # Verificamos normalidad (Test de Shapiro-Wilk)
        print("\nTest de normalidad (Shapiro-Wilk):")
        _, p_val1 = stats.shapiro(group1)
        _, p_val2 = stats.shapiro(group2)

        print("p-valor del grupo 1:", p_val1)
        print("p-valor del grupo 2:", p_val2)

        is_normal = (p_val1 > alpha) and (p_val2 > alpha)

        print("Los datos son normales:", is_normal)

        # Verificamos la homocedasticidad (Test de Levene)
        print("\nTest de homocedasticidad (Levene):")
        _, p_val_levene = stats.levene(group1, group2)

        print("p-valor del test de Levene:", p_val_levene)
        is_equal_var = p_val_levene > alpha
        print("Las varianzas son iguales:", is_equal_var)

        # Realizamos el test correspondiente
        test_name = ""
        t_stat, p_val = 0.0, 0.0

        # Si tanto los datos tienen normalidad como homocedasticidad aplicamos T de Student
        if is_normal and is_equal_var:
            t_stat, p_val = stats.ttest_ind(group1, group2)
            test_name = "T-Student"
        # Si los datos tienen solo normalidad aplicamos T de Welch
        elif is_normal:
            t_stat, p_val = stats.ttest_ind(group1, group2, is_equal_var=False)
            test_name = "T-Welch"
        # Si los datos no tienen normalidad y/o homocedasticidad aplicamos U de Mann-Whitney
        else:
            t_stat, p_val = stats.mannwhitneyu(group1, group2, alternative="two-sided")
            test_name = "Test U de Mann-Whitney"

        print("\nTest de prueba por contraste de hipótesis utilizado:", test_name)
        print("Valor estadístico:", t_stat)
        print("p-valor:", p_val)
        print("Significant difference:", p_val < alpha)

        # Representamos visualmente con un QQ-Plot y un Box Plot
        plt.figure(figsize=(10, 4))

        # QQ-plot
        plt.subplot(1, 2, 1)
        stats.probplot(np.concatenate([group1, group2]), dist="norm", plot=plt)
        plt.title("Q-Q Plot para la variable " + num_var)

        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot([group1, group2], tick_labels=["Grupo 1 (Stroke)", "Grupo 2 (No Stroke)"])
        plt.title("Box Plot para la variable " + num_var)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        log.error("There was an error trying to apply a test by hypothesis contrast.")
        log.error("The error was " + str(e))
        return None

# Cargamos el fichero con la información de los ataques cardiácos en un DataFrame
stroke_data = csvfile_to_dataframe(FILEPATH)

# Verificamos la forma del DataFrame
print("El DataFrame (dataset) de \"Stroke\" cuenta con", stroke_data.shape[1], "columnas y", stroke_data.shape[0], "filas")

# Imprimimos información relevante sobre las columnas que conforman el DataFrame
# La información que mostraremos será la prudente según el propósito y tipo de dato de la columna
for column_name in stroke_data.columns:
    print("\nInformación de la columna:", column_name)
    print("Cantidad de registros no nulos:", stroke_data[column_name].count())

    # Obtenemos la cantidad de valores únicos de cada columna
    # y si son menos o igual a 10 (cantidad manejable) los mostramos también
    nunique_values = stroke_data[column_name].nunique()
    print("Cantidad de valores únicos:", nunique_values)

    if nunique_values <= 10:
        print("Los valores únicos son:", ", ".join(stroke_data[column_name].unique().astype(str)))

    # Ignoramos las estadísticas numéricas de la columna "Patient ID" ya que no nos son relevantes en absoluto
    # Para las variables que solo contienen valores como 0 o 1 podría no ser muy útiles, pero al variar en cantidad
    # ver sus estadísticas, nos permitirá rápidamente deducir si hay más 1 (positivo) o 0 (negativo)
    if stroke_data[column_name].dtype in (int, float) and column_name not in ("Patient ID"):
        print("Valor mínimo:", stroke_data[column_name].min())
        print("Valor máximo:", stroke_data[column_name].max())
        print("Media:", stroke_data[column_name].mean())
        print("Moda:", stroke_data[column_name].mode().iloc[0])
        print("Mediana:", stroke_data[column_name].median())
        print("Desviación estándar:", stroke_data[column_name].std())

# Integración
# Creamos las dos nuevas columnas "Cholesterol Levels HDL" y "Cholesterol Levels LDL" con los valores HDL y LDL de "Cholesterol Levels"
# Utilizamos str.extract para obtener los valores y los convertimos a tipo int
stroke_data["Cholesterol Levels HDL"] = stroke_data["Cholesterol Levels"].str.extract(r"HDL:\s*(\d+)").astype(int)
stroke_data["Cholesterol Levels LDL"] = stroke_data["Cholesterol Levels"].str.extract(r"LDL:\s*(\d+)").astype(int)

# Selección
# Eliminamos del DataFrame las columnas que no nos serán relevantes, o complican demasiado, en predecir si el paciente sufrirá de un Stroke
stroke_data = stroke_data.drop(columns=["Patient ID", "Patient Name", "Blood Pressure Levels", "Cholesterol Levels", "Symptoms"])

# Tratamiento de valores nulos
# * Valores categóricas (String)
# Para las variables categóricas, si la cantidad de valores vacíos es menor o igual al 5% del total de registros, se utilizará la moda de la variable
# en caso de que se supere este valor, se utilizará la "String" definida "NO_VALUE"
# * Valores numéricas
# Para las variables numéricas, utilizaremos la mediana de la columna para reemplazar los valores vacíos, ya que suele ser una medida que rara vez
# se ve afectada por los valores extremos
for column_name in stroke_data.columns:
    if stroke_data[column_name].count() < stroke_data.shape[0]:
        if stroke_data[column_name].dtype in (int, float):
            replace_numeric_values(df=stroke_data, column_name=column_name, method="median")
        else:
            replace_categorical_values(df=stroke_data, column_name=column_name)

# Falso reemplazamiento de datos
# Para poder probar la imputación de datos utilizando la mediana y media, crearemos una copia del DataFrame y reemplazaremos el 20% de los valores de las columnas
# "Average Glucose Level" y "Body Mass Index (BMI)" con la mediana y media respectivamente.
stroke_fake_replace = stroke_data.copy()
stroke_fake_replace.loc[4::5, "Average Glucose Level"] = np.nan
stroke_fake_replace.loc[4::5, "Body Mass Index (BMI)"] = np.nan

# Verificamos que los datos se hayan eliminado efectivamente
print("\nFalso reemplazamiento de datos en el DataFrame \"stroke_fake_replace\"")
print("Pre-Reemplazo - La columna \"Body Mass Index (BMI)\" tiene un total de", stroke_fake_replace["Body Mass Index (BMI)"].count(), "registros.")
print("Pre-Reemplazo - La columna \"Average Glucose Level\" tiene un total de", stroke_fake_replace["Average Glucose Level"].count(), "registros.")

# Reemplazamos los datos con la mediana y media
replace_numeric_values(df=stroke_fake_replace, column_name="Body Mass Index (BMI)", method="median")
replace_numeric_values(df=stroke_fake_replace, column_name="Average Glucose Level", method="mean")

# Verificamos el efectivo reemplazamiento de los datos
print("Post-Reemplazo - La columna \"Body Mass Index (BMI)\" tiene un total de", stroke_fake_replace["Body Mass Index (BMI)"].count(), "registros.")
print("Post-Reemplazo - La columna \"Average Glucose Level\" tiene un total de", stroke_fake_replace["Average Glucose Level"].count(), "registros.")

# Gestión de outliers
# Para la gestión de los outliers, utilizaremos el rango intercuartílico, aplicaremos esto tanto al DataFrame normal como aquel de falso reemplazamiento
# Gestionaremos únicamente los valores numéricos
for column_name in stroke_data:
    if stroke_data[column_name].dtype in (int, float):
        handling_outliers_methods(df=stroke_data, column_name=column_name, method="iqr")
        handling_outliers_methods(df=stroke_fake_replace, column_name=column_name, method="iqr")

# Gestión de los tipos de datos
# Convertimos las variables que sea prudente a tipo de dato "category"
categorical_variables = ["Gender", "Marital Status", "Work Type", "Residence Type", "Smoking Status", "Alcohol Intake",
                         "Physical Activity", "Family History of Stroke", "Dietary Habits"]

for cat_var in categorical_variables:
    stroke_data[cat_var] = stroke_data[cat_var].astype("category")
    stroke_fake_replace[cat_var] = stroke_fake_replace[cat_var].astype("category")

# Creamos una categoría llamada "Age Range" en base a la variable "Age" dividida en tres intervalos iguales
bins = [17, 41, 65, 90]
labels = ["18-41", "42-65", "66-90"]

stroke_data["Age Range"] = pd.cut(stroke_data["Age"], bins=bins, labels=labels, right=True)
stroke_fake_replace["Age Range"] = pd.cut(stroke_fake_replace["Age"], bins=bins, labels=labels, right=True)

# Añadimos "Age Range" a la lista de variables categóricas
categorical_variables.append("Age Range")

# Para evitar tener problemas con los modelos, utilizaremos LabelEncoder para las variables categóricas
label_encoder = LabelEncoder()
for cat_var in categorical_variables:
    stroke_data[cat_var] = label_encoder.fit_transform(stroke_data[cat_var])
    stroke_fake_replace[cat_var] = label_encoder.fit_transform(stroke_fake_replace[cat_var])

# Aplicación de método de aprendizaje supervisado

# Separamos el dataset en características y variable objetivo ("Diagnosis")
X = stroke_data.drop(columns=["Diagnosis"])
y = stroke_data["Diagnosis"]

# Separamos los datos en entrenamiento y pruebas (80% entrenamiento, 20% pruebas y utilizamos el parámetro random_state a 42 para mantener reproducibilidad en el código)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Realizamos lo mismo para el DataFrame de falso reemplazamiento
X_fake_replace = stroke_fake_replace.drop(columns=["Diagnosis"])
X_train_fake_replace, X_test_fake_replace, y_train_fake_replace, y_test_fake_replace = train_test_split(X_fake_replace, y, test_size=0.2, random_state=42)

# Utilizaremos el método de aprendizaje supervisado Random Forest
# Existen diversos métodos de aprendizaje supervisado pero utilizaremos Random Forest ya que es un buen método para capturar relaciones complejas en los datos
# Usa múltiples árboles de decisión y funciona de forma efectiva tanto para datos numéricos como categóricos
model_accuracy, model_classification_report = apply_supervised_learning_model(X_train=X_train, y_train=y_train, X_test=X_test, model="random_forest", random_state=42)

print("\nLa precisión del modelo supervisado Random Forest es:", model_accuracy)
print("El reporte de clasificación es:\n", model_classification_report)

# Probamos con el DataFrame de falso reemplazamiento
model_accuracy_fake_replace, model_classification_report_fake_replace = apply_supervised_learning_model(X_train=X_train_fake_replace, y_train=y_train_fake_replace, X_test=X_test_fake_replace, model="random_forest", random_state=42)

print("\nFalso Reemplazamiento - La precisión del modelo supervisado Random Forest es::", model_accuracy_fake_replace)
print("El reporte de clasificación es:\n", model_classification_report_fake_replace)

# Aplicación de método de aprendizaje no supervisado
# Utilizaremos el método de aprendizaje no supervisado K-Means
df_features = stroke_data.drop(columns=["Diagnosis"])
clusters, pca_data = apply_unsupervised_learning_algorithm(df=df_features, algorithm="kmeans", n_clusters=3, random_state=42)
plot_clusters(pca_data=pca_data, clusters=clusters, title="Algoritmo K-Means sobre el dataset de \"Stroke\"")

df_features_fake_replace = stroke_fake_replace.drop(columns=["Diagnosis"])
clusters_fake_replace, pca_data_fake_replace = apply_unsupervised_learning_algorithm(df=df_features_fake_replace, algorithm="kmeans", n_clusters=3, random_state=42)
plot_clusters(pca_data=pca_data_fake_replace, clusters=clusters_fake_replace, title="Falso Reempl. - Algoritmo K-Means sobre el dataset de \"Stroke\"")

# Aplicación de prueba por contraste de hipótesis
# Seleccionamos las variables numéricas continuas y las tratamos una a una
# Esto lo realizaremos únicamente para el DataFrame de stroke_data para ahorrar tiempo
numerical_vars = ["Age", "Average Glucose Level", "Body Mass Index (BMI)", "Stress Levels"]

for num_var in numerical_vars:
    group1 = stroke_data[stroke_data["Diagnosis"] == "Stroke"][num_var]
    group2 = stroke_data[stroke_data["Diagnosis"] == "No Stroke"][num_var]
    check_and_perform_ttest(num_var, group1, group2)

# Finalmente, guardamos el dataset como CSV con todos los cambios que se han aplicado
stroke_data.to_csv("./csvs/stroke_prediction_dataset_PROCESSED.csv")