# Importación de librerías necesarias
import os  # Para manipular archivos y directorios
import pandas as pd  # Para manejo y análisis de datos
import numpy as np  # Operaciones matemáticas y manipulaciones de matrices
import matplotlib.pyplot as plt  # Para graficar resultados
import torch  # No se utiliza explícitamente en este código, pero puede ser útil para modelos avanzados
from surprise import KNNBasic, Dataset, Reader, accuracy  # Librerías para filtrado colaborativo
from surprise.model_selection import train_test_split  # División de datos para entrenamiento y pruebas
from sklearn.metrics import mean_absolute_error  # Cálculo de error absoluto medio (MAE)
from sklearn.feature_extraction.text import TfidfVectorizer  # Para convertir texto en representaciones numéricas
from sklearn.metrics.pairwise import cosine_similarity  # Para medir similitud entre vectores
from sklearn.decomposition import TruncatedSVD  # Para reducción de dimensionalidad
from sklearn.neighbors import NearestNeighbors  # Algoritmo k-NN (k-Nearest Neighbors)

# -------------------------
# Función para buscar un archivo en el directorio actual
# -------------------------
def find_file_local(filename, search_path):
    """
    Busca un archivo en el directorio especificado.
    :param filename: Nombre del archivo a buscar.
    :param search_path: Ruta del directorio donde buscar.
    :return: Ruta completa del archivo si se encuentra, None si no.
    """
    for root, dirs, files in os.walk(search_path):  # Recorrer carpetas y subcarpetas
        if filename in files:  # Verificar si el archivo está en la lista
            return os.path.join(root, filename)  # Retornar la ruta completa
    return None  # Retorna None si no encuentra el archivo

# Nombre del archivo CSV que contiene los datos
file_name = 'Gift_Cards.csv'

# Obtener el directorio actual donde se ejecuta el script
current_dir = os.getcwd()

# Buscar el archivo en el directorio actual
file_path = find_file_local(file_name, current_dir)

# -------------------------
# Cargar y procesar el archivo CSV
# -------------------------
if file_path:
    print(f"Archivo encontrado en: {file_path}")
    # Leer el archivo CSV con pandas
    df = pd.read_csv(file_path)
    print("Nombres de las columnas:", df.columns.tolist())  # Mostrar las columnas del dataset
    print(df.head(10))  # Mostrar las primeras 10 filas del dataset

    # Filtrar usuarios con al menos 50 interacciones
    min_interactions = 50
    user_counts = df['user_id'].value_counts()  # Contar las interacciones por usuario
    filtered_users = user_counts[user_counts >= min_interactions].index  # Usuarios con al menos 50 interacciones
    df = df[df['user_id'].isin(filtered_users)]  # Filtrar usuarios en el dataset
else:
    print(f"El archivo '{file_name}' no se encontró en la carpeta actual.")
    exit()  # Finalizar el programa si no se encuentra el archivo

# -------------------------
# Configuración y entrenamiento del modelo k-NN
# -------------------------
# Convertir datos al formato requerido por Surprise
reader = Reader(rating_scale=(1, 5))  # Escala de calificaciones de 1 a 5
data = Dataset.load_from_df(df[['user_id', 'title', 'rating']], reader)  # Cargar datos para filtrado colaborativo

# Dividir el dataset en conjuntos de entrenamiento (70%) y prueba (30%)
trainset, testset = train_test_split(data, test_size=0.3)

# Configurar el algoritmo k-NN
sim_options = {
    'name': 'cosine',  # Usar similitud coseno
    'user_based': False  # Similitud entre productos en lugar de usuarios
}
algo_knn = KNNBasic(k=30, sim_options=sim_options)  # Configurar k-NN con 30 vecinos más cercanos

# Entrenar el modelo k-NN con los datos de entrenamiento
algo_knn.fit(trainset)

# Realizar predicciones y evaluar el modelo
predictions = algo_knn.test(testset)
print("k-NN Coseno - RMSE:", accuracy.rmse(predictions))  # Error cuadrático medio (RMSE)

# Calcular MAE (Error absoluto medio)
y_true = [pred.r_ui for pred in predictions]  # Calificaciones reales
y_pred = [pred.est for pred in predictions]  # Calificaciones predichas
mae = mean_absolute_error(y_true, y_pred)
print("MAE:", mae)

# -------------------------
# Probar diferentes valores de k
# -------------------------
k_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # Lista de valores para k
mae_values = []  # Para almacenar MAE de cada k
rmse_values = []  # Para almacenar RMSE de cada k

for k in k_values:
    algo_knn = KNNBasic(k=k, sim_options=sim_options)  # Configurar k-NN con k actual
    algo_knn.fit(trainset)  # Entrenar el modelo
    predictions = algo_knn.test(testset)  # Predecir en el conjunto de prueba
    y_true = [pred.r_ui for pred in predictions]  # Calificaciones reales
    y_pred = [pred.est for pred in predictions]  # Calificaciones predichas
    mae = mean_absolute_error(y_true, y_pred)  # Calcular MAE
    rmse = accuracy.rmse(predictions, verbose=False)  # Calcular RMSE
    mae_values.append(mae)  # Agregar MAE a la lista
    rmse_values.append(rmse)  # Agregar RMSE a la lista

# -------------------------
# Procesamiento de texto con TF-IDF
# -------------------------
# Crear una nueva columna combinando título y texto del producto
df['product_features'] = df['title'] + " " + df['text']

# Rellenar valores nulos en las características de texto
df['product_features'] = df['product_features'].fillna('')

# Filtrar productos con descripciones válidas
df_filtered = df[df['product_features'].str.strip() != ''].reset_index(drop=True)

# Aplicar TF-IDF para convertir texto a representaciones numéricas
tfidf = TfidfVectorizer(stop_words='english')  # Ignorar palabras comunes en inglés
tfidf_matrix = tfidf.fit_transform(df_filtered['product_features'])  # Generar matriz TF-IDF
print("Matriz TF-IDF:", tfidf_matrix.shape)  # Mostrar dimensiones de la matriz

# -------------------------
# Función para recomendaciones basadas en contenido
# -------------------------
def get_content_recommendations(asin, df_filtered, cosine_sim, top_n=5):
    """
    Recomendar productos similares usando similitud coseno y TF-IDF.
    :param asin: Código ASIN del producto a buscar similitudes.
    :param df_filtered: DataFrame con los datos de productos.
    :param cosine_sim: Matriz de similitudes calculadas.
    :param top_n: Número de recomendaciones a generar.
    :return: Lista de productos similares.
    """
    try:
        if asin not in df_filtered['asin'].values:
            raise ValueError(f"El ASIN '{asin}' no se encuentra en los datos filtrados.")
        idx = df_filtered[df_filtered['asin'] == asin].index[0]  # Índice del producto en el DataFrame
        sim_scores = list(enumerate(cosine_sim[idx]))  # Calcular similitudes
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Ordenar por similitud
        sim_indices = [i[0] for i in sim_scores[1:top_n+1]]  # Obtener índices de los productos más similares
        return [(df_filtered.iloc[i]['title'], df_filtered.iloc[i]['asin']) for i in sim_indices]  # Títulos y códigos
    except Exception as e:
        print(f"Error al buscar recomendaciones: {e}")
        return []

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def hybrid_recommendation(user_id, algo, df, alpha=0.5, top_n=10):
    """
    Genera recomendaciones híbridas combinando filtrado colaborativo y contenido.
    
    :param user_id: ID del usuario
    :param algo: Modelo de filtrado colaborativo (surprise)
    :param df: DataFrame con los datos de los productos
    :param alpha: Peso para el filtrado colaborativo (0 < alpha < 1)
    :param top_n: Número de recomendaciones a generar
    :return: Lista de tuplas con las recomendaciones híbridas (nombre del producto, ASIN, puntuación híbrida)
    """
    # Filtrado colaborativo: obtener predicciones para el usuario
    user_items = df[df['user_id'] == user_id]['asin'].unique()
    item_recs = []

    for item in user_items:
        est_rating = algo.predict(user_id, item).est
        item_recs.append((item, est_rating))

    # Normalizar las calificaciones del contenido
    item_ratings = df[df['asin'].isin(user_items)].groupby('asin')['rating'].mean().reset_index()
    scaler = MinMaxScaler(feature_range=(0, 1))
    item_ratings['norm_rating'] = scaler.fit_transform(item_ratings[['rating']])

    # Filtrado de contenido: usar las calificaciones normalizadas
    content_recs = [(row['asin'], row['norm_rating']) for _, row in item_ratings.iterrows()]

    # Calcular la puntuación híbrida combinando ambas recomendaciones
    hybrid_scores = []
    for (item, cf_score), (_, content_score) in zip(item_recs, content_recs):
        hybrid_score = alpha * cf_score + (1 - alpha) * content_score
        product_name = df.loc[df['asin'] == item, 'title'].iloc[0]  # Obtener el nombre del producto
        hybrid_scores.append((product_name, item, hybrid_score))

    # Ordenar las recomendaciones híbridas por puntuación y seleccionar las mejores
    hybrid_scores.sort(key=lambda x: x[2], reverse=True)
    return hybrid_scores[:top_n]
def plot_hybrid_recommendations(hybrid_recs):
    """
    Genera un gráfico de barras para las recomendaciones híbridas con el nombre del producto y el ASIN.

    :param hybrid_recs: Lista de tuplas con (nombre del producto, ASIN, puntuación híbrida).
    """
    if not hybrid_recs or not all(isinstance(score, (int, float)) for _, _, score in hybrid_recs):
        print("Advertencia: No hay datos válidos para graficar.")
        return

    # Crear etiquetas combinando nombre del producto y ASIN
    labels = [f"{nombre} (ASIN: {asin})" for nombre, asin, _ in hybrid_recs]
    scores = [score for _, _, score in hybrid_recs]

    # Crear el gráfico de barras
    plt.figure(figsize=(12, 8))
    plt.barh(labels, scores, color='skyblue')
    plt.xlabel('Puntuación Híbrida', fontsize=14)
    plt.ylabel('Producto', fontsize=14)
    plt.title('Recomendaciones Híbridas para el Usuario', fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()



#############################################################################
##Prueba KNN y SVD####

# 1. Preprocesamiento de texto con TF-IDF
def aplicar_tfidf(df, columna_texto):
    if columna_texto not in df.columns:
        raise ValueError(f"La columna '{columna_texto}' no se encuentra en el DataFrame.")
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[columna_texto].fillna(''))
    return tfidf_matrix, tfidf

# 2. Reducción de dimensionalidad con SVD
def aplicar_svd(tfidf_matrix, n_components=55):
    if tfidf_matrix.shape[1] < n_components:
        raise ValueError(f"El número de componentes ({n_components}) excede la dimensionalidad de la matriz TF-IDF ({tfidf_matrix.shape[1]}).")
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    tfidf_reducida = svd.fit_transform(tfidf_matrix)
    print(f"Dimensiones reducidas: {tfidf_reducida.shape}")
    return tfidf_reducida

# 3. Recomendación con k-NN sobre las características reducidas
def knn_recommendation_svd(user_id, df, tfidf_reducida, top_n=5):
    knn = NearestNeighbors(n_neighbors=top_n + 1, metric='cosine')
    knn.fit(tfidf_reducida)

    # Seleccionar un artículo aleatorio comprado por el usuario
    usuario_articulos = df[df['user_id'] == user_id]['title'].values
    if len(usuario_articulos) == 0:
        print(f"El usuario '{user_id}' no tiene artículos asociados.")
        return []

    # Obtener el índice del primer artículo del usuario en la matriz reducida
    idx_usuario = df[df['title'] == usuario_articulos[0]].index[0] % len(tfidf_reducida)

    # Encontrar los índices de los productos más similares
    _, indices = knn.kneighbors([tfidf_reducida[idx_usuario]])

    # Concatenar ASIN y TITLE para mostrar en las recomendaciones
    recomendaciones = df.iloc[indices[0][1:]][['asin', 'title']].copy()
    recomendaciones['asin_title'] = recomendaciones['asin'] + " - " + recomendaciones['title']

    return recomendaciones['asin_title'].tolist()



# Función para graficar las recomendaciones SVD + k-NN
def plot_svd_knn_recommendations(recommendations):

    if not recommendations:
        print("No hay recomendaciones para graficar.")
        return
    
    # Crear el gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.barh(recommendations, range(len(recommendations), 0, -1), color='skyblue')
    plt.xlabel('Posición en el Ranking')
    plt.ylabel('Productos Recomendados')
    plt.title('Recomendaciones con SVD + k-NN')
    plt.gca().invert_yaxis()  # Invertir el eje Y para mostrar el más relevante arriba
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()


##########################################################################3
##Validacion##

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# 1. Preprocesamiento de texto con TF-IDF
def aplicar_tfidf(df, columna_texto):
    # Inicializar el vectorizador TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')

    # Aplicar TF-IDF a la columna de texto
    tfidf_matrix = tfidf.fit_transform(df[columna_texto].fillna(''))

    print(f"Matriz TF-IDF: {tfidf_matrix.shape}")
    return tfidf_matrix

# 2. Aplicar TF-IDF a los títulos de los artículos
tfidf_matrix = aplicar_tfidf(df, 'title')

# 3. Crear un modelo de recomendación basado en contenido utilizando TF-IDF y similitud coseno

"""
Esta función genera recomendaciones basadas en el contenido utilizando TF-IDF y similitud de coseno.
:param user_id: ID del usuario para el que se generarán las recomendaciones
:param df: DataFrame con los datos de los productos
:param tfidf_matrix: Matriz TF-IDF de los títulos de los productos
:param top_n: Número de recomendaciones a generar
"""

def content_based_recommendation_tfidf(user_id, df, tfidf_matrix, top_n=5):

    # Seleccionar un artículo aleatorio comprado por el usuario
    usuario_articulos = df[df['user_id'] == user_id]['title'].values
    if len(usuario_articulos) == 0:
        return []

    # Obtener el índice del primer artículo del usuario
    idx_usuario = df[df['title'] == usuario_articulos[0]].index[0]

    # Calcular la similitud coseno con TF-IDF
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx_usuario], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los índices de los productos más similares
    sim_indices = [i[0] for i in sim_scores[1:top_n+1]]

    return df['title'].iloc[sim_indices].tolist()
# 4. Realizar validación cruzada

"""
Esta función realiza validación cruzada para evaluar el modelo de filtrado basado en contenido utilizando TF-IDF.
:param tfidf_matrix: Matriz TF-IDF de los títulos de los productos
:param num_folds: Número de folds para la validación cruzada
"""

def validacion_cruzada_tfidf(df, tfidf_matrix, num_folds=5):
    """
    Realiza validación cruzada utilizando TF-IDF y un clasificador simple.

    :param df: DataFrame con los datos de entrada
    :param tfidf_matrix: Matriz TF-IDF
    :param num_folds: Número de folds para la validación cruzada
    :return: Diccionario con métricas promedio
    """
    # Crear el modelo de clasificación para validación
    X = tfidf_matrix
    y = df['rating'] >= df['rating'].mean()  # Etiquetas: 1 si la calificación es >= media, 0 si es menor

    # Configurar la validación cruzada
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Utilizar un clasificador simple para la validación
    modelo = LogisticRegression(max_iter=1000)

    # Definir los scorers para las métricas
    scorers = {
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    
    # Calcular las métricas para cada fold
    precision_scores = cross_val_score(modelo, X, y, cv=kf, scoring=scorers['precision'])
    recall_scores = cross_val_score(modelo, X, y, cv=kf, scoring=scorers['recall'])
    f1_scores = cross_val_score(modelo, X, y, cv=kf, scoring=scorers['f1'])

    # Calcular las métricas acumuladas
    precision_acumulada = np.cumsum(precision_scores) / np.arange(1, num_folds + 1)
    recall_acumulada = np.cumsum(recall_scores) / np.arange(1, num_folds + 1)
    f1_acumulada = np.cumsum(f1_scores) / np.arange(1, num_folds + 1)

    # Graficar las métricas acumuladas
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, num_folds + 1), precision_acumulada, marker='o', linestyle='-', label='Precisión', color='b')
    plt.plot(range(1, num_folds + 1), recall_acumulada, marker='o', linestyle='-', label='Recall', color='g')
    plt.plot(range(1, num_folds + 1), f1_acumulada, marker='o', linestyle='-', label='F1-score', color='r')

    plt.title('Precisión, Recall y F1-score Acumulados en la Validación Cruzada', fontsize=16)
    plt.xlabel('Fold', fontsize=14)
    plt.ylabel('Métricas Acumuladas', fontsize=14)
    plt.ylim([0, 1])
    plt.xticks(range(1, num_folds + 1))
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Retornar métricas promedio
    return {
        'precision': precision_scores.mean(),
        'recall': recall_scores.mean(),
        'f1': f1_scores.mean()
    }







###################################################################
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def iniciar_interfaz(df, tfidf_matrix, algo_knn):
    """
    Inicia la interfaz gráfica del sistema de recomendación.

    :param df: DataFrame con los datos del proyecto
    :param tfidf_matrix: Matriz TF-IDF calculada
    :param algo_knn: Modelo de filtrado colaborativo entrenado
    """
    # Crear ventana principal
    root = tk.Tk()
    root.title("Sistema de Recomendación")
    root.geometry("900x700")

    # Variables para almacenar selección de producto y usuario
    producto_var = tk.StringVar()
    usuario_var = tk.StringVar()

    # Función para seleccionar un producto
    def seleccionar_producto():
        seleccion = lista_productos.curselection()
        if not seleccion:
            messagebox.showerror("Error", "Debe seleccionar un producto.")
            return
        idx = seleccion[0]
        producto_seleccionado = df.iloc[idx]
        asin_producto = producto_seleccionado['asin']
        producto_nombre = producto_seleccionado['title']
        user_id = producto_seleccionado['user_id']

        # Actualizar variables
        producto_var.set(f"{producto_nombre} (ASIN: {asin_producto}, User ID: {user_id})")
        usuario_var.set(user_id)

        # Mostrar información seleccionada
        messagebox.showinfo("Producto Seleccionado", f"Seleccionaste: {producto_var.get()}")

    # Función para mostrar recomendaciones similares (Modelo 1)
    def mostrar_similares():
        if not producto_var.get():
            messagebox.showerror("Error", "Primero seleccione un producto.")
            return

        try:
            # Obtener el ASIN del producto seleccionado
            asin_example = df.loc[df['title'] == producto_var.get().split(" (ASIN:")[0], 'asin'].iloc[0]
            
            # Verificar que el ASIN exista en df_filtered
            if asin_example not in df_filtered['asin'].values:
                messagebox.showerror("Error", f"El ASIN '{asin_example}' no está en los datos filtrados.")
                return
            
            # Generar recomendaciones
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            recommended_products = get_content_recommendations(asin_example, df_filtered, cosine_sim, top_n=5)
            
            # Mostrar resultados
            resultados.delete(0, tk.END)
            resultados.insert(tk.END, "Modelo 1: Productos similares:")
            for i, (nombre, asin) in enumerate(recommended_products, start=1):
                resultados.insert(tk.END, f"{i}. {nombre} (ASIN: {asin})")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al buscar productos similares: {e}")

    #Función para realizar validación cruzada (Modelo 2)
    def validacion_cruzada():
        """
        Realiza validación cruzada y muestra los resultados en la interfaz.
        """
        try:
            # Aplicar TF-IDF y realizar validación cruzada
            tfidf_matrix_cruz = aplicar_tfidf(df, 'title')
            resultados_cruz = validacion_cruzada_tfidf(df, tfidf_matrix_cruz, num_folds=5)

            # Mostrar resultados en la interfaz
            resultados.delete(0, tk.END)
            resultados.insert(tk.END, "Modelo 2: Resultados de Validación Cruzada:")
            resultados.insert(tk.END, f"Precisión Promedio: {resultados_cruz['precision']:.2f}")
            resultados.insert(tk.END, f"Recall Promedio: {resultados_cruz['recall']:.2f}")
            resultados.insert(tk.END, f"F1-Score Promedio: {resultados_cruz['f1']:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error durante la validación cruzada: {e}")



    # Función para mostrar recomendaciones híbridas (Modelo 3)
    def mostrar_hibridas():
        if not usuario_var.get():
            messagebox.showerror("Error", "Primero seleccione un producto para asignar el usuario.")
            return

        hybrid_recs = hybrid_recommendation(usuario_var.get(), algo_knn, df, alpha=0.5, top_n=5)

        # Mostrar resultados en la lista de resultados
        resultados.delete(0, tk.END)
        resultados.insert(tk.END, "Modelo 3: Recomendaciones híbridas:")
        for i, (nombre, asin, score) in enumerate(hybrid_recs, start=1):
            resultados.insert(tk.END, f"{i}. {nombre} (ASIN: {asin}) - Puntuación: {score:.2f}")

        # Graficar recomendaciones híbridas
        plot_hybrid_recommendations(hybrid_recs)

    # Función para mostrar recomendaciones con SVD + k-NN (Modelo 4)
    def mostrar_svd_knn():
        if not usuario_var.get():
            messagebox.showerror("Error", "Primero seleccione un producto para asignar el usuario.")
            return

        try:
            tfidf_reducida = tfidf_matrix.toarray()
            recomendaciones_svd_knn = knn_recommendation_svd(usuario_var.get(), df, tfidf_reducida, top_n=5)

            # Mostrar resultados en la lista de resultados
            resultados.delete(0, tk.END)
            resultados.insert(tk.END, "Modelo 4: Recomendaciones con SVD + k-NN:")
            for i, rec in enumerate(recomendaciones_svd_knn, start=1):
                resultados.insert(tk.END, f"{i}. {rec}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error con SVD + k-NN: {e}")

    # Crear widgets de la interfaz
    tk.Label(root, text="Productos Disponibles", font=("Helvetica", 14)).pack(pady=10)

    # Crear marco para lista y barra desplazadora
    frame_lista = tk.Frame(root)
    frame_lista.pack(pady=10)

    # Crear la lista de productos
    lista_productos = tk.Listbox(frame_lista, selectmode=tk.SINGLE, width=50, height=15)

    # Crear barra desplazadora
    scrollbar = tk.Scrollbar(frame_lista, orient=tk.VERTICAL, command=lista_productos.yview)

    # Configurar barra desplazadora con la lista
    lista_productos.config(yscrollcommand=scrollbar.set)

    # Posicionar widgets en el marco
    lista_productos.grid(row=0, column=0)
    scrollbar.grid(row=0, column=1, sticky="ns")

    # Llenar lista de productos
    for i, producto in df.iterrows():
        lista_productos.insert(tk.END, f"{producto['title']}")

    # Botones para acciones
    tk.Button(root, text="Seleccionar Producto", command=seleccionar_producto).pack(pady=5)
    tk.Button(root, text="Mostrar Similares (Modelo 1)", command=mostrar_similares).pack(pady=5)
    tk.Button(root, text="Recomendaciones SVD + k-NN (Modelo 2)", command=mostrar_svd_knn).pack(pady=5)
    tk.Button(root, text="Recomendaciones Híbridas (Modelo 3)", command=mostrar_hibridas).pack(pady=5)
    tk.Button(root, text="Validación Cruzada (Modelo 4)", command=validacion_cruzada).pack(pady=5)

    # Área para mostrar resultados
    tk.Label(root, text="Resultados", font=("Helvetica", 14)).pack(pady=10)
    resultados = tk.Listbox(root, width=80, height=15)
    resultados.pack(pady=10)

    # Iniciar la ventana principal
    root.mainloop()

if __name__ == "__main__":
    iniciar_interfaz(df, tfidf_matrix, algo_knn)
