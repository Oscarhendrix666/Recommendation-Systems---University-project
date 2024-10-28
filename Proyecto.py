import os
import pandas as pd
from surprise import KNNBasic, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def find_file_local(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Nombre del archivo a buscar
file_name = 'Gift_Cards2.csv'

# Directorio actual donde se ejecuta el script
current_dir = os.getcwd()

# Buscar el archivo en el directorio actual
file_path = find_file_local(file_name, current_dir)

# Leer el archivo si se encuentra
if file_path:
    print(f"Archivo encontrado en: {file_path}")
    # Leer el archivo CSV con pandas y mostrar las primeras 10 filas
    df = pd.read_csv(file_path)
    print("Nombres de las columnas:", df.columns.tolist())
    print(df.head(10))

    # Filtrar el dataset para pruebas (opcional)
    min_interactions = 50
    user_counts = df['user_id'].value_counts()
    filtered_users = user_counts[user_counts >= min_interactions].index
    df = df[df['user_id'].isin(filtered_users)]

else:
    print(f"El archivo '{file_name}' no se encontró en la carpeta actual.")
    exit()

# 1. Preparar el dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'title', 'rating']], reader)

# 2. Dividir el dataset en entrenamiento y prueba
trainset, testset = train_test_split(data, test_size=0.3)

# 3. Definir el algoritmo de k-NN con vecinos limitados y similitud de productos
sim_options = {
    'name': 'cosine',
    'user_based': False  # Cambia a similitud basada en productos
}
algo_knn = KNNBasic(k=50, sim_options=sim_options)  # Limitar a 50 vecinos

# 4. Entrenar el modelo
algo_knn.fit(trainset)

# 5. Hacer predicciones y evaluar
predictions = algo_knn.test(testset)
print("k-NN Coseno - RMSE:", accuracy.rmse(predictions))

# 6. Calcular el MAE
y_true = [pred.r_ui for pred in predictions]  # Calificaciones reales
y_pred = [pred.est for pred in predictions]   # Calificaciones predichas
mae = mean_absolute_error(y_true, y_pred)
print("MAE:", mae)


from surprise import KNNBasic

# Probar diferentes valores de k
k_values = [10, 20, 30, 40, 50]
mae_values = []
rmse_values = []

for k in k_values:
    # Definir el modelo k-NN
    sim_options = {'name': 'cosine', 'user_based': False}
    algo_knn = KNNBasic(k=k, sim_options=sim_options)
    
    # Entrenar el modelo y hacer predicciones
    algo_knn.fit(trainset)
    predictions = algo_knn.test(testset)
    
    # Calcular MAE y RMSE
    y_true = [pred.r_ui for pred in predictions]
    y_pred = [pred.est for pred in predictions]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = accuracy.rmse(predictions, verbose=False)
    
    mae_values.append(mae)
    rmse_values.append(rmse)

# Graficar MAE y RMSE en función de k
plt.figure(figsize=(10, 6))
plt.plot(k_values, mae_values, marker='o', label='MAE')
plt.plot(k_values, rmse_values, marker='o', label='RMSE')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Error')
plt.title('MAE y RMSE para Diferentes Valores de k en Filtrado Colaborativo (k-NN)')
plt.legend()
plt.grid(True)
plt.show()


#aplicando TF-IDF

# Crear una nueva columna combinando el título y la reseña del producto
df['product_features'] = df['title'] + " " + df['text']

# Rellenar valores nulos (si los hay) en las características de los productos
df['product_features'] = df['product_features'].fillna('')

# Filtrar el DataFrame para productos con características válidas
df_filtered = df[df['product_features'].str.strip() != ''].reset_index(drop=True)

# Aplicar TF-IDF a las características combinadas de los productos
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_filtered['product_features'])

# Mostrar el tamaño de la matriz TF-IDF
print("Matriz TF-IDF:", tfidf_matrix.shape)

# Calcular la similitud coseno entre todos los productos
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Función para recomendar productos similares
def get_content_recommendations(asin, df_filtered, cosine_sim, top_n=5):
    try:
        # Obtener el índice del producto correspondiente en el DataFrame filtrado
        idx = df_filtered[df_filtered['asin'] == asin].index[0]
        
        # Obtener los puntajes de similitud de ese producto con todos los demás
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Ordenar productos por similitud
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Obtener los índices de los productos más similares
        sim_indices = [i[0] for i in sim_scores[1:top_n+1]]
        
        return df_filtered['title'].iloc[sim_indices]

    except IndexError:
        print("Error: Índice fuera de rango.")
        return []

# Ejemplo de uso con el DataFrame filtrado
asin_example = df_filtered['asin'].iloc[0]  # Usar el primer ASIN en el DataFrame filtrado
recommended_products = get_content_recommendations(asin_example, df_filtered, cosine_sim, top_n=5)
print(f"Productos similares a '{asin_example}':\n", recommended_products)