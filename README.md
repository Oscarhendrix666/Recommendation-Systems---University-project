# Sistema de Recomendación para Reseñas de Amazon: Gift Cards

Este proyecto implementa un sistema de recomendación basado en el conjunto de datos **Amazon Product Review: Gift Cards** en su versión 2023. Se han desarrollado múltiples enfoques, incluyendo filtrado colaborativo, filtrado basado en contenido y un modelo híbrido para generar recomendaciones personalizadas. El proyecto fue desarrollado para el ramo de Sistemas de Recomendaciones, impartido en la carrera de Ingeniería Civil en Informática y Telecomunicaciones en la Universidad Finis Terrae.

Fuente: Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using distantly-labeled reviews and fine-grained aspects. arXiv preprint. Recuperado de: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

## Tabla de Contenidos

- [Introducción](#introducción)
- [Características del Proyecto](#características-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Resultados](#resultados)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## Introducción

En este proyecto se desarrolló un sistema de recomendación utilizando el dataset "Amazon Product Review: Gift Cards". El enfoque híbrido combina filtrado colaborativo (k-NN) y filtrado basado en contenido (TF-IDF y similitud coseno), optimizando la relevancia y precisión de las recomendaciones.

Este sistema tiene como objetivo mejorar la experiencia de compra en plataformas de comercio electrónico, sugiriendo productos relevantes basados en las interacciones y preferencias de los usuarios.

## Características del Proyecto

- **Filtrado Colaborativo:** Utiliza k-NN para identificar productos similares a los que los usuarios han calificado positivamente.
- **Filtrado Basado en Contenido:** Emplea TF-IDF para analizar las descripciones y reseñas de productos, calculando similitudes textuales.
- **Modelo Híbrido:** Combina las calificaciones colaborativas y basadas en contenido para recomendaciones más precisas.
- **Validación Cruzada:** Se emplean métricas como MAE, RMSE, precisión, recall y F1-score para evaluar el rendimiento del sistema.

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instaladas las siguientes dependencias:

- Python 3.8 o superior
- Librerías:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `surprise`
  - `tkinter`

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/Oscarhendrix666/Recommendation-Systems---University-project.git
   cd Recommendation-Systems---University-project

2. Instala las dependencias:
   
   `pip install -r requirements.txt`

4. Transformación del dataset:
   El dataset original se descarga en formato JSONL. Es necesario convertirlo a CSV antes de usarlo en el proyecto. Para ello, ejecuta el siguiente script:

   `python JsonL_to_csv.py`

   Asegúrate de que el archivo JSONL esté en el directorio raíz del proyecto. El archivo CSV resultante se llamará Gift_Cards3.csv y será utilizado como entrada por el sistema de recomendación.

   Una vez convertido el dataset, continúa con el paso de uso.

## Uso
 1. Ejecuta el script principal:

    `python Proyecto.py`

 2. Interactúa con la interfaz gráfica para seleccionar productos, generar recomendaciones y visualizar resultados.

## Estructura del Proyecto

   `Proyecto.py` : Contiene toda la lógica para implementar los modelos de recomendación, realizar validaciones y generar visualizaciones.
   
   `JsonL_to_csv.py` : Script para convertir el archivo de entrada de formato JSONL a CSV.
   
   Datos: El dataset Gift_Cards.csv con las reseñas y calificaciones, generado a partir del script de conversión.
   
   Gráficos: Se generan automáticamente para mostrar métricas de evaluación y recomendaciones.

## Resultados

  Los principales resultados incluyen:
  
  Rendimiento del Filtrado Colaborativo:
  
  RMSE: 0.35
  
  MAE: 0.32
  
  Rendimiento del Modelo Híbrido: Combinación efectiva de los métodos con visualización clara de las recomendaciones.

## Contribuciones
Este proyecto fue desarrollado por:
- `Martín Azócar`
- `Sebastián Bustamante`
- `Oscar Horta`
- `Bastián Soto`
  
Profesor: Jorge Bozo

Ayudante: Ernesto Starck
  
## Licencia
  Este proyecto está bajo la licencia MIT.
