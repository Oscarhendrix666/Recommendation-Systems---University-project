import os
import json
import pandas as pd

# Obtener la ruta de la carpeta actual del script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Definir la ruta del archivo JSONL en la carpeta actual
file_name = 'Gift_Cards.jsonl'
file_path = os.path.join(current_directory, file_name)

# Verificar si el archivo existe antes de intentar abrirlo
if os.path.exists(file_path):
    try:
        # Leer el archivo línea por línea y convertir cada línea en un objeto JSON
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Convertir cada línea a un diccionario Python
                json_object = json.loads(line.strip())
                data.append(json_object)  # Agregar cada objeto a la lista de datos

        # Convertir la lista de diccionarios a un DataFrame de pandas
        df = pd.DataFrame(data)

        # Definir el nombre del archivo CSV de salida en la misma carpeta
        output_file = os.path.join(current_directory, 'Gift_Cards2.csv')

        # Exportar el DataFrame a un archivo CSV
        df.to_csv(output_file, index=False, encoding='utf-8')

        print(f"Archivo CSV '{output_file}' creado exitosamente en la misma carpeta del script.")
    except Exception as e:
        print(f"Se produjo un error al intentar abrir el archivo: {e}")
else:
    print(f"El archivo no se encontró en la ruta especificada: {file_path}. Verifica la ubicación y el nombre del archivo.")
