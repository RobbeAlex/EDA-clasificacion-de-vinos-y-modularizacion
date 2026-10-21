import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def evaluar():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 1. Cargar los datos para obtener el set de prueba
    ruta_csv = os.path.join(base_dir, 'data', 'winequality-red.csv')
    if not os.path.exists(ruta_csv):
        print(f"Error: No se encuentra el archivo en {ruta_csv}")
        return

    df = pd.read_csv(ruta_csv, sep=',')
    df.columns = df.columns.str.strip()
    
    # Limpieza: eliminar columna Id si existe
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # 2. Dividir datos (usando el mismo random_state que en entrenamiento)
    X = df.drop('quality', axis=1)
    y = df['quality']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Cargar el modelo guardado
    ruta_modelo = os.path.join(base_dir, 'models', 'mejor_modelo.pkl')
    if not os.path.exists(ruta_modelo):
        print(f"Error: No se encontró el modelo guardado en {ruta_modelo}")
        return

    modelo = joblib.load(ruta_modelo)
    y_pred = modelo.predict(X_test)
    
    # 4. Mostrar Reporte de Clasificación (Sin advertencias)
    print("--- Reporte de Clasificación Final ---")
    # zero_division=0 elimina los avisos de métricas indefinidas en clases con pocas muestras
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 5. Visualizar Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión - Clasificación de Vinos")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluar()