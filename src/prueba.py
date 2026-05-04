import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def evaluar():
    # Re-cargar datos para tener el set de prueba
    df = pd.read_csv('data/winequality-red.csv', sep=',')
    df.columns = df.columns.str.strip()
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    X = df.drop('quality', axis=1)
    y = df['quality']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Cargar modelo
    if not os.path.exists('models/mejor_modelo.pkl'):
        print("Error: No se encontró el modelo guardado.")
        return

    modelo = joblib.load('models/mejor_modelo.pkl')
    y_pred = modelo.predict(X_test)
    
    print("--- Reporte de Clasificación Final ---")
    print(classification_report(y_test, y_pred))
    
    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión - Clasificación de Vinos")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import os
    evaluar()