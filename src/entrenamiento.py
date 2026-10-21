import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def entrenar():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_csv = os.path.join(base_dir, 'data', 'winequality-red.csv')
    ruta_models_dir = os.path.join(base_dir, 'models')
    ruta_modelo = os.path.join(ruta_models_dir, 'mejor_modelo.pkl')

    # 1. Cargar los datos
    df = pd.read_csv(ruta_csv, sep=',')
    df.columns = df.columns.str.strip()
    
    # Limpieza: eliminar columna Id si existe para no afectar el modelo
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # 2. Dividir datos en entrenamiento y prueba
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("--- Iniciando Entrenamiento de Modelos ---")
    
    # 3. Entrenar al menos 2 modelos de clasificación
    modelos = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42)
    }
    
    mejor_accuracy = 0
    mejor_modelo = None
    nombre_mejor_modelo = ""

    for nombre, model in modelos.items():
        # Entrenar modelo
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 4. Calcular métricas (accuracy, precision, recall, F1)
        # Se usa average='weighted' porque es un problema multiclase
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\nResultados para {nombre}:")
        print(f"- Accuracy:  {acc:.4f}")
        print(f"- Precision: {prec:.4f}")
        print(f"- Recall:    {rec:.4f}")
        print(f"- F1-Score:  {f1:.4f}")

        # Lógica para guardar el mejor basado en Accuracy
        if acc > mejor_accuracy:
            mejor_accuracy = acc
            mejor_modelo = model
            nombre_mejor_modelo = nombre

    # 5. Guardar el mejor modelo
    if not os.path.exists(ruta_models_dir):
        os.makedirs(ruta_models_dir)
    
    joblib.dump(mejor_modelo, ruta_modelo)
    print(f"\n--- Proceso Finalizado ---")
    print(f"El mejor modelo fue '{nombre_mejor_modelo}' y ha sido guardado en {ruta_modelo}")

if __name__ == "__main__":
    entrenar()