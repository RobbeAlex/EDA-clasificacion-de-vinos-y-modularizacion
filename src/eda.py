import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def cargar_datos(path):
    # Usamos sep=',' basado en las pruebas anteriores de error
    df = pd.read_csv(path, sep=',')
    df.columns = df.columns.str.strip()
    return df

def realizar_eda(df):
    print("--- Estadísticas Descriptivas ---")
    print(df.describe())
    
    # Seleccionar solo numéricas para evitar errores de tipo
    df_numeric = df.select_dtypes(include=['number'])
    if 'Id' in df_numeric.columns:
        df_numeric = df_numeric.drop('Id', axis=1)

    # 1. Matriz de Correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Matriz de Correlación")
    plt.tight_layout()
    plt.show()

    # 2. Histogramas de todas las variables
    df_numeric.hist(bins=15, figsize=(15, 10), layout=(4, 4))
    plt.suptitle("Distribución de Variables (Histogramas)")
    plt.tight_layout()
    plt.show()

    # 3. Boxplots para detectar outliers en relación a la calidad
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_numeric)
    plt.title("Boxplots de Variables Numéricas")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ruta = 'data/winequality-red.csv'
    if os.path.exists(ruta):
        datos = cargar_datos(ruta)
        realizar_eda(datos)
    else:
        print(f"Error: No se encuentra el archivo en {ruta}")