import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from openpyxl import Workbook
from openpyxl.drawing.image import Image

def cargar_archivo(file):
    """
    Carga el archivo especificado y lo convierte en un DataFrame.

    :param file: Archivo subido.
    :return: DataFrame con los datos del archivo.
    """
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file, sheet_name='Sheet1')  # Cambiar 'Sheet1' si es necesario
    else:
        raise ValueError("El archivo debe ser .csv o .xlsx")
    
    return df

def generar_graficos(df):
    """
    Genera gráficos de barras, torta, histograma, regresión lineal, Random Forest y análisis de correlación.

    :param df: DataFrame con los datos.
    """
    # Gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    df['Club'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Distribución por Club')
    ax.set_xlabel('Club')
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)
    
    # Gráfico de torta
    fig, ax = plt.subplots(figsize=(8, 8))
    df['Nationality'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_title('Distribución por Nacionalidad')
    ax.set_ylabel('')
    st.pyplot(fig)
    
    # Histograma
    fig, ax = plt.subplots(figsize=(10, 6))
    df['Age'].hist(bins=10, ax=ax)
    ax.set_title('Distribución de Edad')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)
    
    # Gráfico de correlación entre edad y cantidad de goles
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='Age', y='Goals', data=df, ax=ax)
    ax.set_title('Correlación entre Edad y Cantidad de Goles')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Goles')
    st.pyplot(fig)
    
    # Gráfico de goles por posición
    fig, ax = plt.subplots(figsize=(10, 6))
    df.groupby('Position')['Goals'].sum().plot(kind='bar', ax=ax)
    ax.set_title('Goles por Posición')
    ax.set_xlabel('Posición')
    ax.set_ylabel('Goles')
    st.pyplot(fig)
    
    # Gráfico de distribución de la precisión de los disparos
    fig, ax = plt.subplots(figsize=(10, 6))
    df['Shooting accuracy %'].dropna().hist(bins=10, ax=ax)
    ax.set_title('Distribución de la Precisión de los Disparos')
    ax.set_xlabel('Precisión de Disparos (%)')
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)
    
    # Gráfico de distribución de asistencias por club
    fig, ax = plt.subplots(figsize=(10, 6))
    df.groupby('Club')['Assists'].sum().plot(kind='bar', ax=ax)
    ax.set_title('Distribución de Asistencias por Club')
    ax.set_xlabel('Club')
    ax.set_ylabel('Asistencias')
    st.pyplot(fig)
    
    # Gráfico de comparación de goles y asistencias por posición
    fig, ax = plt.subplots(figsize=(10, 6))
    df_grouped = df.groupby('Position').agg({'Goals': 'sum', 'Assists': 'sum'})
    df_grouped.plot(kind='bar', ax=ax)
    ax.set_title('Comparación de Goles y Asistencias por Posición')
    ax.set_xlabel('Posición')
    ax.set_ylabel('Cantidad')
    st.pyplot(fig)

    # Verificar si las columnas 'Height' y 'Weight' existen
    if 'Height' in df.columns and 'Weight' in df.columns:
        # Regresión Lineal
        df = df.dropna(subset=['Height', 'Weight'])
        X = df[['Height']]
        y = df['Weight']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Predicciones a futuro
        future_heights = np.linspace(df['Height'].min(), df['Height'].max() + 10, 100).reshape(-1, 1)
        future_weights = model.predict(future_heights)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X_test, y_test, color='red', label='Datos Reales')
        ax.plot(X_test, y_pred, color='blue', linewidth=2, label='Regresión Lineal')
        ax.plot(future_heights, future_weights, color='green', linestyle='--', label='Predicciones Futuras')
        ax.set_title('Regresión Lineal entre Altura y Peso')
        ax.set_xlabel('Altura')
        ax.set_ylabel('Peso')
        ax.legend()
        st.pyplot(fig)
        
        # Random Forest
        model_rf = RandomForestRegressor(n_estimators=100, random_state=0)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        
        # Predicciones a futuro
        future_weights_rf = model_rf.predict(future_heights)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X_test, y_test, color='red', label='Datos Reales')
        ax.scatter(X_test, y_pred_rf, color='green', label='Predicciones Random Forest')
        ax.plot(future_heights, future_weights_rf, color='blue', linestyle='--', label='Predicciones Futuras Random Forest')
        ax.set_title('Predicciones Random Forest')
        ax.set_xlabel('Altura')
        ax.set_ylabel('Peso')
        ax.legend()
        st.pyplot(fig)

def main():
    st.title('Análisis de Datos de Jugadores de Fútbol')
    
    uploaded_file = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        df = cargar_archivo(uploaded_file)
        st.write(df)
        
        if st.button('Generar Gráficos'):
            try:
                generar_graficos(df)
                st.success("Gráficos generados con éxito.")
            except Exception as e:
                st.error(f"Error al generar gráficos: {e}")

if __name__ == "__main__":
    main()
