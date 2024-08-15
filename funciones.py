# funciones.py

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import missingno as msno
import seaborn as sns
from collections import Counter
import locale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet



# IMAGEN

def Imagen(path_relativo: str, size: tuple[int,int], path_absoluto: None|str= None):
    if path_absoluto is None:
        current_dir = os.path.dirname(os.path.realpath('IA_0.jpg'))
        image_path = os.path.join(current_dir, path_relativo)
    else:
        image_path = path_absoluto
    
    try:
        image = mpimg.imread(image_path)
        plt.figure(figsize=size)
        plt.axis('off')
        image_plot = plt.imshow(image)
        plt.show()
    except FileNotFoundError:
        print(f"No se encontró el archivo: {image_path}")



# VERIFICAR TIPO DE DATOS.

def verificar_tipo_datos(df):
    mi_dict = {"nombre_campo": [], "tipo_datos": [], "no_nulos_%": [], "nulos_%": [], "nulos": []}

    for columna in df.columns:
        porcentaje_no_nulos = (df[columna].count() / len(df)) * 100
        mi_dict["nombre_campo"].append(columna)
        mi_dict["tipo_datos"].append(df[columna].apply(type).unique())
        mi_dict["no_nulos_%"].append(round(porcentaje_no_nulos, 2))
        mi_dict["nulos_%"].append(round(100-porcentaje_no_nulos, 2))
        mi_dict["nulos"].append(df[columna].isnull().sum())

    df_info = pd.DataFrame(mi_dict)
    return df_info


# INFORME DEL DATAFRAME.

def informe_dataframe(dataframe: pd.DataFrame) -> None:
    df = dataframe

    print('INFORME PRELIMINAR SOBRE CARACTERISTICAS DEL DATASET:\n')
    print(f'--Dimensiones del DataFrame--\nFilas: {df.shape[0]}\nColumnas: {df.shape[1]}\n')
    print(f'--Numero de datos--\n{df[df.isna() == False].count().sum()}\n')
    print(f'--Filas y Columnas--\nFilas: muestra de indices-------> {list(df.index)[0:5]}  -----> Desde {list(df.index)[0]}  Hasta {list(df.index)[-1]}\nColumnas: {list(df.columns)}\n')
    print(f'--Tipo de columnas--\n{df.dtypes}\n')
    columnas= df.columns

    if 'hours' in columnas and 'attributes' not in columnas:
        print(f'--Cantidad de registros duplicados--\n{df.drop(columns=["hours"]).duplicated().sum()}\n')
    elif 'hours' not in columnas and 'attributes' in columnas:
        print(f'--Cantidad de registros duplicados--\n{df.drop(columns=["attributes"]).duplicated().sum()}\n')
    elif 'hours' in columnas and 'attributes' in columnas:
        print(f'--Cantidad de registros duplicados--\n{df.drop(columns=["hours","attributes"]).duplicated().sum()}\n')
    else:
        print(f'--Cantidad de registros duplicados--\n{df.duplicated().sum()}\n')

    return ('~'*50)+'oo'+('~'*50)


# INFORME DE COLUMNAS.

def informe_columna(df: None|pd.DataFrame, columna: None|str) -> None:
    data = df[columna]
    
    print(f'INFORME PRELIMINAR SOBRE LA COLUMNA/FEATURE {columna}:\n')
    if data.dtype == 'object':
        print(f'--Numero de datos nulos--\n{data.isna().sum()}\n')
        print(f'--Cantidad de valores unicos en la columna--\n{data.describe()[1]}\n')

        if len(data.unique()) > 5:
            print(f'--Valores unicos en la columna (Primeros 5 valores)--\n{data.unique()[0:5]}\n')
        else:
            print(f'--Valores unicos en la columna--\n{data.unique()}\n')
            
        print(f'--Moda de la columna especificada--\nValor modal -----> {data.describe()[2]}\nFrecuencia acumulada ------> {data.describe()[3]}\n')
        print(f'--Distribucion de frecuencias (primeros valores con mayor cantidad de frecuencias)--\n {data.value_counts().nlargest(3)}\n')
        print('-'*120)
        print('-'*120)
    elif data.dtype == 'datetime64[ns]':
        print(f'--Numero de datos nulos--\n{data.isna().sum()}\n')
        print(f'--Cantidad de valores unicos en la columna--\n{data.describe()[1]}\n')
        print(f'--Valores unicos en la columna--\nEj: {data.dt.strftime("%Y-%m-%d").unique()[0:3]}  -----> Desde {list(data.dt.strftime("%Y-%m-%d").unique())[0]}  Hasta {list(data.dt.strftime("%Y-%m-%d").unique())[-1]}\n')
        print(f'--Moda de la columna especificada--\nValor modal -----> {data.describe()[2]}\nFrecuencia acumulada ------> {data.describe()[3]}\n')
        print(f'--Distribucion de frecuencias (primeros valores con mayor cantidad de frecuencias)--\n {data.value_counts().nlargest(3)}\n')
        print(f'--Valor maximo y minimo--\nMaximo: {data.max()}\nMinimo: {data.min()}\n')
        print('-'*120)
        print('-'*120)
    else:
        print(f'--Numero de datos nulos--\n{data.isna().sum()}\n')
        print(f'--Valores unicos en la columna--\nEj: {data.unique()[0:5]}  -----> Desde {list(data.unique())[0]}  Hasta {list(data.unique())[-1]}\n')
        print(f'--Estadisticos Principales de la columna--\nMedia: {round(data.mean(),2)}\nDesviacion Estandar: {round(data.std(),2)}\nPrimer cuartil: {data.quantile(0.25)}\nMediana: {data.median()}\nTercer cuartil: {data.quantile(0.75)}\n')
        print(f'--Valores extremos--\nValor maximo: {data.max()}\nValor minimo: {data.min()}\n')
        print(f'--Distribucion de frecuencias (primeros valores con mayor cantidad de frecuencias)--\n {data.value_counts().nlargest(3)}\n')
        print(f'--Valor maximo y minimo--\nMaximo: {data.max()}\nMinimo: {data.min()}\n')
        print('-'*120)
        print('-'*120)
    return


# VISUALIZACIÓN DE NULOS.

def visualizar_nulos(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    msno.matrix(df, ax=axes[0, 0])
    msno.bar(df, ax=axes[0, 1])  
    msno.heatmap(df, ax=axes[1, 0])
    msno.dendrogram(df, ax=axes[1, 1])

    axes[0, 0].set_title('Matrix Plot')
    axes[0, 1].set_title('Bar Plot')
    axes[1, 0].set_title('Heatmap')
    axes[1, 1].set_title('Dendrogram')

    plt.tight_layout()
    plt.show()


# VISUALIZACIÓN DE VALORES ATÍPICOS, EXTREMOS Y OUTLIERS.
#_Columnas Cualitativas._

def analisis_frecuencia_palabras(dataframe):
    columnas_cualitativas = dataframe.select_dtypes(include=['object']).columns.tolist()
    for columna in columnas_cualitativas:
        dataframe[columna] = dataframe[columna].astype(str)
    num_columnas = len(columnas_cualitativas) 
    if num_columnas == 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        axes = [ax]  
    else:
        fig, axes = plt.subplots(nrows=num_columnas, ncols=1, figsize=(10, num_columnas * 4))
    for ax, columna in zip(axes, columnas_cualitativas):
        texto_columna = dataframe[columna].str.lower().str.replace(r'[^a-zA-Z\s]', '').str.split()
        palabras_columna = [word for sublist in texto_columna for word in sublist]
        frecuencia_palabras = Counter(palabras_columna)
        palabras_mas_frecuentes = frecuencia_palabras.most_common(20)
        ax.bar(*zip(*palabras_mas_frecuentes))
        ax.set_xticklabels([word for word, _ in palabras_mas_frecuentes], rotation=45)
        ax.set_xlabel('Palabra')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Palabras más frecuentes en la columna {columna}')
    
    plt.tight_layout()
    plt.show()

#_Columnas Cuantitativas._

def boxplots_numericas(dataframe: pd.DataFrame | None) -> None:
    df_numericas = dataframe.select_dtypes(include=['number'])
    num_columnas = df_numericas.shape[1]
    fig, axes = plt.subplots(nrows=num_columnas, ncols=1, figsize=(10, num_columnas * 4))
    if num_columnas == 1:
        axes = [axes]
    for ax, columna in zip(axes, df_numericas.columns):
        sns.boxplot(x=df_numericas[columna], ax=ax)
        ax.set_title(f'Diagrama de caja para {columna}')
    plt.tight_layout()
    plt.show()


# VISUALIZACIÓN DE DUPLICADOS.

def hist_duplicados(dataframe):
    columnas = dataframe.columns

    if 'hours' in columnas and 'attributes' not in columnas:
        df_duplicates = dataframe.drop(columns=["hours"]).duplicated().value_counts()
    elif 'hours' not in columnas and 'attributes' in columnas:
        df_duplicates = dataframe.drop(columns=["attributes"]).duplicated().value_counts()
    elif 'hours' in columnas and 'attributes' in columnas:
        df_duplicates = dataframe.drop(columns=["hours","attributes"]).duplicated().value_counts()
    else:
        df_duplicates = dataframe.duplicated().value_counts()
        
    df_duplicates = pd.DataFrame({'Duplicados': df_duplicates.index, 'Frecuencia': df_duplicates.values})
    
    if not df_duplicates.empty: 
        num_columnas = 1
        fig, ax = plt.subplots(nrows=num_columnas, ncols=1, figsize=(10, num_columnas * 4))
        
        sns.barplot(x='Duplicados', y='Frecuencia', data=df_duplicates, ax=ax)
        ax.set_title('Conteo de Frecuencias de Registros Duplicados')
        ax.set_xlabel('Duplicados')
        ax.set_ylabel('Frecuencia')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No Duplicados', 'Duplicados'])
        
        plt.tight_layout()
        plt.show()
    else:
        print("No se encontraron registros duplicados.")


# MOSTRAR FILAS CON NULOS.

def mostrar_filas_con_nulos(df):
    indices_nulos = df[df.isnull().any(axis=1)].index.tolist()
    indices_muestra = np.random.choice(indices_nulos, size=min(5, len(indices_nulos)), replace=False)
    filas_seleccionadas = df.loc[indices_muestra]
    
    return filas_seleccionadas


# MOSTRAR FILAS CON VALORES DUPLICADOS.

def mostrar_filas_duplicadas(df):
    filas_duplicadas = df[df.duplicated(keep=False)]
    
    return filas_duplicadas


# BORRAR VALORES DUPLICADOS.

def eliminar_duplicados(df):
    filas_iniciales = len(df)
    df.drop_duplicates(inplace=True)
    filas_finales = len(df)
    filas_eliminadas = filas_iniciales - filas_finales
    
    return filas_eliminadas


# CAMBIAR EL TIPO DE DATOS DE LAS COLUMNAS LATITUD Y LONGITUD
def reemplazar_puntos_por_comas(AEROPUERTOSFINAL):
    AEROPUERTOSFINAL['Latitude'] = AEROPUERTOSFINAL['Latitude'].astype(str).str.replace('.', ',')
    AEROPUERTOSFINAL['Longitude'] = AEROPUERTOSFINAL['Longitude'].astype(str).str.replace('.', ',')
  
    return AEROPUERTOSFINAL



#CREAR LISTA DE AEROPUERTOS ÚNICOS.

def obtener_aeropuertos_unicos(df, origen_aeropuerto, destino_aeropuerto, origen_oaci, destino_oaci, origen_localidad, origen_provincia, origen_pais, origen_continente, destino_localidad,	destino_provincia, destino_pais, destino_continente):
    origen_aeropuertos = df[[origen_aeropuerto, origen_oaci, origen_localidad, origen_provincia, origen_pais, origen_continente]].drop_duplicates()
    destino_aeropuertos = df[[destino_aeropuerto, destino_oaci, destino_localidad,	destino_provincia, destino_pais, destino_continente]].drop_duplicates()
    
    origen_aeropuertos.columns = ['AEROPUERTO', 'OACI', 'LOCALIDAD', 'PROVINCIA', 'PAIS', 'CONTINENTE']
    destino_aeropuertos.columns = ['AEROPUERTO', 'OACI', 'LOCALIDAD', 'PROVINCIA', 'PAIS', 'CONTINENTE']
    
    AEROPUERTOSUNICOS = pd.concat([origen_aeropuertos, destino_aeropuertos]).drop_duplicates().reset_index(drop=True)

    return AEROPUERTOSUNICOS



# OBTENER DATOS LATITUDE Y LONGIITUDE DE AEROPUERTOS.

def combinar_aeropuertos_con_datos(AEROPUERTOSUNICOS, AEROPUERTOSFILTRADOS,):
    resultados = pd.DataFrame(columns=['AEROPUERTO', 'ICAO', 'LONGITUD', 'LATITUD'])
    no_encontrados = []
    
    for idx1, fila_unico in AEROPUERTOSUNICOS.iterrows():
        encontrado = False
        for idx2, fila_filtrado in AEROPUERTOSFILTRADOS.iterrows():
            if fila_unico['OACI'].lower() == fila_filtrado['ICAO'].lower():
                new_row = pd.DataFrame({
                    'AEROPUERTO': [fila_unico['AEROPUERTO']],
                    'ICAO': [fila_filtrado['ICAO']],
                    'LONGITUD': [fila_filtrado['Longitude']],
                    'LATITUD': [fila_filtrado['Latitude']]
                })
                resultados = pd.concat([resultados, new_row], ignore_index=True)
                encontrado = True
                break
        if not encontrado:
            no_encontrados.append(fila_unico['AEROPUERTO'])
    
    return resultados, no_encontrados


# CREAR TABLA AEROPUERTOFILTRADOS.

def cargar_y_filtrar_aeropuertos(url):
    columns = ["AirportID", "Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude", "Altitude", "Timezone", "DST", "TzDatabaseTimeZone", "Type", "Source"]
    airports_df = pd.read_csv(url, header=None, names=columns)
    AEROPUERTOSFILTRADOS = airports_df[["Name", "IATA", "ICAO", "Latitude", "Longitude"]]

    return AEROPUERTOSFILTRADOS



# PROCESAR TABLAS AEROPUERTOS EN AEROPUERTOSFINAL

def procesar_aeropuertos(AEROPUERTOSFILTRADOS, AEROPUERTOSFALTANTES, AEROPUERTOSUNICOS, output_file='aeropuertosfinal.csv'):
    AEROPUERTOSFILTRADOS.columns = AEROPUERTOSFILTRADOS.columns.str.strip()
    AEROPUERTOSFALTANTES.columns = AEROPUERTOSFALTANTES.columns.str.strip()
    aeropuertos_filtrados = AEROPUERTOSFILTRADOS[['ICAO', 'Latitude', 'Longitude']]
    aeropuertos_faltantes = AEROPUERTOSFALTANTES[['ICAO', 'Latitude', 'Longitude']]
    aeropuertos_filtrados.rename(columns={'ICAO': 'OACI'}, inplace=True)
    aeropuertos_faltantes.rename(columns={'ICAO': 'OACI'}, inplace=True)
    aeropuertos_combinados = pd.concat([aeropuertos_filtrados, aeropuertos_faltantes], ignore_index=True)
    aeropuertos_final = AEROPUERTOSUNICOS.merge(aeropuertos_combinados, on='OACI', how='left')
    aeropuertos_final.to_csv(output_file, index=False)
    return aeropuertos_final


# ELIMINAR COLUMNAS DE DATABASEGENERAL

def eliminar_columnas(df, columnas_a_eliminar, output_file='databasegeneral.csv'):
    df_modificado = df.drop(columns=columnas_a_eliminar)
    df_modificado.to_csv(output_file, index=False)
    return df_modificado


# CREAR TABLA CALENDARIO

def crear_calendario(fecha_inicio, fecha_fin, nombre_archivo):    
    dias_semana = {
        'Monday': 'Lunes',
        'Tuesday': 'Martes',
        'Wednesday': 'Miércoles',
        'Thursday': 'Jueves',
        'Friday': 'Viernes',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }

    rango_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin)
    df_fechas = pd.DataFrame(rango_fechas, columns=['Fecha'])
    df_fechas['Día'] = df_fechas['Fecha'].dt.day_name().map(dias_semana)
    df_fechas['Número de Día'] = df_fechas['Fecha'].dt.day
    df_fechas['Mes'] = df_fechas['Fecha'].dt.month
    df_fechas['Año'] = df_fechas['Fecha'].dt.year
    df_fechas['Semana del Año'] = df_fechas['Fecha'].dt.isocalendar().week
    df_fechas['Trimestre'] = df_fechas['Fecha'].dt.quarter
    df_fechas['Día'] = df_fechas['Día'].replace({'Miércoles': 'Miercoles', 'Sábado': 'Sabado'})
    df_fechas['Quincena'] = df_fechas['Número de Día'].apply(lambda x: 1 if x < 15 else 2)
    df_fechas.to_csv(nombre_archivo, index=False)
    return df_fechas



# REGRESIÓN LINEAL

def regresion_lineal(DATABASEGENERAL):
    DATABASEGENERAL = DATABASEGENERAL.dropna()
    DATABASEGENERAL['fecha'] = pd.to_datetime(DATABASEGENERAL['fecha'])
    DATABASEGENERAL['dias'] = (DATABASEGENERAL['fecha'] - DATABASEGENERAL['fecha'].min()).dt.days

    X = DATABASEGENERAL[['dias']]
    y_pasajeros = DATABASEGENERAL['pasajeros']
    y_vuelos = DATABASEGENERAL['vuelos']

    X_train, X_test, y_ocupacion_train, y_ocupacion_test = train_test_split(X, y_pasajeros, test_size=0.2, random_state=42)
    _, _, y_vuelos_train, y_vuelos_test = train_test_split(X, y_vuelos, test_size=0.2, random_state=42)

    model_ocupacion = LinearRegression()
    model_ocupacion.fit(X_train, y_ocupacion_train)
    ocupacion_pred = model_ocupacion.predict(X_test)

    model_vuelos = LinearRegression()
    model_vuelos.fit(X_train, y_vuelos_train)
    vuelos_pred = model_vuelos.predict(X_test)

    r2_ocupacion = r2_score(y_ocupacion_test, ocupacion_pred)
    mse_ocupacion = mean_squared_error(y_ocupacion_test, ocupacion_pred)

    r2_vuelos = r2_score(y_vuelos_test, vuelos_pred)
    mse_vuelos = mean_squared_error(y_vuelos_test, vuelos_pred)

    print(f"Ocupación - R2: {r2_ocupacion}, MSE: {mse_ocupacion}")
    print(f"Crecimiento de vuelos - R2: {r2_vuelos}, MSE: {mse_vuelos}")

    future_date = pd.to_datetime('2024-12-31')
    days_future = (future_date - DATABASEGENERAL['fecha'].min()).days

    predicted_ocupacion = model_ocupacion.predict([[days_future]])
    predicted_vuelos = model_vuelos.predict([[days_future]])

    return predicted_ocupacion, predicted_vuelos


# REGRESIÓN POLINÓMICA

def regresion_polinomica(DATABASEGENERAL):
    DATABASEGENERAL = DATABASEGENERAL.dropna()
    DATABASEGENERAL['fecha'] = pd.to_datetime(DATABASEGENERAL['fecha'])
    DATABASEGENERAL['dias'] = (DATABASEGENERAL['fecha'] - DATABASEGENERAL['fecha'].min()).dt.days
    X = DATABASEGENERAL[['dias']]
    
    y_pasajeros = DATABASEGENERAL['pasajeros']
    X_train, X_test, y_train_pasajeros, y_test_pasajeros = train_test_split(X, y_pasajeros, test_size=0.2, random_state=42)
    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    model_poly = LinearRegression()
    model_poly.fit(X_poly_train, y_train_pasajeros)
    y_train_pred_pasajeros = model_poly.predict(X_poly_train)
    y_test_pred_pasajeros = model_poly.predict(X_poly_test)
    
    r2_train_pasajeros = r2_score(y_train_pasajeros, y_train_pred_pasajeros)
    mse_train_pasajeros = mean_squared_error(y_train_pasajeros, y_train_pred_pasajeros)
    r2_test_pasajeros = r2_score(y_test_pasajeros, y_test_pred_pasajeros)
    mse_test_pasajeros = mean_squared_error(y_test_pasajeros, y_test_pred_pasajeros)
    
    y_vuelos = DATABASEGENERAL['vuelos']
    X_train, X_test, y_train_vuelos, y_test_vuelos = train_test_split(X, y_vuelos, test_size=0.2, random_state=42)
    X_poly_train_vuelos = poly.fit_transform(X_train)
    X_poly_test_vuelos = poly.transform(X_test)
    model_poly_vuelos = LinearRegression()
    model_poly_vuelos.fit(X_poly_train_vuelos, y_train_vuelos)
    y_train_pred_vuelos = model_poly_vuelos.predict(X_poly_train_vuelos)
    y_test_pred_vuelos = model_poly_vuelos.predict(X_poly_test_vuelos)
    
    r2_train_vuelos = r2_score(y_train_vuelos, y_train_pred_vuelos)
    mse_train_vuelos = mean_squared_error(y_train_vuelos, y_train_pred_vuelos)
    r2_test_vuelos = r2_score(y_test_vuelos, y_test_pred_vuelos)
    mse_test_vuelos = mean_squared_error(y_test_vuelos, y_test_pred_vuelos)
    
    future_date = pd.to_datetime('2024-12-31')
    days_future = (future_date - DATABASEGENERAL['fecha'].min()).days
    days_future_poly = poly.transform([[days_future]])
    
    predicted_pasajeros = model_poly.predict(days_future_poly)
    
    predicted_vuelos = model_poly_vuelos.predict(days_future_poly)
    
    print(f"R2 (Entrenamiento) Pasajeros: {r2_train_pasajeros}, MSE (Entrenamiento) Pasajeros: {mse_train_pasajeros}")
    print(f"R2 (Prueba) Pasajeros: {r2_test_pasajeros}, MSE (Prueba) Pasajeros: {mse_test_pasajeros}")
    print(f"Predicción de pasajeros para el 31 de diciembre de 2024: {predicted_pasajeros[0]}")
    
    print(f"R2 (Entrenamiento) Vuelos: {r2_train_vuelos}, MSE (Entrenamiento) Vuelos: {mse_train_vuelos}")
    print(f"R2 (Prueba) Vuelos: {r2_test_vuelos}, MSE (Prueba) Vuelos: {mse_test_vuelos}")
    print(f"Predicción de vuelos para el 31 de diciembre de 2024: {predicted_vuelos[0]}")
    
    return {
        "predicted_pasajeros": predicted_pasajeros[0],
        "r2_train_pasajeros": r2_train_pasajeros,
        "mse_train_pasajeros": mse_train_pasajeros,
        "r2_test_pasajeros": r2_test_pasajeros,
        "mse_test_pasajeros": mse_test_pasajeros,
        "predicted_vuelos": predicted_vuelos[0],
        "r2_train_vuelos": r2_train_vuelos,
        "mse_train_vuelos": mse_train_vuelos,
        "r2_test_vuelos": r2_test_vuelos,
        "mse_test_vuelos": mse_test_vuelos
    }



# MODELO FBPROPHET

def predecir_pasajeros_prophet(DATABASEGENERAL, periods=8):
    DATABASEGENERAL['fecha'] = pd.to_datetime(DATABASEGENERAL['fecha'])
    df_prophet = DATABASEGENERAL[['fecha', 'pasajeros']].rename(columns={'fecha': 'ds', 'pasajeros': 'y'})
    df_prophet = df_prophet.resample('M', on='ds').sum().reset_index()
    model = Prophet()
    model.fit(df_prophet)
    future_dates = model.make_future_dataframe(periods=periods, freq='M')
    forecastp = model.predict(future_dates)
    resultado = forecastp[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    print(resultado)
    return resultado

def predecir_vuelos_prophet(DATABASEGENERAL, periods=8):
    DATABASEGENERAL['fecha'] = pd.to_datetime(DATABASEGENERAL['fecha'])
    df_prophet = DATABASEGENERAL[['fecha', 'vuelos']].rename(columns={'fecha': 'ds', 'vuelos': 'y'})
    df_prophet = df_prophet.resample('M', on='ds').sum().reset_index()
    model = Prophet()
    model.fit(df_prophet)
    future_dates = model.make_future_dataframe(periods=periods, freq='M')
    forecastp = model.predict(future_dates)
    resultado = forecastp[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    print(resultado)
    return resultado

def predecir_pasajeros_aerolineas(DATABASEGENERAL, periods=8):
    top_airlines = [
    'Aerolíneas Argentinas', 'American Airlines', 'Andes Líneas Aéreas', 'Avianca',
    'Copa Airlines', 'Flybondi', 'Gol Transportes Aéreos', 'JetSMART Airlines',
    'LATAM', 'Sky Airline'
    ]
    DATABASEGENERAL['fecha'] = pd.to_datetime(DATABASEGENERAL['fecha'])
    df_top_airlines = DATABASEGENERAL[DATABASEGENERAL['aerolinea'].isin(top_airlines)]
    forecast_all_airlinesp = pd.DataFrame()
    for airline in top_airlines:
        df_airline = df_top_airlines[df_top_airlines['aerolinea'] == airline]
        df_prophet = df_airline[['fecha', 'pasajeros']].rename(columns={'fecha': 'ds', 'pasajeros': 'y'})
        df_prophet = df_prophet.resample('M', on='ds').sum().reset_index()
        model = Prophet()
        model.fit(df_prophet)
        future_dates = model.make_future_dataframe(periods=periods, freq='M')  # Predice hasta la fecha especificada
        forecast = model.predict(future_dates)
        forecast['aerolinea'] = airline
        forecast_all_airlinesp = pd.concat([forecast_all_airlinesp, forecast[['ds', 'aerolinea', 'yhat', 'yhat_lower', 'yhat_upper']]])
    return forecast_all_airlinesp

def predecir_vuelos_aerolineas(DATABASEGENERAL, periods=8):
    top_airlines = [
    'Aerolíneas Argentinas', 'American Airlines', 'Andes Líneas Aéreas', 'Avianca',
    'Copa Airlines', 'Flybondi', 'Gol Transportes Aéreos', 'JetSMART Airlines',
    'LATAM', 'Sky Airline'
    ]
    DATABASEGENERAL['fecha'] = pd.to_datetime(DATABASEGENERAL['fecha'])
    df_top_airlines = DATABASEGENERAL[DATABASEGENERAL['aerolinea'].isin(top_airlines)]
    forecast_all_airlinesp = pd.DataFrame()
    for airline in top_airlines:
        df_airline = df_top_airlines[df_top_airlines['aerolinea'] == airline]
        df_prophet = df_airline[['fecha', 'vuelos']].rename(columns={'fecha': 'ds', 'vuelos': 'y'})
        df_prophet = df_prophet.resample('M', on='ds').sum().reset_index()
        model = Prophet()
        model.fit(df_prophet)
        future_dates = model.make_future_dataframe(periods=periods, freq='M')  # Predice hasta la fecha especificada
        forecast = model.predict(future_dates)
        forecast['aerolinea'] = airline
        forecast_all_airlinesp = pd.concat([forecast_all_airlinesp, forecast[['ds', 'aerolinea', 'yhat', 'yhat_lower', 'yhat_upper']]])
    return forecast_all_airlinesp

