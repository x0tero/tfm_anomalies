from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
#from tf.keras import models, layers
from sklearn.metrics import f1_score, roc_curve, auc



def isolationForest(df, contamination):
    X_scaled = StandardScaler().fit_transform(df)
    model = IsolationForest(contamination=contamination)
    model.fit(X_scaled)
    outliers = model.predict(X_scaled)
    graficarAnomalias(df, outliers)

    # Nos inventamos un 1% de datos anomalos y vemos si los clasifica bien
    # 68774 FILAS ten o dataframe total, 687 FILAS ten o dataframe anomalo
    filasDf = df.shape[0]
    num_anomalias = round(filasDf * 0.01)
    f1_results = []; auc_results = []

    for i in range(10):
        dfanomalo = pd.DataFrame({'DC_POWER':np.random.randint(0.0,28,size=num_anomalias),
                    'MODULE_TEMPERATURE':np.random.randint(0.0,56,size=num_anomalias),
                    'IRRADIATION':np.random.randint(0.0,171,size=num_anomalias)})
        
        Y = np.concatenate([np.zeros(filasDf, dtype=int), np.ones(num_anomalias, dtype=int)])
        df_train = StandardScaler().fit_transform(pd.concat([df, dfanomalo], ignore_index=True))
        model.fit(df_train); Y_pred = model.predict(df_train)
        Y_pred[Y_pred == 1] = 0; Y_pred[Y_pred == -1] = 1
        f1_results.append(f1_score(Y, Y_pred))
        fpr, tpr, thresholds = roc_curve(Y, Y_pred)
        auc_results.append(auc(fpr,tpr))

    f1_results = np.array(f1_results); auc_results = np.array(auc_results)
    print("media F1: " + str(np.mean(f1_results)) + " std: " + str(np.std(f1_results)))
    print("media AUC: " + str(np.mean(auc_results)) + " std: " + str(np.std(auc_results)))


def graficarAnomalias(data, outliers):

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(data['IRRADIATION'], data['DC_POWER'], c=outliers, cmap='viridis')
    plt.scatter(data['IRRADIATION'][outliers == -1], data['DC_POWER'][outliers == -1], 
            edgecolors='r', facecolors='none', s=100, label='Outliers')
    plt.xlabel('IRRADIATION')
    plt.ylabel('DC_POWER')
    plt.title('IRRADIATION vs Watts')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(data['MODULE_TEMPERATURE'], data['DC_POWER'], c=outliers, cmap='viridis')
    plt.scatter(data['MODULE_TEMPERATURE'][outliers == -1], data['DC_POWER'][outliers == -1], 
            edgecolors='r', facecolors='none', s=100, label='Outliers')
    plt.xlabel('MODULE_TEMPERATURE')
    plt.ylabel('DC_POWER')
    plt.title('MODULE_TEMPERATURE vs DC_POWER')
    plt.legend()

    plt.tight_layout()
    plt.show()


def load_data(weather_path, generation_path):
    weather_data = pd.read_csv(weather_path)
    generation_data = pd.read_csv(generation_path)
    generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'],format = '%d-%m-%Y %H:%M')
    weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
    df_solar = pd.merge(generation_data.drop(columns = ['PLANT_ID']), weather_data.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
    return df_solar.drop(columns = ['DATE_TIME', 'SOURCE_KEY', 'DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'AC_POWER'])

if __name__ == '__main__':
    df = load_data("./Plant_1_Weather_Sensor_Data.csv", "./Plant_1_Generation_Data.csv")
    isolationForest(df, 0.01)