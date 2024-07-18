from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import f1_score, roc_curve, auc


def kmeans(df, n_clusters):
        X_scaled = StandardScaler().fit_transform(df)
        model = KMeans(n_clusters=n_clusters)
        model.fit(X_scaled)
        cluster_labels = model.predict(X_scaled)
        cluster_centers = model.cluster_centers_
        distances = [np.linalg.norm(x - cluster_centers[cluster]) for x, cluster in zip(X_scaled, cluster_labels)]

        percentile_threshold = 99.0
        threshold_distance = np.percentile(distances, percentile_threshold)

        anomalies = [X_scaled[i] for i, distance in enumerate(distances) if distance > threshold_distance]
        anomalies = np.asarray(anomalies, dtype=np.float32)

        colors = cm.nipy_spectral(cluster_labels.astype(float) / 3)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 2], marker='.', s=50, lw=0, alpha=0.7,c=colors, edgecolor='k')
        plt.scatter(anomalies[:, 0], anomalies[:, 2], color='red', marker='.', s=50, label='Anomalies')
        plt.xlabel('IRRADIATION'); plt.ylabel('DC_POWER'); plt.title('IRRADIATION vs Watts'); plt.legend()
        plt.subplot(1, 2, 2)
        plt.scatter(X_scaled[:, 1], X_scaled[:, 2], marker='.', s=50, lw=0, alpha=0.7,c=colors, edgecolor='k')
        plt.scatter(anomalies[:, 1], anomalies[:, 2], color='red', marker='.', s=50, label='Anomalies')
        plt.xlabel('MODULE_TEMPERATURE'); plt.ylabel('DC_POWER'); plt.title('MODULE_TEMPERATURE vs DC_POWER'); plt.legend()
        plt.tight_layout()
        plt.show()

        filasDf = df.shape[0]
        #print(" Forma das anomalias " + str(anomalies2.shape) + " Primeros 10 elementos: " + str(anomalies2[-10:]))ape[0]
        num_anomalias = round(filasDf * 0.01)
        f1_results = []; auc_results = []
        for i in range(10):
            dfanomalo = pd.DataFrame({'DC_POWER':np.random.randint(0.0,28,size=num_anomalias),
                        'MODULE_TEMPERATURE':np.random.randint(0.0,56,size=num_anomalias),
                        'IRRADIATION':np.random.randint(0.0,171,size=num_anomalias)})
            
            Y = np.concatenate([np.zeros(filasDf, dtype=int), np.ones(num_anomalias, dtype=int)])
            df_train = StandardScaler().fit_transform(pd.concat([df, dfanomalo], ignore_index=True))
            model.fit(df_train)
            cluster_labels = model.predict(df_train)
            cluster_centers = model.cluster_centers_
            distances = [np.linalg.norm(x - cluster_centers[cluster]) for x, cluster in zip(df_train, cluster_labels)]
            threshold_distance = np.percentile(distances, percentile_threshold)
            #Y_pred = model.predict(df_train)
            Y_pred = np.array(np.where((distances > threshold_distance), 1, 0))
            #Y_pred[Y_pred == 1] = 0; Y_pred[Y_pred == -1] = 1
            f1_results.append(f1_score(Y, Y_pred))
            fpr, tpr, thresholds = roc_curve(Y, Y_pred)
            auc_results.append(auc(fpr,tpr))

        f1_results = np.array(f1_results); auc_results = np.array(auc_results)
        print("media F1: " + str(np.mean(f1_results)) + " std: " + str(np.std(f1_results)))
        print("media AUC: " + str(np.mean(auc_results)) + " std: " + str(np.std(auc_results)))


def load_data(weather_path, generation_path):
    weather_data = pd.read_csv(weather_path)
    generation_data = pd.read_csv(generation_path)
    generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'],format = '%d-%m-%Y %H:%M')
    weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
    df_solar = pd.merge(generation_data.drop(columns = ['PLANT_ID']), weather_data.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
    return df_solar.drop(columns = ['DATE_TIME', 'SOURCE_KEY', 'DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'AC_POWER'])

if __name__ == '__main__':
    df = load_data("./Plant_1_Weather_Sensor_Data.csv", "./Plant_1_Generation_Data.csv")
    # explorar hiperparametro?
    kmeans(df, 4)