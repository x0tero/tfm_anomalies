from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.metrics import f1_score, roc_curve, auc
from keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam





def create_model(dim_entrada):
    generator = Sequential()
    generator.add(layers.Dense(128, input_dim=dim_entrada))
    generator.add(layers.LeakyReLU(alpha=0.01))
    generator.add(layers.BatchNormalization(momentum=0.8))
    generator.add(layers.Dense(256))
    generator.add(layers.LeakyReLU(alpha=0.01))
    generator.add(layers.BatchNormalization(momentum=0.8))
    generator.add(layers.Dense(512))
    generator.add(layers.LeakyReLU(alpha=0.01))
    generator.add(layers.BatchNormalization(momentum=0.8))
    generator.add(layers.Dense(784, activation='tanh'))

    discriminator = Sequential()
    discriminator.add(layers.Dense(512, input_shape=img_shape))
    discriminator.add(layers.LeakyReLU(alpha=0.01))
    discriminator.add(layers.Dense(256))
    discriminator.add(layers.LeakyReLU(alpha=0.01))
    discriminator.add(layers.Dense(1, activation='sigmoid'))
    return (generator,discriminator)


def graficar(ecm_train, ecm_test, umbral, X_train, X_test):
    plt.figure(figsize=(12,4))
    plt.hist(ecm_train, bins=50)
    plt.xlabel("Error de reconstrucción (entrenamiento)")
    plt.ylabel("Número de datos")
    plt.axvline(umbral, color='r', linestyle='--')
    plt.legend(["Umbral"], loc="upper center")
    plt.show()
    #e_test = autoencoder.predict(X_test_scaled)

    #mse_test = np.mean(np.power(X_test_scaled - e_test, 2), axis=1)
    
    plt.figure(figsize=(12,4))
    plt.plot(range(1,X_train.shape[0]+1),ecm_train,'b.')
    plt.plot(range(X_train.shape[0]+1,X_train.shape[0]+X_test.shape[0]+1),ecm_test,'r.')
    plt.axhline(umbral, color='r', linestyle='--')
    plt.xlabel('Índice del dato')
    plt.ylabel('Error de reconstrucción')
    plt.legend(["Entrenamiento", "Test", "Umbral"], loc="upper left")
    plt.show()

def gan(df):

    df = df[['IRRADIATION', 'MODULE_TEMPERATURE', 'DC_POWER']]
    idx = int(len(df)*0.8)
    x_train_0 = df.head(idx); x_test_0 = df.tail(len(df)-idx)
    n_anomalias = round(len(x_test_0) * 0.01) # 1% anomalias en test
    f1_results = []; auc_results = []
    for i in range(4):
        dfanomalo = pd.DataFrame({'IRRADIATION':np.random.randint(0.0,171,size=n_anomalias),
                                'MODULE_TEMPERATURE':np.random.randint(0.0,56,size=n_anomalias),
                                'DC_POWER':np.random.randint(0.0,28,size=n_anomalias)
                                })
        x_test = pd.concat([x_test_0, dfanomalo])
        y_test = np.concatenate([np.ones(len(x_test_0)-n_anomalias, dtype=int), np.zeros(n_anomalias, dtype=int)])

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_0)
        x_test = scaler.fit_transform(x_test_0)

        autoencoder = create_model(x_train.shape[1:])
        
        history = autoencoder.fit(x_train, x_train, 
            epochs=10, 

            batch_size=32,
            shuffle=True,
            validation_data=(x_test, x_test)
        )

        # Reconstruccion conxunto adestramento
        reconTrain = autoencoder.predict(x_train)
        ecm_train = np.mean(np.power(x_train-reconTrain,2), axis=1) # Erro cuadratico medio
        umbral = np.percentile(ecm_train, 99.1) # Umbral estimado no 99.1 % en base a datos empiricos
        reconTest = autoencoder.predict(x_test)
        ecm_test = np.mean(np.power(x_test-reconTest,2), axis=1)
        y_pred = np.ones(y_test.shape); y_pred[ecm_test > umbral] = 0
        f1_results.append(f1_score(y_test, y_pred))
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc_results.append(auc(fpr, tpr))

    f1_results = np.array(f1_results); auc_results = np.array(auc_results)
    print("aucs: " + str(auc_results))
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
    gan(df)
