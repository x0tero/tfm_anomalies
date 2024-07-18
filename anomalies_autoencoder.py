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





def create_model(dim_entrada):
    capa_entrada = layers.Input(shape=dim_entrada)
    encoder = layers.Dense(2, activation='tanh')(capa_entrada)
    decoder = layers.Dense(1, activation='tanh')(encoder)
    encoder = layers.Dense(2, activation='tanh')(decoder)
    decoder = layers.Dense(3, activation='linear')(decoder)
    autoencoder = models.Model(inputs=capa_entrada, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


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

def autoEncoder(df):

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
    #graficar(ecm_train, ecm_test, umbral, X_train_scaled, X_test_scaled)

    #from sklearn.metrics import precision_recall_curve
    #precision, recall, umbral = precision_recall_curve(Y_test, ecm)
    """

    #Y_test = np.ones(y_test.shape)
    #Y_test[ecm_test > umbral] = -1

    #reconTrain = autoencoder.predict(df_train)
    #ecm_train = np.mean(np.power(df_train-reconTrain,2), axis=1) # Erro cuadratico medio
    #umbral = np.percentile(ecm_train, 99.9)
    #reconTest = autoencoder.predict(X_test_scaled)
    #ecm_test = np.mean(np.power(X_test_scaled-reconTest,2), axis=1)
    #Y_test = np.ones(y_test.shape)
    #Y_test[ecm_test > umbral] = -1


    #print("Y_test: " + str(Y_test) + "\n forma: " + str(Y_test.shape))
    #print("Y_test_scaled: " + str(Y_test_scaled) + "\n forma: " + str(Y_test_scaled.shape))
    #print("F1 = " + str(f1_score(Y_test, Y_test_scaled)))
    #graficar(ecm_train, ecm_test, umbral, X_train_scaled, X_test_scaled)

    
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

    """


    # Cosas extra (comentadas)
    #[ecm_test > umbral] = -1
    #from sklearn.metrics import precision_recall_curve
    #precision, recall, umbral = precision_recall_curve(Y_test_scaled, ecm)
    """



    
    filas_df = df.shape[0]
    num_anomalias = round(filas_df * 0.01)
    dfanomalo = pd.DataFrame({'IRRADIATION':np.random.randint(30,100,size=num_anomalias),
                'MODULE_TEMPERATURE':np.random.randint(0.0,56,size=num_anomalias),
                'DC_POWER':np.random.randint(0,10,size=num_anomalias)})
    df = pd.concat([df[['IRRADIATION', 'MODULE_TEMPERATURE', 'DC_POWER']], dfanomalo], ignore_index=True)
    x_train, x_test, y_train, y_test = train_test_split(df.to_numpy(), df[['DC_POWER']].to_numpy(), test_size=0.2, random_state=111)
    scaler = StandardScaler()
    X_train_a = scaler.fit_transform(x_train)
    X_test_a = scaler.fit_transform(x_test)
    autoencoder = create_model2(X_train_a.shape[1:])
    
    #for i in range(10): 
    
    #salidas = StandardScaler().fit_transform(pd.concat([df, dfanomalo], ignore_index=True)[['DC_POWER']])
    
    #Y_train_scaled = scaler.fit_transform(x_train)
    #Y_test_scaled = scaler.fit_transform(x_test)
    #X_test_a = StandardScaler().fit_transform(pd.concat([df, dfanomalo], ignore_index=True))
    Y_test_a = np.concatenate([np.ones(filas_df, dtype=int), np.zeros(num_anomalias, dtype=int)])

    history = autoencoder.fit(X_train_a, X_train_a, 
        epochs=10, 
        batch_size=32,
        shuffle=True,
        validation_data=(X_test_a, X_test_a))
    
    # Reconstruido adestramento
    reconTrain_a = autoencoder.predict(X_train_a)
    ecm_train_a = np.mean(np.power(X_train_a - reconTrain_a, 2), axis=1) # Erro cuadratico medio

    umbral = np.percentile(ecm_train_a, 99.0)

    # Reconstruido test
    reconTest_a = autoencoder.predict(X_test_a)
    #ecm_test = np.mean(np.power(X_test_scaled-reconTest,2), axis=1)
    ecm_test_a = np.mean(np.power(X_test_a - reconTest_a, 2), axis=1)
    Y_test_n = np.ones(Y_test_a.shape)
    print("ecm_test_a: " + str(ecm_test_a) + "Y_test_n: " + str(Y_test_n))
    print("ecm_test_a shape: " + str(ecm_test_a.shape) + "Y_test_n shape: " + str(Y_test_n.shape))

    #Y_test_n[ecm_test_a > umbral] = 0 # 0 son anomalias (pasan o umbral)


    #mse_test_a = np.mean(np.power(salidas - e_test_a, 2), axis=1)
    #Y_pred_a = np.array(np.where((mse_test_a == 1), 1, 0))
    #f1_results.append(f1_score(Y_test_a, Y_test_n))

    #print("Y_test: " + str(Y_test_a) + "\n forma: " + str(Y_test_a.shape))
    #print("Y_test_scaled: " + str(Y_test_n) + "\n forma: " + str(Y_test_n.shape))
    #print("F1 = " + str(f1_score(Y_test_a, Y_test_n)))


  

    
    plt.plot(umbral, precision[1:], label="Precision",linewidth=5)
    plt.plot(umbral, recall[1:], label="Recall",linewidth=5)
    plt.title('Precision y Recall para diferentes umbrales')
    plt.xlabel('Umbral')
    plt.ylabel('Precision/Recall')
    plt.legend()
    plt.show()
    

    


    mse_train = tf.keras.losses.mse(autoencoder.predict(X_train_scaled), X_train_scaled)
    umbral = np.max(mse_train)

    plt.figure(figsize=(12,4))
    plt.hist(mse_train, bins=50)
    plt.xlabel("Error de reconstrucción (entrenamiento)")
    plt.ylabel("Número de datos")
    plt.axvline(umbral, color='r', linestyle='--')
    plt.legend(["Umbral"], loc="upper center")
    plt.show()
    e_test = autoencoder.predict(X_test_scaled)

    mse_test = np.mean(np.power(X_test_scaled - e_test, 2), axis=1)
    plt.figure(figsize=(12,4))
    plt.plot(range(1,X_train_scaled.shape[0]+1),mse_train,'b.')
    plt.plot(range(X_train_scaled.shape[0]+1,X_train_scaled.shape[0]+X_test_scaled.shape[0]+1),mse_test,'r.')
    plt.axhline(umbral, color='r', linestyle='--')
    plt.xlabel('Índice del dato')
    plt.ylabel('Error de reconstrucción')
    plt.legend(["Entrenamiento", "Test", "Umbral"], loc="upper left")
    plt.show()

    filasDf = df.shape[0]
    num_anomalias = round(filasDf * 0.01)
    f1_results = []
    for i in range(10):
        dfanomalo = pd.DataFrame({'IRRADIATION':np.random.randint(30,100,size=num_anomalias),
                    'MODULE_TEMPERATURE':np.random.randint(0.0,56,size=num_anomalias),
                    'DC_POWER':np.random.randint(0,10,size=num_anomalias)})
        salidas = StandardScaler().fit_transform(pd.concat([df, dfanomalo], ignore_index=True)[['DC_POWER']])
        X_test_a = StandardScaler().fit_transform(pd.concat([df, dfanomalo], ignore_index=True).drop(columns=['DC_POWER']))
        Y_test_a = np.concatenate([np.zeros(filasDf, dtype=int), np.ones(num_anomalias, dtype=int)])
        e_test_a = autoencoder.predict(X_test_a)
        mse_test_a = np.mean(np.power(salidas - e_test_a, 2), axis=1)
        Y_pred_a = np.array(np.where((mse_test_a > umbral), 1, 0))
        f1_results.append(f1_score(Y_test_a, Y_pred_a))

    f1_results = np.array(f1_results)
    print(f1_results)
    mean = np.mean(f1_results); std = np.std(f1_results)
    print("media F1: " + str(mean) + " std: " + str(std))
"""


def load_data(weather_path, generation_path):
    weather_data = pd.read_csv(weather_path)
    generation_data = pd.read_csv(generation_path)
    generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'],format = '%d-%m-%Y %H:%M')
    weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
    df_solar = pd.merge(generation_data.drop(columns = ['PLANT_ID']), weather_data.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
    return df_solar.drop(columns = ['DATE_TIME', 'SOURCE_KEY', 'DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'AC_POWER'])


if __name__ == '__main__':
    df = load_data("./Plant_1_Weather_Sensor_Data.csv", "./Plant_1_Generation_Data.csv")
    autoEncoder(df)
    """


    filas_df = df.shape[0]
    num_anomalias = round(filas_df * 0.01)
    dfanomalo = pd.DataFrame({'IRRADIATION':np.random.randint(30,100,size=num_anomalias),
                'MODULE_TEMPERATURE':np.random.randint(0.0,56,size=num_anomalias),
                'DC_POWER':np.random.randint(0,10,size=num_anomalias)})
    df = pd.concat([df[['IRRADIATION', 'MODULE_TEMPERATURE', 'DC_POWER']], dfanomalo], ignore_index=True)
    x_train, x_test, y_train, y_test = train_test_split(df.to_numpy(), df[['DC_POWER']].to_numpy(), test_size=0.2, random_state=111)
    scaler = StandardScaler()
    X_train_a = scaler.fit_transform(x_train)
    X_test_a = scaler.fit_transform(x_test)
    autoencoder = create_model2(X_train_a.shape[1:])
    
    #for i in range(10): 
    
    #salidas = StandardScaler().fit_transform(pd.concat([df, dfanomalo], ignore_index=True)[['DC_POWER']])
    
    #Y_train_scaled = scaler.fit_transform(x_train)
    #Y_test_scaled = scaler.fit_transform(x_test)
    #X_test_a = StandardScaler().fit_transform(pd.concat([df, dfanomalo], ignore_index=True))
    Y_test_a = np.concatenate([np.ones(filas_df, dtype=int), np.zeros(num_anomalias, dtype=int)])
    history = autoencoder.fit(X_train_a, X_train_a, 
        epochs=10, 
        batch_size=32,
        shuffle=True,
        validation_data=(X_test_a, X_test_a))
    
    # Reconstruido adestramento
    reconTrain_a = autoencoder.predict(X_train_a)
    ecm_train_a = np.mean(np.power(X_train_a - reconTrain_a, 2), axis=1) # Erro cuadratico medio

    umbral = np.percentile(ecm_train_a, 99.0)

    # Reconstruido test
    reconTest_a = autoencoder.predict(X_test_a)
    #ecm_test = np.mean(np.power(X_test_scaled-reconTest,2), axis=1)
    ecm_test_a = np.mean(np.power(X_test_a - reconTest_a, 2), axis=1)
    Y_test_n = np.ones(Y_test_a.shape)
    print("ecm_test_a: " + str(ecm_test_a) + "Y_test_n: " + str(Y_test_n))
    print("ecm_test_a shape: " + str(ecm_test_a.shape) + "Y_test_n shape: " + str(Y_test_n.shape))
    """
