from zipfile import ZipFile
import os
import pandas as pd
import keras
import matplotlib.pyplot as plt

uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall()
csv = "jena_climate_2009_2016.csv"

df = pd.read_csv(csv)

titles = [    "Pressure","Temperature","Temperature in Kelvin","Temperature (dew point)","Relative Humidity",
    "Saturation vapor pressure","Vapor pressure","Vapor pressure deficit","Specific humidity","Water vapor concentration","Airtight","Wind speed","Maximum wind speed",
    "Wind direction in degrees",
]


date_time_key = "Date Time"


split_fraction = float(input('please enter training set fraction ie 0.715:'))
total = len(df)
train_split = int(split_fraction * total)
step = 144
units= int(input('please enter number of units ie 32:'))

past = 720  # 5 days
future = 144 # 1 day
learning_rate = float(input('please enter learning rate ie 0.001:')) #0.001
batch_size = int(input('please enter batch size ie 256:')) #256
epochs = int(input('please enter number of  epochs ie 10:'))
n = int(input('please enter number of days you want to predict ie 5:'))

def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


print(
    "The selected parameters are:",
    ", ".join([titles[i] for i in [0, 1, 5, 7, 8, 10, 11]]),
)


selected_features = [feature_keys[i] for i in [0, 1, 5, 7, 8, 10, 11]]
features = df[selected_features]
features.index = df[date_time_key]
features.head()

features = normalize(features.values, train_split)
features = pd.DataFrame(features)
features.head()

train_data = features.loc[0 : train_split - 1]  # cogemos las primeras 71.5% de las filas para el train data
val_data = features.loc[train_split:]  # el resto para validar
start = past + future  # 6 dias 
end = start + train_split # dias mas el training 

x_train = train_data[[i for i in range(7)]].values # todas las columnas
y_train = features.iloc[start:end][[1]]  # la temperatura

sequence_length = int(past / step) # 1 dia dividido en 144 valores de temperatura

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)



x_end = len(val_data) - past - future # validation data - 6 dias

label_start = train_split + past + future # training set + 6 dias

x_val = val_data.iloc[:x_end][[i for i in range(7)]].values # todas las columnas
y_val = features.iloc[label_start:][[1]] # temperatura

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(units)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)


model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)

def plot_days(n):
    x = np.arange(0,144*n)
    real_values = y_val[0:144*n]
    predictions = model.predict(dataset_val)
    plt.plot(x,real_values)
    plt.plot(x,predictions[0:144*n])
    
plot_days(n)
