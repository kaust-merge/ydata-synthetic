#%%
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/aya/GIT/ydata-synthetic/src")
sys.path.append("/Users/aya/GIT/ydata-synthetic/src/ydata_synthetic")
sys.path.append("/Users/aya/GIT/ydata-synthetic/src/ydata_synthetic/synthesizers")
sys.path.append("/Users/aya/GIT/ydata-synthetic/src/ydata_synthetic/preprocessing/timeseries")
sys.path.append("/Users/aya/GIT/ydata-synthetic/src/ydata_synthetic/synthesizers.timeseries")


from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN

#%%

seq_len = 32        # Timesteps
n_seq = 8          # Features

hidden_dim = 24     # Hidden units for generator (GRU & LSTM).
                    # Also decides output_units for generator

gamma = 1           # Used for discriminator loss

noise_dim = 32      # Used by generator as a starter dimension
dim = 128           # UNUSED
batch_size = 128
log_step = 100

learning_rate = 5e-4



gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           noise_dim=noise_dim,
                           layers_dim=dim)

# %%
stock_data,scaler = processed_stock(path='../../data/CitiesOsEnergyAndTemperature.csv', seq_len=seq_len)
#stock_data = processed_stock(path='../../data/stock_data.csv', seq_len=seq_len)
print(len(stock_data),stock_data[0].shape)
# %%
if path.exists('CitiesOS.pkl'):
    synth = TimeGAN.load('CitiesOS.pkl')
else:
    synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_len, n_seq=n_seq, gamma=1)
    synth.train(stock_data, train_steps=5000)
    synth.save('CitiesOS.pkl')
# %%
synth_data = synth.sample(len(stock_data))
print(synth_data.shape)
# %%
cols = ['Electricity consumption(MW)',
'Electricity production(MW)',
'Electricity consumption player(MW)',
'Electricity consumption Industrial Zones(MW)',
'Electricity consumption Commercial Zones (MW)',
'Electricity consumption Residential Zones(MW)',
'Electricity consumption Office Zones(MW)',
'Temperature(°C)']
#cols = ['Open','High','Low','Close','Adj Close','Volume']
#Plotting some generated samples. Both Synthetic and Original data are still standartized with values between [0,1]
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 10))
axes=axes.flatten()

time = list(range(1,25))
obs = np.random.randint(len(stock_data))

for j, col in enumerate(cols):
    df = pd.DataFrame({'Real': stock_data[obs][:, j],
                   'Synthetic': synth_data[obs][:, j]})
    df.plot(ax=axes[j],
            title = col,
            secondary_y='Synthetic data', style=['-', '--'])
fig.tight_layout()
# %%
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError

#First implement a simple RNN model for prediction
def RNN_regression(units):
    opt = Adam(name='AdamOpt')
    loss = MeanAbsoluteError(name='MAE')
    model = Sequential()
    model.add(GRU(units=units,
                  name=f'RNN_1'))
    model.add(Dense(units=8,
                    activation='sigmoid',
                    name='OUT'))
    model.compile(optimizer=opt, loss=loss)
    return model
# %%
#Prepare the dataset for the regression model
stock_data=np.asarray(stock_data)
synth_data = synth_data[:len(stock_data)]
n_events = len(stock_data)

#Split data on train and test
idx = np.arange(n_events)
n_train = int(.75*n_events)
train_idx = idx[:n_train]
test_idx = idx[n_train:]

#Define the X for synthetic and real data
X_stock_train = stock_data[train_idx, :seq_len-1, :]
#Define the y for synthetic and real datasets
y_stock_train = stock_data[train_idx, -1, :]

X_stock_test = stock_data[test_idx, :seq_len-1, :]
y_stock_test = stock_data[test_idx, -1, :]



print('Real X train: {}'.format(X_stock_train.shape))
print('Real y train: {}'.format(y_stock_train.shape))

print('Real X test: {}'.format(X_stock_test.shape))
print('Real y test: {}'.format(y_stock_test.shape))
# %%
#Training the model with the real train data
ts_real = RNN_regression(12)
early_stopping = EarlyStopping(monitor='val_loss')

real_train = ts_real.fit(x=X_stock_train,
                          y=y_stock_train,
                          validation_data=(X_stock_test, y_stock_test),
                          epochs=100,
                          batch_size=128,
                          callbacks=[early_stopping])
# %%
#Training the model with the synthetic data
ts_synth = RNN_regression(12)
synth_train = ts_synth.fit(x=X_synth_train,
                          y=y_synth_train,
                          validation_data=(X_stock_test, y_stock_test),
                          epochs=100,
                          batch_size=128,
                          callbacks=[early_stopping])
# %%
#Summarize the metrics here as a pandas dataframe
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error
real_predictions = ts_real.predict(X_stock_test)
synth_predictions = ts_synth.predict(X_stock_test)

metrics_dict = {'r2': [r2_score(y_stock_test, real_predictions),
                       r2_score(y_stock_test, synth_predictions)],
                'MAE': [mean_absolute_error(y_stock_test, real_predictions),
                        mean_absolute_error(y_stock_test, synth_predictions)],
                'MRLE': [mean_squared_log_error(y_stock_test, real_predictions),
                         mean_squared_log_error(y_stock_test, synth_predictions)]}

results = pd.DataFrame(metrics_dict, index=['Real', 'Synthetic'])

results
# %%
real_predictions = ts_real.predict(X_stock_test[0:32])
real_predictions.shape
# %%
(X_stock_test[0:1]).shape
# %%
scaler.inverse_transform(X_stock_test[0:1][0])[0]
# %%
scaler.inverse_transform(real_predictions)[0]
# %%
real_predictions2 = scaler.inverse_transform(real_predictions)
y_stock_test2 = scaler.inverse_transform(y_stock_test[0:32])

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 10))
axes=axes.flatten()

time = list(range(1,32))
obs = np.random.randint(len(y_stock_test2))

for j, col in enumerate(cols):
    df = pd.DataFrame({'Real': y_stock_test2[:, j],
                   'Predicted': real_predictions2[:, j]})
    df.plot(ax=axes[j],
            title = col, 
            secondary_y='Synthetic data', style=['-', '--'])
fig.tight_layout()
# %%
