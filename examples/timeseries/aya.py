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

from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error
import matplotlib.dates as md

#%%

seq_len = 100        # Timesteps
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
#%%
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

def RNN_regression(units):
    opt = Adam(name='AdamOpt')
    loss = MeanAbsoluteError(name='MAE')
    model = Sequential()
    model.add(GRU(units=units,
                  name=f'RNN_1'))
    model.add(Dense(units=800,
                    activation='sigmoid',
                    name='OUT'))
    model.compile(optimizer=opt, loss=loss)
    return model
# %%
PSIZE = 100
stock_data=np.asarray(stock_data)
n_events = len(stock_data) - PSIZE

#Split data on train and test
idx = np.arange(n_events)
n_train = int(.75*n_events)
train_idx = idx[:n_train-PSIZE]
test_idx = idx[n_train:-PSIZE]

#%%
stock_data.shape
#%%
X_stock_train = stock_data[train_idx, :, :]
y_stock_train = stock_data[train_idx+PSIZE, 0:PSIZE, :]

X_stock_test = stock_data[test_idx, :, :]
y_stock_test = stock_data[test_idx+PSIZE, 0:PSIZE, :]

print('Real X train: {}'.format(X_stock_train.shape))
print('Real y train: {}'.format(y_stock_train.shape))

print('Real X test: {}'.format(X_stock_test.shape))
print('Real y test: {}'.format(y_stock_test.shape))
#%%
y_stock_train_flatten = y_stock_train.reshape(y_stock_train.shape[0],y_stock_train.shape[-1]*PSIZE)
y_stock_test_flatten = y_stock_test.reshape(y_stock_test.shape[0],y_stock_test.shape[-1]*PSIZE)

print('y_stock_train_flatten: {}'.format(y_stock_train_flatten.shape))
print('y_stock_test_flatten: {}'.format(y_stock_test_flatten.shape))

# %%
ts_real = RNN_regression(12)
early_stopping = EarlyStopping(monitor='val_loss')

real_train = ts_real.fit(x=X_stock_train,
                          y=y_stock_train_flatten,
                          validation_data=(X_stock_test, y_stock_test_flatten),
                          epochs=500,
                          batch_size=128)

# %%

# %%
real_predictions = ts_real.predict(X_stock_test[100:101])
print(real_predictions.shape)
#%%
real_predictions_unflaten = real_predictions.reshape(PSIZE,8)
print(real_predictions_unflaten.shape)

#%%
real_predictions2 = scaler.inverse_transform(real_predictions_unflaten)
tmpPred = np.array(real_predictions2)    
print(tmpPred.shape)

# %%
tmpTRUE = y_stock_test[100:101]
tmpTRUE_unfl = tmpTRUE.reshape(PSIZE,8)

tmpTrue2 = []
for x in tmpTRUE:
    real_predictions2 = scaler.inverse_transform(x)
    tmpTrue2.append(real_predictions2)
tmpTrue2 = np.array(tmpTrue2)    
tmpTrue2 = tmpTrue2.reshape(PSIZE,8)

print(tmpTrue2.shape)

#%%
print('tmpTrue2: {}'.format(tmpTrue2.shape))
print('tmpPred: {}'.format(tmpPred.shape))


#%%
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 10))
axes=axes.flatten()

for j, col in enumerate(cols):
    df = pd.DataFrame({'Real': tmpTrue2[:, j],
                   'Predicted': tmpPred[:, j]})
    df.plot(ax=axes[j],
            title = col, 
            secondary_y='Synthetic data', style=['-', '--'])
fig.tight_layout()
# %%
metrics_dict = {'r2': [r2_score(tmpTRUE_unfl, real_predictions_unflaten)],
                'MAE': [mean_absolute_error(tmpTRUE_unfl, real_predictions_unflaten)],
                'MRLE': [mean_squared_log_error(tmpTRUE_unfl, real_predictions_unflaten)]}

results = pd.DataFrame(metrics_dict, index=['Real', 'Synthetic'])

results
# %%

model = Sequential()
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(300, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(800, activation = "linear"))

model.compile(loss='mse', optimizer='adam')
# %%
model_train = model.fit(x=X_stock_train,
                          y=y_stock_train_flatten,
                          validation_data=(X_stock_test, y_stock_test_flatten),
                          epochs=100,
                          batch_size=128)
# %%
indexA = 250
indexB = indexA+1

real_predictions = model.predict(X_stock_train[indexA:indexB])
print(real_predictions.shape)
real_predictions_unflaten = real_predictions.reshape(PSIZE,8)
print(real_predictions_unflaten.shape)

real_predictions2 = scaler.inverse_transform(real_predictions_unflaten)
tmpPred = np.array(real_predictions2)    
print(tmpPred.shape)

tmpTRUE = y_stock_train[indexA:indexB]
tmpTRUE_unfl = tmpTRUE.reshape(PSIZE,8)

tmpTrue2 = []
for x in tmpTRUE:
    real_predictions2 = scaler.inverse_transform(x)
    tmpTrue2.append(real_predictions2)
tmpTrue2 = np.array(tmpTrue2)    
tmpTrue2 = tmpTrue2.reshape(PSIZE,8)

print(tmpTrue2.shape)

print('tmpTrue2: {}'.format(tmpTrue2.shape))
print('tmpPred: {}'.format(tmpPred.shape))


start = pd.Timestamp('2021-10-19T16:59:33')
end = pd.Timestamp('2021-10-20T16:59:33')
t = np.linspace(start.value, end.value, 100)
t = pd.to_datetime(t)


fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 10))
axes=axes.flatten()
xformatter = md.DateFormatter('%H:%M')

for a in fig.axes:
    a.xaxis.set_major_formatter(xformatter)

for j, col in enumerate(cols):
    tmpTrue2[:, j]
    df = pd.DataFrame(  data = {'Real': tmpTrue2[:, j], 'Predicted': tmpPred[:, j]} ,
                        index = t)
    df.plot(ax=axes[j],
            title = col, 
            secondary_y='Synthetic data', style=['-', '--'])
fig.tight_layout()
# %%
xx = pd.DataFrame(tmpTrue2[: , 0])
df = pd.DataFrame(index = t, data =tmpTrue2)
xx['date'] = t
xx.reset_index()
xx
# %%


# %%
