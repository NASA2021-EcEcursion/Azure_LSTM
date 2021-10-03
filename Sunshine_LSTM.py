#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense


# In[2]:


filename = "data.csv"
df = pd.read_csv(filename)
print(df.info())


# In[3]:


df


# In[4]:


df = df.drop(['YEAR', 'MO', 'DY','ALLSKY_SFC_UVA','ALLSKY_SFC_UVB','ALLSKY_SFC_SW_DWN','CLRSKY_SFC_SW_DWN','WS2M','ALLSKY_KT','ALLSKY_NKT','ALLSKY_SFC_LW_DWN','CLRSKY_SFC_PAR_TOT'], axis = 1)


# In[5]:


df


# In[6]:


allskypar = df['ALLSKY_SFC_PAR_TOT'].values
allskypar = allskypar.reshape((-1,1))
df.insert(0, 'INDEX', range(1, 1 + len(df)))


# In[7]:


split_percent = 0.99
split = int(split_percent*len(allskypar))

allskypar_train = allskypar[:split]
allskypar_test = allskypar[split:]

date_train = df['INDEX'][:split]
date_test = df['INDEX'][split:]

print(len(allskypar_train))
print(len(allskypar_test))


# In[8]:


look_back = 30

train_generator = TimeseriesGenerator(allskypar_train, allskypar_train, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(allskypar_test, allskypar_test, length=look_back, batch_size=1)


# In[16]:


model = Sequential()
model.add(
    LSTM(100,
        activation='relu',
        input_shape=(look_back,1))
)
model.add(Dense(2))
model.compile(optimizer='adam', loss='mse')

num_epochs = 25
model.fit(train_generator, epochs=num_epochs, verbose=1)


# In[18]:


print(test_generator)
prediction = model.predict(test_generator)

allskypar_train = allskypar_train.reshape((-1))
allskypar_test = allskypar_test.reshape((-1))
prediction = prediction.reshape((-1))
import plotly.graph_objects as go

trace1 = go.Scatter(
    x = date_train,
    y = allskypar_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = date_test,
    y = allskypar_test,
    mode='lines',
    name = 'Ground Truth'
)
layout = go.Layout(
    title = "LSTM: AllSkyPar",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "allskypar"}
)
fig = go.Figure(data=[trace3, trace2], layout=layout)
fig.show()


# In[19]:


model.save("ALLSKY_SFC_PAR_TOT_10020148.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('ALLSKY_SFC_PAR_TOT_10020148.tflite', 'wb') as f:
  f.write(tflite_model)


# In[20]:


allskypar = allskypar.reshape((-1))

def predict(num_prediction, model):
    prediction_list = allskypar[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = df['INDEX'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

num_prediction = 30
forecast = predict(num_prediction, model)


# In[21]:


print(forecast)

