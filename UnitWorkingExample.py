#!/usr/bin/env python
# coding: utf-8

# In[23]:



# In[34]:


from glob import glob
import numpy as np
import tensorflow as tf


# In[35]:


for i in glob("ludoHistory/*.npy"):
    game = np.load(i,allow_pickle=True)
    break


# In[36]:


#defining the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(16,)),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1)
])
loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# In[37]:


#converting the input from 4X4 to 1X16
#Right now it is only taking the position of the pawns
#later we will update this to take current dice value
def netInput(states):
    netInput = np.array((np.zeros((1,16))))
    for state in states:
        temp = state[0][0]
        for j in state[0][1:]:
            temp = np.concatenate((temp,j), axis=0)
        netInput = np.vstack((netInput,temp))
    return netInput


# In[38]:


netInputs = netInput(game)


# In[39]:


target = model
for _ in range(10):
    #selecting n(5) random inputs
    randIdx = np.random.randint(netInputs.shape[0],size=5)
    inputs = netInputs[randIdx]
    #predicting the input with the target netowrk
    #this values is not right have to update this for final model
    y = target.predict(inputs)
    #training the model 
    model.fit(inputs,y,epochs=1)
    #updating the target network after training on n samples
    target = model


# In[ ]:




