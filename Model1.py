#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# In[2]:


directory = "/Users/manuelbeltran/Documents/EDUCACION/JAVERIANA/SEMESTRE_3/pgrado"


# In[3]:


os.chdir('/Users/manuelbeltran/Documents/EDUCACION/JAVERIANA/SEMESTRE_3/pgrado')


# In[4]:


print(directory)


# In[5]:


model = tf.keras.models.load_model('Newsaved_model290923')
#model = tf.keras.models.load_model('saved_model')


# In[6]:


model.summary()


# In[7]:


directory = "/Users/manuelbeltran/Documents/EDUCACION/JAVERIANA/SEMESTRE_3/pgrado/"


# In[8]:


file_path = "Predictions.csv"


# In[9]:


os.unlink(file_path)


# In[10]:


df = pd.DataFrame(columns=[ 'imagen', 'Radius', 'Cent_X', 'Cent_y'])


# In[11]:


os.chdir(directory)
import cv2
import numpy as np
from tensorflow import keras

imagen="Mask_Circle_645_29.png"
# Load the image
img = cv2.imread(imagen)

input_shape = ( 1476, 1187,3)

img = np.expand_dims(img, axis=0)

model = model

# Make the prediction
prediction = model.predict(img)

# Print the results
print("Predicted radius: ", prediction[0][0],"  Predicted center_x: ", prediction[0][1]," Predicted center_y: ", prediction[0][2])


# In[12]:


df.loc[len(df)] = [imagen, prediction[0][0], prediction[0][1], prediction[0][2]]


# In[13]:


df.to_csv('Predictions.csv')


# In[ ]:





# In[15]:


pip pipreqs 


# In[ ]:




