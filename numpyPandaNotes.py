import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

#for panda
df = pd.read_csv('tempPandaTest.csv')

print(df)
print(df.shape)
print(df.head(3))
print(df.columns)
print(df.isnull().sum())
#df.dropna()  --- to reset indexing after this do...df = df.reset_index(drop=True)
#df2 = df.replace(['y', 'n'], [1, 0])
#if you want to turn an object into an array ....df2['Temperature'].values
print(df.describe())

#for numpy
x = df['Temperature'].values
mean = np.mean(x)
print(mean)

df4 = df[['Day', 'Temperature']].values

#for ploting
x = df['Day'].values
y = df['Temperature'].values

print(plt.scatter(x,y))








