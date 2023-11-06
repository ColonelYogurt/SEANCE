import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


df = pd.read_csv('DLR_0_1.txt',  delim_whitespace=True, header=None)
#print(mydata)

print(df.loc[0,:])

df.dropna()
print(np.sum(df))

powerdf = np.power(df, 2)
sumdf = np.sum(powerdf)
finalpowerdf = np.power(df, 0.5)
print(finalpowerdf)


