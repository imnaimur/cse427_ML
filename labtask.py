#------NUMPY section----#

import numpy as np
a = np.random.randint(1,100,20)
# print("before: mat a = ",a)
b = np.random.randint(1,100,20)
# print("before: mat b = ",b)

a = a.reshape(5,4)
b = b.reshape(5,4)
# print("after: mat a = ",a)
# print("after: mat b = ",b)

a = a*2
b = b*2
b = b.T

c = a @ b
# print(c[:,2:4])
# print(c[:,:])
# print(c.max())
# print(c.argmax())
d = c.flatten()
# print(d)

#------Pandas section----#

import pandas as pd

df = pd.read_csv("subject_scores.csv")
print(df.head(5))
print(df.info())
print(df.isnull().sum())
print(df.value_counts("Math"))
df.fillna( df['Physics'].mean(), inplace=True)
print(df)