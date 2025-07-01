import numpy as np
arr3 = np.array([1,2,3,4,5])
# print(arr3)
arr3 = np.zeros(5)
# print(arr3)
arr3 = np.zeros(shape=(2,3), dtype='int')
# print(arr3)

# matrix transpose
# print(arr3.T)

#Range function

#r = range(1,11,2.5) # doesnot allow float

"""
but in nupmy it allows float with arange function
"""
r = np.arange(1,11.5,2.5)
# print(r)

"""
identity matrinx with eye. and identity function
"""
mat1 = np.eye(5)
# print(mat1)

"""
element wise multiplication
"""
arr1 = np.array(
    [
        [10,20,30],
        [40,50,60],
    ]
)
arr2 = np.array(
    [
        [10,20,30],
        [40,50,60],
    ]
)
# print(arr1 * arr2)
"""
regular matrix multiplication
"""
arr1 = np.array(
    [
        [10,20,30],
        [40,50,60],
    ]
)
arr2 = np.array(
    [
        [10,20,30],
        [40,50,60],
        [70,80,90]
    ]
)
# print(arr1 @ arr2)
"""
numpy array broadcasting
"""
arr1 = np.array(
    [
        [10,20,30],
    ]
)
arr2 = np.array(
    [
        [10,20,30],
        [40,50,60],
        [70,80,90]
    ]
)
# print(arr1 + arr2)

"""
multidimention to 1D
flatten does it permanently by creating a new arr but if we need for temporary calculation we use ravel 
"""
arr1 = np.array(
    [
        [10,20,30],
        [40,50,60],
        [70,80,90]
    ]
)
arr2 = arr1.flatten()
# print(arr2)
# print(arr2.ravel())

import pandas as pd

list1 = [10,20,30,40,50]

data_series = pd.Series(list1)

# print(data_series)
# print(data_series.index)
# print(data_series.values)

loc = "./LAST12_BOOKUSDT_PRICE.csv"
df = pd.read_csv(loc)
# print(df)

"""
check null values count
descrive
"""
# print(df.isnull.sum)
# df.value_counts("varity")

import matplotlib.pyplot as plt

x = np.array([1,3,4,5,6])
y = np.array([10,20,30,40,50])
plt.plot(x,y)
plt.show()