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

# matplotlib
import matplotlib.pyplot as plt

temperature_dhaka = np.array([25,34,21,45,28,6,43,18,7,2])
humidity_dhaka = np.array([28, 25,29,20, 26, 50, 19, 29, 52, 55])

plt.scatter(temperature_dhaka,humidity_dhaka,marker = "*")
plt.title("temperature_dhaka vs humidity_dhaka")
plt.show()

study_hours = np.array([2,3,4,4, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12])
marks1 = np.array([6, 10, 15, 20, 34, 44, 55, 60, 55, 67, 70, 80, 90, 99, 100])
plt.figure(figsize=(12,8))
plt.title("study_hours vs marks")
plt.plot(study_hours,marks1,linestyle = "solid")

plt.show()

subjects = np.array(['Maths', 'English', 'Science', 'Physics', 'Computer'])
marks2 = np.array([89, 90, 45, 78, 99])
plt.title("Subjects vs marks")
plt.barh(subjects,marks2)
plt.show()
plt.bar(subjects,marks2)
plt.show()

plt.subplot(4,1,1)
plt.scatter(temperature_dhaka,humidity_dhaka,marker = "*")

plt.subplot(4,1,2)
plt.plot(study_hours,marks1)

plt.subplot(4,1,3)
plt.barh(subjects,marks2)

plt.subplot(4,1,4)
plt.bar(subjects,marks2)

plt.show()