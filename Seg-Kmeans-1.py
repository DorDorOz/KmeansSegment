import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import warnings
import os


mall_df = pd.read_csv('D:/Python/data/Mall/mall.csv')
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000

mall_df = mall_df.drop('CustomerID', axis = 'columns')
mall_df.columns = ['Gender', 'Age', 'Income', 'Spending']


km_1_df = mall_df[['Age' , 'Spending']].iloc[: , :].values
km_1_inertia = []
for n in range(1 , 11):
    km_1 = KMeans(n_clusters = n, init = 'k-means++', n_init = 10, max_iter = 300, tol = 0.0001, random_state = 0, algorithm = 'elkan')
    km_1.fit(km_1_df)
    km_1_inertia.append(km_1.inertia_)

plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , km_1_inertia , 'o')
plt.plot(np.arange(1 , 11) , km_1_inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()



km_2_df = mall_df[['Age' , 'Income', 'Spending']].iloc[: , :].values
km_2_inertia = []
for n in range(1 , 11):
    km_2 = KMeans(n_clusters = n, init = 'k-means++', n_init = 10, max_iter = 300, tol = 0.0001, random_state = 0, algorithm = 'elkan')
    km_2.fit(km_1_df)
    km_2_inertia.append(km_2.inertia_)

plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , km_2_inertia , 'o')
plt.plot(np.arange(1 , 11) , km_2_inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()




km = KMeans(n_clusters = 8, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(mall_df[['Age' , 'Income', 'Spending']].iloc[: , :].values)

#df['gender'].replace(['Female','Male'], [0,1],inplace = True)













