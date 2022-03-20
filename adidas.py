import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn import preprocessing
import warnings
import os


pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000

adidas_df = pd.read_csv('D:/Python/data/Mall/adidasKmeans.csv', sep = ';')
adidas_df = adidas_df.drop('ID', axis = 'columns')
adidas_df.head(5)
adidas_df.describe()

adidas_male = adidas_df[adidas_df['Gender'] == 'Male']
adidas_female = adidas_df[adidas_df['Gender'] == 'Female']

print(len(adidas_df))
print(len(adidas_male))
print(len(adidas_female))

##DistPlots
plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['TotalOrderCount', 'Age']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(adidas_df[x] , bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()

##GenderPlot
plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'Gender' , data = adidas_df)
plt.show()


##Age-OrderCount--MultPlots
plt.figure(1 , figsize = (15 , 7))
n = 0 
for x in ['Age' , 'TotalOrderCount']:
    for y in ['Age' , 'TotalOrderCount']:
        n += 1
        plt.subplot(2 , 2 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = adidas_df)
        plt.ylabel(y.split()[0] + ' ' + y.split()[1] if len(y.split()) > 1 else y )
plt.show()

##Age-OrderCount
plt.figure(1 , figsize = (15 , 7))
sns.regplot(x = 'Age' , y = 'TotalOrderCount' , data = adidas_df)
plt.show()


###########################################################################################################
###########################################################################################################
##KMeans-Elkan
###########################################################################################################
###########################################################################################################

##Age-TotalOrderCount
X1 = adidas_df[['Age' , 'TotalOrderCount']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n, init = 'k-means++', n_init = 10, max_iter = 300, 
                        tol = 0.0001, random_state = 111, algorithm = 'elkan') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)
    
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


algorithm = (KMeans(n_clusters = 4 ,init = 'k-means++', n_init = 10,max_iter = 300, 
                        tol = 0.0001,  random_state = 111  , algorithm = 'elkan') )
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_

h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation = 'nearest', 
           extent = (xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter(x = 'Age', y = 'TotalOrderCount', data = adidas_df, c = labels1, s = 200 )
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1], s = 300, c = 'red', alpha = 0.5)
plt.ylabel('TotalOrderCount'), plt.xlabel('Age')
plt.show()

##Age-ShoesTotalCount
X1 = adidas_df[['Age' , 'ShoesTotalCount']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n, init = 'k-means++', n_init = 10, max_iter = 300, 
                        tol = 0.0001, random_state = 111, algorithm = 'elkan') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)
    
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


algorithm = (KMeans(n_clusters = 4 ,init = 'k-means++', n_init = 10,max_iter = 300, 
                        tol = 0.0001,  random_state = 111  , algorithm = 'elkan') )
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_

h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation = 'nearest', 
           extent = (xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter(x = 'Age', y = 'ShoesTotalCount', data = adidas_df, c = labels1, s = 200 )
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1], s = 300, c = 'red', alpha = 0.5)
plt.ylabel('ShoesTotalCount'), plt.xlabel('Age')
plt.show()


##ShoesTotalPrice-Age-Female
adidas_female_shoes = adidas_df[(adidas_df['Gender'] == 'Female') & (adidas_df['ShoesTotalCount'] == 2) & (adidas_df['ShoesTotalPrice'] > 0)]
X1 = adidas_female_shoes[['ShoesTotalPrice' , 'Age']].iloc[: , :].values
meshGrid = adidas_female_shoes[['ShoesTotalPrice' , 'Age']].iloc[: , :].values
X1 = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X1))
X1.columns = ['ShoesTotalPrice' , 'Age']
X1 = X1[['ShoesTotalPrice' , 'Age']].iloc[: , :].values

inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n, init = 'k-means++', n_init = 10, max_iter = 300, 
                        tol = 0.0001, random_state = 111, algorithm = 'elkan') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)
    
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


algorithm = (KMeans(n_clusters = 6, init = 'k-means++', n_init = 10, max_iter = 300, 
                        tol = 0.0001,  random_state = 111  , algorithm = 'elkan'))
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_

h = 0.1
x_min, x_max = meshGrid[:, 0].min() - 1, meshGrid[:, 0].max() + 1
y_min, y_max = meshGrid[:, 1].min() - 1, meshGrid[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

plt.figure(1, figsize = (15,7))
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation = 'nearest', 
           extent = (xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin = 'lower')

plt.scatter(x = 'ShoesTotalPrice', y = 'Age', data = adidas_female_shoes, c = labels1, s = 25)
plt.xlabel('ShoesTotalPrice'), plt.ylabel('Age'), plt.title('Female - Age/ShoePrice')
plt.show()

##ShoesTotalPrice-Age-Male
adidas_male_shoes = adidas_df[(adidas_df['Gender'] == 'Male') & (adidas_df['ShoesTotalCount'] == 1) & (adidas_df['ShoesTotalPrice'] > 0)]
X1 = adidas_male_shoes[['ShoesTotalPrice' , 'Age']].iloc[: , :].values
X1 = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X1))
X1.columns = ['ShoesTotalPrice' , 'Age']
X1 = X1[['ShoesTotalPrice' , 'Age']].iloc[: , :].values

inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n, init = 'k-means++', n_init = 10, max_iter = 300, 
                        tol = 0.0001, random_state = 111, algorithm = 'elkan') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)
    
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


algorithm = (KMeans(n_clusters = 8, init = 'k-means++', n_init = 10, max_iter = 300, 
                        tol = 0.0001,  random_state = 111  , algorithm = 'elkan'))
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_

h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

plt.figure(1, figsize = (15,7))
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation = 'nearest', 
           extent = (xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin = 'lower')

plt.scatter(x = 'ShoesTotalPrice', y = 'Age', data = adidas_male_shoes, c = labels1, s = 10)
plt.xlabel('ShoesTotalPrice'), plt.ylabel('Age'), plt.title('Male - Age/ShoePrice')
plt.show()







