#!/usr/bin/env python
# coding: utf-8

# ## K-Means Clustering

# **Overview**<br>
# <a href="https://archive.ics.uci.edu/ml/datasets/online+retail">Online retail is a transnational data set</a> which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
# 
# The steps are broadly:
# 1. Read and understand the data
# 2. Clean the data
# 3. Prepare the data for modelling
# 4. Modelling
# 5. Final analysis and reco

# # 1. Read and visualise the data

# In[404]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


# In[373]:


# read the dataset
retail_df = pd.read_csv("Online_Retail.csv", sep=",", encoding="ISO-8859-1", header=0)
retail_df.head()


# In[374]:


# basics of the df
retail_df.info()


# # 2. Clean the data

# In[375]:


# missing values
round(100*(retail_df.isnull().sum())/len(retail_df), 2)


# In[376]:


# drop all rows having missing values
retail_df = retail_df.dropna()
retail_df.shape


# In[377]:


retail_df.head()


# In[378]:


# new column: amount 
retail_df['amount'] = retail_df['Quantity']*retail_df['UnitPrice']
retail_df.head()


# # 3. Prepare the data for modelling

# - R (Recency): Number of days since last purchase
# - F (Frequency): Number of tracsactions
# - M (Monetary): Total amount of transactions (revenue contributed)

# In[379]:


# monetary
grouped_df = retail_df.groupby('CustomerID')['amount'].sum()
grouped_df = grouped_df.reset_index()
grouped_df.head()


# In[380]:


# frequency
frequency = retail_df.groupby('CustomerID')['InvoiceNo'].count()
frequency = frequency.reset_index()
frequency.columns = ['CustomerID', 'frequency']
frequency.head()


# In[381]:


# merge the two dfs
grouped_df = pd.merge(grouped_df, frequency, on='CustomerID', how='inner')
grouped_df.head()


# In[382]:


retail_df.head()


# In[383]:


# recency
# convert to datetime
retail_df['InvoiceDate'] = pd.to_datetime(retail_df['InvoiceDate'], 
                                          format='%d-%m-%Y %H:%M')


# In[384]:


retail_df.head()


# In[385]:


# compute the max date
max_date = max(retail_df['InvoiceDate'])
max_date


# In[386]:


# compute the diff
retail_df['diff'] = max_date - retail_df['InvoiceDate']
retail_df.head()


# In[387]:


# recency
last_purchase = retail_df.groupby('CustomerID')['diff'].min()
last_purchase = last_purchase.reset_index()
last_purchase.head()


# In[388]:


# merge
grouped_df = pd.merge(grouped_df, last_purchase, on='CustomerID', how='inner')
grouped_df.columns = ['CustomerID', 'amount', 'frequency', 'recency']
grouped_df.head()


# In[389]:


# number of days only
grouped_df['recency'] = grouped_df['recency'].dt.days
grouped_df.head()


# In[390]:


# 1. outlier treatment
plt.boxplot(grouped_df['recency'])


# In[391]:


# two types of outliers:
# - statistical
# - domain specific


# In[392]:


# removing (statistical) outliers
Q1 = grouped_df.amount.quantile(0.05)
Q3 = grouped_df.amount.quantile(0.95)
IQR = Q3 - Q1
grouped_df = grouped_df[(grouped_df.amount >= Q1 - 1.5*IQR) & (grouped_df.amount <= Q3 + 1.5*IQR)]

# outlier treatment for recency
Q1 = grouped_df.recency.quantile(0.05)
Q3 = grouped_df.recency.quantile(0.95)
IQR = Q3 - Q1
grouped_df = grouped_df[(grouped_df.recency >= Q1 - 1.5*IQR) & (grouped_df.recency <= Q3 + 1.5*IQR)]

# outlier treatment for frequency
Q1 = grouped_df.frequency.quantile(0.05)
Q3 = grouped_df.frequency.quantile(0.95)
IQR = Q3 - Q1
grouped_df = grouped_df[(grouped_df.frequency >= Q1 - 1.5*IQR) & (grouped_df.frequency <= Q3 + 1.5*IQR)]



# In[393]:


# 2. rescaling
rfm_df = grouped_df[['amount', 'frequency', 'recency']]

# instantiate
scaler = StandardScaler()

# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape


# In[394]:


rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['amount', 'frequency', 'recency']
rfm_df_scaled.head()


# # 4. Modelling

# In[395]:


# k-means with some arbitrary k
kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_df_scaled)


# In[396]:


kmeans.labels_


# In[397]:


# help(KMeans)


# ## Finding the Optimal Number of Clusters
# 
# ### SSD

# In[398]:


# elbow-curve/SSD
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
# ssd
plt.plot(ssd)


# ### Silhouette Analysis
# 
# $$\text{silhouette score}=\frac{p-q}{max(p,q)}$$
# 
# $p$ is the mean distance to the points in the nearest cluster that the data point is not a part of
# 
# $q$ is the mean intra-cluster distance to all the points in its own cluster.
# 
# * The value of the silhouette score range lies between -1 to 1. 
# 
# * A score closer to 1 indicates that the data point is very similar to other data points in the cluster, 
# 
# * A score closer to -1 indicates that the data point is not similar to the data points in its cluster.

# In[399]:


# silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    
    


# In[400]:


# final model with k=3
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(rfm_df_scaled)


# In[401]:


kmeans.labels_


# In[402]:


# assign the label
grouped_df['cluster_id'] = kmeans.labels_
grouped_df.head()


# In[403]:


# plot
sns.boxplot(x='cluster_id', y='amount', data=grouped_df)


# ## Hierarchical Clustering

# In[406]:


rfm_df_scaled.head()


# In[407]:


grouped_df.head()


# In[408]:


# single linkage
mergings = linkage(rfm_df_scaled, method="single", metric='euclidean')
dendrogram(mergings)
plt.show()


# In[409]:


# complete linkage
mergings = linkage(rfm_df_scaled, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()


# In[415]:


# 3 clusters
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_labels


# In[416]:


# assign cluster labels
grouped_df['cluster_labels'] = cluster_labels
grouped_df.head()


# In[419]:


# plots
sns.boxplot(x='cluster_labels', y='recency', data=grouped_df)


# In[420]:


# plots
sns.boxplot(x='cluster_labels', y='frequency', data=grouped_df)


# In[421]:


# plots
sns.boxplot(x='cluster_labels', y='amount', data=grouped_df)

