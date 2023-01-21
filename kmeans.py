#!/usr/bin/env python
# coding: utf-8

# # Syed Muhammad Ali Akbar
# Student ID: 21034385
# (1st Part of Code)

# In[1]:


# initial imports
import pandas as pd
import wbgapi as wb


# In[222]:


# Indicators ids
# wb.series.info()


# In[223]:


# Indicators to be tested and analysed
# BX.GSR.GNFS.CD	 Total Exports USD
# BM.GSR.GNFS.CD	 Total Imports USD

indicators = [
    'BX.GSR.GNFS.CD',
    'BM.GSR.GNFS.CD'
]


# In[224]:


# countries chosen are South Asian countries. 
# AFG - Afghanistan
# BGD - Bangladesh
# BTN - Bhutan
# IND - India
# LKA - Sri Lanka
# MDV - Maldieves
# NPL - Nepal
# Pak - Pakistan

df = wb.data.DataFrame(indicators, economy = wb.region.members('SAS'), time=range(2000, 2020), numericTimeKeys=True)


# In[225]:


# import to suppress scientific representation of data and show it in full int or float formats
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df.head()


# In[226]:


# checking null
# df.isnull().sum()


# In[227]:


# filling all the null with 0
df.fillna(0, inplace= True)


# In[228]:


# checking null if they are replaced with zero after filling
# df.isnull().sum()


# In[229]:


# stacking the df to convert columns into rows and then unstacking to convert rows into columns
df = df.stack().unstack(1).reset_index()
df.head()


# In[230]:


df = df.rename(columns={
    'economy': 'Country', 
    'level_1': 'Year', 
    'BM.GSR.GNFS.CD': 'Total Imports USD',
    'BX.GSR.GNFS.CD': 'Total Exports USD'
})

df.head()


# In[231]:


# StandardScaler class import to instantiate it inside normalization function

from sklearn.preprocessing import StandardScaler

# defining normalization function
def scale(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


# In[232]:


# selecting the two columns i.e Total Imports and Total Exports for clustering and visualization
X = df.iloc[:, [2,3]].values


# In[233]:


X[1:5]


# In[234]:


scaled_X = scale(X)


# In[235]:


scaled_X[1:5]


# In[236]:


# importing KMeans and matplotlib in order to find out the optimal number of clusters through elbow function first.
# three clusters are optimal as shown in the graph

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

plt.figure(figsize=(8, 5), dpi=100)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 101)
    kmeans.fit(scaled_X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, 'o--')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[237]:


# instantiating KMeans again this time with optimal number of clusters. Calling fit_predict on it to find out labels.
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 101)
y_kmeans = kmeans.fit_predict(scaled_X)


# In[238]:


y_kmeans


# In[239]:


# We need to instantiate StandardScaler class again as the previous one was not available as that one is called inside of a
# function. We cannot call inverse_transform directly as fit_transform should be called first on this new instance.

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = scaler.inverse_transform(scaled_X)
X[1:5]


# In[240]:


# plotting the clusters.
plt.figure(dpi=100)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'cyan', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'brown', label = 'Cluster 3')
plt.title('Clusters in Terms of Imports and Exports')
plt.xlabel('Total Imports USD')
plt.ylabel('Total Exports USD')
plt.legend()
plt.savefig('fig1.png')
plt.show()


# In[241]:


# subsetting df to compare the values between all countries from 2015-2019
df_subset = df[df['Year'] > 2014]
df_subset.head()


# In[242]:


# importing plotly.express to visualize comparison.
# importin kaleido to save plotly graphs as static png
import plotly.express as px
import kaleido
fig = px.bar(df_subset, x = 'Year', y = 'Total Exports USD', color = 'Country', log_y=True,
             color_discrete_sequence=px.colors.qualitative.Antique, template='plotly_white')
fig.update_layout(barmode='group')
# fig.write_image("fig2.png")
fig.show()


# In[243]:


fig = px.bar(df_subset, x = 'Year', y = 'Total Imports USD', color = 'Country', log_y=True,
             color_discrete_sequence=px.colors.qualitative.Antique, template='plotly_white')
fig.update_layout(barmode='group')
# fig.write_image("fig3.png")
fig.show()


# In[244]:


fig = px.line(df_subset, x = 'Year', y = 'Total Exports USD', color = 'Country', log_y=True,
              color_discrete_sequence=px.colors.qualitative.Alphabet, template='plotly_white')
fig.update_layout(xaxis={'showgrid': False}, yaxis={'showgrid': False})
fig.update_traces(line={'width': 3})
# fig.write_image("fig4.png")
fig.show()


# In[245]:


fig = px.line(df_subset, x = 'Year', y = 'Total Imports USD', color = 'Country', log_y=True,
              color_discrete_sequence=px.colors.qualitative.Alphabet, template='plotly_white')
fig.update_layout(xaxis={'showgrid': False}, yaxis={'showgrid': False})
fig.update_traces(line={'width': 3})
# fig.write_image("fig5.png")
fig.show()


# In[ ]:





# In[ ]:




