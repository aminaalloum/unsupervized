#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd


# In[24]:


df=pd.read_csv("./Downloads/CC GENERAL.csv")
df


# In[25]:


import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as shc


# In[26]:


df.info()


# In[27]:


df.isnull().sum()


# In[28]:


mean=round(df["MINIMUM_PAYMENTS"].mean())
mean


# In[29]:


df["MINIMUM_PAYMENTS"].fillna(mean,inplace=True)


# In[30]:


mean2=round(df["CREDIT_LIMIT"].mean())
mean2


# In[31]:


df["CREDIT_LIMIT"].fillna(mean2,inplace=True)


# In[32]:


df.isnull().sum()


# In[33]:


df=df.drop("CUST_ID",axis=1)
df


# In[34]:


from sklearn.cluster import AgglomerativeClustering
model=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
clut_labels=model.fit_predict(df)
clut_labels


# In[35]:


agglom=pd.DataFrame(clut_labels)
agglom.head(100)


# In[36]:


from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
fig =plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter (df['MINIMUM_PAYMENTS'], df["PAYMENTS"] , c= agglom[0], s=50)
plt.colorbar(scatter)


# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:


from sklearn.cluster import AgglomerativeClustering 
#mdl=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
#clust_labels=mdl.fit_predict(df)
#agglo=pd.DataFrame(clust_labels)
#agglo
plt.figure(figsize=(10,7))
dend=shc.dendrogram(shc.linkage(df,method="complete"))


# In[38]:


plt.figure(figsize=(10,7))
dend2=shc.dendrogram(shc.linkage(df,method="ward"))


# In[39]:


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4,random_state=0)
kmeans.fit(df)


# In[40]:


labels=pd.DataFrame(kmeans.labels_)
labels.head()


# In[41]:


kmeans.predict(df)
print(kmeans.cluster_centers_)


# In[110]:


import numpy as  np
k=range(1,16)
sosd=[]
for k in k:
    km = KMeans(n_clusters=k, init='k-means++', random_state= 0)  
    km=km.fit(df)  
    sosd.append(kmeans.inertia_)


# In[111]:


plt.plot(range(1,16),sosd,"bx-")
plt.xlabel("k")
plt.ylabel("sosd")
plt.title("Elbow method for optimal k")
plt.show()


# In[ ]:





# In[ ]:




