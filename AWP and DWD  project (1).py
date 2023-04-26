#!/usr/bin/env python
# coding: utf-8

# import pandas as pd
# import numpy as np
# import seaborn as sns
# sns.set()
# sns.set(style='darkgrid')
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")

# 
# # World University Rankings 2023
# 
# 
# # About Dataset
# 
# Discover the world's top universities. Explore the QS World University Rankings® 2023 based on 8 key ranking indicators.
# This year’s QS World University Rankings include almost 1,500 institutions from around the world. It’s not just iconic 
# institutions that take the top spots: this year’s highest rankings include universities from diverse locations across Europe, Asia and North America.

# In[32]:


df=pd.read_csv("world university ranking.csv")
df


# In[33]:


df.head()


# In[34]:


df.info()


# In[35]:


df.tail()


# In[36]:


df.describe()


# In[37]:


df.shape


# In[38]:


df.ndim


# In[57]:


sns.lineplot(x="ar score",y="er score",data=df[:500]);


# In[44]:


sns.barplot(x="fsr score",y="cpf score",data=df[:100]);


# In[59]:


sns.distplot(df['ifr score']);


# In[61]:


sns.distplot(df['isr score']);


# In[75]:


sns.boxplot(df["ger score"],orient='vertical')


# In[73]:


sns.violinplot(df["ger score"],orient='vertical',color="purple")


# In[88]:


sns.relplot(x="ar score",y="er score",hue="location",data=df[:200],kind="scatter")


# In[83]:


A=(df["ar score"])
B=(df["er score"])

plt.scatter(A,B)
plt.xlabel("A")
plt.ylabel("B")
plt.title("histogram")

plt.grid(True)
plt.show()


# In[85]:


A=(df["fsr score"])
B=(df["cpf score"])

plt.plot(A,B)
plt.xlabel("A")
plt.ylabel("B")
plt.title("histogram")

plt.grid(True)
plt.show()


# In[86]:


A=(df["ifr score"])
B=(df["isr score"])

plt.bar(A,B)
plt.xlabel("A")
plt.ylabel("B")
plt.title("histogram")

plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




