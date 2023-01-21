#!/usr/bin/env python
# coding: utf-8

# # Syed Muhammad Ali Akbar
# 21034385
# (2nd Part of Code)

# In[34]:


# exponential growth function
def exp_growth(t, scale, growth):
    
    f = scale * np.exp(growth * (t-1950)) 
    return f


# In[35]:


import wbgapi as wb
df  = wb.data.DataFrame('BX.GSR.GNFS.CD', 'PAK', range(1980, 2022), numericTimeKeys=True)


# In[36]:


# import to suppress scientific representation of data and show it in full int or float formats
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df


# In[37]:


df = df.unstack(1).reset_index()


# In[38]:


df = df.rename(columns={'level_0': 'Year', 'economy': 'Country', 0: 'Total Exports USD'})
df.head()


# In[39]:


import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


# In[40]:


# fitting with default initial parameters
popt, covar = opt.curve_fit(exp_growth, df["Year"], df["Total Exports USD"])


# In[41]:


plt.figure(dpi=200)
df["pop_exp"] = exp_growth(df["Year"], *popt)
plt.figure()
plt.plot(df["Year"], df["Total Exports USD"], 'o-', label="data")
plt.plot(df["Year"], df["pop_exp"], label="fit")
plt.legend()
plt.title("First fit attempt")
plt.xlabel("Year")
plt.ylabel("Total Exports")
plt.show()
# plt.savefig('fig6.png')
print()


# In[42]:


# suitable start value the pedestrian way
popt = [4e9, 0.03]
df["pop_exp"] = exp_growth(df["Year"], *popt)
plt.figure()
plt.plot(df["Year"], df["Total Exports USD"], 'o-', label="data")
plt.plot(df["Year"], df["pop_exp"], label="fit")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Total Exports USD")
plt.title("Slightly Improved start value")
# plt.savefig('fig7.png')
plt.show()


# In[43]:


# final fit exponential growth
popt, covar = opt.curve_fit(exp_growth, df["Year"],
df["Total Exports USD"], p0=[4e8, 0.02])
print("Fit parameter", popt)
df["pop_exp"] = exp_growth(df["Year"], *popt)
plt.figure()
plt.plot(df["Year"], df["Total Exports USD"], 'o-', label="data")
plt.plot(df["Year"], df["pop_exp"], label="fit")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Total Exports USD")
plt.title("Final fit exponential growth")
# plt.savefig('fig8.png')
plt.show()


# In[ ]:





# In[ ]:




