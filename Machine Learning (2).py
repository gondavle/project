#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[76]:


df=pd.read_csv("diabetes.csv")
df


# In[77]:


#total number of rows and columns
df.shape


# In[78]:


df.head()


# In[79]:


df=df.iloc[1:]


# In[80]:


df.head(2)


# In[81]:


#Gives information about dataset like data type,columns,null value count,memory usage
df.info()


# In[82]:


# Basic static detail about the data
df.describe()


# In[83]:


df.describe().T


# In[84]:


df_copy=df.copy(deep=True)


# In[85]:


df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)


# In[86]:


df_copy.isnull().sum()


# In[87]:


p=df.hist(figsize=(20,20))


# In[88]:


df_copy['Glucose'].fillna(df_copy['Glucose'].mean(),inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(),inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].mean(),inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].mean(),inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].mean(),inplace=True)


# In[89]:


df_copy.isnull().sum()


# # Plotting after nan remove

# In[90]:


A=df_copy.hist(figsize=(20,20))


# # Heatmap for the unclean dataset

# In[91]:


sns.heatmap(df.corr(),annot=True)


# In[92]:


sns.heatmap(df_copy.corr(),annot=True,cmap='Dark2')


# # Pairplot for clean dataset

# In[93]:


sns.pairplot(df_copy,hue='Outcome')


# In[94]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[95]:


x=df_copy.drop(['Outcome'],axis=1)
x


# In[96]:


x=scaler.fit_transform(x)
x


# In[97]:


cols=[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']]


# In[98]:


x=pd.DataFrame(x)


# In[99]:


x.columns=cols


# In[100]:


x.head()


# In[101]:


y=df_copy.Outcome


# In[102]:


y.head()


# In[103]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.33,random_state=42)


# # K neighbors classifier

# In[104]:


from sklearn.neighbors import KNeighborsClassifier


# In[105]:


train_score=[]
test_score=[]
for i in range(1,15):
    knn=KNeighborsClassifier(i)
    knn.fit(x_train,y_train)
    train_score.append(knn.score(x_train,y_train))
    test_score.append(knn.score(x_test,y_test))
                            


# In[111]:


max_train_score=max(train_score)
train_score_ind=[i for i,v in enumerate(train_score) if v==max_train_score]
print('max train score : {} % and k= {}' . format(max_train_score*100,list(map(lambda x:x+1 ,train_score_ind))))


# In[112]:


max_test_score=max(test_score)
test_score_ind=[i for i ,v in enumerate(test_score) if v==max_test_score]
print('max train score : {} % and k= {}' . format(max_test_score*100,list(map(lambda x:x+1 ,test_score_ind))))


# # Result Visualization

# In[117]:


plt.figure(figsize=(10,4))
p=sns.lineplot(range(1,15),train_score,marker='X',label='train_score')
p=sns.lineplot(range(1,15),test_score,marker='o',label='test_score')


# In[118]:


knn=KNeighborsClassifier(13)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)


# In[120]:


from sklearn import metrics
confusion_matrix=metrics.confusion_matrix


# In[121]:


y_pred=knn.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
pd.crosstab(y_test,y_pred,rownames=['True'],colnames=['Predicted'],margins=True)


# # Classification Report

# In[124]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# # ROC - AUC

# In[126]:


from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test,y_pred_proba)


# In[127]:


plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr, label='knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('knn(n_neighbors=11) Roc Curve')
plt.show()


# # Area under ROC Curve

# In[128]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)


# In[130]:


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
#in case of classifier like knn the paramter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(x,y)
print("Best Score:" +str(knn_cv.best_score_))
print("Best Parameters:" +str(knn_cv.best_params_))

