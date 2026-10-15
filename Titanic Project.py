#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib


# In[42]:


titanic = pd.read_csv("Titanic-Dataset.csv")


# In[43]:


titanic


# In[44]:


titanic.info()


# In[45]:


titanic.isnull().sum()


# In[46]:


titanic.describe()


# In[47]:


titanic.duplicated().sum()


# # handelMISSING value

# In[48]:


titanic["Age"]=titanic["Age"].fillna(titanic["Age"].mean())


# In[49]:


titanic 


# In[50]:


titanic.drop(columns= ["Cabin","Ticket","PassengerId","Name"], axis= 1, inplace = True)


# In[51]:


titanic


# In[52]:


titanic["Embarked"]=titanic["Embarked"].fillna(titanic["Embarked"].mode()[0])


# In[53]:


titanic.isnull().sum()


# In[54]:


titanic["Survived"].value_counts().plot(kind="bar")


# In[55]:


b = titanic.groupby("Sex")["Survived"].sum().plot(kind="bar")


# In[56]:


titanic["Age"].value_counts().plot(kind="bar")


# In[57]:


# ENcoding


# In[58]:


titanic_encoded = pd.get_dummies(
    titanic,
    columns= ["Sex","Embarked"],
    drop_first= True
)


# In[59]:


titanic


# In[60]:


# FEature Engineering


# In[61]:


titanic["family_size"] = titanic["SibSp"] +titanic["Parch"]+1


# In[62]:


titanic["Isalone"] = (titanic["family_size"] == 1).astype(int)


# In[63]:


titanic = pd.get_dummies(
    titanic,
    columns = ["Sex","Embarked"],
    drop_first = True
)


# In[64]:


titanic


# In[65]:


titanic.drop


# In[66]:


x = titanic.drop("Survived", axis = 1)


# In[67]:


y = titanic["Survived"]


# In[68]:


x_train,x_test, y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state = 42
)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_test_scaled


# In[69]:


x_train.shape


# In[70]:


logistic_model= LogisticRegression()


# In[71]:


logistic_model.fit(x_train, y_train)


# In[72]:


prediction = logistic_model.predict(x_test)
if prediction[0] ==1:
    "prediction:Survived"
else:
    print("prediction: did not survived")


# In[73]:


svm_model = SVC(kernel = "linear")


# In[74]:


svm_model.fit(x_train,y_train)


# In[75]:


predict = svm_model.predict(x_test)
if predict[0] ==1:
    "prediction : Survived"
else:
    print("prediction : did not survived")


# In[77]:


knn_model =KNeighborsClassifier(n_neighbors = 3)


# In[78]:


knn_model.fit(x_train,y_train)


# In[81]:


predict1 = knn_model.predict(x_test)
if predict1[0] == 1:
    "prediction: Survived"
else:
    print("pridiction : did not survived")


# In[82]:


joblib.dump(logistic_model, "lregression.pkl")


# In[ ]:





# In[ ]:




