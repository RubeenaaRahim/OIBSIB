#!/usr/bin/env python
# coding: utf-8

# # IRIS FLOWER CLASSIFICATION

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# # Understanding of the dataset

# In[2]:


d1=pd.read_csv("Iris.csv")


# In[3]:


d1


# In[4]:


df=d1.copy(deep=True)


# In[5]:


df


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.columns


# In[9]:


df.shape


# In[10]:


df.count()


# In[11]:


df.dtypes


# # Data Cleaning

# In[12]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[14]:


df.skew()


# In[15]:


df.median()


# There were no null or duplicate values in the dataset. We obtained skew and median of each dataset

# In[16]:


numerical=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
cateorical=["Species"]


# We have separated the numerical and categorical columns and made it into a separate list.

# # Outlier detection and removal

# In[17]:


for i in numerical:
    sns.boxplot(data=df[i])
    plt.show()


# In[18]:


for i in numerical:
    Q1=df[i].quantile(0.25)
    Q3=df[i].quantile(0.75)
    
    IQR=Q3-Q1
    low=Q1-1.5*IQR
    up=Q3+1.5*IQR
    
    for j in df[i]:
        if j<low:
            df=df.replace(j,low)
        if j>up:
            df=df.replace(j,up)


# In[19]:


for i in numerical:
    sns.boxplot(data=df[i])
    plt.show()


# Outliers in the dataset is obtained using boxplot.We removed all the outliers providing them with appropriate lower and upper limit values.

# # Data Visualization 

# In[20]:


sns.pairplot(data=df)
plt.show()


# In[21]:


sns.countplot(data=df, x="Species")
plt.show()


# All the species are equal in number.

# In[22]:


plt.figure(figsize=(15,10))
sns.heatmap(data=df.corr(),annot=True)
plt.show()


# Correlation between each attributes is obtained using heatmap.
# 
# Sepal length and Petal length are highly correlated to each other.When the Sepal length increases Petal length also increases and vice-versa.
# 
# Similarly Sepal length and Petal width and Petal length and Petal width are also positively correlated .
# 
# Sepal width and Sepal length, Sepal width and Petal length and Sepal width and Petal width are negatively correlatedd ie., when one value increases the other value decreases and vice-versa.

# # Encoding the target column

# In[23]:


from sklearn.preprocessing import LabelEncoder


# In[24]:


lb=LabelEncoder()


# In[25]:


df["Species"]=lb.fit_transform(df["Species"])
df


# We encoded the target column i.e 'Species'

# # Model building

# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x=df.drop(["Species"],axis=1)
y=df["Species"]


# In[29]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=0,test_size=0.3)


# In[30]:


X_train.shape, Y_train.shape


# In[31]:


X_test.shape,Y_test.shape


# # Fitting the model

# In[32]:


from sklearn.linear_model import LogisticRegression


# In[33]:


model=LogisticRegression(solver='liblinear')


# In[34]:


model.fit(X_train,Y_train)


# In[35]:


y_pred=model.predict(X_test)


# # Calculating the performance of the model

# In[36]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report


# In[37]:


print("Accuracy=", accuracy_score(Y_test,y_pred))


# The model that we used is logistic regression.
# 
# The accuracy for the model is 93%.

# In[39]:


print(classification_report(Y_test, y_pred))


# In[40]:


print(confusion_matrix(Y_test,y_pred))


# Precision,Recall,f1-score and support of the target column is obtained using classification report

# # Saving the model

# In[41]:


import pickle
saved_model=pickle.dumps(model)
model_from_pickle=pickle.loads(saved_model)
model_from_pickle.predict(X_test)


# Firstly we will be using the dump() function to save the model using pickle.Then we will be loading that saved model.lastly, after loading that model we will use this to make predictions.

# # Prediction on an Unknown Data

# In[42]:


print(model.predict([[3.6,1.6,2.4,0.7,0.9]]))


# In[ ]:




