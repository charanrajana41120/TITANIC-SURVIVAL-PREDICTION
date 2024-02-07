#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[7]:


df = pd.read_csv('C:\\Users\\satya\\Downloads\\archive (1)\\tested.csv')


# In[8]:


df.head()


# In[9]:


df.columns


# In[10]:


df.shape


# In[11]:


df['Survived'].value_counts()


# In[12]:


df['Sex'].value_counts()


# In[13]:


df['Embarked'].value_counts()
# C = Cherbourg; Q = Queenstown; S = Southampton


# In[14]:


df.isnull().sum()


# In[15]:


df = df.drop(['Cabin','PassengerId','Name','Ticket'],axis=1)


# In[17]:


df.head()


# In[19]:


# Missing value imputation - Age.
df['Age'].hist()


# In[21]:


def age_approx(cols):
    age = cols[0]
    pclass = cols[1]
    if (pd.isnull(age)):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        elif pclass == 3:
            return 24
    else:
        return age


# In[20]:


df.groupby('Pclass').median()


# In[51]:


#let's visualize the count of survivals wrt pclass
sns.countplot(x=df['Survived'], hue=df['Pclass'])


# In[22]:


df['Age'] = df[['Age','Pclass']].apply(age_approx,axis=1)


# In[23]:


df.isnull().sum()
# Drop all the records with null values.
df.dropna(inplace=True)


# In[24]:


df.isnull().sum()


# In[25]:


# Dtypes 
df.dtypes


# # Sex and Embarked are categorical. They need to be converted to numeric so that they can be passed to the machine learning algorithm. Otherwise, the algorithm throws error.
# 

# In[26]:


df_dummies_sex = pd.get_dummies(df,columns=['Sex'])


# In[27]:


df_dummies = pd.get_dummies(df_dummies_sex, columns=['Embarked'])


# In[28]:


df = df_dummies


# In[29]:


# find correlation between variables
corr_matrix = df.corr()
sns.heatmap(corr_matrix)


# In[30]:


corr_matrix['Survived'].abs().sort_values(ascending=False)


# In[31]:


X = df.drop(['Survived'],axis=1)
y = df['Survived']


# In[32]:


xtrain,xtest,ytrain,ytest = model_selection.train_test_split(X,y,test_size=0.2,random_state=100)


# In[33]:


print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# In[34]:


logReg = LogisticRegression()


# In[36]:


logReg.fit(xtrain,ytrain)


# In[37]:


predictions = logReg.predict(xtest)


# In[39]:


predictions


# In[40]:


metrics.confusion_matrix(ytest,predictions)


# In[41]:


metrics.accuracy_score(ytest,predictions)


# In[42]:


print(metrics.classification_report(ytest,predictions))


# In[43]:


logReg.coef_


# In[44]:


logReg.intercept_


# In[45]:


logReg.predict_proba(xtest)


# In[47]:


# Create a DataFrame with actual and predicted values
result_df = pd.DataFrame({'Actual': ytest, 'Predicted': predictions})

# Display the first few rows of the DataFrame
print(result_df.head())


# In[ ]:




