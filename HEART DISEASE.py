#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[22]:


hd = pd.read_csv('C:\\Users\\Well\\Downloads\\heart.csv')
hd


# In[23]:


hd.shape


# In[24]:


hd.dtypes


# In[25]:


hd.isnull().sum()


# In[26]:


hd= hd.rename(columns= {'cp': 'chest_pain_type' , 'trestbps': 'resting_blood_pressure' , 'chol' : 'cholesterol',
                                             'fbs': 'fasting_blood_sugar' , 'restecg' : 'rest_ecg' ,'thalach' : 'max_heart_rate_achieved',
                                             'exang' : 'exercise_induced_angina' , 'oldpeak' : 'st_depression' , 'slope' : 'st_slope',
                                             'ca' : 'num_major_vessels' , 'thal' : 'thalassemia'})


# In[27]:


hd


# In[28]:


hd['sex'][hd['sex']== 0] = 'Female'
hd['sex'][hd['sex']==1] = 'Male'


# In[29]:


hd['chest_pain_type'][hd['chest_pain_type'] == 0] = 'typical angina'
hd['chest_pain_type'][hd['chest_pain_type'] == 1] = 'atypical angina'
hd['chest_pain_type'][hd['chest_pain_type'] == 2] = 'non-anginal pain'
hd['chest_pain_type'][hd['chest_pain_type'] == 3] = 'asymptomatic'


# In[30]:


hd['fasting_blood_sugar'][hd['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
hd['fasting_blood_sugar'][hd['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'


# In[31]:


hd['rest_ecg'][hd['rest_ecg'] == 0] = 'normal'
hd['rest_ecg'][hd['rest_ecg'] == 1] = 'ST-T wave abnormality'
hd['rest_ecg'][hd['rest_ecg'] == 2] = 'left ventricular hypertrophy'


# In[32]:


hd['exercise_induced_angina'][hd['exercise_induced_angina'] == 0] = 'no'
hd['exercise_induced_angina'][hd['exercise_induced_angina'] == 1] = 'yes'


# In[33]:


hd['st_slope'][hd['st_slope'] == 1] = 'upsloping'
hd['st_slope'][hd['st_slope'] == 2] = 'flat'
hd['st_slope'][hd['st_slope'] == 3] = 'downsloping'


# In[34]:


hd['thalassemia'][hd['thalassemia'] == 1] = 'normal'
hd['thalassemia'][hd['thalassemia'] == 2] = 'fixed defect'
hd['thalassemia'][hd['thalassemia'] == 3] = 'reversable defect'


# In[35]:


hd.head()


# In[36]:


hd.describe().transpose()


# In[37]:


hd['sex'].value_counts()


# In[38]:


hd['resting_blood_pressure'].value_counts()


# In[39]:


hd['chest_pain_type'].value_counts()


# In[40]:


hd['fasting_blood_sugar'].value_counts()


# In[42]:


hd['rest_ecg'].value_counts()


# In[43]:


hd['exercise_induced_angina'].value_counts()


# In[45]:


hd['st_slope'].value_counts()


# In[47]:


hd['thalassemia'].value_counts()


# In[61]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(hd['age'])
plt.show()


# In[52]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(hd['sex'])
plt.show()


# In[62]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(hd['chest_pain_type'])
plt.show()


# In[63]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(hd['target'])
plt.show()


# In[64]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(hd['exercise_induced_angina'])
plt.show()


# In[65]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(hd['rest_ecg'])
plt.show()


# In[66]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(hd['st_slope'])
plt.show()


# In[67]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(hd['thalassemia'])
plt.show()


# In[68]:


sns.distplot(hd['target'])


# In[69]:


pd.crosstab(hd.age,hd.target).plot(kind="bar",figsize=(25,8),color=['red','blue' ])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[70]:


pd.crosstab(hd.sex,hd.target).plot(kind="bar",figsize=(10,5),color=['green','yellow' ])
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# In[71]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='age',y='sex',data=hd)
plt.show()


# In[72]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='age',y='chest_pain_type',data=hd)
plt.show()


# In[73]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='sex',y='max_heart_rate_achieved',data=hd)
plt.show()


# In[74]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='sex',y='target',data=hd)
plt.show()


# In[75]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='sex',y='cholesterol',data=hd)
plt.show()


# In[76]:


sns.pairplot(data=hd)


# In[77]:


plt.figure(figsize=(14,10))
sns.heatmap(hd.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)
plt.show()


# In[78]:


hd.groupby('chest_pain_type', as_index=False)['target'].mean()


# In[79]:


hd.groupby('st_slope',as_index=False)['target'].mean()


# In[80]:


hd.groupby('thalassemia',as_index=False)['target'].mean()


# In[81]:


hd.groupby('target').mean()


# In[84]:


hd.chest_pain_type = hd.chest_pain_type.astype("category")
hd.exercise_induced_angina = hd.exercise_induced_angina.astype("category")
hd.fasting_blood_sugar = hd.fasting_blood_sugar.astype("category")
hd.rest_ecg = hd.rest_ecg.astype("category")
hd.sex = hd.sex.astype("category")
hd.st_slope = hd.st_slope.astype("category")
hd.thalassemia = hd.thalassemia.astype("category")


# In[85]:


hd1 = pd.get_dummies(hd, drop_first=True)


# In[86]:


hd1.head(10)


# In[87]:


from sklearn.preprocessing import scale
scale(hd1)


# In[88]:


np.exp(scale(hd1))


# In[89]:


x = hd1.drop(['target'], axis = 1)
y = hd1.target.values


# In[90]:


x


# In[91]:


y


# In[117]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)


# In[118]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[119]:


logreg.fit(x_train,y_train)


# In[120]:


Lr_pred = logreg.predict(x_test)
Lr_pred


# In[121]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(Lr_pred,y_test))


# In[122]:


from sklearn.metrics import accuracy_score
Lr_accuracy = accuracy_score(Lr_pred,y_test)
Lr_accuracy


# In[126]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier


# In[127]:


classifier.fit(x_train,y_train)


# In[128]:


knn_pred = classifier.predict(x_test)
knn_pred


# In[129]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(knn_pred,y_test))


# In[130]:


from sklearn.metrics import accuracy_score
accuracy_knn=accuracy_score(knn_pred,y_test)
accuracy_knn


# In[132]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier


# In[133]:


classifier.fit(x_train,y_train)


# In[134]:


Nbc_pred = classifier.predict(x_test)
Nbc_pred


# In[135]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(Nbc_pred,y_test))


# In[136]:


from sklearn.metrics import accuracy_score
Nbc_accuracy = accuracy_score(Nbc_pred,y_test)
Nbc_accuracy


# In[137]:


from sklearn.tree import DecisionTreeClassifier
classifier1 = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier1


# In[138]:


classifier1.fit(x_train,y_train)


# In[139]:


dt_pred = classifier1.predict(x_test)
dt_pred


# In[140]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(dt_pred,y_test))


# In[141]:


from sklearn.metrics import accuracy_score
accuracy_dt = accuracy_score(dt_pred,y_test)
accuracy_dt


# In[142]:


from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(criterion='entropy',random_state=0)
classifier2


# In[143]:


classifier2.fit(x_train,y_train)


# In[145]:


rf_pred = classifier2.predict(x_test)
rf_pred


# In[146]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(rf_pred,y_test))


# In[147]:


from sklearn.metrics import accuracy_score
accuracy_rf = accuracy_score(rf_pred,y_test)
accuracy_rf


# In[148]:


from sklearn.svm import SVC
classifier3 = SVC(kernel = 'linear', random_state = 0)
classifier3


# In[149]:


classifier3.fit(x_train,y_train)


# In[150]:


SVC_pred = classifier3.predict(x_test)
SVC_pred


# In[151]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(SVC_pred,y_test))


# In[152]:


from sklearn.metrics import accuracy_score
accuracy_SVC = accuracy_score(SVC_pred,y_test)
accuracy_SVC


# In[153]:


from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier()


# In[155]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
classifier4 = GridSearchCV(clf, parameters, cv=5, scoring='accuracy')
classifier4


# In[156]:


acc_scorer = make_scorer(accuracy_score)


# In[157]:


grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(x_train, y_train)


# In[158]:


clf = grid_obj.best_estimator_


# In[159]:


clf.fit(x_train, y_train)


# In[160]:


predictions = clf.predict(x_test)
print(accuracy_score(y_test, predictions))


# In[ ]:




