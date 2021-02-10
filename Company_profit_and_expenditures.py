#!/usr/bin/env python
# coding: utf-8

# Loading the libraries

# In[1]:


# Loading the required libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[2]:


#Loading the dataset
data = pd.read_csv("/Users/ashutoshshanker/Desktop/Data_Git/Company_Dataset/50_Startups.csv")


# In[3]:


# Renaming the columns
data = data.rename(columns={'R&D Spend':'R&D_Spend','Administration':'Administration','Marketing Spend':'Marketing_Spend','State':'State', 'Profit':'Profit' })


# In[4]:


# First 5 rows in the dataset
data.head(5)


# In[5]:


# Statistical Description
data.describe().T


# In[6]:


data.Profit.unique


# In[7]:


data.Profit.median()


# In[8]:


data.count


# In[9]:


data.describe().T


# Columns with number and percentage of missing data

# In[10]:


# Columns with number and percentage of missing data
missing_data = pd.DataFrame([data.isnull().sum(), data.isnull().sum() * 100.0/data.shape[0]]).T
missing_data.columns = ['No. of Missing Data', 'Percentage of Missing data']
missing_data


# # Data Correlation

# In[11]:


data.corr()


# In[12]:


data.cov()


# In[13]:


plt.figure(figsize=(10,6))
data['R&D_Spend'].plot(kind='hist')
plt.xlabel("Profit")
plt.ioff()
plt.show()


# Correlation Matrix

# In[14]:


plt.figure(figsize=(20,10))
palette = sns.diverging_palette(20, 220, n=256)
corr=data.corr(method='pearson')
sns.heatmap(corr, annot=True, fmt=".2f", cmap=palette, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(ylim=(6, 0))
plt.title("Correlation Matrix",size=15, weight='bold')


# In[15]:


plt.figure(figsize=(6,4))
sns.barplot(x='State', y='Profit', data=data)

plt.xlabel("State", size=13)
plt.ylabel("Profit", size=13)
plt.title("State vs Profit",size=15, weight='bold')


# In[16]:


# Correlation matrix heatmap

import seaborn as sns
fig, ax = plt.subplots(figsize=(6,6));
sns.heatmap(data.corr(), ax=ax, annot=False, linewidths=.1, cmap = "YlGnBu");
plt.title('Pearson Correlation matrix heatmap');


# Boxplot

# In[17]:


# Boxplot for each attribute
get_ipython().run_line_magic('matplotlib', 'inline')
data.boxplot(figsize=(150,50))


# In[18]:


data.Marketing_Spend.unique


# In[19]:


data


# Pearson's Correlation

# In[20]:


plt.figure(figsize=(8,2))
data.corr()['Profit'].sort_values()[:-1].plot(kind='bar')
plt.show()


# In[21]:


data.hist(figsize=(20,20))


# In[22]:


data_box=data.drop('Profit',axis=1)


# In[24]:


# Binning 'R&D_Spend' column
bins_RD_Spend = [0.00, 40000, 80000, 120000, 160000, np.inf]
labels_RD_Spend = [40000, 80000, 120000, 160000, 200000]
data['binned_R&D_Spend'] = pd.cut(data['R&D_Spend'], bins_RD_Spend, labels=labels_RD_Spend)


# In[25]:


data['binned_R&D_Spend'] = data['binned_R&D_Spend'].replace(np.nan, 0)


# In[26]:


# Binning 'Administration' column
bins_Administration = [0.00, 40000, 80000, 120000, 160000, np.inf]
labels_Administration = [40000, 80000, 120000, 160000, 200000]
data['binned_administration'] = pd.cut(data['Administration'], bins_Administration, labels=labels_Administration)


# In[27]:


# Binned 'Marketing_Spend' column
bin_Marketing_Spend = [0, 100000, 200000, 300000, 400000, np.inf]
labels_Marketing_Spend = [100000, 200000, 300000, 400000, 500000]
data['binned_marketing_spend'] = pd.cut(data['Marketing_Spend'], bin_Marketing_Spend, labels = labels_Marketing_Spend)


# In[28]:


data['binned_marketing_spend'] = data['binned_marketing_spend'].replace(np.nan, 0)


# In[29]:


# Binned 'Profit' column
bins_profit = [0.00, 108000, np.inf]
label_profit = ['<108000','>=108000']
data['binned_profit'] = pd.cut(data['Profit'],bins_profit, labels = label_profit)


# In[30]:


data


# In[31]:


data_processed = data
data_processed = data_processed.drop('Profit', axis = 1)
data_processed = data_processed.drop('R&D_Spend', axis = 1)
data_processed = data_processed.drop('Administration', axis = 1)
data_processed = data_processed.drop('Marketing_Spend', axis = 1)


# In[32]:


data_processed['binned_profit'] = data_processed['binned_profit'].replace('>=108000', 'More')
data_processed['binned_profit'] = data_processed['binned_profit'].replace('<108000', 'Less')


# In[33]:


data_processed


# In[34]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[35]:


le.fit(data_processed['State'])
data_processed['State_val'] = le.transform(data_processed['State'])


# In[36]:


data_processed = data_processed.drop('State', axis = 1)


# In[37]:


data_processed


# # Data Prediction

# In[38]:


# Dividing the dataset into training and testing data:
X = data_processed.drop('binned_profit', axis = 1)
Y = data_processed.binned_profit


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Weight of transaction
#wp = y_train.value_counts()[0] / len(y_train)
#wn = y_train.value_counts()[1] / len(y_train)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {X_train.shape}")
print(f"y_test: {y_test.shape}")


# In[39]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[40]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[41]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[42]:


# Calculating the precision, recall, f1-score, and support
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[43]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()


# In[44]:


# Decision tree model on original dataset
classifier.fit(X_train, y_train)


# In[45]:


y_pred_dt = classifier.predict(X_test)


# In[46]:


from sklearn.metrics import classification_report, confusion_matrix

# Confusion Matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_dt))

# Calculating the precision, recall, f1-score, and support
print(classification_report(y_test, y_pred_dt))


# In[47]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
model = GaussianNB()


# In[48]:


# Naive Bayes on original Data
model.fit(X_train, y_train)


# In[49]:


y_pred_NB = model.predict(X_test)
y_pred_NB


# In[50]:


accuracy = accuracy_score(y_test,y_pred_NB)*100
accuracy


# In[51]:


from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel


# In[52]:


# SVM on dataset
#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred_svm = clf.predict(X_test)


# In[53]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)


# In[54]:


# Neural Network on original dataset
clf.fit(X_train, y_train)


# In[55]:


y_pred_NN = clf.predict(X_test)


# In[56]:


# Accuracy
Accuracy_NN = metrics.accuracy_score(y_test, y_pred_NN)
Accuracy_NN


# In[57]:


data_feature = data_processed


# In[58]:


#data_feature = data_feature.drop('binned_administration', axis = 1)
data_feature = data_feature.drop('State_val', axis = 1)


# In[59]:


data_feature


# In[ ]:




